"""
SVM Training via SGD (Stochastic Gradient Descent) for EMG-EPN612 dataset.

Implements a Linear SVM solved via SGD on ~8.8M windowed EMG feature rows
from 306 patients x 6 gestures x 25 repetitions.

Strategy (from notes):
 - The standard kernel SVM (sklearn.SVC) is infeasible for ~8.8M samples
   because the kernel matrix would require O(n^2) memory.
 - Instead, we solve the Primal problem directly using SGDClassifier with
   hinge loss, which is mathematically equivalent to a Linear SVM.
 - Data is loaded from a single training_set.parquet file, split by patient
   ID to prevent leakage, then shuffled at the patient level each epoch.
 - Each epoch sees every window exactly once (sampling without replacement).
 - Patient order is reshuffled between epochs to decorrelate gradients.

Architecture:
 - Outer loop  : Epochs  (full passes over all training patients)
 - Middle loop : Patient chunks  (groups of N patients selected from memory)
 - Inner loop  : Mini-batches (small slices fed to partial_fit)

Key SVM concepts mapped to SGDClassifier:
 - Hinge loss        -> loss='hinge'   (max(0, 1 - y*f(x)))
 - Regularization    -> alpha          (lambda = 1/C; higher alpha = simpler model)
 - Weight vector w   -> clf.coef_      (orientation of the hyperplane)
 - Bias b            -> clf.intercept_ (position of the hyperplane)
 - Multiclass        -> OvR by default (one hyperplane per class, highest score wins)

Usage:
    cd "EMG-EPN612 project"
    python scripts/train_svm.py
"""

import sys
import time
import json
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# --- Project paths (relative to project root) --------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
TRAINING_FILE  = PROJECT_ROOT / "training_set.parquet"
MODELS_DIR     = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# --- Dataset constants --------------------------------------------------------
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES  = len(ALL_LABELS)

# 72 feature columns: 8 channels x 9 TD9 features
CHANNELS   = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES  = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
N_FEATURES = len(FEATURE_COLS)  # 72

# --- Default hyper-parameters -------------------------------------------------
DEFAULT_EPOCHS       = 20      # max full passes (SGD iterations) over the dataset
DEFAULT_ALPHA        = 1e-4    # regularization strength (lambda = 1/C)
DEFAULT_LEARNING_RATE = "optimal"  # adaptive schedule: eta = 1 / (alpha * (t0 + t))
DEFAULT_VAL_FRAC     = 0.15    # fraction of patients held out for validation
DEFAULT_N_COMPONENTS = 500     # Nystroem: number of RBF kernel features
DEFAULT_GAMMA        = None    # Nystroem: RBF bandwidth (None = 1/n_features)


# --- Helpers ------------------------------------------------------------------

def split_patients(all_users: np.ndarray, val_frac: float, seed: int = 42):
    """Split unique patient IDs into training and validation sets.

    This is a true patient-level split: every window from a given patient
    ends up in either training or validation, never both.  This prevents
    data leakage (the model never sees any data from validation patients
    during training).
    """
    rng = np.random.RandomState(seed)
    users_shuffled = all_users.copy()
    rng.shuffle(users_shuffled)
    n_val = max(1, int(len(users_shuffled) * val_frac))
    val_users   = sorted(users_shuffled[:n_val].tolist())
    train_users = sorted(users_shuffled[n_val:].tolist())
    return train_users, val_users


def evaluate(clf, scaler, kernel_map, X_val, y_val, label_encoder, batch_size=4096):
    """Evaluate the classifier on a validation set (in batches to save RAM).

    Returns accuracy and the classification report string.
    """
    y_pred_parts = []
    for start in range(0, len(X_val), batch_size):
        X_batch = scaler.transform(X_val[start : start + batch_size])
        if kernel_map is not None:
            X_batch = kernel_map.transform(X_batch)
        y_pred_parts.append(clf.predict(X_batch))
    y_pred = np.concatenate(y_pred_parts)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(
        y_val,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
    )
    return acc, report


# --- Main training loop ------------------------------------------------------

def train(args):
    print("=" * 70)
    print("  SVM Training via SGD  -  EMG-EPN612 Dataset")
    print("=" * 70)

    # -- 1. Load full dataset --------------------------------------------------
    if not TRAINING_FILE.exists():
        print(f"ERROR: {TRAINING_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading dataset from {TRAINING_FILE.name} ...")
    t0 = time.time()
    df = pd.read_parquet(TRAINING_FILE)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} cols  "
          f"({time.time()-t0:.1f}s)")
    print()

    # -- 2. Patient-level train/val split --------------------------------------
    #
    # Split by unique patient ID -- not by chunk file.
    # This guarantees zero data leakage: every window from a patient is either
    # in training or in validation, never both.
    #
    all_users = df["user"].unique()
    train_users, val_users = split_patients(all_users, args.val_frac)
    print(f"  Total patients     : {len(all_users)}")
    print(f"  Training patients  : {len(train_users)}")
    print(f"  Validation patients: {len(val_users)}")
    print()

    # -- 3. Encode labels ------------------------------------------------------
    le = LabelEncoder()
    le.fit(ALL_LABELS)
    classes = le.transform(ALL_LABELS)  # integer class ids for partial_fit
    print(f"  Classes ({N_CLASSES}): {dict(zip(le.classes_, classes))}")
    print()

    # -- 4. Build train/val arrays ---------------------------------------------
    #
    # Separate the full DataFrame into numpy arrays for train and val.
    # We keep X_train_all in memory (float32) so we can index into it
    # by patient during each epoch without re-reading from disk.
    #
    print("  Splitting data by patient ...")
    t0 = time.time()
    train_mask = df["user"].isin(set(train_users))

    X_train_all = df.loc[train_mask, FEATURE_COLS].values.astype(np.float32)
    y_train_all = le.transform(df.loc[train_mask, "label"].values)
    user_train  = df.loc[train_mask, "user"].values

    X_val = df.loc[~train_mask, FEATURE_COLS].values.astype(np.float32)
    y_val = le.transform(df.loc[~train_mask, "label"].values)

    del df  # free the DataFrame -- we only need the numpy arrays now
    gc.collect()

    print(f"  Train: {X_train_all.shape[0]:,} rows  |  "
          f"Val: {X_val.shape[0]:,} rows  ({time.time()-t0:.1f}s)")
    print()

    # -- 5. Pre-fit global scaler on ALL training data (single pass) -----------
    #
    # Fit the scaler once on the full training set before any SGD step.
    # This ensures consistent feature scaling across all patients and epochs.
    # Because the raw data was already z-scored per-subject during
    # preprocessing, this global scaling harmonises across subjects.
    #
    print("  Fitting global scaler on full training data ...")
    t0 = time.time()
    scaler = StandardScaler()
    scaler.fit(X_train_all)
    print(f"  Scaler fitted ({time.time()-t0:.1f}s)")
    print()

    # -- 5b. Scale and save the full dataset to disk ---------------------------
    #
    # Save the scaled train and val arrays (+ labels, user ids) as parquet so
    # downstream notebooks or scripts can load ready-to-use data directly.
    #
    scaled_path = PROJECT_ROOT / "training_set_scaled.parquet"
    print(f"  Scaling and saving full dataset to {scaled_path.name} ...")
    t0 = time.time()

    # Re-read just the metadata columns we need (user, label) from the
    # original file -- cheaper than keeping the whole DataFrame in memory.
    meta_df = pd.read_parquet(TRAINING_FILE, columns=["user", "label"])
    train_mask_save = meta_df["user"].isin(set(train_users))

    # Scale train and val features
    X_train_scaled = scaler.transform(X_train_all)
    X_val_scaled   = scaler.transform(X_val)

    # Build DataFrames
    df_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS)
    df_train_scaled["label"] = le.inverse_transform(y_train_all)
    df_train_scaled["user"]  = meta_df.loc[train_mask_save, "user"].values
    df_train_scaled["split"] = "train"

    df_val_scaled = pd.DataFrame(X_val_scaled, columns=FEATURE_COLS)
    df_val_scaled["label"] = le.inverse_transform(y_val)
    df_val_scaled["user"]  = meta_df.loc[~train_mask_save, "user"].values
    df_val_scaled["split"] = "val"

    df_scaled = pd.concat([df_train_scaled, df_val_scaled], ignore_index=True)
    df_scaled.to_parquet(scaled_path, index=False)

    del X_train_scaled, X_val_scaled, df_train_scaled, df_val_scaled
    del df_scaled, meta_df
    gc.collect()
    print(f"  Saved {scaled_path.name}  ({time.time()-t0:.1f}s)")
    print()

    # -- 6. Precompute balanced class weights ----------------------------------
    #
    # weight_k = n_total / (n_classes * n_k)
    # We compute sample weights and pass them to fit().
    #
    del user_train  # no longer needed (patient index not required with fit())
    gc.collect()
    print("  Computing class weights ...")
    label_counts = np.bincount(y_train_all, minlength=N_CLASSES)
    n_train_total = label_counts.sum()
    class_weight_array = n_train_total / (N_CLASSES * label_counts.astype(np.float64))
    cw_display = {str(le.classes_[c]): round(class_weight_array[c], 4)
                  for c in range(N_CLASSES)}
    print(f"  Class weights: {cw_display}")
    print()

    # -- 8. Initialise SGDClassifier -------------------------------------------
    #
    # SGDClassifier with loss='hinge' is a linear SVM solved via SGD.
    #
    # Key parameters:
    #   loss='hinge'     : Hinge loss -> equivalent to the SVM primal objective
    #   alpha            : Regularization strength lambda (= 1/C).
    #                      Higher alpha -> simpler model, wider margin, more errors.
    #   learning_rate    : Schedule for eta (step size).
    #     'optimal'      : eta = 1 / (alpha*(t0+t)), auto-tuned t0
    #     'invscaling'   : eta = eta0 / t^power_t
    #     'constant'     : eta = eta0  (fixed)
    #   penalty='l2'     : L2 regularization (minimise ||w||^2 -> max margin)
    #
    # w is initialised to the zero vector at the first partial_fit call.
    # Each subsequent partial_fit call updates w and b incrementally.
    #
    # -- 8. Scale ALL training data once (in-place to save memory) ------------
    #
    # Since all data is already in RAM, we scale it once and use clf.fit()
    # instead of the chunked partial_fit approach.  clf.fit() internally
    # shuffles and iterates for max_iter epochs with proper convergence
    # detection — no manual epoch/chunk/batch loops needed.
    #
    print("  Scaling full training set in-place ...")
    t0 = time.time()
    X_train_scaled = scaler.transform(X_train_all)
    del X_train_all  # free unscaled copy
    gc.collect()
    print(f"  Scaled ({time.time()-t0:.1f}s)")
    print()

    # -- 9. Kernel approximation (Nystroem) ------------------------------------
    #
    # A linear SVM cannot model the non-linear decision boundaries between
    # gesture classes.  Nystroem maps the 72 features to a higher-dimensional
    # space (n_components) where the RBF kernel is approximated by a linear
    # inner product.  The subsequent linear SVM in this space is equivalent
    # to an approximate RBF-kernel SVM in the original space.
    #
    # Complexity: O(n * n_components) for transform — much cheaper than the
    # full O(n^2) kernel matrix.
    #
    kernel_map = None
    if args.n_components > 0:
        gamma_val = args.gamma if args.gamma else 1.0 / N_FEATURES
        print(f"  Fitting Nystroem kernel approximation ...")
        print(f"    n_components : {args.n_components}")
        print(f"    gamma (RBF)  : {gamma_val:.6f}")
        t0 = time.time()
        kernel_map = Nystroem(
            kernel="rbf",
            gamma=gamma_val,
            n_components=args.n_components,
            random_state=42,
        )
        # Fit on a random subsample (Nystroem only needs n_components points
        # to build the basis — using more is wasteful).
        FIT_SAMPLE = min(args.n_components * 10, len(X_train_scaled))
        rng_fit = np.random.RandomState(42)
        fit_idx = rng_fit.choice(len(X_train_scaled), size=FIT_SAMPLE, replace=False)
        kernel_map.fit(X_train_scaled[fit_idx])
        print(f"  Kernel map fitted on {FIT_SAMPLE:,} samples  ({time.time()-t0:.1f}s)")

        # Transform in batches to avoid allocating the full (n × n_components)
        # matrix at once (which would exceed available RAM).
        BATCH = 500_000
        n_rows = len(X_train_scaled)
        X_kernel = np.empty((n_rows, args.n_components), dtype=np.float32)
        for bstart in range(0, n_rows, BATCH):
            bend = min(bstart + BATCH, n_rows)
            X_kernel[bstart:bend] = kernel_map.transform(
                X_train_scaled[bstart:bend]
            ).astype(np.float32)
        del X_train_scaled
        X_train_scaled = X_kernel
        del X_kernel
        gc.collect()
        print(f"  Transform complete: {N_FEATURES} -> {X_train_scaled.shape[1]} dims  "
              f"({time.time()-t0:.1f}s)")
        print()

    # -- 10. Compute per-sample weights (balanced class weighting) -------------
    sample_weights = class_weight_array[y_train_all]

    # -- 11. Initialise and train SGDClassifier --------------------------------
    #
    # SGDClassifier with loss='hinge' is a linear SVM solved via SGD.
    # Using fit() instead of partial_fit() lets sklearn handle:
    #   - proper data shuffling each epoch (all samples, not chunked)
    #   - learning rate scheduling with correct per-epoch reset
    #   - early stopping / convergence detection
    #
    clf = SGDClassifier(
        loss="hinge",                      # Hinge loss = Linear SVM
        penalty="l2",                      # L2 reg -> minimise ||w||^2 (max margin)
        alpha=args.alpha,                  # lambda = 1/C  (regularisation strength)
        learning_rate=args.learning_rate,  # step-size schedule
        eta0=0.01 if args.learning_rate != "optimal" else 0.0,
        max_iter=args.epochs,              # number of full passes over the data
        shuffle=True,                      # shuffle data before each epoch
        random_state=42,
        verbose=1,                         # print loss per epoch
        n_jobs=-1,                         # use all cores
    )

    print(f"  Hyperparameters:")
    print(f"    alpha (lambda=1/C)   : {args.alpha}")
    print(f"    learning_rate        : {args.learning_rate}")
    print(f"    max_iter (epochs)    : {args.epochs}")
    print(f"    n_components (kernel): {args.n_components}")
    print()
    print("-" * 70)

    t0 = time.time()
    clf.fit(X_train_scaled, y_train_all, sample_weight=sample_weights)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")
    print()

    # -- Validation ------------------------------------------------------------
    print("  Evaluating on validation set ...")
    val_acc, val_report = evaluate(clf, scaler, kernel_map, X_val, y_val, le)
    best_val_acc = val_acc

    history = [{
        "epoch": args.epochs,
        "val_acc": val_acc,
        "n_samples": len(y_train_all),
        "time_s": train_time,
    }]

    # -- Save model ------------------------------------------------------------
    joblib.dump(clf, MODELS_DIR / "svm_sgd_best.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_best.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    if kernel_map is not None:
        joblib.dump(kernel_map, MODELS_DIR / "kernel_map.joblib")

    print(
        f"  val_acc: {val_acc:.4f}  |  "
        f"samples: {len(y_train_all):,}  |  "
        f"{train_time:.0f}s"
    )

    # -- 10. Final summary -----------------------------------------------------
    print()
    print("-" * 70)
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print()

    print("  Classification Report (validation set):")
    print(val_report)

    # Confusion matrix
    y_pred_parts = []
    for start in range(0, len(X_val), 4096):
        X_b = scaler.transform(X_val[start : start + 4096])
        if kernel_map is not None:
            X_b = kernel_map.transform(X_b)
        y_pred_parts.append(clf.predict(X_b))
    y_pred = np.concatenate(y_pred_parts)
    cm = confusion_matrix(y_val, y_pred)
    print("  Confusion Matrix:")
    print(f"  {'':>12s}", "  ".join(f"{c:>10s}" for c in le.classes_))
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:>12s}", "  ".join(f"{v:10d}" for v in row))

    # Save training history
    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history saved to: {history_path}")

    print(f"  Models saved to: {MODELS_DIR.resolve()}")
    print("=" * 70)


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Linear SVM via SGD on EMG-EPN612 windowed features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="Max number of SGD iterations (full passes) over the training data.")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Regularization strength (lambda = 1/C). "
                        "Higher -> simpler model, wider margin.")
    p.add_argument("--learning-rate", type=str, default=DEFAULT_LEARNING_RATE,
                   choices=["optimal", "constant", "invscaling", "adaptive"],
                   help="Learning rate schedule for SGD.")
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                   help="Fraction of patients to reserve for validation.")
    p.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS,
                   help="Nystroem kernel approximation: number of RBF components. "
                        "Set to 0 to disable (pure linear SVM).")
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                   help="RBF kernel bandwidth for Nystroem. "
                        "None (default) = 1/n_features.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
