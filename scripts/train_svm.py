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
from sklearn.utils import shuffle as sklearn_shuffle
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
DEFAULT_EPOCHS       = 5       # full passes over the dataset
DEFAULT_CHUNK_SIZE   = 20      # number of patients per training chunk
DEFAULT_BATCH_SIZE   = 512     # mini-batch size for partial_fit
DEFAULT_ALPHA        = 1e-4    # regularization strength (lambda = 1/C)
DEFAULT_LEARNING_RATE = "optimal"  # adaptive schedule: eta = 1 / (alpha * (t0 + t))
DEFAULT_VAL_FRAC     = 0.15    # fraction of patients held out for validation


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


def evaluate(clf, scaler, X_val, y_val, label_encoder, batch_size=4096):
    """Evaluate the classifier on a validation set (in batches to save RAM).

    Returns accuracy and the classification report string.
    """
    y_pred_parts = []
    for start in range(0, len(X_val), batch_size):
        X_batch = scaler.transform(X_val[start : start + batch_size])
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

    # -- 6. Build patient-to-row-index mapping ---------------------------------
    #
    # For each patient, store the row indices into X_train_all / y_train_all
    # so we can efficiently select patient groups during training.
    #
    print("  Building patient index ...")
    t0 = time.time()
    patient_indices = {}
    for user in train_users:
        patient_indices[user] = np.where(user_train == user)[0]
    del user_train  # no longer needed
    gc.collect()

    total_indexed = sum(len(v) for v in patient_indices.values())
    print(f"  Indexed {len(patient_indices)} patients, "
          f"{total_indexed:,} rows  ({time.time()-t0:.1f}s)")
    print()

    # -- 7. Precompute balanced class weights ----------------------------------
    #
    # weight_k = n_total / (n_classes * n_k)
    # 'balanced' class_weight is not supported with partial_fit, so we compute
    # sample weights manually and pass them to each partial_fit call.
    #
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
    clf = SGDClassifier(
        loss="hinge",                      # Hinge loss = Linear SVM
        penalty="l2",                      # L2 reg -> minimise ||w||^2 (max margin)
        alpha=args.alpha,                  # lambda = 1/C  (regularisation strength)
        learning_rate=args.learning_rate,  # step-size schedule
        eta0=0.01 if args.learning_rate == "constant" else 0.0,
        random_state=42,
        warm_start=False,                  # we use partial_fit, not refit
        verbose=0,
    )

    # -- 9. Training epochs ----------------------------------------------------
    print(f"  Hyperparameters:")
    print(f"    alpha (lambda=1/C)   : {args.alpha}")
    print(f"    learning_rate        : {args.learning_rate}")
    print(f"    batch_size           : {args.batch_size}")
    print(f"    patients_per_chunk   : {args.chunk_size}")
    print(f"    epochs               : {args.epochs}")
    print()
    print("-" * 70)

    rng = np.random.RandomState(42)
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # -- Shuffle PATIENT order every epoch ---------------------------------
        # This decorrelates the gradient: different patient sequences prevent
        # the SVM from "forgetting" earlier patients (catastrophic forgetting
        # mitigation).  We shuffle patients, not chunk files.
        patient_order = list(train_users)
        rng.shuffle(patient_order)

        n_batches_total = 0
        epoch_samples = 0
        n_chunks = (len(patient_order) + args.chunk_size - 1) // args.chunk_size
        chunk_idx = 0

        # -- Middle loop: group patients into chunks ---------------------------
        for chunk_start in range(0, len(patient_order), args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, len(patient_order))
            chunk_patients = patient_order[chunk_start:chunk_end]
            chunk_idx += 1

            # Gather row indices for all patients in this chunk
            idx = np.concatenate([patient_indices[u] for u in chunk_patients])

            # Select rows and scale
            X_chunk = scaler.transform(X_train_all[idx])
            y_chunk = y_train_all[idx]

            # -- Shuffle windows within the chunk ------------------------------
            # Mixes windows from different patients and gestures so that
            # each mini-batch is a representative random sample.
            X_chunk, y_chunk = sklearn_shuffle(
                X_chunk, y_chunk, random_state=rng.randint(0, 2**31)
            )

            # -- Inner loop: mini-batches for partial_fit ----------------------
            n_windows = len(X_chunk)
            for batch_start in range(0, n_windows, args.batch_size):
                batch_end = min(batch_start + args.batch_size, n_windows)
                X_batch = X_chunk[batch_start:batch_end]
                y_batch = y_chunk[batch_start:batch_end]

                # Per-sample weights via numpy fancy indexing (fast)
                sample_weight = class_weight_array[y_batch]

                # partial_fit updates w and b using the hinge-loss gradient:
                #   If y*(w.x + b) >= 1 (correct): w <- (1-eta*alpha)*w
                #   If y*(w.x + b) <  1 (error)  : w <- (1-eta*alpha)*w + eta*y*x
                #                                   b <- b + eta*y
                clf.partial_fit(
                    X_batch, y_batch,
                    classes=classes,
                    sample_weight=sample_weight,
                )
                n_batches_total += 1

            epoch_samples += n_windows

            # -- Progress update (inline, overwrite same line) ----------------
            elapsed = time.time() - epoch_start
            pct = chunk_idx / n_chunks * 100
            bar_len = 30
            filled = int(bar_len * chunk_idx // n_chunks)
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write(
                f"\r  Epoch {epoch}/{args.epochs}  "
                f"[{bar}] {pct:5.1f}%  "
                f"chunk {chunk_idx}/{n_chunks}  "
                f"{epoch_samples:,} samples  "
                f"{elapsed:.0f}s"
            )
            sys.stdout.flush()

            # Free chunk arrays
            del X_chunk, y_chunk
            gc.collect()

        # Clear the progress line
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()

        # -- Epoch validation --------------------------------------------------
        print(f"  Epoch {epoch}/{args.epochs}  Evaluating on validation set ...")
        val_acc, val_report = evaluate(clf, scaler, X_val, y_val, le)
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "val_acc": val_acc,
            "n_batches": n_batches_total,
            "n_samples": epoch_samples,
            "time_s": epoch_time,
        })

        # -- Save best model ---------------------------------------------------
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            improved = "  * NEW BEST"
            joblib.dump(clf, MODELS_DIR / "svm_sgd_best.joblib")
            joblib.dump(scaler, MODELS_DIR / "scaler_best.joblib")
            joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

        print(
            f"  Epoch {epoch}/{args.epochs}  |  "
            f"val_acc: {val_acc:.4f}  |  "
            f"batches: {n_batches_total:,}  |  "
            f"samples: {epoch_samples:,}  |  "
            f"{epoch_time:.0f}s{improved}"
        )

    # -- 10. Final summary -----------------------------------------------------
    print()
    print("-" * 70)
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print()

    # Reload best model for final report
    clf_best = joblib.load(MODELS_DIR / "svm_sgd_best.joblib")
    val_acc, val_report = evaluate(clf_best, scaler, X_val, y_val, le)
    print("  Classification Report (best model on validation set):")
    print(val_report)

    # Confusion matrix
    y_pred_parts = []
    for start in range(0, len(X_val), 4096):
        X_b = scaler.transform(X_val[start : start + 4096])
        y_pred_parts.append(clf_best.predict(X_b))
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

    # Save final model (last epoch, might differ from best)
    joblib.dump(clf, MODELS_DIR / "svm_sgd_final.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_final.joblib")
    print(f"  Models saved to: {MODELS_DIR.resolve()}")
    print("=" * 70)


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Linear SVM via SGD on EMG-EPN612 windowed features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="Number of full passes over the training data.")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help="Number of patients to group per training chunk.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="Mini-batch size for each partial_fit call.")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Regularization strength (lambda = 1/C). "
                        "Higher -> simpler model, wider margin.")
    p.add_argument("--learning-rate", type=str, default=DEFAULT_LEARNING_RATE,
                   choices=["optimal", "constant", "invscaling", "adaptive"],
                   help="Learning rate schedule for SGD.")
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                   help="Fraction of patients to reserve for validation.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
