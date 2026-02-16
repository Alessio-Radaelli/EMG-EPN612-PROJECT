"""
SVM Training via SGD (Stochastic Gradient Descent) for EMG-EPN612 dataset.

Implements an approximate RBF-kernel SVM solved via SGD on ~8.8M windowed
EMG feature rows from 306 patients x 6 gestures x 25 repetitions.

Strategy:
 - The standard kernel SVM (sklearn.SVC) is infeasible for ~8.8M samples
   because the kernel matrix would require O(n^2) memory.
 - Instead we use the Nystroem kernel approximation to map the 72 original
   features into a higher-dimensional space where the RBF kernel is
   approximated by a linear inner product.  A linear SVM (SGDClassifier
   with hinge loss) in this space is equivalent to an approximate
   RBF-kernel SVM in the original space.
 - Kernel-mapped data is written to disk in chunks and streamed back
   during training so that only one chunk is in RAM at a time.

Multiclass strategy: One-vs-One (OvO) via manual partial_fit
 - Trains C*(C-1)/2 = 15 binary SGDClassifiers (one per pair of classes).
 - Each chunk is loaded once; all 15 classifiers process it before it is
   released, minimising disk I/O and peak RAM usage.
 - Prediction is by majority vote among all 15 classifiers.

Key SVM concepts mapped to SGDClassifier:
 - Hinge loss        -> loss='hinge'   (max(0, 1 - y*f(x)))
 - Regularization    -> alpha          (lambda = 1/C; higher alpha = simpler model)
 - Weight vector w   -> clf.coef_      (orientation of the hyperplane)
 - Bias b            -> clf.intercept_ (position of the hyperplane)

Usage:
    cd "EMG-EPN612 project"
    python scripts/train_svm.py
"""

import sys
import os
import time
import json
import gc
import shutil
import argparse
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

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
CHUNKS_DIR     = PROJECT_ROOT / "kernel_chunks"   # temp storage for mapped chunks

# --- Dataset constants --------------------------------------------------------
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES  = len(ALL_LABELS)

# 72 feature columns: 8 channels x 9 TD9 features
CHANNELS   = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES  = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
N_FEATURES = len(FEATURE_COLS)  # 72

# --- Default hyper-parameters -------------------------------------------------
DEFAULT_EPOCHS       = 20      # full passes over the dataset
DEFAULT_ALPHA        = 1e-6    # regularization strength (lambda = 1/C)
DEFAULT_LEARNING_RATE = "constant"    # fixed step size (no decay during partial_fit)
DEFAULT_ETA0         = 0.01    # constant learning rate
DEFAULT_VAL_FRAC     = 0.15    # fraction of patients held out for validation
DEFAULT_N_COMPONENTS = 2000    # Nystroem: number of RBF kernel features
DEFAULT_GAMMA        = "0.03"  # Nystroem: RBF bandwidth (higher = sharper locality)
DEFAULT_CHUNK_SIZE   = 250_000 # rows per disk chunk (~1.9 GB at 2000 components)


# --- Helpers ------------------------------------------------------------------

def split_patients(all_users: np.ndarray, val_frac: float, seed: int = 42):
    """Split unique patient IDs into training and validation sets.

    Patient-level split: every window from a given patient ends up in
    either training or validation, never both.  Prevents data leakage.
    """
    rng = np.random.RandomState(seed)
    users_shuffled = all_users.copy()
    rng.shuffle(users_shuffled)
    n_val = max(1, int(len(users_shuffled) * val_frac))
    val_users   = sorted(users_shuffled[:n_val].tolist())
    train_users = sorted(users_shuffled[n_val:].tolist())
    return train_users, val_users


# --- OvO helpers --------------------------------------------------------------

def ovo_pairs(n_classes: int):
    """Return list of (i, j) class-index pairs for One-vs-One."""
    return list(combinations(range(n_classes), 2))


def ovo_predict(classifiers, pairs, X):
    """Majority-vote prediction across OvO binary classifiers (vectorised)."""
    n_classes = max(c for p in pairs for c in p) + 1
    votes = np.zeros((len(X), n_classes), dtype=np.int32)
    for clf_ij, (ci, cj) in zip(classifiers, pairs):
        pred = clf_ij.predict(X)      # returns ci or cj
        np.add.at(votes, (np.arange(len(pred)), pred), 1)
    return votes.argmax(axis=1)


def ovo_predict_batched(classifiers, pairs, scaler, kernel_map,
                        X_raw, batch_size=4096):
    """Predict on raw (unscaled) data in batches to save RAM."""
    parts = []
    for start in range(0, len(X_raw), batch_size):
        X_b = scaler.transform(X_raw[start : start + batch_size])
        if kernel_map is not None:
            X_b = kernel_map.transform(X_b)
        parts.append(ovo_predict(classifiers, pairs, X_b))
    return np.concatenate(parts)


# --- Main training loop ------------------------------------------------------

def train(args):
    print("=" * 70)
    print("  SVM Training via SGD + Nystroem OvO  -  EMG-EPN612")
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
    all_users = df["user"].unique()
    train_users, val_users = split_patients(all_users, args.val_frac)
    print(f"  Total patients     : {len(all_users)}")
    print(f"  Training patients  : {len(train_users)}")
    print(f"  Validation patients: {len(val_users)}")
    print()

    # -- 3. Encode labels ------------------------------------------------------
    le = LabelEncoder()
    le.fit(ALL_LABELS)
    classes = le.transform(ALL_LABELS)
    print(f"  Classes ({N_CLASSES}): {dict(zip(le.classes_, classes))}")
    print()

    # -- 4. Build train/val arrays ---------------------------------------------
    print("  Splitting data by patient ...")
    t0 = time.time()
    train_mask = df["user"].isin(set(train_users))

    X_train_all = df.loc[train_mask, FEATURE_COLS].values.astype(np.float32)
    y_train_all = le.transform(df.loc[train_mask, "label"].values)

    X_val = df.loc[~train_mask, FEATURE_COLS].values.astype(np.float32)
    y_val = le.transform(df.loc[~train_mask, "label"].values)

    del df
    gc.collect()

    n_train = len(y_train_all)
    print(f"  Train: {n_train:,} rows  |  Val: {len(y_val):,} rows  "
          f"({time.time()-t0:.1f}s)")
    print()

    # -- 5. Fit global scaler --------------------------------------------------
    print("  Fitting global scaler on full training data ...")
    t0 = time.time()
    scaler = StandardScaler()
    scaler.fit(X_train_all)
    print(f"  Scaler fitted ({time.time()-t0:.1f}s)")
    print()

    # -- 5b. Optionally save scaled parquet ------------------------------------
    scaled_path = PROJECT_ROOT / "training_set_scaled.parquet"
    if scaled_path.exists():
        print(f"  {scaled_path.name} already exists — skipping save.")
    else:
        print(f"  Scaling and saving full dataset to {scaled_path.name} ...")
        t0 = time.time()
        meta_df = pd.read_parquet(TRAINING_FILE, columns=["user", "label"])
        train_mask_save = meta_df["user"].isin(set(train_users))

        X_ts = scaler.transform(X_train_all)
        X_vs = scaler.transform(X_val)

        df_ts = pd.DataFrame(X_ts, columns=FEATURE_COLS)
        df_ts["label"] = le.inverse_transform(y_train_all)
        df_ts["user"]  = meta_df.loc[train_mask_save, "user"].values
        df_ts["split"] = "train"

        df_vs = pd.DataFrame(X_vs, columns=FEATURE_COLS)
        df_vs["label"] = le.inverse_transform(y_val)
        df_vs["user"]  = meta_df.loc[~train_mask_save, "user"].values
        df_vs["split"] = "val"

        pd.concat([df_ts, df_vs], ignore_index=True).to_parquet(
            scaled_path, index=False)
        del X_ts, X_vs, df_ts, df_vs, meta_df
        gc.collect()
        print(f"  Saved {scaled_path.name}  ({time.time()-t0:.1f}s)")
    print()

    # -- 6–8. Scale, fit Nystroem, transform & save chunks ----------------------
    #
    # If kernel chunks already exist on disk from a previous run, skip the
    # expensive scaling + Nystroem fit + transform pipeline entirely.
    # The saved kernel_map.joblib is loaded instead for validation transforms.
    #
    CHUNK = args.chunk_size
    n_chunks = (n_train + CHUNK - 1) // CHUNK
    CHUNKS_DIR.mkdir(exist_ok=True)

    existing_chunks = sorted(CHUNKS_DIR.glob("X_chunk_*.npy"))
    kernel_map_path = MODELS_DIR / "kernel_map.joblib"

    if len(existing_chunks) == n_chunks:
        # ---- Fast path: reuse chunks from a previous run ---------------------
        print(f"  {n_chunks} kernel chunks already on disk — skipping "
              f"scaling/transform.")
        if kernel_map_path.exists() and args.n_components > 0:
            kernel_map = joblib.load(kernel_map_path)
            print(f"  Loaded kernel_map from {kernel_map_path.name}")
        else:
            kernel_map = None

        # Free training features — not needed when chunks are on disk
        del X_train_all
        gc.collect()
    else:
        # ---- Full path: scale, fit kernel map, write chunks ------------------
        print("  Scaling full training set ...")
        t0 = time.time()
        X_train_scaled = scaler.transform(X_train_all)
        del X_train_all
        gc.collect()
        print(f"  Scaled ({time.time()-t0:.1f}s)")

        # Shuffle rows so kernel chunks contain a mix of all users/classes.
        # Important because partial_fit does NOT shuffle internally.
        print("  Shuffling training rows ...")
        rng_shuf = np.random.RandomState(42)
        shuf_idx = rng_shuf.permutation(len(X_train_scaled))
        X_train_scaled = X_train_scaled[shuf_idx]
        y_train_all    = y_train_all[shuf_idx]
        print()

        # Fit Nystroem kernel map
        kernel_map = None
        if args.n_components > 0:
            if args.gamma == "scale":
                gamma_val = 1.0 / (N_FEATURES * X_train_scaled.var())
            else:
                gamma_val = float(args.gamma)

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
            # Stratified sampling: pick equal numbers of each class so
            # the Nystroem basis vectors are well-balanced across classes.
            FIT_PER_CLASS = min(10_000, len(X_train_scaled) // N_CLASSES)
            FIT_SAMPLE = FIT_PER_CLASS * N_CLASSES    # e.g. 60 000
            rng_fit = np.random.RandomState(42)
            fit_idx = []
            for c in range(N_CLASSES):
                c_idx = np.where(y_train_all == c)[0]
                fit_idx.append(rng_fit.choice(c_idx, size=FIT_PER_CLASS,
                                              replace=False))
            fit_idx = np.concatenate(fit_idx)
            rng_fit.shuffle(fit_idx)
            kernel_map.fit(X_train_scaled[fit_idx])
            joblib.dump(kernel_map, kernel_map_path)
            print(f"  Kernel map fitted on {FIT_SAMPLE:,} samples  "
                  f"({time.time()-t0:.1f}s)")
            print()

        # Transform + write chunks to disk
        print(f"  Transforming and writing {n_chunks} chunks to disk ...")
        t0 = time.time()
        for ci in range(n_chunks):
            bstart = ci * CHUNK
            bend   = min(bstart + CHUNK, n_train)
            X_batch = X_train_scaled[bstart:bend]
            if kernel_map is not None:
                X_batch = kernel_map.transform(X_batch).astype(np.float32)
            np.save(CHUNKS_DIR / f"X_chunk_{ci:04d}.npy", X_batch)
            np.save(CHUNKS_DIR / f"y_chunk_{ci:04d}.npy",
                    y_train_all[bstart:bend])
            print(f"    chunk {ci+1}/{n_chunks}  rows {bstart:,}–{bend:,}")
        print(f"  All chunks saved ({time.time()-t0:.1f}s)")

        del X_train_scaled
        gc.collect()

    print()

    # -- 9. Compute balanced class weights & init 15 OvO classifiers -----------
    #
    # class_weight='balanced' is not supported with partial_fit, so we
    # pre-compute the weights from the full training labels and pass them
    # as a dict {class_label: weight}.
    #
    pairs = ovo_pairs(N_CLASSES)
    n_pairs = len(pairs)

    label_counts = np.bincount(y_train_all, minlength=N_CLASSES)
    n_total = label_counts.sum()
    class_weight_dict = {
        c: n_total / (N_CLASSES * label_counts[c])
        for c in range(N_CLASSES)
    }
    print(f"  Class weights: { {le.classes_[c]: round(w, 4) for c, w in class_weight_dict.items()} }")

    classifiers = []
    for _ in pairs:
        classifiers.append(SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=args.alpha,
            learning_rate=args.learning_rate,
            eta0=args.eta0,
            shuffle=True,
            class_weight=class_weight_dict,
            random_state=42,
            verbose=0,
        ))

    print(f"  Hyperparameters:")
    print(f"    alpha (lambda=1/C)   : {args.alpha}")
    print(f"    learning_rate        : {args.learning_rate}")
    print(f"    eta0                 : {args.eta0}")
    print(f"    epochs               : {args.epochs}")
    print(f"    n_components (kernel): {args.n_components}")
    print(f"    chunk_size           : {CHUNK:,}")
    print(f"    multiclass           : OvO ({n_pairs} binary classifiers)")
    print()
    print("-" * 70)

    # -- 10. Chunked OvO training loop -----------------------------------------
    #
    # Outer loop : epochs (full passes over all chunks)
    # Inner loop : chunks (loaded one at a time from disk)
    #   For each chunk, all 15 OvO classifiers call partial_fit()
    #   on the rows belonging to their two classes, then the chunk
    #   is released before the next one is loaded.
    #
    chunk_paths_X = sorted(CHUNKS_DIR.glob("X_chunk_*.npy"))
    chunk_paths_y = sorted(CHUNKS_DIR.glob("y_chunk_*.npy"))

    rng_epoch = np.random.RandomState(42)
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        # Shuffle chunk order each epoch to decorrelate gradients
        order = np.arange(n_chunks)
        rng_epoch.shuffle(order)

        for ci in order:
            X_chunk = np.load(chunk_paths_X[ci], mmap_mode=None)  # load into RAM
            y_chunk = np.load(chunk_paths_y[ci], mmap_mode=None)

            # Pre-compute per-pair boolean masks (cheap, avoids
            # recomputing inside each thread).
            masks = []
            for (c_i, c_j) in pairs:
                masks.append((y_chunk == c_i) | (y_chunk == c_j))

            def _fit_one(idx):
                """Train one OvO classifier on its class-pair rows."""
                mask = masks[idx]
                if mask.sum() == 0:
                    return
                c_i, c_j = pairs[idx]
                classifiers[idx].partial_fit(
                    X_chunk[mask],
                    y_chunk[mask],
                    classes=[c_i, c_j],
                )

            # Run all 15 partial_fit calls in parallel threads.
            # Threads share the same X_chunk/y_chunk memory — no copies.
            # sklearn SGD releases the GIL during the C-level hot loop.
            with ThreadPoolExecutor(max_workers=min(n_pairs, os.cpu_count())) as pool:
                list(pool.map(_fit_one, range(n_pairs)))

            del X_chunk, y_chunk

        elapsed = time.time() - epoch_t0

        # Quick validation accuracy every epoch
        y_pred_val = ovo_predict_batched(
            classifiers, pairs, scaler, kernel_map, X_val, batch_size=8192)
        val_acc = accuracy_score(y_val, y_pred_val)
        print(f"  Epoch {epoch:2d}/{args.epochs}  |  "
              f"val_acc: {val_acc:.4f}  |  {elapsed:.1f}s")

    train_time = time.time() - t0
    print(f"\n  Training completed in {train_time:.1f}s")
    print()

    # -- 11. Final evaluation --------------------------------------------------
    print("  Evaluating on validation set ...")
    y_pred = ovo_predict_batched(
        classifiers, pairs, scaler, kernel_map, X_val, batch_size=8192)
    val_acc = accuracy_score(y_val, y_pred)
    report = classification_report(
        y_val, y_pred,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
    )

    # -- 12. Save models -------------------------------------------------------
    joblib.dump(classifiers, MODELS_DIR / "svm_ovo_classifiers.joblib")
    joblib.dump(pairs, MODELS_DIR / "svm_ovo_pairs.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_best.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    if kernel_map is not None:
        joblib.dump(kernel_map, MODELS_DIR / "kernel_map.joblib")

    print(f"  val_acc: {val_acc:.4f}  |  "
          f"samples: {n_train:,}  |  {train_time:.0f}s")

    # -- 13. Summary -----------------------------------------------------------
    print()
    print("-" * 70)
    print(f"  Best validation accuracy: {val_acc:.4f}")
    print()
    print("  Classification Report (validation set):")
    print(report)

    cm = confusion_matrix(y_val, y_pred)
    print("  Confusion Matrix:")
    print(f"  {'':>12s}", "  ".join(f"{c:>10s}" for c in le.classes_))
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:>12s}", "  ".join(f"{v:10d}" for v in row))

    # Save training history
    history = [{"epochs": args.epochs, "val_acc": val_acc,
                "n_samples": n_train, "time_s": train_time}]
    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history saved to: {history_path}")
    print(f"  Models saved to: {MODELS_DIR.resolve()}")
    print("=" * 70)


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train an approx-RBF SVM via SGD + Nystroem (OvO) "
                    "on EMG-EPN612 windowed features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="Number of full passes over the training data.")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Regularization strength (lambda = 1/C).")
    p.add_argument("--learning-rate", type=str, default=DEFAULT_LEARNING_RATE,
                   choices=["optimal", "constant", "invscaling", "adaptive"],
                   help="Learning rate schedule for SGD.")
    p.add_argument("--eta0", type=float, default=DEFAULT_ETA0,
                   help="Initial / fixed learning rate (used with constant/invscaling).")
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                   help="Fraction of patients to reserve for validation.")
    p.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS,
                   help="Nystroem kernel approximation: number of RBF components. "
                        "Set to 0 to disable (pure linear SVM).")
    p.add_argument("--gamma", type=str, default=DEFAULT_GAMMA,
                   help="RBF kernel bandwidth for Nystroem. "
                        "'scale' = 1/(d*Var(X)), or a float value.")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help="Rows per disk chunk for kernel-mapped data.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
