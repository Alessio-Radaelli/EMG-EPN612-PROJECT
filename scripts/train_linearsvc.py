"""
Linear SVM Training via LinearSVC for EMG-EPN612 dataset.

Uses sklearn's LinearSVC (liblinear) which solves the primal SVM problem
directly on the 72 TD9 features — no kernel approximation needed.

LinearSVC advantages over SGDClassifier + Nystroem:
 - Exact solver (coordinate descent), not stochastic — no learning rate tuning
 - Works directly on 72 features — no Nystroem approximation bottleneck
 - Built-in OvR multiclass (no manual OvO + partial_fit needed)
 - Handles millions of rows efficiently via the liblinear C implementation

Usage:
    cd "EMG-EPN612 project"
    python scripts/train_linearsvc.py
    python scripts/train_linearsvc.py --C 1.0
    python scripts/train_linearsvc.py --C 10.0 --max-iter 5000
"""

import sys
import time
import json
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
DEFAULT_C        = 1.0     # inverse regularization strength (higher = less reg)
DEFAULT_MAX_ITER = 2000    # max iterations for the solver
DEFAULT_VAL_FRAC = 0.15    # fraction of patients held out for validation


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


# --- Main training ------------------------------------------------------------

def train(args):
    print("=" * 70)
    print("  LinearSVC Training  -  EMG-EPN612")
    print("=" * 70)

    # -- 1. Load full dataset --------------------------------------------------
    if not TRAINING_FILE.exists():
        print(f"ERROR: {TRAINING_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Loading dataset from {TRAINING_FILE.name} ...")
    t0 = time.time()
    df = pd.read_parquet(TRAINING_FILE)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} cols  "
          f"({time.time()-t0:.1f}s)")

    # -- 2. Patient-level train/val split --------------------------------------
    all_users = df["user"].unique()
    train_users, val_users = split_patients(all_users, args.val_frac)
    print(f"\n  Total patients     : {len(all_users)}")
    print(f"  Training patients  : {len(train_users)}")
    print(f"  Validation patients: {len(val_users)}")

    # -- 3. Encode labels ------------------------------------------------------
    le = LabelEncoder()
    le.fit(ALL_LABELS)
    classes = le.transform(ALL_LABELS)
    print(f"\n  Classes ({N_CLASSES}): {dict(zip(le.classes_, classes))}")

    # -- 4. Build train/val arrays ---------------------------------------------
    print("\n  Splitting data by patient ...")
    t0 = time.time()
    train_mask = df["user"].isin(set(train_users))

    X_train = df.loc[train_mask, FEATURE_COLS].values.astype(np.float32)
    y_train = le.transform(df.loc[train_mask, "label"].values)

    X_val = df.loc[~train_mask, FEATURE_COLS].values.astype(np.float32)
    y_val = le.transform(df.loc[~train_mask, "label"].values)

    del df
    gc.collect()

    n_train = len(y_train)
    print(f"  Train: {n_train:,} rows  |  Val: {len(y_val):,} rows  "
          f"({time.time()-t0:.1f}s)")

    # -- 5. Fit global scaler --------------------------------------------------
    print("\n  Fitting global scaler on training data ...")
    t0 = time.time()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    print(f"  Scaler fitted & data scaled ({time.time()-t0:.1f}s)")

    # -- 6. Train LinearSVC ----------------------------------------------------
    print(f"\n  Training LinearSVC ...")
    print(f"    C (inv. regularization): {args.C}")
    print(f"    max_iter               : {args.max_iter}")
    print(f"    dual                   : auto")
    print(f"    class_weight           : balanced")
    print(f"    multi_class            : ovr (one-vs-rest)")
    print(f"    n_train                : {n_train:,}")
    print(f"    n_features             : {N_FEATURES}")
    print()

    t0 = time.time()
    clf = LinearSVC(
        C=args.C,
        dual="auto",               # liblinear picks primal or dual automatically
        class_weight="balanced",    # handle class imbalance
        max_iter=args.max_iter,
        random_state=42,
        verbose=1,                  # show convergence progress
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"\n  Training completed in {train_time:.1f}s")

    # -- 7. Evaluate -----------------------------------------------------------
    print("\n  Evaluating on validation set ...")
    t0 = time.time()
    y_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    print(f"  Prediction done ({time.time()-t0:.1f}s)")

    report = classification_report(
        y_val, y_pred,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
    )

    # -- 8. Save models --------------------------------------------------------
    joblib.dump(clf,    MODELS_DIR / "linearsvc_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "linearsvc_scaler.joblib")
    joblib.dump(le,     MODELS_DIR / "linearsvc_label_encoder.joblib")

    # -- 9. Summary ------------------------------------------------------------
    print()
    print("-" * 70)
    print(f"  Validation accuracy: {val_acc:.4f}")
    print()
    print("  Classification Report (validation set):")
    print(report)

    cm = confusion_matrix(y_val, y_pred)
    print("  Confusion Matrix:")
    print(f"  {'':>12s}", "  ".join(f"{c:>10s}" for c in le.classes_))
    for i, row_vals in enumerate(cm):
        print(f"  {le.classes_[i]:>12s}", "  ".join(f"{v:10d}" for v in row_vals))

    # Save training history
    history = {
        "model": "LinearSVC",
        "C": args.C,
        "max_iter": args.max_iter,
        "val_acc": float(val_acc),
        "n_train": n_train,
        "n_val": len(y_val),
        "train_time_s": round(train_time, 1),
    }
    history_path = MODELS_DIR / "linearsvc_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  History saved to : {history_path}")
    print(f"  Models saved to  : {MODELS_DIR.resolve()}")
    print("=" * 70)


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a LinearSVC on EMG-EPN612 windowed features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--C", type=float, default=DEFAULT_C,
                   help="Inverse regularization strength. "
                        "Larger C = less regularization = more complex model.")
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                   help="Maximum solver iterations.")
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                   help="Fraction of patients to reserve for validation.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
