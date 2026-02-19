"""
XGBoost Training & Validation for EMG-EPN612 dataset.

Trains a gradient-boosted tree ensemble (XGBClassifier) on pre-computed
windowed TD9 features from the SVM-ready parquet file (~1M rows, 72 feats).

Key design choices:
 - Patient-level train/val split to prevent data leakage.
 - GPU acceleration via 'cuda' device when available, CPU fallback.
 - Early stopping on the validation set to avoid overfitting.
 - Class-weight balancing via `sample_weight` computed from inverse
   class frequency (mirrors `class_weight='balanced'` in sklearn).
 - Saves model, label encoder, and full training history to models/.

Usage:
    cd "EMG-EPN612 project"
    python scripts/train_xgboost.py
    python scripts/train_xgboost.py --n-estimators 500 --max-depth 8 --lr 0.1
    python scripts/train_xgboost.py --val-frac 0.2 --no-gpu
"""

import sys
import time
import json
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# --- Project paths (relative to project root) --------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATASET_FILE  = PROJECT_ROOT / "preprocessed_output" / "dataset_SVM.parquet"
MODELS_DIR    = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# --- Dataset constants --------------------------------------------------------
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES  = len(ALL_LABELS)

# 72 feature columns: 8 channels x 9 TD9 features
CHANNELS     = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES    = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
N_FEATURES   = len(FEATURE_COLS)  # 72

# --- Default hyper-parameters -------------------------------------------------
DEFAULT_N_ESTIMATORS   = 300       # number of boosting rounds
DEFAULT_MAX_DEPTH      = 6        # max tree depth
DEFAULT_LEARNING_RATE  = 0.1      # shrinkage / step size
DEFAULT_SUBSAMPLE      = 0.8      # row subsampling per tree
DEFAULT_COLSAMPLE      = 0.8      # column subsampling per tree
DEFAULT_MIN_CHILD_W    = 5        # min sum of instance weight in a child
DEFAULT_GAMMA          = 0.1      # min loss reduction for a split
DEFAULT_REG_ALPHA      = 0.0      # L1 regularization
DEFAULT_REG_LAMBDA     = 1.0      # L2 regularization
DEFAULT_EARLY_STOP     = 20       # early stopping patience (rounds)
DEFAULT_VAL_FRAC       = 0.15     # fraction of patients held out


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


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute per-sample weights to balance class frequencies.

    Equivalent to sklearn's class_weight='balanced':
        w_c = n_samples / (n_classes * n_c)
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    weight_map = {c: n_samples / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y], dtype=np.float32)


def detect_gpu() -> bool:
    """Return True if XGBoost can use CUDA."""
    try:
        _tmp = xgb.XGBClassifier(
            n_estimators=1, max_depth=1, device="cuda", verbosity=0
        )
        _tmp.fit(np.zeros((2, 2)), np.array([0, 1]))
        return True
    except xgb.core.XGBoostError:
        return False


# --- Main training ------------------------------------------------------------

def train(args):
    print("=" * 70)
    print("  XGBoost Training  -  EMG-EPN612")
    print("=" * 70)

    # -- 1. Load dataset -------------------------------------------------------
    if not DATASET_FILE.exists():
        print(f"ERROR: {DATASET_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Loading dataset from {DATASET_FILE.name} ...")
    t0 = time.time()
    df = pd.read_parquet(DATASET_FILE)
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
    n_val   = len(y_val)
    print(f"  Train: {n_train:,} rows  |  Val: {n_val:,} rows  "
          f"({time.time()-t0:.1f}s)")

    # -- 5. Compute balanced sample weights ------------------------------------
    sample_w_train = compute_sample_weights(y_train)
    sample_w_val   = compute_sample_weights(y_val)
    print("  Balanced sample weights computed.")

    # -- 6. Device selection ---------------------------------------------------
    if args.no_gpu:
        device = "cpu"
    else:
        print("\n  Probing GPU availability ...")
        device = "cuda" if detect_gpu() else "cpu"
    tree_method = "hist"   # histogram-based (works for both CPU and GPU)
    print(f"  Device      : {device}")
    print(f"  Tree method : {tree_method}")

    # -- 7. Train XGBoost ------------------------------------------------------
    print(f"\n  Training XGBClassifier ...")
    print(f"    n_estimators  : {args.n_estimators}")
    print(f"    max_depth     : {args.max_depth}")
    print(f"    learning_rate : {args.lr}")
    print(f"    subsample     : {args.subsample}")
    print(f"    colsample     : {args.colsample}")
    print(f"    min_child_w   : {args.min_child_weight}")
    print(f"    gamma         : {args.gamma}")
    print(f"    reg_alpha     : {args.reg_alpha}")
    print(f"    reg_lambda    : {args.reg_lambda}")
    print(f"    early_stop    : {args.early_stop}")
    print()

    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        tree_method=tree_method,
        device=device,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=args.early_stop,
    )

    t0 = time.time()
    clf.fit(
        X_train, y_train,
        sample_weight=sample_w_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight_eval_set=[sample_w_train, sample_w_val],
        verbose=True,
    )
    train_time = time.time() - t0
    best_iteration = clf.best_iteration
    best_score     = clf.best_score
    print(f"\n  Training completed in {train_time:.1f}s")
    print(f"  Best iteration: {best_iteration}  |  Best val mlogloss: {best_score:.6f}")

    # -- 8. Evaluate -----------------------------------------------------------
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

    # Also compute training accuracy for reference
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # -- 9. Save models --------------------------------------------------------
    model_path = MODELS_DIR / "xgboost_model.json"
    clf.save_model(str(model_path))
    joblib.dump(le, MODELS_DIR / "xgboost_label_encoder.joblib")

    # -- 10. Summary -----------------------------------------------------------
    print()
    print("-" * 70)
    print(f"  Training accuracy  : {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print()
    print("  Classification Report (validation set):")
    print(report)

    cm = confusion_matrix(y_val, y_pred)
    print("  Confusion Matrix:")
    print(f"  {'':>12s}", "  ".join(f"{c:>10s}" for c in le.classes_))
    for i, row_vals in enumerate(cm):
        print(f"  {le.classes_[i]:>12s}", "  ".join(f"{v:10d}" for v in row_vals))

    # Feature importance (top 15)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\n  Top 15 features by importance (gain):")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {FEATURE_COLS[idx]:>15s}  {importances[idx]:.4f}")

    # -- 11. Save training history ---------------------------------------------
    # Extract eval results from the evals_result
    evals_result = clf.evals_result()
    history = {
        "model": "XGBClassifier",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "early_stopping_rounds": args.early_stop,
        "best_iteration": int(best_iteration),
        "best_val_mlogloss": float(best_score),
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "n_train": n_train,
        "n_val": n_val,
        "device": device,
        "train_time_s": round(train_time, 1),
        "train_mlogloss": evals_result["validation_0"]["mlogloss"],
        "val_mlogloss": evals_result["validation_1"]["mlogloss"],
    }
    history_path = MODELS_DIR / "xgboost_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  History saved to : {history_path}")
    print(f"  Model saved to   : {model_path}")
    print(f"  Label encoder    : {MODELS_DIR / 'xgboost_label_encoder.joblib'}")
    print("=" * 70)


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train an XGBoost classifier on EMG-EPN612 windowed features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS,
                   help="Max number of boosting rounds.")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                   help="Maximum tree depth.")
    p.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                   help="Learning rate (shrinkage).")
    p.add_argument("--subsample", type=float, default=DEFAULT_SUBSAMPLE,
                   help="Row subsampling ratio per tree.")
    p.add_argument("--colsample", type=float, default=DEFAULT_COLSAMPLE,
                   help="Column subsampling ratio per tree.")
    p.add_argument("--min-child-weight", type=float, default=DEFAULT_MIN_CHILD_W,
                   help="Min sum of instance weight in a leaf.")
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                   help="Min loss reduction to make a further split.")
    p.add_argument("--reg-alpha", type=float, default=DEFAULT_REG_ALPHA,
                   help="L1 regularization on weights.")
    p.add_argument("--reg-lambda", type=float, default=DEFAULT_REG_LAMBDA,
                   help="L2 regularization on weights.")
    p.add_argument("--early-stop", type=int, default=DEFAULT_EARLY_STOP,
                   help="Early-stopping patience (boosting rounds).")
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                   help="Fraction of patients to reserve for validation.")
    p.add_argument("--no-gpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
