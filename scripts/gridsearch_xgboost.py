"""
XGBoost Grid Search for EMG-EPN612 dataset.

Performs an exhaustive grid search over key XGBoost hyperparameters using the
same patient-level train/val split as the single-run training script.

Medium grid (~162 combinations):
 - max_depth       : [4, 6, 8, 10]
 - learning_rate   : [0.05, 0.1, 0.2]
 - min_child_weight: [1, 5, 10]
 - subsample       : [0.7, 0.8, 0.9]
 - colsample_bytree: [0.7, 0.8]

Fixed across all runs:
 - n_estimators    : 500 (early stopping selects the real count)
 - gamma           : 0.1
 - reg_alpha       : 0.0
 - reg_lambda      : 1.0
 - early_stopping  : 20 rounds on val mlogloss

Results are saved incrementally to models/xgboost_gridsearch.csv so that
progress is preserved if the run is interrupted.

Usage:
    python scripts/gridsearch_xgboost.py
    python scripts/gridsearch_xgboost.py --no-gpu
    python scripts/gridsearch_xgboost.py --resume   # skip combos already in CSV
"""

import sys
import time
import json
import gc
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# --- Project paths ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_ROOT / "preprocessed_output" / "dataset_SVM.parquet"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_CSV  = MODELS_DIR / "xgboost_gridsearch.csv"

# --- Dataset constants --------------------------------------------------------
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES  = len(ALL_LABELS)

CHANNELS     = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES    = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]

# --- Grid definition ---------------------------------------------------------
GRID = {
    "max_depth":        [4, 6, 8, 10],
    "learning_rate":    [0.05, 0.1, 0.2],
    "min_child_weight": [1, 5, 10],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8],
}

# Fixed hyperparameters (not searched)
FIXED = {
    "n_estimators":         500,   # high cap — early stopping picks the optimum
    "gamma":                0.1,
    "reg_alpha":            0.0,
    "reg_lambda":           1.0,
    "early_stopping_rounds": 20,
}

VAL_FRAC = 0.15


# --- Helpers ------------------------------------------------------------------

def split_patients(all_users: np.ndarray, val_frac: float, seed: int = 42):
    rng = np.random.RandomState(seed)
    users_shuffled = all_users.copy()
    rng.shuffle(users_shuffled)
    n_val = max(1, int(len(users_shuffled) * val_frac))
    return (sorted(users_shuffled[n_val:].tolist()),
            sorted(users_shuffled[:n_val].tolist()))


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    wmap = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([wmap[yi] for yi in y], dtype=np.float32)


def detect_gpu() -> bool:
    try:
        tmp = xgb.XGBClassifier(n_estimators=1, max_depth=1,
                                device="cuda", verbosity=0)
        tmp.fit(np.zeros((2, 2)), np.array([0, 1]))
        return True
    except xgb.core.XGBoostError:
        return False


def combo_key(combo: dict) -> str:
    """Deterministic string key for a parameter combo (for resume lookup)."""
    return "|".join(f"{k}={combo[k]}" for k in sorted(combo))


# --- Main ---------------------------------------------------------------------

def run_grid(args):
    print("=" * 70)
    print("  XGBoost Grid Search  -  EMG-EPN612")
    print("=" * 70)

    # -- Load data -------------------------------------------------------------
    if not DATASET_FILE.exists():
        print(f"ERROR: {DATASET_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Loading dataset ...")
    t0 = time.time()
    df = pd.read_parquet(DATASET_FILE)
    print(f"  {len(df):,} rows loaded ({time.time()-t0:.1f}s)")

    # -- Patient-level split ---------------------------------------------------
    all_users = df["user"].unique()
    train_users, val_users = split_patients(all_users, VAL_FRAC)
    print(f"  Train patients: {len(train_users)}  |  Val patients: {len(val_users)}")

    le = LabelEncoder()
    le.fit(ALL_LABELS)

    train_mask = df["user"].isin(set(train_users))
    X_train = df.loc[train_mask, FEATURE_COLS].values.astype(np.float32)
    y_train = le.transform(df.loc[train_mask, "label"].values)
    X_val   = df.loc[~train_mask, FEATURE_COLS].values.astype(np.float32)
    y_val   = le.transform(df.loc[~train_mask, "label"].values)
    del df; gc.collect()

    sw_train = compute_sample_weights(y_train)
    sw_val   = compute_sample_weights(y_val)
    print(f"  Train: {len(y_train):,}  |  Val: {len(y_val):,}")

    # -- Device ----------------------------------------------------------------
    device = "cpu" if args.no_gpu else ("cuda" if detect_gpu() else "cpu")
    print(f"  Device: {device}\n")

    # -- Build combo list ------------------------------------------------------
    keys = sorted(GRID.keys())
    all_combos = [dict(zip(keys, vals)) for vals in product(*(GRID[k] for k in keys))]
    total = len(all_combos)
    print(f"  Total grid combinations: {total}")

    # -- CSV column order -------------------------------------------------------
    csv_cols = keys + [
        "n_estimators_cap", "best_iteration", "best_val_mlogloss",
        "train_acc", "val_acc", "train_time_s",
    ]

    # -- Resume support --------------------------------------------------------
    #    Always preserve existing results.  --resume skips them; without it
    #    they stay in the CSV but every combo is re-run (and appended).
    done_keys: set = set()
    if RESULTS_CSV.exists():
        try:
            prev = pd.read_csv(RESULTS_CSV)
            if not args.no_resume and len(prev) > 0:
                for _, row in prev.iterrows():
                    ck = combo_key({k: row[k] for k in keys})
                    done_keys.add(ck)
                print(f"  Resuming — {len(done_keys)} combos already completed, "
                      f"{total - len(done_keys)} remaining.")
            else:
                print(f"  Found existing CSV with {len(prev)} rows (will append).")
        except Exception:
            pass  # corrupted CSV — just start fresh

    # Write header only if file doesn't exist or is empty
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        with open(RESULTS_CSV, "w") as f:
            f.write(",".join(csv_cols) + "\n")

    # -- Run grid --------------------------------------------------------------
    best_val_acc = 0.0
    best_combo   = None
    global_t0    = time.time()

    for idx, combo in enumerate(all_combos, 1):
        ck = combo_key(combo)
        if ck in done_keys:
            continue

        # Short label for logging
        tag = "  ".join(f"{k[:5]}={combo[k]}" for k in keys)
        print(f"\n  [{idx}/{total}]  {tag}")

        clf = xgb.XGBClassifier(
            **combo,
            **{k: v for k, v in FIXED.items() if k != "early_stopping_rounds"},
            objective="multi:softprob",
            num_class=N_CLASSES,
            eval_metric="mlogloss",
            tree_method="hist",
            device=device,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=FIXED["early_stopping_rounds"],
        )

        t0 = time.time()
        clf.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[sw_val],
            verbose=False,
        )
        elapsed = time.time() - t0

        best_iter  = clf.best_iteration
        best_loss  = clf.best_score
        val_acc    = accuracy_score(y_val, clf.predict(X_val))
        train_acc  = accuracy_score(y_train, clf.predict(X_train))

        # Print result
        print(f"           best_iter={best_iter:3d}  "
              f"val_mlogloss={best_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"train_acc={train_acc:.4f}  "
              f"({elapsed:.1f}s)")

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_combo   = {**combo, "best_iteration": best_iter,
                            "val_acc": val_acc, "train_acc": train_acc}
            # Save best model
            clf.save_model(str(MODELS_DIR / "xgboost_best_gridsearch.json"))
            joblib.dump(le, MODELS_DIR / "xgboost_label_encoder.joblib")
            print(f"           *** NEW BEST — val_acc={val_acc:.4f} ***")

        # Append row to CSV immediately (crash-safe)
        row_vals = [combo[k] for k in keys] + [
            FIXED["n_estimators"], best_iter, best_loss,
            train_acc, val_acc, round(elapsed, 1),
        ]
        with open(RESULTS_CSV, "a") as f:
            f.write(",".join(str(v) for v in row_vals) + "\n")

        del clf; gc.collect()

    total_time = time.time() - global_t0

    # -- Final summary ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  Grid search completed in {total_time/60:.1f} min")
    print(f"  Results saved to: {RESULTS_CSV}")
    print(f"\n  Best validation accuracy: {best_val_acc:.4f}")
    if best_combo:
        for k, v in best_combo.items():
            print(f"    {k:>20s} : {v}")
    print(f"\n  Best model saved to: {MODELS_DIR / 'xgboost_best_gridsearch.json'}")
    print("=" * 70)


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="XGBoost grid search on EMG-EPN612.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-gpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-run all combos even if results CSV exists.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_grid(args)
