"""
Feature Selection (Cross-Validated) for EMG-EPN612 dataset.

Evaluates all 72 TD9 features (8 channels × 9 features) using five
complementary selection methods, each cross-validated over 5 patient-level
folds (GroupKFold on `user`) for robust, split-independent rankings.

Methods:
  1. Pearson Correlation Filter  — drops features with |ρ| > 0.95 (redundancy)
  2. ANOVA F-value               — univariate F-test per feature
  3. Mutual Information           — non-linear dependency with label
  4. XGBoost Feature Importance   — gain-based importance from a quick model
  5. Recursive Feature Elimination — RFE with LinearSVC estimator

Outputs:
  • Console summary comparing rank-based and score-based consensus.
  • models/feature_ranking_ranks.csv  — rank-based (Borda count) ranking.
  • models/feature_ranking_scores.csv — score-based (min-max norm) ranking.
  • preprocessed_output/dataset_TRAINING_reduced{36,18}.parquet (default on).

Usage:
    cd "EMG-EPN612 project"
    python scripts/feature_selection.py
    python scripts/feature_selection.py --top-k 36 --no-save-reduced
    python scripts/feature_selection.py --n-folds 10 --top-k 24
"""

import sys
import os
import time
import gc
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif,
    RFE,
)
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

# --- Project paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SVM_FILE     = PROJECT_ROOT / "preprocessed_output" / "dataset_TRAINING.parquet"
DTW_FILE     = PROJECT_ROOT / "preprocessed_output" / "dataset_DTW.parquet"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# --- Dataset constants -------------------------------------------------------
ALL_LABELS   = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES    = len(ALL_LABELS)
CHANNELS     = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES    = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
N_FEATURES   = len(FEATURE_COLS)   # 72

# --- Defaults ----------------------------------------------------------------
DEFAULT_TOP_K      = 36       # half of 72
DEFAULT_N_FOLDS    = 5
DEFAULT_CORR_THR   = 0.95
DEFAULT_SUBSAMPLE  = 200_000  # max rows for slow methods (MI, RFE); 0 = all


# =============================================================================
# Helpers
# =============================================================================

def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Handles constant arrays."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return np.ones_like(scores)
    return (scores - mn) / (mx - mn)


def _format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# =============================================================================
# Method 1 — Pearson Correlation Filter (redundancy removal)
# =============================================================================

def correlation_filter(X_train: np.ndarray, threshold: float = 0.95):
    """Score features by how many others they are highly correlated with.

    Features that are strongly correlated with many others get a LOW score
    (they are redundant).  Returns a 72-element score array where a higher
    score means the feature is LESS redundant and should be kept.
    """
    corr = np.abs(np.corrcoef(X_train, rowvar=False))  # (72, 72)
    np.fill_diagonal(corr, 0)
    n_high = (corr > threshold).sum(axis=1)            # per feature
    # Invert: fewer high-corr partners → higher score
    scores = 1.0 - n_high / max(n_high.max(), 1)
    return scores.astype(np.float64)


# =============================================================================
# Method 2 — ANOVA F-value
# =============================================================================

def anova_scores(X_train: np.ndarray, y_train: np.ndarray):
    """Per-feature ANOVA F-statistic (higher = more discriminative)."""
    f_vals, _ = f_classif(X_train, y_train)
    f_vals = np.nan_to_num(f_vals, nan=0.0)
    return f_vals


# =============================================================================
# Method 3 — Mutual Information
# =============================================================================

def mi_scores(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    """Per-feature mutual information with the class label."""
    mi = mutual_info_classif(
        X_train, y_train,
        discrete_features=False,
        n_neighbors=5,
        random_state=seed,
        n_jobs=-1,
    )
    return mi


# =============================================================================
# Method 4 — XGBoost Feature Importance (gain)
# =============================================================================

def xgb_importance(X_train: np.ndarray, y_train: np.ndarray,
                   seed: int = 42):
    """Train a quick XGBoost and return feature importances (gain)."""
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf.feature_importances_


# =============================================================================
# Method 5 — Recursive Feature Elimination (LinearSVC)
# =============================================================================

def rfe_ranking(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    """RFE ranking using LinearSVC.  Returns scores (higher = selected earlier)."""
    estimator = LinearSVC(
        max_iter=5000,
        dual="auto",
        random_state=seed,
        class_weight="balanced",
    )
    rfe = RFE(estimator, n_features_to_select=1, step=5)
    rfe.fit(X_train, y_train)
    # rfe.ranking_ is 1..n_features (1 = selected last → best).
    # Invert so that best feature gets highest score.
    scores = (N_FEATURES - rfe.ranking_ + 1).astype(np.float64)
    return scores


# =============================================================================
# Helpers — Stratified subsample for slow methods
# =============================================================================

def _stratified_subsample(X, y, max_rows, seed=42):
    """Return a stratified subsample of at most `max_rows` rows.

    Preserves class proportions.  If max_rows >= len(y), returns originals.
    """
    if max_rows <= 0 or max_rows >= len(y):
        return X, y
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    frac = max_rows / len(y)
    idx = []
    for c, cnt in zip(classes, counts):
        n_take = max(1, int(cnt * frac))
        c_idx = np.where(y == c)[0]
        idx.append(rng.choice(c_idx, size=min(n_take, len(c_idx)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


# =============================================================================
# Single-fold worker  (runs in a child process)
# =============================================================================

_LABEL_MAP = {
    "corr_filter":  "Correlation filter",
    "anova_f":      "ANOVA F-value",
    "mutual_info":  "Mutual Information",
    "xgb_gain":     "XGBoost importance",
    "rfe_rank":     "RFE (LinearSVC)",
}

def _run_one_fold(fold_idx, train_idx, X, y, corr_thr, subsample):
    """Execute all 5 methods for a single fold — called in a child process."""
    fold_num = fold_idx + 1
    X_tr = X[train_idx]
    y_tr = y[train_idx]

    # Subsample for the expensive methods
    X_sub, y_sub = _stratified_subsample(X_tr, y_tr, subsample,
                                          seed=42 + fold_idx)
    sub_tag = (f", sub={len(y_sub):,}" if len(y_sub) < len(y_tr) else "")

    t_fold = time.time()
    scores = {}
    timings = {}

    for name, fn, args in [
        ("corr_filter",  correlation_filter, (X_tr, corr_thr)),
        ("anova_f",      anova_scores,       (X_tr, y_tr)),
        ("mutual_info",  mi_scores,          (X_sub, y_sub, 42 + fold_idx)),
        ("xgb_gain",     xgb_importance,     (X_tr, y_tr, 42 + fold_idx)),
        ("rfe_rank",     rfe_ranking,         (X_sub, y_sub, 42 + fold_idx)),
    ]:
        t0 = time.time()
        scores[name] = fn(*args)
        timings[name] = time.time() - t0

    total_time = time.time() - t_fold
    return {
        "fold_idx":   fold_idx,
        "fold_num":   fold_num,
        "n_train":    len(train_idx),
        "sub_tag":    sub_tag,
        "scores":     scores,
        "timings":    timings,
        "total_time": total_time,
    }


# =============================================================================
# Cross-Validated Feature Selection  (parallelised across folds)
# =============================================================================

METHOD_NAMES = [
    "corr_filter",
    "anova_f",
    "mutual_info",
    "xgb_gain",
    "rfe_rank",
]

def run_cv_feature_selection(X, y, groups, n_folds=5, corr_thr=0.95,
                             subsample=DEFAULT_SUBSAMPLE):
    """Run all 5 methods for each fold, with folds in parallel processes.

    Parameters
    ----------
    X         : np.ndarray (n_samples, 72)
    y         : np.ndarray (n_samples,) — integer-encoded labels
    groups    : np.ndarray (n_samples,) — user IDs for GroupKFold
    n_folds   : int
    corr_thr  : float — threshold for correlation filter
    subsample : int — max training rows for slow methods (MI, RFE). 0 = all.

    Returns
    -------
    avg_scores : dict  { method_name: np.ndarray(72,) }
    """
    gkf = GroupKFold(n_splits=n_folds)

    # Pre-compute fold splits (indices only — cheap)
    splits = list(gkf.split(X, y, groups))

    fold_scores = {m: [] for m in METHOD_NAMES}
    n_workers = min(n_folds, 2)  # Limit to 2 workers to reduce memory usage

    print(f"  Launching {n_folds} folds across {n_workers} worker processes ...")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _run_one_fold, fold_idx, train_idx, X, y, corr_thr, subsample
            ): fold_idx
            for fold_idx, (train_idx, _val_idx) in enumerate(splits)
        }

        for future in as_completed(futures):
            r = future.result()
            fn = r["fold_num"]
            print(f"\n  ── Fold {fn}/{n_folds}  "
                  f"({r['n_train']:,} train rows{r['sub_tag']}) ──")
            for i, m in enumerate(METHOD_NAMES, 1):
                fold_scores[m].append(r["scores"][m])
                print(f"    [{i}/5] {_LABEL_MAP[m]:<22s}  "
                      f"({r['timings'][m]:.1f}s)")
            print(f"    Fold {fn} done  ({_format_duration(r['total_time'])})")

    # Average across folds
    avg_scores = {}
    for m in METHOD_NAMES:
        stacked = np.stack(fold_scores[m])        # (n_folds, 72)
        avg_scores[m] = stacked.mean(axis=0)      # (72,)

    return avg_scores


# =============================================================================
# Consensus Ranking
# =============================================================================

def build_score_ranking(avg_scores: dict, feature_cols: list):
    """Consensus via min-max normalised scores (original method).

    Each method's raw scores are normalised to [0, 1], then averaged.
    Higher consensus_score = more important.
    """
    df = pd.DataFrame({"feature": feature_cols})

    norm_cols = []
    for method, raw_scores in avg_scores.items():
        col_raw = f"{method}_raw"
        col_norm = f"{method}_norm"
        df[col_raw] = raw_scores
        df[col_norm] = _normalise_scores(raw_scores)
        norm_cols.append(col_norm)

    df["consensus_score"] = df[norm_cols].mean(axis=1)
    df = df.sort_values("consensus_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    return df


def build_rank_voting(avg_scores: dict, feature_cols: list):
    """Consensus via rank-based voting (Borda count).

    Each method's raw scores are converted to per-method ranks (1 = best),
    then averaged.  Lower avg_rank = more important.  Immune to outliers
    and scale differences between methods.
    """
    df = pd.DataFrame({"feature": feature_cols})

    rank_cols = []
    for method, raw_scores in avg_scores.items():
        col_raw = f"{method}_raw"
        col_mrank = f"{method}_mrank"
        df[col_raw] = raw_scores
        df[col_mrank] = df[col_raw].rank(ascending=False, method="min").astype(int)
        rank_cols.append(col_mrank)

    df["avg_rank"] = df[rank_cols].mean(axis=1)
    df = df.sort_values("avg_rank", ascending=True).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    return df


# =============================================================================
# Quick Validation (top-K vs full features)
# =============================================================================

def quick_validation(X, y, groups, feature_sets, n_folds=5):
    """Compare XGBoost accuracy across multiple feature subsets using the
    same GroupKFold splits for a fair comparison.

    Parameters
    ----------
    feature_sets : list of (label, feature_name_list) tuples
    n_folds      : int

    Returns
    -------
    results : list of (label, n_features, mean_acc, std_acc)
    """
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(X, y, groups))

    results = []
    for label, feats in feature_sets:
        fidx = [FEATURE_COLS.index(f) for f in feats]
        accs = []
        for train_idx, val_idx in splits:
            clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", num_class=N_CLASSES,
                eval_metric="mlogloss", tree_method="hist",
                random_state=42, n_jobs=-1, verbosity=0,
            )
            clf.fit(X[train_idx][:, fidx], y[train_idx])
            accs.append(accuracy_score(y[val_idx], clf.predict(X[val_idx][:, fidx])))
            del clf
            gc.collect()
        results.append((label, len(feats), np.mean(accs), np.std(accs)))

    return results


# =============================================================================
# Main
# =============================================================================

def main(args):
    t_start = time.time()

    print("=" * 70)
    print("  Feature Selection (Cross-Validated)  —  EMG-EPN612")
    print("=" * 70)

    # -- 1. Load dataset -------------------------------------------------------
    if not SVM_FILE.exists():
        print(f"ERROR: {SVM_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Loading {SVM_FILE.name} ...")
    t0 = time.time()
    df = pd.read_parquet(SVM_FILE)
    print(f"  {len(df):,} rows × {len(df.columns)} cols  ({time.time()-t0:.1f}s)")

    # -- 2. Encode labels ------------------------------------------------------
    le = LabelEncoder()
    le.fit(ALL_LABELS)

    X      = df[FEATURE_COLS].values.astype(np.float32)
    y      = le.transform(df["label"].values)
    groups = df["user"].values

    n_users = len(np.unique(groups))
    print(f"  Features  : {N_FEATURES}")
    print(f"  Classes   : {N_CLASSES}  {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"  Patients  : {n_users}")
    print(f"  Folds     : {args.n_folds}  (patient-level GroupKFold)")

    del df
    gc.collect()

    # -- 3. Feature ranking (compute or load from CSV) -------------------------
    csv_ranks  = MODELS_DIR / "feature_ranking_ranks.csv"
    csv_scores = MODELS_DIR / "feature_ranking_scores.csv"

    if args.from_csv:
        missing = [p for p in (csv_ranks, csv_scores) if not p.exists()]
        if missing:
            print(f"ERROR: missing {[p.name for p in missing]}. "
                  f"Run without --from-csv first.", file=sys.stderr)
            sys.exit(1)
        print(f"\n  Loading pre-computed rankings ...")
        df_rnk = pd.read_csv(csv_ranks)
        df_scr = pd.read_csv(csv_scores)
        print(f"  {len(df_rnk)} features loaded  (skipping CV scoring)")
    else:
        print(f"\n{'─'*70}")
        print("  Running cross-validated feature selection ...")
        print(f"{'─'*70}")

        avg_scores = run_cv_feature_selection(
            X, y, groups,
            n_folds=args.n_folds,
            corr_thr=args.corr_thr,
            subsample=args.subsample,
        )

        print(f"\n{'─'*70}")
        print("  Building consensus rankings (rank-based + score-based) ...")
        print(f"{'─'*70}")

        df_rnk = build_rank_voting(avg_scores, FEATURE_COLS)
        df_scr = build_score_ranking(avg_scores, FEATURE_COLS)

        df_rnk.to_csv(csv_ranks,  index=False, float_format="%.6f")
        df_scr.to_csv(csv_scores, index=False, float_format="%.6f")
        print(f"  Saved: {csv_ranks.name}")
        print(f"  Saved: {csv_scores.name}")

    # -- 4. Print rank-based table (primary) -----------------------------------
    top_k = min(args.top_k, N_FEATURES)

    print(f"\n{'─'*70}")
    print(f"  RANK-BASED consensus (Borda count)  —  top {top_k}")
    print(f"{'─'*70}\n")
    print(f"  {'Rk':>4s}  {'Feature':<15s}  "
          f"{'Corr':>5s}  {'ANOVA':>5s}  {'MI':>5s}  "
          f"{'XGB':>5s}  {'RFE':>5s}  {'AvgRk':>6s}")
    print(f"  {'─'*4}  {'─'*15}  " + "  ".join(["─"*5]*5) + f"  {'─'*6}")
    for _, row in df_rnk.head(top_k).iterrows():
        print(f"  {int(row['rank']):4d}  {row['feature']:<15s}  "
              f"{int(row['corr_filter_mrank']):5d}  "
              f"{int(row['anova_f_mrank']):5d}  "
              f"{int(row['mutual_info_mrank']):5d}  "
              f"{int(row['xgb_gain_mrank']):5d}  "
              f"{int(row['rfe_rank_mrank']):5d}  "
              f"{row['avg_rank']:6.1f}")

    print(f"\n{'─'*70}")
    print(f"  SCORE-BASED consensus (min-max norm)  —  top {top_k}")
    print(f"{'─'*70}\n")
    print(f"  {'Rk':>4s}  {'Feature':<15s}  "
          f"{'Corr':>6s}  {'ANOVA':>6s}  {'MI':>6s}  "
          f"{'XGB':>6s}  {'RFE':>6s}  {'Score':>6s}")
    print(f"  {'─'*4}  {'─'*15}  " + "  ".join(["─"*6]*6))
    for _, row in df_scr.head(top_k).iterrows():
        print(f"  {int(row['rank']):4d}  {row['feature']:<15s}  "
              f"{row['corr_filter_norm']:6.3f}  {row['anova_f_norm']:6.3f}  "
              f"{row['mutual_info_norm']:6.3f}  {row['xgb_gain_norm']:6.3f}  "
              f"{row['rfe_rank_norm']:6.3f}  {row['consensus_score']:6.3f}")

    # -- 4b. Comparison --------------------------------------------------------
    for k in (36, 18):
        rnk_set = set(df_rnk.head(k)["feature"])
        scr_set = set(df_scr.head(k)["feature"])
        only_rnk = sorted(rnk_set - scr_set)
        only_scr = sorted(scr_set - rnk_set)
        if not only_rnk and not only_scr:
            print(f"\n  Top-{k}: both methods agree on the same features.")
        else:
            print(f"\n  Top-{k}: {len(only_rnk)} feature(s) differ")
            if only_rnk:
                print(f"    Rank-only : {', '.join(only_rnk)}")
            if only_scr:
                print(f"    Score-only: {', '.join(only_scr)}")

    # -- 5. Quick validation (using rank-based as primary) ---------------------
    print(f"\n{'─'*70}")
    print(f"  Quick Validation: XGBoost accuracy (rank-based feature sets)")
    print(f"  ({args.n_folds}-fold GroupKFold)")
    print(f"{'─'*70}")

    feature_sets = [
        ("All 72", FEATURE_COLS),
        ("Top 36", df_rnk.head(36)["feature"].tolist()),
        ("Top 18", df_rnk.head(18)["feature"].tolist()),
    ]
    t0 = time.time()
    results = quick_validation(X, y, groups, feature_sets, n_folds=args.n_folds)
    for label, n_feat, mean_acc, std_acc in results:
        print(f"  {label:<8s} ({n_feat:>2d} feat) : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  ({_format_duration(time.time()-t0)})")

    # -- 6. Save reduced datasets (using rank-based) ---------------------------
    if args.save_reduced:
        print(f"\n{'─'*70}")
        print(f"  Saving reduced-feature datasets (rank-based selection) ...")
        print(f"{'─'*70}")

        meta_cols = ["label", "user", "sample_id", "window_idx"]

        if SVM_FILE.exists():
            t0 = time.time()
            df_src = pd.read_parquet(SVM_FILE)

            for topk, suffix in [(36, "reduced36"), (18, "reduced18")]:
                top_feats = df_rnk.head(topk)["feature"].tolist()
                keep = [c for c in top_feats + meta_cols if c in df_src.columns]
                out = SVM_FILE.parent / f"dataset_TRAINING_{suffix}.parquet"
                df_src[keep].to_parquet(out, index=False)
                print(f"  → {out.name}  ({len(df_src):,} rows × "
                      f"{len(keep)} cols)  ({time.time()-t0:.1f}s)")

            del df_src
            gc.collect()
        else:
            print(f"  SKIP: {SVM_FILE.name} not found.")

    # -- Done ------------------------------------------------------------------
    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  ✓ Feature selection complete in {_format_duration(total)}")
    print(f"  Rank-based  : {csv_ranks}")
    print(f"  Score-based : {csv_scores}")
    if args.save_reduced:
        print(f"  Reduced sets: {SVM_FILE.parent}")
    print(f"{'='*70}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-validated feature selection for EMG-EPN612 "
                    "TD9 features (72 features, 5 methods, GroupKFold).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help="Number of top features to highlight / export.")
    p.add_argument("--from-csv", action="store_true",
                   help="Load ranking from existing feature_ranking.csv "
                        "(skip CV scoring, just validate & export).")
    p.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS,
                   help="Number of GroupKFold folds for cross-validation.")
    p.add_argument("--corr-thr", type=float, default=DEFAULT_CORR_THR,
                   help="Pearson |ρ| threshold for the correlation filter.")
    p.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE,
                   help="Max training rows for slow methods (MI, RFE). "
                        "0 = use all rows (slower).")
    p.add_argument("--no-save-reduced", dest="save_reduced",
                   action="store_false",
                   help="Skip saving reduced-feature parquet files.")
    p.set_defaults(save_reduced=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
