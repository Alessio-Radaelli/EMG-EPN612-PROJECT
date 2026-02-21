"""
Run feature selection on dataset_TRAINING.parquet  (sequential folds).

Outputs:
  models/feature_ranking.csv
  preprocessed_output/dataset_TRAINING_reduced36.parquet  (top 36 features)
  preprocessed_output/dataset_TRAINING_reduced18.parquet  (top 18 features)

Usage:
    cd "EMG-EPN612-PROJECT"
    python scripts/run_feature_selection_training.py
"""

import sys, os, time, gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

# ── paths ────────────────────────────────────────────────────────────────
PROJECT   = Path(__file__).resolve().parent.parent
DATA_DIR  = PROJECT / "preprocessed_output"
TRAIN_F   = DATA_DIR / "dataset_TRAINING.parquet"
MODEL_DIR = PROJECT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES  = len(ALL_LABELS)
CHANNELS   = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES  = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
FEAT_COLS  = [f"{ch}_{f}" for ch in CHANNELS for f in TD9_NAMES]
N_FEAT     = len(FEAT_COLS)  # 72

N_FOLDS    = 5
CORR_THR   = 0.95
SUBSAMPLE  = 200_000

def p(msg=""):
    print(msg, flush=True)

def fmt(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")

def norm01(a):
    mn, mx = a.min(), a.max()
    return np.ones_like(a) if mx - mn < 1e-12 else (a - mn) / (mx - mn)

# ── individual methods ───────────────────────────────────────────────────

def m_corr(Xtr, _y):
    c = np.abs(np.corrcoef(Xtr, rowvar=False))
    np.fill_diagonal(c, 0)
    nh = (c > CORR_THR).sum(axis=1)
    return (1.0 - nh / max(nh.max(), 1)).astype(np.float64)

def m_anova(Xtr, ytr):
    f, _ = f_classif(Xtr, ytr)
    return np.nan_to_num(f, nan=0.0)

def m_mi(Xtr, ytr, seed):
    return mutual_info_classif(Xtr, ytr, discrete_features=False,
                               n_neighbors=5, random_state=seed, n_jobs=-1)

def m_xgb(Xtr, ytr, seed):
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", tree_method="hist",
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    clf.fit(Xtr, ytr)
    return clf.feature_importances_

def m_rfe(Xtr, ytr, seed):
    est = LinearSVC(max_iter=5000, dual="auto", random_state=seed,
                    class_weight="balanced")
    rfe = RFE(est, n_features_to_select=1, step=5)
    rfe.fit(Xtr, ytr)
    return (N_FEAT - rfe.ranking_ + 1).astype(np.float64)

def subsample(X, y, maxr, seed=42):
    if maxr <= 0 or maxr >= len(y):
        return X, y
    rng = np.random.RandomState(seed)
    cls, cnts = np.unique(y, return_counts=True)
    frac = maxr / len(y)
    idx = []
    for c, cnt in zip(cls, cnts):
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, size=min(max(1, int(cnt*frac)), len(ci)), replace=False))
    idx = np.concatenate(idx); rng.shuffle(idx)
    return X[idx], y[idx]

# ── main ─────────────────────────────────────────────────────────────────

def main():
    t0_all = time.time()
    p("=" * 70)
    p("  Feature Selection  —  dataset_TRAINING.parquet")
    p("=" * 70)

    # 1. Load
    p(f"\n  Loading {TRAIN_F.name} ...")
    t0 = time.time()
    df = pd.read_parquet(TRAIN_F)
    p(f"  {len(df):,} rows × {len(df.columns)} cols  ({time.time()-t0:.1f}s)")

    le = LabelEncoder(); le.fit(ALL_LABELS)
    X = df[FEAT_COLS].values.astype(np.float32)
    y = le.transform(df["label"].values)
    groups = df["user"].values
    p(f"  Features: {N_FEAT}   Classes: {N_CLASSES}   Users: {len(np.unique(groups))}")

    csv_path = MODEL_DIR / "feature_ranking.csv"

    # 2. Check if ranking already computed
    if csv_path.exists():
        p(f"\n  Found existing {csv_path.name} — loading (skip CV scoring)")
        df_rank = pd.read_csv(csv_path)
    else:
        # 3. Cross-validated feature selection (SEQUENTIAL folds)
        p(f"\n{'─'*70}")
        p(f"  Running {N_FOLDS}-fold CV feature selection (sequential) ...")
        p(f"{'─'*70}")

        gkf = GroupKFold(n_splits=N_FOLDS)
        splits = list(gkf.split(X, y, groups))
        METHOD_KEYS = ["corr_filter", "anova_f", "mutual_info", "xgb_gain", "rfe_rank"]
        fold_scores = {m: [] for m in METHOD_KEYS}

        for fi, (tr_idx, _) in enumerate(splits):
            fn = fi + 1
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xsub, ysub = subsample(Xtr, ytr, SUBSAMPLE, seed=42+fi)
            sub_tag = f", sub={len(ysub):,}" if len(ysub) < len(ytr) else ""
            p(f"\n  ── Fold {fn}/{N_FOLDS}  ({len(tr_idx):,} train{sub_tag}) ──")

            methods = [
                ("corr_filter",  "Correlation",  lambda: m_corr(Xtr, ytr)),
                ("anova_f",      "ANOVA F",      lambda: m_anova(Xtr, ytr)),
                ("mutual_info",  "Mutual Info",   lambda: m_mi(Xsub, ysub, 42+fi)),
                ("xgb_gain",     "XGBoost",       lambda: m_xgb(Xtr, ytr, 42+fi)),
                ("rfe_rank",     "RFE(SVC)",      lambda: m_rfe(Xsub, ysub, 42+fi)),
            ]
            for j, (key, label, fn_call) in enumerate(methods, 1):
                ts = time.time()
                scores = fn_call()
                fold_scores[key].append(scores)
                p(f"    [{j}/5] {label:<16s}  ({time.time()-ts:.1f}s)")

        # Average across folds
        avg = {}
        for m in METHOD_KEYS:
            avg[m] = np.stack(fold_scores[m]).mean(axis=0)

        # Build ranking table
        p(f"\n{'─'*70}")
        p("  Building consensus ranking ...")
        tbl = pd.DataFrame({"feature": FEAT_COLS})
        norm_cols = []
        for m, raw in avg.items():
            tbl[f"{m}_raw"] = raw
            tbl[f"{m}_norm"] = norm01(raw)
            norm_cols.append(f"{m}_norm")
        tbl["consensus_score"] = tbl[norm_cols].mean(axis=1)
        tbl = tbl.sort_values("consensus_score", ascending=False).reset_index(drop=True)
        tbl.insert(0, "rank", range(1, len(tbl) + 1))
        df_rank = tbl

        df_rank.to_csv(csv_path, index=False, float_format="%.6f")
        p(f"  Saved: {csv_path}")

    # 4. Print full ranking
    p(f"\n  Full ranking ({N_FEAT} features):\n")
    p(f"  {'Rk':>4s}  {'Feature':<15s}  "
      f"{'Corr':>6s}  {'ANOVA':>6s}  {'MI':>6s}  "
      f"{'XGB':>6s}  {'RFE':>6s}  {'Score':>6s}")
    p(f"  {'─'*4}  {'─'*15}  " + "  ".join(["─"*6]*6))
    for _, r in df_rank.iterrows():
        p(f"  {int(r['rank']):4d}  {r['feature']:<15s}  "
          f"{r['corr_filter_norm']:6.3f}  {r['anova_f_norm']:6.3f}  "
          f"{r['mutual_info_norm']:6.3f}  {r['xgb_gain_norm']:6.3f}  "
          f"{r['rfe_rank_norm']:6.3f}  {r['consensus_score']:6.3f}")

    # 5. Save reduced datasets
    meta = [c for c in ["label", "user", "sample_id", "window_idx"] if c in df.columns]

    for topk, suffix in [(36, "reduced36"), (18, "reduced18")]:
        p(f"\n{'─'*70}")
        p(f"  Saving dataset_TRAINING_{suffix}.parquet  (top {topk}) ...")
        top_feats = df_rank.head(topk)["feature"].tolist()
        keep = top_feats + meta
        out = DATA_DIR / f"dataset_TRAINING_{suffix}.parquet"
        df[keep].to_parquet(out, index=False)
        p(f"  → {out.name}  ({len(df):,} rows × {len(keep)} cols)")
        p(f"  Features: {top_feats}")

    # 6. Quick validation
    p(f"\n{'─'*70}")
    p(f"  Quick Validation: XGBoost accuracy across feature sets")
    p(f"  ({N_FOLDS}-fold GroupKFold)")
    p(f"{'─'*70}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    for label, feats in [("All 72", FEAT_COLS),
                          ("Top 36", df_rank.head(36)["feature"].tolist()),
                          ("Top 18", df_rank.head(18)["feature"].tolist())]:
        fidx = [FEAT_COLS.index(f) for f in feats]
        accs = []
        tv = time.time()
        for tr_i, va_i in gkf.split(X, y, groups):
            clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", num_class=N_CLASSES,
                eval_metric="mlogloss", tree_method="hist",
                random_state=42, n_jobs=-1, verbosity=0,
            )
            clf.fit(X[tr_i][:, fidx], y[tr_i])
            accs.append(accuracy_score(y[va_i], clf.predict(X[va_i][:, fidx])))
            del clf; gc.collect()
        am, astd = np.mean(accs), np.std(accs)
        p(f"  {label:<8s} ({len(feats):>2d} feat) : {am:.4f} ± {astd:.4f}  ({fmt(time.time()-tv)})")

    total = time.time() - t0_all
    p(f"\n{'='*70}")
    p(f"  ✓ Done in {fmt(total)}")
    p(f"  Ranking        : {csv_path}")
    p(f"  Reduced (36)   : {DATA_DIR / 'dataset_TRAINING_reduced36.parquet'}")
    p(f"  Reduced (18)   : {DATA_DIR / 'dataset_TRAINING_reduced18.parquet'}")
    p(f"{'='*70}")

    del df; gc.collect()


if __name__ == "__main__":
    main()
