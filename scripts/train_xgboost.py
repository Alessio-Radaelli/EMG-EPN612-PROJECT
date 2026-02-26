"""
Successive Halving Hyperparameter Search for XGBoost on EMG-EPN612.

Manual implementation of the successive halving algorithm (tournament-style)
that replaces sklearn's HalvingRandomSearchCV, adding:
  - StratifiedGroupKFold (patient-level + class-distribution stratification)
  - Per-round checkpointing with auto-resume (crash-safe, use --no-resume to start fresh)

How it works (100 candidates, factor=3, ~1.5M samples):
  Round 0:  100 candidates × 3-fold CV on  ~55K samples → keep top 33
  Round 1:   33 candidates × 3-fold CV on ~167K samples → keep top 11
  Round 2:   11 candidates × 3-fold CV on ~500K samples → keep top  3
  Round 3:    3 candidates × 3-fold CV on ~1.5M samples → winner
  Then: train winner on full 1.5M → evaluate on held-out test set

Usage:
  python scripts/hyperparam_searchXGBoost.py
  python scripts/hyperparam_searchXGBoost.py --features 36   # top-36 features
  python scripts/hyperparam_searchXGBoost.py --no-resume     # discard checkpoints
"""

import sys
import time
import json
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform, randint as sp_randint
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# Project paths & dataset constants
# =============================================================================

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PREPROC_DIR   = PROJECT_ROOT / "preprocessed_output"

ALL_LABELS   = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES    = len(ALL_LABELS)
CHANNELS     = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES    = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
ALL_FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
META_COLS    = {"label", "user", "sample_id", "window_idx"}

DATASET_FILES = {
    72: ("dataset_TRAINING.parquet",            "dataset_TEST.parquet"),
    36: ("dataset_TRAINING_reduced36.parquet",  "dataset_TESTING_reduced36.parquet"),
    18: ("dataset_TRAINING_reduced18.parquet",  "dataset_TESTING_reduced18.parquet"),
}

# These globals are set in main() after CLI parsing
MODELS_DIR      = None
CHECKPOINTS_DIR = None
FEATURE_COLS    = None
N_FEATURES      = None

MODEL_NAME = "xgboost"


# =============================================================================
# XGBoost model configuration
# =============================================================================

def get_model_config(no_gpu=False):
    """Return (estimator, param_distributions) for XGBoost."""
    import xgboost as xgb

    device = "cpu"
    if not no_gpu:
        try:
            _t = xgb.XGBClassifier(n_estimators=1, max_depth=1,
                                   device="cuda", verbosity=0)
            _t.fit(np.zeros((2, 2)), np.array([0, 1]))
            device = "cuda"
        except xgb.core.XGBoostError:
            pass

    estimator = xgb.XGBClassifier(
        n_estimators=1000,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device=device,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=20,
    )
    param_dist = {
        "max_depth":       sp_randint(3, 10),
        "learning_rate":   loguniform(0.01, 0.3),
        "min_child_weight": sp_randint(1, 15),
        "subsample":       uniform(0.6, 0.4),       # [0.6, 1.0]
        "colsample_bytree": uniform(0.6, 0.4),      # [0.6, 1.0]
        "reg_alpha":       loguniform(1e-3, 10),
        "reg_lambda":      loguniform(1e-3, 10),
        "gamma":           uniform(0, 5),
    }
    return estimator, param_dist


# =============================================================================
# Helpers
# =============================================================================

def _json_safe(obj):
    """Convert numpy / tuple types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def _params_to_json(params: dict) -> dict:
    return {k: _json_safe(v) for k, v in params.items()}


def _format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _sample_candidates(param_distributions, n_candidates, rng):
    """Draw n_candidates random parameter sets from the distributions.

    If every distribution is a finite list, exhaustively enumerate all
    combinations (ignoring n_candidates) so nothing is missed or duplicated.
    """
    all_lists = all(isinstance(d, list) for d in param_distributions.values())

    if all_lists:
        from itertools import product
        names = list(param_distributions.keys())
        combos = list(product(*(param_distributions[n] for n in names)))
        rng.shuffle(combos)
        candidates = [{n: _json_safe(v) for n, v in zip(names, combo)}
                      for combo in combos]
        return candidates

    candidates = []
    for _ in range(n_candidates):
        params = {}
        for name, dist in param_distributions.items():
            if hasattr(dist, "rvs"):
                val = dist.rvs(random_state=rng)
            elif isinstance(dist, list):
                val = dist[rng.randint(len(dist))]
            else:
                val = dist
            params[name] = _json_safe(val)
        candidates.append(params)
    return candidates


def _subsample_by_group(y, groups, n_target, rng):
    """Subsample entire patient groups to reach ~n_target samples.

    Maintains group integrity: every sample from a selected patient
    is included.  Approximate class balance is preserved because
    each patient performs all gestures.

    Returns sorted index array.
    """
    n_total = len(y)
    if n_target >= n_total:
        return np.arange(n_total)

    unique_g = np.unique(groups)
    rng.shuffle(unique_g)

    # Pre-compute group → indices mapping for speed
    g_to_idx = {}
    for g in unique_g:
        g_to_idx[g] = np.where(groups == g)[0]

    selected_idx = []
    for g in unique_g:
        if len(selected_idx) >= n_target:
            break
        selected_idx.extend(g_to_idx[g].tolist())

    return np.sort(selected_idx)


def _evaluate_candidate(estimator, params, X, y, groups, cv):
    """Evaluate one candidate via cross-validation, return fold scores."""
    est = clone(estimator)
    est.set_params(**params)

    scores = []
    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        y_pred = est.predict(X_va)
        scores.append(float(accuracy_score(y_va, y_pred)))

    return np.array(scores)


# =============================================================================
# Tournament schedule management
# =============================================================================

def compute_schedule(n_total, n_candidates, factor):
    """Compute the halving round schedule.

    Returns list of dicts: [{round, n_candidates, n_samples}, ...]
    """
    # Number of elimination rounds needed to reach 1 survivor
    c, n_rounds = n_candidates, 0
    while c > 1:
        c = max(1, int(c / factor))
        n_rounds += 1

    # min_resources: back-calculate so the last round uses all data
    min_resources = max(
        N_CLASSES * 10,  # absolute minimum for stratified splits
        int(n_total / (factor ** (n_rounds - 1))),
    )

    schedule = []
    cand = n_candidates
    for r in range(n_rounds):
        n_samples = min(int(min_resources * (factor ** r)), n_total)
        schedule.append({
            "round": r,
            "n_candidates": int(cand),
            "n_samples": n_samples,
        })
        cand = max(1, int(cand / factor))

    return schedule


def load_or_create_schedule(n_total, n_candidates, factor, n_splits,
                            random_state):
    """Load saved schedule or create + save a new one."""
    schedule_file = MODELS_DIR / "halving_schedule_xgboost.json"

    meta = {
        "dataset_rows": n_total,
        "n_candidates": n_candidates,
        "factor": factor,
        "n_splits": n_splits,
        "random_state": random_state,
    }

    if schedule_file.exists():
        saved = json.loads(schedule_file.read_text())
        compat = True
        for key in ("dataset_rows", "n_candidates", "factor",
                     "n_splits", "random_state"):
            if saved.get(key) != meta[key]:
                compat = False
                break
        if compat:
            print(f"  Loaded existing tournament schedule from {schedule_file.name}")
            return saved["rounds"]
        else:
            print(f"  WARNING: schedule parameters changed — regenerating.")
            print(f"    Saved: { {k: saved.get(k) for k in meta} }")
            print(f"    Current: {meta}")

    rounds = compute_schedule(n_total, n_candidates, factor)
    saved = {**meta, "rounds": rounds}
    schedule_file.write_text(json.dumps(saved, indent=2))
    print(f"  Saved tournament schedule to {schedule_file.name}")
    return rounds


# =============================================================================
# Successive halving engine
# =============================================================================

def run_halving(estimator, param_distributions, X, y, groups,
                n_candidates, factor, n_splits, random_state, resume,
                n_features=72):
    """Run the full successive halving tournament for XGBoost."""

    n_total = len(y)
    rng = np.random.RandomState(random_state)

    # --- Checkpoint directory -------------------------------------------
    cp_dir = CHECKPOINTS_DIR / MODEL_NAME
    cp_dir.mkdir(parents=True, exist_ok=True)

    # --- Sample or load initial candidates ------------------------------
    cand_file = cp_dir / "initial_candidates.json"
    if resume and cand_file.exists():
        candidates = json.loads(cand_file.read_text())
        print(f"\n  Loaded {len(candidates)} initial candidates from checkpoint")
    else:
        candidates = _sample_candidates(param_distributions, n_candidates, rng)
        cand_file.write_text(json.dumps(candidates, indent=2, default=str))
        print(f"\n  Sampled {len(candidates)} initial candidates")

    actual_n_candidates = len(candidates)

    # --- Schedule (based on actual candidate count) ---------------------
    rounds = load_or_create_schedule(
        n_total, actual_n_candidates, factor, n_splits, random_state)
    n_rounds = len(rounds)

    print(f"\n  Tournament: {n_rounds} rounds, factor={factor}")
    for r in rounds:
        print(f"    Round {r['round']}: {r['n_candidates']:3d} candidates "
              f"× {r['n_samples']:>10,} samples")

    # --- Resume: find last completed round ------------------------------
    start_round = 0
    if resume:
        for r in range(n_rounds):
            cp_file = cp_dir / f"round_{r}.json"
            if cp_file.exists():
                cp = json.loads(cp_file.read_text())
                candidates = cp["survivors"]
                start_round = r + 1
            else:
                break
        if start_round > 0:
            print(f"  Resuming from round {start_round} "
                  f"({len(candidates)} surviving candidates)")

    # --- CV splitter ----------------------------------------------------
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)

    # --- Run rounds -----------------------------------------------------
    t_total = time.time()

    for round_k in range(start_round, n_rounds):
        round_info = rounds[round_k]
        n_samples_k = round_info["n_samples"]
        n_cand_k = len(candidates)

        tournament_elapsed = time.time() - t_total
        print(f"\n{'='*70}")
        print(f"  ROUND {round_k}/{n_rounds - 1}:  "
              f"{n_cand_k} candidates × {n_samples_k:,} samples "
              f"× {n_splits}-fold CV"
              f"  [elapsed: {_format_duration(tournament_elapsed)}]")
        print(f"{'='*70}")

        # Subsample (deterministic per round — fresh RNG from seed + round)
        round_rng = np.random.RandomState(random_state + round_k * 1000)
        sub_idx = _subsample_by_group(y, groups, n_samples_k, round_rng)
        X_round = X[sub_idx]
        y_round = y[sub_idx]
        groups_round = groups[sub_idx]
        actual_n = len(sub_idx)
        print(f"  Subsampled: {actual_n:,} rows "
              f"({len(np.unique(groups_round))} patients)")

        # Evaluate each surviving candidate
        all_scores = []
        t_round = time.time()

        for c_idx, params in enumerate(candidates):
            t0 = time.time()
            try:
                fold_scores = _evaluate_candidate(
                    estimator, params, X_round, y_round, groups_round, cv)
                mean_sc = float(fold_scores.mean())
                std_sc = float(fold_scores.std())
            except Exception as e:
                print(f"    [{c_idx+1}/{n_cand_k}] FAILED: {e}")
                mean_sc, std_sc = 0.0, 0.0
                fold_scores = np.zeros(n_splits)

            elapsed = time.time() - t0

            # ETA
            done = c_idx + 1
            remaining_cand = n_cand_k - done
            avg_time = (time.time() - t_round) / done
            eta = _format_duration(avg_time * remaining_cand)

            print(f"    [{done:3d}/{n_cand_k}] "
                  f"score={mean_sc:.4f}±{std_sc:.4f}  "
                  f"({elapsed:.1f}s)  ETA: {eta}")

            all_scores.append({
                "params": _params_to_json(params),
                "mean_score": mean_sc,
                "std_score": std_sc,
                "fold_scores": fold_scores.tolist(),
                "time_s": round(elapsed, 1),
            })

        # Rank and eliminate
        all_scores.sort(key=lambda x: x["mean_score"], reverse=True)
        n_survivors = max(1, int(n_cand_k / factor))
        survivors = [s["params"] for s in all_scores[:n_survivors]]

        round_elapsed = time.time() - t_round
        round_work = {r: rounds[r]["n_candidates"] * rounds[r]["n_samples"]
                      for r in range(n_rounds)}
        work_done = sum(round_work[r] for r in range(start_round, round_k + 1))
        work_left = sum(round_work[r] for r in range(round_k + 1, n_rounds))
        time_per_work = (time.time() - t_total) / work_done if work_done else 0
        tournament_eta = _format_duration(time_per_work * work_left)
        rounds_left = n_rounds - round_k - 1

        print(f"\n  Round {round_k} complete ({_format_duration(round_elapsed)})")
        print(f"    Best:  {all_scores[0]['mean_score']:.4f}")
        print(f"    Worst: {all_scores[-1]['mean_score']:.4f}")
        print(f"    Eliminated: {n_cand_k - n_survivors}  |  "
              f"Surviving: {n_survivors}")
        if rounds_left > 0:
            print(f"    Tournament ETA: ~{tournament_eta} "
                  f"({rounds_left} round{'s' if rounds_left > 1 else ''} left)")

        # Save checkpoint
        checkpoint = {
            "round": round_k,
            "n_samples": actual_n,
            "n_candidates": n_cand_k,
            "n_survivors": n_survivors,
            "all_scores": all_scores,
            "survivors": survivors,
            "elapsed_s": round(round_elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        cp_file = cp_dir / f"round_{round_k}.json"
        cp_file.write_text(json.dumps(checkpoint, indent=2, default=str))
        print(f"    Checkpoint saved: {cp_file.name}")

        candidates = survivors
        gc.collect()

    # --- Final summary --------------------------------------------------
    total_elapsed = time.time() - t_total
    best_params = candidates[0]

    print(f"\n{'='*70}")
    print(f"  TOURNAMENT COMPLETE  ({_format_duration(total_elapsed)})")
    print(f"{'='*70}")
    print(f"\n  Best hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k:>20s} : {v}")

    # Save best params
    best_file = MODELS_DIR / f"{MODEL_NAME}_best_params.json"
    best_file.write_text(json.dumps(best_params, indent=2, default=str))
    print(f"\n  Best params saved to: {best_file}")

    # Save full results CSV
    _save_results_csv(MODEL_NAME, cp_dir, n_rounds)

    # Train final model with best params on full data (no early stopping)
    print(f"\n  Training final model with best params on full data ...")
    t0 = time.time()
    final_est = clone(estimator)
    final_est.set_params(**best_params)
    try:
        import xgboost as xgb
        if isinstance(final_est, xgb.XGBClassifier):
            final_est.set_params(early_stopping_rounds=None)
    except ImportError:
        pass
    final_est.fit(X, y)
    print(f"  Final model trained ({time.time()-t0:.1f}s)")

    # Save final model
    _save_final_model(final_est)

    # Evaluate on held-out test set
    _evaluate_on_test_set(final_est, n_features)

    return best_params


def _save_results_csv(model_name, cp_dir, n_rounds):
    """Compile all round checkpoints into a single results CSV."""
    rows = []
    for r in range(n_rounds):
        cp_file = cp_dir / f"round_{r}.json"
        if not cp_file.exists():
            continue
        cp = json.loads(cp_file.read_text())
        for entry in cp["all_scores"]:
            row = {"round": r, "n_samples": cp["n_samples"]}
            row.update(entry["params"])
            row["mean_score"] = entry["mean_score"]
            row["std_score"] = entry["std_score"]
            row["time_s"] = entry["time_s"]
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = MODELS_DIR / f"{model_name}_halving_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Results CSV: {csv_path}")


def _save_final_model(estimator):
    """Save the final trained XGBoost model."""
    path = MODELS_DIR / f"{MODEL_NAME}_best_halving.json"
    estimator.save_model(str(path))
    print(f"  Model saved: {path}")


def _evaluate_on_test_set(estimator, n_features):
    """Load the held-out test set, predict with the fitted model, and report."""
    _, test_name = DATASET_FILES[n_features]
    test_file = PREPROC_DIR / test_name

    if not test_file.exists():
        print(f"\n  WARNING: {test_file} not found — skipping test eval.")
        return

    print(f"\n  Evaluating on test set ({test_file.name}) ...")
    t0 = time.time()
    df = pd.read_parquet(test_file)
    X_test = df[FEATURE_COLS].values.astype(np.float32)

    le = LabelEncoder()
    le.fit(ALL_LABELS)
    y_test = le.transform(df["label"].values)
    label_map = {i: name for i, name in enumerate(le.classes_)}

    del df
    gc.collect()

    y_pred = estimator.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report_str = classification_report(
        y_test, y_pred,
        target_names=[label_map[i] for i in range(len(label_map))],
        digits=4,
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=[label_map[i] for i in range(len(label_map))],
        digits=4,
        output_dict=True,
    )
    elapsed = time.time() - t0

    print(f"  Test accuracy: {acc:.4f}  ({elapsed:.1f}s)")
    print(f"\n{report_str}")

    results = {
        "model": MODEL_NAME,
        "test_accuracy": acc,
        "classification_report": report_dict,
        "test_samples": len(X_test),
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_file = MODELS_DIR / f"{MODEL_NAME}_test_results.json"
    results_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Test results saved to: {results_file}")


# =============================================================================
# Data loading
# =============================================================================

def load_data(n_features):
    """Load training parquet for the requested feature set and return X, y, groups."""
    global FEATURE_COLS, N_FEATURES

    train_name, _ = DATASET_FILES[n_features]
    dataset_file = PREPROC_DIR / train_name

    if not dataset_file.exists():
        print(f"ERROR: {dataset_file} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Loading {dataset_file.name} ...")
    t0 = time.time()
    df = pd.read_parquet(dataset_file)
    print(f"  {len(df):,} rows × {len(df.columns)} cols  ({time.time()-t0:.1f}s)")

    FEATURE_COLS = [c for c in df.columns if c in ALL_FEATURE_COLS]
    N_FEATURES = len(FEATURE_COLS)
    X = df[FEATURE_COLS].values.astype(np.float32)

    le = LabelEncoder()
    le.fit(ALL_LABELS)
    y = le.transform(df["label"].values)
    label_map = dict(zip(le.transform(le.classes_), le.classes_))

    groups = df["user"].values

    del df
    gc.collect()

    print(f"  Features: {X.shape}  ({N_FEATURES}f)")
    print(f"  Classes:  {label_map}")
    print(f"  Patients: {len(np.unique(groups))}")

    return X, y, groups


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Successive Halving Hyperparameter Search for XGBoost on EMG-EPN612.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-candidates", type=int, default=100,
                   help="Number of initial random candidates.")
    p.add_argument("--factor", type=int, default=3,
                   help="Elimination factor (keep 1/factor each round).")
    p.add_argument("--n-splits", type=int, default=3,
                   help="Number of CV folds (StratifiedGroupKFold).")
    p.add_argument("--features", type=int, default=72, choices=[72, 36, 18],
                   help="Feature set to use (72=full, 36=top36, 18=top18).")
    p.add_argument("--no-gpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    p.add_argument("--no-resume", action="store_false", dest="resume",
                   help="Start fresh, ignoring existing checkpoints.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    return p.parse_args()


def main():
    global MODELS_DIR, CHECKPOINTS_DIR

    args = parse_args()

    # Set up output directories based on feature set
    nf = args.features
    MODELS_DIR = PROJECT_ROOT / "models" / f"{nf}f"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR = MODELS_DIR / "halving_checkpoints"

    print("=" * 70)
    print(f"  Successive Halving Hyperparameter Search  —  EMG-EPN612  ({nf}f)")
    print("=" * 70)

    X, y, groups = load_data(nf)

    print(f"\n\n{'#'*70}")
    print(f"  XGBOOST  ({nf}f)")
    print(f"{'#'*70}")

    estimator, param_dist = get_model_config(no_gpu=args.no_gpu)

    run_halving(
        estimator=estimator,
        param_distributions=param_dist,
        X=X, y=y, groups=groups,
        n_candidates=args.n_candidates,
        factor=args.factor,
        n_splits=args.n_splits,
        random_state=args.seed,
        resume=args.resume,
        n_features=nf,
    )

    print(f"\n\n{'='*70}")
    print("  SEARCH COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

