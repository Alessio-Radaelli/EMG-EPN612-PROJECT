# ══════════════════════════════════════════════════════════════════════════════
# EMG-EPN612 -- Preprocessing Pipeline  (dataset_A -> dataset_TRAINING)
# ══════════════════════════════════════════════════════════════════════════════
#
# Five-phase pipeline:
#   Phase 1  Signal Conditioning (bandpass + 50 Hz notch) & Segmentation
#   Phase 2  TD9 Feature Extraction (72 features per window)
#   Phase 3  Outlier Detection (IQR voting)
#   Phase 4  Outlier Removal (drop bad windows)
#   Phase 5  Subject-Specific Z-Score Normalization
#
# Input : datasets/dataset_A.pkl  (459 users, 150 registrations each)
# Output: preprocessed_output/dataset_TRAINING.parquet
#
# Dataset JSON key layout (per user inside the pkl):
#   trainingSamples -> idx_N -> gestureName        (str)
#                            -> emg -> ch1..ch8    (list[float])
#                            -> groundTruthIndex   [start, end]  (1-based)
#                            -> groundTruth        list[0|1]
#   (noGesture samples have NO groundTruthIndex / groundTruth keys)
#
# Usage:
#   cd <project root>
#   python scripts/preprocess_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

import json
import os
import sys
import time
import gc
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from concurrent.futures import ProcessPoolExecutor

# -- Paths (relative to project root) ----------------------------------------
DATASET_A_PATH = Path("datasets") / "dataset_A.pkl"
OUTPUT_DIR     = Path("preprocessed_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# -- Global Parameters -------------------------------------------------------
FS            = 200       # sampling frequency in Hz
WINDOW_LENGTH = 40        # 200 ms window at 200 Hz
WINDOW_SHIFT  = 10        # 50 ms step -> 75 % overlap
THRESHOLD     = 0.00001   # 10 uV threshold for WAMP / ZC (noise floor)
CHANNELS      = [f"ch{i}" for i in range(1, 9)]  # ch1 .. ch8
TD9_NAMES     = ["LS", "MFL", "MSR", "WAMP", "ZC",
                 "RMS", "IAV", "DASDV", "VAR"]
NUM_WORKERS   = 8         # parallel worker processes
CHUNK_SIZE    = 10        # users per processing chunk

# noGesture: fallback crop length if no gesture segments are available
NO_GESTURE_CROP_FALLBACK = int(1.0 * FS)  # 260 samples

# Outlier-detection thresholds
IQR_VOTE_PCT  = 0.25      # >25 % of features out-of-bounds -> BAD

# Column names
feature_columns = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
meta_columns    = ["label", "user", "sample_id", "window_idx"]
all_columns     = feature_columns + meta_columns


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Signal Conditioning
# ══════════════════════════════════════════════════════════════════════════════

def filter_emg(emg_signal, fs=FS, lowcut=20, highcut=95,
               notch_freq=50, notch_q=30):
    """Bandpass (20-95 Hz) + 50 Hz notch filter via zero-phase filtfilt.

    1. 2nd-order Butterworth IIR bandpass (20-95 Hz).
    2. IIR notch at 50 Hz (Q=30) to remove mains-frequency interference.
    Both applied with filtfilt for zero-phase distortion.
    """
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    # Bandpass
    b_bp, a_bp = signal.butter(2, [low, high], btype="band")
    filtered = signal.filtfilt(b_bp, a_bp, emg_signal)
    # 50 Hz notch (power-line removal)
    b_notch, a_notch = signal.iirnotch(w0=notch_freq, Q=notch_q, fs=fs)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)
    return filtered


def segment_trial(filtered_channels, sample_meta, no_gesture_crop=None):
    """Crop the filtered signal to the gesture region.

    For gestures   → use groundTruthIndex [start, end] (1-based → 0-based).
    For noGesture  → crop `no_gesture_crop` samples from the centre of the trial.
                     If not provided, falls back to NO_GESTURE_CROP_FALLBACK.

    Returns a dict  {ch: np.array} with the cropped signals.
    """
    gesture = sample_meta["gestureName"]

    if gesture != "noGesture":
        # groundTruthIndex is [start, end], 1-based inclusive
        gti   = sample_meta["groundTruthIndex"]
        start = gti[0] - 1                         # convert to 0-based
        end   = gti[1]                              # Python slice end (exclusive)
        return {ch: arr[start:end] for ch, arr in filtered_channels.items()}
    else:
        # noGesture: take median-gesture-length from the centre of the trial
        length  = len(next(iter(filtered_channels.values())))
        crop    = no_gesture_crop if no_gesture_crop else NO_GESTURE_CROP_FALLBACK
        centre  = length // 2
        start   = max(centre - crop // 2, 0)
        end     = start + crop
        if end > length:                            # safety clamp
            end   = length
            start = end - crop
        return {ch: arr[start:end] for ch, arr in filtered_channels.items()}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — TD9 Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def _LS(x):
    """L-Scale: robust dispersion measure."""
    n = len(x)
    if n < 2:
        return 0.0
    xs = np.sort(x)
    i  = np.arange(1, n + 1)
    return np.sum((2 * i - n - 1) * xs) / (n * (n - 1))

def _MFL(x):
    """Maximum Fractal Length."""
    return np.log(np.sqrt(np.sum(np.diff(x) ** 2)) + 1e-10)

def _MSR(x):
    """Mean Square Root."""
    return np.mean(np.sqrt(np.abs(x)))

def _WAMP(x, thr):
    """Willison Amplitude."""
    return np.sum(np.abs(np.diff(x)) > thr)

def _ZC(x, thr):
    """Zero Crossings with amplitude check."""
    x1, x2 = x[:-1], x[1:]
    return np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) > thr))

def _RMS(x):
    """Root Mean Square."""
    return np.sqrt(np.mean(x ** 2))

def _IAV(x):
    """Integrated Absolute Value."""
    return np.sum(np.abs(x))

def _DASDV(x):
    """Difference Absolute Standard Deviation Value."""
    return np.sqrt(np.mean(np.diff(x) ** 2))

def _VAR(x):
    """Variance (ddof=1)."""
    return np.var(x, ddof=1)


def extract_td9(window, thr=THRESHOLD):
    """Return 9-element feature array for one window."""
    return np.array([
        _LS(window),  _MFL(window),  _MSR(window),
        _WAMP(window, thr), _ZC(window, thr),
        _RMS(window), _IAV(window),  _DASDV(window), _VAR(window)
    ])


def windowed_features(cropped_channels, label, user, sample_id):
    """Slide a window over the cropped segment and extract 72 features.

    Returns a list of row-lists (each of length 76: 72 features + 4 meta).
    """
    n_samples = len(next(iter(cropped_channels.values())))
    rows = []
    win_idx = 0
    for start in range(0, n_samples - WINDOW_LENGTH + 1, WINDOW_SHIFT):
        end = start + WINDOW_LENGTH
        feat = np.concatenate([
            extract_td9(cropped_channels[ch][start:end])
            for ch in CHANNELS
        ])
        rows.append(feat.tolist() + [label, user, sample_id, win_idx])
        win_idx += 1
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Per-user worker (Phases 1 + 2, no normalisation yet)
# ══════════════════════════════════════════════════════════════════════════════

# -- Global holder for user data passed to workers ---------------------------
# (set by main before launching the pool; workers inherit via fork/spawn)
_USER_DATA_STORE = {}


def process_user(user_id):
    """Filter, segment, extract features for one user.

    Reads from _USER_DATA_STORE[user_id] (trainingSamples dict).
    Returns a list of 76-element rows (features + metadata).
    Normalisation is deferred to Phase 5 (after outlier handling).
    """
    samples     = _USER_DATA_STORE[user_id]
    sample_keys = list(samples.keys())
    user_label  = f"user{user_id}"      # consistent with folder-style naming
    all_rows    = []

    # Compute per-user median gesture length for noGesture cropping
    gesture_lengths = []
    for sk in sample_keys:
        gti = samples[sk].get("groundTruthIndex")
        if gti:
            gesture_lengths.append(gti[1] - gti[0] + 1)
    median_gesture_len = (int(np.median(gesture_lengths))
                          if gesture_lengths else NO_GESTURE_CROP_FALLBACK)

    for sample_key in sample_keys:
        sample  = samples[sample_key]
        emg     = sample["emg"]
        gesture = sample["gestureName"]

        # Phase 1a -- Bandpass + notch filter on full 5 s trial
        filtered = {ch: filter_emg(np.array(emg[ch])) for ch in CHANNELS}

        # Phase 1b -- Segment (crop to gesture or centre-crop for noGesture)
        cropped = segment_trial(filtered, sample,
                                no_gesture_crop=median_gesture_len)

        # Phase 2 -- Windowed TD9 feature extraction
        rows = windowed_features(cropped, gesture, user_label, sample_key)
        all_rows.extend(rows)

    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Outlier Detection & Branching
# ══════════════════════════════════════════════════════════════════════════════

def detect_outliers(df):
    """Add 'is_bad_window' column using Subject- & Gesture-Specific IQR voting.

    For each (user, gesture) group independently:
      1. Compute Q1, Q3, IQR for every feature column.
      2. For each window, count how many features fall outside
         [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
      3. Mark as BAD if >25 % of feature values are outside the fences.

    Grouping by gesture avoids penalising gestures whose activation patterns
    differ from the pooled distribution (e.g. strong fist vs. quiet noGesture).
    """
    df = df.copy()
    df["vote_count"]    = 0
    df["is_bad_window"] = False

    for (user, label), grp in df.groupby(["user", "label"]):
        idx = grp.index
        feat_vals = grp[feature_columns].values              # (n_windows, 72)

        q1  = np.percentile(feat_vals, 25, axis=0)           # (72,)
        q3  = np.percentile(feat_vals, 75, axis=0)           # (72,)
        iqr = q3 - q1                                        # (72,)

        lower = q1 - 1.5 * iqr                               # (72,)
        upper = q3 + 1.5 * iqr                               # (72,)

        outside = (feat_vals < lower) | (feat_vals > upper)   # bool (n, 72)
        votes   = outside.sum(axis=1)                         # (n,)

        n_feats = len(feature_columns)                        # 72
        bad     = votes > (IQR_VOTE_PCT * n_feats)            # >18 features OOB

        df.loc[idx, "vote_count"]    = votes
        df.loc[idx, "is_bad_window"] = bad

    return df


def generate_outlier_report(df_flagged, output_dir):
    """Generate, print, and save outlier analysis reports (CSV).

    Saves three files:
      - outlier_report_by_gesture.csv       -- per-gesture window stats
      - outlier_report_by_user.csv          -- per-user   window stats
      - outlier_report_by_user_gesture.csv  -- per (user, gesture) cross-tab

    Also prints a detailed console summary.
    """
    n_total_all = len(df_flagged)
    n_bad_all   = int(df_flagged["is_bad_window"].sum())

    # -- Window-level stats per gesture -----------------------------------
    gesture_rows = []
    for label, grp in df_flagged.groupby("label"):
        n_total = len(grp)
        n_bad   = int(grp["is_bad_window"].sum())
        # count how many registrations have at least 1 bad window
        reps_with_bad = 0
        for (_, sid), rep_grp in grp.groupby(["user", "sample_id"]):
            if rep_grp["is_bad_window"].any():
                reps_with_bad += 1
        total_reps = grp.groupby(["user", "sample_id"]).ngroups
        gesture_rows.append({
            "gesture":              label,
            "total_windows":        n_total,
            "bad_windows":          n_bad,
            "pct_bad_windows":      round(n_bad / n_total * 100, 2) if n_total else 0,
            "total_registrations":  total_reps,
            "reps_with_outliers":   reps_with_bad,
            "pct_reps_affected":    round(reps_with_bad / total_reps * 100, 2) if total_reps else 0,
        })
    df_gesture = pd.DataFrame(gesture_rows).sort_values("gesture")

    # -- Window-level stats per user --------------------------------------
    user_rows = []
    for user, grp in df_flagged.groupby("user"):
        n_total = len(grp)
        n_bad   = int(grp["is_bad_window"].sum())
        total_reps = grp["sample_id"].nunique()
        reps_with_bad = 0
        for sid, rep_grp in grp.groupby("sample_id"):
            if rep_grp["is_bad_window"].any():
                reps_with_bad += 1
        user_rows.append({
            "user":                 user,
            "total_windows":        n_total,
            "bad_windows":          n_bad,
            "pct_bad_windows":      round(n_bad / n_total * 100, 2) if n_total else 0,
            "total_registrations":  total_reps,
            "reps_with_outliers":   reps_with_bad,
            "pct_reps_affected":    round(reps_with_bad / total_reps * 100, 2) if total_reps else 0,
        })
    df_user = pd.DataFrame(user_rows).sort_values(
        "user", key=lambda s: s.str.replace("user", "").astype(int)
    )

    # -- Per (user, gesture) cross-tab ------------------------------------
    ug_rows = []
    for (user, label), grp in df_flagged.groupby(["user", "label"]):
        n_total = len(grp)
        n_bad   = int(grp["is_bad_window"].sum())
        total_reps = grp["sample_id"].nunique()
        reps_with_bad = 0
        for sid, rep_grp in grp.groupby("sample_id"):
            if rep_grp["is_bad_window"].any():
                reps_with_bad += 1
        ug_rows.append({
            "user":                 user,
            "gesture":              label,
            "total_windows":        n_total,
            "bad_windows":          n_bad,
            "pct_bad_windows":      round(n_bad / n_total * 100, 2) if n_total else 0,
            "total_registrations":  total_reps,
            "reps_with_outliers":   reps_with_bad,
        })
    df_user_gesture = pd.DataFrame(ug_rows)

    # -- Save CSVs --------------------------------------------------------
    path_gesture = output_dir / "outlier_report_by_gesture.csv"
    path_user    = output_dir / "outlier_report_by_user.csv"
    path_detail  = output_dir / "outlier_report_by_user_gesture.csv"

    df_gesture.to_csv(path_gesture, index=False)
    df_user.to_csv(path_user, index=False)
    df_user_gesture.to_csv(path_detail, index=False)

    # -- Console summary --------------------------------------------------
    print()
    print("  " + "-" * 68)
    print("  OUTLIER ANALYSIS SUMMARY")
    print("  " + "-" * 68)
    print(f"  Total windows        : {n_total_all:,}")
    print(f"  Bad windows (removed): {n_bad_all:,}  "
          f"({n_bad_all/n_total_all*100:.2f} %)")
    print()

    # Per-gesture table
    print("  --- By Gesture " + "-" * 51)
    print(f"  {'Gesture':<14} {'Total':>10} {'Bad':>8} {'%Bad':>7}  "
          f"{'Regs':>6} {'Affected':>8} {'%Aff':>6}")
    for _, r in df_gesture.iterrows():
        print(f"  {r['gesture']:<14} {r['total_windows']:>10,} {r['bad_windows']:>8,} "
              f"{r['pct_bad_windows']:>6.2f}%  "
              f"{r['total_registrations']:>6} {r['reps_with_outliers']:>8} "
              f"{r['pct_reps_affected']:>5.1f}%")
    print()

    # Per-user summary (top 10 worst + distribution stats)
    print("  --- By User (top 10 worst) " + "-" * 40)
    print(f"  {'User':<12} {'Total':>10} {'Bad':>8} {'%Bad':>7}  "
          f"{'Regs':>6} {'Affected':>8}")
    top10 = df_user.nlargest(10, "pct_bad_windows")
    for _, r in top10.iterrows():
        print(f"  {r['user']:<12} {r['total_windows']:>10,} {r['bad_windows']:>8,} "
              f"{r['pct_bad_windows']:>6.2f}%  "
              f"{r['total_registrations']:>6} {r['reps_with_outliers']:>8}")

    # Distribution of user-level bad %
    pcts = df_user["pct_bad_windows"]
    n_zero  = int((pcts == 0).sum())
    n_low   = int(((pcts > 0) & (pcts <= 1)).sum())
    n_mid   = int(((pcts > 1) & (pcts <= 3)).sum())
    n_high  = int((pcts > 3).sum())
    print()
    print("  --- User outlier-rate distribution ---")
    print(f"    0% outliers         : {n_zero} users")
    print(f"    (0%, 1%] outliers   : {n_low} users")
    print(f"    (1%, 3%] outliers   : {n_mid} users")
    print(f"    >3% outliers        : {n_high} users")
    print(f"    mean % bad/user     : {pcts.mean():.2f}%")
    print(f"    median % bad/user   : {pcts.median():.2f}%")
    print(f"    max % bad/user      : {pcts.max():.2f}%")
    print("  " + "-" * 68)

    return path_gesture, path_user, path_detail


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Outlier Correction & Analysis
# ══════════════════════════════════════════════════════════════════════════════

def correct_outliers(df):
    """Drop all windows flagged as outliers (bad windows)."""
    n_before = len(df)
    df_clean = df[~df["is_bad_window"]].copy()
    n_after  = len(df_clean)
    pct_drop = (n_before - n_after) / n_before * 100 if n_before else 0
    return df_clean, pct_drop


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Subject-Specific Z-Score Normalization
# ══════════════════════════════════════════════════════════════════════════════

def zscore_normalize(df):
    """Apply per-subject Z-score to all 72 feature columns.

    For each user, µ and σ are computed from that user's cleaned data.
    """
    df = df.copy()
    for user, grp in df.groupby("user"):
        idx  = grp.index
        vals = grp[feature_columns].values.astype(np.float64)
        mu   = vals.mean(axis=0)
        sig  = vals.std(axis=0) + 1e-8         # epsilon avoids division by zero
        df.loc[idx, feature_columns] = ((vals - mu) / sig).astype(np.float32)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USER_DATA_STORE

    # ------------------------------------------------------------------
    # Load dataset_A.pkl
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  EMG-EPN612 Preprocessing Pipeline  (dataset_A -> dataset_TRAINING)")
    print("=" * 72)

    print(f"\n  Loading {DATASET_A_PATH} ...")
    t_load = time.time()
    with open(DATASET_A_PATH, "rb") as f:
        dataset_a = pickle.load(f)       # {overall_user_id: trainingSamples}
    total_users = len(dataset_a)
    print(f"  Loaded {total_users} users in {time.time()-t_load:.1f}s")

    user_ids = sorted(dataset_a.keys())  # sorted overall user ids

    print(f"  Workers        : {NUM_WORKERS}")
    print(f"  Window         : {WINDOW_LENGTH} samples ({WINDOW_LENGTH/FS*1000:.0f} ms)")
    print(f"  Step           : {WINDOW_SHIFT} samples ({WINDOW_SHIFT/FS*1000:.0f} ms)")
    print(f"  Filter         : bandpass 20-95 Hz + 50 Hz notch")
    print(f"  WAMP/ZC thr    : {THRESHOLD}")
    print(f"  IQR vote thr   : {IQR_VOTE_PCT*100:.0f} % of 72 features")
    print(f"  Output dir     : {OUTPUT_DIR.resolve()}")
    print(f"  CPU cores      : {os.cpu_count()}")
    print()

    # ==================================================================
    # PHASES 1 + 2 -- Filter + Notch -> Segment -> Extract
    # ==================================================================
    print("> Phase 1+2: Signal conditioning (BP+notch) & feature extraction ...")
    t0       = time.time()
    BAR_W    = 40
    all_rows = []

    # Process users sequentially (data is already in memory; avoids
    # pickling multi-GB data to worker processes)
    for i, uid in enumerate(user_ids):
        _USER_DATA_STORE = {uid: dataset_a[uid]}
        rows = process_user(uid)
        all_rows.extend(rows)

        # progress bar
        progress = (i + 1) / total_users
        filled   = int(BAR_W * progress)
        bar      = "#" * filled + "." * (BAR_W - filled)
        elapsed  = time.time() - t0
        eta      = (elapsed / progress - elapsed) if progress > 0 else 0
        sys.stdout.write(
            f"\r  [{bar}] {progress*100:5.1f}%  |  "
            f"Users {i+1:>3}/{total_users}  |  "
            f"{len(all_rows):>9} windows  |  "
            f"{elapsed/60:.1f}m  ETA {eta/60:.1f}m"
        )
        sys.stdout.flush()

        # Free memory for processed user
        del dataset_a[uid]
        if (i + 1) % 50 == 0:
            gc.collect()

    del dataset_a
    _USER_DATA_STORE = {}
    gc.collect()

    print()
    df = pd.DataFrame(all_rows, columns=all_columns)
    df[feature_columns] = df[feature_columns].astype(np.float32)
    df["window_idx"]    = df["window_idx"].astype(int)
    del all_rows
    gc.collect()
    print(f"  -> {len(df):,} windows extracted in {(time.time()-t0)/60:.1f} min")
    print()

    # ==================================================================
    # PHASE 3 -- Outlier Detection
    # ==================================================================
    print("> Phase 3: Outlier detection (IQR voting) ...")
    t1 = time.time()
    df = detect_outliers(df)
    n_bad_total = int(df["is_bad_window"].sum())
    print(f"  -> {n_bad_total:,} / {len(df):,} windows flagged as BAD "
          f"({n_bad_total/len(df)*100:.2f} %)  [{time.time()-t1:.1f}s]")

    # Outlier audit reports (before removal)
    print("  Generating outlier audit reports ...")
    rpt_gesture, rpt_user, rpt_detail = generate_outlier_report(df, OUTPUT_DIR)
    print(f"  -> {rpt_gesture}")
    print(f"  -> {rpt_user}")
    print(f"  -> {rpt_detail}")
    print()

    # ==================================================================
    # PHASE 4 -- Outlier Removal (drop bad windows)
    # ==================================================================
    print("> Phase 4: Outlier removal (drop bad windows) ...")
    t2 = time.time()
    dataset_training, pct_drop = correct_outliers(df)
    print(f"  -> {pct_drop:.2f} % windows dropped  ->  {len(dataset_training):,} remain")
    print(f"  [{time.time()-t2:.1f}s]")

    del df
    gc.collect()
    print()

    # ==================================================================
    # PHASE 5 -- Subject-Specific Z-Score Normalization
    # ==================================================================
    print("> Phase 5: Z-score normalization (per subject) ...")
    t3 = time.time()
    dataset_training = zscore_normalize(dataset_training)
    print(f"  [{time.time()-t3:.1f}s]")
    print()

    # ==================================================================
    # Save output
    # ==================================================================
    print("> Saving ...")

    # Drop helper columns before saving
    drop_cols = ["vote_count", "is_bad_window"]
    for c in drop_cols:
        if c in dataset_training.columns:
            dataset_training.drop(columns=c, inplace=True)

    out_path = OUTPUT_DIR / "dataset_TRAINING.parquet"
    dataset_training.to_parquet(out_path, index=False)

    total_elapsed = time.time() - t0
    print(f"  -> {out_path}  ({len(dataset_training):,} rows)")
    print()

    # ==================================================================
    # Verification summary
    # ==================================================================
    n_users = dataset_training["user"].nunique()
    n_cols  = len(dataset_training.columns)
    feat_c  = [c for c in dataset_training.columns if c in feature_columns]
    meta_c  = [c for c in dataset_training.columns if c in meta_columns]

    print("=" * 72)
    print("  VERIFICATION")
    print("=" * 72)
    print(f"  Unique users            : {n_users}")
    print(f"  Total rows (windows)    : {len(dataset_training):,}")
    print(f"  Total columns           : {n_cols}  "
          f"({len(feat_c)} features + {len(meta_c)} meta)")
    print(f"  Feature columns         : {feat_c[:3]} ... {feat_c[-3:]}")
    print(f"  Meta columns            : {meta_c}")

    # Per-user registration count
    reg_per_user = (dataset_training
                    .groupby("user")["sample_id"]
                    .nunique())
    print(f"  Registrations per user  : min={reg_per_user.min()}, "
          f"max={reg_per_user.max()}, median={reg_per_user.median():.0f}")
    print(f"  Gestures found          : "
          f"{sorted(dataset_training['label'].unique())}")
    print()
    print("=" * 72)
    print(f"  [OK] Pipeline complete in {total_elapsed/60:.1f} minutes")
    print("=" * 72)


if __name__ == "__main__":
    main()
