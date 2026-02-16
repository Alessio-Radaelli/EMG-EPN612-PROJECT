# ══════════════════════════════════════════════════════════════════════════════
# EMG-EPN612 — Full Preprocessing Pipeline
# ══════════════════════════════════════════════════════════════════════════════
#
# Five-phase pipeline:
#   Phase 1  Signal Conditioning & Segmentation
#   Phase 2  TD9 Feature Extraction (72 features per window)
#   Phase 3  Outlier Detection & Branching (SVM / DTW copies)
#   Phase 4  Outlier Correction & Audit Report
#   Phase 5  Subject-Specific Z-Score Normalization
#
# Dataset JSON key layout (per user):
#   trainingSamples -> idx_N -> gestureName        (str)
#                            -> emg -> ch1…ch8     (list[float])
#                            -> groundTruthIndex   [start, end]  ← 1-based
#                            -> groundTruth        list[0|1]
#   (noGesture samples have NO groundTruthIndex / groundTruth keys)
#
# Usage:
#   cd "EMG-EPN612 project"
#   python scripts/preprocess_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

import json
import os
import sys
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from concurrent.futures import ProcessPoolExecutor

# ── Paths (relative to project root) ────────────────────────────────────────
BASE_PATH     = Path("EMG-EPN612 Dataset")
TRAINING_PATH = BASE_PATH / "trainingJSON"
OUTPUT_DIR    = Path("preprocessed_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Global Parameters ───────────────────────────────────────────────────────
FS            = 200       # sampling frequency in Hz
WINDOW_LENGTH = 40        # 200 ms window at 200 Hz
WINDOW_SHIFT  = 10        # 50 ms step  → 75 % overlap
THRESHOLD     = 0.00001   # 10 µV threshold for WAMP / ZC (noise floor)
CHANNELS      = [f"ch{i}" for i in range(1, 9)]  # ch1 … ch8
TD9_NAMES     = ["LS", "MFL", "MSR", "WAMP", "ZC",
                 "RMS", "IAV", "DASDV", "VAR"]
NUM_WORKERS   = 8         # parallel worker processes
CHUNK_SIZE    = 10        # users per Parquet chunk

# noGesture: crop 1.3 s centred in the trial
NO_GESTURE_CROP_SAMPLES = int(1.3 * FS)   # 260 samples

# Outlier-detection thresholds
IQR_VOTE_PCT  = 0.25      # >25 % of features out-of-bounds → BAD
BAD_REP_PCT   = 0.20      # >20 % bad windows → drop entire repetition
BAD_CONSEC    = 3          # ≥3 consecutive bad windows → drop repetition

# Column names
feature_columns = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
meta_columns    = ["label", "user", "sample_id", "window_idx"]
all_columns     = feature_columns + meta_columns


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Signal Conditioning
# ══════════════════════════════════════════════════════════════════════════════

def bandpass_filter(emg_signal, fs=FS, lowcut=20, highcut=95):
    """2nd-order Butterworth IIR bandpass (20–95 Hz) via filtfilt.

    The plan explicitly omits the notch filter — only a bandpass is applied
    to the full 5 s trial before segmentation.
    """
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = signal.butter(2, [low, high], btype="band")
    return signal.filtfilt(b, a, emg_signal)


def segment_trial(filtered_channels, sample_meta):
    """Crop the filtered signal to the gesture region.

    For gestures   → use groundTruthIndex [start, end] (1-based → 0-based).
    For noGesture  → crop 1.3 s from the centre of the trial.

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
        # noGesture: take 1.3 s from the centre of the rest trial
        length  = len(next(iter(filtered_channels.values())))
        crop    = NO_GESTURE_CROP_SAMPLES
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

def process_user(user_folder):
    """Load one user's JSON, filter, segment, extract features.

    Returns a list of 76-element rows (features + metadata).
    Normalisation is deferred to Phase 5 (after outlier handling).
    """
    json_path = TRAINING_PATH / user_folder / f"{user_folder}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    samples     = user_data.get("trainingSamples", {})
    sample_keys = list(samples.keys())
    all_rows    = []

    for sample_key in sample_keys:
        sample = samples[sample_key]
        emg    = sample["emg"]
        gesture = sample["gestureName"]

        # Phase 1a — Bandpass filter full 5 s trial
        filtered = {ch: bandpass_filter(np.array(emg[ch])) for ch in CHANNELS}

        # Phase 1b — Segment (crop to gesture or centre-crop for noGesture)
        cropped = segment_trial(filtered, sample)

        # Phase 2 — Windowed TD9 feature extraction
        rows = windowed_features(cropped, gesture, user_folder, sample_key)
        all_rows.extend(rows)

    del user_data, samples
    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Outlier Detection & Branching
# ══════════════════════════════════════════════════════════════════════════════

def detect_outliers(df):
    """Add 'is_bad_window' column using Subject-Specific IQR voting.

    For each user independently:
      1. Compute Q1, Q3, IQR for every feature column.
      2. For each window, count how many features fall outside
         [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
      3. Mark as BAD if >25 % of feature values are outside the fences.
    """
    df = df.copy()
    df["vote_count"]    = 0
    df["is_bad_window"] = False

    for user, grp in df.groupby("user"):
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


def branch_datasets(df):
    """Create two copies from the flagged DataFrame.

    dataset_SVM  — flat copy (all windows + flags).
    dataset_DTW  — same data, used later for repetition-level analysis.
    """
    dataset_svm = df.copy()
    dataset_dtw = df.copy()
    return dataset_svm, dataset_dtw


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Outlier Correction & Analysis
# ══════════════════════════════════════════════════════════════════════════════

def correct_svm(df):
    """SVM branch: simply drop all bad windows."""
    n_before = len(df)
    df_clean = df[~df["is_bad_window"]].copy()
    n_after  = len(df_clean)
    pct_drop = (n_before - n_after) / n_before * 100 if n_before else 0
    return df_clean, pct_drop


def _max_consecutive_true(flags):
    """Return the length of the longest consecutive True run."""
    max_run = 0
    run     = 0
    for f in flags:
        if f:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def correct_dtw(df):
    """DTW branch: per-repetition analysis.

    For each (user, sample_id) group:
      • If <20 % windows are bad AND no run of ≥3 consecutive bad windows
        → interpolate bad windows from their neighbours.
      • Otherwise → drop the entire repetition.

    Returns (cleaned_df, stats_dict).
    """
    kept_parts        = []
    total_reps        = 0
    dropped_reps      = 0
    interpolated_reps = 0

    for (user, sid), grp in df.groupby(["user", "sample_id"]):
        total_reps += 1
        grp = grp.sort_values("window_idx").reset_index(drop=True)

        bad_flags  = grp["is_bad_window"].values
        n_windows  = len(bad_flags)
        n_bad      = bad_flags.sum()
        bad_ratio  = n_bad / n_windows if n_windows else 0
        max_consec = _max_consecutive_true(bad_flags)

        if n_bad == 0:
            # clean repetition — keep as-is
            kept_parts.append(grp)
            continue

        if bad_ratio > BAD_REP_PCT or max_consec >= BAD_CONSEC:
            # too many bad windows or a long gap — drop entire repetition
            dropped_reps += 1
            continue

        # Interpolate isolated bad windows with linear average of neighbours
        interpolated_reps += 1
        feat_arr = grp[feature_columns].values.copy()        # (n_windows, 72)
        for i in range(n_windows):
            if not bad_flags[i]:
                continue
            # Determine neighbour indices
            prev_i = i - 1 if i > 0 and not bad_flags[i - 1] else None
            next_i = i + 1 if i < n_windows - 1 and not bad_flags[i + 1] else None

            if prev_i is not None and next_i is not None:
                feat_arr[i] = (feat_arr[prev_i] + feat_arr[next_i]) / 2.0
            elif prev_i is not None:
                feat_arr[i] = feat_arr[prev_i]
            elif next_i is not None:
                feat_arr[i] = feat_arr[next_i]
            # else: leave as-is (isolated edge case — single-window rep)

        grp = grp.copy()
        grp[feature_columns]  = feat_arr
        grp["is_bad_window"]  = False        # all corrected
        kept_parts.append(grp)

    if kept_parts:
        df_clean = pd.concat(kept_parts, ignore_index=True)
    else:
        df_clean = pd.DataFrame(columns=df.columns)

    stats = {
        "total_reps":        total_reps,
        "dropped_reps":      dropped_reps,
        "interpolated_reps": interpolated_reps,
        "pct_dropped":       dropped_reps / total_reps * 100 if total_reps else 0,
        "pct_interpolated":  interpolated_reps / total_reps * 100 if total_reps else 0,
    }
    return df_clean, stats


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
    # ------------------------------------------------------------------
    # Enumerate users
    # ------------------------------------------------------------------
    user_folders = sorted(
        [d for d in os.listdir(TRAINING_PATH) if d.startswith("user")],
        key=lambda x: int(x.replace("user", "")),
    )
    total_users = len(user_folders)

    print("═" * 72)
    print("  EMG-EPN612 Preprocessing Pipeline")
    print("═" * 72)
    print(f"  Users          : {total_users}")
    print(f"  Workers        : {NUM_WORKERS}")
    print(f"  Window         : {WINDOW_LENGTH} samples ({WINDOW_LENGTH/FS*1000:.0f} ms)")
    print(f"  Step           : {WINDOW_SHIFT} samples ({WINDOW_SHIFT/FS*1000:.0f} ms)")
    print(f"  WAMP/ZC thr    : {THRESHOLD}")
    print(f"  IQR vote thr   : {IQR_VOTE_PCT*100:.0f} % of 72 features")
    print(f"  Output dir     : {OUTPUT_DIR.resolve()}")
    print(f"  CPU cores      : {os.cpu_count()}")
    print()

    # ==================================================================
    # PHASES 1 + 2 — Filter → Segment → Extract  (parallelised)
    # ==================================================================
    print("▸ Phase 1+2: Signal conditioning & feature extraction …")
    t0       = time.time()
    BAR_W    = 40
    n_chunks = (total_users + CHUNK_SIZE - 1) // CHUNK_SIZE
    all_rows = []

    for ci, cs in enumerate(range(0, total_users, CHUNK_SIZE)):
        ce = min(cs + CHUNK_SIZE, total_users)
        batch = user_folders[cs:ce]

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            results = list(pool.map(process_user, batch))

        for user_rows in results:
            all_rows.extend(user_rows)

        progress = (ci + 1) / n_chunks
        filled   = int(BAR_W * progress)
        bar      = "█" * filled + "░" * (BAR_W - filled)
        elapsed  = time.time() - t0
        eta      = (elapsed / progress - elapsed) if progress > 0 else 0
        sys.stdout.write(
            f"\r  [{bar}] {progress*100:5.1f}%  |  "
            f"Users {ce:>3}/{total_users}  |  "
            f"{len(all_rows):>9} windows  |  "
            f"{elapsed/60:.1f}m  ETA {eta/60:.1f}m"
        )
        sys.stdout.flush()
        del results
        gc.collect()

    print()
    df = pd.DataFrame(all_rows, columns=all_columns)
    df[feature_columns] = df[feature_columns].astype(np.float32)
    df["window_idx"]    = df["window_idx"].astype(int)
    del all_rows
    gc.collect()
    print(f"  → {len(df):,} windows extracted in {(time.time()-t0)/60:.1f} min")
    print()

    # ==================================================================
    # PHASE 3 — Outlier Detection & Branching
    # ==================================================================
    print("▸ Phase 3: Outlier detection (IQR voting) …")
    t1 = time.time()
    df = detect_outliers(df)
    n_bad_total = df["is_bad_window"].sum()
    print(f"  → {n_bad_total:,} / {len(df):,} windows flagged as BAD "
          f"({n_bad_total/len(df)*100:.2f} %)  [{time.time()-t1:.1f}s]")

    dataset_svm, dataset_dtw = branch_datasets(df)
    del df
    gc.collect()
    print()

    # ==================================================================
    # PHASE 4 — Outlier Correction & Report
    # ==================================================================
    print("▸ Phase 4: Outlier correction …")
    t2 = time.time()

    # SVM
    dataset_svm, svm_pct = correct_svm(dataset_svm)
    print(f"  SVM : {svm_pct:.2f} % windows dropped  →  {len(dataset_svm):,} remain")

    # DTW
    dataset_dtw, dtw_stats = correct_dtw(dataset_dtw)
    print(f"  DTW : {dtw_stats['pct_dropped']:.2f} % repetitions dropped, "
          f"{dtw_stats['pct_interpolated']:.2f} % interpolated  "
          f"→  {len(dataset_dtw):,} windows remain")
    print(f"        ({dtw_stats['dropped_reps']} dropped / "
          f"{dtw_stats['interpolated_reps']} interpolated / "
          f"{dtw_stats['total_reps']} total repetitions)")
    print(f"  [{time.time()-t2:.1f}s]")
    print()

    # ==================================================================
    # PHASE 5 — Subject-Specific Z-Score Normalization
    # ==================================================================
    print("▸ Phase 5: Z-score normalization (per subject) …")
    t3 = time.time()
    dataset_svm = zscore_normalize(dataset_svm)
    dataset_dtw = zscore_normalize(dataset_dtw)
    print(f"  [{time.time()-t3:.1f}s]")
    print()

    # ==================================================================
    # Save outputs
    # ==================================================================
    print("▸ Saving …")

    # Drop helper columns before saving
    drop_cols = ["vote_count", "is_bad_window"]
    for c in drop_cols:
        if c in dataset_svm.columns:
            dataset_svm.drop(columns=c, inplace=True)
        if c in dataset_dtw.columns:
            dataset_dtw.drop(columns=c, inplace=True)

    svm_path = OUTPUT_DIR / "dataset_SVM.parquet"
    dtw_path = OUTPUT_DIR / "dataset_DTW.parquet"
    dataset_svm.to_parquet(svm_path, index=False)
    dataset_dtw.to_parquet(dtw_path, index=False)

    total_elapsed = time.time() - t0
    print(f"  → {svm_path}  ({len(dataset_svm):,} rows)")
    print(f"  → {dtw_path}  ({len(dataset_dtw):,} rows)")
    print()
    print("═" * 72)
    print(f"  ✓ Pipeline complete in {total_elapsed/60:.1f} minutes")
    print("═" * 72)


if __name__ == "__main__":
    main()
