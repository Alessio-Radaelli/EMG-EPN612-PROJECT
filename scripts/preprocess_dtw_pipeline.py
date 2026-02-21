# ══════════════════════════════════════════════════════════════════════════════
# EMG-EPN612 -- DTW Preprocessing Pipeline
# ══════════════════════════════════════════════════════════════════════════════
#
# Produces DTW-ready datasets from raw JSON, using the SAME samples and user
# split as the existing TRAINING/TEST pipelines.
#
# Sub-windowing strategy:
#   Each 200 ms macro-window (40 samples) is divided into 8 contiguous
#   mini-windows of 25 ms (5 samples).  TD9 features (9 per channel × 8 ch
#   = 72) are extracted from each mini-window.
#
#   A repetition with N macro-windows becomes a DTW time-series of N × 8
#   steps, each 72-dimensional.
#
# Phases:
#   1  Signal Conditioning (bandpass + 50 Hz notch) & Segmentation
#   2  Macro-windowed Feature Extraction (for outlier detection compat.)
#   3  Outlier Detection + Removal (IQR voting, training only)
#   4  Sub-windowed Feature Re-extraction (8 mini-windows per macro-window)
#   5  Subject-Specific Z-Score Normalization
#
# User split (from build_datasets_AB.py):
#   Dataset A (train) — overall users 1-459:
#       trainingJSON/user1..user306     (overall 1-306)
#       testingJSON/user1..user153      (overall 307-459)
#   Dataset B (test)  — overall users 460-612:
#       testingJSON/user154..user306    (overall 460-612)
#
# Output:
#   preprocessed_output/dataset_DTW_TRAINING.parquet
#   preprocessed_output/dataset_DTW_TEST.parquet
#
# Usage:
#   cd <project root>
#   python scripts/preprocess_dtw_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal

# -- Paths (relative to project root) ----------------------------------------
BASE_PATH      = Path("EMG-EPN612 Dataset")
TRAINING_PATH  = BASE_PATH / "trainingJSON"
TESTING_PATH   = BASE_PATH / "testingJSON"
OUTPUT_DIR     = Path("preprocessed_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# -- Global Parameters -------------------------------------------------------
FS             = 200       # sampling frequency in Hz
WINDOW_LENGTH  = 40        # 200 ms macro-window at 200 Hz
WINDOW_SHIFT   = 10        # 50 ms step -> 75% overlap
MINI_WIN_LEN   = 5         # 25 ms mini-window (5 samples)
N_MINI_WINDOWS = WINDOW_LENGTH // MINI_WIN_LEN   # 8
THRESHOLD      = 0.00001   # 10 µV threshold for WAMP / ZC
CHANNELS       = [f"ch{i}" for i in range(1, 9)]  # ch1..ch8
TD9_NAMES      = ["LS", "MFL", "MSR", "WAMP", "ZC",
                   "RMS", "IAV", "DASDV", "VAR"]

# noGesture: fallback crop length
NO_GESTURE_CROP_FALLBACK = int(1.3 * FS)  # 260 samples

# Outlier-detection thresholds
IQR_VOTE_PCT = 0.25  # >25% of features out-of-bounds -> BAD

# Column names
feature_columns     = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
macro_meta_columns  = ["label", "user", "sample_id", "window_idx"]
macro_all_columns   = feature_columns + macro_meta_columns
mini_meta_columns   = ["label", "user", "sample_id", "window_idx",
                        "miniwindow_idx"]
mini_all_columns    = feature_columns + mini_meta_columns


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Signal Conditioning & Segmentation
# ══════════════════════════════════════════════════════════════════════════════

def filter_emg(emg_signal, fs=FS, lowcut=20, highcut=95,
               notch_freq=50, notch_q=30):
    """Bandpass (20-95 Hz) + 50 Hz notch filter via zero-phase filtfilt."""
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b_bp, a_bp = signal.butter(2, [low, high], btype="band")
    filtered = signal.filtfilt(b_bp, a_bp, emg_signal)
    b_notch, a_notch = signal.iirnotch(w0=notch_freq, Q=notch_q, fs=fs)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)
    return filtered


def segment_trial(filtered_channels, sample_meta, no_gesture_crop=None):
    """Crop the filtered signal to the gesture region."""
    gesture = sample_meta["gestureName"]

    if gesture != "noGesture":
        gti   = sample_meta["groundTruthIndex"]
        start = gti[0] - 1
        end   = gti[1]
        return {ch: arr[start:end] for ch, arr in filtered_channels.items()}
    else:
        length  = len(next(iter(filtered_channels.values())))
        crop    = no_gesture_crop if no_gesture_crop else NO_GESTURE_CROP_FALLBACK
        centre  = length // 2
        start   = max(centre - crop // 2, 0)
        end     = start + crop
        if end > length:
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


def windowed_features_macro(cropped_channels, label, user, sample_id):
    """Extract 72 features per macro-window (for outlier detection).

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


def windowed_features_mini(cropped_channels, label, user, sample_id,
                           valid_window_indices=None):
    """Extract 72 features per mini-window (8 per macro-window).

    Parameters
    ----------
    cropped_channels : dict
        {ch: np.array} with the cropped signals.
    label, user, sample_id : str
        Metadata for each row.
    valid_window_indices : set or None
        If not None, only emit mini-windows for macro-windows whose
        window_idx is in this set (i.e. outlier-free windows).

    Returns a list of row-lists (each 77: 72 features + 5 meta).
    """
    n_samples = len(next(iter(cropped_channels.values())))
    rows = []
    win_idx = 0
    for start in range(0, n_samples - WINDOW_LENGTH + 1, WINDOW_SHIFT):
        if valid_window_indices is not None and win_idx not in valid_window_indices:
            win_idx += 1
            continue

        # Sub-divide the macro-window into 8 mini-windows
        for mini_idx in range(N_MINI_WINDOWS):
            mini_start = start + mini_idx * MINI_WIN_LEN
            mini_end   = mini_start + MINI_WIN_LEN
            feat = np.concatenate([
                extract_td9(cropped_channels[ch][mini_start:mini_end])
                for ch in CHANNELS
            ])
            rows.append(feat.tolist() + [label, user, sample_id,
                                         win_idx, mini_idx])
        win_idx += 1
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# JSON Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_training_samples(json_path):
    """Load and return the trainingSamples dict from a user JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["trainingSamples"]


def resolve_json_path(folder, user_folder_name):
    """Return the path to the JSON file inside a user folder."""
    return folder / user_folder_name / f"{user_folder_name}.json"


# ══════════════════════════════════════════════════════════════════════════════
# Per-user worker
# ══════════════════════════════════════════════════════════════════════════════

def process_user_macro(user_label, samples):
    """Filter, segment, extract MACRO-window features for one user.

    Returns a list of 76-element rows (72 features + 4 meta).
    """
    sample_keys = list(samples.keys())
    all_rows = []

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

        # Phase 1a -- Bandpass + notch filter
        filtered = {ch: filter_emg(np.array(emg[ch], dtype=np.float64))
                    for ch in CHANNELS}

        # Phase 1b -- Segment
        cropped = segment_trial(filtered, sample,
                                no_gesture_crop=median_gesture_len)

        # Phase 2 -- Macro-window TD9 feature extraction
        rows = windowed_features_macro(cropped, gesture, user_label,
                                       sample_key)
        all_rows.extend(rows)

    return all_rows


def process_user_mini(user_label, samples, valid_windows=None):
    """Filter, segment, extract SUB-WINDOWED features for one user.

    Parameters
    ----------
    valid_windows : dict or None
        {sample_id: set(window_idx)} — only emit mini-windows for these
        macro-windows (post-outlier-removal).  None = keep all.

    Returns a list of 77-element rows (72 features + 5 meta).
    """
    sample_keys = list(samples.keys())
    all_rows = []

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

        filtered = {ch: filter_emg(np.array(emg[ch], dtype=np.float64))
                    for ch in CHANNELS}
        cropped = segment_trial(filtered, sample,
                                no_gesture_crop=median_gesture_len)

        valid_idx = (valid_windows.get(sample_key) if valid_windows
                     else None)
        rows = windowed_features_mini(cropped, gesture, user_label,
                                      sample_key,
                                      valid_window_indices=valid_idx)
        all_rows.extend(rows)

    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# Outlier Detection (macro-window level, same as preprocess_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def detect_outliers(df):
    """Add 'is_bad_window' column using Subject- & Gesture-Specific IQR voting."""
    df = df.copy()
    df["is_bad_window"] = False

    for (user, label), grp in df.groupby(["user", "label"]):
        idx = grp.index
        feat_vals = grp[feature_columns].values

        q1  = np.percentile(feat_vals, 25, axis=0)
        q3  = np.percentile(feat_vals, 75, axis=0)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outside = (feat_vals < lower) | (feat_vals > upper)
        votes   = outside.sum(axis=1)

        n_feats = len(feature_columns)
        bad     = votes > (IQR_VOTE_PCT * n_feats)

        df.loc[idx, "is_bad_window"] = bad

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Z-Score Normalization
# ══════════════════════════════════════════════════════════════════════════════

def zscore_normalize(df):
    """Apply per-subject Z-score to all 72 feature columns."""
    df = df.copy()
    for user, grp in df.groupby("user"):
        idx  = grp.index
        vals = grp[feature_columns].values.astype(np.float64)
        mu   = vals.mean(axis=0)
        sig  = vals.std(axis=0) + 1e-8
        df.loc[idx, feature_columns] = ((vals - mu) / sig).astype(np.float32)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Dataset builder (one dataset at a time)
# ══════════════════════════════════════════════════════════════════════════════

def build_user_map():
    """Return two lists of (overall_user_id, json_path) for dataset A and B.

    Replicates the split from build_datasets_AB.py.
    """
    dataset_a_users = []  # (overall_uid, json_path)
    dataset_b_users = []

    # Part 1: trainingJSON users 1-306  →  Dataset A  (overall 1-306)
    for uid in range(1, 307):
        folder_name = f"user{uid}"
        json_path = resolve_json_path(TRAINING_PATH, folder_name)
        dataset_a_users.append((uid, json_path))

    # Part 2: testingJSON users 1-153  →  Dataset A  (overall 307-459)
    for local_uid in range(1, 154):
        overall_uid = local_uid + 306
        folder_name = f"user{local_uid}"
        json_path = resolve_json_path(TESTING_PATH, folder_name)
        dataset_a_users.append((overall_uid, json_path))

    # Part 3: testingJSON users 154-306  →  Dataset B  (overall 460-612)
    for local_uid in range(154, 307):
        overall_uid = local_uid + 306
        folder_name = f"user{local_uid}"
        json_path = resolve_json_path(TESTING_PATH, folder_name)
        dataset_b_users.append((overall_uid, json_path))

    return dataset_a_users, dataset_b_users


def process_dataset(user_list, dataset_name, apply_outlier_removal=True):
    """Process a list of (overall_uid, json_path) into a DTW parquet.

    For training: uses outlier detection at the macro-window level, then
    re-extracts sub-windowed features only for surviving macro-windows.
    For testing: skips outlier detection; all windows kept.

    Returns the final DataFrame.
    """
    total_users = len(user_list)
    BAR_W = 40

    # ── PASS 1 (training only): macro-window extraction + outlier detection ──
    if apply_outlier_removal:
        print(f"\n> {dataset_name} — Pass 1: Macro-window extraction "
              f"(for outlier detection) ...")
        t0 = time.time()
        macro_rows = []

        for i, (uid, json_path) in enumerate(user_list):
            if not json_path.exists():
                print(f"\n  WARNING: {json_path} not found, skipping")
                continue
            samples = load_training_samples(json_path)
            user_label = f"user{uid}"
            rows = process_user_macro(user_label, samples)
            macro_rows.extend(rows)

            progress = (i + 1) / total_users
            filled = int(BAR_W * progress)
            bar = "#" * filled + "." * (BAR_W - filled)
            elapsed = time.time() - t0
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            sys.stdout.write(
                f"\r  [{bar}] {progress*100:5.1f}%  |  "
                f"Users {i+1:>3}/{total_users}  |  "
                f"{len(macro_rows):>9} windows  |  "
                f"{elapsed/60:.1f}m  ETA {eta/60:.1f}m"
            )
            sys.stdout.flush()

        print()
        df_macro = pd.DataFrame(macro_rows, columns=macro_all_columns)
        df_macro[feature_columns] = df_macro[feature_columns].astype(
            np.float32)
        df_macro["window_idx"] = df_macro["window_idx"].astype(int)
        del macro_rows
        gc.collect()
        print(f"  -> {len(df_macro):,} macro-windows extracted  "
              f"({(time.time()-t0)/60:.1f} min)")

        # Outlier detection
        print(f"\n> {dataset_name} — Outlier detection (IQR voting) ...")
        t1 = time.time()
        df_macro = detect_outliers(df_macro)
        n_bad = int(df_macro["is_bad_window"].sum())
        print(f"  -> {n_bad:,} / {len(df_macro):,} windows flagged as BAD  "
              f"({n_bad/len(df_macro)*100:.2f}%)  [{time.time()-t1:.1f}s]")

        # Build valid_windows lookup: {user: {sample_id: set(window_idx)}}
        df_good = df_macro[~df_macro["is_bad_window"]]
        valid_windows_by_user = {}
        for (user, sid), grp in df_good.groupby(["user", "sample_id"]):
            valid_windows_by_user.setdefault(user, {})[sid] = set(
                grp["window_idx"].values)

        del df_macro, df_good
        gc.collect()
    else:
        valid_windows_by_user = None

    # ── PASS 2: sub-windowed feature extraction ──────────────────────────────
    outlier_tag = ("(outlier-filtered)" if apply_outlier_removal
                   else "(no outlier filter)")
    print(f"\n> {dataset_name} — Pass 2: Sub-windowed feature extraction "
          f"{outlier_tag} ...")
    t2 = time.time()
    mini_rows = []

    for i, (uid, json_path) in enumerate(user_list):
        if not json_path.exists():
            continue
        samples = load_training_samples(json_path)
        user_label = f"user{uid}"

        valid_wins = (valid_windows_by_user.get(user_label)
                      if valid_windows_by_user else None)
        rows = process_user_mini(user_label, samples,
                                 valid_windows=valid_wins)
        mini_rows.extend(rows)

        progress = (i + 1) / total_users
        filled = int(BAR_W * progress)
        bar = "#" * filled + "." * (BAR_W - filled)
        elapsed = time.time() - t2
        eta = (elapsed / progress - elapsed) if progress > 0 else 0
        sys.stdout.write(
            f"\r  [{bar}] {progress*100:5.1f}%  |  "
            f"Users {i+1:>3}/{total_users}  |  "
            f"{len(mini_rows):>9} mini-windows  |  "
            f"{elapsed/60:.1f}m  ETA {eta/60:.1f}m"
        )
        sys.stdout.flush()

    print()
    df = pd.DataFrame(mini_rows, columns=mini_all_columns)
    df[feature_columns] = df[feature_columns].astype(np.float32)
    df["window_idx"]       = df["window_idx"].astype(int)
    df["miniwindow_idx"]   = df["miniwindow_idx"].astype(int)
    del mini_rows
    gc.collect()
    print(f"  -> {len(df):,} mini-windows extracted  "
          f"({(time.time()-t2)/60:.1f} min)")

    # ── Z-Score Normalization ────────────────────────────────────────────────
    print(f"\n> {dataset_name} — Z-score normalization (per subject) ...")
    t3 = time.time()
    df = zscore_normalize(df)
    print(f"  [{time.time()-t3:.1f}s]")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("=" * 72)
    print("  EMG-EPN612 DTW Preprocessing Pipeline")
    print("  (raw JSON -> sub-windowed DTW datasets)")
    print("=" * 72)
    print(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Macro-window  : {WINDOW_LENGTH} samples ({WINDOW_LENGTH/FS*1000:.0f} ms)")
    print(f"  Macro-step    : {WINDOW_SHIFT} samples ({WINDOW_SHIFT/FS*1000:.0f} ms)")
    print(f"  Mini-window   : {MINI_WIN_LEN} samples ({MINI_WIN_LEN/FS*1000:.0f} ms)")
    print(f"  Mini-windows  : {N_MINI_WINDOWS} per macro-window")
    print(f"  Filter        : bandpass 20-95 Hz + 50 Hz notch")
    print(f"  WAMP/ZC thr   : {THRESHOLD}")
    print(f"  IQR vote thr  : {IQR_VOTE_PCT*100:.0f}% of 72 features")
    print(f"  Output dir    : {OUTPUT_DIR.resolve()}")
    print(f"  CPU cores     : {os.cpu_count()}")

    # Build user lists
    dataset_a_users, dataset_b_users = build_user_map()
    print(f"\n  Dataset A (training): {len(dataset_a_users)} users  "
          f"(overall 1-459)")
    print(f"  Dataset B (testing) : {len(dataset_b_users)} users  "
          f"(overall 460-612)")

    # ── Process Dataset A (training) with outlier removal ────────────────────
    print(f"\n{'═'*72}")
    print(f"  DATASET A — Training  (with outlier detection)")
    print(f"{'═'*72}")
    df_train = process_dataset(dataset_a_users, "Dataset A",
                               apply_outlier_removal=True)

    out_train = OUTPUT_DIR / "dataset_DTW_TRAINING.parquet"
    df_train.to_parquet(out_train, index=False)
    print(f"\n  -> Saved {out_train}  ({len(df_train):,} rows)")

    # ── Process Dataset B (testing) without outlier removal ──────────────────
    print(f"\n{'═'*72}")
    print(f"  DATASET B — Testing  (no outlier detection)")
    print(f"{'═'*72}")
    df_test = process_dataset(dataset_b_users, "Dataset B",
                              apply_outlier_removal=False)

    out_test = OUTPUT_DIR / "dataset_DTW_TEST.parquet"
    df_test.to_parquet(out_test, index=False)
    print(f"\n  -> Saved {out_test}  ({len(df_test):,} rows)")

    # ── Verification ─────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  VERIFICATION")
    print(f"{'═'*72}")

    for tag, df, expected_users in [("TRAINING", df_train, 459),
                                     ("TEST", df_test, 153)]:
        n_users = df["user"].nunique()
        n_reps  = df.groupby(["user", "sample_id"]).ngroups
        n_cols  = len(df.columns)

        # Check mini-windows per macro-window
        mini_per_macro = df.groupby(
            ["user", "sample_id", "window_idx"])["miniwindow_idx"].nunique()
        all_8 = (mini_per_macro == N_MINI_WINDOWS).all()

        print(f"\n  {tag}:")
        print(f"    Unique users        : {n_users}   "
              f"{'[OK]' if n_users == expected_users else f'[FAIL — expected {expected_users}]'}")
        print(f"    Total repetitions   : {n_reps:,}")
        print(f"    Total rows (mini-w) : {len(df):,}")
        print(f"    Columns             : {n_cols}  "
              f"({'[OK]' if n_cols == 77 else '[FAIL — expected 77]'})")
        print(f"    8 mini-win/macro    : "
              f"{'[OK]' if all_8 else '[FAIL — some groups ≠ 8]'}")
        print(f"    Gestures            : "
              f"{sorted(df['label'].unique())}")

    del df_train, df_test
    gc.collect()

    total_elapsed = time.time() - t_total
    print(f"\n{'═'*72}")
    print(f"  [OK] Pipeline complete in {total_elapsed/60:.1f} minutes")
    print(f"  Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*72}")


if __name__ == "__main__":
    main()
