# ══════════════════════════════════════════════════════════════════════════════
# EMG-EPN612 -- Testing Pipeline  (dataset_B -> dataset_TEST)
# ══════════════════════════════════════════════════════════════════════════════
#
# Three-phase pipeline (no outlier detection/removal):
#   Phase 1  Signal Conditioning (bandpass + 50 Hz notch) & Segmentation
#   Phase 2  TD9 Feature Extraction (72 features per window)
#   Phase 3  Subject-Specific Z-Score Normalization
#
# Input : datasets/dataset_B.pkl  (153 users, 150 registrations each)
# Output: preprocessed_output/dataset_TEST.parquet
#
# Expected verification:
#   - 153 unique users
#   - 22,950 total repetitions  (153 × 150)
#   - 76 columns  (72 EMG features + 4 metadata)
#
# Usage:
#   cd <project root>
#   python scripts/Creation_dataset_TESTING.py
# ══════════════════════════════════════════════════════════════════════════════

import gc
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal

# -- Paths (relative to project root) ----------------------------------------
DATASET_B_PATH = Path("datasets") / "dataset_B.pkl"
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

# noGesture: fallback crop length if no gesture segments are available
NO_GESTURE_CROP_FALLBACK = int(1.3 * FS)  # 260 samples

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
# Per-user worker (Phases 1 + 2)
# ══════════════════════════════════════════════════════════════════════════════

_USER_DATA_STORE = {}


def process_user(user_id):
    """Filter, segment, extract features for one user.

    Reads from _USER_DATA_STORE[user_id] (trainingSamples dict).
    Returns a list of 76-element rows (features + metadata).
    """
    samples     = _USER_DATA_STORE[user_id]
    sample_keys = list(samples.keys())
    user_label  = f"user{user_id}"
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
# PHASE 3 — Subject-Specific Z-Score Normalization
# ══════════════════════════════════════════════════════════════════════════════

def zscore_normalize(df):
    """Apply per-subject Z-score to all 72 feature columns.

    For each user, µ and σ are computed from that user's data.
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
    # Load dataset_B.pkl
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  EMG-EPN612 Testing Pipeline  (dataset_B -> dataset_TEST)")
    print("=" * 72)

    print(f"\n  Loading {DATASET_B_PATH} ...")
    t_load = time.time()
    with open(DATASET_B_PATH, "rb") as f:
        dataset_b = pickle.load(f)       # {overall_user_id: trainingSamples}
    total_users = len(dataset_b)
    print(f"  Loaded {total_users} users in {time.time()-t_load:.1f}s")

    user_ids = sorted(dataset_b.keys())

    print(f"  Window         : {WINDOW_LENGTH} samples ({WINDOW_LENGTH/FS*1000:.0f} ms)")
    print(f"  Step           : {WINDOW_SHIFT} samples ({WINDOW_SHIFT/FS*1000:.0f} ms)")
    print(f"  Filter         : bandpass 20-95 Hz + 50 Hz notch")
    print(f"  WAMP/ZC thr    : {THRESHOLD}")
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

    for i, uid in enumerate(user_ids):
        _USER_DATA_STORE = {uid: dataset_b[uid]}
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
        del dataset_b[uid]
        if (i + 1) % 50 == 0:
            gc.collect()

    del dataset_b
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
    # PHASE 3 -- Subject-Specific Z-Score Normalization
    # ==================================================================
    print("> Phase 3: Z-score normalization (per subject) ...")
    t1 = time.time()
    dataset_test = zscore_normalize(df)
    del df
    gc.collect()
    print(f"  [{time.time()-t1:.1f}s]")
    print()

    # ==================================================================
    # Save output
    # ==================================================================
    print("> Saving ...")
    out_path = OUTPUT_DIR / "dataset_TEST.parquet"
    dataset_test.to_parquet(out_path, index=False)

    total_elapsed = time.time() - t0
    print(f"  -> {out_path}  ({len(dataset_test):,} rows)")
    print()

    # ==================================================================
    # Verification summary
    # ==================================================================
    n_users   = dataset_test["user"].nunique()
    n_cols    = len(dataset_test.columns)
    feat_c    = [c for c in dataset_test.columns if c in feature_columns]
    meta_c    = [c for c in dataset_test.columns if c in meta_columns]
    total_reps = dataset_test.groupby(["user", "sample_id"]).ngroups

    print("=" * 72)
    print("  VERIFICATION")
    print("=" * 72)
    print(f"  Unique users            : {n_users}   "
          f"{'[OK]' if n_users == 153 else '[FAIL — expected 153]'}")
    print(f"  Total repetitions       : {total_reps:,}   "
          f"{'[OK]' if total_reps == 22950 else '[FAIL — expected 22,950]'}")
    print(f"  Total columns           : {n_cols}  "
          f"({len(feat_c)} features + {len(meta_c)} meta)   "
          f"{'[OK]' if n_cols == 76 else '[FAIL — expected 76]'}")
    print(f"  Total rows (windows)    : {len(dataset_test):,}")
    print(f"  Feature columns         : {feat_c[:3]} ... {feat_c[-3:]}")
    print(f"  Meta columns            : {meta_c}")

    # Per-user registration count
    reg_per_user = (dataset_test
                    .groupby("user")["sample_id"]
                    .nunique())
    print(f"  Registrations per user  : min={reg_per_user.min()}, "
          f"max={reg_per_user.max()}, median={reg_per_user.median():.0f}")
    print(f"  Gestures found          : "
          f"{sorted(dataset_test['label'].unique())}")
    print()

    # Final pass/fail
    all_ok = (n_users == 153) and (total_reps == 22950) and (n_cols == 76)
    if all_ok:
        print("  *** ALL VERIFICATION CHECKS PASSED ***")
    else:
        print("  *** SOME VERIFICATION CHECKS FAILED — see above ***")

    print()
    print("=" * 72)
    print(f"  [OK] Pipeline complete in {total_elapsed/60:.1f} minutes")
    print("=" * 72)


if __name__ == "__main__":
    main()
