# Build the full training set from EMG-EPN612 dataset.
# Parallel processing + chunked Parquet saving.
#
# Usage:
#   cd to the project root directory
#   python scripts/build_training_set.py

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from concurrent.futures import ProcessPoolExecutor
import gc

# ── Paths (relative to project root) ──────────────────────────────────────────
BASE_PATH     = Path("EMG-EPN612 Dataset")
TRAINING_PATH = BASE_PATH / "trainingJSON"
CHUNKS_DIR    = Path("training_chunks")
CHUNKS_DIR.mkdir(exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
WINDOW_LENGTH = 40
WINDOW_SHIFT  = 5
THRESHOLD     = 0.01
CHANNELS      = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES     = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
NUM_WORKERS   = 10
CHUNK_SIZE    = 10

feature_columns = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
meta_columns    = ["label", "user", "sample_id", "window_start"]
all_columns     = feature_columns + meta_columns


# ── Signal preprocessing ─────────────────────────────────────────────────────
def preprocess_emg(emg_signal, fs=200, lowcut=20, highcut=500, notch_freq=50, notch_q=30):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = min(highcut, nyquist * 0.99) / nyquist
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    numtaps = 5
    b_bp = signal.firwin(numtaps, [low, high], pass_zero=False)
    filtered = signal.filtfilt(b_bp, [1.0], emg_signal)
    b_notch, a_notch = signal.iirnotch(w0=notch_freq, Q=notch_q, fs=fs)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)
    normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    return normalized


# ── TD9 feature functions ─────────────────────────────────────────────────────
def _LS(x):
    n = len(x)
    if n < 2:
        return 0.0
    x_sorted = np.sort(x)
    i = np.arange(1, n + 1)
    return np.sum((2 * i - n - 1) * x_sorted) / (n * (n - 1))

def _MFL(x):
    return np.log(np.sqrt(np.sum(np.diff(x) ** 2)) + 1e-10)

def _MSR(x):
    return np.mean(np.sqrt(np.abs(x)))

def _WAMP(x, threshold):
    return np.sum(np.abs(np.diff(x)) > threshold)

def _ZC(x, threshold):
    x1, x2 = x[:-1], x[1:]
    return np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) > threshold))

def _RMS(x):
    return np.sqrt(np.mean(x ** 2))

def _IAV(x):
    return np.sum(np.abs(x))

def _DASDV(x):
    return np.sqrt(np.mean(np.diff(x) ** 2))

def _VAR(x):
    return np.var(x, ddof=1)

def extract_td9_array(window, threshold=THRESHOLD):
    return np.array([
        _LS(window), _MFL(window), _MSR(window),
        _WAMP(window, threshold), _ZC(window, threshold),
        _RMS(window), _IAV(window), _DASDV(window), _VAR(window)
    ])


# ── Per-user processing function ──────────────────────────────────────────────
def process_user(user_folder):
    """Process one user entirely — safe for multiprocessing."""
    json_path = TRAINING_PATH / user_folder / f"{user_folder}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    samples = user_data.get("trainingSamples", {})
    rows = []

    for sample_key, sample in samples.items():
        emg_dict = sample["emg"]
        gesture = sample["gestureName"]

        norm_channels = {}
        for ch in CHANNELS:
            norm_channels[ch] = preprocess_emg(np.array(emg_dict[ch]))

        n_samples = len(norm_channels[CHANNELS[0]])

        for start in range(0, n_samples - WINDOW_LENGTH + 1, WINDOW_SHIFT):
            end = start + WINDOW_LENGTH
            feature_vec = np.concatenate([
                extract_td9_array(norm_channels[ch][start:end])
                for ch in CHANNELS
            ])
            rows.append(np.append(feature_vec, [gesture, user_folder, sample_key, start]))

    del user_data, samples
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    user_folders = sorted(
        [d for d in os.listdir(TRAINING_PATH) if d.startswith("user")],
        key=lambda x: int(x.replace("user", ""))
    )
    total_users = len(user_folders)

    print(f"EMG-EPN612 Training Set Builder")
    print(f"  Users       : {total_users}")
    print(f"  Workers     : {NUM_WORKERS}")
    print(f"  Chunk size  : {CHUNK_SIZE} users")
    print(f"  Output dir  : {CHUNKS_DIR.resolve()}")
    print(f"  CPU cores   : {os.cpu_count()}")
    print()

    start_time = time.time()
    total_rows = 0
    n_chunks = (total_users + CHUNK_SIZE - 1) // CHUNK_SIZE
    BAR_WIDTH = 40

    for chunk_idx, chunk_start in enumerate(range(0, total_users, CHUNK_SIZE)):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_users)
        chunk_file = CHUNKS_DIR / f"chunk_{chunk_start:04d}.parquet"

        # Skip if already processed (resumable)
        if chunk_file.exists():
            total_rows += len(pd.read_parquet(chunk_file))
            progress = (chunk_idx + 1) / n_chunks
            filled = int(BAR_WIDTH * progress)
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            sys.stdout.write(f"\r  [{bar}] {progress*100:5.1f}%  |  Users {chunk_end}/{total_users}  [SKIP]")
            sys.stdout.flush()
            continue

        chunk_users = user_folders[chunk_start:chunk_end]

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            results = list(pool.map(process_user, chunk_users))

        chunk_rows = [row for user_rows in results for row in user_rows]
        chunk_df = pd.DataFrame(chunk_rows, columns=all_columns)
        chunk_df[feature_columns] = chunk_df[feature_columns].astype(np.float32)
        chunk_df["window_start"]  = chunk_df["window_start"].astype(int)
        chunk_df.to_parquet(chunk_file, index=False)

        total_rows += len(chunk_rows)
        elapsed = time.time() - start_time
        progress = (chunk_idx + 1) / n_chunks
        eta = (elapsed / progress - elapsed) if progress > 0 else 0
        filled = int(BAR_WIDTH * progress)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)

        sys.stdout.write(
            f"\r  [{bar}] {progress*100:5.1f}%  |  "
            f"Users {chunk_end:>3}/{total_users}  |  "
            f"{total_rows:>8} rows  |  "
            f"{elapsed/60:.1f}m elapsed  |  ETA {eta/60:.1f}m"
        )
        sys.stdout.flush()

        del results, chunk_rows, chunk_df
        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n\n✓ All done in {elapsed/60:.1f} minutes")
    print(f"  Chunks saved to: {CHUNKS_DIR.resolve()}")

    # Verify
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.parquet"))
    total = sum(len(pd.read_parquet(f)) for f in chunk_files)
    print(f"  Total rows across {len(chunk_files)} chunks: {total}")


if __name__ == "__main__":
    main()
