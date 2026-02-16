# Build the full training set from EMG-EPN612 dataset.
# Parallel processing + chunked Parquet saving.
#
# ── Dataset JSON structure ────────────────────────────────────────────────────
# Each user folder (e.g. user1/) contains a single JSON file (user1.json).
# The JSON has this structure:
#
#   {
#       "trainingSamples": {           # top-level key containing all repetitions
#           "sample_0": {              # key = repetition id ("sample_0", "sample_1", …)
#               "gestureName": "fist", # the gesture label for this repetition
#               "emg": {               # key holding the 8-channel EMG data
#                   "ch1": [0.12, …],  # list of float samples for channel 1
#                   "ch2": [0.05, …],  # list of float samples for channel 2
#                   …                  #   … through ch8
#                   "ch8": [0.09, …]
#               }
#           },
#           "sample_1": { … },
#           …
#       }
#   }
#
# Usage:
#   cd to the project root directory
#   python scripts/build_training_set.py

import json                                  # to load JSON dataset files
import os                                    # for listing directories
import sys                                   # for writing progress to stdout
import time                                  # to track elapsed / ETA
import numpy as np                           # numerical operations on EMG arrays
import pandas as pd                          # create DataFrames & save as Parquet
from pathlib import Path                     # cross-platform path handling
from scipy import signal                     # FIR / IIR filter design & application
from concurrent.futures import ProcessPoolExecutor  # parallel user processing
import gc                                    # explicit garbage collection between chunks

# ── Paths (relative to project root) ──────────────────────────────────────────
BASE_PATH     = Path("EMG-EPN612 Dataset")   # root folder of the dataset
TRAINING_PATH = BASE_PATH / "trainingJSON"   # folder with one sub-folder per user
CHUNKS_DIR    = Path("training_chunks")      # output folder for Parquet chunk files
CHUNKS_DIR.mkdir(exist_ok=True)              # create output dir if it doesn't exist

# ── Parameters ─────────────────────────────────────────────────────────────────
WINDOW_LENGTH = 40    # number of samples per sliding window (40 samples @ 200 Hz = 200 ms)
WINDOW_SHIFT  = 10    # step size of the sliding window (10 samples @ 200 Hz = 50 ms, 75% overlap)
THRESHOLD     = 0.1   # amplitude threshold used by WAMP and ZC features (tuned for z-scored data)
CHANNELS      = [f"ch{i}" for i in range(1, 9)]  # 8 EMG channels: "ch1" … "ch8"
TD9_NAMES     = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]  # 9 time-domain features
NUM_WORKERS   = 10    # max parallel processes (one per user)
CHUNK_SIZE    = 10    # how many users to batch into one Parquet file

# Build column names: 8 channels × 9 features = 72 feature columns
feature_columns = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
# Metadata columns appended after the 72 features
meta_columns    = ["label", "user", "sample_id", "window_start"]
# Full column list for the output DataFrame (72 + 4 = 76 columns)
all_columns     = feature_columns + meta_columns


# ── Signal preprocessing ─────────────────────────────────────────────────────
def filter_emg(emg_signal, fs=200, lowcut=20, highcut=95, notch_freq=50, notch_q=30):
    """Apply bandpass + notch filtering only (no normalization).
    Normalization is handled later at the subject level.

    Uses a 4th-order Butterworth IIR bandpass (20–95 Hz) instead of
    an FIR filter.  At fs=200 Hz the Nyquist frequency is 100 Hz,
    so highcut is set to 95 Hz to remain safely below Nyquist.
    """
    nyquist = fs / 2                                      # Nyquist freq = half the sampling rate
    low  = lowcut  / nyquist                              # normalise low cutoff to [0, 1]
    high = highcut / nyquist                              # normalise high cutoff to [0, 1]
    # 4th-order Butterworth bandpass (effective 8th-order after filtfilt)
    b_bp, a_bp = signal.butter(4, [low, high], btype="band")
    filtered = signal.filtfilt(b_bp, a_bp, emg_signal)    # zero-phase bandpass filtering
    b_notch, a_notch = signal.iirnotch(w0=notch_freq, Q=notch_q, fs=fs)  # design 50 Hz notch filter
    filtered = signal.filtfilt(b_notch, a_notch, filtered) # zero-phase notch filtering (remove powerline)
    return filtered                                       # return filtered signal (not normalised yet)


# ── TD9 feature functions ─────────────────────────────────────────────────────
# Each function takes a window (1-D numpy array) and returns a single scalar.

def _LS(x):
    """L-Scale: a robust dispersion measure (like a trimmed std dev)."""
    n = len(x)                                  # number of samples in the window
    if n < 2:                                   # guard: need at least 2 points
        return 0.0
    x_sorted = np.sort(x)                       # sort values ascending
    i = np.arange(1, n + 1)                     # rank indices 1..n
    return np.sum((2 * i - n - 1) * x_sorted) / (n * (n - 1))  # L-scale formula

def _MFL(x):
    """Maximum Fractal Length: log of the total waveform length."""
    return np.log(np.sqrt(np.sum(np.diff(x) ** 2)) + 1e-10)  # log(path length + eps)

def _MSR(x):
    """Mean Square Root: average of sqrt(|x|)."""
    return np.mean(np.sqrt(np.abs(x)))          # mean of element-wise sqrt(|x|)

def _WAMP(x, threshold):
    """Willison Amplitude: count of successive differences exceeding threshold."""
    return np.sum(np.abs(np.diff(x)) > threshold)  # how many |Δx| > threshold

def _ZC(x, threshold):
    """Zero Crossings: count of sign changes with amplitude check."""
    x1, x2 = x[:-1], x[1:]                     # pairs of consecutive samples
    return np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) > threshold))  # sign flip + magnitude

def _RMS(x):
    """Root Mean Square: sqrt(mean(x²))."""
    return np.sqrt(np.mean(x ** 2))             # classic RMS

def _IAV(x):
    """Integrated Absolute Value: sum(|x|)."""
    return np.sum(np.abs(x))                    # total absolute amplitude

def _DASDV(x):
    """Difference Absolute Standard Deviation Value."""
    return np.sqrt(np.mean(np.diff(x) ** 2))    # RMS of first differences

def _VAR(x):
    """Variance (sample variance with ddof=1)."""
    return np.var(x, ddof=1)                    # unbiased variance estimator

def extract_td9_array(window, threshold=THRESHOLD):
    """Compute all 9 TD features for one window, return as 1-D array of length 9."""
    return np.array([
        _LS(window), _MFL(window), _MSR(window),       # features 0-2
        _WAMP(window, threshold), _ZC(window, threshold),  # features 3-4
        _RMS(window), _IAV(window), _DASDV(window), _VAR(window)  # features 5-8
    ])


# ── Per-user processing function ──────────────────────────────────────────────
def process_user(user_folder):
    """Process one user entirely — safe for multiprocessing.

    Z-score normalization is computed per-subject, per-channel:
      For each channel j, μ and σ are calculated from every sample of this
      subject (all repetitions concatenated), then applied to every sample.
    """
    # ── Load the user's JSON ──────────────────────────────────────────────
    # Path example: EMG-EPN612 Dataset/trainingJSON/user1/user1.json
    json_path = TRAINING_PATH / user_folder / f"{user_folder}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)               # full JSON dict for this user

    # user_data["trainingSamples"] is a dict of repetitions
    # keys are like "sample_0", "sample_1", …
    samples = user_data.get("trainingSamples", {})  # access the top-level key
    sample_keys = list(samples.keys())              # list of repetition ids

    # ══════════════════════════════════════════════════════════════════════
    # PASS 1 — Filter every repetition & collect all data per channel
    #          so we can compute subject-level μ and σ for z-score.
    # ══════════════════════════════════════════════════════════════════════
    filtered_data = {}                              # will hold {sample_key: {ch: np.array}}
    channel_all = {ch: [] for ch in CHANNELS}       # accumulator: all filtered arrays per channel

    for sample_key in sample_keys:
        # samples[sample_key]["emg"] is a dict with keys "ch1"…"ch8"
        # each value is a list of float EMG samples for that channel
        emg_dict = samples[sample_key]["emg"]       # access the "emg" key for this repetition
        filtered_data[sample_key] = {}               # prepare storage for this repetition
        for ch in CHANNELS:                          # iterate "ch1" through "ch8"
            filt = filter_emg(np.array(emg_dict[ch]))  # bandpass + notch filter the raw signal
            filtered_data[sample_key][ch] = filt     # store filtered signal per repetition & channel
            channel_all[ch].append(filt)             # also collect it for global stats

    # Compute per-subject, per-channel mean (μ) & std (σ)
    # by concatenating ALL repetitions for this user into one long array per channel
    channel_stats = {}                               # will hold {ch: (μ, σ)}
    for ch in CHANNELS:
        concat = np.concatenate(channel_all[ch])     # one long array with every sample for this channel
        channel_stats[ch] = (np.mean(concat),        # μ = mean across all repetitions
                             np.std(concat) + 1e-8)  # σ = std dev (+ epsilon to avoid /0)
    del channel_all                                  # free the accumulator memory

    # ══════════════════════════════════════════════════════════════════════
    # PASS 2 — Z-score normalise each repetition using subject-level stats,
    #          then slide a window and extract TD9 features.
    # ══════════════════════════════════════════════════════════════════════
    rows = []                                        # accumulate one row per window
    for sample_key in sample_keys:
        # samples[sample_key]["gestureName"] is the label string, e.g. "fist"
        gesture = samples[sample_key]["gestureName"] # access the gesture label for this repetition

        # Normalise: x_norm = (x - μ) / σ   using subject-level stats
        norm_channels = {}                           # will hold {ch: normalised np.array}
        for ch in CHANNELS:
            mu, sigma = channel_stats[ch]            # unpack μ and σ for this channel
            norm_channels[ch] = (filtered_data[sample_key][ch] - mu) / sigma  # z-score formula

        n_samples = len(norm_channels[CHANNELS[0]])  # length of the normalised signal (same for all ch)

        # Slide a window of WINDOW_LENGTH samples, stepping by WINDOW_SHIFT
        for start in range(0, n_samples - WINDOW_LENGTH + 1, WINDOW_SHIFT):
            end = start + WINDOW_LENGTH              # end index of the current window
            # Extract 9 features for each of the 8 channels → 72-element vector
            feature_vec = np.concatenate([
                extract_td9_array(norm_channels[ch][start:end])  # 9 features for one channel
                for ch in CHANNELS                   # repeat for all 8 channels
            ])
            # Append feature vector + metadata as a Python list (avoids
            # numpy dtype coercion from mixing floats and strings).
            row = feature_vec.tolist() + [gesture, user_folder, sample_key, start]
            rows.append(row)

    del user_data, samples, filtered_data            # free memory before returning
    return rows                                      # list of 76-element arrays (one per window)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # List all user folders ("user1", "user2", …) and sort numerically
    user_folders = sorted(
        [d for d in os.listdir(TRAINING_PATH) if d.startswith("user")],
        key=lambda x: int(x.replace("user", ""))  # sort by number, not lexicographically
    )
    total_users = len(user_folders)                 # total number of subjects

    # Print run configuration
    print(f"EMG-EPN612 Training Set Builder")
    print(f"  Users       : {total_users}")
    print(f"  Workers     : {NUM_WORKERS}")
    print(f"  Chunk size  : {CHUNK_SIZE} users")
    print(f"  Output dir  : {CHUNKS_DIR.resolve()}")
    print(f"  CPU cores   : {os.cpu_count()}")
    print()

    start_time = time.time()                        # timestamp for progress tracking
    total_rows = 0                                  # running count of all output rows
    n_chunks = (total_users + CHUNK_SIZE - 1) // CHUNK_SIZE  # ceil division → number of chunks
    BAR_WIDTH = 40                                  # width of the text progress bar

    # Process users in chunks of CHUNK_SIZE, saving each chunk to a separate Parquet file
    for chunk_idx, chunk_start in enumerate(range(0, total_users, CHUNK_SIZE)):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_users)  # end index for this chunk
        chunk_file = CHUNKS_DIR / f"chunk_{chunk_start:04d}.parquet"  # output file for this chunk

        # Skip if already processed (makes the script resumable after interruption)
        if chunk_file.exists():
            total_rows += len(pd.read_parquet(chunk_file))  # count existing rows
            progress = (chunk_idx + 1) / n_chunks           # fraction complete
            filled = int(BAR_WIDTH * progress)               # filled portion of bar
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)  # visual progress bar
            sys.stdout.write(f"\r  [{bar}] {progress*100:5.1f}%  |  Users {chunk_end}/{total_users}  [SKIP]")
            sys.stdout.flush()                               # force immediate display
            continue                                         # skip to next chunk

        chunk_users = user_folders[chunk_start:chunk_end]     # list of user folder names in this chunk

        # Process each user in parallel using multiple worker processes
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
            results = list(pool.map(process_user, chunk_users))  # list of lists of row arrays

        # Flatten: merge all users' rows into one list
        chunk_rows = [row for user_rows in results for row in user_rows]
        # Build a DataFrame with 76 columns (72 features + 4 metadata)
        chunk_df = pd.DataFrame(chunk_rows, columns=all_columns)
        chunk_df[feature_columns] = chunk_df[feature_columns].astype(np.float32)  # cast features to float32
        chunk_df["window_start"]  = chunk_df["window_start"].astype(int)          # cast to integer
        chunk_df.to_parquet(chunk_file, index=False)  # save chunk as compressed Parquet

        # Update progress
        total_rows += len(chunk_rows)                 # accumulate row count
        elapsed = time.time() - start_time            # total time so far
        progress = (chunk_idx + 1) / n_chunks         # fraction complete
        eta = (elapsed / progress - elapsed) if progress > 0 else 0  # estimated time remaining
        filled = int(BAR_WIDTH * progress)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)

        # Overwrite same line with updated progress bar
        sys.stdout.write(
            f"\r  [{bar}] {progress*100:5.1f}%  |  "
            f"Users {chunk_end:>3}/{total_users}  |  "
            f"{total_rows:>8} rows  |  "
            f"{elapsed/60:.1f}m elapsed  |  ETA {eta/60:.1f}m"
        )
        sys.stdout.flush()

        del results, chunk_rows, chunk_df             # free memory for processed chunk
        gc.collect()                                  # force garbage collection

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n\n✓ All done in {elapsed/60:.1f} minutes")
    print(f"  Chunks saved to: {CHUNKS_DIR.resolve()}")

    # Verify: re-read all chunks and count total rows as a sanity check
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.parquet"))
    total = sum(len(pd.read_parquet(f)) for f in chunk_files)
    print(f"  Total rows across {len(chunk_files)} chunks: {total}")


if __name__ == "__main__":
    main()
