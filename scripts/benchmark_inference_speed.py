"""
EMG-EPN612 — Inference Speed Benchmark (18-feature models)

Evaluates inference speed of KNN, TDCNN, XGBoost, and SVM models, taking into
account preprocessing overhead. TDCNN requires NO feature extraction (raw windows
directly); KNN/XGBoost/SVM require TD9 feature extraction per window.

Data (all from datasets/dataset_B.pkl when available):
  - TDCNN: filter→segment→window→z-score→inference (reduced pipeline, no TD9)
  - KNN/XGBoost/SVM: filter→segment→window→TD9→z-score→inference (shared + TD9)

Requirements: torch, xgboost, joblib, faiss-gpu (for KNN), pyarrow (for parquet)
Models: models/18f/knn18_*.joblib, tdcnn_emg_model.pth, xgboost18_*.json, svm_val_best18.pt

Usage:
    cd "EMG-EPN612 project"
    python scripts/benchmark_inference_speed.py
    python scripts/benchmark_inference_speed.py --max-samples 10000 --repetitions 3
    python scripts/benchmark_inference_speed.py --load-only   # Show saved results, no re-run
"""

import sys
import time
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

PREPROC_DIR = PROJECT_ROOT / "preprocessed_output"
MODELS_DIR = PROJECT_ROOT / "models" / "18f"
DATASET_B_PATH = PROJECT_ROOT / "datasets" / "dataset_B.pkl"

# Model paths (18-feature case)
KNN_PATH = MODELS_DIR / "knn18_faiss_gpu_enn_manhattan_k1_wuniform.joblib"
TDCNN_PATH = MODELS_DIR / "tdcnn_emg_model.pth"
XGB_PATH = MODELS_DIR / "xgboost18_best_halving.json"
SVM_PATH = MODELS_DIR / "svm_val_best18.pt"

# Test data
NPZ_TEST_PATH = PREPROC_DIR / "dataset_TESTING.npz"
PARQUET_18_PATH = PREPROC_DIR / "dataset_TESTING_reduced18.parquet"
BENCHMARK_RESULTS_PATH = MODELS_DIR / "benchmark_inference_results.json"

# Constants (from preprocess_pipeline / train_tcnn)
FS = 200
WINDOW_LENGTH = 40
WINDOW_SHIFT = 10
THRESHOLD = 1e-5
NO_GESTURE_CROP_FALLBACK = int(1.3 * FS)  # 260 samples
CHANNELS = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
ALL_FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])

# Top-18 features from ranking (same order as feature_ranking_ranks.csv)
TOP_18_FEATURES = [
    "ch4_MFL", "ch4_MSR", "ch5_MFL", "ch4_RMS", "ch3_RMS", "ch4_DASDV",
    "ch4_IAV", "ch6_MFL", "ch7_MFL", "ch3_MFL", "ch4_LS", "ch3_MSR",
    "ch8_MFL", "ch2_RMS", "ch1_RMS", "ch2_MFL", "ch3_LS", "ch1_MFL",
]


# ═══════════════════════════════════════════════════════════════════════════════
# TDCNN Preprocessing Pipeline (from dataset_B: filter → segment → window → z-score)
# ═══════════════════════════════════════════════════════════════════════════════
def _filter_emg(emg_signal, fs=FS, lowcut=20, highcut=95, notch_freq=50, notch_q=30):
    """Bandpass (20-95 Hz) + 50 Hz notch filter."""
    nyq = fs / 2
    b_bp, a_bp = signal.butter(2, [lowcut / nyq, highcut / nyq], btype="band")
    filtered = signal.filtfilt(b_bp, a_bp, emg_signal)
    b_notch, a_notch = signal.iirnotch(w0=notch_freq, Q=notch_q, fs=fs)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)
    return filtered


def _segment_trial(filtered_channels, sample_meta, no_gesture_crop):
    """Crop to ground truth or center crop for noGesture."""
    gesture = sample_meta["gestureName"]
    if gesture != "noGesture" and "groundTruthIndex" in sample_meta:
        gti = sample_meta["groundTruthIndex"]
        start, end = gti[0] - 1, gti[1]
        return {ch: arr[start:end] for ch, arr in filtered_channels.items()}
    else:
        length = len(next(iter(filtered_channels.values())))
        centre = length // 2
        start = max(centre - no_gesture_crop // 2, 0)
        end = min(start + no_gesture_crop, length)
        start = max(end - no_gesture_crop, 0)
        return {ch: arr[start:end] for ch, arr in filtered_channels.items()}


def _extract_raw_windows(cropped_signals):
    """Slice segmented channels into overlapping windows (40 samples, 8 channels)."""
    sig_matrix = np.column_stack([cropped_signals[ch] for ch in CHANNELS])
    num_samples = sig_matrix.shape[0]
    windows = []
    for start in range(0, num_samples - WINDOW_LENGTH + 1, WINDOW_SHIFT):
        windows.append(sig_matrix[start : start + WINDOW_LENGTH, :])
    return windows


def process_dataset_B_to_tdcnn_windows(dataset_path):
    """
    Process dataset_B.pkl: filter → segment → window → per-user z-score.
    Returns (X, y) with X shape (N, 40, 8), y labels.
    """
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    all_X, all_y = [], []
    for user_id, user_data in dataset.items():
        samples = user_data.get("trainingSamples", user_data)
        gesture_lengths = [
            s["groundTruthIndex"][1] - s["groundTruthIndex"][0] + 1
            for s in samples.values()
            if s.get("gestureName") != "noGesture" and s.get("groundTruthIndex")
        ]
        median_gesture_len = (
            int(np.median(gesture_lengths)) if gesture_lengths else NO_GESTURE_CROP_FALLBACK
        )
        user_windows, user_labels = [], []
        for sample_key, sample in samples.items():
            emg = sample["emg"]
            gesture = sample["gestureName"]
            filtered = {
                ch: _filter_emg(np.array(emg[ch])) for ch in CHANNELS
            }
            cropped = _segment_trial(filtered, sample, median_gesture_len)
            windows = _extract_raw_windows(cropped)
            if windows:
                user_windows.extend(windows)
                user_labels.extend([gesture] * len(windows))
        if user_windows:
            user_windows = np.array(user_windows, dtype=np.float32)
            mu = np.mean(user_windows, axis=(0, 1))
            sig = np.std(user_windows, axis=(0, 1)) + 1e-8
            user_windows = (user_windows - mu) / sig
            all_X.append(user_windows)
            all_y.extend(user_labels)
    X = np.concatenate(all_X, axis=0) if all_X else np.array([])
    y = np.array(all_y)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# TD9 Feature Extraction (for preprocessing timing)
# ═══════════════════════════════════════════════════════════════════════════════
def _LS(x):
    n = len(x)
    if n < 2:
        return 0.0
    xs = np.sort(x)
    i = np.arange(1, n + 1)
    return np.sum((2 * i - n - 1) * xs) / (n * (n - 1))


def _MFL(x):
    return np.log(np.sqrt(np.sum(np.diff(x) ** 2)) + 1e-10)


def _MSR(x):
    return np.mean(np.sqrt(np.abs(x)))


def _WAMP(x, thr):
    return np.sum(np.abs(np.diff(x)) > thr)


def _ZC(x, thr):
    x1, x2 = x[:-1], x[1:]
    return np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) > thr))


def _RMS(x):
    return np.sqrt(np.mean(x ** 2))


def _IAV(x):
    return np.sum(np.abs(x))


def _DASDV(x):
    return np.sqrt(np.mean(np.diff(x) ** 2))


def _VAR(x):
    return np.var(x, ddof=1)


def extract_td9(window, thr=THRESHOLD):
    """Return 9-element feature array for one channel."""
    return np.array([
        _LS(window), _MFL(window), _MSR(window),
        _WAMP(window, thr), _ZC(window, thr),
        _RMS(window), _IAV(window), _DASDV(window), _VAR(window),
    ])


def raw_windows_to_18_features(raw_windows):
    """
    Convert raw EMG windows (N, 40, 8) to 18 features (N, 18).
    Mirrors preprocessing: TD9 extraction per channel, then select top-18.
    """
    n_samples = raw_windows.shape[0]
    features_72 = np.empty((n_samples, 72), dtype=np.float32)
    for i in range(n_samples):
        w = raw_windows[i]  # (40, 8)
        feat = np.concatenate([extract_td9(w[:, c]) for c in range(8)])
        features_72[i] = feat
    top_18_indices = [ALL_FEATURE_COLS.index(f) for f in TOP_18_FEATURES]
    return features_72[:, top_18_indices]


def process_dataset_B_to_features(dataset_path):
    """
    Full feature pipeline from dataset_B: filter→segment→window→TD9→z-score→select18.
    Same shared steps as TDCNN (filter, segment, window) plus TD9 and feature z-score.
    Returns (X_18, y) with X shape (N, 18), z-scored per user.
    """
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    top_18_indices = [ALL_FEATURE_COLS.index(f) for f in TOP_18_FEATURES]
    all_X, all_y, all_users = [], [], []
    for user_id, user_data in dataset.items():
        samples = user_data.get("trainingSamples", user_data)
        gesture_lengths = [
            s["groundTruthIndex"][1] - s["groundTruthIndex"][0] + 1
            for s in samples.values()
            if s.get("gestureName") != "noGesture" and s.get("groundTruthIndex")
        ]
        median_gesture_len = (
            int(np.median(gesture_lengths)) if gesture_lengths else NO_GESTURE_CROP_FALLBACK
        )
        user_rows = []  # list of (feat_72, label)
        for sample_key, sample in samples.items():
            emg = sample["emg"]
            gesture = sample["gestureName"]
            filtered = {ch: _filter_emg(np.array(emg[ch])) for ch in CHANNELS}
            cropped = _segment_trial(filtered, sample, median_gesture_len)
            n_samples = len(next(iter(cropped.values())))
            for start in range(0, n_samples - WINDOW_LENGTH + 1, WINDOW_SHIFT):
                end = start + WINDOW_LENGTH
                feat_72 = np.concatenate([
                    extract_td9(cropped[ch][start:end]) for ch in CHANNELS
                ])
                user_rows.append((feat_72, gesture))
        if user_rows:
            feat_72_arr = np.array([r[0] for r in user_rows], dtype=np.float32)
            labels = [r[1] for r in user_rows]
            mu = np.mean(feat_72_arr, axis=0)
            sig = np.std(feat_72_arr, axis=0) + 1e-8
            feat_72_arr = (feat_72_arr - mu) / sig
            feat_18 = feat_72_arr[:, top_18_indices]
            all_X.append(feat_18)
            all_y.extend(labels)
    X = np.concatenate(all_X, axis=0) if all_X else np.array([])
    y = np.array(all_y)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# Load Test Data
# ═══════════════════════════════════════════════════════════════════════════════
def load_raw_windows_from_npz():
    """Load raw (filtered, windowed, z-scored) windows from npz."""
    if not NPZ_TEST_PATH.exists():
        return None, None
    data = np.load(NPZ_TEST_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    # Ensure shape (N, 40, 8) for raw windows
    if X.ndim == 3 and X.shape[2] == 8:
        pass  # (N, 40, 8)
    elif X.ndim == 3 and X.shape[1] == 8:
        X = np.transpose(X, (0, 2, 1))  # (N, L, 8) -> (N, 40, 8)
    return X.astype(np.float32), y


def load_features_from_parquet():
    """Load pre-extracted 18 features from parquet."""
    if not PARQUET_18_PATH.exists():
        return None, None
    try:
        df = pd.read_parquet(PARQUET_18_PATH)
    except ImportError as e:
        print(f"Warning: Cannot read parquet (install pyarrow: pip install pyarrow): {e}")
        return None, None
    # Use TOP_18 order; ensure we have exactly 18 feature columns
    feat_cols = [c for c in TOP_18_FEATURES if c in df.columns]
    if len(feat_cols) < 18:
        extra = [c for c in df.columns if c in ALL_FEATURE_COLS and c not in feat_cols]
        feat_cols = (feat_cols + extra)[:18]
    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].values if "label" in df.columns else None
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# Load Models
# ═══════════════════════════════════════════════════════════════════════════════
def load_knn(path=None):
    import joblib
    from train_knn import FaissKNNClassifierGPU
    p = path or KNN_PATH
    data = joblib.load(p)
    clf = FaissKNNClassifierGPU(
        n_neighbors=data["params"]["n_neighbors"],
        metric=data["params"]["metric"],
    )
    clf.fit(data["X_store"], data["y_store"])
    return clf


def load_tdcnn():
    from tdcnn_eca import TDCNNClassifier
    return TDCNNClassifier.load(str(TDCNN_PATH))


def load_xgboost():
    import xgboost as xgb
    clf = xgb.XGBClassifier()
    clf.load_model(str(XGB_PATH))
    return clf


def load_svm():
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import LabelEncoder
    from svm_val import RFFSVMClassifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(SVM_PATH, map_location=device)
    params = ckpt.get("params", {})
    clf = RFFSVMClassifier(**params)
    W = ckpt["W"].to(device)
    b = ckpt["b"].to(device)
    n_cls = len(ALL_LABELS)
    linear = nn.Linear(W.shape[1], n_cls).to(device)
    linear.load_state_dict(ckpt["model_state_dict"])
    linear.eval()
    clf.model_ = (W, b, linear)
    clf.le_ = LabelEncoder()
    clf.le_.fit(ALL_LABELS)
    clf.classes_ = clf.le_.classes_
    return clf


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _descriptive_stats(times_ms):
    """Compute mean, std, 95%% CI from a list of times (ms)."""
    arr = np.array(times_ms, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return {"mean": float(arr[0]) if n else 0.0, "std": 0.0, "ci95_low": arr[0], "ci95_high": arr[0]}
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    t_val = stats.t.ppf(0.975, df=n - 1)
    sem = std / np.sqrt(n)
    return {"mean": mean, "std": std, "ci95_low": mean - t_val * sem, "ci95_high": mean + t_val * sem}


def time_preprocess(raw_windows, n_reps=3):
    """Time TD9 feature extraction (preprocessing for feature-based models)."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = raw_windows_to_18_features(raw_windows)
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return np.median(times), times


def time_inference(model_name, model, X, n_reps=5, warmup=2):
    """Time model inference. Returns (median_ms, list_of_times_ms) for statistics."""
    for _ in range(warmup):
        model.predict(X)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model.predict(X)
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return np.median(times), times


def run_benchmark(repetitions=5, warmup=2, max_samples=20000):
    results = {}

    print("=" * 70)
    print("  EMG-EPN612 — Inference Speed Benchmark (18 features)")
    print("=" * 70)
    print("\nNote: All models start from dataset_B.pkl when available.")
    print("      TDCNN: filter→segment→window→z-score (no TD9).")
    print("      KNN/XGBoost/SVM: filter→segment→window→TD9→z-score (shared + TD9).\n")

    # Load test data
    raw_X, raw_y = load_raw_windows_from_npz()
    feat_X, feat_y = load_features_from_parquet()

    use_raw = raw_X is not None
    use_feat = feat_X is not None

    if not use_raw and not use_feat and not DATASET_B_PATH.exists():
        print("ERROR: No test data found. Expected at least one of:")
        print(f"  - {DATASET_B_PATH}")
        print(f"  - {NPZ_TEST_PATH}  (run train_tcnn.py to create)")
        print(f"  - {PARQUET_18_PATH}")
        return None

    n_raw = len(raw_X) if use_raw else 0
    n_feat = len(feat_X) if use_feat else 0
    n_samples = max(n_raw, n_feat)
    print(f"Test samples: raw={n_raw or 'N/A'}, features={n_feat or 'N/A'}\n")

    # ─── TDCNN: full pipeline from dataset_B (filter→segment→window→z-score→inference) ─
    tdcnn_raw_X = None
    if TDCNN_PATH.exists() and DATASET_B_PATH.exists():
        print("[TDCNN] Processing dataset_B.pkl (filter→segment→window→z-score)...")
        prep_times = []
        for _ in range(1):  # 1 rep for preprocess (subsample speeds up inference)
            t0 = time.perf_counter()
            tdcnn_raw_X, _ = process_dataset_B_to_tdcnn_windows(DATASET_B_PATH)
            prep_times.append((time.perf_counter() - t0) * 1000)
        tdcnn_preprocess_ms = np.median(prep_times)
        n_tdcnn = len(tdcnn_raw_X)
        # Subsample for faster inference timing (extrapolate per-window from subset)
        tdcnn_X_timing = tdcnn_raw_X
        if max_samples and n_tdcnn > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_tdcnn, size=max_samples, replace=False)
            tdcnn_X_timing = tdcnn_raw_X[idx]
            n_tdcnn_timing = max_samples
        else:
            n_tdcnn_timing = n_tdcnn
        if n_tdcnn > 0:
            print(f"       Preprocess: {tdcnn_preprocess_ms:.1f} ms for {n_tdcnn:,} windows")
            if n_tdcnn_timing < n_tdcnn:
                print(f"       Inference timing on {n_tdcnn_timing:,} samples (subsampled for speed)")
            print(f"       Per window: {tdcnn_preprocess_ms*1000/n_tdcnn:.2f} µs")
            print("[TDCNN] Loading model and timing inference...")
            model = load_tdcnn()
            t_inf, inf_times = time_inference("TDCNN", model, tdcnn_X_timing, n_reps=repetitions, warmup=warmup)
            total_ms = tdcnn_preprocess_ms + t_inf
            prep_us = tdcnn_preprocess_ms * 1000 / n_tdcnn_timing
            inf_us = t_inf * 1000 / n_tdcnn_timing
            results["TDCNN"] = {
                "preprocess_ms": tdcnn_preprocess_ms,
                "inference_ms": t_inf,
                "total_ms": total_ms,
                "preprocess_per_window_ms": prep_us / 1000,
                "inference_per_window_ms": inf_us / 1000,
                "total_per_window_ms": (prep_us + inf_us) / 1000,
                "preprocess_per_sample_us": prep_us,
                "inference_per_sample_us": inf_us,
                "samples_per_sec": n_tdcnn_timing / (total_ms / 1000),
                "n_windows": n_tdcnn_timing,
                "inference_times_ms": inf_times,
                "preprocess_times_ms": prep_times,
            }
            print(f"       Inference: {t_inf:.1f} ms")
            print(f"       Total (preprocess+inference): {total_ms:.1f} ms, {n_tdcnn/(total_ms/1000):,.0f} samples/sec")
    elif TDCNN_PATH.exists() and use_raw:
        # Fallback: use npz (no preprocessing timing)
        print("[TDCNN] Using dataset_TESTING.npz (inference only; no dataset_B for preprocess timing)")
        tdcnn_X = raw_X
        if max_samples and len(raw_X) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(raw_X), size=max_samples, replace=False)
            tdcnn_X = raw_X[idx]
            print(f"       Subsampled to {max_samples:,} for inference timing")
        model = load_tdcnn()
        t_inf, inf_times = time_inference("TDCNN", model, tdcnn_X, n_reps=repetitions, warmup=warmup)
        n_timing = len(tdcnn_X)
        inf_us = t_inf * 1000 / n_timing
        results["TDCNN"] = {
            "preprocess_ms": 0.0,
            "inference_ms": t_inf,
            "total_ms": t_inf,
            "preprocess_per_window_ms": 0.0,
            "inference_per_window_ms": inf_us / 1000,
            "total_per_window_ms": inf_us / 1000,
            "preprocess_per_sample_us": 0.0,
            "inference_per_sample_us": inf_us,
            "samples_per_sec": n_timing / (t_inf / 1000),
            "n_windows": n_timing,
            "inference_times_ms": inf_times,
            "preprocess_times_ms": [],
        }
        print(f"       Inference: {t_inf:.1f} ms, {n_timing/(t_inf/1000):,.0f} samples/sec")
    else:
        if not TDCNN_PATH.exists():
            print(f"[TDCNN] Skipped: model not found at {TDCNN_PATH}")
        elif not DATASET_B_PATH.exists():
            print(f"[TDCNN] Skipped: dataset_B.pkl not found (needed for preprocess timing)")
        else:
            print("[TDCNN] Skipped: no raw data")

    # ─── Feature models: full pipeline from dataset_B (filter→segment→window→TD9→z-score) ───
    feat_preprocess_ms = 0.0
    feat_preprocess_times = []
    feat_X_in = None  # Input for KNN/XGB/SVM inference
    n_feat_in = 0
    if DATASET_B_PATH.exists():
        print("\n[Feature models] Processing dataset_B.pkl (filter→segment→window→TD9→z-score)...")
        prep_times = []
        for _ in range(1):  # 1 rep for preprocess
            t0 = time.perf_counter()
            feat_X_in, _ = process_dataset_B_to_features(DATASET_B_PATH)
            prep_times.append((time.perf_counter() - t0) * 1000)
        feat_preprocess_ms = np.median(prep_times)
        feat_preprocess_times = prep_times
        n_feat_full = len(feat_X_in)
        if max_samples and n_feat_full > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_feat_full, size=max_samples, replace=False)
            feat_X_in = feat_X_in[idx]
            n_feat_in = max_samples
            print(f"       Full preprocess: {feat_preprocess_ms:.1f} ms for {n_feat_full:,} windows")
            print(f"       Subsampled to {n_feat_in:,} for inference timing")
        else:
            n_feat_in = n_feat_full
        if n_feat_in > 0:
            print(f"       Per window: {feat_preprocess_ms*1000/n_feat_full:.2f} µs")
    elif use_raw:
        # Fallback: TD9 only on npz windows (shared filter/segment/window not timed)
        print("\n[Feature models] No dataset_B; timing TD9 extraction on npz windows...")
        _, prep_times = time_preprocess(raw_X, n_reps=3)
        feat_preprocess_ms = np.median(prep_times)
        feat_preprocess_times = prep_times
        feat_X_in = raw_windows_to_18_features(raw_X)
        n_feat_full = len(feat_X_in)
        if max_samples and n_feat_full > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_feat_full, size=max_samples, replace=False)
            feat_X_in = feat_X_in[idx]
            n_feat_in = max_samples
        else:
            n_feat_in = n_feat_full
        print(f"       TD9 extraction only: {feat_preprocess_ms:.1f} ms (filter/segment/window not included)")
    elif use_feat:
        # No dataset_B, no npz: use parquet, estimate TD9 time
        print("\n[Feature models] Using parquet (inference only; no dataset_B for preprocess timing)")
        feat_X_in = feat_X
        n_feat_in = len(feat_X_in)
        synthetic = np.random.randn(min(5000, n_feat_in), 40, 8).astype(np.float32)
        _, prep_times = time_preprocess(synthetic, n_reps=3)
        feat_preprocess_ms = np.median(prep_times) * (n_feat_in / len(synthetic))
        feat_preprocess_times = []  # Not comparable across dataset sizes
        print(f"       Estimated TD9+shared: ~{feat_preprocess_ms:.1f} ms (no dataset_B)")

    # ─── KNN ────────────────────────────────────────────────────────────────
    knn_path = KNN_PATH
    if not knn_path.exists():
        knn_files = list(MODELS_DIR.glob("knn*_faiss_gpu_enn_*.joblib"))
        knn_path = max(knn_files, key=lambda p: p.stat().st_mtime) if knn_files else None
    if knn_path and knn_path.exists() and feat_X_in is not None and n_feat_in > 0:
        print("\n[KNN] Loading model...")
        model = load_knn(path=knn_path)
        t_ms, inf_times = time_inference("KNN", model, feat_X_in, n_reps=repetitions, warmup=warmup)
        total_ms = feat_preprocess_ms + t_ms
        prep_us = feat_preprocess_ms * 1000 / n_feat_in
        inf_us = t_ms * 1000 / n_feat_in
        results["KNN"] = {
            "preprocess_ms": feat_preprocess_ms,
            "inference_ms": t_ms,
            "total_ms": total_ms,
            "preprocess_per_window_ms": prep_us / 1000,
            "inference_per_window_ms": inf_us / 1000,
            "total_per_window_ms": (prep_us + inf_us) / 1000,
            "preprocess_per_sample_us": prep_us,
            "inference_per_sample_us": inf_us,
            "samples_per_sec": n_feat_in / (total_ms / 1000),
            "n_windows": n_feat_in,
            "inference_times_ms": inf_times,
            "preprocess_times_ms": feat_preprocess_times,
        }
        print(f"       Preprocess: {feat_preprocess_ms:.1f} ms | Inference: {t_ms:.1f} ms")
        print(f"       Total: {total_ms:.1f} ms ({total_ms*1000/n_feat_in:.2f} µs/sample)")

    # ─── XGBoost ────────────────────────────────────────────────────────────
    if XGB_PATH.exists() and feat_X_in is not None and n_feat_in > 0:
        print("\n[XGBoost] Loading model...")
        model = load_xgboost()
        t_ms, inf_times = time_inference("XGBoost", model, feat_X_in, n_reps=repetitions, warmup=warmup)
        total_ms = feat_preprocess_ms + t_ms
        prep_us = feat_preprocess_ms * 1000 / n_feat_in
        inf_us = t_ms * 1000 / n_feat_in
        results["XGBoost"] = {
            "preprocess_ms": feat_preprocess_ms,
            "inference_ms": t_ms,
            "total_ms": total_ms,
            "preprocess_per_window_ms": prep_us / 1000,
            "inference_per_window_ms": inf_us / 1000,
            "total_per_window_ms": (prep_us + inf_us) / 1000,
            "preprocess_per_sample_us": prep_us,
            "inference_per_sample_us": inf_us,
            "samples_per_sec": n_feat_in / (total_ms / 1000),
            "n_windows": n_feat_in,
            "inference_times_ms": inf_times,
            "preprocess_times_ms": feat_preprocess_times,
        }
        print(f"       Preprocess: {feat_preprocess_ms:.1f} ms | Inference: {t_ms:.1f} ms")
        print(f"       Total: {total_ms:.1f} ms ({total_ms*1000/n_feat_in:.2f} µs/sample)")

    # ─── SVM ────────────────────────────────────────────────────────────────
    if SVM_PATH.exists() and feat_X_in is not None and n_feat_in > 0:
        print("\n[SVM] Loading model...")
        model = load_svm()
        t_ms, inf_times = time_inference("SVM", model, feat_X_in, n_reps=repetitions, warmup=warmup)
        total_ms = feat_preprocess_ms + t_ms
        prep_us = feat_preprocess_ms * 1000 / n_feat_in
        inf_us = t_ms * 1000 / n_feat_in
        results["SVM"] = {
            "preprocess_ms": feat_preprocess_ms,
            "inference_ms": t_ms,
            "total_ms": total_ms,
            "preprocess_per_window_ms": prep_us / 1000,
            "inference_per_window_ms": inf_us / 1000,
            "total_per_window_ms": (prep_us + inf_us) / 1000,
            "preprocess_per_sample_us": prep_us,
            "inference_per_sample_us": inf_us,
            "samples_per_sec": n_feat_in / (total_ms / 1000),
            "n_windows": n_feat_in,
            "inference_times_ms": inf_times,
            "preprocess_times_ms": feat_preprocess_times,
        }
        print(f"       Preprocess: {feat_preprocess_ms:.1f} ms | Inference: {t_ms:.1f} ms")
        print(f"       Total: {total_ms:.1f} ms ({total_ms*1000/n_feat_in:.2f} µs/sample)")

    # ─── Statistical analysis ───────────────────────────────────────────────
    if results:
        _print_statistical_analysis(results)

    # ─── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  SUMMARY — Average ms per window (preprocess + inference)")
    print("=" * 85)
    for name, r in results.items():
        prep_pw = r.get("preprocess_per_window_ms", 0.0)
        inf_pw = r.get("inference_per_window_ms", 0.0)
        total_pw = r.get("total_per_window_ms", prep_pw + inf_pw)
        sps = r["samples_per_sec"]
        print(f"  {name:10} | Preprocess: {prep_pw:8.4f} ms/window | Inference: {inf_pw:8.4f} ms/window | "
              f"Total: {total_pw:8.4f} ms/window | {sps:,.0f} samples/sec")
    print("=" * 85)

    # Save results
    out_path = BENCHMARK_RESULTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON (inference_times_ms, etc.)
    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    # Pairwise p-values (raw + Bonferroni-corrected) for saved reference
    pairwise_pvalues = {}
    model_list = [m for m in results if results[m].get("inference_times_ms")]
    n_comp = len(model_list) * (len(model_list) - 1) // 2
    for i, a in enumerate(model_list):
        for b in model_list[i + 1 :]:
            ta = results[a].get("inference_times_ms", [])
            tb = results[b].get("inference_times_ms", [])
            if len(ta) >= 2 and len(tb) >= 2:
                _, p = stats.mannwhitneyu(ta, tb, alternative="two-sided")
                pairwise_pvalues[f"{a}_vs_{b}"] = {
                    "p_raw": float(p),
                    "p_bonferroni": float(min(p * n_comp, 1.0)),
                }
    save_data = {
        "n_samples": n_samples,
        "repetitions": repetitions,
        "warmup": warmup,
        "max_samples_used": max((k.get("n_windows", 0) for k in results.values()), default=0),
        "results": _to_serializable(results),
        "pairwise_pvalues": pairwise_pvalues,
    }
    out_path.write_text(json.dumps(save_data, indent=2))
    print(f"\nResults saved to {out_path}")
    _save_visualizations(results, MODELS_DIR)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def _print_statistical_analysis(results):
    """Print descriptive statistics and pairwise significance tests."""
    models = [m for m in results if results[m].get("inference_times_ms")]
    if len(models) < 2:
        return
    print("\n" + "=" * 85)
    print("  STATISTICAL ANALYSIS")
    print("=" * 85)
    # 1. Descriptive stats: mean ± 95%% CI for total ms per window
    print("\n  Descriptive statistics (total ms per window, mean [95%% CI]):")
    print("  " + "-" * 60)
    stats_per_model = {}
    for name in models:
        r = results[name]
        nw = r.get("n_windows", 1)
        inf_times = r.get("inference_times_ms", [])
        prep_times = r.get("preprocess_times_ms", [])
        if inf_times:
            inf_per_win = [t / nw for t in inf_times]  # ms per window per rep
            prep_mean = np.mean(prep_times) / nw if prep_times else 0.0
            inf_stats = _descriptive_stats(inf_per_win)  # ms per window
            total_per_win = [prep_mean + t for t in inf_per_win]
            total_stats = _descriptive_stats(total_per_win)
            stats_per_model[name] = {"inference": inf_stats, "total": total_stats}
            ci_str = f"[{total_stats['ci95_low']:.4f}, {total_stats['ci95_high']:.4f}]"
            print(f"    {name:10}  mean={total_stats['mean']:.4f} ms  std={total_stats['std']:.4f}  95%% CI {ci_str}")
    # 2. Pairwise tests (Mann-Whitney U) with Bonferroni correction
    model_list = list(models)
    n_comparisons = len(model_list) * (len(model_list) - 1) // 2
    alpha_bonf = 0.05 / n_comparisons if n_comparisons else 0.05
    print("\n  Pairwise significance tests (Mann-Whitney U, Bonferroni α=0.05/{}={:.4f}):".format(
        n_comparisons, alpha_bonf))
    print("  " + "-" * 60)
    for i, a in enumerate(model_list):
        for b in model_list[i + 1 :]:
            times_a = results[a].get("inference_times_ms", [])
            times_b = results[b].get("inference_times_ms", [])
            if len(times_a) >= 2 and len(times_b) >= 2:
                stat, p = stats.mannwhitneyu(times_a, times_b, alternative="two-sided")
                p_bonf = min(p * n_comparisons, 1.0)  # Bonferroni-corrected p
                sig = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else "n.s."
                print(f"    {a} vs {b:10}  p={p:.4f}  p_bonf={p_bonf:.4f}  {sig}")
    print("=" * 85)


def _save_visualizations(results, out_dir):
    """Save bar chart and pairwise p-value heatmap to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(results.keys())
    if not models:
        return
    # 1. Bar chart: mean total ms per window with error bars
    means = []
    err_lo = []
    err_hi = []
    for name in models:
        r = results[name]
        nw = r.get("n_windows", 1)
        inf_times = r.get("inference_times_ms", [])
        prep_times = r.get("preprocess_times_ms", [])
        if inf_times and nw:
            inf_per_win = [t / nw for t in inf_times]
            prep_mean = np.mean(prep_times) / nw if prep_times else 0.0
            total_per_win = [prep_mean + t for t in inf_per_win]
            s = _descriptive_stats(total_per_win)
            means.append(s["mean"])
            err_lo.append(s["mean"] - s["ci95_low"])
            err_hi.append(s["ci95_high"] - s["mean"])
        else:
            total_pw = r.get("total_per_window_ms", 0.0)
            means.append(total_pw)
            err_lo.append(0.0)
            err_hi.append(0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=[err_lo, err_hi], capsize=5, color="steelblue", edgecolor="navy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mean total time (ms/window)")
    ax.set_title("Inference Speed Comparison (preprocess + inference)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "benchmark_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    # 2. Heatmap of pairwise Bonferroni-corrected p-values
    models_with_times = [m for m in models if results[m].get("inference_times_ms")]
    if len(models_with_times) >= 2:
        n = len(models_with_times)
        n_comp = n * (n - 1) // 2
        pmat = np.ones((n, n))
        for i, a in enumerate(models_with_times):
            for j, b in enumerate(models_with_times):
                if i == j:
                    pmat[i, j] = 0.0
                elif i < j:
                    ta = results[a]["inference_times_ms"]
                    tb = results[b]["inference_times_ms"]
                    if len(ta) >= 2 and len(tb) >= 2:
                        _, p = stats.mannwhitneyu(ta, tb, alternative="two-sided")
                        p_bonf = min(p * n_comp, 1.0)
                        pmat[i, j] = pmat[j, i] = p_bonf
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pmat, cmap="RdYlGn_r", vmin=0, vmax=0.1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(models_with_times)
        ax.set_yticklabels(models_with_times)
        for i in range(n):
            for j in range(n):
                if i != j:
                    txt = f"{pmat[i,j]:.3f}" if pmat[i,j] < 0.1 else "n.s."
                    ax.text(j, i, txt, ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, label="p (Bonferroni-corrected)")
        n_comp = n * (n - 1) // 2
        alpha_bonf = 0.05 / n_comp if n_comp else 0.05
        ax.set_title("Pairwise p-values (Bonferroni α={:.4f})".format(alpha_bonf))
        plt.tight_layout()
        fig.savefig(out_dir / "benchmark_pvalues_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"\nVisualizations saved to {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Load saved results (no re-run)
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_print_saved_results():
    """Load existing benchmark results and print summary. Skips all model loading and inference."""
    if not BENCHMARK_RESULTS_PATH.exists():
        print(f"No saved results found at {BENCHMARK_RESULTS_PATH}")
        print("Run the benchmark first (without --load-only).")
        return None
    data = json.loads(BENCHMARK_RESULTS_PATH.read_text())
    results = data.get("results", {})
    n_samples = data.get("n_samples", 0)
    print("=" * 85)
    print("  BENCHMARK RESULTS (loaded from cache — no re-run)")
    print("=" * 85)
    print(f"  n_samples: {n_samples:,} | repetitions: {data.get('repetitions')} | warmup: {data.get('warmup')}")
    print("-" * 85)
    print("  Average ms per window (preprocess + inference)")
    print("-" * 85)
    for name, r in results.items():
        nw = r.get("n_windows") or n_samples or 1
        if "preprocess_per_window_ms" in r and "inference_per_window_ms" in r:
            prep_pw = r["preprocess_per_window_ms"]
            inf_pw = r["inference_per_window_ms"]
            total_pw = r.get("total_per_window_ms", prep_pw + inf_pw)
        else:
            # Compute from totals: avg ms/window = total_ms / n_windows
            prep_pw = (r.get("preprocess_ms", 0) / nw) if nw else 0.0
            inf_pw = (r.get("inference_ms", 0) / nw) if nw else 0.0
            total_pw = (r.get("total_ms", 0) / nw) if nw else 0.0
        sps = r.get("samples_per_sec", 0)
        if sps == 0 and r.get("total_ms"):
            sps = nw / (r["total_ms"] / 1000)
        print(f"  {name:10} | Preprocess: {prep_pw:8.4f} ms/window | Inference: {inf_pw:8.4f} ms/window | "
              f"Total: {total_pw:8.4f} ms/window | {sps:,.0f} samples/sec")
    print("=" * 85)
    # Statistical analysis and visualizations if raw times are available
    if any(r.get("inference_times_ms") for r in results.values()):
        _print_statistical_analysis(results)
        _save_visualizations(results, MODELS_DIR)
    print(f"\nLoaded from: {BENCHMARK_RESULTS_PATH}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark inference speed of 18f models")
    parser.add_argument("--repetitions", type=int, default=3, help="Inference timing repetitions")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Max samples for inference timing (subsample for speed). 0 = use all.",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load and display saved results without re-running the benchmark.",
    )
    args = parser.parse_args()
    if args.load_only:
        load_and_print_saved_results()
    else:
        run_benchmark(
            repetitions=args.repetitions,
            warmup=args.warmup,
            max_samples=args.max_samples if args.max_samples > 0 else None,
        )


if __name__ == "__main__":
    main()
