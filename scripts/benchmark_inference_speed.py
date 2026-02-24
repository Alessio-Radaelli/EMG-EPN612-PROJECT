"""
EMG-EPN612 — Refactored Inference Speed Benchmark (18-feature models)

FIXES APPLIED:
1. Separates Single-Sample Latency (Batch=1) vs High Throughput (Batch=N).
2. Isolates mathematical preprocessing (Z-scoring, TD9) from file I/O and offline filtering.
3. Fully vectorizes TD9 feature extraction for realistic deployment-level speed.
4. Adds proper GPU synchronization for PyTorch/Faiss to prevent asynchronous timing bugs.
5. Generates synthetic fallback data if real data is missing to ensure the script always runs.

Requirements: torch, xgboost, joblib, faiss-gpu (for KNN)
Usage:
    cd "EMG-EPN612 project"
    python scripts/benchmark_inference_speed.py
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

PREPROC_DIR = PROJECT_ROOT / "preprocessed_output"
MODELS_DIR = PROJECT_ROOT / "models" / "18f"
NPZ_TEST_PATH = PREPROC_DIR / "dataset_TESTING.npz"
BENCHMARK_RESULTS_PATH = MODELS_DIR / "benchmark_inference_results.json"

# Model paths (TDCNN lives in models/ root; others in models/18f/)
KNN_PATH = MODELS_DIR / "knn_faiss_gpu_enn_manhattan_k1_wuniform.joblib"
TDCNN_PATH = PROJECT_ROOT / "models" / "tdcnn_emg_model.pth"
XGB_PATH = MODELS_DIR / "xgboost18_best_halving.json"
SVM_PATH = MODELS_DIR / "svm_val_best18.pt"

# Constants
CHANNELS = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
ALL_FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
TOP_18_FEATURES = [
    "ch4_MFL", "ch4_MSR", "ch5_MFL", "ch4_RMS", "ch3_RMS", "ch4_DASDV",
    "ch4_IAV", "ch6_MFL", "ch7_MFL", "ch3_MFL", "ch4_LS", "ch3_MSR",
    "ch8_MFL", "ch2_RMS", "ch1_RMS", "ch2_MFL", "ch3_LS", "ch1_MFL",
]
TOP_18_IDX = [ALL_FEATURE_COLS.index(f) for f in TOP_18_FEATURES]

# Dummy standardization constants (simulating pre-computed deployment values)
TDCNN_MU = np.zeros((40, 8), dtype=np.float32)
TDCNN_SIG = np.ones((40, 8), dtype=np.float32)
FEAT_MU = np.zeros(72, dtype=np.float32)
FEAT_SIG = np.ones(72, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Refactored Preprocessing (Isolated Memory/Math Operations)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_td9_vectorized(X_batch, thr=1e-5):
    """
    Highly optimized, fully vectorized TD9 feature extraction.
    Takes input shape (Batch, Time=40, Channels=8) and returns (Batch, 72).
    """
    B, n, _ = X_batch.shape

    # LS
    xs = np.sort(X_batch, axis=1)
    i = np.arange(1, n + 1).reshape(1, n, 1)
    ls = np.sum((2 * i - n - 1) * xs, axis=1) / (n * (n - 1))

    # MFL
    mfl = np.log(np.sqrt(np.sum(np.diff(X_batch, axis=1)**2, axis=1)) + 1e-10)

    # MSR
    msr = np.mean(np.sqrt(np.abs(X_batch)), axis=1)

    # WAMP
    wamp = np.sum(np.abs(np.diff(X_batch, axis=1)) > thr, axis=1)

    # ZC
    x1, x2 = X_batch[:, :-1, :], X_batch[:, 1:, :]
    zc = np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) > thr), axis=1)

    # RMS
    rms = np.sqrt(np.mean(X_batch**2, axis=1))

    # IAV
    iav = np.sum(np.abs(X_batch), axis=1)

    # DASDV
    dasdv = np.sqrt(np.mean(np.diff(X_batch, axis=1)**2, axis=1))

    # VAR
    var = np.var(X_batch, axis=1, ddof=1)

    # Stack into (B, 8 channels, 9 features)
    stacked = np.stack([ls, mfl, msr, wamp, zc, rms, iav, dasdv, var], axis=2)
    
    # Reshape to (B, 72) preserving the exact feature order of the original code
    return stacked.reshape(B, 72)


def preprocess_tdcnn(X_raw):
    """Pipeline for TDCNN: just Z-score standardisation on raw windows."""
    return (X_raw - TDCNN_MU) / TDCNN_SIG


def preprocess_features(X_raw):
    """Pipeline for Feature Models: TD9 -> Z-score -> Select Top 18."""
    feats_72 = extract_td9_vectorized(X_raw)
    feats_72_norm = (feats_72 - FEAT_MU) / FEAT_SIG
    return feats_72_norm[:, TOP_18_IDX]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers & Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _sync():
    """Ensure GPU completes tasks before stopping the timer."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def load_data(max_samples=20000):
    """Loads raw test windows, or creates synthetic data if unavailable."""
    if NPZ_TEST_PATH.exists():
        print(f"Loading real test data from {NPZ_TEST_PATH.name}...")
        data = np.load(NPZ_TEST_PATH, allow_pickle=True)
        X = data["X"]
        if X.ndim == 3 and X.shape[1] == 8:
            X = np.transpose(X, (0, 2, 1))  # Fix shape if (N, 8, 40)
        X = X[:max_samples].astype(np.float32)
    else:
        print("Test data missing. Generating synthetic data for benchmark...")
        X = np.random.randn(max_samples, 40, 8).astype(np.float32)
    return X


def load_knn():
    import joblib
    from train_knn import FaissKNNClassifierGPU
    path = KNN_PATH
    if not path.exists():
        paths = list(MODELS_DIR.glob("knn*_faiss_gpu_enn_*.joblib"))
        if not paths: raise FileNotFoundError("KNN model not found")
        path = max(paths, key=lambda p: p.stat().st_mtime)
    data = joblib.load(path)
    clf = FaissKNNClassifierGPU(
        n_neighbors=data["params"]["n_neighbors"],
        metric=data["params"]["metric"],
    )
    clf.fit(data["X_store"], data["y_store"])
    return clf


def load_tdcnn():
    from tdcnn_eca import TDCNNClassifier
    path = TDCNN_PATH
    if not path.exists():
        path = MODELS_DIR / "tdcnn_emg_model.pth"  # fallback: 18f subdir
    if not path.exists():
        raise FileNotFoundError("TDCNN model not found")
    return TDCNNClassifier.load(str(path))


def load_xgboost():
    import xgboost as xgb
    if not XGB_PATH.exists(): raise FileNotFoundError("XGBoost model not found")
    clf = xgb.XGBClassifier()
    clf.load_model(str(XGB_PATH))
    return clf


def load_svm():
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import LabelEncoder
    from train_svm import RFFSVMClassifier
    if not SVM_PATH.exists(): raise FileNotFoundError("SVM model not found")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(SVM_PATH, map_location=device)
    clf = RFFSVMClassifier(**ckpt.get("params", {}))
    
    W = ckpt["W"].to(device)
    b = ckpt["b"].to(device)
    linear = nn.Linear(W.shape[1], 6).to(device) # 6 classes
    linear.load_state_dict(ckpt["model_state_dict"])
    linear.eval()
    
    clf.model_ = (W, b, linear)
    clf.le_ = LabelEncoder()
    clf.le_.fit(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
    clf.classes_ = clf.le_.classes_
    return clf


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Engine
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_pipeline(model, X_raw, prep_fn, n_reps):
    """Times preprocessing and inference separately."""
    # 1. Warmup & verify
    try:
        X_prep = prep_fn(X_raw)
        model.predict(X_prep)
    except Exception as e:
        print(f"      [!] Model failed during warmup: {e}")
        return None

    # 2. Time Preprocessing (Math & Normalization only)
    prep_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = prep_fn(X_raw)
        prep_times.append((time.perf_counter() - t0) * 1000)

    # 3. Time Inference
    inf_times = []
    for _ in range(n_reps):
        _sync()
        t0 = time.perf_counter()
        _ = model.predict(X_prep)
        _sync()
        inf_times.append((time.perf_counter() - t0) * 1000)

    return {
        "prep_ms": np.median(prep_times),
        "inf_ms": np.median(inf_times),
        "total_ms": np.median(prep_times) + np.median(inf_times),
        "raw_total_times": [p + i for p, i in zip(prep_times, inf_times)]
    }


def run_benchmark():
    print("=" * 80)
    print("  EMG-EPN612 — Real-time Latency & Throughput Benchmark")
    print("=" * 80)
    
    X_test = load_data(max_samples=10000)
    B = len(X_test)
    
    models_config = {
        "TDCNN":   (load_tdcnn,   preprocess_tdcnn),
        "KNN":     (load_knn,     preprocess_features),
        "XGBoost": (load_xgboost, preprocess_features),
        "SVM":     (load_svm,     preprocess_features)
    }

    results = {}

    for name, (loader_fn, prep_fn) in models_config.items():
        print(f"\n[Benchmarking {name}]")
        try:
            model = loader_fn()
        except Exception as e:
            print(f"  -> Skipped: {e}")
            continue
            
        # 1. Latency: Batch Size = 1 (Critical for Real-time Control)
        print("  -> Measuring Latency (Batch = 1, Replications = 500)")
        X_single = X_test[0:1]
        lat_res = benchmark_pipeline(model, X_single, prep_fn, n_reps=500)

        # 2. Throughput: Batch Size = N
        print(f"  -> Measuring Throughput (Batch = {B:,}, Replications = 10)")
        thr_res = benchmark_pipeline(model, X_test, prep_fn, n_reps=10)

        if lat_res and thr_res:
            results[name] = {
                "latency_ms": lat_res,
                "throughput_ms": thr_res,
                "samples_per_sec": B / (thr_res["total_ms"] / 1000)
            }
            print(f"     Latency: {lat_res['total_ms']:.3f} ms / window")
            print(f"     Throughput: {results[name]['samples_per_sec']:,.0f} windows / sec")

    # Display & Save Results
    _print_summary(results)
    _print_statistical_analysis(results)
    _save_visualizations(results, MODELS_DIR)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {BENCHMARK_RESULTS_PATH}")


def _print_summary(results):
    print("\n" + "=" * 80)
    print("  SUMMARY: LATENCY VS THROUGHPUT")
    print("=" * 80)
    print(f"  {'Model':<10} | {'Latency (Prep + Inf) [ms]':<25} | {'Throughput [samples/sec]':<20}")
    print("-" * 80)
    for name, r in results.items():
        lat = r["latency_ms"]
        lat_str = f"{lat['prep_ms']:.2f} + {lat['inf_ms']:.2f} = {lat['total_ms']:.2f} ms"
        print(f"  {name:<10} | {lat_str:<25} | {r['samples_per_sec']:,.0f}")
    print("=" * 80)


def _print_statistical_analysis(results):
    """Performs non-parametric statistical tests on Latency execution times."""
    print("\n" + "=" * 80)
    print("  STATISTICAL ANALYSIS: PAIRWISE LATENCY DIFFERENCES")
    print("=" * 80)
    
    models = list(results.keys())
    if len(models) < 2:
        return
        
    n_comp = len(models) * (len(models) - 1) // 2
    alpha_bonf = 0.05 / n_comp
    
    print(f"  Test: Mann-Whitney U (Non-parametric)")
    print(f"  Correction: Bonferroni (α = {alpha_bonf:.4f} for {n_comp} comparisons)\n")
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            t1 = results[m1]["latency_ms"]["raw_total_times"]
            t2 = results[m2]["latency_ms"]["raw_total_times"]
            
            # Two-sided Mann-Whitney U test
            stat, p_raw = stats.mannwhitneyu(t1, t2, alternative="two-sided")
            p_adj = min(p_raw * n_comp, 1.0)
            
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
            diff_ms = np.median(t1) - np.median(t2)
            faster = m2 if diff_ms > 0 else m1
            
            print(f"  {m1} vs {m2:10} | p_adj = {p_adj:.4e} {sig:<4} | {faster} is faster by {abs(diff_ms):.3f} ms")
    print("=" * 80)


def _save_visualizations(results, out_dir):
    """Creates clear comparative bar charts for the benchmark."""
    if not results: return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(results.keys())

    # Plot 1: Single-Window Latency Breakdown
    prep_lats = [results[m]["latency_ms"]["prep_ms"] for m in models]
    inf_lats = [results[m]["latency_ms"]["inf_ms"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    ax.bar(x, prep_lats, label="Preprocessing (TD9, Z-score)", color="#f4a261")
    ax.bar(x, inf_lats, bottom=prep_lats, label="Inference", color="#2a9d8f")
    
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Latency Time (ms)")
    ax.set_title("Real-Time Latency per Window (Batch Size = 1)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_dir / "benchmark_latency.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Throughput
    sps = [results[m]["samples_per_sec"] for m in models]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, sps, color="#457b9d", edgecolor="navy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Samples per Second")
    ax.set_title("System Throughput (Large Batch Processing)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_dir / "benchmark_throughput.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_benchmark()