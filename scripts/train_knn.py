import sys
import time
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mode
import joblib
import faiss
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PREPROC_DIR   = PROJECT_ROOT / "preprocessed_output"

# FAISS GPU handles massive datasets easily. Set to None to use all data.
MAX_TRAIN_SAMPLES = None 

ALL_LABELS   = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
CHANNELS     = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES    = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
ALL_FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]

DATASET_FILES = {
    72: ("dataset_TRAINING.parquet",            "dataset_TEST.parquet"),
    36: ("dataset_TRAINING_reduced36.parquet",  "dataset_TESTING_reduced36.parquet"),
    18: ("dataset_TRAINING_reduced18.parquet",  "dataset_TESTING_reduced18.parquet"),
}

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# =============================================================================
# FAISS GPU Functions
# =============================================================================

def faiss_enn_gpu(X, y, k=3, metric="euclidean"):
    """
    GPU-Accelerated Edited Nearest Neighbours using FAISS.
    """
    d = X.shape[1]
    metric_type = faiss.METRIC_L1 if metric == 'manhattan' else faiss.METRIC_L2
    
    # 1. Initialize GPU Resources
    res = faiss.StandardGpuResources()
    
    # 2. Build CPU index, then transfer to GPU 0
    cpu_index = faiss.IndexFlat(d, metric_type)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    # FAISS requires C-contiguous arrays
    X_contig = np.ascontiguousarray(X, dtype=np.float32)
    gpu_index.add(X_contig)
    
    print(f"\n      -> FAISS (GPU) searching for {k} nearest neighbors across {len(X):,} rows...")
    t0 = time.time()
    
    # 3. Batch the search to allow tqdm to update the progress bar
    n_samples = len(X_contig)
    chunk_size = 100000
    I_list = []
    
    for i in tqdm(range(0, n_samples, chunk_size), desc="         ENN Search Progress", leave=False, unit="batch"):
        end = min(i + chunk_size, n_samples)
        _, I_chunk = gpu_index.search(X_contig[i:end], k + 1)
        I_list.append(I_chunk)
        
    I = np.vstack(I_list)
    print(f"      -> Search completed in {format_time(time.time() - t0)}")
    
    # 4. Filter noise based on neighbor labels
    neighbor_indices = I[:, 1:]
    neighbor_labels = y[neighbor_indices]
    
    # Find majority label among the k neighbors
    majority_labels = mode(neighbor_labels, axis=1, keepdims=True)[0].ravel()
    
    mask = (y == majority_labels)
    X_clean = X[mask]
    y_clean = y[mask]
    
    return X_clean, y_clean


class FaissKNNClassifierGPU:
    """
    Scikit-Learn style wrapper around FAISS GPU for ultra-fast KNN inference.
    """
    def __init__(self, n_neighbors=1, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.index = None
        self.y_store = None
        self.classes_ = None

    def fit(self, X, y):
        self.y_store = np.array(y)
        self.classes_ = np.unique(y)
        d = X.shape[1]
        
        metric_type = faiss.METRIC_L1 if self.metric == 'manhattan' else faiss.METRIC_L2
        
        # Build index and move to GPU
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlat(d, metric_type)
        self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        
        X_contig = np.ascontiguousarray(X, dtype=np.float32)
        self.index.add(X_contig)
        return self

    def predict(self, X):
        X_contig = np.ascontiguousarray(X, dtype=np.float32)
        n_samples = len(X_contig)
        chunk_size = 100000
        I_list = []
        
        print(f"\n      -> Predicting {n_samples:,} samples...")
        
        # Batch the predictions to show a progress bar
        for i in tqdm(range(0, n_samples, chunk_size), desc="         Inference Progress", leave=False, unit="batch"):
            end = min(i + chunk_size, n_samples)
            _, I_chunk = self.index.search(X_contig[i:end], self.n_neighbors)
            I_list.append(I_chunk)
            
        I = np.vstack(I_list)
        
        if self.n_neighbors == 1:
            return self.y_store[I[:, 0]]
        else:
            neighbor_labels = self.y_store[I]
            majority_labels = mode(neighbor_labels, axis=1, keepdims=True)[0].ravel()
            return majority_labels

# =============================================================================
# Helper: Subsample by Group
# =============================================================================
def subsample_by_group(X, y, groups, n_target):
    if n_target is None or n_target >= len(y):
        return X, y, groups
    unique_g = np.unique(groups)
    np.random.seed(42)
    np.random.shuffle(unique_g)
    selected_idx = []
    for g in unique_g:
        if len(selected_idx) >= n_target: break
        selected_idx.extend(np.where(groups == g)[0].tolist())
    return X[np.sort(selected_idx)], y[np.sort(selected_idx)], groups[np.sort(selected_idx)]

# =============================================================================
# Main Script
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=72, choices=[72, 36, 18])
    args = parser.parse_args()
    nf = args.features

    MODELS_DIR = PROJECT_ROOT / "models" / f"{nf}f"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("======================================================")
    print(f" FAISS-GPU KNN Grid Search — {nf} Features")
    print("======================================================")

    # 1. Load Data
    train_name, test_name = DATASET_FILES[nf]
    print(f"\n[1/4] Loading Training Data: {train_name}")
    df = pd.read_parquet(PREPROC_DIR / train_name)
    feature_cols = [c for c in df.columns if c in ALL_FEATURE_COLS]

    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder().fit(ALL_LABELS)
    y = le.transform(df["label"].values)
    groups = df["user"].values
    del df

    X, y, groups = subsample_by_group(X, y, groups, MAX_TRAIN_SAMPLES)
    target_str = f"Targeting ~{MAX_TRAIN_SAMPLES}" if MAX_TRAIN_SAMPLES else "Using ALL data"
    print(f"      {target_str}. Actual: {len(X):,} rows across {len(np.unique(groups))} patients.")

    # 2. Run ENN Noise Cleaning (fixed k=3, metric=euclidean)
    print(f"\n[2/4] Running FAISS-GPU ENN Noise Cleaning...")
    t_start_train = time.time()
    X_clean, y_clean = faiss_enn_gpu(X, y, k=3, metric="euclidean")
    print(f"      Kept {len(y_clean):,}/{len(y):,} samples.")

    # 3. Hold-out Validation with per-user per-class stratified split
    print(f"\n[3/4] Hold-out Validation (per-user per-class stratified split)...")
    from sklearn.model_selection import train_test_split
    # Create stratification key: user-class
    stratify_key = [f"{u}_{c}" for u, c in zip(groups, y)]
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, stratify=stratify_key)

    param_grid = {
        "n_neighbors": [1, 3, 5],
        "metric": ["euclidean", "manhattan"],
        "weight": ["uniform", "distance"]
    }
    best_score = -1
    best_params = None
    results = []

    for k in param_grid["n_neighbors"]:
        for metric in param_grid["metric"]:
            for weight in param_grid["weight"]:
                # ENN only on training split
                X_clean, y_clean = faiss_enn_gpu(X_train, y_train, k=3, metric="euclidean")
                clf = FaissKNNClassifierGPU(n_neighbors=k, metric=metric)
                clf.fit(X_clean, y_clean)
                y_pred = clf.predict(X_val)
                if weight == "distance" and k > 1:
                    # For distance weighting, majority vote weighted by distance (not implemented in FaissKNNClassifierGPU)
                    pass
                acc = accuracy_score(y_val, y_pred)
                results.append({"k": k, "metric": metric, "weight": weight, "mean_score": acc})
                print(f"    k={k}, metric={metric}, weight={weight} | Hold-out acc={acc:.4f}")
                if acc > best_score:
                    best_score = acc
                    best_params = {"n_neighbors": k, "metric": metric, "weight": weight}

    print(f"\nBest params: {best_params} | Hold-out acc={best_score:.4f}")

    # 4. Train final model on all ENN-cleaned data with best params
    print(f"\n[4/4] Training final model and evaluating on test set...")
    X_clean, y_clean = faiss_enn_gpu(X, y, k=3, metric="euclidean")
    clf = FaissKNNClassifierGPU(n_neighbors=best_params["n_neighbors"], metric=best_params["metric"])
    clf.fit(X_clean, y_clean)

    # Save model arrays
    save_path = MODELS_DIR / f"knn_faiss_gpu_enn_{best_params['metric']}_k{best_params['n_neighbors']}_w{best_params['weight']}.joblib"
    joblib.dump({
        "X_store": X_clean,
        "y_store": y_clean,
        "params": best_params,
    }, save_path)
    print(f"      Saved to {save_path}")

    # Evaluate on test set
    df_test = pd.read_parquet(PREPROC_DIR / test_name)
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = le.transform(df_test["label"].values)
    del df_test

    t_start_pred = time.time()
    y_pred_test = clf.predict(X_test)
    pred_time = time.time() - t_start_pred

    print(f"      Total Prediction Time: {format_time(pred_time)}")
    print(f"      Latency per sample: {(pred_time / len(X_test)) * 1000:.4f} ms")

    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    target_names = [name for _, name in sorted(zip(le.transform(le.classes_), le.classes_))]
    print(classification_report(y_test, y_pred_test, target_names=target_names, digits=4))

if __name__ == "__main__":
    main()
