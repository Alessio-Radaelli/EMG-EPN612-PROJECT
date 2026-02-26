"""
EMG-EPN612 — Full Test Set Evaluation with Per-Model Confusion Matrices

Evaluates TDCNN, KNN, XGBoost, and SVM on the complete test set:
  - TDCNN: preprocessed_output/dataset_TESTING.npz (raw windows)
  - KNN, XGBoost, SVM: preprocessed_output/dataset_TESTING_reduced18.parquet (18 features)

Generates one confusion matrix figure per model and saves metrics to JSON.

Requirements: torch, xgboost, joblib, faiss-gpu, pandas, pyarrow (or fastparquet)

Usage:
    cd "EMG-EPN612 project"
    python scripts/evaluate_confusion_matrices.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

PREPROC_DIR = PROJECT_ROOT / "preprocessed_output"
MODELS_DIR = PROJECT_ROOT / "models" / "18f"
NPZ_TEST_PATH = PREPROC_DIR / "dataset_TESTING.npz"
PARQUET_TEST_PATH = PREPROC_DIR / "dataset_TESTING_reduced18.parquet"
OUTPUT_DIR = MODELS_DIR / "confusion_matrices"
RESULTS_JSON = MODELS_DIR / "evaluation_confusion_results.json"

# Model paths
KNN_PATH = MODELS_DIR / "knn_faiss_gpu_enn_manhattan_k1_wuniform.joblib"
TDCNN_PATH = PROJECT_ROOT / "models" / "tdcnn_emg_model.pth"
TDCNN_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
XGB_PATH = MODELS_DIR / "xgboost18_best_halving.json"
SVM_PATH = MODELS_DIR / "svm_val_best18.pt"

# 18-feature column names (must match parquet)
TOP_18_FEATURES = [
    "ch4_MFL", "ch4_MSR", "ch5_MFL", "ch4_RMS", "ch3_RMS", "ch4_DASDV",
    "ch4_IAV", "ch6_MFL", "ch7_MFL", "ch3_MFL", "ch4_LS", "ch3_MSR",
    "ch8_MFL", "ch2_RMS", "ch1_RMS", "ch2_MFL", "ch3_LS", "ch1_MFL",
]
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_tdcnn_data():
    """Load TDCNN test data from npz: raw windows (N, 40, 8) and labels."""
    if not NPZ_TEST_PATH.exists():
        raise FileNotFoundError(f"TDCNN test data not found: {NPZ_TEST_PATH}")
    data = np.load(NPZ_TEST_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y_labels = np.array(data["y"], dtype=str)
    if X.ndim == 3 and X.shape[1] == 8:
        X = np.transpose(X, (0, 2, 1))  # (N, 8, 40) -> (N, 40, 8)
    return X, y_labels


def load_parquet_data():
    """Load 18-feature test data from parquet."""
    if not PARQUET_TEST_PATH.exists():
        raise FileNotFoundError(f"Parquet test data not found: {PARQUET_TEST_PATH}")
    df = pd.read_parquet(PARQUET_TEST_PATH)
    missing = [c for c in TOP_18_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    X = df[TOP_18_FEATURES].values.astype(np.float32)
    y_labels = df["label"].values
    return X, y_labels


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loaders (same pattern as benchmark_inference_speed.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_tdcnn():
    from tdcnn_eca import TDCNNClassifier
    path = TDCNN_PATH
    if not path.exists():
        path = MODELS_DIR / "tdcnn_emg_model.pth"
    if not path.exists():
        raise FileNotFoundError("TDCNN model not found")
    return TDCNNClassifier.load(str(path))


def load_knn():
    import joblib
    from train_knn import FaissKNNClassifierGPU
    path = KNN_PATH
    if not path.exists():
        paths = list(MODELS_DIR.glob("knn*_faiss_gpu_enn_*.joblib"))
        if not paths:
            raise FileNotFoundError("KNN model not found")
        path = max(paths, key=lambda p: p.stat().st_mtime)
    data = joblib.load(path)
    clf = FaissKNNClassifierGPU(
        n_neighbors=data["params"]["n_neighbors"],
        metric=data["params"]["metric"],
    )
    clf.fit(data["X_store"], data["y_store"])
    return clf


def load_xgboost():
    import xgboost as xgb
    if not XGB_PATH.exists():
        raise FileNotFoundError("XGBoost model not found")
    clf = xgb.XGBClassifier()
    clf.load_model(str(XGB_PATH))
    return clf


def load_svm():
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import LabelEncoder as SKLabelEncoder
    from train_svm import RFFSVMClassifier
    if not SVM_PATH.exists():
        raise FileNotFoundError("SVM model not found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(SVM_PATH, map_location=device)
    clf = RFFSVMClassifier(**ckpt.get("params", {}))
    W = ckpt["W"].to(device)
    b = ckpt["b"].to(device)
    linear = nn.Linear(W.shape[1], 6).to(device)
    linear.load_state_dict(ckpt["model_state_dict"])
    linear.eval()
    clf.model_ = (W, b, linear)
    clf.le_ = SKLabelEncoder()
    clf.le_.fit(ALL_LABELS)
    clf.classes_ = clf.le_.classes_
    return clf


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation & Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, class_names, model_name, output_path):
    """Create and save a per-model confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        ax=ax,
        square=False,
        linewidths=0.5,
    )
    ax.set_title(f"{model_name} — Confusion Matrix (Complete Test Set)", fontweight="bold", pad=20)
    ax.set_ylabel("True Gesture", fontweight="bold")
    ax.set_xlabel("Predicted Gesture", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_tdcnn():
    """Evaluate TDCNN on npz test set."""
    print("\n[TDCNN] Loading data and model...")
    X, y_labels = load_tdcnn_data()

    import pickle
    if not TDCNN_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {TDCNN_ENCODER_PATH}")
    with open(TDCNN_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    valid_idx = [i for i, lb in enumerate(y_labels) if lb in le.classes_]
    if len(valid_idx) < len(y_labels):
        print(f"  Filtered {len(y_labels) - len(valid_idx)} samples with unseen labels")
    X = X[valid_idx]
    y_labels = np.array([y_labels[i] for i in valid_idx])
    y_true = le.transform(y_labels)
    class_names = list(le.classes_)

    model = load_tdcnn()
    y_pred = model.predict(X)

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    return {
        "model": "TDCNN",
        "test_samples": int(len(y_true)),
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def evaluate_18f_model(name, load_fn, X, y_labels):
    """Evaluate KNN/XGBoost/SVM on parquet features."""
    print(f"\n[{name}] Loading model and predicting...")
    le = LabelEncoder()
    le.fit(ALL_LABELS)
    y_true = le.transform(y_labels)
    class_names = list(le.classes_)

    model = load_fn()
    y_pred = model.predict(X)

    # Ensure integer indices (KNN/XGBoost/SVM return class indices)
    if y_pred.size > 0 and isinstance(y_pred.flat[0], (str, np.str_)):
        y_pred = le.transform(y_pred)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    return {
        "model": name,
        "test_samples": int(len(y_true)),
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def run_evaluation():
    print("=" * 70)
    print("  EMG-EPN612 — Full Test Set Evaluation & Per-Model Confusion Matrices")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    models_config = []

    # TDCNN: npz
    try:
        res = evaluate_tdcnn()
        all_results.append(res)
        models_config.append(("TDCNN", res))
    except Exception as e:
        print(f"  [SKIP] TDCNN: {e}")

    # 18f models: parquet
    try:
        X_parquet, y_parquet = load_parquet_data()
        print(f"\n  Parquet test set: {len(X_parquet):,} samples")
    except Exception as e:
        print(f"  [SKIP] Parquet: {e}")
        X_parquet = y_parquet = None

    if X_parquet is not None:
        for name, load_fn in [
            ("KNN", load_knn),
            ("XGBoost", load_xgboost),
            ("SVM", load_svm),
        ]:
            try:
                res = evaluate_18f_model(name, load_fn, X_parquet, y_parquet)
                all_results.append(res)
                models_config.append((name, res))
            except Exception as e:
                print(f"  [SKIP] {name}: {e}")

    # Generate confusion matrix plots
    print("\n" + "-" * 70)
    print("  Saving confusion matrices...")
    for name, res in models_config:
        cm = np.array(res["confusion_matrix"])
        out_path = OUTPUT_DIR / f"confusion_matrix_{name}.png"
        plot_confusion_matrix(cm, res["class_names"], name, out_path)
        print(f"    -> {out_path}")

    # Save JSON summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": {
            "TDCNN": str(NPZ_TEST_PATH),
            "18f_models": str(PARQUET_TEST_PATH),
        },
        "models": {r["model"]: {k: v for k, v in r.items() if k != "confusion_matrix"} for r in all_results},
        "confusion_matrices": {r["model"]: r["confusion_matrix"] for r in all_results},
    }
    RESULTS_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Results saved to {RESULTS_JSON}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<12} | {'Samples':>10} | {'Accuracy':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"  {r['model']:<12} | {r['test_samples']:>10,} | {r['accuracy']:>10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
