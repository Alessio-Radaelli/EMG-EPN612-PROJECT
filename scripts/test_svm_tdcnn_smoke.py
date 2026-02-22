"""
Quick smoke test for RFFSVMClassifier and TDCNNSklearnWrapper.

Verifies the full pipeline used by hyperparam_search.py:
  1. fit / predict with default params
  2. sklearn clone() + set_params() + fit / predict  (same as halving rounds)
  3. _save_final_model()  (model serialization)
  4. _evaluate_on_test_set() path — skipped if dataset_TEST.parquet is absent,
     but we manually test predict-on-new-data here

Uses synthetic data (600 samples, 72 features, 6 classes, 10 "patients")
so the test finishes in seconds on CPU or GPU.

Usage:
  python scripts/test_svm_tdcnn_smoke.py
"""

import sys
import traceback
import numpy as np
from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.hyperparam_search import (
    RFFSVMClassifier,
    TDCNNSklearnWrapper,
    _save_final_model,
    MODELS_DIR,
)

N_SAMPLES = 600
N_FEATURES = 72
N_CLASSES = 6
N_PATIENTS = 10


def make_synthetic_data():
    rng = np.random.RandomState(0)
    X = rng.randn(N_SAMPLES, N_FEATURES).astype(np.float32)
    y = np.tile(np.arange(N_CLASSES), N_SAMPLES // N_CLASSES)
    rng.shuffle(y)
    groups = np.repeat(np.arange(N_PATIENTS), N_SAMPLES // N_PATIENTS)
    return X, y, groups


def test_model(name, estimator, override_params):
    """Run fit/predict, clone+set_params+fit/predict, and save."""
    X, y, _ = make_synthetic_data()
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # --- 1. Default params ---
    print(f"\n  [{name}] fit with default params ...")
    estimator.fit(X_tr, y_tr)
    preds = estimator.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"  [{name}] predict OK  (acc={acc:.4f}, preds shape={preds.shape})")
    assert preds.shape == y_te.shape, "prediction shape mismatch"

    # --- 2. clone + set_params (same path as _evaluate_candidate) ---
    print(f"  [{name}] clone + set_params({override_params}) ...")
    est2 = clone(estimator)
    est2.set_params(**override_params)
    est2.fit(X_tr, y_tr)
    preds2 = est2.predict(X_te)
    acc2 = accuracy_score(y_te, preds2)
    print(f"  [{name}] cloned predict OK  (acc={acc2:.4f})")
    assert preds2.shape == y_te.shape, "cloned prediction shape mismatch"

    # --- 3. Save (same path as end of tournament) ---
    print(f"  [{name}] _save_final_model ...")
    _save_final_model(est2, name)
    expected_path = MODELS_DIR / f"{name}_best_halving.pt"
    assert expected_path.exists(), f"saved model not found at {expected_path}"
    print(f"  [{name}] model file exists: {expected_path} "
          f"({expected_path.stat().st_size:,} bytes)")

    # --- 4. Predict on fresh unseen data (simulates test-set eval) ---
    X_new = np.random.RandomState(99).randn(100, N_FEATURES).astype(np.float32)
    preds_new = est2.predict(X_new)
    assert preds_new.shape == (100,), "fresh-data prediction shape mismatch"
    print(f"  [{name}] predict on fresh data OK  (shape={preds_new.shape})")

    print(f"  [{name}] ALL CHECKS PASSED")


def main():
    passed = []
    failed = []

    # --- SVM ---
    try:
        test_model("svm", RFFSVMClassifier(D=500, epochs=3, batch_size=128),
                   {"D": 300, "gamma": 0.05, "C": 1.0, "lr": 0.01, "epochs": 2})
        passed.append("svm")
    except Exception:
        traceback.print_exc()
        failed.append("svm")

    # --- TDCNN ---
    try:
        test_model("tdcnn", TDCNNSklearnWrapper(hidden_channels=(16, 32),
                                                 epochs=2, batch_size=64),
                   {"hidden_channels": (8, 16), "kernel_size": 5,
                    "dropout": 0.3, "learning_rate": 0.005,
                    "batch_size": 32, "epochs": 2})
        passed.append("tdcnn")
    except Exception:
        traceback.print_exc()
        failed.append("tdcnn")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"  PASSED: {', '.join(passed) if passed else 'none'}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'='*50}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
