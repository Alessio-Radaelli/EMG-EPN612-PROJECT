"""
Successive Halving Hyperparameter Search for EMG-EPN612.

Manual implementation of the successive halving algorithm (tournament-style)
that replaces sklearn's HalvingRandomSearchCV, adding:
  - StratifiedGroupKFold (patient-level + class-distribution stratification)
  - Per-round checkpointing with auto-resume (crash-safe, use --no-resume to start fresh)
  - Reusable tournament schedules across models
  - Model-aware parallelization (CPU threading vs GPU)

How it works (100 candidates, factor=3, ~1.5M samples):
  Round 0:  100 candidates × 3-fold CV on  ~55K samples → keep top 33
  Round 1:   33 candidates × 3-fold CV on ~167K samples → keep top 11
  Round 2:   11 candidates × 3-fold CV on ~500K samples → keep top  3
  Round 3:    3 candidates × 3-fold CV on ~1.5M samples → winner
  Then: train winner on full 1.5M → evaluate on held-out test set

Supported models:
  xgboost  – XGBClassifier (CPU/GPU trees)
  svm      – RFF-based SVM approximation (GPU PyTorch)
  tdcnn    – Time-Delay CNN + ECA (GPU PyTorch)
  knn      – ENN + CNN + KNN pipeline (CPU, imblearn)

Usage:
  python scripts/hyperparam_search.py --model xgboost
  python scripts/hyperparam_search.py --model xgboost --no-resume   # discard checkpoints
  python scripts/hyperparam_search.py --model xgboost --n-candidates 50 --factor 2
  python scripts/hyperparam_search.py --model all
"""

import sys
import time
import json
import gc
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform, randint as sp_randint
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# Project paths & dataset constants
# =============================================================================

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATASET_FILE      = PROJECT_ROOT / "preprocessed_output" / "dataset_TRAINING.parquet"
TEST_DATASET_FILE = PROJECT_ROOT / "preprocessed_output" / "dataset_TEST.parquet"
MODELS_DIR    = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR = MODELS_DIR / "halving_checkpoints"

ALL_LABELS   = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES    = len(ALL_LABELS)
CHANNELS     = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES    = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
N_FEATURES   = len(FEATURE_COLS)  # 72


# =============================================================================
# PyTorch modules (inlined from training scripts to avoid package imports)
# =============================================================================

class RandomFourierFeatures(nn.Module):
    """Fixed RFF transform: z(x) = sqrt(2/D) * cos(Wx + b)."""
    def __init__(self, in_features, out_features, gamma):
        super().__init__()
        W = torch.randn(in_features, out_features) * math.sqrt(2.0 * gamma)
        b = torch.rand(out_features) * (2.0 * math.pi)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / out_features)

    def forward(self, x):
        return self.scale * torch.cos(x @ self.W + self.b)


class _ECA(nn.Module):
    """Efficient Channel Attention."""
    def __init__(self, channels=None, k_size=3, gamma=2, b=1):
        super().__init__()
        if channels is not None:
            t = int(abs((math.log2(channels) + b) / gamma))
            k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class _Chomp1d(nn.Module):
    """Causal trim for time-delay convolutions."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class _TDCNNModel(nn.Module):
    """Time-Delay CNN with ECA blocks."""
    def __init__(self, input_channels, num_classes,
                 hidden_channels=(64, 128), kernel_size=3, dropout=0.5):
        super().__init__()
        layers = []
        in_c = input_channels
        dilation = 1
        for out_c in hidden_channels:
            padding = (kernel_size - 1) * dilation
            layers.extend([
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size,
                          padding=padding, dilation=dilation),
                _Chomp1d(padding),
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(0.1),
                _ECA(channels=out_c),
                nn.Dropout(0.1),
            ])
            in_c = out_c
            dilation *= 2
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(in_c, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


# =============================================================================
# Sklearn-compatible wrappers
# =============================================================================

class RFFSVMClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible GPU RFF-SVM (approximate RBF kernel)."""

    def __init__(self, D=10000, gamma=0.01, C=5.0, lr=0.05,
                 max_epochs=70, batch_size=4096, patience=10):
        self.D = D
        self.gamma = gamma
        self.C = C
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience

    def fit(self, X, y, X_val=None, y_val=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler_ = StandardScaler()
        X_sc = self.scaler_.fit_transform(X).astype(np.float32)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y).astype(np.int64)
        self.classes_ = self.le_.classes_
        n_cls = len(self.classes_)

        if X_val is not None:
            X_val_sc = self.scaler_.transform(X_val).astype(np.float32)
            y_val_enc = self.le_.transform(y_val).astype(np.int64)
            X_val_t = torch.from_numpy(X_val_sc).to(device)
            y_val_t = torch.from_numpy(y_val_enc).to(device)

        rff = RandomFourierFeatures(X.shape[1], int(self.D), float(self.gamma))
        linear = nn.Linear(int(self.D), n_cls)
        self.model_ = nn.Sequential(rff, linear).to(device)

        wd = 1.0 / (2.0 * self.C * len(X))
        opt = torch.optim.Adam(linear.parameters(), lr=self.lr, weight_decay=wd)

        ds = TensorDataset(torch.from_numpy(X_sc),
                           torch.from_numpy(y_enc))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            pin_memory=(device.type == "cuda"), num_workers=0)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        self.model_.train()
        for epoch in range(self.max_epochs):
            for Xb, yb in loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                scores = self.model_(Xb)
                n = scores.size(0)
                correct = scores[torch.arange(n, device=device), yb]
                margins = scores - correct.unsqueeze(1) + 1.0
                margins[torch.arange(n, device=device), yb] = 0.0
                loss = margins.clamp(min=0).max(dim=1)[0].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            if X_val is not None:
                self.model_.eval()
                val_losses = []
                with torch.no_grad():
                    for vi in range(0, len(X_val_sc), self.batch_size):
                        Xvb = X_val_t[vi:vi + self.batch_size]
                        yvb = y_val_t[vi:vi + self.batch_size]
                        vs = self.model_(Xvb)
                        nv = vs.size(0)
                        cv = vs[torch.arange(nv, device=device), yvb]
                        mv = vs - cv.unsqueeze(1) + 1.0
                        mv[torch.arange(nv, device=device), yvb] = 0.0
                        val_losses.append(float(mv.clamp(min=0).max(dim=1)[0].sum()))
                    val_loss = sum(val_losses) / len(X_val_sc)
                self.model_.train()

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in
                                  self.model_.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        return self

    def predict(self, X):
        device = next(self.model_.parameters()).device
        X_sc = self.scaler_.transform(X).astype(np.float32)
        bs = self.batch_size * 2
        preds = []
        self.model_.eval()
        with torch.no_grad():
            for i in range(0, len(X_sc), bs):
                batch = torch.from_numpy(X_sc[i:i + bs]).to(device)
                preds.append(self.model_(batch).argmax(dim=1).cpu().numpy())
        return self.le_.inverse_transform(np.concatenate(preds))


class TDCNNSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for TDCNN + ECA.

    Reshapes flat 72-feature rows into (8 channels, 9 TD features)
    for 1D causal convolution.
    """

    def __init__(self, hidden_channels=(64, 128), kernel_size=3,
                 dropout=0.5, learning_rate=0.001,
                 batch_size=32, max_epochs=70, patience=10):
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

    # sklearn clone() needs get_params to return the exact init values;
    # we must ensure hidden_channels stays as a tuple.
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["hidden_channels"] = tuple(params["hidden_channels"])
        return params

    def _reshape(self, X):
        """(n, 72) → (n, 8, 9):  8 EMG channels × 9 TD9 features."""
        return X.reshape(-1, 8, 9).astype(np.float32)

    def fit(self, X, y, X_val=None, y_val=None):
        X_3d = self._reshape(X)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y).astype(np.int64)
        self.classes_ = self.le_.classes_

        if X_val is not None:
            X_val_3d = self._reshape(X_val)
            y_val_enc = self.le_.transform(y_val).astype(np.int64)
            X_val_t = torch.from_numpy(X_val_3d).to(device)
            y_val_t = torch.from_numpy(y_val_enc).to(device)

        hc = tuple(self.hidden_channels)
        self.model_ = _TDCNNModel(
            input_channels=8, num_classes=len(self.classes_),
            hidden_channels=hc,
            kernel_size=self.kernel_size, dropout=self.dropout,
        ).to(device)

        opt = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        crit = nn.CrossEntropyLoss()

        ds = TensorDataset(torch.from_numpy(X_3d),
                           torch.from_numpy(y_enc))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            pin_memory=(device.type == "cuda"), num_workers=0)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        self.model_.train()
        for epoch in range(self.max_epochs):
            for Xb, yb in loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                loss = crit(self.model_(Xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if X_val is not None:
                self.model_.eval()
                val_losses, val_n = 0.0, 0
                with torch.no_grad():
                    for vi in range(0, len(X_val_3d), self.batch_size):
                        Xvb = X_val_t[vi:vi + self.batch_size]
                        yvb = y_val_t[vi:vi + self.batch_size]
                        val_losses += float(crit(self.model_(Xvb), yvb)) * len(yvb)
                        val_n += len(yvb)
                val_loss = val_losses / val_n
                self.model_.train()

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in
                                  self.model_.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        return self

    def predict(self, X):
        X_3d = self._reshape(X)
        device = next(self.model_.parameters()).device
        bs = self.batch_size * 2
        preds = []
        self.model_.eval()
        with torch.no_grad():
            for i in range(0, len(X_3d), bs):
                batch = torch.from_numpy(X_3d[i:i + bs]).to(device)
                preds.append(self.model_(batch).argmax(dim=1).cpu().numpy())
        return self.le_.inverse_transform(np.concatenate(preds))


class ENNCNNKNNClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible ENN + CNN + KNN pipeline.

    Applies Edited Nearest Neighbours (noise cleaning) then Condensed
    Nearest Neighbour (prototype selection) during fit, producing a
    compact reference set for KNN inference.

    ENN/CNN parameters are fixed at domain-appropriate defaults;
    only KNN hyperparameters are tuned via the halving search.
    """

    def __init__(self, n_neighbors=3, weights="uniform", metric="euclidean",
                 enn_n_neighbors=3, enn_kind_sel="mode"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.enn_n_neighbors = enn_n_neighbors
        self.enn_kind_sel = enn_kind_sel

    def fit(self, X, y):
        from imblearn.under_sampling import (
            EditedNearestNeighbours, CondensedNearestNeighbour,
        )
        from sklearn.neighbors import KNeighborsClassifier

        enn = EditedNearestNeighbours(
            n_neighbors=self.enn_n_neighbors,
            kind_sel=self.enn_kind_sel,
        )
        X_enn, y_enn = enn.fit_resample(X, y)

        cnn = CondensedNearestNeighbour(random_state=42)
        X_cnn, y_cnn = cnn.fit_resample(X_enn, y_enn)

        self.n_prototypes_ = len(y_cnn)
        self.X_store_ = X_cnn
        self.y_store_ = y_cnn

        self.knn_ = KNeighborsClassifier(
            n_neighbors=min(self.n_neighbors, len(y_cnn)),
            weights=self.weights,
            metric=self.metric,
            n_jobs=-1,
        )
        self.knn_.fit(X_cnn, y_cnn)
        self.classes_ = self.knn_.classes_

        return self

    def predict(self, X):
        return self.knn_.predict(X)


# =============================================================================
# Model configurations  (estimator + param distributions)
# =============================================================================

def get_model_config(model_name, no_gpu=False):
    """Return (estimator, param_distributions, is_gpu_model)."""

    if model_name == "xgboost":
        import xgboost as xgb

        device = "cpu"
        if not no_gpu:
            try:
                _t = xgb.XGBClassifier(n_estimators=1, max_depth=1,
                                       device="cuda", verbosity=0)
                _t.fit(np.zeros((2, 2)), np.array([0, 1]))
                device = "cuda"
            except xgb.core.XGBoostError:
                pass

        estimator = xgb.XGBClassifier(
            n_estimators=1000,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            device=device,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=20,
        )
        param_dist = {
            "max_depth":       sp_randint(3, 10),
            "learning_rate":   loguniform(0.01, 0.3),
            "min_child_weight": sp_randint(1, 15),
            "subsample":       uniform(0.6, 0.4),       # [0.5, 1.0]
            "colsample_bytree": uniform(0.6, 0.4),      # [0.5, 1.0]
            "reg_alpha":       loguniform(1e-3, 10),
            "reg_lambda":      loguniform(1e-3, 10),
            "gamma":           uniform(0, 5),
        }
        return estimator, param_dist, False

    elif model_name == "svm":
        estimator = RFFSVMClassifier()
        param_dist = {
            "D":      [10000, 20000, 50000],
            "gamma":  loguniform(0.001, 1),
            "C":      loguniform(0.1, 100),
            "lr":     loguniform(1e-3, 0.1),
        }
        return estimator, param_dist, True

    elif model_name == "tdcnn":
        estimator = TDCNNSklearnWrapper()
        param_dist = {
            "hidden_channels": [(32, 64), (64, 128), (64, 128, 256)],
            "kernel_size":     [3, 5, 7],
            "dropout":         uniform(0.2, 0.2),      # [0.2, 0.4]
            "learning_rate":   loguniform(1e-4, 1e-2),
            "batch_size":      [32, 64, 128],
        }
        return estimator, param_dist, True

    elif model_name == "knn":
        estimator = ENNCNNKNNClassifier()
        param_dist = {
            "n_neighbors":  [1, 3, 5],
            "weights":      ["uniform", "distance"],
            "metric":       ["euclidean", "manhattan"],
        }
        return estimator, param_dist, False

    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Helpers
# =============================================================================

def _json_safe(obj):
    """Convert numpy / tuple types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def _params_to_json(params: dict) -> dict:
    return {k: _json_safe(v) for k, v in params.items()}


def _format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _sample_candidates(param_distributions, n_candidates, rng):
    """Draw n_candidates random parameter sets from the distributions.

    If every distribution is a finite list, exhaustively enumerate all
    combinations (ignoring n_candidates) so nothing is missed or duplicated.
    """
    all_lists = all(isinstance(d, list) for d in param_distributions.values())

    if all_lists:
        from itertools import product
        names = list(param_distributions.keys())
        combos = list(product(*(param_distributions[n] for n in names)))
        rng.shuffle(combos)
        candidates = [{n: _json_safe(v) for n, v in zip(names, combo)}
                      for combo in combos]
        return candidates

    candidates = []
    for _ in range(n_candidates):
        params = {}
        for name, dist in param_distributions.items():
            if hasattr(dist, "rvs"):
                val = dist.rvs(random_state=rng)
            elif isinstance(dist, list):
                val = dist[rng.randint(len(dist))]
            else:
                val = dist
            params[name] = _json_safe(val)
        candidates.append(params)
    return candidates


def _subsample_by_group(y, groups, n_target, rng):
    """Subsample entire patient groups to reach ~n_target samples.

    Maintains group integrity: every sample from a selected patient
    is included.  Approximate class balance is preserved because
    each patient performs all gestures.

    Returns sorted index array.
    """
    n_total = len(y)
    if n_target >= n_total:
        return np.arange(n_total)

    unique_g = np.unique(groups)
    rng.shuffle(unique_g)

    # Pre-compute group → indices mapping for speed
    g_to_idx = {}
    for g in unique_g:
        g_to_idx[g] = np.where(groups == g)[0]

    selected_idx = []
    for g in unique_g:
        if len(selected_idx) >= n_target:
            break
        selected_idx.extend(g_to_idx[g].tolist())

    return np.sort(selected_idx)


def _evaluate_candidate(estimator, params, X, y, groups, cv):
    """Evaluate one candidate via cross-validation, return fold scores."""
    est = clone(estimator)
    est.set_params(**params)

    try:
        import xgboost as xgb
        is_xgb = isinstance(est, xgb.XGBClassifier)
    except ImportError:
        is_xgb = False
    has_val = isinstance(est, (RFFSVMClassifier, TDCNNSklearnWrapper))

    scores = []
    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        if is_xgb:
            est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        elif has_val:
            est.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        else:
            est.fit(X_tr, y_tr)

        y_pred = est.predict(X_va)
        scores.append(float(accuracy_score(y_va, y_pred)))

    return np.array(scores)


# =============================================================================
# Tournament schedule management
# =============================================================================

def compute_schedule(n_total, n_candidates, factor):
    """Compute the halving round schedule.

    Returns list of dicts: [{round, n_candidates, n_samples}, ...]
    """
    # Number of elimination rounds needed to reach 1 survivor
    c, n_rounds = n_candidates, 0
    while c > 1:
        c = max(1, int(c / factor))
        n_rounds += 1

    # min_resources: back-calculate so the last round uses all data
    min_resources = max(
        N_CLASSES * 10,  # absolute minimum for stratified splits
        int(n_total / (factor ** (n_rounds - 1))),
    )

    schedule = []
    cand = n_candidates
    for r in range(n_rounds):
        n_samples = min(int(min_resources * (factor ** r)), n_total)
        schedule.append({
            "round": r,
            "n_candidates": int(cand),
            "n_samples": n_samples,
        })
        cand = max(1, int(cand / factor))

    return schedule


def load_or_create_schedule(n_total, n_candidates, factor, n_splits,
                            random_state, model_name="default"):
    """Load saved schedule or create + save a new one.

    Each model gets its own schedule file so models with different candidate
    counts (e.g. KNN with an exhaustive grid) don't conflict.
    """
    schedule_file = MODELS_DIR / f"halving_schedule_{model_name}.json"

    meta = {
        "dataset_rows": n_total,
        "n_candidates": n_candidates,
        "factor": factor,
        "n_splits": n_splits,
        "random_state": random_state,
    }

    if schedule_file.exists():
        saved = json.loads(schedule_file.read_text())
        compat = True
        for key in ("dataset_rows", "n_candidates", "factor",
                     "n_splits", "random_state"):
            if saved.get(key) != meta[key]:
                compat = False
                break
        if compat:
            print(f"  Loaded existing tournament schedule from {schedule_file.name}")
            return saved["rounds"]
        else:
            print(f"  WARNING: schedule parameters changed — regenerating.")
            print(f"    Saved: { {k: saved.get(k) for k in meta} }")
            print(f"    Current: {meta}")

    rounds = compute_schedule(n_total, n_candidates, factor)
    saved = {**meta, "rounds": rounds}
    schedule_file.write_text(json.dumps(saved, indent=2))
    print(f"  Saved tournament schedule to {schedule_file.name}")
    return rounds


# =============================================================================
# Successive halving engine
# =============================================================================

def run_halving(model_name, estimator, param_distributions, X, y, groups,
                n_candidates, factor, n_splits, random_state, resume):
    """Run the full successive halving tournament for one model."""

    n_total = len(y)
    rng = np.random.RandomState(random_state)

    # --- Checkpoint directory -------------------------------------------
    cp_dir = CHECKPOINTS_DIR / model_name
    cp_dir.mkdir(parents=True, exist_ok=True)

    # --- Sample or load initial candidates ------------------------------
    cand_file = cp_dir / "initial_candidates.json"
    if resume and cand_file.exists():
        candidates = json.loads(cand_file.read_text())
        print(f"\n  Loaded {len(candidates)} initial candidates from checkpoint")
    else:
        candidates = _sample_candidates(param_distributions, n_candidates, rng)
        cand_file.write_text(json.dumps(candidates, indent=2, default=str))
        print(f"\n  Sampled {len(candidates)} initial candidates")

    actual_n_candidates = len(candidates)

    # --- Schedule (based on actual candidate count) ---------------------
    rounds = load_or_create_schedule(
        n_total, actual_n_candidates, factor, n_splits, random_state,
        model_name=model_name)
    n_rounds = len(rounds)

    print(f"\n  Tournament: {n_rounds} rounds, factor={factor}")
    for r in rounds:
        print(f"    Round {r['round']}: {r['n_candidates']:3d} candidates "
              f"× {r['n_samples']:>10,} samples")

    # --- Resume: find last completed round ------------------------------
    start_round = 0
    if resume:
        for r in range(n_rounds):
            cp_file = cp_dir / f"round_{r}.json"
            if cp_file.exists():
                cp = json.loads(cp_file.read_text())
                candidates = cp["survivors"]
                start_round = r + 1
            else:
                break
        if start_round > 0:
            print(f"  Resuming from round {start_round} "
                  f"({len(candidates)} surviving candidates)")

    # --- CV splitter ----------------------------------------------------
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)

    # --- Run rounds -----------------------------------------------------
    t_total = time.time()

    for round_k in range(start_round, n_rounds):
        round_info = rounds[round_k]
        n_samples_k = round_info["n_samples"]
        n_cand_k = len(candidates)

        tournament_elapsed = time.time() - t_total
        print(f"\n{'='*70}")
        print(f"  ROUND {round_k}/{n_rounds - 1}:  "
              f"{n_cand_k} candidates × {n_samples_k:,} samples "
              f"× {n_splits}-fold CV"
              f"  [elapsed: {_format_duration(tournament_elapsed)}]")
        print(f"{'='*70}")

        # Subsample (deterministic per round — fresh RNG from seed + round)
        round_rng = np.random.RandomState(random_state + round_k * 1000)
        sub_idx = _subsample_by_group(y, groups, n_samples_k, round_rng)
        X_round = X[sub_idx]
        y_round = y[sub_idx]
        groups_round = groups[sub_idx]
        actual_n = len(sub_idx)
        print(f"  Subsampled: {actual_n:,} rows "
              f"({len(np.unique(groups_round))} patients)")

        # Evaluate each surviving candidate
        all_scores = []
        t_round = time.time()

        for c_idx, params in enumerate(candidates):
            t0 = time.time()
            try:
                fold_scores = _evaluate_candidate(
                    estimator, params, X_round, y_round, groups_round, cv)
                mean_sc = float(fold_scores.mean())
                std_sc = float(fold_scores.std())
            except Exception as e:
                print(f"    [{c_idx+1}/{n_cand_k}] FAILED: {e}")
                mean_sc, std_sc = 0.0, 0.0
                fold_scores = np.zeros(n_splits)

            elapsed = time.time() - t0

            # ETA
            done = c_idx + 1
            remaining_cand = n_cand_k - done
            avg_time = (time.time() - t_round) / done
            eta = _format_duration(avg_time * remaining_cand)

            print(f"    [{done:3d}/{n_cand_k}] "
                  f"score={mean_sc:.4f}±{std_sc:.4f}  "
                  f"({elapsed:.1f}s)  ETA: {eta}")

            all_scores.append({
                "params": _params_to_json(params),
                "mean_score": mean_sc,
                "std_score": std_sc,
                "fold_scores": fold_scores.tolist(),
                "time_s": round(elapsed, 1),
            })

        # Rank and eliminate
        all_scores.sort(key=lambda x: x["mean_score"], reverse=True)
        n_survivors = max(1, int(n_cand_k / factor))
        survivors = [s["params"] for s in all_scores[:n_survivors]]

        round_elapsed = time.time() - t_round
        round_work = {r: rounds[r]["n_candidates"] * rounds[r]["n_samples"]
                      for r in range(n_rounds)}
        work_done = sum(round_work[r] for r in range(start_round, round_k + 1))
        work_left = sum(round_work[r] for r in range(round_k + 1, n_rounds))
        time_per_work = (time.time() - t_total) / work_done if work_done else 0
        tournament_eta = _format_duration(time_per_work * work_left)
        rounds_left = n_rounds - round_k - 1

        print(f"\n  Round {round_k} complete ({_format_duration(round_elapsed)})")
        print(f"    Best:  {all_scores[0]['mean_score']:.4f}")
        print(f"    Worst: {all_scores[-1]['mean_score']:.4f}")
        print(f"    Eliminated: {n_cand_k - n_survivors}  |  "
              f"Surviving: {n_survivors}")
        if rounds_left > 0:
            print(f"    Tournament ETA: ~{tournament_eta} "
                  f"({rounds_left} round{'s' if rounds_left > 1 else ''} left)")

        # Save checkpoint
        checkpoint = {
            "round": round_k,
            "n_samples": actual_n,
            "n_candidates": n_cand_k,
            "n_survivors": n_survivors,
            "all_scores": all_scores,
            "survivors": survivors,
            "elapsed_s": round(round_elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        cp_file = cp_dir / f"round_{round_k}.json"
        cp_file.write_text(json.dumps(checkpoint, indent=2, default=str))
        print(f"    Checkpoint saved: {cp_file.name}")

        candidates = survivors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Final summary --------------------------------------------------
    total_elapsed = time.time() - t_total
    best_params = candidates[0]

    print(f"\n{'='*70}")
    print(f"  TOURNAMENT COMPLETE  ({_format_duration(total_elapsed)})")
    print(f"{'='*70}")
    print(f"\n  Best hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k:>20s} : {v}")

    # Save best params
    best_file = MODELS_DIR / f"{model_name}_best_params.json"
    best_file.write_text(json.dumps(best_params, indent=2, default=str))
    print(f"\n  Best params saved to: {best_file}")

    # Save full results CSV
    _save_results_csv(model_name, cp_dir, n_rounds)

    # Train final model with best params on full data (no early stopping)
    print(f"\n  Training final model with best params on full data ...")
    t0 = time.time()
    final_est = clone(estimator)
    final_est.set_params(**best_params)
    try:
        import xgboost as xgb
        if isinstance(final_est, xgb.XGBClassifier):
            final_est.set_params(early_stopping_rounds=None)
    except ImportError:
        pass
    final_est.fit(X, y)
    print(f"  Final model trained ({time.time()-t0:.1f}s)")

    # Save final model
    _save_final_model(final_est, model_name)

    # Evaluate on held-out test set
    _evaluate_on_test_set(final_est, model_name)

    return best_params


def _save_results_csv(model_name, cp_dir, n_rounds):
    """Compile all round checkpoints into a single results CSV."""
    rows = []
    for r in range(n_rounds):
        cp_file = cp_dir / f"round_{r}.json"
        if not cp_file.exists():
            continue
        cp = json.loads(cp_file.read_text())
        for entry in cp["all_scores"]:
            row = {"round": r, "n_samples": cp["n_samples"]}
            row.update(entry["params"])
            row["mean_score"] = entry["mean_score"]
            row["std_score"] = entry["std_score"]
            row["time_s"] = entry["time_s"]
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = MODELS_DIR / f"{model_name}_halving_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Results CSV: {csv_path}")


def _save_final_model(estimator, model_name):
    """Save the final trained model."""
    if model_name == "xgboost":
        path = MODELS_DIR / f"{model_name}_best_halving.json"
        estimator.save_model(str(path))
        print(f"  Model saved: {path}")

    elif model_name == "svm":
        path = MODELS_DIR / f"{model_name}_best_halving.pt"
        torch.save({
            "model_state_dict": estimator.model_.cpu().state_dict(),
            "scaler": estimator.scaler_,
            "label_encoder": estimator.le_,
            "params": estimator.get_params(),
        }, path)
        print(f"  Model saved: {path}")

    elif model_name == "tdcnn":
        path = MODELS_DIR / f"{model_name}_best_halving.pt"
        torch.save({
            "model_state_dict": estimator.model_.cpu().state_dict(),
            "label_encoder": estimator.le_,
            "params": estimator.get_params(),
        }, path)
        print(f"  Model saved: {path}")

    elif model_name == "knn":
        import joblib
        path = MODELS_DIR / f"{model_name}_best_halving.joblib"
        joblib.dump({
            "knn": estimator.knn_,
            "X_store": estimator.X_store_,
            "y_store": estimator.y_store_,
            "n_prototypes": estimator.n_prototypes_,
            "params": estimator.get_params(),
        }, path)
        print(f"  Model saved: {path}")
        print(f"  Prototypes: {estimator.n_prototypes_:,} "
              f"(condensed from full training set)")


def _evaluate_on_test_set(estimator, model_name):
    """Load the held-out test set, predict with the fitted model, and report."""
    if not TEST_DATASET_FILE.exists():
        print(f"\n  WARNING: {TEST_DATASET_FILE} not found — skipping test eval.")
        return

    print(f"\n  Evaluating on test set ({TEST_DATASET_FILE.name}) ...")
    t0 = time.time()
    df = pd.read_parquet(TEST_DATASET_FILE)
    X_test = df[FEATURE_COLS].values.astype(np.float32)

    le = LabelEncoder()
    le.fit(ALL_LABELS)
    y_test = le.transform(df["label"].values)
    label_map = {i: name for i, name in enumerate(le.classes_)}

    del df
    gc.collect()

    y_pred = estimator.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report_str = classification_report(
        y_test, y_pred,
        target_names=[label_map[i] for i in range(len(label_map))],
        digits=4,
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=[label_map[i] for i in range(len(label_map))],
        digits=4,
        output_dict=True,
    )
    elapsed = time.time() - t0

    print(f"  Test accuracy: {acc:.4f}  ({elapsed:.1f}s)")
    print(f"\n{report_str}")

    results = {
        "model": model_name,
        "test_accuracy": acc,
        "classification_report": report_dict,
        "test_samples": len(X_test),
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_file = MODELS_DIR / f"{model_name}_test_results.json"
    results_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Test results saved to: {results_file}")


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    """Load dataset_TRAINING.parquet and return X, y, groups."""
    if not DATASET_FILE.exists():
        print(f"ERROR: {DATASET_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Loading {DATASET_FILE.name} ...")
    t0 = time.time()
    df = pd.read_parquet(DATASET_FILE)
    print(f"  {len(df):,} rows × {len(df.columns)} cols  ({time.time()-t0:.1f}s)")

    # Features
    X = df[FEATURE_COLS].values.astype(np.float32)

    # Labels — encode to integers for cross-model compatibility.
    # String labels cause issues with XGBoost's num_class auto-detection
    # when combined with sklearn's clone(). Integer labels work universally.
    le = LabelEncoder()
    le.fit(ALL_LABELS)
    y = le.transform(df["label"].values)
    label_map = dict(zip(le.transform(le.classes_), le.classes_))

    # Groups (patient IDs)
    groups = df["user"].values

    del df
    gc.collect()

    print(f"  Features: {X.shape}")
    print(f"  Classes:  {label_map}")
    print(f"  Patients: {len(np.unique(groups))}")

    return X, y, groups


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Successive Halving Hyperparameter Search for EMG-EPN612.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, required=True,
                   choices=["xgboost", "svm", "tdcnn", "knn", "all"],
                   help="Model to search (or 'all').")
    p.add_argument("--n-candidates", type=int, default=100,
                   help="Number of initial random candidates.")
    p.add_argument("--factor", type=int, default=3,
                   help="Elimination factor (keep 1/factor each round).")
    p.add_argument("--n-splits", type=int, default=3,
                   help="Number of CV folds (StratifiedGroupKFold).")
    p.add_argument("--no-gpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    p.add_argument("--no-resume", action="store_false", dest="resume",
                   help="Start fresh, ignoring existing checkpoints.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  Successive Halving Hyperparameter Search  —  EMG-EPN612")
    print("=" * 70)

    # Load data once
    X, y, groups = load_data()

    # Which models to search
    if args.model == "all":
        models = ["xgboost", "svm", "tdcnn", "knn"]
    else:
        models = [args.model]

    for model_name in models:
        print(f"\n\n{'#'*70}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'#'*70}")

        estimator, param_dist, is_gpu = get_model_config(
            model_name, no_gpu=args.no_gpu)

        run_halving(
            model_name=model_name,
            estimator=estimator,
            param_distributions=param_dist,
            X=X, y=y, groups=groups,
            n_candidates=args.n_candidates,
            factor=args.factor,
            n_splits=args.n_splits,
            random_state=args.seed,
            resume=args.resume,
        )

    print(f"\n\n{'='*70}")
    print("  ALL SEARCHES COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

