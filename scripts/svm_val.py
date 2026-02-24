import sys
import time
import json
import gc
import math
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# =============================================================================
# Project paths & dataset constants
# =============================================================================
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PREPROC_DIR   = PROJECT_ROOT / "preprocessed_output"

ALL_LABELS   = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES    = len(ALL_LABELS)
CHANNELS     = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES    = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
ALL_FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]

DATASET_FILES = {
    72: ("dataset_TRAINING.parquet",            "dataset_TEST.parquet"),
    36: ("dataset_TRAINING_reduced36.parquet",  "dataset_TESTING_reduced36.parquet"),
    18: ("dataset_TRAINING_reduced18.parquet",  "dataset_TESTING_reduced18.parquet"),
}

# =============================================================================
# GPU RFF-SVM Classifier
# =============================================================================
class RFFSVMClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible GPU RFF-SVM (approximate RBF kernel)."""

    def __init__(self, D=10000, gamma=0.01, C=5.0, lr=0.05,
                 max_epochs=50, batch_size=16384, patience=7):
        self.D = D
        self.gamma = gamma
        self.C = C
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience

    def fit(self, X, y, X_val=None, y_val=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Features already z-score normalized — no scaling
        X_sc = np.asarray(X, dtype=np.float32)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y).astype(np.int64)
        self.classes_ = self.le_.classes_
        n_cls = len(self.classes_)

        # Prepare RFF projection matrices
        W = torch.randn(X.shape[1], int(self.D), device=device) * math.sqrt(2.0 * float(self.gamma))
        b = torch.rand(int(self.D), device=device) * (2.0 * math.pi)
        
        # Linear head
        linear = nn.Linear(int(self.D), n_cls).to(device)
        wd = 1.0 / (2.0 * self.C * len(X))
        opt = torch.optim.Adam(linear.parameters(), lr=self.lr, weight_decay=wd)

        ds = TensorDataset(torch.from_numpy(X_sc), torch.from_numpy(y_enc))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        best_avg_loss = float("inf")
        best_val_acc = 0.0
        best_linear_state = None
        wait = 0
        crit = nn.MultiMarginLoss()  # Multiclass hinge loss (Crammer-Singer), equivalent to SVM

        epoch_pbar = tqdm(range(self.max_epochs), desc="Epochs", leave=False)
        for epoch in epoch_pbar:
            linear.train()
            epoch_loss_sum = 0.0
            n_batches = 0
            batch_pbar = tqdm(loader, desc="Batches", leave=False)
            for Xb, yb in batch_pbar:
                Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                
                # RFF Mapping: z(x) = sqrt(2/D) * cos(Wx + b)
                z = math.sqrt(2.0 / self.D) * torch.cos(Xb @ W + b)
                scores = linear(z)
                
                loss = crit(scores, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss_sum += loss.item()
                n_batches += 1
                batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_avg_loss = epoch_loss_sum / n_batches
            val_acc = self._score_internal(X_val, y_val, W, b, linear) if X_val is not None else None

            # Hard cutoff: if by epoch 20 we haven't reached 0.65 val acc, skip to next candidate
            if val_acc is not None and epoch >= 19 and val_acc < 0.65:
                print(f"    epoch {epoch}: val_acc={val_acc:.4f} < 0.65 after 20 epochs — skipping remaining epochs for this candidate.")
                epoch_pbar.close()
                break

            # Patience on validation accuracy (fallback to loss only if no val set)
            if val_acc is not None:
                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    best_linear_state = {k: v.cpu().clone() for k, v in linear.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        epoch_pbar.close()
                        break
            else:
                if epoch_avg_loss < best_avg_loss - 1e-4:
                    best_avg_loss = epoch_avg_loss
                    best_linear_state = {k: v.cpu().clone() for k, v in linear.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        epoch_pbar.close()
                        break

            epoch_pbar.set_postfix(
                avg_loss=f"{epoch_avg_loss:.4f}",
                wait=wait,
                val_acc=f"{val_acc:.4f}" if val_acc is not None else "n/a",
                best_val_acc=f"{best_val_acc:.4f}" if val_acc is not None else "n/a",
            )
            val_str = f", val_acc={val_acc:.4f}, best_val_acc={best_val_acc:.4f}" if val_acc is not None else ""
            print(f"    epoch {epoch}: avg_loss={epoch_avg_loss:.4f} (patience wait={wait}{val_str})")

        if best_linear_state is not None:
            linear.load_state_dict(best_linear_state)
            
        self.model_ = (W, b, linear.eval())
        return self

    def _score_internal(self, X, y, W, b, linear):
        """Validation accuracy in batches to avoid GPU OOM on large val sets."""
        X_sc = np.asarray(X, dtype=np.float32)
        y_enc = self.le_.transform(y).astype(np.int64)
        linear.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_sc), self.batch_size):
                batch = torch.from_numpy(X_sc[i:i+self.batch_size]).to(W.device)
                z = math.sqrt(2.0 / self.D) * torch.cos(batch @ W + b)
                preds.append(linear(z).argmax(dim=1).cpu().numpy())
        preds = np.concatenate(preds)
        return accuracy_score(y_enc, preds)

    def predict(self, X):
        W, b, linear = self.model_
        X_sc = np.asarray(X, dtype=np.float32)
        preds = []
        n_batches = (len(X_sc) + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            for i in tqdm(range(0, len(X_sc), self.batch_size), total=n_batches,
                         desc="Predict", leave=False, unit="batch"):
                batch = torch.from_numpy(X_sc[i:i+self.batch_size]).to(W.device)
                z = math.sqrt(2.0 / self.D) * torch.cos(batch @ W + b)
                preds.append(linear(z).argmax(dim=1).cpu().numpy())
        return self.le_.inverse_transform(np.concatenate(preds))

# =============================================================================
# Helper Functions
# =============================================================================
def _format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def _subsample_intra_patient(X, y, groups, n_target, rng):
    """Proportional stratified slice from every patient to maximize diversity."""
    n_total = len(y)
    if n_target >= n_total: return X, y, groups

    frac = n_target / n_total
    selected_idx = []
    for user in np.unique(groups):
        u_idx = np.where(groups == user)[0]
        u_n = max(N_CLASSES, int(len(u_idx) * frac))
        
        sss = StratifiedShuffleSplit(n_splits=1, train_size=u_n, random_state=rng)
        sub_idx, _ = next(sss.split(u_idx, y[u_idx]))
        selected_idx.extend(u_idx[sub_idx])

    idx = np.sort(selected_idx)
    return X[idx], y[idx], groups[idx]

# =============================================================================
# Eval-only helper (load checkpoint and run on test set)
# =============================================================================
def evaluate_saved_model(n_features: int, ckpt_path: Path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = PROJECT_ROOT / "models" / f"{n_features}f"
    if ckpt_path is None:
        # Try svm_val_best{n}f.pt first (e.g. svm_val_best72.pt), then svm_val_best.pt
        ckpt_path = model_dir / f"svm_val_best{n_features}.pt"
        if not ckpt_path.exists():
            ckpt_path = model_dir / "svm_val_best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}. Run the tournament first.")
        return

    print(f"\n[Eval-only] Loading checkpoint from {ckpt_path} on {device} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    params = ckpt.get("params", {})

    # Rebuild classifier core
    clf = RFFSVMClassifier(**params)
    W = ckpt["W"].to(device)
    b = ckpt["b"].to(device)
    n_cls = N_CLASSES
    linear = nn.Linear(W.shape[1], n_cls).to(device)
    linear.load_state_dict(ckpt["model_state_dict"])
    linear.eval()

    # Set model_ and label encoder (must match ALL_LABELS for correct inverse_transform)
    clf.model_ = (W, b, linear)
    clf.le_ = LabelEncoder()
    clf.le_.fit(ALL_LABELS)
    clf.classes_ = clf.le_.classes_

    # Load held-out test set for this feature count
    _, test_name = DATASET_FILES[n_features]
    df_test = pd.read_parquet(PREPROC_DIR / test_name)
    feature_cols = [c for c in df_test.columns if c in ALL_FEATURE_COLS]
    X_test = df_test[feature_cols].values.astype(np.float32)

    le_global = LabelEncoder().fit(ALL_LABELS)
    y_test_enc = le_global.transform(df_test["label"].values)
    y_test_str = le_global.inverse_transform(y_test_enc)
    del df_test; gc.collect()

    print(f"  Test samples: {len(y_test_str):,}")
    t0 = time.time()
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test_str, y_pred))
    report_str = classification_report(y_test_str, y_pred, target_names=ALL_LABELS, digits=4)
    report_dict = classification_report(y_test_str, y_pred, target_names=ALL_LABELS, digits=4, output_dict=True)
    elapsed = time.time() - t0

    print(f"\n[Eval-only] Test Accuracy: {acc:.4f}  ({elapsed:.1f}s)")
    print(report_str)

    # Save results in same format as tournament (svm_val_test_results{n}f.json)
    out_dir = PROJECT_ROOT / "models" / f"{n_features}f"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "script": "svm_val",
        "n_features": n_features,
        "winner_params": params,
        "test_accuracy": acc,
        "classification_report": report_dict,
        "test_samples": len(y_test_str),
        "total_time_s": round(elapsed, 1),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_file = out_dir / f"svm_val_test_results{n_features}.json"
    results_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[Eval-only] Results saved to: {results_file}")


# =============================================================================
# Main Tournament Script
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=72, choices=[72, 36, 18])
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate the saved checkpoint on the test set (skip tournament).",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to checkpoint (e.g. models/72f/svm_val_best72.pt). Default: models/{n}f/svm_val_best{n}.pt or svm_val_best.pt",
    )
    args = parser.parse_args()
    nf = args.features

    if args.eval_only:
        evaluate_saved_model(nf, ckpt_path=args.ckpt)
        return

    # 1. Load Data
    print(f"\n[1/4] Loading Data ({nf}f)...")
    df = pd.read_parquet(PREPROC_DIR / DATASET_FILES[nf][0])
    feature_cols = [c for c in df.columns if c in ALL_FEATURE_COLS]
    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder().fit(ALL_LABELS)
    y = le.transform(df["label"].values)
    groups = df["user"].values
    del df; gc.collect()

    # 2. Patient-Level Holdout Split (80% Train Patients, 20% Val Patients)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    X_train_full, y_train_full, g_train_full = X[train_idx], y[train_idx], groups[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"      Full Train: {len(y_train_full):,} samples ({len(np.unique(g_train_full))} users)")
    print(f"      Holdout Val: {len(y_val):,} samples ({len(np.unique(groups[val_idx]))} users)")

    # 3. Setup SVM Grid
    # 9 total combinations (D=10k fixed, gamma×C×lr = 3×3×1)
    candidates = []
    for gamma in [0.01, 0.1, 1.0]:
        for c in [0.1, 1.0, 10.0]:
            for lr in [0.001]:
                candidates.append({"D": 10000, "gamma": gamma, "C": c, "lr": lr})

    # 4. Successive Halving Tournament
    factor = 3
    n_rounds = math.ceil(math.log(len(candidates), factor))
    t_start_total = time.time()

    print(f"\n[2/4] Starting Tournament: {len(candidates)} candidates, factor={factor} → {n_rounds} rounds.")
    print(f"       Round 0: ~{100/factor:.0f}% data, keep 1/{factor}; Round {n_rounds-1}: 100% data, 1 winner.")

    round_iter = tqdm(range(n_rounds), desc="Tournament rounds", unit="round")
    for r in round_iter:
        # Sample size this round: geometric progression → 100% by last round
        # Round 0: 9 candidates, ~33% data → keep 2. Round 1: 2 candidates, 100% → 1 winner.
        n_samples_r = int(len(y_train_full) * (factor ** (r - n_rounds + 1)))
        
        # Subsample while keeping patient/class proportions
        X_r, y_r, _ = _subsample_intra_patient(X_train_full, y_train_full, g_train_full, 
                                                n_samples_r, np.random.RandomState(42 + r))
        
        pct = 100 * n_samples_r / len(y_train_full)
        round_iter.set_postfix(candidates=len(candidates), samples=f"{n_samples_r:,}", pct=f"{pct:.0f}%")
        print(f"\n--- ROUND {r}/{n_rounds-1} | {len(candidates)} candidates | {n_samples_r:,} samples ({pct:.0f}% of train) ---")
        scores = []
        
        cand_iter = tqdm(enumerate(candidates), total=len(candidates), desc=f"Round {r} candidates", unit="cand")
        for i, params in cand_iter:
            t0 = time.time()
            clf = RFFSVMClassifier(**params)
            clf.fit(X_r, y_r, X_val=X_val, y_val=y_val)
            
            y_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            scores.append(acc)
            cand_iter.set_postfix(acc=f"{acc:.4f}", best_so_far=f"{max(scores):.4f}", elapsed=f"{time.time()-t0:.1f}s")

        # Eliminate: round 0 keep 2 out of 9; round 1 keep 1 winner
        n_keep = 2 if r == 0 else max(1, len(candidates) // factor)
        top_indices = np.argsort(scores)[-n_keep:]
        candidates = [candidates[idx] for idx in top_indices]
        if n_keep == 1:
            print(f"       → Winner: 1 candidate.")
        else:
            print(f"       → Kept top {n_keep} (of {len(scores)}) → {n_keep} candidates to next round.")

    # 5. Final Model Training (winner from round 1 only)
    winner_params = candidates[0]
    print(f"\n[3/4] Winner: {winner_params}")
    print(f"      Training final model on full training set...")
    final_clf = RFFSVMClassifier(**winner_params)
    final_clf.fit(X_train_full, y_train_full)

    # Save final model for future inference
    model_dir = PROJECT_ROOT / "models" / f"{nf}f"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "svm_val_best.pt"
    W, b, linear = final_clf.model_
    # Keep tensors on their current device for consistent inference; device can be handled at load time.
    torch.save(
        {
            "model_state_dict": linear.state_dict(),
            "W": W,
            "b": b,
            "params": winner_params,
        },
        model_path,
    )
    print(f"      Final model saved to: {model_path}")

    # 6. Evaluation on held-out test set
    print(f"\n[4/4] Final Evaluation on Heldout Test Set...")
    df_test = pd.read_parquet(PREPROC_DIR / DATASET_FILES[nf][1])
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = le.transform(df_test["label"].values)
    y_pred_final = final_clf.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred_final))
    report_str = classification_report(y_test, y_pred_final, target_names=ALL_LABELS, digits=4)
    report_dict = classification_report(y_test, y_pred_final, target_names=ALL_LABELS, digits=4, output_dict=True)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(report_str)

    total_time = time.time() - t_start_total
    print(f"\nTotal Tournament Time: {_format_duration(total_time)}")

    # Save run results
    out_dir = PROJECT_ROOT / "models" / f"{nf}f"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "script": "svm_val",
        "n_features": nf,
        "winner_params": winner_params,
        "test_accuracy": test_acc,
        "classification_report": report_dict,
        "test_samples": len(y_test),
        "total_time_s": round(total_time, 1),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_file = out_dir / "svm_val_test_results.json"
    results_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()