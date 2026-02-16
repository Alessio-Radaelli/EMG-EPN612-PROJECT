"""
GPU-accelerated RBF-SVM Training via Random Fourier Features (RFF) + PyTorch.

Implements an approximate RBF-kernel SVM on the GPU using:
 1. Random Fourier Features (RFF): a fixed random projection that maps
    72 TD9 features into D-dimensional space where the RBF kernel is
    approximated by a linear inner product.
 2. Linear classifier (hinge loss + L2 penalty) trained with SGD
    on the GPU-mapped features — equivalent to an RBF-SVM.

Why this works:
    The key identity is:
        K(x, x') = exp(-gamma * ||x - x'||^2)
                  ≈ z(x)^T z(x')
    where z(x) = sqrt(2/D) * cos(Wx + b), W ~ N(0, 2*gamma*I), b ~ U(0, 2π).
    A linear SVM in z-space is therefore an approximate RBF-SVM in x-space.
    With D = 10,000+ features (feasible on GPU), the approximation is excellent.

GPU advantage:
    - The RFF transform z(x) = cos(X @ W + b) is a matrix multiply — the
      exact operation GPUs are built for.
    - D = 10,000–20,000 features fit in 4 GB VRAM with mini-batches.
    - Training is 5–10× faster than CPU Nystroem + SGD.

Usage:
    cd "EMG-EPN612 project"
    python scripts/train_svm_gpu.py
    python scripts/train_svm_gpu.py --D 20000 --gamma 0.02 --C 1.0
    python scripts/train_svm_gpu.py --batch-size 4096 --epochs 30
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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# --- Project paths ------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TRAINING_FILE = PROJECT_ROOT / "training_set.parquet"
MODELS_DIR    = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# --- Dataset constants --------------------------------------------------------
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES  = len(ALL_LABELS)

CHANNELS     = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES    = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
N_FEATURES   = len(FEATURE_COLS)  # 72

# --- Defaults -----------------------------------------------------------------
DEFAULT_D          = 20_000  # RFF output dimension (20K sufficient for 72 features)
DEFAULT_GAMMA      = "scale" # RBF bandwidth: 'scale' = 1/(d*Var(X)), standard heuristic
DEFAULT_C          = 1.0     # SVM regularization: moderate for faster convergence
DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 2048    # mini-batch size
DEFAULT_LR         = 1e-2    # Adam LR (larger steps to escape flat regions)
DEFAULT_VAL_FRAC   = 0.15


# --- Random Fourier Features layer -------------------------------------------

class RandomFourierFeatures(nn.Module):
    """Fixed (non-learnable) RFF transform: z(x) = sqrt(2/D) * cos(Wx + b).

    W ~ N(0, 2*gamma*I),  b ~ Uniform(0, 2*pi).
    The output dimension is D.  The weights are frozen (never updated).
    """
    def __init__(self, in_features: int, out_features: int, gamma: float):
        super().__init__()
        # W has shape (in_features, D) — sampled from N(0, 2*gamma)
        W = torch.randn(in_features, out_features) * math.sqrt(2.0 * gamma)
        b = torch.rand(out_features) * (2.0 * math.pi)
        # Register as buffers (saved with model, moved to GPU, but not trained)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / out_features)

    def forward(self, x):
        # x: (batch, in_features) -> (batch, D)
        return self.scale * torch.cos(x @ self.W + self.b)


# --- Multiclass hinge loss (Crammer-Singer style) ----------------------------

class MulticlassHingeLoss(nn.Module):
    """Multiclass hinge loss: L = max(0, 1 + max_{j≠y}(s_j) - s_y).

    This is the standard loss used by SVM for multiclass problems
    (Crammer & Singer, 2001).  It penalises the margin between the
    correct class score and the highest incorrect class score.
    """
    def forward(self, scores, targets):
        n = scores.size(0)
        correct_scores = scores[torch.arange(n, device=scores.device), targets]
        # Mask correct class with -inf so it doesn't win the max
        margins = scores - correct_scores.unsqueeze(1) + 1.0
        margins[torch.arange(n, device=scores.device), targets] = 0.0
        loss = margins.clamp(min=0).max(dim=1)[0]
        return loss.mean()


# --- Helpers ------------------------------------------------------------------

def split_patients(all_users: np.ndarray, val_frac: float, seed: int = 42):
    """Patient-level train/val split (no data leakage)."""
    rng = np.random.RandomState(seed)
    users_shuffled = all_users.copy()
    rng.shuffle(users_shuffled)
    n_val = max(1, int(len(users_shuffled) * val_frac))
    return sorted(users_shuffled[n_val:].tolist()), sorted(users_shuffled[:n_val].tolist())


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Compute accuracy on a DataLoader."""
    model.eval()
    correct = 0
    total = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        preds = model(X_batch).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return correct / total if total > 0 else 0.0


# --- Main ---------------------------------------------------------------------

def train(args):
    # -- Device selection ------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        device = torch.device("cpu")
        gpu_name = "CPU (no CUDA)"
        vram_gb = 0

    print("=" * 70)
    print("  GPU RBF-SVM via Random Fourier Features  -  EMG-EPN612")
    print("=" * 70)
    print(f"\n  Device: {gpu_name}"
          + (f"  ({vram_gb:.1f} GB VRAM)" if vram_gb > 0 else ""))

    # -- 1. Load dataset -------------------------------------------------------
    if not TRAINING_FILE.exists():
        print(f"ERROR: {TRAINING_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Loading {TRAINING_FILE.name} ...")
    t0 = time.time()
    df = pd.read_parquet(TRAINING_FILE)
    print(f"  {len(df):,} rows x {len(df.columns)} cols  ({time.time()-t0:.1f}s)")

    # -- 2. Patient-level split ------------------------------------------------
    all_users = df["user"].unique()
    train_users, val_users = split_patients(all_users, args.val_frac)
    print(f"\n  Patients: {len(all_users)} total  "
          f"({len(train_users)} train / {len(val_users)} val)")

    # -- 3. Labels -------------------------------------------------------------
    le = LabelEncoder()
    le.fit(ALL_LABELS)
    print(f"  Classes ({N_CLASSES}): {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # -- 4. Split arrays -------------------------------------------------------
    train_mask = df["user"].isin(set(train_users))
    X_train_np = df.loc[train_mask, FEATURE_COLS].values.astype(np.float32)
    y_train_np = le.transform(df.loc[train_mask, "label"].values)
    X_val_np   = df.loc[~train_mask, FEATURE_COLS].values.astype(np.float32)
    y_val_np   = le.transform(df.loc[~train_mask, "label"].values)
    del df; gc.collect()
    print(f"  Train: {len(y_train_np):,}  |  Val: {len(y_val_np):,}")

    # -- 5. StandardScaler (on CPU) --------------------------------------------
    print("\n  Fitting scaler ...")
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_val_np   = scaler.transform(X_val_np)

    # -- 6. Determine gamma ----------------------------------------------------
    if args.gamma == "scale":
        gamma_val = 1.0 / (N_FEATURES * X_train_np.var())
    else:
        gamma_val = float(args.gamma)
    print(f"  gamma = {gamma_val:.6f}")

    # -- 7. Compute balanced class weights -------------------------------------
    counts = np.bincount(y_train_np, minlength=N_CLASSES).astype(np.float32)
    weights = len(y_train_np) / (N_CLASSES * counts)
    class_weights = torch.tensor(weights, device=device)
    print(f"  Class weights: {dict(zip(le.classes_, [f'{w:.3f}' for w in weights]))}")

    # -- 8. Build DataLoaders --------------------------------------------------
    X_train_t = torch.from_numpy(X_train_np); del X_train_np
    y_train_t = torch.from_numpy(y_train_np.astype(np.int64)); del y_train_np
    X_val_t   = torch.from_numpy(X_val_np);   del X_val_np
    y_val_t   = torch.from_numpy(y_val_np.astype(np.int64));   del y_val_np
    gc.collect()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t, y_val_t)

    # pin_memory=True enables async CPU→GPU transfer
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, pin_memory=pin,
                              num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2,
                              shuffle=False, pin_memory=pin,
                              num_workers=0)

    # -- 9. Build model --------------------------------------------------------
    #
    #   Input (72) → RFF (D, fixed) → Linear (D → 6)
    #
    # The RFF layer is frozen.  Only the linear layer is trained.
    # This is mathematically equivalent to an RBF-SVM.
    #
    rff = RandomFourierFeatures(N_FEATURES, args.D, gamma_val)
    linear = nn.Linear(args.D, N_CLASSES)

    model = nn.Sequential(rff, linear).to(device)

    # Weight decay on the linear layer = L2 regularization = 1/(2*C*n)
    weight_decay = 1.0 / (2.0 * args.C * len(train_ds))
    optimizer = torch.optim.Adam(
        linear.parameters(),       # only train the linear layer
        lr=args.lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler: reduce LR by 0.5× when val_acc plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6,
        verbose=True,
    )

    criterion = MulticlassHingeLoss()

    print(f"\n  Model:")
    print(f"    RFF dimension (D)  : {args.D:,}")
    print(f"    gamma (RBF)        : {gamma_val:.6f}")
    print(f"    C (regularization) : {args.C}")
    print(f"    weight_decay       : {weight_decay:.2e}")
    print(f"    lr                 : {args.lr}")
    print(f"    batch_size         : {args.batch_size}")
    print(f"    epochs             : {args.epochs}")
    rff_mb = args.D * N_FEATURES * 4 / 1024**2
    print(f"    RFF W matrix       : {rff_mb:.0f} MB")
    print()
    print("-" * 70)

    # -- 10. Training loop -----------------------------------------------------
    best_acc = 0.0
    best_state = None
    t0_total = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0_epoch = time.time()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # Forward: RFF transform + linear classifier
            scores = model(X_batch)

            # Weighted hinge loss: scale per-sample loss by class weight
            batch_weights = class_weights[y_batch]

            n = scores.size(0)
            correct_scores = scores[torch.arange(n, device=device), y_batch]
            margins = scores - correct_scores.unsqueeze(1) + 1.0
            margins[torch.arange(n, device=device), y_batch] = 0.0
            per_sample_loss = margins.clamp(min=0).max(dim=1)[0]
            loss = (per_sample_loss * batch_weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        elapsed = time.time() - t0_epoch

        # Validate
        val_acc = evaluate(model, val_loader, device)

        scheduler.step(val_acc)   # ReduceLROnPlateau uses val metric

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            marker = ""

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:2d}/{args.epochs}  |  "
              f"loss: {avg_loss:.4f}  |  val_acc: {val_acc:.4f}  |  "
              f"lr: {lr_now:.5f}  |  {elapsed:.1f}s{marker}")

    total_time = time.time() - t0_total
    print(f"\n  Training completed in {total_time:.1f}s")
    print(f"  Best validation accuracy: {best_acc:.4f}")

    # -- 11. Final evaluation with best model ----------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    print("\n  Final evaluation on validation set ...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    final_acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
    )

    # -- 12. Save models -------------------------------------------------------
    # Save PyTorch model (RFF + linear)
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "gamma": gamma_val,
        "D": args.D,
        "n_features": N_FEATURES,
        "n_classes": N_CLASSES,
    }, MODELS_DIR / "svm_gpu_model.pt")

    joblib.dump(scaler, MODELS_DIR / "svm_gpu_scaler.joblib")
    joblib.dump(le, MODELS_DIR / "svm_gpu_label_encoder.joblib")

    # -- 13. Summary -----------------------------------------------------------
    print()
    print("-" * 70)
    print(f"  Best validation accuracy: {best_acc:.4f}")
    print()
    print("  Classification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix:")
    print(f"  {'':>12s}", "  ".join(f"{c:>10s}" for c in le.classes_))
    for i, row_vals in enumerate(cm):
        print(f"  {le.classes_[i]:>12s}", "  ".join(f"{v:10d}" for v in row_vals))

    history = {
        "model": "GPU RFF-SVM",
        "D": args.D,
        "gamma": gamma_val,
        "C": args.C,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_acc": float(best_acc),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "train_time_s": round(total_time, 1),
        "device": str(device),
    }
    history_path = MODELS_DIR / "svm_gpu_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  History: {history_path}")
    print(f"  Models : {MODELS_DIR.resolve()}")
    print("=" * 70)


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a GPU-accelerated RBF-SVM via Random Fourier Features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--D", type=int, default=DEFAULT_D,
                   help="Number of Random Fourier Features (RFF dimension).")
    p.add_argument("--gamma", type=str, default=DEFAULT_GAMMA,
                   help="RBF bandwidth. 'scale' = 1/(d*Var(X)), or a float.")
    p.add_argument("--C", type=float, default=DEFAULT_C,
                   help="SVM regularization (inverse of weight decay).")
    p.add_argument("--lr", type=float, default=DEFAULT_LR,
                   help="Adam learning rate.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="Mini-batch size.")
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                   help="Fraction of patients for validation.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
