"""
DTW-KNN with k-Medoids Template Generation for EMG-EPN612 dataset.

Two-phase algorithm:
  Phase 1 (Offline — Template Generation):
    1. Load the DTW-ready dataset (preprocessed, feature-extracted,
       outlier-cleaned, subject-normalised time series).
    2. Group repetitions by gesture class.
    3. Run k-Medoids clustering (with DTW metric) inside each class
       to discover representative "Master Templates" (Medoids).
    4. Store the Medoid templates as the Reference Set.

  Phase 2 (Online — Classification):
    1. For each test repetition, compute the Dependent DTW distance
       to every Medoid in the Reference Set.
    2. Classify via 1-NN (nearest Medoid vote).

Hyperparameter search (grid):
    - sakoe_chiba_radius (r):  controls max time warping
    - n_clusters_per_class:    number of Medoid templates per gesture
    - dtw_metric:              local cost function ("euclidean" / "cityblock")

Dependent (multivariate) DTW forces all 72 features to warp together,
preserving physical channel correlations.

Usage:
    cd "EMG-EPN612 project"
    python scripts/train_dtw_knn.py
    python scripts/train_dtw_knn.py --n-clusters 3 --radius auto --max-reps 500
    python scripts/train_dtw_knn.py --n-clusters 5 --max-reps 0 --pca 20 --clara-samples 5
    python scripts/train_dtw_knn.py --grid-search
"""

import sys
import os
import time
import json
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# tslearn for DTW-based distance computation and classification
from tslearn.metrics import cdist_dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset

# --- Project paths ------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DTW_TRAIN_FILE = PROJECT_ROOT / "preprocessed_output" / "dataset_DTW_TRAINING.parquet"
DTW_TEST_FILE  = PROJECT_ROOT / "preprocessed_output" / "dataset_DTW_TEST.parquet"
MODELS_DIR     = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# --- Dataset constants --------------------------------------------------------
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES  = len(ALL_LABELS)

CHANNELS     = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES    = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
N_FEATURES   = len(FEATURE_COLS)  # 72

# --- Defaults -----------------------------------------------------------------
DEFAULT_N_CLUSTERS    = 3        # medoids per class
DEFAULT_RADIUS        = "auto"   # "auto" = calculated safe radius
DEFAULT_VAL_FRAC      = 0.15     # fraction of patients for validation
DEFAULT_MAX_REPS      = 0        # 0 = use all (WARNING: very slow)
DEFAULT_BUFFER_RATIO  = 0.10     # 10% buffer for safe radius calculation
DEFAULT_PCA_COMPONENTS = 0       # 0 = no PCA; e.g. 20 retains ~95% variance
DEFAULT_CLARA_SAMPLES  = 0       # 0 = full PAM (original); 5 = use CLARA
DEFAULT_CLARA_SAMPLE_SIZE = 0    # 0 = auto  =  min(n, max(40+2k, 200))


# --- Utility ------------------------------------------------------------------

def _format_duration(seconds):
    """Format seconds into human-readable HH:MM:SS or MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


# ==============================================================================
# STEP 1 — Data Loading & Reshaping
# ==============================================================================

def load_and_reshape(parquet_path: Path, feature_cols: list):
    """Load the flat DTW parquet and reshape into 3D time-series sequences.

    Each unique (user, sample_id) group is one repetition/sequence.

    Returns
    -------
    sequences : list of np.ndarray
        Each element has shape (T_i, 72) where T_i varies per repetition.
    labels : np.ndarray of str
        Gesture label for each sequence.
    users : np.ndarray of str
        User identifier for each sequence.
    rep_ids : list of str
        The sample_id for each sequence.
    """
    print(f"  Loading {parquet_path.name} ...")
    t0 = time.time()
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df):,} rows x {len(df.columns)} cols  ({time.time()-t0:.1f}s)")

    # Sort by user, sample_id, window_idx, miniwindow_idx
    t_sort = time.time()
    sort_cols = ["user", "sample_id", "window_idx"]
    if "miniwindow_idx" in df.columns:
        sort_cols.append("miniwindow_idx")
    print(f"  Sorting by {sort_cols} ...")
    df = df.sort_values(sort_cols)
    print(f"  Sorted  ({time.time()-t_sort:.1f}s)")

    print("  Reshaping into 3D sequences ...")
    t0 = time.time()

    sequences = []
    labels = []
    users = []
    rep_ids = []

    grouped = df.groupby(["user", "sample_id"], sort=False)
    n_groups = len(grouped)
    for i, ((user, sample_id), group) in enumerate(grouped):
        seq = group[feature_cols].values.astype(np.float32)  # (T_i, 72)
        label = group["label"].iloc[0]
        sequences.append(seq)
        labels.append(label)
        users.append(user)
        rep_ids.append(sample_id)

        # Progress every 10%
        if (i + 1) % max(1, n_groups // 10) == 0 or (i + 1) == n_groups:
            pct = 100 * (i + 1) / n_groups
            elapsed_so_far = time.time() - t0
            print(f"    [{pct:5.1f}%] {i+1:,}/{n_groups:,} sequences  "
                  f"({elapsed_so_far:.1f}s)", flush=True)

    labels = np.array(labels)
    users = np.array(users)
    lengths = np.array([len(s) for s in sequences])

    elapsed = time.time() - t0
    print(f"  {len(sequences):,} sequences reshaped  ({elapsed:.1f}s)")
    print(f"  Sequence lengths: min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}, std={lengths.std():.1f}")
    # Memory estimate
    total_windows = int(lengths.sum())
    mem_mb = total_windows * N_FEATURES * 4 / 1024**2
    print(f"  Total windows: {total_windows:,}  (~{mem_mb:.0f} MB float32)")

    return sequences, labels, users, rep_ids


# ==============================================================================
# STEP 2 — Patient-Level Train / Val Split
# ==============================================================================

def split_patients(users: np.ndarray, val_frac: float, seed: int = 42):
    """Patient-level split to avoid data leakage."""
    unique_users = np.unique(users)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_users)
    n_val = max(1, int(len(unique_users) * val_frac))
    val_set = set(unique_users[:n_val])
    train_set = set(unique_users[n_val:])
    return train_set, val_set


def apply_split(sequences, labels, users, train_set, val_set):
    """Split sequences into train and val sets based on user membership."""
    train_idx = [i for i, u in enumerate(users) if u in train_set]
    val_idx   = [i for i, u in enumerate(users) if u in val_set]

    X_train = [sequences[i] for i in train_idx]
    y_train = labels[train_idx]
    X_val   = [sequences[i] for i in val_idx]
    y_val   = labels[val_idx]

    return X_train, y_train, X_val, y_val


# ==============================================================================
# STEP 3 — Safe Sakoe-Chiba Radius Calculation
# ==============================================================================

def calculate_safe_radius(sequences, buffer_ratio=DEFAULT_BUFFER_RATIO):
    """Calculate a global Sakoe-Chiba radius that guarantees reachability.

    r = (max_len - min_len) + int(avg_len * buffer_ratio)

    This ensures the DTW warping path can reach the bottom-right corner
    of the cost matrix even for the most extreme length pair.
    """
    lengths = np.array([len(s) for s in sequences])
    min_len = int(lengths.min())
    max_len = int(lengths.max())
    avg_len = float(lengths.mean())

    max_diff = max_len - min_len
    warp_buffer = int(avg_len * buffer_ratio)
    safe_radius = max_diff + warp_buffer

    print(f"\n  Sakoe-Chiba radius calculation:")
    print(f"    Lengths: min={min_len}, max={max_len}, avg={avg_len:.1f}")
    print(f"    max_diff={max_diff} + buffer={warp_buffer} → r={safe_radius}")

    return safe_radius


# ==============================================================================
# STEP 4 — Subsample (for tractability)
# ==============================================================================

def subsample_per_class(X, y, max_per_class, seed=42):
    """Randomly subsample to at most max_per_class repetitions per label.

    k-Medoids with DTW is O(n² * T²) per class per iteration,
    so we need to limit n for tractability.
    """
    if max_per_class <= 0:
        return X, y  # no limit

    rng = np.random.RandomState(seed)
    selected_X = []
    selected_y = []

    for label in np.unique(y):
        idxs = np.where(y == label)[0]
        if len(idxs) > max_per_class:
            chosen = rng.choice(idxs, max_per_class, replace=False)
            print(f"    {label:>12s}: {len(idxs):5d} → {max_per_class} (subsampled)")
        else:
            chosen = idxs
            print(f"    {label:>12s}: {len(idxs):5d} → {len(chosen)} (kept all)")
        selected_X.extend([X[i] for i in chosen])
        selected_y.extend([y[i] for i in chosen])

    return selected_X, np.array(selected_y)


# ==============================================================================
# Dimensionality Reduction (optional)
# ==============================================================================

def apply_pca(sequences, n_components, pca=None):
    """Apply PCA to reduce feature dimensionality of time-series windows.

    Fits PCA on all stacked windows (if pca is None), then transforms
    each sequence independently.  Reduces per-pair DTW cost linearly
    (72 -> n_components features per window).

    Parameters
    ----------
    sequences : list of np.ndarray, each (T_i, D)
    n_components : int
        Target dimensionality.
    pca : PCA or None
        Pre-fitted PCA object (for val data). If None, fits on sequences.

    Returns
    -------
    transformed : list of np.ndarray, each (T_i, n_components)
    pca : PCA
        The fitted PCA object.
    """
    all_windows = np.vstack(sequences)  # (total_windows, D)
    original_dim = all_windows.shape[1]

    if pca is None:
        n_components = min(n_components, original_dim, all_windows.shape[0])
        pca = PCA(n_components=n_components)
        pca.fit(all_windows)
        var_kept = float(pca.explained_variance_ratio_.sum())
        print(f"  PCA fitted: {original_dim} -> {n_components} features  "
              f"({var_kept:.1%} variance retained)")
    else:
        var_kept = float(pca.explained_variance_ratio_.sum())
        print(f"  PCA transform: {original_dim} -> {pca.n_components_} features  "
              f"({var_kept:.1%} variance retained)")

    transformed = [pca.transform(seq).astype(np.float32) for seq in sequences]
    return transformed, pca


# ==============================================================================
# PHASE 1 — Build Reference Set (k-Medoids Clustering)
# ==============================================================================

def _kmedoids_plus_plus_init(dist_matrix, k, rng):
    """k-Medoids++ initialisation (D² probability-proportional sampling).

    Similar to k-means++ but uses precomputed distance matrix.
    Selects initial medoids that are well spread out.
    """
    n = dist_matrix.shape[0]
    # First medoid: random
    medoids = [rng.randint(n)]

    for _ in range(1, k):
        # Distance from each point to its nearest existing medoid
        dists_to_medoids = dist_matrix[:, medoids].min(axis=1)
        # Probability proportional to D²
        probs = dists_to_medoids ** 2
        prob_sum = probs.sum()
        if prob_sum == 0:
            # All distances zero — pick randomly
            probs = np.ones(n) / n
        else:
            probs /= prob_sum
        new_medoid = rng.choice(n, p=probs)
        medoids.append(new_medoid)

    return medoids


def _pam(dist_matrix, k, max_iter=50, seed=42):
    """Partitioning Around Medoids (PAM) using a precomputed distance matrix.

    The medoid is the real data point within each cluster that minimises
    the sum of distances to all other cluster members — preserving the
    jagged, realistic physics of the EMG signal (unlike a barycenter).

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (n, n)
        Symmetric pairwise DTW distance matrix.
    k : int
        Number of clusters / medoids.
    max_iter : int
        Maximum PAM iterations.
    seed : int
        Random state.

    Returns
    -------
    medoid_indices : list of int
        Indices into the original data of the selected medoids.
    labels : np.ndarray
        Cluster assignment for each data point.
    inertia : float
        Sum of distances of each point to its assigned medoid.
    """
    n = dist_matrix.shape[0]
    rng = np.random.RandomState(seed)

    # --- Initialise medoids (k-Medoids++) ---
    print(f"      PAM: initialising {k} medoids (k-Medoids++) from {n} points ...",
          flush=True)
    medoids = _kmedoids_plus_plus_init(dist_matrix, k, rng)

    converged_at = max_iter
    for iteration in range(max_iter):
        # ASSIGN: each point → nearest medoid
        dists_to_medoids = dist_matrix[:, medoids]          # (n, k)
        labels = np.argmin(dists_to_medoids, axis=1)        # (n,)
        current_inertia = float(dists_to_medoids[np.arange(n), labels].sum())

        # UPDATE: for each cluster find the point that minimises
        #         the total within-cluster distance
        new_medoids = []
        cluster_sizes = []
        for c in range(k):
            cluster_mask = np.where(labels == c)[0]
            cluster_sizes.append(len(cluster_mask))
            if len(cluster_mask) == 0:
                # Empty cluster — keep old medoid
                new_medoids.append(medoids[c])
                continue
            # Sub-matrix of distances within this cluster
            sub_dist = dist_matrix[np.ix_(cluster_mask, cluster_mask)]
            # Point with smallest sum of distances
            best_local = np.argmin(sub_dist.sum(axis=1))
            new_medoids.append(cluster_mask[best_local])

        n_changed = len(set(new_medoids) - set(medoids))
        print(f"      PAM iter {iteration+1:2d}/{max_iter}: "
              f"inertia={current_inertia:.1f}  "
              f"medoids_changed={n_changed}  "
              f"cluster_sizes={cluster_sizes}", flush=True)

        # CHECK convergence
        if set(new_medoids) == set(medoids):
            converged_at = iteration + 1
            print(f"      PAM converged at iteration {converged_at}", flush=True)
            break
        medoids = new_medoids

    # Final assignment & inertia
    dists_to_medoids = dist_matrix[:, medoids]
    labels = np.argmin(dists_to_medoids, axis=1)
    inertia = float(dists_to_medoids[np.arange(n), labels].sum())

    return medoids, labels, inertia


def _clara(class_data, k, sakoe_chiba_radius=None,
           n_samples=5, sample_size=0, seed=42):
    """CLARA (Clustering LARge Applications) - scalable k-Medoids.

    Instead of computing the full n x n DTW distance matrix (infeasible
    for n > ~1000), CLARA draws multiple random samples of size s,
    runs PAM on each sample, and evaluates the resulting medoids
    against the FULL dataset.  Keeps the best solution.

    Complexity per sample:  O(s^2 * T^2 * D) for PAM  +  O(n*k * T^2 * D) for eval
    vs full PAM:            O(n^2 * T^2 * D)

    For n=6407, k=3, s=200, n_samples=5:
      CLARA ~ 5 x (200^2 + 6407*3) ~ 296K DTW computations
      PAM   ~ 41,000,000 DTW computations  -> ~138x faster
    """
    n = len(class_data)
    rng = np.random.RandomState(seed)

    # Auto sample size
    if sample_size <= 0:
        sample_size = min(n, max(40 + 2 * k, 200))
    sample_size = min(sample_size, n)

    # If sample covers everything, fall back to full PAM
    if sample_size >= n:
        print(f"      CLARA: sample_size ({sample_size}) >= n ({n}), "
              f"falling back to full PAM", flush=True)
        class_ts = to_time_series_dataset(class_data)
        dtw_kw = {"n_jobs": -1}
        if sakoe_chiba_radius is not None:
            dtw_kw["global_constraint"] = "sakoe_chiba"
            dtw_kw["sakoe_chiba_radius"] = sakoe_chiba_radius
        dist_matrix = cdist_dtw(class_ts, **dtw_kw)
        med, _, inertia = _pam(dist_matrix, k, max_iter=50, seed=seed)
        return med, inertia

    print(f"      CLARA: {n_samples} samples x {sample_size} points  "
          f"(vs full {n}x{n} = {n**2:,})", flush=True)

    best_medoids = None
    best_cost = np.inf

    # Pre-convert all sequences for evaluation step
    all_ts = to_time_series_dataset(class_data)
    dtw_kw = {"n_jobs": -1}
    if sakoe_chiba_radius is not None:
        dtw_kw["global_constraint"] = "sakoe_chiba"
        dtw_kw["sakoe_chiba_radius"] = sakoe_chiba_radius

    for s_idx in range(n_samples):
        t_s = time.time()

        # 1. Draw random sample
        sample_idx = rng.choice(n, sample_size, replace=False)
        sample_ts = to_time_series_dataset([class_data[i] for i in sample_idx])

        # 2. Compute sample x sample DTW distance matrix
        print(f"      CLARA sample {s_idx+1}/{n_samples}: "
              f"{sample_size}x{sample_size} DTW matrix ...", flush=True)
        dist_matrix = cdist_dtw(sample_ts, **dtw_kw)

        # 3. Run PAM on sample
        local_med, _, _ = _pam(dist_matrix, k, max_iter=50, seed=seed + s_idx)

        # 4. Map local indices -> global indices
        global_med = [int(sample_idx[m]) for m in local_med]

        # 5. Evaluate medoids against ALL data  (n x k DTW computations)
        med_ts = to_time_series_dataset([class_data[i] for i in global_med])
        eval_dists = cdist_dtw(all_ts, med_ts, **dtw_kw)  # (n, k)
        cost = float(eval_dists.min(axis=1).sum())

        elapsed_s = time.time() - t_s
        improved = cost < best_cost
        tag = " * BEST" if improved else ""
        print(f"      CLARA sample {s_idx+1}/{n_samples}: "
              f"cost={cost:.1f}{tag}  ({elapsed_s:.1f}s)", flush=True)

        if improved:
            best_cost = cost
            best_medoids = global_med

    return best_medoids, best_cost


def build_reference_set(X_train, y_train, n_clusters_per_class=3,
                        sakoe_chiba_radius=None, seed=42,
                        clara_samples=DEFAULT_CLARA_SAMPLES,
                        clara_sample_size=DEFAULT_CLARA_SAMPLE_SIZE):
    """Run k-Medoids (PAM / CLARA) with DTW metric inside each gesture class.

    Uses full PAM for small classes and CLARA for large classes
    (auto-selected based on class size vs sample_size).
    Stores the selected Medoid sequences as the Reference Set.

    Parameters
    ----------
    X_train : list of np.ndarray
        Each element has shape (T_i, 72).
    y_train : np.ndarray
        Labels.
    n_clusters_per_class : int
        Number of medoid templates per class.
    sakoe_chiba_radius : int or None
        Sakoe-Chiba band radius. None = unconstrained DTW.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    ref_X : np.ndarray, shape (n_classes * n_clusters, max_T, 72)
        Medoid templates (tslearn padded format with NaN).
    ref_y : np.ndarray
        Labels for each medoid.
    """
    unique_classes = np.unique(y_train)
    ref_templates = []
    ref_labels = []

    print(f"\n  Building Reference Set ({n_clusters_per_class} medoids/class) ...")
    if sakoe_chiba_radius is not None:
        print(f"  Sakoe-Chiba radius: {sakoe_chiba_radius}")
    print()

    for label in unique_classes:
        t0 = time.time()
        # Isolate sequences for this class
        class_idx = np.where(y_train == label)[0]
        class_data = [X_train[i] for i in class_idx]
        n_class = len(class_data)

        # Handle edge case: fewer samples than clusters
        actual_k = min(n_clusters_per_class, n_class)
        if actual_k < n_clusters_per_class:
            print(f"    WARNING: class '{label}' has only {n_class} samples, "
                  f"using k={actual_k}")

        class_lengths = [len(s) for s in class_data]
        print(f"    [{label:>12s}] {n_class} reps, "
              f"len range [{min(class_lengths)}-{max(class_lengths)}]",
              flush=True)

        # Decide: CLARA (scalable) vs full PAM
        auto_ss = min(n_class, max(40 + 2 * actual_k, 200))
        use_clara = (clara_samples > 0 and n_class > auto_ss)

        if use_clara:
            medoid_indices, inertia = _clara(
                class_data, actual_k,
                sakoe_chiba_radius=sakoe_chiba_radius,
                n_samples=clara_samples,
                sample_size=clara_sample_size,
                seed=seed,
            )
            method_tag = "CLARA"
        else:
            # Full PAM: compute entire n x n DTW distance matrix
            class_ts = to_time_series_dataset(class_data)
            print(f"    [{label:>12s}] computing {n_class}x{n_class} = "
                  f"{n_class**2:,} DTW distances ...", flush=True)
            t_dtw = time.time()
            dtw_kw = {"n_jobs": -1}
            if sakoe_chiba_radius is not None:
                dtw_kw["global_constraint"] = "sakoe_chiba"
                dtw_kw["sakoe_chiba_radius"] = sakoe_chiba_radius
            dist_matrix = cdist_dtw(class_ts, **dtw_kw)
            dtw_time = time.time() - t_dtw
            print(f"    [{label:>12s}] DTW matrix done  ({dtw_time:.1f}s)",
                  flush=True)
            medoid_indices, _, inertia = _pam(
                dist_matrix, actual_k, max_iter=50, seed=seed
            )
            method_tag = "PAM"

        # Store the medoid sequences (actual data points, not averages)
        for idx in medoid_indices:
            ref_templates.append(class_data[idx])
        ref_labels.extend([label] * actual_k)

        elapsed = time.time() - t0
        print(f"    Class '{label:>12s}': {n_class:5d} reps -> "
              f"{actual_k} medoids  ({method_tag}, inertia={inertia:.1f}, "
              f"total={elapsed:.1f}s)")

    # Stack into tslearn-compatible 3D array
    ref_X = to_time_series_dataset(ref_templates)
    ref_y = np.array(ref_labels)

    print(f"\n  Reference Set: {len(ref_y)} templates, "
          f"shape {ref_X.shape}")

    return ref_X, ref_y


# ==============================================================================
# PHASE 2 — Classification (DTW-KNN)
# ==============================================================================

def classify(ref_X, ref_y, X_val, y_val, le,
             n_neighbors=1, sakoe_chiba_radius=None):
    """Classify validation set using DTW-1NN against the Reference Set.

    Parameters
    ----------
    ref_X : np.ndarray
        Medoid templates in tslearn format.
    ref_y : np.ndarray
        Labels (string) for each medoid.
    X_val : list of np.ndarray
        Validation sequences.
    y_val : np.ndarray
        True labels.
    le : LabelEncoder
        For classification report.
    n_neighbors : int
        Number of nearest neighbors (default 1).
    sakoe_chiba_radius : int or None
        Sakoe-Chiba constraint on DTW.

    Returns
    -------
    accuracy : float
    y_pred : np.ndarray
    """
    # Build metric_params
    metric_params = {}
    if sakoe_chiba_radius is not None:
        metric_params["global_constraint"] = "sakoe_chiba"
        metric_params["sakoe_chiba_radius"] = sakoe_chiba_radius

    # Create KNN classifier
    clf = KNeighborsTimeSeriesClassifier(
        n_neighbors=n_neighbors,
        metric="dtw",
        metric_params=metric_params if metric_params else None,
        n_jobs=-1,   # use all CPU cores
    )

    # "Fit" the classifier (stores templates in memory)
    print(f"\n  Fitting KNN classifier with {ref_X.shape[0]} templates ...")
    clf.fit(ref_X, ref_y)

    # Convert validation data to tslearn format
    n_val = len(X_val)
    print(f"  Converting {n_val} validation sequences to tslearn format ...")
    t_conv = time.time()
    X_val_ts = to_time_series_dataset(X_val)
    print(f"  Conversion done  ({time.time()-t_conv:.1f}s)")

    n_templates = ref_X.shape[0]
    total_dtw = n_val * n_templates
    print(f"  Classifying {n_val} sequences (DTW-{n_neighbors}NN "
          f"vs {n_templates} templates = "
          f"{total_dtw:,} DTW computations) ...")
    print(f"  This may take a while ...", flush=True)

    # Predict in batches so we can report progress
    BATCH = max(1, min(200, n_val // 10))  # ~10 progress updates
    t0 = time.time()
    y_pred_parts = []
    for start in range(0, n_val, BATCH):
        end = min(start + BATCH, n_val)
        batch_pred = clf.predict(X_val_ts[start:end])
        y_pred_parts.append(batch_pred)

        done = end
        elapsed_so_far = time.time() - t0
        speed_so_far = done / elapsed_so_far if elapsed_so_far > 0 else 0
        remaining = (n_val - done) / speed_so_far if speed_so_far > 0 else 0
        pct = 100 * done / n_val
        print(f"    [{pct:5.1f}%] {done:,}/{n_val:,}  "
              f"({elapsed_so_far:.1f}s elapsed, "
              f"~{_format_duration(remaining)} remaining, "
              f"{speed_so_far:.1f} reps/s)", flush=True)

    y_pred = np.concatenate(y_pred_parts)
    elapsed = time.time() - t0

    acc = accuracy_score(y_val, y_pred)
    speed = n_val / elapsed if elapsed > 0 else 0
    print(f"  Classification done: accuracy={acc:.4f}  ({elapsed:.1f}s, "
          f"{speed:.1f} reps/s)")

    return acc, y_pred


# ==============================================================================
# Grid Search
# ==============================================================================

def run_grid_search(X_train, y_train, X_val, y_val, le,
                    safe_radius, args):
    """Run hyperparameter grid search over n_clusters and radius.

    Evaluates all combinations and returns results sorted by accuracy.
    """
    # Define search grid
    cluster_grid = [1, 3, 5]
    # Radius grid: from safe_radius down to ~60% of safe_radius, step of 3-5
    r_min = max(1, int(safe_radius * 0.6))
    r_step = max(1, (safe_radius - r_min) // 4)
    radius_grid = list(range(r_min, safe_radius + r_step, r_step))
    # Make sure safe_radius is included
    if safe_radius not in radius_grid:
        radius_grid.append(safe_radius)
    radius_grid = sorted(set(radius_grid))

    total = len(cluster_grid) * len(radius_grid)
    print(f"\n{'='*70}")
    print(f"  GRID SEARCH: {total} configurations")
    print(f"  n_clusters : {cluster_grid}")
    print(f"  radius     : {radius_grid}")
    print(f"{'='*70}\n")

    results = []
    run_idx = 0
    t_grid = time.time()

    for n_clust in cluster_grid:
        for radius in radius_grid:
            run_idx += 1
            grid_elapsed = time.time() - t_grid
            if run_idx > 1 and grid_elapsed > 0:
                avg_per_run = grid_elapsed / (run_idx - 1)
                remaining = avg_per_run * (total - run_idx + 1)
                eta_str = f"  ETA: {_format_duration(remaining)}"
            else:
                eta_str = ""
            print(f"\n  --- Run {run_idx}/{total}: "
                  f"n_clusters={n_clust}, radius={radius}{eta_str} ---")

            try:
                # Phase 1: Build reference set
                ref_X, ref_y = build_reference_set(
                    X_train, y_train,
                    n_clusters_per_class=n_clust,
                    sakoe_chiba_radius=radius,
                    seed=42,
                    clara_samples=args.clara_samples,
                    clara_sample_size=args.clara_sample_size,
                )

                # Phase 2: Classify
                acc, y_pred = classify(
                    ref_X, ref_y, X_val, y_val, le,
                    n_neighbors=1,
                    sakoe_chiba_radius=radius,
                )

                results.append({
                    "n_clusters": n_clust,
                    "radius": radius,
                    "accuracy": float(acc),
                    "n_templates": len(ref_y),
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "n_clusters": n_clust,
                    "radius": radius,
                    "accuracy": 0.0,
                    "error": str(e),
                })

    # Sort by accuracy
    results.sort(key=lambda x: x.get("accuracy", 0), reverse=True)

    print(f"\n{'='*70}")
    print("  GRID SEARCH RESULTS (sorted by accuracy):")
    print(f"{'='*70}")
    for i, r in enumerate(results):
        marker = " ← BEST" if i == 0 else ""
        err = f"  ERROR: {r['error']}" if "error" in r else ""
        print(f"  {i+1:2d}. clusters={r['n_clusters']}  radius={r['radius']:3d}  "
              f"acc={r['accuracy']:.4f}{err}{marker}")

    return results


# ==============================================================================
# Main
# ==============================================================================


def train(args):
    t_pipeline = time.time()
    print("=" * 70)
    print("  DTW-KNN with k-Medoids  —  EMG-EPN612")
    print("=" * 70)
    print(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config: n_clusters={args.n_clusters}, radius={args.radius}, "
          f"max_reps={args.max_reps}, "
          f"pca={args.pca}, clara_samples={args.clara_samples}, "
          f"grid_search={args.grid_search}")

    # -- 1. Load & reshape -----------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  STEP 1/6: Loading & reshaping data")
    print(f"{'─'*70}")
    for fpath in (DTW_TRAIN_FILE, DTW_TEST_FILE):
        if not fpath.exists():
            print(f"  ERROR: {fpath} not found.", file=sys.stderr)
            print(f"  Run  python scripts/preprocess_dtw_pipeline.py  first.",
                  file=sys.stderr)
            sys.exit(1)

    print("  --- Training set ---")
    X_train, y_train, users_train, _ = load_and_reshape(
        DTW_TRAIN_FILE, FEATURE_COLS)
    print("  --- Test set ---")
    X_val, y_val, users_val, _ = load_and_reshape(
        DTW_TEST_FILE, FEATURE_COLS)

    # -- 2. Label encoder ------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  STEP 2/6: Encoding labels")
    print(f"{'─'*70}")
    le = LabelEncoder()
    le.fit(ALL_LABELS)
    print(f"  Classes ({N_CLASSES}): "
          f"{dict(zip(le.classes_, le.transform(le.classes_)))}")

    n_train_users = len(np.unique(users_train))
    n_val_users   = len(np.unique(users_val))
    print(f"\n  Patients: {n_train_users + n_val_users} total  "
          f"({n_train_users} train / {n_val_users} test)")
    print(f"  Repetitions: {len(y_train):,} train / {len(y_val):,} test")

    # -- 3. Report class distribution ------------------------------------------
    print("\n  Train/test class distribution:")
    for lbl in ALL_LABELS:
        n_train_lbl = np.sum(y_train == lbl)
        n_val_lbl = np.sum(y_val == lbl)
        print(f"    {lbl:>12s}: {n_train_lbl:5d} train / {n_val_lbl:5d} test")

    # -- 4. Subsample for tractability -----------------------------------------
    print(f"\n{'─'*70}")
    print(f"  STEP 3/6: Subsampling for tractability")
    print(f"{'─'*70}")
    if args.max_reps > 0:
        print(f"\n  Subsampling to {args.max_reps} reps/class for clustering ...")
        X_train_sub, y_train_sub = subsample_per_class(
            X_train, y_train, args.max_reps, seed=42
        )
        print(f"  Subsampled: {len(y_train_sub):,} training repetitions")
    else:
        X_train_sub, y_train_sub = X_train, y_train
        print(f"  No subsampling — using all {len(y_train_sub):,} training repetitions")
        est_pairs = 0
        for lbl in np.unique(y_train_sub):
            n_lbl = np.sum(y_train_sub == lbl)
            est_pairs += n_lbl ** 2
        print(f"  WARNING: Total DTW pairs to compute: ~{est_pairs:,}  "
              f"(this may be very slow!)")
        if args.clara_samples == 0:
            print(f"  TIP: Use --clara-samples 5 for scalable CLARA, "
                  f"or --max-reps 500 to subsample.")

    # -- 4b. Optional PCA dimensionality reduction ----------------------------
    pca_obj = None
    if args.pca > 0:
        print(f"\n{'─'*70}")
        print(f"  STEP 3b/6: PCA dimensionality reduction ({N_FEATURES} -> {args.pca})")
        print(f"{'─'*70}")
        X_train_sub, pca_obj = apply_pca(X_train_sub, args.pca, pca=None)
        X_val, _ = apply_pca(X_val, args.pca, pca=pca_obj)
        print(f"  Train sequences: {len(X_train_sub)}, "
              f"feature dim: {X_train_sub[0].shape[1]}")
        print(f"  Test  sequences: {len(X_val)}, "
              f"feature dim: {X_val[0].shape[1]}")

    # -- 5. Calculate safe Sakoe-Chiba radius ----------------------------------
    print(f"\n{'─'*70}")
    print(f"  STEP 4/6: Calculating Sakoe-Chiba radius")
    print(f"{'─'*70}")
    all_seqs = X_train_sub + X_val
    safe_radius = calculate_safe_radius(all_seqs)

    if args.radius == "auto":
        radius = safe_radius
        print(f"  Using auto radius: {radius}")
    else:
        radius = int(args.radius)
        if radius < safe_radius:
            print(f"  WARNING: Requested radius {radius} < safe radius "
                  f"{safe_radius}. Some DTW pairs may fail.")
        print(f"  Using manual radius: {radius}")

    # -- 6. Run ----------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  STEP 5/6: {'Grid search' if args.grid_search else 'Training (single run)'}")
    print(f"{'─'*70}")
    t_train = time.time()
    if args.grid_search:
        # Grid search mode
        results = run_grid_search(
            X_train_sub, y_train_sub, X_val, y_val, le,
            safe_radius, args
        )

        # Use best configuration for final evaluation
        best = results[0]
        best_n_clusters = best["n_clusters"]
        best_radius = best["radius"]
        print(f"\n  Best config: n_clusters={best_n_clusters}, "
              f"radius={best_radius}, acc={best['accuracy']:.4f}")

        # Save grid search results
        grid_path = MODELS_DIR / "dtw_knn_grid_search.json"
        with open(grid_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Grid results saved: {grid_path}")

        # Re-build reference set with best params for final save
        ref_X, ref_y = build_reference_set(
            X_train_sub, y_train_sub,
            n_clusters_per_class=best_n_clusters,
            sakoe_chiba_radius=best_radius,
            seed=42,
            clara_samples=args.clara_samples,
            clara_sample_size=args.clara_sample_size,
        )
        final_acc, y_pred = classify(
            ref_X, ref_y, X_val, y_val, le,
            n_neighbors=1,
            sakoe_chiba_radius=best_radius,
        )
        final_radius = best_radius
        final_n_clusters = best_n_clusters

    else:
        # Single run mode
        ref_X, ref_y = build_reference_set(
            X_train_sub, y_train_sub,
            n_clusters_per_class=args.n_clusters,
            sakoe_chiba_radius=radius,
            seed=42,
            clara_samples=args.clara_samples,
            clara_sample_size=args.clara_sample_size,
        )
        final_acc, y_pred = classify(
            ref_X, ref_y, X_val, y_val, le,
            n_neighbors=1,
            sakoe_chiba_radius=radius,
        )
        final_radius = radius
        final_n_clusters = args.n_clusters

    train_elapsed = time.time() - t_train
    print(f"\n  Phase 1+2 completed in {_format_duration(train_elapsed)}")

    # -- 7. Final report -------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  STEP 6/6: Results & saving")
    print(f"{'─'*70}")
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  n_clusters_per_class: {final_n_clusters}")
    print(f"  sakoe_chiba_radius: {final_radius}")
    print(f"  n_neighbors: 1")
    print(f"  Reference set: {ref_X.shape}")
    print()

    report = classification_report(
        y_val, y_pred,
        target_names=ALL_LABELS,
        digits=4,
        zero_division=0,
    )
    print("  Classification Report:")
    print(report)

    cm = confusion_matrix(y_val, y_pred, labels=ALL_LABELS)
    print("  Confusion Matrix:")
    print(f"  {'':>12s}", "  ".join(f"{c:>10s}" for c in ALL_LABELS))
    for i, row_vals in enumerate(cm):
        print(f"  {ALL_LABELS[i]:>12s}", "  ".join(f"{v:10d}" for v in row_vals))

    # -- 9. Save model artifacts -----------------------------------------------
    print(f"\n  Saving model artifacts ...")

    # Save reference set (medoid templates + labels)
    joblib.dump({
        "ref_X": ref_X,
        "ref_y": ref_y,
        "sakoe_chiba_radius": final_radius,
        "n_clusters_per_class": final_n_clusters,
        "n_neighbors": 1,
    }, MODELS_DIR / "dtw_knn_reference_set.joblib")

    # Save label encoder
    joblib.dump(le, MODELS_DIR / "dtw_knn_label_encoder.joblib")

    # Save PCA (if used)
    if pca_obj is not None:
        joblib.dump(pca_obj, MODELS_DIR / "dtw_knn_pca.joblib")
        print(f"  PCA saved: dtw_knn_pca.joblib")

    # Save training history / metadata
    history = {
        "model": "DTW-KNN with k-Medoids",
        "n_clusters_per_class": final_n_clusters,
        "sakoe_chiba_radius": final_radius,
        "n_neighbors": 1,
        "accuracy": float(final_acc),
        "n_templates": int(ref_X.shape[0]),
        "template_shape": list(ref_X.shape),
        "n_train_reps": len(y_train),
        "n_test_reps": len(y_val),
        "n_train_reps_used": len(y_train_sub),
        "max_reps_per_class": args.max_reps if args.max_reps > 0 else "all",
        "pca_components": args.pca if args.pca > 0 else "none",
        "clara_samples": args.clara_samples,
        "n_train_patients": n_train_users,
        "n_test_patients": n_val_users,
    }
    history_path = MODELS_DIR / "dtw_knn_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    total_elapsed = time.time() - t_pipeline
    print(f"\n  History : {history_path}")
    print(f"  Models  : {MODELS_DIR.resolve()}")
    print(f"\n  Total pipeline time: {_format_duration(total_elapsed)}")
    print(f"  Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a DTW-KNN classifier with k-Medoids template generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS,
                   help="Number of medoid templates per gesture class.")
    p.add_argument("--radius", type=str, default=DEFAULT_RADIUS,
                   help="Sakoe-Chiba radius. 'auto' = calculated safe radius, "
                        "or an integer value.")
    p.add_argument("--max-reps", type=int, default=DEFAULT_MAX_REPS,
                   help="Max repetitions per class for clustering (0 = all). "
                        "k-Medoids with DTW is O(n^2), so limit this for "
                        "tractability (e.g. 200-500).")
    p.add_argument("--buffer-ratio", type=float, default=DEFAULT_BUFFER_RATIO,
                   help="Buffer ratio for safe radius (fraction of avg length).")
    p.add_argument("--pca", type=int, default=DEFAULT_PCA_COMPONENTS,
                   help="PCA components for dimensionality reduction (0 = off). "
                        "Reduces DTW cost linearly (72->N). Try 15-25.")
    p.add_argument("--clara-samples", type=int, default=DEFAULT_CLARA_SAMPLES,
                   help="CLARA random restarts for scalable k-Medoids. "
                        "0 = force full PAM (O(n^2) distance matrix).")
    p.add_argument("--clara-sample-size", type=int, default=DEFAULT_CLARA_SAMPLE_SIZE,
                   help="CLARA sample size per restart (0 = auto ~ 200). "
                        "Only used when class size > sample_size.")
    p.add_argument("--grid-search", action="store_true",
                   help="Run hyperparameter grid search.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
