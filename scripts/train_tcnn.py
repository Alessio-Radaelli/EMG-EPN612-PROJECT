import os
import pickle
import numpy as np
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor

# Import your custom model wrapper
from tdcnn_eca import TDCNNClassifier

# ══════════════════════════════════════════════════════════════════════════════
# Configuration & Paths
# ══════════════════════════════════════════════════════════════════════════════
DATASET_A_PATH = Path("datasets") / "dataset_A.pkl"
DATASET_B_PATH = Path("datasets") / "dataset_B.pkl"
MODEL_SAVE_PATH = Path("models") / "tdcnn_emg_model.pth"
OUTPUT_DIR = Path("preprocessed_output")

FS = 200               # Sampling frequency
WINDOW_LENGTH = 40     # 200 ms window
WINDOW_SHIFT = 10      # 50 ms step
NO_GESTURE_CROP_FALLBACK = int(1.3 * FS)
CHANNELS = [f"ch{i}" for i in range(1, 9)]

# Use a fixed number of workers to avoid memory issues
NUM_WORKERS = 4

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Signal Conditioning & Segmentation
# ══════════════════════════════════════════════════════════════════════════════
def filter_emg(emg_signal, fs=FS, lowcut=20, highcut=95, notch_freq=50, notch_q=30):
    """Bandpass and notch filter."""
    nyq = fs / 2
    b_bp, a_bp = signal.butter(2, [lowcut / nyq, highcut / nyq], btype="band")
    filtered = signal.filtfilt(b_bp, a_bp, emg_signal)
    
    b_notch, a_notch = signal.iirnotch(w0=notch_freq, Q=notch_q, fs=fs)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)
    
    return filtered

def segment_trial(filtered_channels, sample_meta, no_gesture_crop):
    """Crop signal to ground truth or center crop for noGesture."""
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

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Raw Windowing (No Features)
# ══════════════════════════════════════════════════════════════════════════════
def extract_raw_windows(cropped_signals):
    """Slices the segmented channels into overlapping temporal windows."""
    sig_matrix = np.column_stack([cropped_signals[ch] for ch in CHANNELS])
    num_samples = sig_matrix.shape[0]
    
    windows = []
    for start in range(0, num_samples - WINDOW_LENGTH + 1, WINDOW_SHIFT):
        windows.append(sig_matrix[start : start + WINDOW_LENGTH, :])
        
    return windows

# ══════════════════════════════════════════════════════════════════════════════
# Worker Function for Parallel Processing
# ══════════════════════════════════════════════════════════════════════════════
def process_single_user(user_tuple):
    """Processes a single user's data. Must be top-level for Multiprocessing."""
    user_id, user_data = user_tuple
    samples = user_data.get('trainingSamples', user_data) 
    
    user_windows, user_labels = [], []
    
    # Calculate dynamic fallback length for noGesture
    gesture_lengths = [s["groundTruthIndex"][1] - s["groundTruthIndex"][0] + 1 
                       for s in samples.values() if s["gestureName"] != "noGesture" and "groundTruthIndex" in s]
    median_gesture_len = int(np.median(gesture_lengths)) if gesture_lengths else NO_GESTURE_CROP_FALLBACK

    for sample_key, sample in samples.items():
        emg = sample["emg"]
        gesture = sample["gestureName"]
        
        # Phase 1
        filtered = {ch: filter_emg(np.array(emg[ch])) for ch in CHANNELS}
        cropped = segment_trial(filtered, sample, median_gesture_len)
        
        # Phase 2 (Raw Windows)
        windows = extract_raw_windows(cropped)
        
        if windows:
            user_windows.extend(windows)
            user_labels.extend([gesture] * len(windows))
            
    if not user_windows:
        return np.array([]), []
        
    # Phase 5: Subject-Specific Z-Score Normalization
    user_windows = np.array(user_windows, dtype=np.float32)
    
    # Calculate mean & std per channel for THIS user
    mu = np.mean(user_windows, axis=(0, 1))
    sig = np.std(user_windows, axis=(0, 1)) + 1e-8
    
    user_windows_normalized = (user_windows - mu) / sig
    
    return user_windows_normalized, user_labels

# ══════════════════════════════════════════════════════════════════════════════
# Dataset Processing Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def process_dataset(filepath: Path):
    """Distributes raw pkl dataset processing across parallel workers."""
    print(f"Processing {filepath.name} using {NUM_WORKERS} workers...")
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
        
    all_X, all_y = [], []
    user_items = list(dataset.items())
    
    # Spin up the process pool
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(process_single_user, user_items))
        
    # Aggregate results from all workers
    for X_user, y_user in results:
        if len(X_user) > 0:
            all_X.append(X_user)
            all_y.extend(y_user)
            
    X_final = np.concatenate(all_X, axis=0) if all_X else np.array([])
    y_final = np.array(all_y)
    print(f"  -> Generated {len(X_final)} windows.")
    return X_final, y_final

# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1. Load or Preprocess Datasets
    train_save_path = OUTPUT_DIR / "dataset_TRAINING.npz"
    test_save_path = OUTPUT_DIR / "dataset_TESTING.npz"
    OUTPUT_DIR.mkdir(exist_ok=True)

    if train_save_path.exists() and test_save_path.exists():
        print("Loading preprocessed datasets...")
        train_data = np.load(train_save_path, allow_pickle=True)
        test_data = np.load(test_save_path, allow_pickle=True)
        X_train, y_train_labels = train_data['X'], train_data['y']
        X_test, y_test_labels = test_data['X'], test_data['y']
        print(f"✓ Loaded Training data from {train_save_path}")
        print(f"✓ Loaded Testing data from {test_save_path}")
    else:
        print("Preprocessing datasets from raw files...")
        X_train, y_train_labels = process_dataset(DATASET_A_PATH)
        X_test, y_test_labels = process_dataset(DATASET_B_PATH)
        print("\nSaving preprocessed datasets...")
        np.savez_compressed(train_save_path, X=X_train, y=y_train_labels)
        np.savez_compressed(test_save_path, X=X_test, y=y_test_labels)
        print(f"✓ Saved Training data to {train_save_path}")
        print(f"✓ Saved Testing data to {test_save_path}")
    
    # 3. Encode Labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_labels)
    
    # Handle potentially unseen labels in test set gracefully
    test_classes = set(y_test_labels)
    train_classes = set(le.classes_)
    if not test_classes.issubset(train_classes):
        print(f"Warning: Test set contains unseen classes: {test_classes - train_classes}")
        valid_indices = [i for i, label in enumerate(y_test_labels) if label in train_classes]
        X_test = X_test[valid_indices]
        y_test_labels = [y_test_labels[i] for i in valid_indices]
        
    y_test = le.transform(y_test_labels) 
    
    num_classes = len(le.classes_)
    print(f"\nFound {num_classes} gesture classes: {le.classes_}")
    
    # 4. Initialize and Train Model
    print("\nInitializing TDCNN Classifier...")
    model = TDCNNClassifier(
        input_channels=8,
        num_classes=num_classes,
        hidden_channels=(64, 128,256),
        kernel_size=3,
        dropout=0.4,
        learning_rate=0.001,
        batch_size=64,
        epochs=50, 
        device='auto'
    )
    
    print("Starting Training...")
    model.fit(X_train, y_train, X_test, y_test)
    
    # 5. Save the trained model
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_SAVE_PATH))
    
    # 6. Save the Label Encoder mapping
    encoder_path = MODEL_SAVE_PATH.parent / "label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"✓ Saved LabelEncoder to {encoder_path}")