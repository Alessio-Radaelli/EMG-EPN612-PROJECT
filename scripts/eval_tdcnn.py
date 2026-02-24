import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tdcnn_eca import TDCNNClassifier

# ══════════════════════════════════════════════════════════════════════════════
# Configuration & Paths
# ══════════════════════════════════════════════════════════════════════════════
TEST_DATA_PATH = Path("preprocessed_output") / "dataset_TESTING.npz"
MODEL_PATH = Path("models") / "tdcnn_emg_model.pth"
ENCODER_PATH = Path("models") / "label_encoder.pkl"
RESULTS_PATH = Path("models") / "tdcnn_test_results.json"

def main():
    # 1. Load the preprocessed testing data
    print(f"Loading testing data from {TEST_DATA_PATH}...")
    try:
        data = np.load(TEST_DATA_PATH, allow_pickle=True)
        X_test = data['X']
        y_test_labels = data['y']
    except FileNotFoundError:
        print("Error: Could not find the preprocessed test data. Did you run the training script first?")
        return

    # 2. Load the LabelEncoder
    print(f"Loading LabelEncoder from {ENCODER_PATH}...")
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)

    # Clean the test set of any labels the model wasn't trained on
    valid_indices = [i for i, label in enumerate(y_test_labels) if label in le.classes_]
    if len(valid_indices) < len(y_test_labels):
        print(f"Filtered out {len(y_test_labels) - len(valid_indices)} test samples with unseen labels.")
    
    X_test = X_test[valid_indices]
    y_test_labels = [y_test_labels[i] for i in valid_indices]
    y_true = le.transform(y_test_labels)

    # 3. Load the trained TDCNN model
    print(f"Loading model from {MODEL_PATH}...")
    model = TDCNNClassifier.load(str(MODEL_PATH))

    # 4. Run Inference (timed)
    print("Running predictions on the test set. This may take a moment...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    total_time_s = time.time() - t0

    test_accuracy = float(accuracy_score(y_true, y_pred))
    report_dict = classification_report(
        y_true, y_pred, target_names=le.classes_, digits=4, output_dict=True
    )

    # 5. Generate Classification Report
    print("\n" + "═"*60)
    print("CLASSIFICATION REPORT")
    print("═"*60)
    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=4))

    # Save metrics in same format as svm_val_test_results*.json
    n_features = int(np.prod(X_test.shape[1:])) if len(X_test.shape) > 1 else X_test.shape[1]
    results = {
        "script": "eval_tcnn",
        "n_features": n_features,
        "winner_params": {},
        "test_accuracy": test_accuracy,
        "classification_report": report_dict,
        "test_samples": len(y_true),
        "total_time_s": round(total_time_s, 1),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n✓ Results saved to {RESULTS_PATH}")

    # 6. Generate and Plot Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                cbar_kws={'label': 'Number of Predictions'})
    
    plt.title('TDCNN EMG Gesture Classification - Confusion Matrix', pad=20)
    plt.ylabel('True Gesture (Ground Truth)', fontweight='bold')
    plt.xlabel('Predicted Gesture (Model Output)', fontweight='bold')
    
    # Rotate the x-axis labels so they don't overlap
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save and show the plot
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix plot to {plot_path}")
    
    # This will open a window displaying the plot
    plt.show()

if __name__ == "__main__":
    main()