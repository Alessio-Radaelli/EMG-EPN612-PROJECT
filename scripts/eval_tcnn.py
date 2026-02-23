import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from tdcnn_eca import TDCNNClassifier

# ══════════════════════════════════════════════════════════════════════════════
# Configuration & Paths
# ══════════════════════════════════════════════════════════════════════════════
TEST_DATA_PATH = Path("preprocessed_output") / "dataset_TESTING.npz"
MODEL_PATH = Path("models") / "tdcnn_emg_model.pth"
ENCODER_PATH = Path("models") / "label_encoder.pkl"

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

    # 4. Run Inference
    print("Running predictions on the test set. This may take a moment...")
    y_pred = model.predict(X_test)

    # 5. Generate Classification Report
    print("\n" + "═"*60)
    print("CLASSIFICATION REPORT")
    print("═"*60)
    # The classification report calculates precision, recall, and F1-score for each gesture
    print(classification_report(y_true, y_pred, target_names=le.classes_))

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