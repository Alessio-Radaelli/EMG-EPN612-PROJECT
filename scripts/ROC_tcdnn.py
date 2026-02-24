import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Import your TDCNN model wrapper
from tdcnn_eca import TDCNNClassifier

# ══════════════════════════════════════════════════════════════════════════════
# Configuration & Paths (independent of CWD)
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = PROJECT_ROOT / "preprocessed_output" / "dataset_TESTING.npz"
MODEL_PATH = PROJECT_ROOT / "models" / "tdcnn_emg_model.pth"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
PLOT_OUTPUT_PATH = PROJECT_ROOT / "ROC_TDCNN.png"

def main():
    # 1. Load the preprocessed testing data
    print(f"Loading testing data from {TEST_DATA_PATH}...")
    try:
        data = np.load(TEST_DATA_PATH, allow_pickle=True)
        X_test = data['X']
        y_test_labels = data['y']
    except FileNotFoundError:
        print("Error: Could not find the preprocessed test data. Please run the training script first.")
        return

    # 2. Load the LabelEncoder
    print(f"Loading LabelEncoder from {ENCODER_PATH}...")
    try:
        with open(ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {ENCODER_PATH}. Run train_tdcnn.py first to create it.")
        return

    # 3. Clean the test set and transform labels
    valid_indices = [i for i, label in enumerate(y_test_labels) if label in le.classes_]
    if len(valid_indices) < len(y_test_labels):
        print(f"Filtered out {len(y_test_labels) - len(valid_indices)} test samples with unseen labels.")
    
    X_test = X_test[valid_indices]
    y_test_labels = [y_test_labels[i] for i in valid_indices]
    y_true = le.transform(y_test_labels)

    # Binarize the output for multi-class ROC (One-vs-Rest)
    n_classes = len(le.classes_)
    y_test_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Handle binary classification edge case if only 2 gestures are present
    if n_classes == 2:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    # 4. Load the trained TDCNN model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = TDCNNClassifier.load(str(MODEL_PATH))
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_PATH}. Run train_tdcnn.py first to train and save the model.")
        return

    # 5. Get probability predictions
    print("Computing prediction probabilities. This may take a moment...")
    y_score = model.predict_proba(X_test)

    # 6. Compute ROC curve and ROC area for each class
    print("Calculating ROC curves and AUC...")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Calculate for each individual gesture class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 7. Plot all ROC curves
    print("Generating ROC plot...")
    plt.figure(figsize=(12, 9))

    # Set up a color cycle for individual classes
    colors = cycle(plt.cm.tab10.colors)
    
    for i, color in zip(range(n_classes), colors):
        class_name = le.classes_[i]
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {class_name} (area = {roc_auc[i]:.3f})')

    # Plot the random chance diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')

    # Formatting the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.title('Multi-class ROC Curve - TDCNN EMG Classifier', pad=20, fontsize=16, fontweight='bold')
    
    # Position legend outside the plot if there are many classes, otherwise bottom right
    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=10)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 8. Save and show
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curve plot to {PLOT_OUTPUT_PATH}")
    
    plt.show()

if __name__ == "__main__":
    main()