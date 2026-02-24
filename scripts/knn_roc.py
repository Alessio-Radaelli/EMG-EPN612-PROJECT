import numpy as np
import pandas as pd
import joblib
import faiss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 1. Configuration

KNN_MODEL_PATH = "models/18f/knn18_faiss_gpu_enn_manhattan_k1_wuniform.joblib"
TRAIN_DATA_PATH = "preprocessed_output/dataset_TRAINING_reduced18.parquet"
TEST_DATA_PATH = "preprocessed_output/dataset_TESTING_reduced18.parquet"
ENCODER_PATH = "preprocessed_output/label_encoder.pkl"

ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES = len(ALL_LABELS)

def main():
    print("1. Loading Data and Label Encoder...")

    # Load training set to get feature columns
    df_train = pd.read_parquet(TRAIN_DATA_PATH)
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ("label", "window_idx")]

    # Load test set and use the same feature columns
    df_test = pd.read_parquet(TEST_DATA_PATH)
    X_test = df_test[feature_cols].values.astype(np.float32)
    print(f"Feature columns used: {feature_cols}")
    print(f"Shape of X_test: {X_test.shape}")
    le = joblib.load(ENCODER_PATH)
    y_test = le.transform(df_test["label"].values)
    y_test_bin = label_binarize(y_test, classes=range(N_CLASSES))

    print("2. Loading k-NN Model Data...")
    knn_data = joblib.load(KNN_MODEL_PATH)
    X_store = knn_data["X_store"].astype(np.float32)
    y_store = knn_data["y_store"]
    print(f"Shape of X_store: {X_store.shape}")

    print("3. Rebuilding FAISS CPU Index (Manhattan / L1)...")
    d = X_store.shape[1]
    cpu_index = faiss.IndexFlat(d, faiss.METRIC_L1)
    cpu_index.add(np.ascontiguousarray(X_store))

    print("4. Running Inference on Test Set...")
    chunk_size = 100000
    I_list = []
    X_test_contig = np.ascontiguousarray(X_test)
    for i in range(0, len(X_test_contig), chunk_size):
        end = min(i + chunk_size, len(X_test_contig))
        _, I_chunk = cpu_index.search(X_test_contig[i:end], 1)
        I_list.append(I_chunk)
    I = np.vstack(I_list)
    y_pred = y_store[I[:, 0]]

    print("5. Generating Probabilities & ROC Curves...")
    y_score = np.zeros((len(y_pred), N_CLASSES))
    y_score[np.arange(len(y_pred)), y_pred] = 1.0

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print("6. Plotting...")
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, color in zip(range(N_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{ALL_LABELS[i]} (AUC = {roc_auc[i]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('k-NN (k=1) Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    filename = "ROC_KNN_18f.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Done! ROC plot saved to {filename}")

if __name__ == "__main__":
    main()
