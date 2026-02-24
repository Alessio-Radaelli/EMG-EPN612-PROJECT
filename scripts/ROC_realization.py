import joblib
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator, ClassifierMixin

# ==========================================
# 1. GLOBALS & CONFIG
# ==========================================
DEVICE = 'cpu'
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
N_CLASSES = len(ALL_LABELS)
MODELS_DIR = "models/18f"
FEATURES = 18

# ==========================================
# 2. SCORING FUNCTIONS (One per model)
# ==========================================

def get_svm_scores(X_test_flat):
    """Loads RFF SVM and returns decision function logits."""
    print("Computing SVM probabilities...")
    import joblib
    import numpy as np
    import torch
    import math
    import torch.nn as nn
    ckpt_path = f"{MODELS_DIR}/svm_val_best{FEATURES}.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    W = torch.as_tensor(ckpt["W"])
    b = torch.as_tensor(ckpt["b"])
    D = ckpt.get("params", {}).get("D", 10000)
    linear = nn.Linear(W.shape[1], N_CLASSES)
    linear.load_state_dict(ckpt["model_state_dict"])
    linear.eval()
    X_tensor = torch.from_numpy(X_test_flat).float()
    all_logits = []
    batch_size = 16384
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            Z = math.sqrt(2.0 / D) * torch.cos(batch @ W + b)
            logits = linear(Z)
            probs = torch.softmax(logits, dim=1)
            all_logits.append(probs.cpu().numpy())
    return np.concatenate(all_logits, axis=0)

def get_xgb_scores(X_test_flat):
    """Loads XGBoost and returns probabilities."""
    print("Computing XGBoost probabilities...")
    import xgboost as xgb
    model_path = f"{MODELS_DIR}/xgboost_model{FEATURES}.json"
    booster = xgb.Booster()
    booster.load_model(model_path)
    dtest = xgb.DMatrix(X_test_flat)
    y_score = booster.predict(dtest)
    return y_score

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================

def plot_multiclass_roc(y_test, y_score, model_name):
    y_test_bin = label_binarize(y_test, classes=range(N_CLASSES))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, color in zip(range(N_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC {ALL_LABELS[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    filename = f"Multiclass_ROC_-_{model_name}_{FEATURES}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved ROC plot: {filename}")
    plt.close()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    import sys
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if mode in ["svm", "xgb", "all"]:
        print("Loading flat dataset for traditional models...")
        df_test = pd.read_parquet("preprocessed_output/dataset_TESTING_reduced18.parquet")
        numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != "label"]
        X_test_flat = df_test[feature_cols].values.astype(np.float32)
        le = joblib.load("preprocessed_output/label_encoder.pkl")
        y_test_flat = le.transform(df_test["label"].values)
        del df_test
        if mode in ["svm", "all"]:
            svm_scores = get_svm_scores(X_test_flat)
            plot_multiclass_roc(y_test_flat, svm_scores, "SVM (RFF)")
        if mode in ["xgb", "all"]:
            xgb_scores = get_xgb_scores(X_test_flat)
            plot_multiclass_roc(y_test_flat, xgb_scores, "XGBoost")
