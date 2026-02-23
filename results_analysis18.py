import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
import joblib


# --- File paths ---
MODELS_DIR = "models/18f"
RESULTS_DIR = os.path.join("result_collection", "18f")
os.makedirs(RESULTS_DIR, exist_ok=True)


# SVM: Load model and label encoder
svm_model = joblib.load(f"{MODELS_DIR}/svm18_model.joblib")
svm_le = joblib.load(f"{MODELS_DIR}/svm18_label_encoder.joblib")

# XGBoost
xgb_results = pd.read_csv(f"{MODELS_DIR}/xgboost_halving_results.csv")
with open(f"{MODELS_DIR}/xgboost_best_halving.json") as f:
    xgb_best = json.load(f)

# KNN
knn_model = joblib.load(f"{MODELS_DIR}/knn18_faiss_gpu_enn_manhattan_k1_wuniform.joblib")

# --- Class labels ---
class_labels = ["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"]

# --- Helper to plot classification report ---
def plot_classification_report(report_dict, title):
    df = pd.DataFrame(report_dict).T.iloc[:-3, :]
    df = df[["precision", "recall", "f1-score", "support"]]
    df[["precision", "recall", "f1-score"]] = df[["precision", "recall", "f1-score"]].astype(float)
    df["support"] = df["support"].astype(int)
    plt.figure(figsize=(10, 4))
    sns.heatmap(df[["precision", "recall", "f1-score"]], annot=True, cmap="Blues", fmt=".2f")
    plt.title(title)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    fname = title.lower().replace(" ", "_").replace("/", "_") + ".png"
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname))
    plt.close()

# --- Helper to plot grouped bar charts for inter-classifier comparison ---
def plot_grouped_bars(metrics_dict, metric, class_labels):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(class_labels))
    width = 0.25
    for i, (clf, values) in enumerate(metrics_dict.items()):
        plt.bar(x + i*width, values, width, label=clf)
    plt.xticks(x + width, class_labels)
    plt.ylabel(metric)
    plt.title(f"{metric} by Class and Classifier")
    plt.legend()
    fname = f"grouped_{metric.lower()}_by_classifier.png"
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname))
    plt.close()


# --- Load test set for predictions ---
import os
PREPROC_DIR = os.path.join(os.path.dirname(__file__), "preprocessed_output")
test_file = os.path.join(PREPROC_DIR, "dataset_TESTING_reduced18.parquet")
df_test = pd.read_parquet(test_file)
feature_cols = [c for c in df_test.columns if c in class_labels or any(c.startswith(f"ch{i}_") for i in range(1,9))]
X_test = df_test[feature_cols].values.astype(np.float32)
y_true = svm_le.transform(df_test["label"].values)

# SVM predictions and report
y_pred_svm = svm_model.predict(X_test)
svm_report_dict = classification_report(y_true, y_pred_svm, target_names=class_labels, output_dict=True)

# XGBoost
xgb_report = xgb_best["classification_report"] if "classification_report" in xgb_best else None
xgb_report_dict = json.loads(xgb_report.replace("'", '"')) if isinstance(xgb_report, str) else xgb_report

# KNN (Assume you have y_pred_knn and y_true)
# y_pred_knn = knn_model.predict(X_test)  # Uncomment and provide X_test
# knn_report_dict = classification_report(y_true, y_pred_knn, target_names=class_labels, output_dict=True)
knn_report_dict = None  # Replace with actual report


# --- Plot confusion matrices and classification reports ---
for clf_name, report_dict in zip(["SVM", "XGBoost", "KNN"], [svm_report_dict, xgb_report_dict, knn_report_dict]):
    if report_dict is not None:
        plot_classification_report(report_dict, f"{clf_name} Classification Report")

# --- Compute average accuracy for each classifier ---
avg_acc = {}
for clf_name, report_dict in zip(["SVM", "XGBoost", "KNN"], [svm_report_dict, xgb_report_dict, knn_report_dict]):
    if report_dict is not None:
        avg_acc[clf_name] = np.mean([report_dict[c]["f1-score"] for c in class_labels])

# --- Inter-classifier comparison plots ---
metrics = ["precision", "recall", "f1-score", "support"]
for metric in metrics:
    metric_dict = {}
    for clf_name, report_dict in zip(["SVM", "XGBoost", "KNN"], [svm_report_dict, xgb_report_dict, knn_report_dict]):
        if report_dict is not None:
            metric_dict[clf_name] = [report_dict[c][metric] for c in class_labels]
    plot_grouped_bars(metric_dict, metric, class_labels)

# --- Plot average accuracy for all classifiers ---
plt.figure(figsize=(6, 4))
plt.bar(avg_acc.keys(), avg_acc.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.ylabel("Average F1-score")
plt.title("Average F1-score by Classifier")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "average_f1score_by_classifier.png"))
plt.close()
