"""
Visualize model test results from models/18f/ - one plot per model:
KNN, SVM, TDCNN, XGBoost.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "models" / "18f"

MODEL_FILES = {
    "KNN": RESULTS_DIR / "knn_test_results18.json",
    "SVM": RESULTS_DIR / "svm_val_test_results18.json",
    "TDCNN": RESULTS_DIR / "tdcnn_test_results.json",
    "XGBoost": RESULTS_DIR / "xgboost18_test_results.json",
}

GESTURES = ["fist", "noGesture", "open", "pinch", "waveIn", "waveOut"]
METRICS = ["precision", "recall", "f1-score"]


def load_model_data(model_name: str, filepath: Path):
    """Load a single model's results from JSON."""
    if not filepath.exists():
        print(f"[WARN] File not found: {filepath}")
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def plot_model(model_name: str, data: dict, output_path: Path):
    """Create and save a dedicated visualization for one model."""
    report = data.get("classification_report", {})
    acc = data.get("test_accuracy", 0) * 100
    f1_macro = report.get("macro avg", {}).get("f1-score", 0) * 100
    time_s = data.get("total_time_s") or data.get("elapsed_s") or 0

    # Build per-gesture dataframe
    rows = []
    for gesture in GESTURES:
        g_report = report.get(gesture, {})
        for metric in METRICS:
            val = g_report.get(metric, 0) * 100
            label = "F1-Score" if metric == "f1-score" else metric.replace("-", " ").title()
            rows.append({"Gesture": gesture, "Metric": label, "Value (%)": val})

    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2.5]})

    # Left: summary (accuracy, macro F1) + time as text
    ax = axes[0]
    summary_data = [("Accuracy", acc), ("Macro F1", f1_macro)]
    labels = [s[0] for s in summary_data]
    values = [s[1] for s in summary_data]
    colors = sns.color_palette("magma", n_colors=2)
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(f"{model_name} - Summary", fontweight="bold")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.1f}%", xy=(b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=10)
    ax.text(0.5, -0.12, f"Inference time: {time_s:.1f}s", transform=ax.transAxes, ha="center", fontsize=10)

    # Right: precision, recall, F1 per gesture
    ax = axes[1]
    sns.barplot(data=df, x="Gesture", y="Value (%)", hue="Metric", ax=ax, palette="magma")
    ax.set_title(f"{model_name} - Metrics per Gesture", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.suptitle(f"{model_name} Test Results", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {model_name}: {output_path}")


def save_metrics_table_image(df_table: pd.DataFrame, output_path: Path, title: str = "Metrics by Model and Gesture (%)"):
    """Render the metrics table as an image and save it."""
    if df_table.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Format values for display
    df_display = df_table.copy()
    for col in ["Precision (%)", "Recall (%)", "F1-Score (%)"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}")

    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Style header row
    for j in range(len(df_display.columns)):
        table[(0, j)].set_facecolor("#4a4a4a")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(df_display) + 1):
        color = "#f5f5f5" if i % 2 == 0 else "#ffffff"
        for j in range(len(df_display.columns)):
            table[(i, j)].set_facecolor(color)

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def build_metrics_table():
    """Build a table of precision, recall, F1 for each model and gesture."""
    rows = []
    for model_name, filepath in MODEL_FILES.items():
        data = load_model_data(model_name, filepath)
        if data is None:
            continue
        report = data.get("classification_report", {})
        for gesture in GESTURES:
            g_report = report.get(gesture, {})
            row = {
                "Model": model_name,
                "Gesture": gesture,
                "Precision (%)": round(g_report.get("precision", 0) * 100, 2),
                "Recall (%)": round(g_report.get("recall", 0) * 100, 2),
                "F1-Score (%)": round(g_report.get("f1-score", 0) * 100, 2),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    # 1. Build and save metrics table
    df_table = build_metrics_table()
    if not df_table.empty:
        csv_path = RESULTS_DIR / "metrics_table.csv"
        df_table.to_csv(csv_path, index=False)
        print(f"[OK] Metrics table saved: {csv_path}")

        # One table image per model
        for model in df_table["Model"].unique():
            subset = df_table[df_table["Model"] == model].drop(columns=["Model"])
            table_img_path = RESULTS_DIR / f"{model.lower()}_metrics_table.png"
            save_metrics_table_image(subset, table_img_path, title=f"{model} - Metrics per Gesture (%)")
            print(f"[OK] {model} table image: {table_img_path}")

        # Print table to console (per model for readability)
        print("\n" + "=" * 80)
        print(" METRICS BY MODEL AND GESTURE (%)")
        print("=" * 80)
        for model in df_table["Model"].unique():
            subset = df_table[df_table["Model"] == model]
            print(f"\n--- {model} ---")
            print(subset.to_string(index=False))
        print()

    # 2. Generate plots per model
    for model_name, filepath in MODEL_FILES.items():
        data = load_model_data(model_name, filepath)
        if data is None:
            continue
        out_path = RESULTS_DIR / f"{model_name.lower()}_results.png"
        plot_model(model_name, data, out_path)
