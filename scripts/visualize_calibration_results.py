"""
Visualize calibration results, zero-shot vs calibrated accuracies,
and XGBoost 18f model performance.
Generates separate plots per model and reports statistical significance.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "preprocessed_output" / "calibration_results"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = CALIBRATION_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["TDCNN", "SVM", "XGBoost", "KNN"]


def format_pvalue(p):
    """Format p-value for display."""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def sig_stars(p):
    """Return significance stars: *** p<0.001, ** p<0.01, * p<0.05."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def load_data():
    """Load all data sources."""
    data = {}
    
    # Calibration CSVs (0-1 scale)
    for name, fname in [
        ("zero_shot", "zero_shot_accuracies.csv"),
        ("calibrated", "calibrated_accuracies.csv"),
        ("all", "all_results.csv"),
    ]:
        path = CALIBRATION_DIR / fname
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            data[name] = df
        else:
            data[name] = None
    
    # Calibration stats summary
    path = CALIBRATION_DIR / "calibration_stats_summary.json"
    data["summary"] = json.loads(path.read_text()) if path.exists() else None
    
    # XGBoost 18f test results
    path = MODELS_DIR / "18f" / "xgboost18_test_results.json"
    data["xgboost18"] = json.loads(path.read_text()) if path.exists() else None
    
    return data


def plot_single_model(df_all, summary, model, ax=None):
    """Per-model plot: each user zero-shot → calibrated, with means and Wilcoxon p-value."""
    if df_all is None or summary is None:
        return
    zs_col = f"{model}_zs"
    cal_col = f"{model}_cal"
    if zs_col not in df_all.columns or cal_col not in df_all.columns:
        return
    ax = ax or plt.gca()

    zs = (df_all[zs_col] * 100).values
    cal = (df_all[cal_col] * 100).values
    users = df_all.index.astype(str).tolist()
    n = len(users)

    # Per-user lines (light gray)
    for i in range(n):
        ax.plot([0, 1], [zs[i], cal[i]], "o-", color="lightgray", alpha=0.6, linewidth=1, markersize=4, zorder=1)

    # Mean line (bold)
    m_zs = summary["means_zero_shot"][model]
    m_cal = summary["means_calibrated"][model]
    ax.plot([0, 1], [m_zs, m_cal], "o-", color="steelblue", linewidth=3, markersize=12, zorder=2, label="Mean")
    ax.annotate(f"{m_zs:.1f}%", (0, m_zs), textcoords="offset points", xytext=(-10, 0), ha="right", fontsize=11, fontweight="bold")
    ax.annotate(f"{m_cal:.1f}%", (1, m_cal), textcoords="offset points", xytext=(10, 0), ha="left", fontsize=11, fontweight="bold")

    # Wilcoxon calibration effect
    wilcoxon = summary.get("wilcoxon_calibration_effect", {}).get(model, {})
    p_val = wilcoxon.get("p_value", 1)
    stars = sig_stars(p_val)
    ax.set_title(f"{model}: Zero-shot → Calibrated\nWilcoxon {format_pvalue(p_val)} {stars}", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Zero-shot", "Calibrated"])
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.15, 1.15)
    ax.set_yticks(range(0, 101, 20))


def fig1_zero_shot_vs_calibrated_means(summary, ax=None):
    """Mean accuracy: zero-shot vs calibrated — slope graph (before → after)."""
    if not summary:
        return
    ax = ax or plt.gca()
    
    models = list(summary["means_zero_shot"].keys())
    zs = [summary["means_zero_shot"][m] for m in models]
    cal = [summary["means_calibrated"][m] for m in models]
    
    colors = plt.cm.tab10.colors[:len(models)]
    for i, (m, v0, v1) in enumerate(zip(models, zs, cal)):
        ax.plot([0, 1], [v0, v1], "o-", color=colors[i], linewidth=2, markersize=8, label=m)
        ax.annotate(f"{v0:.0f}", (0, v0), textcoords="offset points", xytext=(-8, 0), ha="right", fontsize=9)
        ax.annotate(f"{v1:.0f}", (1, v1), textcoords="offset points", xytext=(8, 0), ha="left", fontsize=9)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Zero-shot", "Calibrated"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Mean Accuracy: Zero-shot → Calibrated")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.15, 1.15)


def fig2_calibration_gain_per_user(df_all, ax=None):
    """Per-user calibration gain (calibrated - zero-shot) per model."""
    if df_all is None:
        return
    ax = ax or plt.gca()
    
    records = []
    for model in MODELS:
        zs_col = f"{model}_zs"
        cal_col = f"{model}_cal"
        if zs_col not in df_all.columns or cal_col not in df_all.columns:
            continue
        gain = (df_all[cal_col] - df_all[zs_col]) * 100
        for g in gain:
            records.append({"Model": model, "Gain (%)": g})
    
    df = pd.DataFrame(records)
    sns.boxplot(data=df, x="Model", y="Gain (%)", hue="Model", ax=ax, palette="Set2", legend=False)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Calibration Gain per User (% points)")
    ax.set_ylabel("Gain (%)")


def fig3_model_comparison_boxplots(df_all, ax=None):
    """Box plots: zero-shot vs calibrated accuracies per model."""
    if df_all is None:
        return
    ax = ax or plt.gca()
    
    records = []
    for model in MODELS:
        for suffix, label in [("_zs", "Zero-shot"), ("_cal", "Calibrated")]:
            col = f"{model}{suffix}"
            if col not in df_all.columns:
                continue
            for val in df_all[col] * 100:
                records.append({"Model": model, "Setting": label, "Accuracy (%)": val})
    
    df = pd.DataFrame(records)
    sns.boxplot(data=df, x="Model", y="Accuracy (%)", hue="Setting", ax=ax, palette={"Zero-shot": "steelblue", "Calibrated": "coral"})
    ax.set_title("Accuracy Distribution per Model (Zero-shot vs Calibrated)")
    ax.set_ylim(0, 100)
    ax.legend(title="Setting")


def fig4_user_heatmap(df_all, ax=None):
    """Heatmap: per-user accuracy for calibrated models."""
    if df_all is None:
        return
    ax = ax or plt.gca()
    
    cal_cols = [c for c in df_all.columns if c.endswith("_cal")]
    if not cal_cols:
        return
    df_cal = df_all[cal_cols].copy()
    df_cal.columns = [c.replace("_cal", "") for c in cal_cols]
    df_cal = df_cal * 100
    
    sns.heatmap(df_cal, annot=False, cmap="RdYlGn", ax=ax, vmin=50, vmax=100, cbar_kws={"label": "Accuracy (%)"})
    ax.set_title("Calibrated Accuracy by User and Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("User ID")


def fig5_xgboost18_per_gesture(xgboost_data, ax=None):
    """XGBoost 18f: F1-score per gesture — horizontal bars, sorted."""
    if not xgboost_data:
        return
    ax = ax or plt.gca()
    
    report = xgboost_data.get("classification_report", {})
    gestures = ["fist", "noGesture", "open", "pinch", "waveIn", "waveOut"]
    f1_scores = [report.get(g, {}).get("f1-score", 0) * 100 for g in gestures]
    
    # Sort by F1 descending for clearer reading
    pairs = sorted(zip(gestures, f1_scores), key=lambda x: x[1], reverse=True)
    gestures, f1_scores = [p[0] for p in pairs], [p[1] for p in pairs]
    
    y_pos = range(len(gestures))
    bars = ax.barh(y_pos, f1_scores, height=0.6, color=plt.cm.Blues([0.3 + 0.6 * (v/100) for v in f1_scores]))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gestures)
    ax.set_xlabel("F1-Score (%)")
    ax.set_title("XGBoost (18f) F1 per Gesture")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    for i, v in enumerate(f1_scores):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)


def write_statistics_report(summary, out_path):
    """Write statistical significance report to text file."""
    if not summary:
        return
    lines = [
        "=" * 60,
        "STATISTICAL SIGNIFICANCE REPORT",
        "=" * 60,
        "",
        "1. Wilcoxon signed-rank: Calibration effect (zero-shot vs calibrated)",
        "   Per-model paired test across users.",
        "",
    ]
    for model in MODELS:
        w = summary.get("wilcoxon_calibration_effect", {}).get(model, {})
        if w:
            stat, p = w.get("statistic", ""), w.get("p_value", "")
            lines.append(f"   {model}: W = {stat}, {format_pvalue(p)} {sig_stars(p)}")
    lines.extend([
        "",
        "2. Friedman test: Post-calibration model comparison",
        "   Tests whether calibrated accuracies differ across models.",
        "",
    ])
    f = summary.get("friedman_postcalibration", {})
    if f:
        lines.append(f"   chi² = {f.get('statistic', 0):.2f}, {format_pvalue(f.get('p_value', 1))} {sig_stars(f.get('p_value', 1))}")
    lines.extend([
        "",
        "3. Wilcoxon post-hoc: TDCNN vs baselines (calibrated)",
        "   Pairwise comparisons after Friedman.",
        "",
    ])
    wh = summary.get("wilcoxon_posthoc_tdcnn_vs_baselines", {})
    for key, val in wh.items():
        if isinstance(val, dict) and "p_value" in val:
            comp = key.replace("TDCNN_vs_", "TDCNN vs ")
            adv = val.get("tdcnn_advantage_pct", 0)
            lines.append(f"   {comp}: {format_pvalue(val['p_value'])} {sig_stars(val['p_value'])} (TDCNN +{adv:.2f}% points)")
    lines.extend(["", "Legend: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant", ""])
    out_path.write_text("\n".join(lines))


def write_significance_csv(summary, out_path):
    """Export significance report as CSV table."""
    if not summary:
        return
    records = []
    for model in MODELS:
        w = summary.get("wilcoxon_calibration_effect", {}).get(model, {})
        if w:
            records.append({
                "Test": "Wilcoxon calibration effect",
                "Model/Comparison": model,
                "Statistic": w.get("statistic", ""),
                "p_value": w.get("p_value", ""),
                "Significance": sig_stars(w.get("p_value", 1)),
                "Notes": "",
            })
    f = summary.get("friedman_postcalibration", {})
    if f:
        records.append({
            "Test": "Friedman (post-calibration)",
            "Model/Comparison": "all models",
            "Statistic": f"{f.get('statistic', 0):.2f}",
            "p_value": f.get("p_value", ""),
            "Significance": sig_stars(f.get("p_value", 1)),
            "Notes": "Model comparison",
        })
    wh = summary.get("wilcoxon_posthoc_tdcnn_vs_baselines", {})
    for key, val in wh.items():
        if isinstance(val, dict) and "p_value" in val:
            comp = key.replace("TDCNN_vs_", "TDCNN vs ")
            adv = val.get("tdcnn_advantage_pct", 0)
            records.append({
                "Test": "Wilcoxon post-hoc",
                "Model/Comparison": comp,
                "Statistic": val.get("statistic", ""),
                "p_value": val["p_value"],
                "Significance": sig_stars(val["p_value"]),
                "Notes": f"TDCNN +{adv:.2f}%",
            })
    pd.DataFrame(records).to_csv(out_path, index=False)


def write_improvement_csv(summary, out_path):
    """Export calibration improvement table as CSV."""
    if not summary:
        return
    records = []
    for model in MODELS:
        zs = summary.get("means_zero_shot", {}).get(model, 0)
        cal = summary.get("means_calibrated", {}).get(model, 0)
        imp = cal - zs
        w = summary.get("wilcoxon_calibration_effect", {}).get(model, {})
        p_val = w.get("p_value", 1)
        records.append({
            "Model": model,
            "Zero-shot (%)": round(zs, 2),
            "Calibrated (%)": round(cal, 2),
            "Improvement (pp)": round(imp, 2),
            "p-value": p_val,
            "Significant": "Yes" if p_val < 0.05 else "No",
        })
    pd.DataFrame(records).to_csv(out_path, index=False)


def plot_improvement_table(summary, out_path):
    """Create table image: calibration improvement and statistical significance."""
    if not summary:
        return

    headers = ["Model", "Zero-shot (%)", "Calibrated (%)", "Improvement (pp)", "Significant"]
    rows = []
    for model in MODELS:
        zs = summary.get("means_zero_shot", {}).get(model, 0)
        cal = summary.get("means_calibrated", {}).get(model, 0)
        imp = cal - zs
        w = summary.get("wilcoxon_calibration_effect", {}).get(model, {})
        p_val = w.get("p_value", 1)
        sig = "Yes ***" if p_val < 0.001 else ("Yes **" if p_val < 0.01 else ("Yes *" if p_val < 0.05 else "No (n.s.)"))
        rows.append([
            model,
            f"{zs:.1f}",
            f"{cal:.1f}",
            f"+{imp:.1f}" if imp >= 0 else f"{imp:.1f}",
            sig,
        ])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table = ax.table(
        cellText=[headers] + rows,
        loc="center",
        cellLoc="center",
        colWidths=[0.18, 0.2, 0.2, 0.22, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        cell.set_linewidth(0.8)
        if i == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if i % 2 else "#F8F9FA")
            if j == 4 and "Yes" in str(rows[i - 1][4]):
                cell.set_text_props(color="#27AE60", fontweight="bold")

    ax.set_title("Calibration Improvement: Zero-shot → Calibrated\nWilcoxon signed-rank test per model (pp = percentage points)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.text(0.5, 0.02, "*** p < 0.001  ** p < 0.01  * p < 0.05  n.s. = not significant",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555", style="italic")
    fig.subplots_adjust(left=0.12, right=0.9, top=0.85, bottom=0.12)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_significance_table(summary, out_path):
    """Create a publication-quality table image from the statistical significance report."""
    if not summary:
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [1.4, 0.7, 1.2], "hspace": 0.5})
    fig.patch.set_facecolor("#FAFAFA")
    for ax in axes:
        ax.axis("off")

    # --- Section 1: Wilcoxon calibration effect ---
    ax1 = axes[0]
    headers1 = ["Model", "W statistic", "p-value", "Significance"]
    rows1 = []
    for model in MODELS:
        w = summary.get("wilcoxon_calibration_effect", {}).get(model, {})
        if w:
            rows1.append([
                model,
                str(int(w.get("statistic", ""))),
                format_pvalue(w.get("p_value", 1)),
                sig_stars(w.get("p_value", 1)),
            ])
    tab1 = ax1.table(
        cellText=[headers1] + rows1,
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.25, 0.28, 0.22],
    )
    tab1.auto_set_font_size(False)
    tab1.set_fontsize(11)
    tab1.scale(1.3, 2.4)
    for (i, j), cell in tab1.get_celld().items():
        cell.set_edgecolor("#333")
        cell.set_linewidth(0.8)
        if i == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=12)
        else:
            cell.set_facecolor("#FFFFFF" if i % 2 else "#F8F9FA")
    ax1.set_title("1. Wilcoxon signed-rank: Calibration effect (zero-shot vs calibrated)\nPer-model paired test across users",
                  fontsize=12, fontweight="bold", pad=20, loc="left")

    # --- Section 2: Friedman ---
    ax2 = axes[1]
    f = summary.get("friedman_postcalibration", {})
    headers2 = ["Test", "Statistic", "p-value", "Significance", "Interpretation"]
    rows2 = [[
        "Friedman (post-calibration model comparison)",
        f"\u03C7\u00B2 = {f.get('statistic', 0):.2f}",
        format_pvalue(f.get("p_value", 1)) if f else "",
        sig_stars(f.get("p_value", 1)) if f else "",
        "Calibrated accuracies differ across models",
    ]] if f else []
    tab2 = ax2.table(
        cellText=[headers2] + rows2,
        loc="center",
        cellLoc="center",
        colWidths=[0.35, 0.15, 0.2, 0.15, 0.35],
    )
    tab2.auto_set_font_size(False)
    tab2.set_fontsize(11)
    tab2.scale(1.3, 2.4)
    for (i, j), cell in tab2.get_celld().items():
        cell.set_edgecolor("#333")
        cell.set_linewidth(0.8)
        if i == 0:
            cell.set_facecolor("#34495E")
            cell.set_text_props(color="white", fontweight="bold", fontsize=12)
        else:
            cell.set_facecolor("#FFFFFF")
    ax2.set_title("2. Friedman test: Do calibrated models differ?\nTests whether calibrated accuracies differ across models",
                  fontsize=12, fontweight="bold", pad=20, loc="left")

    # --- Section 3: Post-hoc ---
    ax3 = axes[2]
    wh = summary.get("wilcoxon_posthoc_tdcnn_vs_baselines", {})
    headers3 = ["Comparison", "p-value", "Significance", "TDCNN advantage"]
    rows3 = []
    for key, val in wh.items():
        if isinstance(val, dict) and "p_value" in val:
            comp = key.replace("TDCNN_vs_", "TDCNN vs ")
            adv = val.get("tdcnn_advantage_pct", 0)
            rows3.append([comp, format_pvalue(val["p_value"]), sig_stars(val["p_value"]), f"+{adv:.2f} percentage points"])
    tab3 = ax3.table(
        cellText=[headers3] + rows3,
        loc="center",
        cellLoc="center",
        colWidths=[0.3, 0.25, 0.2, 0.25],
    )
    tab3.auto_set_font_size(False)
    tab3.set_fontsize(11)
    tab3.scale(1.3, 2.4)
    for (i, j), cell in tab3.get_celld().items():
        cell.set_edgecolor("#333")
        cell.set_linewidth(0.8)
        if i == 0:
            cell.set_facecolor("#2980B9")
            cell.set_text_props(color="white", fontweight="bold", fontsize=12)
        else:
            cell.set_facecolor("#FFFFFF" if i % 2 else "#EBF5FB")
    ax3.set_title("3. Wilcoxon post-hoc: TDCNN vs baselines (calibrated)\nPairwise comparisons after significant Friedman",
                  fontsize=12, fontweight="bold", pad=20, loc="left")

    fig.suptitle("Statistical Significance Report", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02, "*** p < 0.001    ** p < 0.01    * p < 0.05    n.s. = not significant",
              ha="center", fontsize=10, color="#555", style="italic")
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.06, hspace=0.6)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FAFAFA", edgecolor="none")
    plt.close()


def fig6_tdcnn_advantage(summary, ax=None):
    """TDCNN advantage over baselines from post-hoc Wilcoxon."""
    if not summary:
        return
    ax = ax or plt.gca()
    
    wh = summary.get("wilcoxon_posthoc_tdcnn_vs_baselines", {})
    if not wh:
        return
    
    labels = []
    advantages = []
    for key, val in wh.items():
        if isinstance(val, dict) and "tdcnn_advantage_pct" in val:
            label = key.replace("TDCNN_vs_", "vs ")
            labels.append(label)
            advantages.append(val["tdcnn_advantage_pct"])
    
    if not labels:
        return
    
    colors = ["green" if a > 0 else "red" for a in advantages]
    ax.barh(labels, advantages, color=colors, alpha=0.8)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("TDCNN Advantage (% points)")
    ax.set_title("TDCNN vs Baselines (Post-calibration)")


def main():
    data = load_data()
    
    summary = data["summary"]
    df_all = data["all"]
    xgboost18 = data["xgboost18"]
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    # --- Per-model plots (separate figure each) ---
    for model in MODELS:
        zs_col = f"{model}_zs"
        if df_all is not None and zs_col in df_all.columns and summary:
            fig, ax = plt.subplots(figsize=(6, 5))
            plot_single_model(df_all, summary, model, ax)
            plt.tight_layout()
            out_path = OUTPUT_DIR / f"per_model_{model.lower()}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[OK] Per-model plot saved: {out_path}")

    # --- Statistics report (text + table image) ---
    report_path = OUTPUT_DIR / "statistical_significance_report.txt"
    write_statistics_report(summary, report_path)
    print(f"[OK] Statistics report saved: {report_path}")
    table_path = OUTPUT_DIR / "statistical_significance_table.png"
    plot_significance_table(summary, table_path)
    print(f"[OK] Significance table image saved: {table_path}")
    csv_path = OUTPUT_DIR / "statistical_significance_table.csv"
    write_significance_csv(summary, csv_path)
    print(f"[OK] Significance table CSV saved: {csv_path}")

    # --- Calibration improvement table (improvement + significance) ---
    imp_csv = OUTPUT_DIR / "calibration_improvement_table.csv"
    write_improvement_csv(summary, imp_csv)
    print(f"[OK] Improvement table CSV saved: {imp_csv}")
    imp_png = OUTPUT_DIR / "calibration_improvement_table.png"
    plot_improvement_table(summary, imp_png)
    print(f"[OK] Improvement table image saved: {imp_png}")
    
    # --- XGBoost 18f F1 per gesture (separate) ---
    if xgboost18:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig5_xgboost18_per_gesture(xgboost18, ax)
        plt.tight_layout()
        out_f1 = OUTPUT_DIR / "xgboost18_f1_per_gesture.png"
        plt.savefig(out_f1, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] XGBoost F1 per gesture saved: {out_f1}")

    # --- Summary dashboard (first row only: slope, gain, accuracy distribution) ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig1_zero_shot_vs_calibrated_means(summary, axes[0])
    fig2_calibration_gain_per_user(df_all, axes[1])
    fig3_model_comparison_boxplots(df_all, axes[2])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Dashboard saved: {OUTPUT_DIR / 'calibration_dashboard.png'}")

    # --- Print stats to console ---
    if summary:
        print("\n--- Summary ---")
        print("Mean zero-shot:", {k: f"{v:.1f}%" for k, v in summary["means_zero_shot"].items()})
        print("Mean calibrated:", {k: f"{v:.1f}%" for k, v in summary["means_calibrated"].items()})
    if xgboost18:
        print(f"XGBoost 18f test accuracy: {xgboost18['test_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
