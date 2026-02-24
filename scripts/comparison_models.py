import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# Configurazione File
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "models"

# Mappa dei file JSON da confrontare. 
# Usiamo il set a 36 feature per i modelli ML come "baseline" ottimizzata, 
# e il raw signal per la TDCNN.
MODEL_FILES = {
    "SVM (36f)": RESULTS_DIR / "36f" / "svm_val_test_results36.json",
    "XGBoost (36f)": RESULTS_DIR / "36f" / "xgboost36_test_results.json",
    "KNN (36f)": RESULTS_DIR / "36f" / "knn_test_results36.json",
    "TDCNN (Raw)": RESULTS_DIR / "tdcnn_test_results.json"
}

GESTURES = ["fist", "noGesture", "open", "pinch", "waveIn", "waveOut"]

def main():
    global_records = []
    gesture_records = []

    # 1. Parsing dei file JSON
    for model_name, filepath in MODEL_FILES.items():
        if not filepath.exists():
            print(f"[ATTENZIONE] File non trovato per {model_name}: {filepath}")
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        acc = data.get("test_accuracy", 0)
        report = data.get("classification_report", {})
        f1_macro = report.get("macro avg", {}).get("f1-score", 0)
        
        # Salvataggio metriche globali
        global_records.append({
            "Modello": model_name,
            "Accuratezza": acc * 100,
            "Macro F1": f1_macro * 100
        })
        
        # Salvataggio F1-Score per singolo gesto
        for gesture in GESTURES:
            f1_gesto = report.get(gesture, {}).get("f1-score", 0)
            gesture_records.append({
                "Modello": model_name,
                "Gesto": gesture,
                "F1-Score": f1_gesto * 100
            })

    df_global = pd.DataFrame(global_records)
    df_gestures = pd.DataFrame(gesture_records)

    if df_global.empty:
        print("Nessun dato caricato. Controlla i percorsi dei file.")
        return

    # --- Stampa Tabella Riassuntiva ---
    print("\n" + "="*50)
    print(" CONFRONTO GLOBALE MODELLI (ZERO-SHOT)")
    print("="*50)
    print(df_global.to_string(index=False))

    # --- Creazione Grafici ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    
    # Figura con 2 subplot (1 riga, 2 colonne)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 2.5]})

    # Grafico 1: Accuratezza Globale
    sns.barplot(data=df_global, x="Modello", y="Accuratezza", ax=axes[0], palette="magma")
    axes[0].set_title("Accuratezza Globale", fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuratezza (%)")
    axes[0].tick_params(axis='x', rotation=45)

    # Grafico 2: F1-Score per Gesto
    sns.barplot(data=df_gestures, x="Gesto", y="F1-Score", hue="Modello", ax=axes[1], palette="magma")
    axes[1].set_title("F1-Score per Gesto (Robustezza per Classe)", fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("F1-Score (%)")
    axes[1].legend(title="Modello", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Salvataggio
    plot_path = RESULTS_DIR / "model_comparison_gestures.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico comparativo salvato in: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()