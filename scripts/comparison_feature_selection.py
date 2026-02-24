import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# Configurazione
# =============================================================================
# Script in scripts/ -> parent.parent = radice del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "models"

# Modelli tradizionali (che usano le feature)
ML_MODELS = ["SVM", "XGBoost", "KNN"]
FEATURE_SETS = [72, 36, 18]

# Mappatura dei file JSON previsti per i modelli ML
# (modifica questi nomi se nel tuo file system sono leggermente diversi)
FILE_MAP = {
    "SVM": {
        72: "72f/svm_val_test_results72.json",
        36: "36f/svm_val_test_results36.json",
        18: "18f/svm_val_test_results18.json"
    },
    "XGBoost": {
        72: "72f/xgboost72_test_results.json",
        36: "36f/xgboost36_test_results.json",
        18: "18f/xgboost18_test_results.json"
    },
    "KNN": {
        72: "72f/knn_test_results72.json",
        36: "36f/knn_test_results36.json",
        18: "18f/knn_test_results18.json"
    }
}

# Modello Deep Learning (Raw Signal, nessuna feature)
TDCNN_JSON_PATH = "tdcnn_test_results.json" 

def main():
    records = []

    # 1. Estrazione dati modelli Machine Learning (SVM, XGBoost, KNN)
    for model in ML_MODELS:
        for feat in FEATURE_SETS:
            filepath = RESULTS_DIR / FILE_MAP[model][feat]
            
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                # Estrae Accuratezza e Macro F1-Score
                acc = data.get("test_accuracy", 0)
                # Il dizionario di classification_report ha chiavi diverse a volte, 
                # ma di solito contiene "macro avg"
                f1_macro = data.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0)
                
                records.append({
                    "Modello": model,
                    "Input": f"{feat} Features",
                    "Ordinamento": feat, # Usato per ordinare le barre nel grafico
                    "Accuratezza (%)": acc * 100,
                    "Macro F1 (%)": f1_macro * 100
                })
            else:
                print(f"[ATTENZIONE] File mancante: {filepath}")

    # 2. Estrazione dati TDCNN (Raw Signals)
    tdcnn_path = RESULTS_DIR / TDCNN_JSON_PATH
    if tdcnn_path.exists():
        with open(tdcnn_path, 'r') as f:
            data = json.load(f)
            acc = data.get("test_accuracy", 0)
            f1_macro = data.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0)
            records.append({
                "Modello": "TDCNN",
                "Input": "Raw Signal",
                "Ordinamento": 999, # Mettiamo il Raw Signal alla fine
                "Accuratezza (%)": acc * 100,
                "Macro F1 (%)": f1_macro * 100
            })
    else:
        print(f"[ATTENZIONE] File TDCNN mancante: {tdcnn_path}")

    # 3. Creazione DataFrame
    df = pd.DataFrame(records)
    if df.empty:
        print("Nessun dato trovato. Verifica i percorsi dei file.")
        return

    # Ordina il dataframe per avere 72 -> 36 -> 18 -> Raw
    df = df.sort_values(by=["Modello", "Ordinamento"], ascending=[True, False])

    print("\n" + "="*60)
    print(" TABELLA RIASSUNTIVA ZERO-SHOT (AGGREGATA SU TUTTI GLI UTENTI)")
    print("="*60)
    # Stampa la tabella formattata per il report
    print(df[["Modello", "Input", "Accuratezza (%)", "Macro F1 (%)"]].to_string(index=False))

    # 4. Generazione Grafico a Barre
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ordine: ML models prima, TDCNN alla fine
    model_order = ["KNN", "SVM", "XGBoost", "TDCNN"]

    # Creiamo un barplot che raggruppa per modello e colora per tipo di input
    sns.barplot(
        data=df, 
        x="Modello", 
        y="Accuratezza (%)", 
        hue="Input",
        order=model_order,
        palette="viridis",
        ax=ax
    )

    # Centra le etichette sull'asse x
    ax.set_xticklabels(model_order, ha="center")
    ax.set_title("Confronto Zero-Shot: Impatto della Riduzione Dimensionalità", fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuratezza Globale sul Test Set (%)")
    ax.set_xlabel("Modello di Classificazione")
    
    # Sposta la legenda fuori dal grafico
    plt.legend(title="Tipo di Input", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Salva il grafico
    plot_path = RESULTS_DIR / "zero_shot_comparison_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico salvato in: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()