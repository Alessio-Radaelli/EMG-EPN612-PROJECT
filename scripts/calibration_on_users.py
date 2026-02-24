import copy
import pickle
import random
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import joblib
from pathlib import Path
from scipy import stats
from scipy import signal
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Importa i tuoi moduli custom
from tdcnn_eca import TDCNNClassifier
from train_knn import FaissKNNClassifierGPU, faiss_enn_gpu
from train_svm import RFFSVMClassifier

# =============================================================================
# 1. Configurazione e Percorsi
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_B_PATH = PROJECT_ROOT / "datasets" / "dataset_B.pkl" 
MODELS_DIR = PROJECT_ROOT / "models"/"18f"
USER_DATASETS_DIR = PROJECT_ROOT / "preprocessed_users"
USER_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Cartelle per i salvataggi finali
CALIBRATED_MODELS_DIR = MODELS_DIR / "calibrated_users"
CALIBRATED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_EXPORT_DIR = PROJECT_ROOT / "results"
RESULTS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_EXPORT_DIR = PROJECT_ROOT / "preprocessed_output" / "calibration_results"
CALIBRATION_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Flag per salvare i pesi dei modelli (Mettilo a True se vuoi salvare i 120 file dei modelli)
SAVE_CALIBRATED_MODELS = True

# File dei Modelli Base
KNN_MODEL_PATTERN = "knn*_faiss_gpu_enn_*.joblib"
SVM_MODEL_FILE = MODELS_DIR / "svm_val_best18.pt"
TDCNN_MODEL_FILE = MODELS_DIR / "tdcnn_emg_model.pth"
XGB_MODEL_FILE = MODELS_DIR / "xgboost18_best_halving.json"

# File Ranking (fallback a models/ se non in 18f)
RANKING_CSV_FILE = MODELS_DIR / "feature_ranking_ranks.csv"
if not RANKING_CSV_FILE.exists():
    RANKING_CSV_FILE = PROJECT_ROOT / "models" / "feature_ranking_ranks.csv"

# Impostazioni Test e Ripetizioni
CALIBRATION_REPS = 5 
ALL_LABELS = sorted(["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"])
MAX_TEST_USERS = 30
RANDOM_SEED = 42

# Parametri Globali
FS            = 200
WINDOW_LENGTH = 40
WINDOW_SHIFT  = 10
THRESHOLD     = 0.00001
CHANNELS      = [f"ch{i}" for i in range(1, 9)]
TD9_NAMES     = ["LS", "MFL", "MSR", "WAMP", "ZC", "RMS", "IAV", "DASDV", "VAR"]
ALL_FEATURE_COLS = [f"{ch}_{feat}" for ch in CHANNELS for feat in TD9_NAMES]
NO_GESTURE_CROP_FALLBACK = int(1.0 * FS)

# =============================================================================
# 2. PHASE 1 — Signal Conditioning
# =============================================================================
def filter_emg(emg_signal, fs=FS, lowcut=20, highcut=95, notch_freq=50, notch_q=30):
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b_bp, a_bp = signal.butter(2, [low, high], btype="band")
    filtered = signal.filtfilt(b_bp, a_bp, emg_signal, axis=0)
    b_notch, a_notch = signal.iirnotch(w0=notch_freq, Q=notch_q, fs=fs)
    filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=0)
    return filtered

def segment_trial(filtered_channels, sample_meta, no_gesture_crop=None):
    gesture = sample_meta["gestureName"]
    if gesture != "noGesture":
        gti   = sample_meta["groundTruthIndex"]
        start = gti[0] - 1
        end   = gti[1]
        return {ch: arr[start:end] for ch, arr in filtered_channels.items()}
    else:
        length  = len(next(iter(filtered_channels.values())))
        crop    = no_gesture_crop if no_gesture_crop else NO_GESTURE_CROP_FALLBACK
        centre  = length // 2
        start   = max(centre - crop // 2, 0)
        end     = start + crop
        if end > length:
            end   = length
            start = max(end - crop, 0)
        return {ch: arr[start:end] for ch, arr in filtered_channels.items()}

# =============================================================================
# 3. PHASE 2 — TD9 Feature Extraction
# =============================================================================
def _LS(x):
    n = len(x)
    if n < 2: return 0.0
    xs = np.sort(x)
    i  = np.arange(1, n + 1)
    return np.sum((2 * i - n - 1) * xs) / (n * (n - 1))

def _MFL(x): return np.log(np.sqrt(np.sum(np.diff(x) ** 2)) + 1e-10)
def _MSR(x): return np.mean(np.sqrt(np.abs(x)))
def _WAMP(x, thr): return np.sum(np.abs(np.diff(x)) > thr)
def _ZC(x, thr):
    x1, x2 = x[:-1], x[1:]
    return np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) > thr))
def _RMS(x): return np.sqrt(np.mean(x ** 2))
def _IAV(x): return np.sum(np.abs(x))
def _DASDV(x): return np.sqrt(np.mean(np.diff(x) ** 2))
def _VAR(x): return np.var(x, ddof=1)

def extract_td9(window, thr=THRESHOLD):
    return np.array([
        _LS(window),  _MFL(window),  _MSR(window),
        _WAMP(window, thr), _ZC(window, thr),
        _RMS(window), _IAV(window),  _DASDV(window), _VAR(window)
    ])

# =============================================================================
# 4. Utilities di Parsing e Gestione Dataset Globale
# =============================================================================
def carica_indici_top18(csv_path):
    try:
        df = pd.read_csv(csv_path)
        col_name = 'feature' if 'feature' in df.columns else df.columns[0]
        top_18_names = df[col_name].head(18).tolist()
        indici = [ALL_FEATURE_COLS.index(f) for f in top_18_names if f in ALL_FEATURE_COLS]
        return indici
    except FileNotFoundError:
        print(f"\n[ERRORE CRITICO] File {csv_path} non trovato!")
        exit(1)

def crea_dataset_globale_non_scalato(dataset_b, top_18_indices):
    unscaled_cache_path = USER_DATASETS_DIR / "dataset_B_unscaled_windows.pkl"
    if unscaled_cache_path.exists():
        print(f"\n      [Cache] Trovato dataset globale grezzo: {unscaled_cache_path.name}")
        with open(unscaled_cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"\n      [Preprocessing] Creazione dataset globale intermedio (richiederà qualche minuto)...")
    unscaled_dataset = {}
    for user_id, user_data in tqdm(dataset_b.items(), desc="Signal Cond. & Feature Extr."):
        samples = user_data.get('trainingSamples', user_data)
        user_unscaled_data = []
        
        gesture_lengths = []
        for key, sample in samples.items():
            gti = sample.get("groundTruthIndex")
            if gti and sample["gestureName"] != "noGesture":
                gesture_lengths.append(gti[1] - gti[0] + 1)
        median_gesture_len = (int(np.median(gesture_lengths)) if gesture_lengths else NO_GESTURE_CROP_FALLBACK)
        
        gesture_samples = {label: [] for label in ALL_LABELS}
        for key, sample in samples.items():
            gesture_samples[sample["gestureName"]].append(sample)
            
        for gesture, reps in gesture_samples.items():
            for rep_idx, sample_meta in enumerate(reps):
                emg = sample_meta["emg"]
                filtered = {ch: filter_emg(np.array(emg[ch])) for ch in CHANNELS}
                cropped = segment_trial(filtered, sample_meta, no_gesture_crop=median_gesture_len)
                n_samples = len(next(iter(cropped.values())))
                
                for start in range(0, n_samples - WINDOW_LENGTH + 1, WINDOW_SHIFT):
                    end = start + WINDOW_LENGTH
                    raw_win = np.column_stack([cropped[ch][start:end] for ch in CHANNELS])
                    all_72_feat = np.concatenate([extract_td9(cropped[ch][start:end], thr=THRESHOLD) for ch in CHANNELS])
                    features_18 = all_72_feat[top_18_indices]
                    
                    user_unscaled_data.append({
                        "window": raw_win.astype(np.float32),
                        "features": features_18,
                        "label": gesture,
                        "rep_idx": rep_idx
                    })
        unscaled_dataset[user_id] = user_unscaled_data

    with open(unscaled_cache_path, 'wb') as f:
        pickle.dump(unscaled_dataset, f)
    return unscaled_dataset

# =============================================================================
# 5. Scaling e Suddivisione per Utente Singolo
# =============================================================================
def scala_e_dividi_utente(user_id, user_unscaled_data, le):
    raw_filepath = USER_DATASETS_DIR / f"{user_id}_scaled_raw.npz"
    feat_filepath = USER_DATASETS_DIR / f"{user_id}_scaled_feat.npz"
    if raw_filepath.exists() and feat_filepath.exists():
        raw_npz = np.load(raw_filepath)
        feat_npz = np.load(feat_filepath)
        return (
            (raw_npz['X_cal'], raw_npz['y_cal'], raw_npz['X_eval'], raw_npz['y_eval']),
            (feat_npz['X_cal'], feat_npz['y_cal'], feat_npz['X_eval'], feat_npz['y_eval'])
        )

    calib_raw, calib_features, calib_labels = [], [], []
    eval_raw, eval_features, eval_labels = [], [], []
    for item in user_unscaled_data:
        if item["rep_idx"] < CALIBRATION_REPS:
            calib_raw.append(item["window"])
            calib_features.append(item["features"])
            calib_labels.append(item["label"])
        else:
            eval_raw.append(item["window"])
            eval_features.append(item["features"])
            eval_labels.append(item["label"])

    X_cal_raw, X_eval_raw = np.array(calib_raw), np.array(eval_raw)
    X_cal_feat, X_eval_feat = np.array(calib_features), np.array(eval_features)
    y_cal, y_eval = le.transform(calib_labels), le.transform(eval_labels)

    mu_raw, sig_raw = np.mean(X_cal_raw, axis=(0, 1)), np.std(X_cal_raw, axis=(0, 1)) + 1e-8
    X_cal_raw = (X_cal_raw - mu_raw) / sig_raw
    X_eval_raw = (X_eval_raw - mu_raw) / sig_raw

    scaler = StandardScaler()
    X_cal_feat = scaler.fit_transform(X_cal_feat)
    X_eval_feat = scaler.transform(X_eval_feat)

    np.savez_compressed(raw_filepath, X_cal=X_cal_raw, y_cal=y_cal, X_eval=X_eval_raw, y_eval=y_eval)
    np.savez_compressed(feat_filepath, X_cal=X_cal_feat, y_cal=y_cal, X_eval=X_eval_feat, y_eval=y_eval)

    return (X_cal_raw, y_cal, X_eval_raw, y_eval), (X_cal_feat, y_cal, X_eval_feat, y_eval)

# =============================================================================
# 6. Pipeline Principale di Valutazione
# =============================================================================
def main():
    print(f"\n[1/5] Inizializzazione e Caricamento Feature Ranking...")
    top_18_indices = carica_indici_top18(RANKING_CSV_FILE)
    le = LabelEncoder().fit(ALL_LABELS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Verifica presenza base models
    missing = []
    if not TDCNN_MODEL_FILE.exists():
        missing.append(f"TDCNN: {TDCNN_MODEL_FILE}")
    if not XGB_MODEL_FILE.exists():
        missing.append(f"XGBoost: {XGB_MODEL_FILE}")
    if not SVM_MODEL_FILE.exists():
        missing.append(f"SVM: {SVM_MODEL_FILE}")
    knn_files = list(MODELS_DIR.glob(KNN_MODEL_PATTERN))
    if not knn_files:
        missing.append(f"KNN: {MODELS_DIR}/{KNN_MODEL_PATTERN}")
    if not DATASET_B_PATH.exists():
        missing.append(f"Dataset: {DATASET_B_PATH}")
    if missing:
        print("\n" + "=" * 60)
        print(" ERRORE: Base model o dataset non trovati!")
        print("=" * 60)
        for m in missing:
            print(f"  • {m}")
        print("=" * 60)
        raise FileNotFoundError(f"Base model/dataset mancanti: {len(missing)} file")

    print(f"[2/5] Caricamento Modelli Globali...")
    base_tdcnn = TDCNNClassifier.load(str(TDCNN_MODEL_FILE))
    base_xgb = xgb.XGBClassifier()
    base_xgb.load_model(str(XGB_MODEL_FILE))

    knn_path = max(knn_files, key=lambda p: p.stat().st_mtime)
    knn_data = joblib.load(knn_path)
    X_train_global_knn, y_train_global_knn = knn_data["X_store"], knn_data["y_store"]
    global_knn = FaissKNNClassifierGPU(n_neighbors=1, metric="manhattan")
    global_knn.fit(X_train_global_knn, y_train_global_knn)
    
    svm_ckpt = torch.load(SVM_MODEL_FILE, map_location=device)
    base_svm_params = svm_ckpt["params"]
    base_svm_state = svm_ckpt["model_state_dict"]
    base_svm_W, base_svm_b = svm_ckpt["W"], svm_ckpt["b"]

    print(f"\n[3/5] Elaborazione Dataset e Signal Conditioning...")
    with open(DATASET_B_PATH, 'rb') as f:
        dataset_test = pickle.load(f)
    
    unscaled_dataset = crea_dataset_globale_non_scalato(dataset_test, top_18_indices)

    all_users = sorted(list(unscaled_dataset.keys()))
    np.random.seed(RANDOM_SEED)
    shuffled_users = list(np.random.permutation(all_users))
    selected_users = shuffled_users[:min(MAX_TEST_USERS, len(all_users))]
    
    print(f"      Selezionati {len(selected_users)} pazienti su {len(all_users)} (Seme={RANDOM_SEED})")

    user_results_zs = {}
    user_results_cal = {}

    print(f"\n[4/5] Avvio Valutazione (Zero-Shot vs Calibrato) Utente per Utente...")
    for user_id in tqdm(selected_users, desc="Inferenza & Fine-Tuning"):
        results_cache_path = USER_DATASETS_DIR / f"{user_id}_results.json"
        
        # Se i risultati ci sono già, ricaricali per risparmiare tempo
        if results_cache_path.exists():
            with open(results_cache_path, 'r') as f:
                cached_res = json.load(f)
            user_results_zs[user_id] = cached_res["zero_shot"]
            user_results_cal[user_id] = cached_res["calibrated"]
            continue

        user_unscaled_data = unscaled_dataset[user_id]
        raw_data, feat_data = scala_e_dividi_utente(user_id, user_unscaled_data, le)
        X_cal_raw, y_cal, X_eval_raw, y_eval = raw_data
        X_cal_feat, _, X_eval_feat, _ = feat_data
        
        if len(y_eval) == 0: continue
        
        accs_zs, accs_cal = {}, {}

        # --- FASE ZERO-SHOT ---
        accs_zs["TDCNN"] = accuracy_score(y_eval, base_tdcnn.predict(X_eval_raw))
        accs_zs["XGBoost"] = accuracy_score(y_eval, base_xgb.predict(X_eval_feat))
        accs_zs["KNN"] = accuracy_score(y_eval, global_knn.predict(X_eval_feat))
        
        linear_zs = nn.Linear(base_svm_W.shape[1], len(le.classes_)).to(device)
        linear_zs.load_state_dict(base_svm_state)
        linear_zs.eval()
        with torch.no_grad():
            X_eval_t = torch.from_numpy(X_eval_feat).float().to(device)
            z = math.sqrt(2.0 / base_svm_params['D']) * torch.cos(X_eval_t @ base_svm_W + base_svm_b)
            preds_svm_zs = linear_zs(z).argmax(dim=1).cpu().numpy()
        accs_zs["SVM"] = accuracy_score(y_eval, preds_svm_zs)
        user_results_zs[user_id] = accs_zs

        # --- FASE CALIBRAZIONE ---
        user_tdcnn = copy.deepcopy(base_tdcnn)
        user_tdcnn.epochs = 5 
        user_tdcnn.fit(X_cal_raw, y_cal)
        accs_cal["TDCNN"] = accuracy_score(y_eval, user_tdcnn.predict(X_eval_raw))

        user_xgb = xgb.XGBClassifier(**base_xgb.get_params())
        user_xgb.fit(
            X_cal_feat, y_cal,
            xgb_model=base_xgb.get_booster(),
            eval_set=[(X_cal_feat, y_cal)],
            verbose=10,
        )
        accs_cal["XGBoost"] = accuracy_score(y_eval, user_xgb.predict(X_eval_feat))

        X_combined_knn = np.vstack([X_train_global_knn, X_cal_feat])
        y_combined_knn = np.hstack([y_train_global_knn, y_cal])
        X_user_cleaned, y_user_cleaned = faiss_enn_gpu(X_combined_knn, y_combined_knn, k=3)
        user_knn = FaissKNNClassifierGPU(n_neighbors=1, metric="manhattan")
        user_knn.fit(X_user_cleaned, y_user_cleaned)
        accs_cal["KNN"] = accuracy_score(y_eval, user_knn.predict(X_eval_feat))

        user_svm = RFFSVMClassifier(**base_svm_params)
        user_svm.max_epochs = 10 
        user_svm.le_, user_svm.classes_ = le, le.classes_
        linear_cal = nn.Linear(base_svm_W.shape[1], len(le.classes_)).to(device)
        linear_cal.load_state_dict(base_svm_state)
        opt = torch.optim.Adam(linear_cal.parameters(), lr=user_svm.lr)
        crit = nn.CrossEntropyLoss()
        X_cal_t = torch.from_numpy(X_cal_feat).float().to(device)
        y_cal_t = torch.from_numpy(y_cal).long().to(device)
        linear_cal.train()
        pbar_svm = tqdm(range(user_svm.max_epochs), desc="SVM epochs", leave=False)
        for _ in pbar_svm:
            opt.zero_grad()
            z = math.sqrt(2.0 / user_svm.D) * torch.cos(X_cal_t @ base_svm_W + base_svm_b)
            loss = crit(linear_cal(z), y_cal_t)
            loss.backward()
            opt.step()
            pbar_svm.set_postfix(loss=f"{loss.item():.4f}")
        user_svm.model_ = (base_svm_W, base_svm_b, linear_cal.eval())
        preds_svm_str = user_svm.predict(X_eval_feat)
        accs_cal["SVM"] = accuracy_score(y_eval, le.transform(preds_svm_str))

        user_results_cal[user_id] = accs_cal

        # --- SALVATAGGI SINGOLO UTENTE ---
        # 1. Salva la cache JSON delle accuratezze
        with open(results_cache_path, 'w') as f:
            json.dump({"zero_shot": accs_zs, "calibrated": accs_cal}, f)
            
        # 2. (Opzionale) Salva i pesi dei modelli calibrati
        if SAVE_CALIBRATED_MODELS:
            user_model_dir = CALIBRATED_MODELS_DIR / str(user_id)
            user_model_dir.mkdir(parents=True, exist_ok=True)
            
            user_tdcnn.save(str(user_model_dir / "tdcnn.pth"))
            user_xgb.save_model(str(user_model_dir / "xgboost.json"))
            joblib.dump({
                "X_store": X_user_cleaned, "y_store": y_user_cleaned,
                "n_neighbors": 1, "metric": "manhattan"
            }, user_model_dir / "knn.joblib")
            torch.save({
                "model_state_dict": linear_cal.state_dict(), 
                "W": base_svm_W, "b": base_svm_b, "params": base_svm_params
            }, user_model_dir / "svm.pt")


    # =============================================================================
    # 7. Analisi Statistica e Salvataggio Risultati Finali
    # =============================================================================
    print(f"\n[5/5] Analisi Statistica ed Esportazione...")
    df_zs = pd.DataFrame.from_dict(user_results_zs, orient='index')
    df_cal = pd.DataFrame.from_dict(user_results_cal, orient='index')
    
    # Crea una stringa report per salvare i risultati testuali
    report_lines = []
    
    msg_medie = f"\n--- MEDIE SUI {len(selected_users)} PAZIENTI ---"
    print(msg_medie)
    report_lines.append(msg_medie)
    
    for model in df_zs.columns:
        mean_zs = df_zs[model].mean() * 100
        mean_cal = df_cal[model].mean() * 100
        delta = mean_cal - mean_zs
        msg = f" {model:>8}: Zero-Shot = {mean_zs:.2f}% | Calibrato = {mean_cal:.2f}% | Delta = {delta:+.2f}%"
        print(msg)
        report_lines.append(msg)

    msg_wil1 = "\n--- TEST DI WILCOXON: LA CALIBRAZIONE MIGLIORA LE PRESTAZIONI? ---"
    print(msg_wil1)
    report_lines.append(msg_wil1)
    
    for model in df_zs.columns:
        try:
            stat_w, p_wilcoxon = stats.wilcoxon(df_zs[model], df_cal[model])
            is_sig = "★ MIGLIORAMENTO SIGNIFICATIVO" if p_wilcoxon < 0.05 else "Non Significativo"
        except ValueError:
            p_wilcoxon = 1.0
            is_sig = "Nessun cambiamento (Identici)"
        msg = f"{model:<8} (p-value: {p_wilcoxon:.5e}) -> {is_sig}"
        print(msg)
        report_lines.append(msg)

    msg_fried = "\n" + "="*50 + "\n CONFRONTO TRA MODELLI CALIBRATI (FRIEDMAN + WILCOXON)\n" + "="*50
    print(msg_fried)
    report_lines.append(msg_fried)
    
    stat, p_friedman = stats.friedmanchisquare(df_cal['TDCNN'], df_cal['SVM'], df_cal['XGBoost'], df_cal['KNN'])
    msg_fp = f"Friedman p-value: {p_friedman:.5e}"
    print(msg_fp)
    report_lines.append(msg_fp)

    if p_friedman >= 0.05:
        msg = "-> Nessuna differenza statistica significativa tra i modelli Calibrati."
        print(msg)
        report_lines.append(msg)
    else:
        msg = "-> Differenza SIGNIFICATIVA. Test Post-Hoc vs TDCNN:\n"
        print(msg)
        report_lines.append(msg)
        
        alpha_bonferroni = 0.05 / 3
        baselines = ['SVM', 'XGBoost', 'KNN']
        for baseline in baselines:
            try:
                stat_w, p_wilcoxon = stats.wilcoxon(df_cal['TDCNN'], df_cal[baseline])
                is_sig = "★ SIGNIFICATIVO" if p_wilcoxon < alpha_bonferroni else "Non Sig."
            except ValueError:
                p_wilcoxon = 1.0
                is_sig = "Nessun cambiamento (Identici)"
                
            diff_media = (df_cal['TDCNN'].mean() - df_cal[baseline].mean()) * 100
            msg_ph = f"TDCNN vs {baseline:<7} | Vantaggio TDCNN: {diff_media:+.2f}% | p-value: {p_wilcoxon:.5e} -> {is_sig}"
            print(msg_ph)
            report_lines.append(msg_ph)

    # --- ESPORTAZIONE FINALE (CSV E REPORT TESTUALE) ---
    csv_zs_path = RESULTS_EXPORT_DIR / "zero_shot_metrics.csv"
    csv_cal_path = RESULTS_EXPORT_DIR / "calibrated_metrics.csv"
    report_path = RESULTS_EXPORT_DIR / "statistical_report.txt"
    
    df_zs.to_csv(csv_zs_path, index_label="user_id")
    df_cal.to_csv(csv_cal_path, index_label="user_id")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # --- Export per visualize_calibration_results.py ---
    model_order = ["TDCNN", "SVM", "XGBoost", "KNN"]
    df_zs_exp = df_zs[[c for c in model_order if c in df_zs.columns]]
    df_cal_exp = df_cal[[c for c in model_order if c in df_cal.columns]]
    df_zs_exp.to_csv(CALIBRATION_EXPORT_DIR / "zero_shot_accuracies.csv", index_label="")
    df_cal_exp.to_csv(CALIBRATION_EXPORT_DIR / "calibrated_accuracies.csv", index_label="")
    df_all = pd.concat([df_zs_exp.add_suffix("_zs"), df_cal_exp.add_suffix("_cal")], axis=1)
    df_all.to_csv(CALIBRATION_EXPORT_DIR / "all_results.csv")
    wilcoxon_cal = {}
    for model in model_order:
        if model in df_zs.columns:
            try:
                stat_w, p_w = stats.wilcoxon(df_zs[model], df_cal[model])
                wilcoxon_cal[model] = {"statistic": float(stat_w), "p_value": float(p_w)}
            except ValueError:
                wilcoxon_cal[model] = {"statistic": 0, "p_value": 1.0}
    wilcoxon_posthoc = {}
    for baseline in ["SVM", "XGBoost", "KNN"]:
        if baseline in df_cal.columns:
            try:
                stat_w, p_w = stats.wilcoxon(df_cal["TDCNN"], df_cal[baseline])
                adv = (df_cal["TDCNN"].mean() - df_cal[baseline].mean()) * 100
                wilcoxon_posthoc[f"TDCNN_vs_{baseline}"] = {
                    "statistic": float(stat_w), "p_value": float(p_w), "tdcnn_advantage_pct": float(adv)
                }
            except ValueError:
                wilcoxon_posthoc[f"TDCNN_vs_{baseline}"] = {"statistic": 0, "p_value": 1.0, "tdcnn_advantage_pct": 0}
    calib_summary = {
        "config": {
            "calibration_reps": CALIBRATION_REPS,
            "random_seed": RANDOM_SEED,
            "max_test_users": MAX_TEST_USERS,
            "selected_users": [str(u) for u in selected_users],
        },
        "means_zero_shot": {m: float(df_zs[m].mean() * 100) for m in model_order if m in df_zs.columns},
        "means_calibrated": {m: float(df_cal[m].mean() * 100) for m in model_order if m in df_cal.columns},
        "wilcoxon_calibration_effect": wilcoxon_cal,
        "friedman_postcalibration": {"statistic": float(stat), "p_value": float(p_friedman)},
        "wilcoxon_posthoc_tdcnn_vs_baselines": wilcoxon_posthoc,
    }
    with open(CALIBRATION_EXPORT_DIR / "calibration_stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(calib_summary, f, indent=2)
        
    print(f"\n✓ Salvataggio completato! Risultati esportati in: {RESULTS_EXPORT_DIR}")
    print(f"✓ Dati per visualizzazione esportati in: {CALIBRATION_EXPORT_DIR}")

if __name__ == "__main__":
    main()