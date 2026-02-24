# EMG-EPN612 Gesture Recognition Project

A comprehensive machine learning pipeline for **hand gesture recognition** using electromyography (EMG) signals from the **EMG-EPN612** dataset. The project implements multiple classifier families (traditional ML and deep learning), feature selection, preprocessing pipelines, user calibration, and inference benchmarking.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Pipeline Workflow](#pipeline-workflow)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Models](#models)
8. [Scripts Reference](#scripts-reference)
9. [Output Artifacts](#output-artifacts)
10. [Calibration](#calibration)
11. [Benchmarking](#benchmarking)
12. [Notebooks](#notebooks)
13. [References](#references)
14. [License](#license)

---

## Overview

This project processes EMG signals from the Myo Armband (8 channels @ 200 Hz) to recognize six hand gestures. It features:

- **Signal conditioning**: Bandpass filter (20–95 Hz), 50 Hz notch filter, segmentation
- **TD9 feature extraction**: 9 time-domain features per channel (72 total)
- **Feature selection**: Cross-validated ranking (ANOVA, Mutual Information, XGBoost, RFE, correlation filter)
- **Multiple classifiers**: KNN (FAISS GPU), SVM (RFF approximation), XGBoost, TDCNN (deep learning)
- **User calibration**: Per-user fine-tuning using few-shot data from new users
- **Inference benchmarking**: Latency and throughput comparison across models

---

## Dataset

The **EMG-EPN612** dataset contains EMG signals from **612 users** recorded with the Myo Armband.

### Dataset Structure

| Split | Users | Source Folder | Use |
|-------|-------|---------------|-----|
| **Dataset A** | 459 (1–459) | trainingJSON (1–306) + testingJSON (1–153) | Training & validation |
| **Dataset B** | 153 (460–612) | testingJSON (154–306) | Hold-out testing |

### Gestures (6 classes)

| Label | Gesture |
|-------|---------|
| 0 | noGesture |
| 1 | fist |
| 2 | waveIn |
| 3 | waveOut |
| 4 | open |
| 5 | pinch |

### Samples per User

- **150 registrations** (25 per gesture × 6 gestures)
- **300 total samples** per user when including test data (for split users)

### Sensor Data

- **EMG**: 8 channels @ 200 Hz (~992 samples per 5 s trial)
- **Gyroscope**: x, y, z @ 50 Hz
- **Accelerometer**: x, y, z @ 50 Hz
- **Quaternion**: w, x, y, z @ 50 Hz

### Download

- [EMG-EPN612 Official Page](https://laboratorio-ia.epn.edu.ec/en/resources/dataset/2020_emg_dataset_612)

---

## Project Structure

```
EMG-EPN612 project/
├── EMG-EPN612 Dataset/           # Dataset (download separately)
│   ├── trainingJSON/             # user1 … user306
│   └── testingJSON/              # user1 … user306
├── datasets/                     # Built datasets (pkl)
│   ├── dataset_A.pkl             # 459 users for training
│   └── dataset_B.pkl              # 153 users for testing
├── preprocessed_output/          # Parquet/NPZ outputs
│   ├── dataset_TRAINING.parquet   # 72 features, full
│   ├── dataset_TEST.parquet
│   ├── dataset_TRAINING_reduced18.parquet
│   ├── dataset_TRAINING_reduced36.parquet
│   ├── dataset_TESTING_reduced18.parquet
│   ├── dataset_TESTING_reduced36.parquet
│   ├── dataset_TESTING.npz       # For TDCNN (raw windows)
│   └── calibration_results/      # Calibration CSVs, plots
├── preprocessed_users/           # Per-user calibration data
├── models/
│   ├── 18f/                     # 18-feature models
│   │   ├── knn_faiss_gpu_enn_*.joblib
│   │   ├── svm_val_best18.pt
│   │   ├── xgboost18_best_halving.json
│   │   ├── tdcnn_emg_model.pth
│   │   ├── feature_ranking_ranks.csv
│   │   └── benchmark_inference_results.json
│   ├── 36f/                     # 36-feature models
│   ├── 72f/                     # 72-feature models
│   └── tdcnn_emg_model.pth       # Root TDCNN model
├── scripts/                     # All Python scripts
├── notebooks/                   # Jupyter notebooks
├── requirements.txt
└── README.md
```

---

## Pipeline Workflow

### Data Flow

```
EMG-EPN612 JSON
       │
       ▼
  build_datasets_AB.py  ──► dataset_A.pkl, dataset_B.pkl
       │
       ├──────────────────────────────────────────────────┐
       │                                                    │
       ▼                                                    ▼
preprocess_pipeline.py                        Creation_dataset_TESTING.py
(dataset_A)                                   (dataset_B)
       │                                                    │
       ▼                                                    ▼
dataset_TRAINING.parquet (72f)               dataset_TEST.parquet (72f)
       │                                                    │
       ▼                                                    │
feature_selection.py                                         │
       │                                                     │
       ├──► dataset_TRAINING_reduced18/36.parquet            │
       │                                                     │
       ▼                                                     ▼
creation_TESTset_reduced.py  ──► dataset_TESTING_reduced18/36.parquet
       │
       ▼
train_tdcnn.py (raw windows from dataset_B)  ──► dataset_TESTING.npz
```

### Preprocessing Steps

1. **Phase 1 – Signal conditioning**: 2nd-order Butterworth bandpass (20–95 Hz), 50 Hz notch (Q=30), `filtfilt` (zero-phase)
2. **Phase 2 – Segmentation**: Ground-truth cropping for gestures; center crop for noGesture
3. **Phase 3 – Windowing**: 200 ms windows (40 samples), 50 ms step (75% overlap)
4. **Phase 4 – TD9 features**: LS, MFL, MSR, WAMP, ZC, RMS, IAV, DASDV, VAR per channel
5. **Phase 5 – Outlier detection**: IQR voting (>25% of features out-of-bounds → drop window)
6. **Phase 6 – Z-score normalization**: Subject-specific (per user)

---

## Installation

### Prerequisites

- Python 3.11+
- CUDA (for GPU models: TDCNN, KNN/FAISS, RFF-SVM)
- ~8 GB RAM recommended; GPU recommended for training

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/EMG-EPN612-project.git
cd EMG-EPN612-project
```

2. **Create a conda environment**

```bash
conda create -n emg-epn612 python=3.11
conda activate emg-epn612
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Optional: FAISS and tqdm (for KNN and progress bars)**

```bash
pip install faiss-gpu tqdm
```

- Use `faiss-cpu` instead of `faiss-gpu` if no NVIDIA GPU is available (KNN will run on CPU).

5. **Download the dataset**

Download from [EMG-EPN612 Official Page](https://laboratorio-ia.epn.edu.ec/en/resources/dataset/2020_emg_dataset_612) and extract to `EMG-EPN612 Dataset/` in the project root.

---

## Usage

All commands assume you are in the project root with the conda environment activated.

### Full Pipeline (from raw JSON to trained models)

```bash
# 1. Build datasets A and B from JSON
python scripts/build_datasets_AB.py

# 2. Preprocess training data (72 features, outlier removal, z-score)
python scripts/preprocess_pipeline.py

# 3. Preprocess testing data
python scripts/Creation_dataset_TESTING.py

# 4. Feature selection (rank 72 features, output 18/36 reduced sets)
python scripts/feature_selection.py

# 5. Create reduced test sets
python scripts/creation_TESTset_reduced.py

# 6. Train TDCNN on raw windows (generates dataset_TESTING.npz)
python scripts/train_tdcnn.py

# 7. Train ML models (KNN, SVM, XGBoost) on 18/36/72 features
python scripts/train_knn.py --n_features 18
python scripts/train_svm.py --n_features 18
python scripts/train_xgboost.py  # default 18 features

# 8. Evaluate TDCNN
python scripts/eval_tdcnn.py
```

### Per-model training

| Model | Script | Default features |
|-------|--------|-------------------|
| KNN   | `train_knn.py`   | 18 (configurable) |
| SVM   | `train_svm.py`   | 18 (configurable) |
| XGBoost | `train_xgboost.py` | 18 |
| TDCNN | `train_tdcnn.py` | Raw (8×40 windows) |

### Visualization & comparison

```bash
python scripts/comparison_models.py       # Model comparison plots
python scripts/visualize_18f_results.py  # 18f model metrics tables
python scripts/visualize_calibration_results.py  # Calibration plots
```

### Calibration (per-user fine-tuning)

```bash
python scripts/calibration_on_users.py
```

### Inference benchmark

```bash
python scripts/benchmark_inference_speed.py
```

---

## Models

### Traditional ML (feature-based)

| Model | Input | Key characteristics |
|-------|--------|----------------------|
| **KNN** | 18/36/72 TD9 features | FAISS GPU, Edited NN (ENN) denoising, Manhattan/Euclidean |
| **SVM** | 18/36/72 TD9 features | RFF kernel approximation, GPU, multiclass hinge loss |
| **XGBoost** | 18/36/72 TD9 features | Halving search, gradient boosting |

### Deep learning

| Model | Input | Architecture |
|-------|--------|--------------|
| **TDCNN** | Raw EMG (40×8 windows) | Time-delay CNN + ECA (Efficient Channel Attention), causal convolutions |

### Top 18 features (from cross-validated ranking)

```
ch4_MFL, ch4_MSR, ch5_MFL, ch4_RMS, ch3_RMS, ch4_DASDV, ch4_IAV, ch6_MFL,
ch7_MFL, ch3_MFL, ch4_LS, ch3_MSR, ch8_MFL, ch2_RMS, ch1_RMS, ch2_MFL,
ch3_LS, ch1_MFL
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `build_datasets_AB.py` | Load JSON → `dataset_A.pkl`, `dataset_B.pkl` |
| `preprocess_pipeline.py` | Train preprocessing: filter, TD9, outliers, z-score |
| `Creation_dataset_TESTING.py` | Test preprocessing (no outlier removal) |
| `feature_selection.py` | Cross-validated feature ranking, reduced parquets |
| `creation_TESTset_reduced.py` | Filter test set to 18/36 features |
| `train_knn.py` | Train FAISS KNN with ENN denoising |
| `train_svm.py` | Train RFF-SVM with successive halving |
| `train_xgboost.py` | Train XGBoost with halving search |
| `train_tdcnn.py` | Train TDCNN on raw windows |
| `eval_tdcnn.py` | Evaluate TDCNN on test set |
| `calibration_on_users.py` | Per-user calibration (few-shot) |
| `benchmark_inference_speed.py` | Latency/throughput benchmark |
| `comparison_models.py` | Compare model accuracies and F1 |
| `comparison_feature_selection.py` | Compare feature subsets |
| `visualize_18f_results.py` | Metrics tables for 18f models |
| `visualize_calibration_results.py` | Calibration plots and stats |
| `ROC_svm_xgboost.py` | ROC curves for SVM and XGBoost |
| `ROC_tcdnn.py` | ROC curves for TDCNN |

---

## Output Artifacts

### Preprocessed data

- `preprocessed_output/dataset_TRAINING.parquet` — 72 features, training
- `preprocessed_output/dataset_TEST.parquet` — 72 features, testing
- `preprocessed_output/dataset_TRAINING_reduced{18,36}.parquet`
- `preprocessed_output/dataset_TESTING_reduced{18,36}.parquet`
- `preprocessed_output/dataset_TESTING.npz` — Raw windows for TDCNN

### Models

- `models/18f/knn_faiss_gpu_enn_*.joblib`
- `models/18f/svm_val_best18.pt`
- `models/18f/xgboost18_best_halving.json`
- `models/tdcnn_emg_model.pth` or `models/18f/tdcnn_emg_model.pth`

### Feature ranking

- `models/feature_ranking_ranks.csv`
- `models/feature_ranking_scores.csv`
- `models/18f/feature_ranking_ranks.csv`

### Benchmark

- `models/18f/benchmark_inference_results.json`
- `models/18f/benchmark_stats_analysis.json`
- `models/18f/benchmark_stats_table.png`

### Calibration

- `preprocessed_output/calibration_results/*.csv`
- `preprocessed_output/calibration_results/plots/*.png`

---

## Calibration

User calibration improves zero-shot performance by fine-tuning on a small amount of data from new users:

1. Load base model (KNN, SVM, XGBoost, TDCNN)
2. For each test user: combine base training + user’s few samples
3. Retrain or adapt the model on the combined set
4. Evaluate on the user’s hold-out samples

Run:

```bash
python scripts/calibration_on_users.py
```

Output: `zero_shot_accuracies.csv`, `calibrated_accuracies.csv`, per-user calibrated models (if enabled).

---

## Benchmarking

Inference speed is benchmarked for:

- **Single-sample latency** (batch size 1)
- **High throughput** (batch size N)
- **Preprocessing** (TD9 extraction, z-score) vs **inference only**

Typical order (fastest → slowest): **TDCNN** < **SVM** < **XGBoost** < **KNN** (median latency).

Run:

```bash
python scripts/benchmark_inference_speed.py
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `dataset_exploration.ipynb` | Inspect JSON structure, splits, labels |
| `dataset_balance_analysis.ipynb` | Class balance and sample counts |
| `compile_training_file.ipynb` | Compile/create training data |

Start Jupyter:

```bash
jupyter lab
```

---

## References

- **EMG-EPN612 Dataset**: Laboratorio de Investigación en Inteligencia y Visión Artificial, Escuela Politécnica Nacional
- **ECA-Net**: Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"

---

## License

This project is for educational and research purposes.
