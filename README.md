# EMG-EPN612 Gesture Recognition Project

A machine learning project for hand gesture recognition using EMG (Electromyography) signals from the EMG-EPN612 dataset.

## Dataset

The **EMG-EPN612** dataset contains EMG signals recorded from 612 users using the Myo Armband for hand gesture recognition.

### Dataset Structure

- **Training Split**: 306 users (for training & validation)
- **Testing Split**: 306 users (for training & testing)
- **Total**: 612 unique users

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

- **Training samples**: 150 (25 per gesture × 6 gestures)
- **Testing samples**: 150 (25 per gesture × 6 gestures)
- **Total per user**: 300 samples

### Sensor Data

Each sample contains 5 seconds of recording at:
- **EMG**: 8 channels @ 200 Hz (~992 samples)
- **Gyroscope**: x, y, z @ 50 Hz (~249 samples)
- **Accelerometer**: x, y, z @ 50 Hz (~249 samples)
- **Quaternion**: w, x, y, z @ 50 Hz (~249 samples)

## Project Structure

```
EMG-EPN612 project/
├── EMG-EPN612 Dataset/     # Dataset (not included, download separately)
│   ├── trainingJSON/       # 306 users for training/validation
│   └── testingJSON/        # 306 users for training/testing
├── notebooks/              # Jupyter notebooks for analysis
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/EMG-EPN612-project.git
cd EMG-EPN612-project
```

2. Create a conda environment:
```bash
conda create -n emg-epn612 python=3.11
conda activate emg-epn612
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from [EMG-EPN612 Official Page](https://laboratorio-ia.epn.edu.ec/en/resources/dataset/2020_emg_dataset_612) and extract it to the project folder.

## Usage

Open the Jupyter notebooks in the `notebooks/` folder to explore the dataset and run experiments:

```bash
jupyter lab
```

## Dataset Download

The EMG-EPN612 dataset can be downloaded from:
- [Official Dataset Page](https://laboratorio-ia.epn.edu.ec/en/resources/dataset/2020_emg_dataset_612)

## References

- EMG-EPN612 Dataset: Laboratorio de Investigación en Inteligencia y Visión Artificial, Escuela Politécnica Nacional

## License

This project is for educational and research purposes.
