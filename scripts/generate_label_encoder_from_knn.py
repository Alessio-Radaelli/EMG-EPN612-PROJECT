import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Load the KNN joblib file
knn_path = "models/18f/knn18_faiss_gpu_enn_manhattan_k1_wuniform.joblib"
model_data = joblib.load(knn_path)
y_store = model_data["y_store"]

# Fit a new LabelEncoder on the unique labels
le = LabelEncoder()
le.fit(np.unique(y_store))

# Ensure output directory exists
os.makedirs("preprocessed_output", exist_ok=True)

# Save the label encoder
joblib.dump(le, "preprocessed_output/label_encoder.pkl")
print("LabelEncoder saved to preprocessed_output/label_encoder.pkl")
