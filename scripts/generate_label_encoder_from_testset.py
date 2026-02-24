import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

df_test = pd.read_parquet("preprocessed_output/dataset_TESTING_reduced18.parquet")
le = LabelEncoder()
le.fit(df_test["label"].values)
os.makedirs("preprocessed_output", exist_ok=True)
joblib.dump(le, "preprocessed_output/label_encoder.pkl")
print("LabelEncoder saved to preprocessed_output/label_encoder.pkl (fit on string labels)")
