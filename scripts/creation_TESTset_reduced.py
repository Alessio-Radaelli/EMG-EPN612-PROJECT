import pandas as pd

# Path to the original test dataset
input_path = 'preprocessed_output/dataset_TEST.parquet'

# Output paths
output_path_18 = 'preprocessed_output/dataset_TESTING_reduced18.parquet'
output_path_36 = 'preprocessed_output/dataset_TESTING_reduced36.parquet'

# Top features from feature ranking
features_18 = [
    'ch4_MFL','ch4_MSR','ch5_MFL','ch4_RMS','ch3_RMS','ch4_DASDV','ch4_IAV','ch6_MFL','ch7_MFL','ch3_MFL',
    'ch4_LS','ch3_MSR','ch8_MFL','ch2_RMS','ch1_RMS','ch2_MFL','ch3_LS','ch1_MFL'
]
features_36 = [
    'ch4_MFL','ch4_MSR','ch5_MFL','ch4_RMS','ch3_RMS','ch4_DASDV','ch4_IAV','ch6_MFL','ch7_MFL','ch3_MFL',
    'ch4_LS','ch3_MSR','ch8_MFL','ch2_RMS','ch1_RMS','ch2_MFL','ch3_LS','ch1_MFL','ch2_LS','ch3_IAV','ch5_MSR',
    'ch5_RMS','ch5_DASDV','ch7_MSR','ch8_RMS','ch4_VAR','ch1_LS','ch2_MSR','ch1_MSR','ch7_RMS','ch8_MSR',
    'ch7_LS','ch3_DASDV','ch6_MSR','ch6_LS','ch6_IAV','ch5_LS'
]

# Try to include metadata columns if present
metadata_cols = ['user','gesture','timestamp','segment_id']

def get_columns(df, feature_list):
    cols = feature_list.copy()
    # Always include label, user, sample_id, window_idx if present, but only once
    for col in ['label', 'user', 'sample_id', 'window_idx']:
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols

def main():
    df = pd.read_parquet(input_path)
    # 18 features + 4 metadata
    cols_18 = features_18 + [col for col in ['label', 'user', 'sample_id', 'window_idx'] if col in df.columns]
    # 36 features + 4 metadata
    cols_36 = features_36 + [col for col in ['label', 'user', 'sample_id', 'window_idx'] if col in df.columns]
    df_reduced18 = df[cols_18]
    df_reduced36 = df[cols_36]
    df_reduced18.to_parquet(output_path_18)
    df_reduced36.to_parquet(output_path_36)
    # Verify first 18 columns in reduced36 match reduced18
    match = df_reduced36.columns[:18].tolist() == df_reduced18.columns[:18].tolist()
    print(f"Created {output_path_18} and {output_path_36}.")
    print(f"First 18 columns match: {match}")

if __name__ == '__main__':
    main()
