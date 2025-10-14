import pandas as pd

# Input and output file paths
input_path = "/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/Predictive Maintenance/output_simple/telemetry_labeled_with_features_7d.csv"
output_path = "/Users/gowthamsivaraman/Desktop/Vehicle Maintenance Project/Predictive Maintenance/output_simple/single_row_index6.csv"

# Read CSV
df = pd.read_csv(input_path)

# Select only the 6th index (7th row) and keep header
df_subset = df.iloc[[4]]  # zero-based index

# Save new CSV with header
df_subset.to_csv(output_path, index=False)

print(f"✅ Saved new CSV with header and 6th index to: {output_path}")
