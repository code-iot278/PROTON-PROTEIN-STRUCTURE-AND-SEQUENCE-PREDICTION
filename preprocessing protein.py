import pandas as pd

# Load CSV
csv_path = "input.csv"   # change path
df = pd.read_csv(csv_path)

# Apply Z-score normalization (column-wise)
df_zscore = (df - df.mean()) / df.std()

# Save normalized CSV
df_zscore.to_csv("zscore_normalized.csv", index=False)

print("Z-score normalization completed and saved.")

import pandas as pd

# -------------------------------
# Load CSV file
# -------------------------------
input_csv = "input.csv"     # change path if needed
df = pd.read_csv(input_csv)

# -------------------------------
# Select numeric columns only
# -------------------------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# -------------------------------
# IQR filtering
# -------------------------------
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

# Keep rows within IQR range
df_iqr = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) |
              (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# -------------------------------
# Save cleaned CSV
# -------------------------------
df_iqr.to_csv("iqr_filtered.csv", index=False)

print("IQR filtering completed. Cleaned file saved as iqr_filtered.csv")
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("input.csv")

# Select numeric columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Apply log transformation
df[num_cols] = np.log1p(df[num_cols])   # log(1 + x)

# Save result
df.to_csv("log_transformed.csv", index=False)

print("Log transformation applied successfully.")
