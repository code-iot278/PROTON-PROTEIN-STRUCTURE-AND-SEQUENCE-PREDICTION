# =========================================================
# BiLSTM-Based Feature Extraction (CSV → CSV)
# =========================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load CSV file
# -----------------------------
input_csv = "input.csv"     # <-- change path
output_csv = "bilstm_features.csv"

df = pd.read_csv(input_csv)

# -----------------------------
# 2. Select numerical columns
# -----------------------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
data = df[num_cols].values.astype(np.float32)

# -----------------------------
# 3. Normalize data (Z-score)
# -----------------------------
scaler = StandardScaler()
data = scaler.fit_transform(data)

# -----------------------------
# 4. Create sequences
# -----------------------------
SEQ_LEN = 20

def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

X_seq = create_sequences(data, SEQ_LEN)

# Convert to tensor
X_tensor = torch.tensor(X_seq)

# -----------------------------
# 5. BiLSTM Feature Extractor
# -----------------------------
class BiLSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        _, (hn, _) = self.bilstm(x)
        features = torch.cat((hn[-2], hn[-1]), dim=1)
        return features

# Model parameters
INPUT_SIZE = X_tensor.shape[2]
HIDDEN_SIZE = 64

model = BiLSTMFeatureExtractor(INPUT_SIZE, HIDDEN_SIZE)

# -----------------------------
# 6. Extract features
# -----------------------------
with torch.no_grad():
    features = model(X_tensor)

features_np = features.numpy()

# -----------------------------
# 7. Save features to CSV
# -----------------------------
feature_columns = [f"BiLSTM_Feature_{i+1}" for i in range(features_np.shape[1])]
df_features = pd.DataFrame(features_np, columns=feature_columns)

df_features.to_csv(output_csv, index=False)

print("BiLSTM feature extraction completed.")
print("Output saved to:", output_csv)
print("Feature shape:", features_np.shape)
