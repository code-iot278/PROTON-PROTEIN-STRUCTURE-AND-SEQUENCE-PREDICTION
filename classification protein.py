# =========================================================
# Trans-CNN-TCN for Protein Structure Classification
# with Full Metrics Evaluation
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)

# -----------------------------
# 1. Load CSV
# -----------------------------
input_csv = "rco_selected_features.csv"  # CSV with selected features + label
LABEL_COLUMN = "label"                  # Name of label column

df = pd.read_csv(input_csv)

X = df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)
y = df[LABEL_COLUMN].values.astype(int)

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Normalize Features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train).unsqueeze(1)  # (batch, seq_len=1, features)
X_test = torch.tensor(X_test).unsqueeze(1)

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# -----------------------------
# 4. TCN Block
# -----------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.relu(self.conv(x))
        res = x if self.downsample is None else self.downsample(x)
        return out[:, :, :res.size(2)] + res

# -----------------------------
# 5. Trans-CNN-TCN Model
# -----------------------------
class TransCNN_TCN(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # CNN
        self.cnn = nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1)

        # TCN
        self.tcn1 = TemporalBlock(128, 128, kernel_size=3, dilation=1)
        self.tcn2 = TemporalBlock(128, 128, kernel_size=3, dilation=2)

        # Classifier
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.transformer(x)

        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = F.relu(self.cnn(x))

        # TCN
        x = self.tcn1(x)
        x = self.tcn2(x)

        # Global Average Pooling
        x = torch.mean(x, dim=2)

        return self.fc(x)

# -----------------------------
# 6. Model Setup
# -----------------------------
NUM_CLASSES = len(np.unique(y))
FEATURE_DIM = X_train.shape[2]

model = TransCNN_TCN(FEATURE_DIM, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 7. Training Loop
# -----------------------------
EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# -----------------------------
# 8. Full Evaluation Metrics
# -----------------------------
model.eval()
with torch.no_grad():
    logits = model(X_test)
    preds = torch.argmax(logits, dim=1).numpy()

y_true = y_test.numpy()

# Confusion Matrix
cm = confusion_matrix(y_true, preds)
if NUM_CLASSES == 2:
    TN, FP, FN, TP = cm.ravel()
else:
    # For multi-class: calculate metrics macro average
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)
    TP, FP, FN, TN = TP.sum(), FP.sum(), FN.sum(), TN.sum()

# Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = precision_score(y_true, preds, average='macro')
recall = recall_score(y_true, preds, average='macro')
f1 = f1_score(y_true, preds, average='macro')
specificity = TN / (TN + FP)
mcc = matthews_corrcoef(y_true, preds)
npv = TN / (TN + FN)
fpr = FP / (FP + TN)
fnr = FN / (FN + TP)

# -----------------------------
# 9. Display Results
# -----------------------------
print("\n================ MODEL PERFORMANCE =================")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1-Score      : {f1:.4f}")
print(f"Specificity   : {specificity:.4f}")
print(f"MCC           : {mcc:.4f}")
print(f"NPV           : {npv:.4f}")
print(f"FPR           : {fpr:.4f}")
print(f"FNR           : {fnr:.4f}")
print("===================================================")
