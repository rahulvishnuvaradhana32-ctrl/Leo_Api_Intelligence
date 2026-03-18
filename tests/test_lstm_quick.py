#!/usr/bin/env python3
"""Quick test: dataset + model + 1 epoch training"""
import sys
print("=== test_lstm_quick.py start ===", file=sys.stderr, flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

print("Imports OK", file=sys.stderr, flush=True)

# Minimal dataset
n_samples = 200
np.random.seed(42)
data = {
    'response_time': np.random.exponential(100, n_samples) + np.random.normal(0, 20, n_samples),
    'status_code': np.random.choice([200,201,400,404,500], n_samples, p=[0.7,0.1,0.1,0.05,0.05]),
    'request_count': np.random.poisson(50, n_samples),
    'hour': np.random.randint(0, 24, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),
    'response_time_rolling_mean': np.random.exponential(100, n_samples),
    'response_time_rolling_std': np.abs(np.random.normal(15, 5, n_samples)),
    'error_rate_rolling': np.random.beta(2, 20, n_samples),
    'response_time_variance': np.abs(np.random.normal(20, 10, n_samples)),
    'error_volatility': np.random.exponential(0.1, n_samples),
}
df = pd.DataFrame(data)
df['success'] = (df['status_code'] < 400).astype(int)

print(f"Dataset shape: {df.shape}", file=sys.stderr, flush=True)

class TimeSeriesDataset(Dataset):
    def __init__(self, df, sequence_length, horizons, feature_cols=None):
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.feature_cols = feature_cols or [c for c in df.columns if c != 'success']
        self.numeric_cols = self.feature_cols
        self._data = df[self.feature_cols].to_numpy(dtype=np.float32)
        self._targets = df['success'].to_numpy(dtype=np.float32)
        self.max_start = len(self._data) - self.sequence_length - max(self.horizons) + 1
        if self.max_start < 1:
            raise ValueError('Dataframe too small')

    def __len__(self):
        return max(0, self.max_start)

    def __getitem__(self, idx):
        start = idx
        end = start + self.sequence_length
        seq = self._data[start:end].copy()
        targets = []
        for h in self.horizons:
            target_idx = end + h - 1
            success_val = self._targets[target_idx]
            targets.append(float(1 - success_val))
        seq_tensor = torch.from_numpy(seq)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        return {'sequence': seq_tensor, 'targets': target_tensor}

feature_cols = ['response_time', 'status_code', 'request_count', 'hour', 'day_of_week',
                'response_time_rolling_mean', 'response_time_rolling_std', 'error_rate_rolling',
                'response_time_variance', 'error_volatility']
dataset = TimeSeriesDataset(df, sequence_length=10, horizons=[1,5,15], feature_cols=feature_cols)
print(f"Dataset len: {len(dataset)}", file=sys.stderr, flush=True)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                       generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print(f"Data split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}", file=sys.stderr, flush=True)

class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiHorizonLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

device = torch.device('cpu')
model = MultiHorizonLSTM(len(feature_cols), 32, 2, 3).to(device)
print("Model created", file=sys.stderr, flush=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        X = batch['sequence'].to(device)
        y = batch['targets'].to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / max(1, len(train_loader))
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}", file=sys.stderr, flush=True)

# Evaluate
model.eval()
all_targets = []
all_probas = []
with torch.no_grad():
    for batch in test_loader:
        X = batch['sequence'].to(device)
        y = batch['targets'].to(device)
        outputs = model(X)
        probas = torch.sigmoid(outputs)
        all_targets.append(y.cpu().numpy())
        all_probas.append(probas.cpu().numpy())
all_targets = np.vstack(all_targets)
all_probas = np.vstack(all_probas)

print(f"Test targets shape: {all_targets.shape}, probas shape: {all_probas.shape}", file=sys.stderr, flush=True)

from sklearn.metrics import roc_auc_score
for i, h in enumerate([1, 5, 15]):
    y_true = all_targets[:, i]
    y_score = all_probas[:, i]
    if len(np.unique(y_true)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(y_true, y_score)
    print(f"horizon_{h}: AUC={auc:.3f}", file=sys.stderr, flush=True)

print("=== test_lstm_quick.py done ===", file=sys.stderr, flush=True)
