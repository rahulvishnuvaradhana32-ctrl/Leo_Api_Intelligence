#!/usr/bin/env python3
"""Fast hyperparameter tuning with improved feature preprocessing and class balancing."""

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print('=== Fast Hyperparameter Tuning v2 (with preprocessing) ===\n')


def generate_dataset(n_samples=100000):
    """Generate balanced synthetic dataset."""
    print(f"Generating {n_samples:,} samples...")
    np.random.seed(42)
    
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
    hour = timestamps.hour.values
    dow = timestamps.dayofweek.values
    
    data = {
        'timestamp': timestamps,
        'response_time': np.random.exponential(100, n_samples) + 20 * np.sin(2 * np.pi * hour / 24),
        'status_code': np.random.choice([200, 201, 400, 404, 500], n_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        'request_count': np.random.poisson(50, n_samples),
        'hour': hour,
        'day_of_week': dow,
        'is_weekend': (dow >= 5).astype(int),
        'is_market_hours': ((hour >= 9) & (hour <= 16)).astype(int),
    }
    
    df = pd.DataFrame(data)
    df['success'] = (df['status_code'] < 400).astype(int)
    
    # Rolling statistics
    for col in ['response_time']:
        for window in [10, 30]:
            df[f'{col}_ma_{window}'] = df[col].rolling(window, min_periods=1).mean()
            df[f'{col}_std_{window}'] = df[col].rolling(window, min_periods=1).std().fillna(0)
    
    # Lag features
    for lag in [1, 5]:
        df[f'response_time_lag_{lag}'] = df['response_time'].shift(lag).fillna(0)
        df[f'success_lag_{lag}'] = df['success'].shift(lag).fillna(0)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Add failure clustering
    failure_idx = np.where(df['success'] == 0)[0]
    for idx in failure_idx[:len(failure_idx)//2]:
        for offset in range(1, 5):
            if idx + offset < len(df) and np.random.random() < 0.4:
                df.loc[idx + offset, 'success'] = 0
    
    print(f"Generated: {len(df)} samples, failure rate: {1 - df['success'].mean():.2%}\n")
    return df


class TimeSeriesDataset(Dataset):
    def __init__(self, df, sequence_length, horizons, feature_cols=None):
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.feature_cols = feature_cols or [c for c in df.columns if c not in ['timestamp', 'success']]
        
        # Standardize features
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(df[self.feature_cols])
        
        self._data = scaled_data.astype(np.float32)
        self._targets = df['success'].to_numpy(dtype=np.float32)
        self.max_start = len(self._data) - self.sequence_length - max(self.horizons) + 1

    def __len__(self):
        return max(0, self.max_start)

    def __getitem__(self, idx):
        start = idx
        end = start + self.sequence_length
        seq = self._data[start:end].copy()
        targets = []
        for h in self.horizons:
            target_idx = end + h - 1
            if target_idx < len(self._targets):
                targets.append(float(1 - self._targets[target_idx]))
            else:
                targets.append(0.0)
        return {
            'sequence': torch.from_numpy(seq),
            'targets': torch.tensor(targets, dtype=torch.float32)
        }


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_horizons, dropout=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_horizons)
        )

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        out = h[-1]  # Use last hidden state
        out = self.fc(out)
        return out


def train_and_eval(model, train_loader, val_loader, test_loader, criterion, optimizer, 
                   epochs=15, device='cpu', patience=4):
    """Train with early stopping and evaluate."""
    best_val_loss = float('inf')
    patience_count = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            X = batch['sequence'].to(device)
            y = batch['targets'].to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch['sequence'].to(device)
                y = batch['targets'].to(device)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break
    
    # Evaluate on test set
    model.eval()
    test_targets, test_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            X = batch['sequence'].to(device)
            y = batch['targets'].to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            test_targets.append(y.cpu().numpy())
            test_probs.append(probs.cpu().numpy())
    
    targets = np.vstack(test_targets)
    probs = np.vstack(test_probs)
    
    # Compute AUC per horizon
    aucs = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) > 1:
            auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(auc)
    
    avg_auc = np.mean(aucs) if aucs else 0.5
    return avg_auc


def main(args):
    # Data
    df = generate_dataset(n_samples=args.n_samples)
    
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'success']]
    print(f"Features: {len(feature_cols)}")
    
    dataset = TimeSeriesDataset(df, args.sequence_length, args.horizons, feature_cols)
    print(f"Dataset size: {len(dataset)} sequences\n")
    
    # Train/val/test split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size],
                                             generator=torch.Generator().manual_seed(42))
    
    device = torch.device('cpu')
    
    # Hyperparameter grid
    configs = []
    for h in args.hidden_sizes:
        for l in args.num_layers_list:
            for b in args.batch_sizes:
                for lr in args.learning_rates:
                    configs.append({'hidden': h, 'layers': l, 'batch': b, 'lr': lr})
    
    print(f"Testing {len(configs)} configurations:\n")
    results = []
    best_auc = 0
    best_cfg = None
    
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] hidden={cfg['hidden']} layers={cfg['layers']} "
              f"batch={cfg['batch']} lr={cfg['lr']:.4f}", end=' -> ')
        
        train_loader = DataLoader(train_ds, batch_size=cfg['batch'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg['batch'])
        test_loader = DataLoader(test_ds, batch_size=cfg['batch'])
        
        model = LSTM(len(feature_cols), cfg['hidden'], cfg['layers'], len(args.horizons)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        criterion = nn.BCEWithLogitsLoss()
        
        auc = train_and_eval(model, train_loader, val_loader, test_loader, 
                            criterion, optimizer, epochs=args.epochs, device=device)
        
        result = {**cfg, 'auc': float(auc)}
        results.append(result)
        
        if auc > best_auc:
            best_auc = auc
            best_cfg = result
            torch.save(model.state_dict(), 'models/hypertuned_best.pth')
            print(f"AUC={auc:.4f} ✓ NEW BEST")
        else:
            print(f"AUC={auc:.4f}")
    
    # Save results
    os.makedirs('models', exist_ok=True)
    with open('models/hypertuning_results.json', 'w') as f:
        json.dump({'best': best_cfg, 'all': results}, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION: {best_cfg}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 15])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[32, 64])
    parser.add_argument('--num_layers_list', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[32, 64])
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[0.001, 0.005])
    args = parser.parse_args()
    main(args)
