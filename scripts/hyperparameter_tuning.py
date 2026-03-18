#!/usr/bin/env python3
"""LSTM hyperparameter tuning with enhanced feature engineering and large dataset generation."""

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('=== Hyperparameter Tuning with Enhanced Features ===\n')


def generate_large_dataset_enhanced(n_samples=500000):
    """Generate large synthetic banking API dataset with realistic patterns."""
    print(f"Generating {n_samples:,} synthetic samples...")
    np.random.seed(42)
    
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1min')
    data = {}
    data['timestamp'] = timestamps
    
    # Base response time with time-of-day and day-of-week effects
    hour = timestamps.hour.values
    dow = timestamps.dayofweek.values
    
    # Response time: baseline + time-of-day effect + day-of-week effect + noise
    base_response = np.random.exponential(100, n_samples)
    hour_effect = 50 * np.sin(2 * np.pi * hour / 24)  # slower during business hours
    dow_effect = 20 * (1 if (dow >= 5).any() else 0)  # slightly slower weekends
    data['response_time'] = np.maximum(10, base_response + hour_effect + dow_effect)
    
    # Status codes with correlated success/failure
    error_prob = 0.15 + 0.05 * (hour_effect > 30) / 100  # higher errors during peak hours
    status_codes = np.where(
        np.random.random(n_samples) < error_prob,
        np.random.choice([400, 404, 500, 503], n_samples),
        np.random.choice([200, 201], n_samples)
    )
    data['status_code'] = status_codes
    data['success'] = (status_codes < 400).astype(int)
    data['request_count'] = np.random.poisson(50 + 20 * (hour >= 9) & (hour <= 16), n_samples)
    
    # Time features
    data['hour'] = hour
    data['day_of_week'] = dow
    data['is_weekend'] = (dow >= 5).astype(int)
    data['is_market_hours'] = ((hour >= 9) & (hour <= 16)).astype(int)
    
    # Rolling statistics (simulated)
    data['response_time_rolling_mean'] = pd.Series(data['response_time']).rolling(60, min_periods=1).mean().values
    data['response_time_rolling_std'] = pd.Series(data['response_time']).rolling(60, min_periods=1).std().fillna(0).values
    data['error_rate_rolling'] = pd.Series(1 - data['success']).rolling(60, min_periods=1).mean().values
    data['response_time_variance'] = data['response_time_rolling_std'] ** 2
    data['error_volatility'] = np.random.exponential(0.1, n_samples)
    
    df = pd.DataFrame(data)
    
    # Add autocorrelation: failures tend to cluster
    failure_indices = np.where(df['success'] == 0)[0]
    for idx in failure_indices:
        for offset in range(1, 10):
            if idx + offset < len(df) and np.random.random() < 0.3:
                df.loc[idx + offset, 'success'] = 0
    
    print(f"Dataset generated: {len(df)} samples, failure rate: {1 - df['success'].mean():.2%}\n")
    return df


def engineer_features(df, include_lags=True, include_ema=True, include_cyclical=True):
    """Enhanced feature engineering with lag features, EMA, and cyclical encoding."""
    df = df.copy()
    
    # Lag features (1, 5, 15 step lags for response time and success)
    if include_lags:
        for lag in [1, 5, 15]:
            df[f'response_time_lag_{lag}'] = df['response_time'].shift(lag).fillna(0)
            df[f'success_lag_{lag}'] = df['success'].shift(lag).fillna(0)
            df[f'error_rate_lag_{lag}'] = (1 - df['success']).shift(lag).fillna(0)
    
    # Exponential moving averages (EMA)
    if include_ema:
        for span in [10, 30, 60]:
            df[f'response_time_ema_{span}'] = df['response_time'].ewm(span=span).mean()
            df[f'error_rate_ema_{span}'] = (1 - df['success']).ewm(span=span).mean()
    
    # Cyclical encoding for hour and day_of_week (to capture periodicity)
    if include_cyclical:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Interaction features
    df['response_time_x_request_count'] = df['response_time'] * df['request_count'] / 100
    df['error_rate_x_is_peak'] = df['error_rate_rolling'] * df['is_market_hours']
    
    return df.fillna(0)


class TimeSeriesDataset(Dataset):
    def __init__(self, df, sequence_length, horizons, feature_cols=None):
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.feature_cols = feature_cols or [c for c in df.columns if c not in ['timestamp', 'success']]
        
        self._data = df[self.feature_cols].to_numpy(dtype=np.float32)
        self._targets = df['success'].to_numpy(dtype=np.float32)
        self.max_start = len(self._data) - self.sequence_length - max(self.horizons) + 1
        
        if self.max_start < 1:
            raise ValueError(f'Dataset too small: {len(self._data)} < {self.sequence_length + max(self.horizons)}')

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


class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(MultiHorizonLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, 
                              epochs=30, device='cpu', patience=5):
    """Train with early stopping."""
    best_val_loss = float('inf')
    patience_counter = 0
    training_stats = {'epoch_times': [], 'train_losses': [], 'val_losses': []}
    
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            X = batch['sequence'].to(device)
            y = batch['targets'].to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / max(1, len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X = batch['sequence'].to(device)
                y = batch['targets'].to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        val_loss = val_loss / max(1, len(val_loader))
        
        epoch_time = time.time() - t0
        training_stats['epoch_times'].append(epoch_time)
        training_stats['train_losses'].append(train_loss)
        training_stats['val_losses'].append(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    return best_val_loss, training_stats


def evaluate_model(model, test_loader, prediction_horizons, device='cpu'):
    """Evaluate model on test set."""
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
    
    per_horizon = {}
    horizon_aucs = []
    for i, h in enumerate(prediction_horizons):
        y_true = all_targets[:, i]
        y_score = all_probas[:, i]
        if len(np.unique(y_true)) < 2:
            auc = float('nan')
        else:
            auc = roc_auc_score(y_true, y_score)
            horizon_aucs.append(auc)
        per_horizon[f'horizon_{h}'] = float(auc)
    
    avg_auc = np.nanmean(horizon_aucs) if horizon_aucs else 0
    return per_horizon, avg_auc


def run_hyperparameter_tuning(args):
    """Run grid search over hyperparameters."""
    
    # Load or generate data
    features_path = 'data/banking_api_features.csv'
    if os.path.exists(features_path) and args.use_existing:
        df = pd.read_csv(features_path)
        print(f"Loaded existing data: {features_path} ({len(df)} samples)")
    else:
        df = generate_large_dataset_enhanced(n_samples=args.n_samples)
        # Save for future use
        os.makedirs('data', exist_ok=True)
        df.to_csv(features_path, index=False)
        print(f"Saved to {features_path}")
    
    # Feature engineering
    print("Applying advanced feature engineering...")
    df = engineer_features(df, include_lags=True, include_ema=True, include_cyclical=True)
    
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'success']]
    print(f"Total features after engineering: {len(feature_cols)}")
    
    # Prepare dataset
    df_data = df[feature_cols + ['success']].copy()
    df_data[feature_cols] = df_data[feature_cols].fillna(0).astype(np.float32)
    
    sequence_length = args.sequence_length
    prediction_horizons = args.horizons
    
    dataset = TimeSeriesDataset(df_data, sequence_length, prediction_horizons, feature_cols=feature_cols)
    print(f"Dataset ready: {len(dataset)} sequences\n")
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    device = torch.device('cpu')
    
    # Hyperparameter grid
    param_grid = {
        'hidden_size': args.hidden_sizes,
        'num_layers': args.num_layers_list,
        'batch_size': args.batch_sizes,
        'lr': args.learning_rates,
    }
    
    results = []
    total_trials = (len(param_grid['hidden_size']) * len(param_grid['num_layers']) * 
                   len(param_grid['batch_size']) * len(param_grid['lr']))
    
    print(f"Running {total_trials} hyperparameter combinations...\n")
    
    trial = 0
    best_auc = 0
    best_config = None
    
    for hidden_size in param_grid['hidden_size']:
        for num_layers in param_grid['num_layers']:
            for batch_size in param_grid['batch_size']:
                for lr in param_grid['lr']:
                    trial += 1
                    print(f"[{trial}/{total_trials}] hidden={hidden_size}, layers={num_layers}, batch={batch_size}, lr={lr:.4f}")
                    
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    
                    model = MultiHorizonLSTM(len(feature_cols), hidden_size, num_layers, 
                                            len(prediction_horizons), dropout=0.2).to(device)
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    
                    t0 = time.time()
                    best_val, stats = train_with_early_stopping(
                        model, train_loader, val_loader, criterion, optimizer,
                        epochs=args.epochs, device=device, patience=args.patience
                    )
                    training_time = time.time() - t0
                    
                    # Evaluate
                    per_horizon, avg_auc = evaluate_model(model, test_loader, prediction_horizons, device)
                    
                    result = {
                        'trial': trial,
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'batch_size': batch_size,
                        'lr': lr,
                        'best_val_loss': float(best_val),
                        'avg_auc': float(avg_auc),
                        'per_horizon': per_horizon,
                        'training_time_sec': training_time,
                    }
                    results.append(result)
                    
                    if avg_auc > best_auc:
                        best_auc = avg_auc
                        best_config = result
                        # Save best model
                        os.makedirs('models', exist_ok=True)
                        torch.save(model.state_dict(), 'models/hypertuned_lstm_best.pth')
                        print(f"  ✓ New best AUC: {avg_auc:.4f}")
                    else:
                        print(f"    AUC: {avg_auc:.4f}")
    
    # Save all results
    os.makedirs('models', exist_ok=True)
    with open('models/hyperparameter_tuning_results.json', 'w') as f:
        json.dump({
            'total_trials': total_trials,
            'best_config': best_config,
            'all_results': results,
            'feature_count': len(feature_cols),
            'dataset_size': len(dataset),
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Best Configuration:")
    for key in ['hidden_size', 'num_layers', 'batch_size', 'lr']:
        print(f"  {key}: {best_config[key]}")
    print(f"  Average AUC: {best_config['avg_auc']:.4f}")
    print(f"  Per-horizon AUCs: {best_config['per_horizon']}")
    print(f"{'='*80}\n")
    print(f"Results saved to models/hyperparameter_tuning_results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Hyperparameter Tuning')
    parser.add_argument('--n_samples', type=int, default=500000, help='Number of synthetic samples')
    parser.add_argument('--use_existing', action='store_true', help='Use existing data instead of generating')
    parser.add_argument('--sequence_length', type=int, default=30)
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 15])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument('--num_layers_list', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[32, 64])
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[0.0005, 0.001, 0.005])
    
    args = parser.parse_args()
    run_hyperparameter_tuning(args)
