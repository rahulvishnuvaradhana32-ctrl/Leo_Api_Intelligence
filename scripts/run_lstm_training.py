#!/usr/bin/env python3
"""Run LSTM training (creates models/ outputs).

Usage:
    python scripts/run_lstm_training.py --epochs 30
    python scripts/run_lstm_training.py --epochs 30 --end_date 2024-12-31
    python scripts/run_lstm_training.py --epochs 30 --balance   # 50/50 balanced sampling
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print('=== run_lstm_training.py start ===')
import argparse
import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ── Feature columns ──────────────────────────────────────────────────────────
# 28 original CSV features
CSV_BASE_FEATURES = [
    'response_time', 'status_code', 'request_count',
    'hour', 'day_of_week', 'is_weekend', 'is_market_hours',
    'response_time_rolling_mean', 'response_time_rolling_std',
    'error_rate_rolling',
    'response_time_variance', 'error_volatility',
    'response_time_lag_1', 'response_time_lag_5',
    'error_rate_lag_1',
    'response_time_ema_10', 'response_time_ema_30',
    'error_rate_ema_10',
    'request_count_rolling_mean', 'request_count_rolling_std',
    'response_time_pct_change', 'error_rate_pct_change',
    'response_time_zscore', 'error_rate_zscore',
    'time_since_last_failure', 'failure_streak',
    'api_failure_rate_global', 'hour_sin',
]

# 2 precursor features that live in the CSV (computed by add_precursor_features.py)
CSV_PRECURSOR_FEATURES = [
    'traffic_change', 'burst_ratio',
]

# 9 on-the-fly precursor features (computed below from CSV base columns)
ONTHEFLY_FEATURES = [
    'latency_diff_1', 'latency_diff_5',
    'error_rate_diff_1', 'error_rate_diff_5',
    'latency_spike', 'error_burst', 'instability_index',
    'latency_slope', 'error_slope',
]

ALL_FEATURE_COLS = CSV_BASE_FEATURES + CSV_PRECURSOR_FEATURES + ONTHEFLY_FEATURES
EPS = 1e-6


# ── Precursor computation ─────────────────────────────────────────────────────
def add_precursor_features_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Add on-the-fly precursor features if not already present."""
    if 'latency_diff_1' not in df.columns:
        df['latency_diff_1'] = (df['response_time'] - df['response_time_lag_1']).fillna(0)
    if 'latency_diff_5' not in df.columns:
        df['latency_diff_5'] = (df['response_time'] - df['response_time_lag_5']).fillna(0)
    if 'error_rate_diff_1' not in df.columns:
        df['error_rate_diff_1'] = (df['error_rate_rolling'] - df['error_rate_lag_1']).fillna(0)
    if 'error_rate_diff_5' not in df.columns:
        df['error_rate_diff_5'] = (df['error_rate_rolling'] - df['error_rate_ema_10']).fillna(0)
    if 'latency_spike' not in df.columns:
        df['latency_spike'] = df['response_time'] / (df['response_time_rolling_mean'] + EPS)
    if 'error_burst' not in df.columns:
        df['error_burst'] = df['error_rate_rolling'] / (df['error_rate_ema_10'] + EPS)
    if 'instability_index' not in df.columns:
        df['instability_index'] = (
            df['latency_diff_1'].abs() + df['error_rate_diff_1'].abs()
        )
    if 'latency_slope' not in df.columns:
        df['latency_slope'] = (
            (df['response_time_ema_10'] - df['response_time_ema_30']) / 20.0
        ).fillna(0)
    if 'error_slope' not in df.columns:
        df['error_slope'] = (
            (df['error_rate_ema_10'] - df['error_rate_lag_1']) / 10.0
        ).fillna(0)
    return df


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        prob = torch.sigmoid(logits)
        p_t = torch.where(targets >= 0.5, prob, 1 - prob)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int, horizons: list):
        self.seq_len = sequence_length
        self.horizons = horizons
        max_h = max(horizons)
        self.max_start = len(X) - sequence_length - max_h + 1
        if self.max_start < 1:
            raise ValueError('Dataset too small for sequence_length and horizons')
        self.X = X
        self.y = y

    def __len__(self):
        return max(0, self.max_start)

    def __getitem__(self, idx):
        seq = self.X[idx: idx + self.seq_len]
        targets = []
        for h in self.horizons:
            target_idx = idx + self.seq_len + h - 1
            targets.append(float(1 - self.y[target_idx]))   # 1 = failure
        return (
            torch.from_numpy(seq),
            torch.tensor(targets, dtype=torch.float32)
        )


# ── Model v4 ─────────────────────────────────────────────────────────────────
class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_horizons, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_horizons)])

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.layer_norm(out)
        out = self.dropout(out)
        return torch.cat([head(out) for head in self.heads], dim=1)


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    features_path = os.path.join('data', 'banking_api_features.csv')
    if not os.path.exists(features_path):
        print(f"ERROR: {features_path} not found. Run integrate_kaggle_data.py first.")
        sys.exit(1)

    print(f"Loading {features_path} ...")
    df = pd.read_csv(features_path, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    if args.start_date:
        df = df[df['timestamp'] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        df = df[df['timestamp'] <= pd.to_datetime(args.end_date)]
    print(f"  After date filtering: {len(df):,} rows")

    # Add on-the-fly precursor features
    print("Computing precursor features ...")
    df = add_precursor_features_inplace(df)

    # Select available features
    available = [c for c in ALL_FEATURE_COLS if c in df.columns]
    missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [!] Missing {len(missing)} feature(s) — will train on {len(available)}: {missing}")
    feature_cols = available
    print(f"  Features: {len(feature_cols)}")

    df_feat = df[feature_cols + ['success']].copy()
    df_feat[feature_cols] = df_feat[feature_cols].fillna(0).astype(np.float32)
    df_feat['success'] = df_feat['success'].fillna(0).astype(np.float32)

    raw_X = df_feat[feature_cols].to_numpy(dtype=np.float32)
    raw_y = df_feat['success'].to_numpy(dtype=np.float32)

    # Chronological split indices (80/10/10)
    n = len(raw_X)
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)

    # Stratified split: use sequence-level failure label at h=max horizon
    max_h = max(args.horizons)
    seq_len = args.sequence_length
    n_sequences = n - seq_len - max_h + 1
    if n_sequences < 100:
        print(f"ERROR: Only {n_sequences} sequences available — too few to train.")
        sys.exit(1)

    seq_starts = np.arange(n_sequences)
    strat_labels = np.array([int(raw_y[i + seq_len + max_h - 1] == 0) for i in seq_starts])

    # Chronological boundary indices for sequences
    train_seq_end = sum(1 for i in seq_starts if i + seq_len + max_h - 1 < train_end)
    val_seq_end   = sum(1 for i in seq_starts if i + seq_len + max_h - 1 < val_end)

    tr = seq_starts[:train_seq_end]
    va = seq_starts[train_seq_end:val_seq_end]
    te = seq_starts[val_seq_end:]

    # Fit scaler on training rows only
    print("Fitting StandardScaler on training rows ...")
    train_row_end = tr[-1] + seq_len + max_h if len(tr) else train_end
    scaler = StandardScaler()
    scaler.fit(raw_X[:train_row_end])
    scaled_X = scaler.transform(raw_X).astype(np.float32)

    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("  Scaler saved -> models/scaler.pkl")

    # Optional balanced sampling (training split only)
    if args.balance:
        _fail_tr   = tr[strat_labels[tr] == 1]
        _normal_tr = tr[strat_labels[tr] == 0]
        if len(_fail_tr) > 0 and len(_normal_tr) > len(_fail_tr):
            _rng_bal   = np.random.default_rng(42)
            _normal_tr = _rng_bal.choice(_normal_tr, size=len(_fail_tr), replace=False)
            tr         = np.concatenate([_fail_tr, _normal_tr])
            _rng_bal.shuffle(tr)
            print(f"Balanced train: {len(_fail_tr):,} failures + {len(_fail_tr):,} normal "
                  f"= {len(tr):,} sequences (50/50)")
        else:
            print(f"Balance flag set but insufficient failures ({len(_fail_tr)}) — skipping balance.")
    else:
        fail_count = int(strat_labels[tr].sum())
        print(f"Natural distribution: {len(tr):,} train sequences "
              f"({fail_count:,} failures = {100*fail_count/max(1,len(tr)):.1f}%)")

    # Cap training sequences
    if args.max_train_sequences and len(tr) > args.max_train_sequences:
        rng = np.random.default_rng(42)
        tr = rng.choice(tr, size=args.max_train_sequences, replace=False)
        print(f"Capped train to {len(tr):,} sequences")

    # Compute pos_weight for FocalLoss
    n_fail = int(strat_labels[tr].sum())
    n_norm = len(tr) - n_fail
    if args.balance:
        pos_weight_val = 1.0
    else:
        pos_weight_val = n_norm / max(1, n_fail)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    print(f"  pos_weight = {pos_weight_val:.2f}")

    # Build datasets
    train_dataset = TimeSeriesDataset(scaled_X, raw_y, seq_len, args.horizons)
    train_dataset.max_start = len(tr)
    # Override __getitem__ index mapping via a wrapper
    class SubsetDataset(Dataset):
        def __init__(self, base_ds, indices):
            self.base = base_ds
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.base[self.indices[i]]

    train_ds = SubsetDataset(TimeSeriesDataset(scaled_X, raw_y, seq_len, args.horizons), tr)
    val_ds   = SubsetDataset(TimeSeriesDataset(scaled_X, raw_y, seq_len, args.horizons), va)
    test_ds  = SubsetDataset(TimeSeriesDataset(scaled_X, raw_y, seq_len, args.horizons), te)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nDataset sizes — train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}")

    device = torch.device('cpu')
    model = MultiHorizonLSTM(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        n_horizons=len(args.horizons),
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    training_stats = {'epoch_times': [], 'train_losses': [], 'val_losses': []}

    print(f"\nTraining for up to {args.epochs} epochs (early stopping patience={args.patience}) ...")
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for seq, targets in train_loader:
            seq, targets = seq.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / max(1, len(train_loader))
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, targets in val_loader:
                seq, targets = seq.to(device), targets.to(device)
                outputs = model(seq)
                val_loss += criterion(outputs, targets).item()
        val_loss /= max(1, len(val_loader))

        epoch_time = time.time() - t0
        training_stats['epoch_times'].append(epoch_time)
        training_stats['train_losses'].append(train_loss)
        training_stats['val_losses'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/stress_test_best_model.pth')
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:>3}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    total_time = sum(training_stats['epoch_times'])

    # ── Evaluation on test set ────────────────────────────────────────────────
    model.load_state_dict(torch.load('models/stress_test_best_model.pth', map_location=device))
    model.eval()
    all_targets, all_probas = [], []
    with torch.no_grad():
        for seq, targets in test_loader:
            seq = seq.to(device)
            probas = torch.sigmoid(model(seq))
            all_targets.append(targets.cpu().numpy())
            all_probas.append(probas.cpu().numpy())
    all_targets = np.vstack(all_targets)
    all_probas  = np.vstack(all_probas)

    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    per_horizon = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, h in enumerate(args.horizons):
        y_true  = all_targets[:, i]
        y_score = all_probas[:, i]
        if len(np.unique(y_true)) < 2:
            auc = float('nan')
            print(f"  horizon_{h}: AUC=nan (only one class in test set)")
        else:
            auc = float(roc_auc_score(y_true, y_score))
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, label=f'h={h} (AUC={auc:.3f})')
        per_horizon[f'horizon_{h}'] = {'auc': auc}
        print(f"  horizon_{h}: AUC = {auc:.4f}")

    avg_auc = float(np.nanmean([v['auc'] for v in per_horizon.values()]))
    print(f"\n  Average AUC: {avg_auc:.4f}")

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Multi-Horizon LSTM')
    ax.legend()
    fig.tight_layout()
    roc_path = 'models/lstm_roc_curves.png'
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)

    results = {
        'total_time': total_time,
        'avg_epoch_time': float(np.mean(training_stats['epoch_times'])),
        'best_val_loss': float(best_val_loss),
        'train_losses': training_stats['train_losses'],
        'val_losses': training_stats['val_losses'],
        'per_horizon': per_horizon,
        'avg_auc': avg_auc,
        'n_features': len(feature_cols),
        'n_train': len(train_ds),
        'n_val': len(val_ds),
        'n_test': len(test_ds),
        'balance': args.balance,
        'summary': {
            'training_time_sec': total_time,
            'best_val_loss': float(best_val_loss),
            'avg_auc': avg_auc,
        },
        'artifact_paths': {
            'roc_plot':   roc_path,
            'model_path': 'models/stress_test_best_model.pth',
            'scaler_path': 'models/scaler.pkl',
        }
    }
    with open('models/lstm_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nTraining & evaluation complete.')
    print(f'  Model  -> models/stress_test_best_model.pth')
    print(f'  Scaler -> models/scaler.pkl')
    print(f'  ROC    -> {roc_path}')
    print(f'  JSON   -> models/lstm_results.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',            type=int,   default=30)
    parser.add_argument('--batch_size',        type=int,   default=128)
    parser.add_argument('--sequence_length',   type=int,   default=30)
    parser.add_argument('--horizons',          nargs='+',  type=int,   default=[1, 5, 15])
    parser.add_argument('--hidden_size',       type=int,   default=128)
    parser.add_argument('--num_layers',        type=int,   default=2)
    parser.add_argument('--dropout',           type=float, default=0.3)
    parser.add_argument('--lr',                type=float, default=0.001)
    parser.add_argument('--focal_gamma',       type=float, default=2.0)
    parser.add_argument('--patience',          type=int,   default=6)
    parser.add_argument('--max_train_sequences', type=int, default=None,
                        help='Cap training sequences (None = no cap)')
    parser.add_argument('--balance',           action='store_true', default=False,
                        help='Enable 50/50 balanced sampling on training split (default: off)')
    parser.add_argument('--start_date',        type=str,   default=None,
                        help='Filter rows from this YYYY-MM-DD (inclusive)')
    parser.add_argument('--end_date',          type=str,   default=None,
                        help='Filter rows up to this YYYY-MM-DD (inclusive)')
    args = parser.parse_args()
    train(args)
