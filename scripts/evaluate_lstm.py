#!/usr/bin/env python3
"""Evaluate LSTM against baseline models (LogReg, RF, XGBoost)"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
import warnings
import argparse
warnings.filterwarnings('ignore')

# command-line arguments
parser = argparse.ArgumentParser(description='Evaluate LSTM vs baselines with optional cross-validation')
parser.add_argument('--cv', type=int, default=1,
                    help='number of folds for cross-validation (default 1 -> no CV)')
parser.add_argument('--start_date', type=str,
                    help='filter features from this YYYY-MM-DD (inclusive)')
parser.add_argument('--end_date', type=str,
                    help='filter features up to this YYYY-MM-DD (inclusive)')
parser.add_argument('--data_path', type=str, default='data/banking_api_features_v7.csv',
                    help='path to features CSV')
args = parser.parse_args()

print("=== LSTM Evaluation vs Baselines ===\n")

# Load banking API features
features_path = args.data_path
if not os.path.exists(features_path):
    print(f"ERROR: {features_path} not found")
    exit(1)

df = pd.read_csv(features_path)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
print(f"Loaded {features_path}: {df.shape}")
# optional slicing
if args.start_date:
    df = df[df['timestamp'] >= pd.to_datetime(args.start_date)]
if args.end_date:
    df = df[df['timestamp'] <= pd.to_datetime(args.end_date)]
print(f"After date filtering: {df.shape}")

# All 43 feature columns — same as run_lstm_training.py.
# Baselines now use the full engineered feature set for a fair comparison.
# Previously 11 features; expanding gives XGBoost/RF access to precursor signals
# and cross-API features, making the comparison honest.
FEATURE_COLS = [
    "response_time", "request_count",
    "hour", "day_of_week", "is_market_hours", "is_financial_peak",
    "is_weekend", "is_holiday",
    "response_time_rolling_mean", "response_time_rolling_std",
    "error_rate_rolling", "response_time_variance", "error_volatility",
    "response_time_lag_1", "response_time_lag_5", "error_rate_lag_1",
    "response_time_ema_10", "response_time_ema_30", "error_rate_ema_10",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "high_frequency_api", "api_complexity",
    "error_rate_boost", "rt_multiplier",
    "latency_diff_1", "latency_diff_5",
    "error_rate_diff_1", "error_rate_diff_5",
    "latency_spike", "error_burst", "instability_index",
    "latency_slope", "error_slope",
    "traffic_change", "burst_ratio",
    "avg_error_rate_others", "max_error_rate_others",
    "n_apis_elevated", "corr_with_similar_api",
    "systemic_stress_index",
]
feature_cols = FEATURE_COLS

# Prepare data (filter to existing columns)
available_cols = [c for c in feature_cols if c in df.columns]

# ensure success is binary numeric
if df['success'].dtype == object:
    df['success'] = df['success'].map({'True':1, 'False':0}).fillna(0).astype(int)
else:
    df['success'] = df['success'].astype(int)

X = df[available_cols].fillna(0).astype(np.float32)
# y = 1 for failure (inverse of success)
y = (1 - df['success']).astype(int)

print(f"Features: {len(available_cols)}, Failure rate: {y.mean():.3f}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

results = {
    'baselines': {},
    'lstm': {},
    'comparison': None
}

# Baseline 1: Logistic Regression
print("Training Logistic Regression...")
if args.cv > 1:
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(LogisticRegression(random_state=42, max_iter=1000),
                                X, y, cv=args.cv, scoring='roc_auc')
    lr_auc = cv_scores.mean()
    print(f"  CV AUC ({args.cv}-fold): {lr_auc:.4f} (+/- {cv_scores.std()*2:.4f})")
else:
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    print(f"  AUC: {lr_auc:.4f}")
results['baselines']['LogisticRegression'] = {'auc': float(lr_auc)}

# Baseline 2: Random Forest
print("Training Random Forest...")
if args.cv > 1:
    cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42),
                                X, y, cv=args.cv, scoring='roc_auc')
    rf_auc = cv_scores.mean()
    print(f"  CV AUC ({args.cv}-fold): {rf_auc:.4f} (+/- {cv_scores.std()*2:.4f})")
else:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"  AUC: {rf_auc:.4f}")
results['baselines']['RandomForest'] = {'auc': float(rf_auc)}

# Baseline 3: XGBoost
print("Training XGBoost...")
if args.cv > 1:
    cv_scores = cross_val_score(xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
                                X, y, cv=args.cv, scoring='roc_auc')
    xgb_auc = cv_scores.mean()
    print(f"  CV AUC ({args.cv}-fold): {xgb_auc:.4f} (+/- {cv_scores.std()*2:.4f})")
else:
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    print(f"  AUC: {xgb_auc:.4f}\n")
results['baselines']['XGBoost'] = {'auc': float(xgb_auc)}

# LSTM Evaluation (with optional cross-validation)
print("Loading LSTM model...")
model_path = 'models/stress_test_best_model.pth'
if os.path.exists(model_path):
    # ── Correct architecture — must match run_lstm_training.py exactly ──────
    # Previous version was: unidirectional, hidden=64, no attention, no LayerNorm,
    # single FC layer. That caused a RuntimeError on load and fell back to
    # lstm_results.json, making the LSTM look weaker than it really is.

    class AttentionPooling(nn.Module):
        """Scalar attention over timesteps — identical to run_lstm_training.py."""
        def __init__(self, hidden_size):
            super().__init__()
            self.score = nn.Linear(hidden_size, 1, bias=False)

        def forward(self, x):                                    # x: (B, T, H)
            weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)  # (B, T)
            return (weights.unsqueeze(-1) * x).sum(dim=1)               # (B, H)

    class MultiHorizonLSTM(nn.Module):
        """BiLSTM + LayerNorm + AttentionPooling + per-horizon heads (v5)."""
        def __init__(self, input_size, hidden_size, num_layers,
                     n_horizons, dropout=0.3, bidirectional=True):
            super().__init__()
            self.lstm_out = hidden_size * (2 if bidirectional else 1)
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.layer_norm = nn.LayerNorm(self.lstm_out)
            self.attn_pool  = AttentionPooling(self.lstm_out)
            self.dropout    = nn.Dropout(dropout)
            self.heads = nn.ModuleList(
                [nn.Linear(self.lstm_out, 1) for _ in range(n_horizons)]
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            out    = self.layer_norm(out)
            out    = self.attn_pool(out)
            out    = self.dropout(out)
            return torch.cat([h(out) for h in self.heads], dim=1)

    def _detect_dims(state_dict):
        """Auto-detect hidden_size, n_features, n_horizons, bidirectional
        from checkpoint weights so this script never needs updating when
        the training config changes."""
        lstm_hh       = state_dict["lstm.weight_hh_l0"]
        lstm_ih       = state_dict["lstm.weight_ih_l0"]
        bidirectional = "lstm.weight_ih_l0_reverse" in state_dict
        hidden        = lstm_hh.shape[1]
        n_in          = lstm_ih.shape[1]
        n_horizons    = sum(1 for k in state_dict
                           if k.startswith("heads.") and k.endswith(".weight"))
        return n_in, hidden, n_horizons, bidirectional

    class TimeSeriesDataset(Dataset):
        def __init__(self, df, sequence_length, horizons, feature_cols=None):
            self.sequence_length = sequence_length
            self.horizons = horizons
            self.feature_cols = feature_cols or [c for c in df.columns if c != 'success']
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

    # Use all 43 features — same as training.
    # Previously used only 10 hardcoded features which meant the model was
    # evaluated on a different input space than it was trained on.
    lstm_available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"  LSTM features available: {len(lstm_available)}/43")

    df_lstm = df[lstm_available + ['success']].copy()
    df_lstm[lstm_available] = df_lstm[lstm_available].fillna(0).astype(np.float32)

    # Apply the same scaler used during training so inputs are normalised
    scaler_path = 'models/scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        df_lstm[lstm_available] = scaler.transform(
            df_lstm[lstm_available]
        ).astype(np.float32)
        print("  Scaler applied from models/scaler.pkl")
    else:
        print("  Warning: models/scaler.pkl not found — using unscaled features")
    
    sequence_length = 30
    prediction_horizons = [1, 5, 15]
    dataset = TimeSeriesDataset(df_lstm, sequence_length, prediction_horizons, feature_cols=lstm_available)
    
    if len(dataset) > 0:
        if args.cv > 1:
            # perform simple k-fold cross-validation for LSTM
            from sklearn.model_selection import KFold
            print(f"Performing {args.cv}-fold cross-validation for LSTM... (this may be slow)")
            kf = KFold(n_splits=args.cv, shuffle=True, random_state=42)
            fold_aucs = []
            for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
                subtrain = torch.utils.data.Subset(dataset, train_idx)
                subtest = torch.utils.data.Subset(dataset, test_idx)
                loader = DataLoader(subtest, batch_size=32, shuffle=False)
                # train a fresh model on subtrain for few epochs
                sd_cv = torch.load(model_path, map_location='cpu')
                n_in_cv, hid_cv, nh_cv, bd_cv = _detect_dims(sd_cv)
                model_fold = MultiHorizonLSTM(n_in_cv, hid_cv, 2, nh_cv, bidirectional=bd_cv).to('cpu')
                optim = torch.optim.Adam(model_fold.parameters(), lr=0.001)
                crit = nn.BCEWithLogitsLoss()
                for ep in range(3):
                    model_fold.train()
                    for batch in DataLoader(subtrain, batch_size=32, shuffle=True):
                        X = batch['sequence']
                        y = batch['targets']
                        optim.zero_grad()
                        out = model_fold(X)
                        loss = crit(out, y)
                        loss.backward()
                        optim.step()
                # evaluate
                all_t, all_p = [], []
                model_fold.eval()
                with torch.no_grad():
                    for batch in loader:
                        out = model_fold(batch['sequence'])
                        prob= torch.sigmoid(out)
                        all_t.append(batch['targets'].numpy())
                        all_p.append(prob.numpy())
                all_t = np.vstack(all_t)
                all_p = np.vstack(all_p)
                # compute average auc
                aucs = []
                for i in range(all_t.shape[1]):
                    if len(np.unique(all_t[:,i]))<2:
                        aucs.append(np.nan)
                    else:
                        aucs.append(roc_auc_score(all_t[:,i], all_p[:,i]))
                fold_aucs.append(np.nanmean(aucs))
                print(f" Fold {fold+1} avg AUC: {fold_aucs[-1]:.4f}")
            lstm_auc_avg = np.nanmean(fold_aucs)
            results['lstm']['avg_auc'] = float(lstm_auc_avg)
            results['lstm']['per_horizon'] = {}  # omitted for cv
        else:
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                                   generator=torch.Generator().manual_seed(42))
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            device = torch.device('cpu')
            try:
                sd = torch.load(model_path, map_location=device)
                n_in, hidden, n_heads, bidir = _detect_dims(sd)
                print(f"  Checkpoint: input={n_in}  hidden={hidden}  "
                      f"horizons={n_heads}  bidir={bidir}")
                lstm_model = MultiHorizonLSTM(
                    n_in, hidden, num_layers=2,
                    n_horizons=n_heads, bidirectional=bidir
                ).to(device)
                lstm_model.load_state_dict(sd)
                lstm_model.eval()

                all_targets = []
                all_probas = []
                with torch.no_grad():
                    for batch in test_loader:
                        X = batch['sequence'].to(device)
                        y = batch['targets'].to(device)
                        outputs = lstm_model(X)
                        probas = torch.sigmoid(outputs)
                        all_targets.append(y.cpu().numpy())
                        all_probas.append(probas.cpu().numpy())

                all_targets = np.vstack(all_targets)
                all_probas  = np.vstack(all_probas)

                print(f"LSTM test set: {len(all_targets)} samples")
                per_horizon = {}
                lstm_auc_avg = 0
                for i, h in enumerate(prediction_horizons):
                    y_true  = all_targets[:, i]
                    y_score = all_probas[:, i]
                    if len(np.unique(y_true)) < 2:
                        auc = float('nan')
                    else:
                        auc = roc_auc_score(y_true, y_score)
                        lstm_auc_avg += auc
                    per_horizon[f'horizon_{h}'] = float(auc)
                    auc_str = 'NaN' if np.isnan(auc) else f"{auc:.4f}"
                    print(f"  Horizon {h}: AUC={auc_str}")

                lstm_auc_avg = lstm_auc_avg / len(prediction_horizons)
                results['lstm']['per_horizon'] = per_horizon
                results['lstm']['avg_auc'] = float(lstm_auc_avg) if not np.isnan(lstm_auc_avg) else None

            except (RuntimeError, Exception) as model_err:
                # v5 architecture differs from this script's stub — load from saved results
                print(f"  Note: model architecture mismatch ({model_err.__class__.__name__}) "
                      f"— reading AUC from models/lstm_results.json")
                _res_path = 'models/lstm_results.json'
                if os.path.exists(_res_path):
                    with open(_res_path) as _f:
                        _saved = json.load(_f)
                    results['lstm']['avg_auc']     = _saved.get('avg_auc')
                    results['lstm']['per_horizon'] = {
                        k: v.get('auc') for k, v in _saved.get('per_horizon', {}).items()
                    }
                    print(f"  Loaded LSTM avg_auc={results['lstm']['avg_auc']:.4f} "
                          f"from saved results")
                else:
                    print("  lstm_results.json not found — LSTM AUC unavailable")
    else:
        print("  ERROR: LSTM dataset too small")
else:
    print(f"  ERROR: {model_path} not found")

# Summary
print("\n=== Summary ===")
baseline_aucs = [v['auc'] for v in results['baselines'].values()]
best_baseline = max(baseline_aucs)
print(f"Best Baseline AUC: {best_baseline:.4f}")

if results['lstm'].get('avg_auc'):
    lstm_auc = results['lstm']['avg_auc']
    improvement = ((lstm_auc - best_baseline) / best_baseline * 100) if best_baseline > 0 else 0
    print(f"LSTM Avg AUC: {lstm_auc:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    results['comparison'] = {
        'best_baseline_auc': float(best_baseline),
        'lstm_avg_auc': float(lstm_auc),
        'improvement_percent': float(improvement)
    }

# Save results
os.makedirs('models', exist_ok=True)
with open('models/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to models/evaluation_results.json")
