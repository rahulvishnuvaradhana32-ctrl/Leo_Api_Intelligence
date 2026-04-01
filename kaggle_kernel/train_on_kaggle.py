#!/usr/bin/env python3
"""
train_on_kaggle.py — LEO API Multi-Horizon LSTM Training (Kaggle GPU Edition)

This is an exact copy of run_lstm_training.py adapted for the Kaggle environment:
  - Data read from /kaggle/input/leo-api-v7-dataset/banking_api_features_v7.csv
  - Outputs saved to /kaggle/working/models/ (downloaded back by run_kaggle_training.py)
  - GPU auto-detected (T4 x2 supported via DataParallel)
  - Batch size 512 and hidden 256 for GPU efficiency

DO NOT edit the architecture here independently — keep it in sync with run_lstm_training.py.
"""
import json, os, sys, time
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ── Kaggle paths ───────────────────────────────────────────────────────────────
KAGGLE_WORKING = "/kaggle/working"
OUT_DIR        = os.path.join(KAGGLE_WORKING, "models")

def _find_input_csv() -> str:
    """Auto-detect the v7 CSV regardless of dataset slug name."""
    import glob
    # Search all attached datasets for any CSV containing 'banking_api_features'
    candidates = glob.glob("/kaggle/input/**/*.csv", recursive=True)
    print("\nAvailable input files:")
    for f in candidates:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {f}  ({size_mb:.0f} MB)")
    # Prefer v7, then any banking_api_features CSV, then largest CSV
    for c in candidates:
        if "banking_api_features_v7" in c:
            return c
    for c in candidates:
        if "banking_api_features" in c:
            return c
    if candidates:
        largest = max(candidates, key=os.path.getsize)
        print(f"  No banking_api_features CSV found — using largest: {largest}")
        return largest
    print("ERROR: No CSV files found in /kaggle/input/")
    print("Make sure you attached the dataset to this notebook.")
    sys.exit(1)

KAGGLE_INPUT = _find_input_csv()

# ── Hyperparameters (tune here before pushing) ─────────────────────────────────
EPOCHS          = 30
BATCH_SIZE      = 512       # larger batch = faster on GPU
SEQUENCE_LENGTH = 30
HORIZONS        = [1, 5, 15]
HIDDEN_SIZE     = 256       # doubled from CPU default (128) — GPU has room
NUM_LAYERS      = 2
DROPOUT         = 0.3
LR              = 0.001
PATIENCE        = 6
FOCAL_GAMMA     = 2.0
BIDIRECTIONAL   = True
MAX_TRAIN_SEQ   = None      # None = use all sequences
BALANCE         = False

# ── Feature columns (must match run_lstm_training.py exactly) ─────────────────
FEATURE_COLS: List[str] = [
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
assert len(FEATURE_COLS) == 43, f"Expected 43 features, got {len(FEATURE_COLS)}"

EPS = 1e-6


# ── On-the-fly precursor features (identical to run_lstm_training.py) ─────────
def add_precursor_features(df: pd.DataFrame) -> pd.DataFrame:
    def _col(name, fill=0.0):
        return df[name] if name in df.columns else pd.Series(fill, index=df.index)

    rt   = df["response_time"]
    rl1  = _col("response_time_lag_1",    0.0)
    rl5  = _col("response_time_lag_5",    0.0)
    er   = _col("error_rate_rolling",     0.0)
    el1  = _col("error_rate_lag_1",       0.0)
    rm   = _col("response_time_rolling_mean", 1.0)
    e10  = _col("error_rate_ema_10",      0.0)
    re10 = _col("response_time_ema_10",   1.0)
    re30 = _col("response_time_ema_30",   1.0)

    df["latency_diff_1"]    = (rt  - rl1).fillna(0)
    df["latency_diff_5"]    = (rt  - rl5).fillna(0)
    df["error_rate_diff_1"] = (er  - el1).fillna(0)
    df["error_rate_diff_5"] = (er  - e10).fillna(0)
    df["latency_spike"]     = rt  / (rm  + EPS)
    df["error_burst"]       = er  / (e10 + EPS)
    df["instability_index"] = df["latency_diff_1"].abs() + df["error_rate_diff_1"].abs()
    df["latency_slope"]     = ((re10 - re30) / 20.0).fillna(0)
    df["error_slope"]       = ((e10  - el1)  / 10.0).fillna(0)
    return df


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw  = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        prob = torch.sigmoid(logits)
        p_t  = torch.where(targets >= 0.5, prob, 1 - prob)
        return ((1 - p_t) ** self.gamma * bce).mean()


# ── Dataset ────────────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len, horizons):
        self.X        = X
        self.y        = y
        self.seq_len  = seq_len
        self.horizons = horizons
        self.max_h    = max(horizons)
        self.n        = max(0, len(X) - seq_len - self.max_h + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seq     = self.X[idx : idx + self.seq_len]
        targets = [float(self.y[idx + self.seq_len + h - 1] == 0) for h in self.horizons]
        return torch.from_numpy(seq).float(), torch.tensor(targets, dtype=torch.float32)


class SubsetDataset(Dataset):
    def __init__(self, base, indices):
        self.base    = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]


# ── Attention Pooling ──────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


# ── Model v5 (identical to run_lstm_training.py) ──────────────────────────────
class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_horizons,
                 dropout=0.3, bidirectional=True):
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
        self.heads      = nn.ModuleList(
            [nn.Linear(self.lstm_out, 1) for _ in range(n_horizons)]
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.layer_norm(out)
        out    = self.attn_pool(out)
        out    = self.dropout(out)
        return torch.cat([h(out) for h in self.heads], dim=1)


# ── Main training ──────────────────────────────────────────────────────────────
def train():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {KAGGLE_INPUT} ...")
    df = pd.read_csv(KAGGLE_INPUT, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    print("Computing precursor features ...")
    df = add_precursor_features(df)

    available    = [c for c in FEATURE_COLS if c in df.columns]
    missing      = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [!] {len(missing)} feature(s) absent — training on {len(available)}: {missing}")
    feature_cols = available
    print(f"  Features used: {len(feature_cols)}")

    df_work = df[feature_cols + ["success"]].copy()
    df_work[feature_cols] = df_work[feature_cols].fillna(0).astype(np.float32)
    df_work["success"]    = df_work["success"].fillna(0).astype(np.float32)

    raw_X = df_work[feature_cols].to_numpy(dtype=np.float32)
    raw_y = df_work["success"].to_numpy(dtype=np.float32)

    seq_len = SEQUENCE_LENGTH
    max_h   = max(HORIZONS)
    n_seqs  = len(raw_X) - seq_len - max_h + 1
    seq_idx = np.arange(n_seqs)
    fail_labels = np.array(
        [int(raw_y[i + seq_len + max_h - 1] == 0) for i in seq_idx]
    )
    labels_per_h = {
        h: np.array([int(raw_y[i + seq_len + h - 1] == 0) for i in seq_idx])
        for h in HORIZONS
    }

    tr_idx, tmp_idx, _, tmp_lbl = train_test_split(
        seq_idx, fail_labels, test_size=0.20, random_state=42, stratify=fail_labels
    )
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.50, random_state=42, stratify=tmp_lbl
    )

    train_row_end = int(0.80 * len(raw_X))
    print("Fitting StandardScaler on training rows ...")
    scaler   = StandardScaler()
    scaler.fit(raw_X[:train_row_end])
    scaled_X = scaler.transform(raw_X).astype(np.float32)

    scaler_path = os.path.join(OUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved → {scaler_path}")

    if BALANCE:
        fail_tr   = tr_idx[fail_labels[tr_idx] == 1]
        normal_tr = tr_idx[fail_labels[tr_idx] == 0]
        if len(fail_tr) > 0 and len(normal_tr) > len(fail_tr):
            rng       = np.random.default_rng(42)
            normal_tr = rng.choice(normal_tr, size=len(fail_tr), replace=False)
            tr_idx    = np.concatenate([fail_tr, normal_tr])
            rng.shuffle(tr_idx)

    if MAX_TRAIN_SEQ and len(tr_idx) > MAX_TRAIN_SEQ:
        tr_idx = np.random.default_rng(42).choice(tr_idx, size=MAX_TRAIN_SEQ, replace=False)

    pw_vals = []
    for h in HORIZONS:
        h_labels = labels_per_h[h][tr_idx]
        n_f = int(h_labels.sum())
        n_n = len(tr_idx) - n_f
        cap = 20.0 if h == 15 else 10.0
        pw_vals.append(min(n_n / max(1, n_f), cap))
    print(f"  pos_weight per horizon {HORIZONS}: {[round(w, 3) for w in pw_vals]}")

    base_ds      = TimeSeriesDataset(scaled_X, raw_y, seq_len, HORIZONS)
    train_loader = DataLoader(SubsetDataset(base_ds, tr_idx),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(SubsetDataset(base_ds, va_idx),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(SubsetDataset(base_ds, te_idx),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Dataset — train:{len(tr_idx):,}  val:{len(va_idx):,}  test:{len(te_idx):,}")

    # ── GPU setup ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device    = torch.device("cuda")
        n_gpu     = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
        print(f"  GPU: {n_gpu}× {gpu_names}")
    else:
        device = torch.device("cpu")
        n_gpu  = 0
        print("  No GPU found — training on CPU")

    model = MultiHorizonLSTM(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        n_horizons=len(HORIZONS),
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ).to(device)

    # Use both T4 GPUs if available
    if n_gpu > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel enabled across {n_gpu} GPUs")

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    pos_weight = torch.tensor(pw_vals, dtype=torch.float32)
    criterion  = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_val   = float("inf")
    patience_c = 0
    train_losses, val_losses, epoch_times = [], [], []
    model_path = os.path.join(OUT_DIR, "stress_test_best_model.pth")

    print(f"\nTraining up to {EPOCHS} epochs (early-stop patience={PATIENCE}) ...")
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        run_loss = 0.0
        for seq, tgt in train_loader:
            seq, tgt = seq.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq), tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item()
        train_loss = run_loss / max(1, len(train_loader))

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for seq, tgt in val_loader:
                v_loss += criterion(model(seq.to(device)), tgt.to(device)).item()
        val_loss = v_loss / max(1, len(val_loader))

        if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
            print(f"FATAL — NaN/Inf at epoch {epoch+1}")
            sys.exit(1)

        scheduler.step(val_loss)
        dt = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(dt)

        # Save best checkpoint (unwrap DataParallel if needed)
        if val_loss < best_val:
            best_val   = val_loss
            patience_c = 0
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, model_path)
        else:
            patience_c += 1

        print(f"Epoch {epoch+1:>3}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Patience: {patience_c}/{PATIENCE} | {dt:.1f}s")

        if patience_c >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    total_time = sum(epoch_times)

    # Reload best checkpoint for evaluation (always single-model state_dict)
    raw_model = MultiHorizonLSTM(
        input_size=len(feature_cols), hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, n_horizons=len(HORIZONS),
        dropout=DROPOUT, bidirectional=BIDIRECTIONAL,
    ).to(device)
    raw_model.load_state_dict(torch.load(model_path, map_location=device))
    raw_model.eval()

    # Test evaluation
    all_tgt, all_prob = [], []
    with torch.no_grad():
        for seq, tgt in test_loader:
            all_prob.append(torch.sigmoid(raw_model(seq.to(device))).cpu().numpy())
            all_tgt.append(tgt.numpy())
    all_tgt  = np.vstack(all_tgt)
    all_prob = np.vstack(all_prob)

    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def precision_at_k(y_true, y_score, k=100):
        k = min(k, len(y_true))
        top_k = np.argsort(y_score)[-k:]
        return float(y_true[top_k].sum() / k)

    per_horizon = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, h in enumerate(HORIZONS):
        y_t = all_tgt[:, i]
        y_s = all_prob[:, i]
        if len(np.unique(y_t)) < 2:
            auc = pr_auc = p_at_k = float("nan")
        else:
            auc    = float(roc_auc_score(y_t, y_s))
            pr_auc = float(average_precision_score(y_t, y_s))
            p_at_k = precision_at_k(y_t, y_s)
            fpr, tpr, _ = roc_curve(y_t, y_s)
            ax.plot(fpr, tpr, label=f"h={h} (ROC={auc:.3f} PR={pr_auc:.3f})")
        per_horizon[f"horizon_{h}"] = {
            "auc": auc, "pr_auc": pr_auc, "precision_at_100": p_at_k
        }
        print(f"  horizon_{h}: ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  P@100={p_at_k:.3f}")

    avg_auc    = float(np.nanmean([v["auc"]    for v in per_horizon.values()]))
    avg_pr_auc = float(np.nanmean([v["pr_auc"] for v in per_horizon.values()]))
    print(f"\n  Average ROC-AUC: {avg_auc:.4f}  |  Average PR-AUC: {avg_pr_auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Multi-Horizon LSTM v5")
    ax.legend()
    fig.tight_layout()
    roc_path = os.path.join(OUT_DIR, "lstm_roc_curves.png")
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)

    results = {
        "version":       f"v5-h{HIDDEN_SIZE}-s{SEQUENCE_LENGTH}-g{FOCAL_GAMMA}",
        "total_time":    total_time,
        "best_val_loss": float(best_val),
        "avg_auc":       avg_auc,
        "avg_pr_auc":    avg_pr_auc,
        "n_features":    len(feature_cols),
        "hidden_size":   HIDDEN_SIZE,
        "focal_gamma":   FOCAL_GAMMA,
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "per_horizon":   per_horizon,
        "summary": {
            "training_time_sec": total_time,
            "best_val_loss":     float(best_val),
            "avg_auc":           avg_auc,
            "avg_pr_auc":        avg_pr_auc,
        },
        "artifact_paths": {
            "roc_plot":    roc_path,
            "model_path":  model_path,
            "scaler_path": scaler_path,
        },
    }
    json_path = os.path.join(OUT_DIR, "lstm_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete.")
    print(f"  Model  → {model_path}")
    print(f"  Scaler → {scaler_path}")
    print(f"  ROC    → {roc_path}")
    print(f"  JSON   → {json_path}")


if __name__ == "__main__":
    train()
