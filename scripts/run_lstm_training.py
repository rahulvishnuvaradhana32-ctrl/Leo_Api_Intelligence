#!/usr/bin/env python3
"""LEO API — Multi-Horizon LSTM Predictive Reliability Training  (v5)

Usage:
    python scripts/run_lstm_training.py --epochs 30
    python scripts/run_lstm_training.py --epochs 30 --end_date 2024-12-31
    python scripts/run_lstm_training.py --epochs 30 --balance   # 50/50 on train split only
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=== run_lstm_training.py start ===")

import argparse
import json
import os
import time
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ── Exactly 43 feature columns in defined order ────────────────────────────────
FEATURE_COLS: List[str] = [
    # Core telemetry (status_code removed — r=0.98 with label, dominates learning)
    "response_time", "request_count",
    # Time features
    "hour", "day_of_week", "is_market_hours", "is_financial_peak",
    "is_weekend", "is_holiday",
    # Rolling statistics
    "response_time_rolling_mean", "response_time_rolling_std",
    "error_rate_rolling", "response_time_variance", "error_volatility",
    # Lag features
    "response_time_lag_1", "response_time_lag_5", "error_rate_lag_1",
    # EMA features
    "response_time_ema_10", "response_time_ema_30", "error_rate_ema_10",
    # Cyclical encoding
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # API flags
    "high_frequency_api", "api_complexity",
    # Event stress signals
    "error_rate_boost", "rt_multiplier",
    # Precursor signals — computed on-the-fly
    "latency_diff_1", "latency_diff_5",
    "error_rate_diff_1", "error_rate_diff_5",
    "latency_spike", "error_burst", "instability_index",
    "latency_slope", "error_slope",
    # Advanced signals — from CSV if available
    "traffic_change", "burst_ratio",
    # Cross-API correlation features — from banking_api_features_v6.csv
    "avg_error_rate_others", "max_error_rate_others",
    "n_apis_elevated", "corr_with_similar_api",
    "systemic_stress_index",
]

assert len(FEATURE_COLS) == 43, f"Expected 43 features, got {len(FEATURE_COLS)}"

EPS = 1e-6


# ── On-the-fly precursor feature computation ───────────────────────────────────
def add_precursor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute precursor features after loading and sorting the dataframe."""

    def _col(name: str, fill: float = 0.0) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(fill, index=df.index)

    rt  = df["response_time"]
    rl1 = _col("response_time_lag_1", 0.0)
    rl5 = _col("response_time_lag_5", 0.0)
    er  = _col("error_rate_rolling",  0.0)
    el1 = _col("error_rate_lag_1",    0.0)
    rm  = _col("response_time_rolling_mean", 1.0)
    e10 = _col("error_rate_ema_10",   0.0)
    re10 = _col("response_time_ema_10", 1.0)
    re30 = _col("response_time_ema_30", 1.0)

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
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to the same device as logits here, not at construction,
        # so the module is device-agnostic (CPU/GPU safe)
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        prob = torch.sigmoid(logits)
        p_t  = torch.where(targets >= 0.5, prob, 1 - prob)
        return ((1 - p_t) ** self.gamma * bce).mean()


# ── Dataset ────────────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_len: int, horizons: List[int]):
        self.X        = X
        self.y        = y
        self.seq_len  = seq_len
        self.horizons = horizons
        self.max_h    = max(horizons)
        self.n        = max(0, len(X) - seq_len - self.max_h + 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        seq     = self.X[idx : idx + self.seq_len]
        targets = [float(self.y[idx + self.seq_len + h - 1] == 0)
                   for h in self.horizons]
        return (
            torch.from_numpy(seq).float(),
            torch.tensor(targets, dtype=torch.float32),
        )


class SubsetDataset(Dataset):
    def __init__(self, base: TimeSeriesDataset, indices: np.ndarray):
        self.base    = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[int(self.indices[i])]


# ── Attention pooling over LSTM timesteps ──────────────────────────────────────
class AttentionPooling(nn.Module):
    """Learns a scalar score per timestep; weighted sum over the sequence."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden)
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)  # (batch, seq)
        return (weights.unsqueeze(-1) * x).sum(dim=1)               # (batch, hidden)


# ── Model v5 ───────────────────────────────────────────────────────────────────
class MultiHorizonLSTM(nn.Module):
    """LSTM → LayerNorm → AttentionPooling → Dropout → per-horizon Linear heads.

    LayerNorm is applied to the full (B, T, H) sequence before attention pooling
    so that attention scores are computed on normalised activations.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 n_horizons: int, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.lstm_out = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(self.lstm_out)  # on full sequence, not pooled vector
        self.attn_pool  = AttentionPooling(self.lstm_out)
        self.dropout    = nn.Dropout(dropout)
        # Separate Linear head per horizon — no single shared fc layer
        self.heads = nn.ModuleList(
            [nn.Linear(self.lstm_out, 1) for _ in range(n_horizons)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                # (batch, seq, lstm_out)
        out    = self.layer_norm(out)         # normalise before attention sees it
        out    = self.attn_pool(out)          # (batch, lstm_out) — weighted sum over seq
        out    = self.dropout(out)
        return torch.cat([h(out) for h in self.heads], dim=1)


# ── Training ───────────────────────────────────────────────────────────────────
def train(args) -> None:
    features_path = args.data
    if not os.path.exists(features_path):
        print(f"ERROR: {features_path} not found — run integrate_kaggle_data.py first.")
        sys.exit(1)

    print(f"Loading {features_path} ...")
    df = pd.read_csv(features_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    if args.start_date:
        df = df[df["timestamp"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        df = df[df["timestamp"] <= pd.to_datetime(args.end_date)]
    print(f"  After date filtering: {len(df):,} rows")

    # Compute on-the-fly precursor features
    print("Computing precursor features ...")
    df = add_precursor_features(df)

    # Resolve available features (graceful degradation for absent columns)
    available  = [c for c in FEATURE_COLS if c in df.columns]
    missing    = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [!] {len(missing)} feature(s) absent from CSV — "
              f"training on {len(available)}: {missing}")
    feature_cols = available
    print(f"  Features used: {len(feature_cols)}")

    df_work = df[feature_cols + ["success"]].copy()
    df_work[feature_cols]  = df_work[feature_cols].fillna(0).astype(np.float32)
    df_work["success"]     = df_work["success"].fillna(0).astype(np.float32)

    raw_X = df_work[feature_cols].to_numpy(dtype=np.float32)
    raw_y = df_work["success"].to_numpy(dtype=np.float32)

    # Sequence index pool
    seq_len = args.sequence_length
    max_h   = max(args.horizons)
    n_seqs  = len(raw_X) - seq_len - max_h + 1
    if n_seqs < 100:
        print(f"ERROR: only {n_seqs} sequences — dataset too small.")
        sys.exit(1)

    seq_idx     = np.arange(n_seqs)
    fail_labels = np.array(
        [int(raw_y[i + seq_len + max_h - 1] == 0) for i in seq_idx]
    )

    # Pre-compute per-horizon failure labels for every sequence index.
    # Used for stratified pw_vals later — avoids re-running the index arithmetic
    # after tr_idx has been mutated by balancing/capping.
    labels_per_h = {
        h: np.array([int(raw_y[i + seq_len + h - 1] == 0) for i in seq_idx])
        for h in args.horizons
    }

    # Stratified 80 / 10 / 10 split
    tr_idx, tmp_idx, tr_lbl, tmp_lbl = train_test_split(
        seq_idx, fail_labels, test_size=0.20, random_state=42, stratify=fail_labels
    )
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.50, random_state=42, stratify=tmp_lbl
    )

    # Fit scaler on first 80% of rows — clean chronological boundary,
    # independent of the stratified split shuffling (tr_idx.max() is post-shuffle
    # and can leak val/test rows into scaler fitting)
    train_row_end = int(0.80 * len(raw_X))
    print("Fitting StandardScaler on training rows ...")
    scaler   = StandardScaler()
    scaler.fit(raw_X[:train_row_end])
    scaled_X = scaler.transform(raw_X).astype(np.float32)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("  Scaler saved → models/scaler.pkl")

    # Optional 50/50 balanced sampling — training split only
    if args.balance:
        fail_tr   = tr_idx[fail_labels[tr_idx] == 1]
        normal_tr = tr_idx[fail_labels[tr_idx] == 0]
        if len(fail_tr) > 0 and len(normal_tr) > len(fail_tr):
            rng       = np.random.default_rng(42)
            normal_tr = rng.choice(normal_tr, size=len(fail_tr), replace=False)
            tr_idx    = np.concatenate([fail_tr, normal_tr])
            rng.shuffle(tr_idx)
            print(f"Balanced train: {len(fail_tr):,} failures + {len(fail_tr):,} normal "
                  f"= {len(tr_idx):,} sequences (50/50)")
        else:
            print("Balance flag set but insufficient failures — skipping.")
    else:
        fail_count = int(fail_labels[tr_idx].sum())
        print(f"Natural distribution: {len(tr_idx):,} train sequences "
              f"({fail_count:,} failures = {100*fail_count/max(1,len(tr_idx)):.1f}%)")

    # Cap training sequences
    if args.max_train_sequences and len(tr_idx) > args.max_train_sequences:
        rng    = np.random.default_rng(42)
        tr_idx = rng.choice(tr_idx, size=args.max_train_sequences, replace=False)
        print(f"Capped train to {len(tr_idx):,} sequences")

    # Per-horizon pos_weight — horizon 15 has far fewer failures than horizon 1,
    # so each head gets an independently calibrated weight, capped at 10.0.
    # Index pre-computed labels_per_h instead of re-running raw_y arithmetic
    # (safe even when tr_idx is a capped/shuffled subset).
    pw_vals: List[float] = []
    for h in args.horizons:
        h_labels = labels_per_h[h][tr_idx]
        n_f = int(h_labels.sum())
        n_n = len(tr_idx) - n_f
        cap = 20.0 if h == 15 else 10.0
        pw_vals.append(min(n_n / max(1, n_f), cap))
    print(f"  pos_weight per horizon {args.horizons}: "
          f"{[round(w, 3) for w in pw_vals]}")

    base_ds  = TimeSeriesDataset(scaled_X, raw_y, seq_len, args.horizons)
    train_ds = SubsetDataset(base_ds, tr_idx)
    val_ds   = SubsetDataset(base_ds, va_idx)
    test_ds  = SubsetDataset(base_ds, te_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print(f"\nDataset — train:{len(train_ds):,}  val:{len(val_ds):,}  "
          f"test:{len(test_ds):,}")

    device = torch.device("cpu")
    model  = MultiHorizonLSTM(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        n_horizons=len(args.horizons),
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
          f"  (bidirectional={args.bidirectional})")

    # Keep pos_weight on CPU at construction; FocalLoss.forward() moves it to
    # logits.device at runtime so the module stays device-agnostic.
    pos_weight = torch.tensor(pw_vals, dtype=torch.float32)
    criterion  = FocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    # ReduceLROnPlateau reacts to actual val loss — correct for early stopping
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_val_loss  = float("inf")
    patience_count = 0
    train_losses: List[float] = []
    val_losses:   List[float] = []
    epoch_times:  List[float] = []

    print(f"\nTraining up to {args.epochs} epochs "
          f"(early stopping patience={args.patience}) ...")

    for epoch in range(args.epochs):
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
                v_loss += criterion(model(seq.to(device)),
                                    tgt.to(device)).item()
        val_loss = v_loss / max(1, len(val_loader))

        # ── NaN/Inf loss guard ─────────────────────────────────────────────
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            print(
                f"FATAL — NaN/Inf loss at epoch {epoch+1}: "
                f"train={train_loss}  val={val_loss}  pw_vals={pw_vals}",
                file=sys.stderr,
            )
            print("  Fix suggestions:", file=sys.stderr)
            print("    1. Lower pos_weight cap below 10.0 (try 5.0)", file=sys.stderr)
            print("    2. Lower --lr to 1e-4", file=sys.stderr)
            print("    3. Lower --focal_gamma to 1.0", file=sys.stderr)
            print("    4. Check for NaN in input features: np.isnan(scaled_X).any()", file=sys.stderr)
            if os.path.exists("models/stress_test_best_model.pth"):
                print("  Prior checkpoint models/stress_test_best_model.pth is preserved.", file=sys.stderr)
            sys.exit(1)
        # ───────────────────────────────────────────────────────────────────

        # ReduceLROnPlateau reacts to val_loss — step after val is computed
        scheduler.step(val_loss)

        dt = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(dt)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), "models/stress_test_best_model.pth")
        else:
            patience_count += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:>3}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_count}/{args.patience} | {dt:.1f}s")

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    total_time = sum(epoch_times)

    # Reload best checkpoint before evaluation
    model.load_state_dict(
        torch.load("models/stress_test_best_model.pth", map_location=device)
    )
    model.eval()

    # Test-set evaluation
    all_tgt, all_prob = [], []
    with torch.no_grad():
        for seq, tgt in test_loader:
            all_prob.append(torch.sigmoid(model(seq.to(device))).cpu().numpy())
            all_tgt.append(tgt.numpy())
    all_tgt  = np.vstack(all_tgt)
    all_prob = np.vstack(all_prob)

    from sklearn.metrics import (
        average_precision_score, roc_auc_score, roc_curve,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> float:
        """Fraction of actual failures among the top-k highest-scored predictions.
        Directly answers: 'of the top-K alerts fired, how many were real failures?'
        """
        k = min(k, len(y_true))
        top_k = np.argsort(y_score)[-k:]
        return float(y_true[top_k].sum() / k)

    per_horizon: dict = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, h in enumerate(args.horizons):
        y_t = all_tgt[:, i]
        y_s = all_prob[:, i]
        if len(np.unique(y_t)) < 2:
            auc = pr_auc = p_at_k = float("nan")
            print(f"  horizon_{h}: AUC=nan (single class in test set)")
        else:
            auc    = float(roc_auc_score(y_t, y_s))
            pr_auc = float(average_precision_score(y_t, y_s))
            p_at_k = precision_at_k(y_t, y_s, k=100)
            fpr, tpr, _ = roc_curve(y_t, y_s)
            ax.plot(fpr, tpr, label=f"h={h} (ROC={auc:.3f} PR={pr_auc:.3f})")
        per_horizon[f"horizon_{h}"] = {
            "auc":              auc,
            "pr_auc":           pr_auc,
            "precision_at_100": p_at_k,
        }
        print(f"  horizon_{h}: ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}"
              f"  Precision@100={p_at_k:.3f}")

    avg_auc    = float(np.nanmean([v["auc"]    for v in per_horizon.values()]))
    avg_pr_auc = float(np.nanmean([v["pr_auc"] for v in per_horizon.values()]))
    print(f"\n  Average ROC-AUC: {avg_auc:.4f}  |  Average PR-AUC: {avg_pr_auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Multi-Horizon LSTM v5")
    ax.legend()
    fig.tight_layout()
    fig.savefig("models/lstm_roc_curves.png", dpi=120)
    plt.close(fig)

    results = {
        "version":        "v5",
        "total_time":     total_time,
        "best_val_loss":  float(best_val_loss),
        "avg_auc":        avg_auc,
        "avg_pr_auc":     avg_pr_auc,
        "n_features":     len(feature_cols),
        "hidden_size":    args.hidden_size,
        "focal_gamma":    args.focal_gamma,
        "train_losses":   train_losses,
        "val_losses":     val_losses,
        "per_horizon":    per_horizon,
        "summary": {
            "training_time_sec": total_time,
            "best_val_loss":     float(best_val_loss),
            "avg_auc":           avg_auc,
            "avg_pr_auc":        avg_pr_auc,
        },
        "artifact_paths": {
            "roc_plot":    "models/lstm_roc_curves.png",
            "model_path":  "models/stress_test_best_model.pth",
            "scaler_path": "models/scaler.pkl",
        },
    }
    with open("models/lstm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nTraining & evaluation complete.")
    print("  Model  → models/stress_test_best_model.pth")
    print("  Scaler → models/scaler.pkl")
    print("  ROC    → models/lstm_roc_curves.png")
    print("  JSON   → models/lstm_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEO API Multi-Horizon LSTM Training v5"
    )
    parser.add_argument("--epochs",              type=int,   default=30)
    parser.add_argument("--batch_size",          type=int,   default=128)
    parser.add_argument("--sequence_length",     type=int,   default=30)
    parser.add_argument("--horizons",            nargs="+",  type=int,
                        default=[1, 5, 15])
    parser.add_argument("--hidden_size",         type=int,   default=128)
    parser.add_argument("--num_layers",           type=int,   default=2)
    parser.add_argument("--dropout",             type=float, default=0.3)
    parser.add_argument("--no_bidirectional",    dest="bidirectional",
                        action="store_false",
                        help="Disable bidirectional LSTM (ablation; bidir is default ON)")
    parser.set_defaults(bidirectional=True)
    parser.add_argument("--lr",                  type=float, default=0.001)
    parser.add_argument("--patience",            type=int,   default=6)
    parser.add_argument("--max_train_sequences", type=int,   default=200_000,
                        help="Cap training sequences (default 200,000)")
    parser.add_argument("--focal_gamma",         type=float, default=2.0)
    parser.add_argument("--balance",             action="store_true",
                        default=False,
                        help="50/50 balanced sampling on training split only")
    parser.add_argument("--start_date",          type=str,   default=None,
                        help="Filter rows from YYYY-MM-DD (inclusive)")
    parser.add_argument("--end_date",            type=str,   default=None,
                        help="Filter rows up to YYYY-MM-DD (inclusive)")
    parser.add_argument("--data",                type=str,
                        default=os.path.join("data", "banking_api_features_clean.csv"))
    args = parser.parse_args()
    train(args)
