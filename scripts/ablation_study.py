#!/usr/bin/env python3
"""
ablation_study.py  --  Feature group importance for the FCE Multi-Horizon LSTM.

Removes one feature group at a time, retrains a lightweight model, and
measures how much average AUC drops.  A larger drop = the group matters more.

Output:
    models/ablation_results.json   -- full per-horizon AUC table
    models/ablation_results.png    -- ranked horizontal bar chart

Usage:
    python scripts/ablation_study.py
    python scripts/ablation_study.py --recent_rows 200000 --epochs 3
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Ablation study for FCE LSTM")
parser.add_argument("--recent_rows", type=int, default=300_000)
parser.add_argument("--epochs",      type=int, default=5)
parser.add_argument("--max_seq",     type=int, default=50_000)
parser.add_argument("--batch_size",  type=int, default=128)
parser.add_argument("--seq_len",     type=int, default=30)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--data_path",   type=str, default="data/banking_api_features.csv")
args = parser.parse_args()

HORIZONS    = [1, 5, 15]
MAX_H       = max(HORIZONS)
SEQ_LEN     = args.seq_len
HIDDEN      = args.hidden_size
NUM_LAYERS  = 2
DROPOUT     = 0.3
FOCAL_GAMMA = 2.0
LR          = 0.001
SEED        = 42

# ── All 28 feature columns (ordered as in run_lstm_training.py) ───────────────
ALL_FEATURES = [
    "response_time", "status_code", "request_count",
    "hour", "day_of_week", "is_market_hours", "is_financial_peak",
    "is_weekend", "is_holiday",
    "response_time_rolling_mean", "response_time_rolling_std",
    "error_rate_rolling", "response_time_variance", "error_volatility",
    "response_time_lag_1", "response_time_lag_5", "error_rate_lag_1",
    "response_time_ema_10", "response_time_ema_30", "error_rate_ema_10",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "high_frequency_api", "api_complexity",
    "error_rate_boost", "rt_multiplier",
    # precursor signals (computed on-the-fly from existing engineered columns)
    "latency_diff_1", "latency_diff_5",
    "error_rate_diff_1", "error_rate_diff_5",
    "latency_spike", "error_burst", "instability_index",
    "latency_slope", "error_slope",
    # advanced (present only if add_precursor_features.py was run)
    "traffic_change", "burst_ratio",
]

# ── 8 experiments: (display name, features to remove) ─────────────────────────
EXPERIMENTS = [
    ("Baseline",              []),
    ("No Event Signals",      ["error_rate_boost", "rt_multiplier"]),
    ("No Rolling Stats",      ["response_time_rolling_mean", "response_time_rolling_std",
                               "response_time_variance", "error_rate_rolling", "error_volatility"]),
    ("No Lag Features",       ["response_time_lag_1", "response_time_lag_5", "error_rate_lag_1"]),
    ("No EMA Features",       ["response_time_ema_10", "response_time_ema_30", "error_rate_ema_10"]),
    ("No Cyclical Enc.",      ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]),
    ("No API Flags",          ["high_frequency_api", "api_complexity"]),
    ("No Precursor Signals",  ["latency_diff_1", "latency_diff_5",
                               "error_rate_diff_1", "error_rate_diff_5",
                               "latency_spike", "error_burst", "instability_index",
                               "latency_slope", "error_slope",
                               "traffic_change", "burst_ratio"]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared model classes  (identical to run_lstm_training.py — do not modify)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


class TimeSeriesDataset(Dataset):
    def __init__(self, data_arr, targets_arr, seq_len, horizons, scaler):
        data_arr       = scaler.transform(data_arr).astype(np.float32)
        self._data     = data_arr
        self._targets  = targets_arr.astype(np.float32)
        self.seq_len   = seq_len
        self.horizons  = horizons
        self.n         = len(self._data) - seq_len - max(horizons) + 1
        if self.n < 1:
            raise ValueError("Dataset too small for seq_len + max_horizon")

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, i):
        seq     = self._data[i: i + self.seq_len].copy()
        targets = [float(1 - self._targets[i + self.seq_len + h - 1])
                   for h in self.horizons]
        return {
            "sequence": torch.from_numpy(seq),
            "targets":  torch.tensor(targets, dtype=torch.float32),
        }


class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 output_size=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.heads   = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(output_size)
        ])

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        return torch.cat([head(out) for head in self.heads], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Core experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(name: str, col_indices: list[int], col_names: list[str],
                   raw_X_full: np.ndarray, raw_y: np.ndarray,
                   tr_idx: np.ndarray, val_idx: np.ndarray, te_idx: np.ndarray,
                   strat: np.ndarray) -> dict:
    """
    Train a model using only the columns at col_indices and evaluate on
    the fixed test split.  Returns a dict with per-horizon AUC and avg AUC.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    raw_X = raw_X_full[:, col_indices]      # slice to this experiment's features
    n_feat = raw_X.shape[1]

    # Scaler fitted on training rows only (no leakage)
    train_row_end = int(tr_idx.max()) + SEQ_LEN + MAX_H
    scaler        = StandardScaler().fit(raw_X[:train_row_end])

    full_ds  = TimeSeriesDataset(raw_X, raw_y, SEQ_LEN, HORIZONS, scaler)
    train_ds = Subset(full_ds, tr_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    test_ds  = Subset(full_ds, te_idx.tolist())

    # Focal loss with pos_weight from training split
    n_fail    = float((strat[tr_idx] == 1).sum())
    n_success = float((strat[tr_idx] == 0).sum())
    pw        = torch.tensor([min(n_success / max(n_fail, 1), 10.0)] * len(HORIZONS))
    criterion = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=pw)

    model     = MultiHorizonLSTM(n_feat, HIDDEN, NUM_LAYERS, len(HORIZONS), DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Cap training sequences
    tr_use = tr_idx
    if args.max_seq > 0 and len(tr_use) > args.max_seq:
        tr_use = np.random.default_rng(SEED).choice(tr_use, size=args.max_seq, replace=False)
    train_ds_capped = Subset(full_ds, tr_use.tolist())

    train_loader = DataLoader(train_ds_capped, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,          batch_size=256,
                              shuffle=False, num_workers=0)

    # ── Training (fixed epochs, no early stopping for reproducibility) ────────
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch["sequence"]), batch["targets"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                vl += criterion(model(batch["sequence"]), batch["targets"]).item()
        val_loss = vl / max(1, len(val_loader))
        print(f"    epoch {epoch+1}/{args.epochs}  val_loss={val_loss:.5f}", flush=True)

    elapsed = time.time() - t0

    # ── Evaluate on fixed test set ────────────────────────────────────────────
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for batch in test_loader:
            all_t.append(batch["targets"].numpy())
            all_p.append(torch.sigmoid(model(batch["sequence"])).numpy())

    all_t = np.vstack(all_t)
    all_p = np.vstack(all_p)

    per_horizon = {}
    valid_aucs  = []
    for i, h in enumerate(HORIZONS):
        y_true = all_t[:, i]
        y_pred = all_p[:, i]
        if len(np.unique(y_true)) < 2:
            per_horizon[h] = float("nan")
        else:
            auc = float(roc_auc_score(y_true, y_pred))
            per_horizon[h] = auc
            valid_aucs.append(auc)

    avg_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    return {
        "name":           name,
        "n_features":     n_feat,
        "per_horizon":    per_horizon,
        "avg_auc":        avg_auc,
        "train_time_sec": round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: list[dict], baseline_auc: float, out_path: str):
    # Sort ablation experiments by AUC drop descending (most important first)
    ablations = [r for r in results if r["name"] != "Baseline"]
    ablations.sort(key=lambda r: r["auc_drop"], reverse=True)

    names  = [r["name"]     for r in ablations]
    drops  = [r["auc_drop"] for r in ablations]
    colors = []
    for d in drops:
        if d > 0.05:
            colors.append("#e74c3c")       # red — critical
        elif d >= 0.02:
            colors.append("#e67e22")       # orange — significant
        else:
            colors.append("#27ae60")       # green — minor

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos   = range(len(names))
    bars    = ax.barh(y_pos, drops, color=colors, height=0.55, edgecolor="white")

    # Value labels on bars
    for bar, drop in zip(bars, drops):
        x_label = bar.get_width() + 0.001
        ax.text(x_label, bar.get_y() + bar.get_height() / 2,
                f"{drop:+.4f}", va="center", fontsize=9, color="#2c3e50")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("AUC Drop vs Baseline", fontsize=11)
    ax.set_title(
        f"Feature Group Importance  (Baseline AUC = {baseline_auc:.4f})\n"
        "FCE Multi-Horizon LSTM — Ablation Study",
        fontsize=12, pad=12,
    )
    ax.axvline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")
    ax.set_xlim(left=min(-0.005, min(drops) - 0.01),
                right=max(drops) + 0.025)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#e74c3c", label="Critical  (drop > 0.05)"),
        Patch(facecolor="#e67e22", label="Significant  (0.02 – 0.05)"),
        Patch(facecolor="#27ae60", label="Minor  (< 0.02)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print("=== FCE Ablation Study ===\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.data_path):
        print(f"ERROR: {args.data_path} not found"); sys.exit(1)

    print(f"Loading {args.data_path} ...")
    df = pd.read_csv(args.data_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["success"].dtype == object:
        df["success"] = df["success"].map({"True": 1, "False": 0}).fillna(0).astype(int)
    else:
        df["success"] = df["success"].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Total rows: {len(df):,}")

    # Prefer the most recent window; fall back to pre-2025 synthetic data if
    # failure rate < 5% (which happens when Kaggle zero-failure rows dominate)
    df_window = df.tail(args.recent_rows)
    fail_rate = 1 - df_window["success"].mean()

    if fail_rate < 0.05:
        print(f"  [!] Recent {args.recent_rows:,} rows have only {fail_rate:.2%} failures.")
        print("      Falling back to pre-2025 synthetic window for meaningful ablation.")
        synthetic = df[df["timestamp"] < "2025-01-01"]
        if len(synthetic) >= args.recent_rows:
            df_window = synthetic.tail(args.recent_rows)
        else:
            df_window = synthetic
            print(f"      Using all {len(df_window):,} available synthetic rows.")
        fail_rate = 1 - df_window["success"].mean()

    df_window = df_window.reset_index(drop=True)
    print(f"  Window: {len(df_window):,} rows | "
          f"failure rate: {fail_rate:.2%} | "
          f"{df_window['timestamp'].min().date()} -> {df_window['timestamp'].max().date()}")

    # ── Compute precursor features on-the-fly (identical logic to run_lstm_training.py)
    _EPS = 1e-6
    df_window["latency_diff_1"]    = (df_window["response_time"] - df_window["response_time_lag_1"]).fillna(0)
    df_window["latency_diff_5"]    = (df_window["response_time"] - df_window["response_time_lag_5"]).fillna(0)
    df_window["error_rate_diff_1"] = (df_window["error_rate_rolling"] - df_window["error_rate_lag_1"]).fillna(0)
    df_window["error_rate_diff_5"] = (df_window["error_rate_rolling"] - df_window["error_rate_ema_10"]).fillna(0)
    df_window["latency_spike"]     = df_window["response_time"] / (df_window["response_time_rolling_mean"] + _EPS)
    df_window["error_burst"]       = df_window["error_rate_rolling"] / (df_window["error_rate_ema_10"] + _EPS)
    df_window["instability_index"] = df_window["latency_diff_1"].abs() + df_window["error_rate_diff_1"].abs()
    df_window["latency_slope"]     = (df_window["response_time_ema_10"] - df_window["response_time_ema_30"]) / 20.0
    df_window["error_slope"]       = (df_window["error_rate_ema_10"] - df_window["error_rate_lag_1"]).fillna(0) / 10.0

    # Keep only columns that actually exist in the CSV
    available = [c for c in ALL_FEATURES if c in df_window.columns]
    missing   = [c for c in ALL_FEATURES if c not in df_window.columns]
    if missing:
        print(f"  [!] Missing features (will skip from all experiments): {missing}")

    raw_X_full = df_window[available].fillna(0).to_numpy(dtype=np.float64)
    raw_y      = df_window["success"].to_numpy(dtype=np.float32)
    print(f"  Features available: {len(available)}/{len(ALL_FEATURES)}\n")

    # ── Compute shared stratified splits ONCE ─────────────────────────────────
    # All 7 experiments use these exact same indices for a fair comparison.
    n_seq = len(raw_X_full) - SEQ_LEN - MAX_H + 1
    if n_seq < 500:
        print("ERROR: Window too small to form sequences."); sys.exit(1)

    strat   = 1 - raw_y[SEQ_LEN + MAX_H - 1: SEQ_LEN + MAX_H - 1 + n_seq].astype(int)
    idx_all = np.arange(n_seq)
    tr_idx, tmp_idx = train_test_split(idx_all, test_size=0.30,
                                       random_state=SEED, stratify=strat)
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.50,
                                       random_state=SEED, stratify=strat[tmp_idx])

    print(f"Shared splits -- train: {len(tr_idx):,}  "
          f"val: {len(val_idx):,}  test: {len(te_idx):,}")
    print(f"Test failure rate (h=15 target): {strat[te_idx].mean():.2%}\n")

    # ── Run all experiments ───────────────────────────────────────────────────
    exp_results = []
    for exp_name, removed in EXPERIMENTS:
        # Build column indices for this experiment
        cols_to_use = [c for c in available if c not in removed]
        # Skip features not present in CSV (they are already excluded from `available`)
        col_indices = [available.index(c) for c in cols_to_use]
        n_removed   = len(removed)

        print(f"{'-'*60}")
        print(f"Experiment: {exp_name}")
        if removed:
            actually_removed = [c for c in removed if c in available]
            skipped          = [c for c in removed if c not in available]
            print(f"  Removing  : {actually_removed}")
            if skipped:
                print(f"  Not in CSV: {skipped} (already absent)")
        else:
            print(f"  Removing  : nothing (full feature set)")
        print(f"  Features  : {len(col_indices)}")

        result = run_experiment(
            exp_name, col_indices, cols_to_use,
            raw_X_full, raw_y,
            tr_idx, val_idx, te_idx, strat,
        )
        print(f"  AUC (avg) : {result['avg_auc']:.4f}  "
              f"({result['train_time_sec']:.0f}s)")
        exp_results.append(result)

    # ── Compute AUC drops relative to baseline ────────────────────────────────
    baseline     = next(r for r in exp_results if r["name"] == "Baseline")
    baseline_auc = baseline["avg_auc"]

    def importance_label(drop: float) -> str:
        if drop > 0.05:   return "Critical"
        if drop >= 0.02:  return "Significant"
        return "Minor"

    out_experiments = []
    for r in exp_results:
        drop = baseline_auc - r["avg_auc"]
        out_experiments.append({
            "name":             r["name"],
            "features_removed": [c for _, removed in EXPERIMENTS
                                  if _ == r["name"] for c in removed],
            "n_features":       r["n_features"],
            "auc":              round(r["avg_auc"], 6),
            "auc_drop":         round(drop, 6),
            "importance":       importance_label(drop),
            "per_horizon_auc":  {f"h{h}": round(r["per_horizon"][h], 6)
                                 for h in HORIZONS},
            "train_time_sec":   r["train_time_sec"],
        })
        # Attach drop + importance back onto the result for charting
        r["auc_drop"] = drop

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    json_path = "models/ablation_results.json"
    payload   = {
        "baseline_auc":   round(baseline_auc, 6),
        "data_window":    args.recent_rows,
        "epochs":         args.epochs,
        "max_sequences":  args.max_seq,
        "experiments":    out_experiments,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved -> {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(exp_results, baseline_auc, "models/ablation_results.png")

    # ── Ranked table ──────────────────────────────────────────────────────────
    ablations = [r for r in out_experiments if r["name"] != "Baseline"]
    ablations.sort(key=lambda r: r["auc_drop"], reverse=True)

    col_w = [22, 8, 8, 8, 8, 13]
    header = (f"{'Feature Group':<{col_w[0]}}  "
              f"{'AUC':>{col_w[1]}}  "
              f"{'Drop':>{col_w[2]}}  "
              f"{'Drop%':>{col_w[3]}}  "
              f"{'N feat':>{col_w[4]}}  "
              f"{'Importance':<{col_w[5]}}")
    sep = "-" * len(header)
    dbl = "=" * len(header)

    print(f"\n{dbl}")
    print("  ABLATION RESULTS -- Feature Group Importance Ranking")
    print(f"{dbl}")
    print(f"  Baseline AUC: {baseline_auc:.4f}  "
          f"(epochs={args.epochs}, max_seq={args.max_seq:,})")
    print(f"\n  {header}")
    print(f"  {sep}")

    for r in ablations:
        drop_pct = (r["auc_drop"] / max(baseline_auc, 1e-9)) * 100
        print(f"  {r['name']:<{col_w[0]}}  "
              f"{r['auc']:>{col_w[1]}.4f}  "
              f"{r['auc_drop']:>+{col_w[2]}.4f}  "
              f"{drop_pct:>{col_w[3]}.1f}%  "
              f"{r['n_features']:>{col_w[4]}}  "
              f"{r['importance']:<{col_w[5]}}")

    print(f"  {sep}")
    print(f"  {'Baseline':<{col_w[0]}}  {baseline_auc:>{col_w[1]}.4f}  "
          f"{'0.0000':>{col_w[2]}}  {'0.0%':>{col_w[3]}}  "
          f"{baseline['n_features']:>{col_w[4]}}  {'Reference':<{col_w[5]}}")

    # ── One-sentence conclusion ───────────────────────────────────────────────
    top          = ablations[0]
    removed_feat = top["features_removed"]
    drop_pct_top = (top["auc_drop"] / max(baseline_auc, 1e-9)) * 100

    feature_desc = {
        "No Event Signals":     "Event stress signals (error_rate_boost, rt_multiplier)",
        "No Rolling Stats":     "Rolling statistics (mean, std, variance, error rate, volatility)",
        "No Lag Features":      "Lag features (response_time_lag_1/5, error_rate_lag_1)",
        "No EMA Features":      "EMA features (response_time_ema_10/30, error_rate_ema_10)",
        "No Cyclical Enc.":     "Cyclical time encoding (hour_sin/cos, dow_sin/cos)",
        "No API Flags":         "API flags (high_frequency_api, api_complexity)",
        "No Precursor Signals": "Precursor signals (diffs, spike, burst, instability, slope)",
    }
    desc = feature_desc.get(top["name"], top["name"])
    print(f"\n  Conclusion: {desc} {'are' if ',' in desc else 'is'} the most important "
          f"feature group -- removing {'them' if ',' in desc else 'it'} drops "
          f"AUC by {top['auc_drop']:.4f} ({drop_pct_top:.1f}%).")

    elapsed_total = time.time() - t_total
    print(f"\n  Total runtime: {elapsed_total/60:.1f} min")
    print(f"{dbl}\n")


if __name__ == "__main__":
    main()
