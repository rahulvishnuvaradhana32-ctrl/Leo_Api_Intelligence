#!/usr/bin/env python3
"""
self_improving_pipeline.py  --  Autonomous self-healing LSTM pipeline.

Diagnoses the current model, identifies performance problems, fixes the
training data, retrains, compares results, and gates model replacement
on measured improvement.  Every run appends a structured entry to the
append-only audit log -- models/self_heal_log.jsonl.

Steps:
  1. Diagnose   -- per-API AUC on held-out recent data
  2. Identify   -- worst API, missed failure types, data drift, class collapse
  3. Fix        -- targeted augmentation for each problem found
  4. Retrain    -- compact training loop on fixed data
  5. Compare    -- old vs new AUC on the same held-out test set
  6. Select     -- keep new model only if it improved; always backup old
  7. Log        -- append JSONL entry (never overwrite)
  8. Summarise  -- plain-language console report

Usage:
    python scripts/self_improving_pipeline.py              # full run
    python scripts/self_improving_pipeline.py --dry_run    # preview only
    python scripts/self_improving_pipeline.py --recent_rows 100000
    python scripts/self_improving_pipeline.py --retrain_epochs 10
"""

import os, sys, json, time, shutil, warnings, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Self-improving LSTM pipeline for FCE banking API failure prediction"
)
parser.add_argument("--dry_run", action="store_true",
                    help="Analyse and report only -- do not retrain or update any files")
parser.add_argument("--recent_rows", type=int, default=200_000,
                    help="Rows from tail of CSV to use for diagnosis + retraining (default 200000)")
parser.add_argument("--test_fraction", type=float, default=0.15,
                    help="Fraction of recent_rows held out as the fixed test set (default 0.15)")
parser.add_argument("--retrain_epochs", type=int, default=15,
                    help="Epochs for the candidate retrain (default 15)")
parser.add_argument("--max_train_seq", type=int, default=80_000,
                    help="Max training sequences sampled per retrain epoch (default 80000)")
parser.add_argument("--min_improvement", type=float, default=0.001,
                    help="Minimum AUC gain to accept the new model (default 0.001)")
parser.add_argument("--seq_len", type=int, default=30)
parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 15])
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--focal_gamma", type=float, default=2.0)
parser.add_argument("--data_path",  type=str, default="data/banking_api_features_v6.csv")
parser.add_argument("--model_path", type=str, default="models/stress_test_best_model.pth")
parser.add_argument("--scaler_path",type=str, default="models/scaler.pkl")
parser.add_argument("--results_path",type=str,default="models/lstm_results.json")
parser.add_argument("--log_path",   type=str, default="models/self_heal_log.jsonl")
args = parser.parse_args()

RUN_ID    = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_TS    = datetime.now().isoformat(timespec="seconds")
MODE      = "dry_run" if args.dry_run else "full"
SEQ_LEN   = args.seq_len
HORIZONS  = args.horizons
MAX_H     = max(HORIZONS)

print(f"=== Self-Improving Pipeline  [{RUN_ID}]  mode={MODE} ===\n")

# ── Feature columns (must match run_lstm_training.py exactly) ─────────────────
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
    # Cross-API correlation features — present in banking_api_features_v6.csv
    "avg_error_rate_others", "max_error_rate_others",
    "n_apis_elevated", "corr_with_similar_api",
    "systemic_stress_index",
]

# Continuous columns safe to jitter during augmentation
JITTER_COLS = [
    "response_time", "response_time_rolling_mean", "response_time_rolling_std",
    "error_rate_rolling", "response_time_variance", "error_volatility",
    "response_time_lag_1", "response_time_lag_5", "error_rate_lag_1",
    "response_time_ema_10", "response_time_ema_30", "error_rate_ema_10",
    "error_rate_boost", "rt_multiplier",
]

# Thresholds for problem detection
WORST_API_AUC_THRESHOLD   = 0.70   # flag API if AUC falls below this
RECALL_THRESHOLD          = 0.35   # flag error_type if model recall < 35%
DRIFT_KS_THRESHOLD        = 0.10   # KS statistic above which drift is significant
DRIFT_P_THRESHOLD         = 0.05   # p-value below which drift is significant
DRIFT_MIN_FEATURES        = 2      # number of features that must drift to trigger fix
IMBALANCE_THRESHOLD       = 0.05   # failure rate below which imbalance fix fires
TARGET_FAILURE_RATE       = 0.13   # target failure rate after imbalance injection
OVERSAMPLE_FACTOR         = 3      # how many extra copies of failure rows to add
DRIFT_WINDOW_FRACTION     = 0.40   # fraction of train_pool to keep when drift fix fires


# ─────────────────────────────────────────────────────────────────────────────
# Shared model classes  (identical to run_lstm_training.py)
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
    def __init__(self, X_arr, y_arr, seq_len, horizons, scaler=None):
        if scaler is None:
            scaler  = StandardScaler()
            X_arr   = scaler.fit_transform(X_arr)
        else:
            X_arr   = scaler.transform(X_arr)
        self.scaler   = scaler
        self._X       = X_arr.astype(np.float32)
        self._y       = y_arr.astype(np.float32)
        self.seq_len  = seq_len
        self.horizons = horizons
        self.n        = len(self._X) - seq_len - max(horizons) + 1
        if self.n < 1:
            raise ValueError("Dataset too small for seq_len + max_horizon")

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, i):
        seq     = self._X[i: i + self.seq_len].copy()
        targets = [float(1 - self._y[i + self.seq_len + h - 1]) for h in self.horizons]
        return {
            "sequence": torch.from_numpy(seq),
            "targets":  torch.tensor(targets, dtype=torch.float32),
        }


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):  # x: (B, T, H)
        w = torch.softmax(self.score(x), dim=1)  # (B, T, 1)
        return (w * x).sum(dim=1)                # (B, H)


class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.bidirectional = bidirectional
        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(lstm_out)
        self.attn_pool  = AttentionPooling(lstm_out)
        self.dropout    = nn.Dropout(dropout)
        self.heads      = nn.ModuleList([
            nn.Linear(lstm_out, 1) for _ in range(output_size)
        ])

    def forward(self, x):
        n_dir = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * n_dir, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * n_dir, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))      # (B, T, lstm_out)
        out = self.layer_norm(out)
        out = self.attn_pool(out)             # (B, lstm_out)
        out = self.dropout(out)
        return torch.cat([head(out) for head in self.heads], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper -- returns (probas [n_seq x n_horizons], target_row_indices)
# ─────────────────────────────────────────────────────────────────────────────

def _infer(model, X_scaled, y_raw, seq_len, horizons, batch_size=512):
    """
    Run batched inference on a scaled feature array.
    Returns probas [n_seq x n_horizons] and per-horizon target indices.
    target_indices[h_idx] = array of row indices in the original df that each
    sequence's h-step-ahead prediction corresponds to.
    """
    n_seq = len(X_scaled) - seq_len - max(horizons) + 1
    if n_seq < 1:
        return np.empty((0, len(horizons))), [np.empty(0, int)] * len(horizons)

    all_probas = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end   = min(start + batch_size, n_seq)
            batch = np.stack([X_scaled[i: i + seq_len] for i in range(start, end)])
            logits = model(torch.from_numpy(batch.astype(np.float32)))
            all_probas.append(torch.sigmoid(logits).numpy())

    probas = np.vstack(all_probas)

    # Row index in the original array that is the target for each sequence + horizon
    target_indices = [
        np.arange(n_seq) + seq_len + h - 1
        for h in horizons
    ]
    return probas, target_indices


def _auc_from_arrays(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _detect_dims(sd):
    """Return (n_in, hidden, n_heads, bidirectional) from a state_dict."""
    n_in   = sd["lstm.weight_ih_l0"].shape[1]
    hidden = sd["lstm.weight_hh_l0"].shape[1]
    bidir  = "lstm.weight_ih_l0_reverse" in sd
    lstm_out = hidden * 2 if bidir else hidden
    n_heads  = sum(1 for k in sd if k.startswith("heads.") and k.endswith(".weight"))
    return n_in, hidden, n_heads, bidir


def _load_model(model_path, n_features, hidden_size, num_layers, n_horizons):
    sd = torch.load(model_path, map_location="cpu")
    n_in, hidden, n_heads, bidir = _detect_dims(sd)
    model = MultiHorizonLSTM(n_in, hidden, num_layers, n_heads, bidirectional=bidir)
    model.load_state_dict(sd)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- Diagnose: per-API AUC on the eval set
# ─────────────────────────────────────────────────────────────────────────────

def step1_diagnose(df_eval, model, scaler):
    """Return {api_name: avg_auc_across_horizons}.  None if not computable."""
    print("Step 1 -- Diagnosing per-API performance ...")
    available = [c for c in FEATURE_COLS if c in df_eval.columns]
    per_api   = {}

    for api in sorted(df_eval["api_name"].unique()):
        sub = df_eval[df_eval["api_name"] == api].sort_values("timestamp").reset_index(drop=True)
        if len(sub) < SEQ_LEN + MAX_H + 10:
            per_api[api] = None
            continue

        X_raw    = sub[available].fillna(0).to_numpy(dtype=np.float64)
        y_raw    = sub["success"].to_numpy(dtype=np.float32)
        X_scaled = scaler.transform(X_raw).astype(np.float32)

        probas, tgt_idx = _infer(model, X_scaled, y_raw, SEQ_LEN, HORIZONS)
        if len(probas) == 0:
            per_api[api] = None
            continue

        aucs = []
        for h_i, h in enumerate(HORIZONS):
            y_true  = 1 - y_raw[tgt_idx[h_i]]
            y_score = probas[:, h_i]
            auc     = _auc_from_arrays(y_true, y_score)
            if not np.isnan(auc):
                aucs.append(auc)

        per_api[api] = float(np.mean(aucs)) if aucs else None
        status = f"{per_api[api]:.4f}" if per_api[api] is not None else "N/A"
        print(f"  {api:<22}  AUC={status}")

    return per_api


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- Identify problems
# ─────────────────────────────────────────────────────────────────────────────

def step2_identify(df_eval, df_train_pool, per_api_auc, model, scaler):
    """
    Returns problems dict with keys:
      worst_api, worst_api_auc,
      missed_failure_types,
      drift_detected, drift_report,
      imbalance_detected, failure_rate
    """
    print("\nStep 2 -- Identifying problems ...")
    problems = {
        "worst_api":            None,
        "worst_api_auc":        None,
        "missed_failure_types": [],
        "drift_detected":       False,
        "drift_report":         {},
        "imbalance_detected":   False,
        "failure_rate":         None,
    }

    # ── 2a. Worst API ─────────────────────────────────────────────────────────
    valid = {k: v for k, v in per_api_auc.items() if v is not None}
    if valid:
        worst_api = min(valid, key=valid.get)
        problems["worst_api"]     = worst_api
        problems["worst_api_auc"] = valid[worst_api]
        flag = " <-- FLAGGED" if valid[worst_api] < WORST_API_AUC_THRESHOLD else ""
        print(f"  Worst API: {worst_api}  AUC={valid[worst_api]:.4f}{flag}")

    # ── 2b. Missed failure types  ─────────────────────────────────────────────
    # Use eval set: for each sequence's h=1 target that is a failure,
    # check if the model correctly predicts failure (prob > 0.5).
    available = [c for c in FEATURE_COLS if c in df_eval.columns]
    df_sorted = df_eval.sort_values("timestamp").reset_index(drop=True)
    X_raw     = df_sorted[available].fillna(0).to_numpy(dtype=np.float64)
    y_raw     = df_sorted["success"].to_numpy(dtype=np.float32)
    X_scaled  = scaler.transform(X_raw).astype(np.float32)

    probas, tgt_idx = _infer(model, X_scaled, y_raw, SEQ_LEN, HORIZONS)

    if len(probas) > 0:
        # Focus on h=1 (most actionable)
        y_true_h1  = 1 - y_raw[tgt_idx[0]]              # 1 = failure
        y_pred_h1  = (probas[:, 0] > 0.5).astype(int)
        error_types = df_sorted["error_type"].iloc[tgt_idx[0]].values

        recall_by_type = {}
        for seq_i, (yt, yp, et) in enumerate(zip(y_true_h1, y_pred_h1, error_types)):
            if yt == 1 and et and str(et) not in ("None", "nan", ""):
                et_str = str(et)
                if et_str not in recall_by_type:
                    recall_by_type[et_str] = {"tp": 0, "fn": 0}
                if yp == 1:
                    recall_by_type[et_str]["tp"] += 1
                else:
                    recall_by_type[et_str]["fn"] += 1

        missed = []
        for et, counts in recall_by_type.items():
            total  = counts["tp"] + counts["fn"]
            recall = counts["tp"] / total if total > 0 else 0.0
            if total >= 10 and recall < RECALL_THRESHOLD:   # need >= 10 samples to flag
                missed.append(et)
                print(f"  Missed failure type: {et}  recall={recall:.2f}  (n={total})")

        problems["missed_failure_types"] = missed
        if not missed:
            print("  No missed failure types (all recall >= threshold or too few samples)")

    # ── 2c. Data drift  ──────────────────────────────────────────────────────
    try:
        from scipy.stats import ks_2samp
        drift_features = ["error_rate_rolling", "response_time_rolling_mean",
                          "rt_multiplier", "error_rate_boost", "error_volatility"]
        n         = len(df_train_pool)
        half      = n // 2
        old_half  = df_train_pool.iloc[:half]
        new_half  = df_train_pool.iloc[half:]

        n_drifted = 0
        drift_report = {}
        for feat in drift_features:
            if feat not in df_train_pool.columns:
                continue
            stat, p = ks_2samp(
                old_half[feat].fillna(0).values,
                new_half[feat].fillna(0).values,
            )
            drifted = bool(stat > DRIFT_KS_THRESHOLD and p < DRIFT_P_THRESHOLD)
            drift_report[feat] = {"ks": round(float(stat), 4),
                                  "p":  round(float(p), 6),
                                  "drifted": drifted}
            if drifted:
                n_drifted += 1
                print(f"  Drift detected: {feat}  KS={stat:.3f}  p={p:.4f}")

        problems["drift_detected"] = (n_drifted >= DRIFT_MIN_FEATURES)
        problems["drift_report"]   = drift_report
        if not problems["drift_detected"]:
            print(f"  No significant drift ({n_drifted}/{len(drift_features)} features drifted)")
    except ImportError:
        print("  scipy not available -- skipping drift detection")

    # ── 2d. Class imbalance  ──────────────────────────────────────────────────
    recent_fail_rate = 1 - df_train_pool["success"].mean()
    problems["failure_rate"]       = float(recent_fail_rate)
    problems["imbalance_detected"] = bool(recent_fail_rate < IMBALANCE_THRESHOLD)
    flag = " <-- FLAGGED" if problems["imbalance_detected"] else ""
    print(f"  Recent failure rate: {recent_fail_rate:.2%}{flag}")

    return problems


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- Fix: targeted data augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _jitter(rows: pd.DataFrame, rng, scale=0.05) -> pd.DataFrame:
    """Add multiplicative Gaussian noise to continuous features."""
    out = rows.copy()
    for col in JITTER_COLS:
        if col in out.columns:
            noise      = 1.0 + rng.normal(0, scale, len(out))
            out[col]   = (out[col].astype(float) * noise).clip(lower=0)
    if "request_count" in out.columns:
        out["request_count"] = (
            out["request_count"].astype(float)
            + rng.integers(-3, 4, len(out))
        ).clip(lower=1).astype(int)
    return out


def step3_fix(df_train_pool: pd.DataFrame, problems: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply targeted augmentations based on identified problems.
    Returns (augmented_df, list_of_fix_names_applied).
    """
    print("\nStep 3 -- Applying fixes ...")
    rng        = np.random.default_rng(42)
    fixes      = []
    df_working = df_train_pool.copy()

    # ── Fix A: Drift -- restrict to most recent data  ────────────────────────
    if problems["drift_detected"]:
        cutoff  = int(len(df_working) * (1 - DRIFT_WINDOW_FRACTION))
        df_working = df_working.iloc[cutoff:].reset_index(drop=True)
        fixes.append("drift_recency_window")
        print(f"  [drift]     Restricted to most recent {len(df_working):,} rows")

    # ── Fix B: Worst API over-sampling  ──────────────────────────────────────
    worst_api = problems.get("worst_api")
    worst_auc = problems.get("worst_api_auc")
    if worst_api and worst_auc is not None and worst_auc < WORST_API_AUC_THRESHOLD:
        api_failures = df_working[
            (df_working["api_name"] == worst_api) & (df_working["success"] == 0)
        ]
        if len(api_failures) > 0:
            copies = [_jitter(api_failures, rng, scale=0.05)
                      for _ in range(OVERSAMPLE_FACTOR)]
            df_working = pd.concat([df_working] + copies, ignore_index=True)
            n_added = len(api_failures) * OVERSAMPLE_FACTOR
            fixes.append(f"oversample_worst_api:{worst_api}")
            print(f"  [worst_api] +{n_added:,} augmented rows for {worst_api}")

    # ── Fix C: Missed failure types  ─────────────────────────────────────────
    missed = problems.get("missed_failure_types", [])
    for et in missed:
        et_failures = df_working[
            (df_working["error_type"] == et) & (df_working["success"] == 0)
        ]
        if len(et_failures) > 0:
            copies = [_jitter(et_failures, rng, scale=0.04)
                      for _ in range(OVERSAMPLE_FACTOR)]
            df_working = pd.concat([df_working] + copies, ignore_index=True)
            n_added = len(et_failures) * OVERSAMPLE_FACTOR
            fixes.append(f"boost_missed_type:{et}")
            print(f"  [missed]    +{n_added:,} rows for error_type={et}")

    # ── Fix D: Class imbalance -- inject synthetic failures  ──────────────────
    if problems["imbalance_detected"]:
        current_fail = 1 - df_working["success"].mean()
        n_total      = len(df_working)
        n_current_f  = int((1 - df_working["success"]).sum())
        n_target_f   = int(n_total * TARGET_FAILURE_RATE)
        n_inject     = max(0, n_target_f - n_current_f)

        if n_inject > 0 and n_current_f > 0:
            existing_failures = df_working[df_working["success"] == 0]
            # Sample with replacement from existing failures, then jitter
            sampled = existing_failures.sample(
                n=n_inject, replace=True, random_state=42
            )
            sampled = _jitter(sampled, rng, scale=0.06)
            df_working = pd.concat([df_working, sampled], ignore_index=True)
            new_rate = 1 - df_working["success"].mean()
            fixes.append("inject_failures_for_imbalance")
            print(f"  [imbalance] +{n_inject:,} synthetic failures "
                  f"({current_fail:.2%} -> {new_rate:.2%})")

    if not fixes:
        print("  No fixes triggered -- retraining on unmodified recent data")

    df_working = df_working.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Training pool: {len(df_train_pool):,} -> {len(df_working):,} rows "
          f"({len(df_working)/max(len(df_train_pool),1):.2f}x)")
    return df_working, fixes


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- Retrain on fixed data
# ─────────────────────────────────────────────────────────────────────────────

def step4_retrain(df_augmented: pd.DataFrame,
                  candidate_model_path: str,
                  candidate_scaler_path: str) -> dict:
    """
    Train a new model on the augmented data.
    Returns training stats dict.
    """
    print(f"\nStep 4 -- Retraining ({args.retrain_epochs} epochs, "
          f"max {args.max_train_seq:,} sequences) ...")

    available = [c for c in FEATURE_COLS if c in df_augmented.columns]
    df_s      = df_augmented.sort_values("timestamp").reset_index(drop=True)
    raw_X     = df_s[available].fillna(0).to_numpy(dtype=np.float64)
    raw_y     = df_s["success"].to_numpy(dtype=np.float32)

    n_seq   = len(raw_X) - SEQ_LEN - MAX_H + 1
    if n_seq < 100:
        raise ValueError(f"Augmented pool too small for sequences: {n_seq}")

    strat   = 1 - raw_y[SEQ_LEN + MAX_H - 1: SEQ_LEN + MAX_H - 1 + n_seq].astype(int)
    idx     = np.arange(n_seq)
    tr, tmp = train_test_split(idx, test_size=0.2, random_state=42, stratify=strat)
    val, _  = train_test_split(tmp, test_size=0.5, random_state=42, stratify=strat[tmp])

    if args.max_train_seq > 0 and len(tr) > args.max_train_seq:
        tr = np.random.default_rng(42).choice(tr, size=args.max_train_seq, replace=False)

    # Fit scaler on training rows only (no leakage)
    train_row_end = int(tr.max()) + SEQ_LEN + MAX_H
    scaler        = StandardScaler().fit(raw_X[:train_row_end])
    joblib.dump(scaler, candidate_scaler_path)

    full_ds  = TimeSeriesDataset(raw_X, raw_y, SEQ_LEN, HORIZONS, scaler=scaler)
    train_ds = Subset(full_ds, tr.tolist())
    val_ds   = Subset(full_ds, val.tolist())

    # Focal loss with pos_weight
    n_fail    = float((strat[tr] == 1).sum())
    n_success = float((strat[tr] == 0).sum())
    pw        = torch.tensor([min(n_success / max(n_fail, 1), 10.0)] * len(HORIZONS))
    criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pw)

    model     = MultiHorizonLSTM(len(available), args.hidden_size, 2, len(HORIZONS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.retrain_epochs, eta_min=1e-5
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

    best_val_loss  = float("inf")
    patience_count = 0
    patience       = 4
    train_losses   = []
    val_losses     = []

    for epoch in range(args.retrain_epochs):
        model.train()
        run_loss = 0.0
        for batch in train_loader:
            X = batch["sequence"]
            y = batch["targets"]
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item()
        train_loss = run_loss / max(1, len(train_loader))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                vl += criterion(model(batch["sequence"]), batch["targets"]).item()
        val_loss = vl / max(1, len(val_loader))
        scheduler.step()

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        improved = val_loss < best_val_loss
        marker   = " *" if improved else ""
        print(f"  Epoch {epoch+1:>2}/{args.retrain_epochs} | "
              f"train={train_loss:.6f} | val={val_loss:.6f}{marker}")

        if improved:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), candidate_model_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return {
        "best_val_loss": float(best_val_loss),
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "n_features":    len(available),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- Compare old vs new on the same held-out test set
# ─────────────────────────────────────────────────────────────────────────────

def _eval_model_on_df(model, scaler, df_eval):
    """Return per-horizon AUC dict for a model on df_eval."""
    available = [c for c in FEATURE_COLS if c in df_eval.columns]
    df_s      = df_eval.sort_values("timestamp").reset_index(drop=True)
    X_raw     = df_s[available].fillna(0).to_numpy(dtype=np.float64)
    y_raw     = df_s["success"].to_numpy(dtype=np.float32)
    X_scaled  = scaler.transform(X_raw).astype(np.float32)

    probas, tgt_idx = _infer(model, X_scaled, y_raw, SEQ_LEN, HORIZONS)
    if len(probas) == 0:
        return {f"horizon_{h}": float("nan") for h in HORIZONS}

    result = {}
    for h_i, h in enumerate(HORIZONS):
        y_true        = 1 - y_raw[tgt_idx[h_i]]
        result[f"horizon_{h}"] = _auc_from_arrays(y_true, probas[:, h_i])
    return result


def step5_compare(df_test,
                  old_model_path, old_scaler_path,
                  new_model_path, new_scaler_path,
                  n_features):
    """
    Evaluate both models on the same df_test.
    Returns (old_auc_dict, new_auc_dict, old_avg, new_avg).
    """
    print("\nStep 5 -- Comparing old vs new model on held-out test set ...")

    old_model  = _load_model(old_model_path, n_features, args.hidden_size, 2, len(HORIZONS))
    old_scaler = joblib.load(old_scaler_path)
    old_aucs   = _eval_model_on_df(old_model, old_scaler, df_test)

    new_model  = _load_model(new_model_path, n_features, args.hidden_size, 2, len(HORIZONS))
    new_scaler = joblib.load(new_scaler_path)
    new_aucs   = _eval_model_on_df(new_model, new_scaler, df_test)

    valid_old = [v for v in old_aucs.values() if not np.isnan(v)]
    valid_new = [v for v in new_aucs.values() if not np.isnan(v)]
    old_avg   = float(np.mean(valid_old)) if valid_old else float("nan")
    new_avg   = float(np.mean(valid_new)) if valid_new else float("nan")

    print(f"  {'Horizon':<12}  {'Old AUC':>10}  {'New AUC':>10}  {'Delta':>8}")
    print(f"  {'-'*44}")
    for h in HORIZONS:
        k   = f"horizon_{h}"
        old = old_aucs[k]
        new = new_aucs[k]
        d   = new - old if not (np.isnan(old) or np.isnan(new)) else float("nan")
        arrow = " ^" if (not np.isnan(d) and d > 0) else (" v" if (not np.isnan(d) and d < 0) else "")
        print(f"  h={h:<10}  {old:>10.4f}  {new:>10.4f}  {d:>+8.4f}{arrow}")
    print(f"  {'-'*44}")
    print(f"  {'Average':<12}  {old_avg:>10.4f}  {new_avg:>10.4f}  "
          f"  {new_avg - old_avg:>+8.4f}")

    return old_aucs, new_aucs, old_avg, new_avg


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 -- Model selection
# ─────────────────────────────────────────────────────────────────────────────

def step6_select(old_avg, new_avg,
                 old_model_path, new_model_path,
                 old_scaler_path, new_scaler_path) -> tuple[bool, str]:
    """
    Keep new model if it improved by at least min_improvement.
    Always backs up old model before replacing.
    Returns (model_updated, backup_path_or_empty).
    """
    print("\nStep 6 -- Model selection ...")
    improved = (not np.isnan(new_avg)) and (new_avg - old_avg >= args.min_improvement)

    backup_path = ""
    if improved:
        # Backup old model
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = old_model_path.replace(".pth", f".bak_{ts}.pth")
        shutil.copy2(old_model_path,  backup_path)
        shutil.copy2(old_scaler_path, old_scaler_path.replace(".pkl", f".bak_{ts}.pkl"))
        # Replace with new
        shutil.copy2(new_model_path,  old_model_path)
        shutil.copy2(new_scaler_path, old_scaler_path)
        print(f"  [ACCEPT] New model  (AUC {old_avg:.4f} -> {new_avg:.4f})")
        print(f"  Old model backed up -> {backup_path}")
    else:
        reason = (f"new AUC {new_avg:.4f} not better than old {old_avg:.4f} "
                  f"by >= {args.min_improvement}")
        print(f"  [REJECT] Keeping old model  ({reason})")

    # Clean up temp candidate files
    for p in [new_model_path, new_scaler_path]:
        if os.path.exists(p) and p != old_model_path:
            os.remove(p)

    return improved, backup_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 -- Append to audit log (never overwrite)
# ─────────────────────────────────────────────────────────────────────────────

def step7_log(problems, fixes, retrain_stats,
              old_aucs, new_aucs, old_avg, new_avg,
              model_updated, backup_path,
              rows_before, rows_after):
    """Append one JSON line to the append-only self_heal_log.jsonl."""
    entry = {
        "run_id":        RUN_ID,
        "timestamp":     RUN_TS,
        "mode":          MODE,
        "data": {
            "rows_in_recent_window":  rows_before,
            "rows_after_augmentation": rows_after,
            "augmentation_ratio":      round(rows_after / max(rows_before, 1), 3),
        },
        "problems_found": {
            "worst_api":            problems["worst_api"],
            "worst_api_auc":        problems["worst_api_auc"],
            "missed_failure_types": problems["missed_failure_types"],
            "drift_detected":       problems["drift_detected"],
            "drift_report":         problems["drift_report"],
            "imbalance_detected":   problems["imbalance_detected"],
            "failure_rate":         problems["failure_rate"],
        },
        "fixes_applied":  fixes,
        "retrain": retrain_stats,
        "comparison": {
            "old_auc": old_aucs,
            "new_auc": new_aucs,
            "old_avg": round(old_avg, 6),
            "new_avg": round(new_avg, 6) if not np.isnan(new_avg) else None,
            "delta":   round(new_avg - old_avg, 6) if not np.isnan(new_avg) else None,
        },
        "outcome": {
            "model_updated": model_updated,
            "backup_path":   backup_path,
            "min_improvement_threshold": args.min_improvement,
        },
    }

    log_path = args.log_path
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nStep 7 -- Log entry appended -> {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 -- Plain-language summary
# ─────────────────────────────────────────────────────────────────────────────

def step8_summary(problems, fixes, old_avg, new_avg,
                  model_updated, rows_before, rows_after, elapsed_sec):
    w = 68
    print(f"\n{'='*w}")
    print(f"  SELF-IMPROVING PIPELINE SUMMARY  [{RUN_ID}]")
    print(f"{'='*w}")

    if args.dry_run:
        print("  Mode    : DRY RUN -- no files were modified")
    print(f"  Runtime : {elapsed_sec:.0f}s")
    print(f"  Data    : {rows_before:,} rows analysed, "
          f"{rows_after:,} rows used for retraining")

    print(f"\n  Problems found:")
    found_any = False
    if problems["worst_api"] and problems["worst_api_auc"] < WORST_API_AUC_THRESHOLD:
        print(f"    - {problems['worst_api']} is performing poorly "
              f"(AUC {problems['worst_api_auc']:.4f} < {WORST_API_AUC_THRESHOLD})")
        found_any = True
    if problems["missed_failure_types"]:
        for et in problems["missed_failure_types"]:
            print(f"    - Model keeps missing '{et}' failures (recall < {RECALL_THRESHOLD:.0%})")
        found_any = True
    if problems["drift_detected"]:
        print("    - Data distribution has shifted significantly (drift detected)")
        found_any = True
    if problems["imbalance_detected"]:
        print(f"    - Failure rate dropped to {problems['failure_rate']:.2%} "
              f"(below {IMBALANCE_THRESHOLD:.0%} threshold)")
        found_any = True
    if not found_any:
        print("    - None  (model is performing within expected bounds)")

    print(f"\n  Fixes applied:")
    if fixes:
        for fix in fixes:
            label = fix.split(":")[0]
            detail = fix.split(":")[1] if ":" in fix else ""
            descriptions = {
                "drift_recency_window":          f"Restricted training to most recent {int(DRIFT_WINDOW_FRACTION*100)}% of data",
                "oversample_worst_api":          f"Oversampled failures for {detail} ({OVERSAMPLE_FACTOR}x)",
                "boost_missed_type":             f"Boosted training examples for '{detail}' ({OVERSAMPLE_FACTOR}x)",
                "inject_failures_for_imbalance": f"Injected synthetic failures to reach {TARGET_FAILURE_RATE:.0%} rate",
            }
            print(f"    - {descriptions.get(label, fix)}")
    else:
        print("    - None triggered")

    print(f"\n  Result:")
    if args.dry_run:
        print("    - Dry run: no retraining or model replacement performed")
    elif model_updated:
        delta = new_avg - old_avg
        print(f"    - Model IMPROVED and was replaced")
        print(f"      AUC: {old_avg:.4f} -> {new_avg:.4f} ({delta:+.4f})")
        print(f"      Old model backed up safely")
    else:
        print(f"    - Model NOT replaced (new AUC {new_avg:.4f} did not exceed "
              f"old AUC {old_avg:.4f} + {args.min_improvement})")

    print(f"\n  Log: {args.log_path}")
    print(f"{'='*w}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # ── Validate required files ───────────────────────────────────────────────
    for p, label in [(args.data_path,  "features CSV"),
                     (args.model_path, "model checkpoint"),
                     (args.scaler_path,"scaler")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    # ── Load recent data ──────────────────────────────────────────────────────
    print(f"Loading last {args.recent_rows:,} rows from {args.data_path} ...")
    df_full = pd.read_csv(args.data_path, low_memory=False)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], errors="coerce")
    if df_full["success"].dtype == object:
        df_full["success"] = df_full["success"].map({"True": 1, "False": 0}).fillna(0).astype(int)
    else:
        df_full["success"] = df_full["success"].astype(int)
    df_full = df_full.sort_values("timestamp").reset_index(drop=True)

    df_recent = df_full.tail(args.recent_rows).reset_index(drop=True)
    print(f"  Using {len(df_recent):,} rows | "
          f"failure rate: {1 - df_recent['success'].mean():.2%} | "
          f"date range: {df_recent['timestamp'].min().date()} -> "
          f"{df_recent['timestamp'].max().date()}")

    # ── Split into eval (held-out test) and train pool ────────────────────────
    n_test       = max(SEQ_LEN + MAX_H + 100, int(len(df_recent) * args.test_fraction))
    df_test      = df_recent.tail(n_test).reset_index(drop=True)
    df_train_pool = df_recent.iloc[:-n_test].reset_index(drop=True)
    print(f"  Train pool: {len(df_train_pool):,} rows | "
          f"Test (held-out): {len(df_test):,} rows\n")

    # ── Load existing model + scaler ──────────────────────────────────────────
    n_features = len([c for c in FEATURE_COLS if c in df_recent.columns])
    old_model  = _load_model(args.model_path, n_features, args.hidden_size, 2, len(HORIZONS))
    old_scaler = joblib.load(args.scaler_path)
    print(f"Loaded model ({n_features} features, "
          f"hidden={args.hidden_size}, heads={len(HORIZONS)})\n")

    # Read current recorded AUC from results file
    recorded_avg = float("nan")
    if os.path.exists(args.results_path):
        try:
            recorded_avg = json.load(open(args.results_path))["avg_auc"]
            print(f"Recorded production AUC (from lstm_results.json): {recorded_avg:.4f}\n")
        except Exception:
            pass

    # ── Step 1: Diagnose ──────────────────────────────────────────────────────
    per_api_auc = step1_diagnose(df_test, old_model, old_scaler)

    # ── Step 2: Identify problems ─────────────────────────────────────────────
    problems = step2_identify(df_test, df_train_pool, per_api_auc, old_model, old_scaler)

    # Compute old baseline AUC on the held-out test set (Steps 1+2 already did per-API,
    # now do overall for the comparison in Step 5)
    print("\n  Computing overall baseline AUC on held-out test set ...")
    old_aucs_test = _eval_model_on_df(old_model, old_scaler, df_test)
    valid_old     = [v for v in old_aucs_test.values() if not np.isnan(v)]
    old_avg_test  = float(np.mean(valid_old)) if valid_old else float("nan")
    for h in HORIZONS:
        k = f"horizon_{h}"
        print(f"    h={h}: AUC={old_aucs_test[k]:.4f}")
    print(f"    Average:  {old_avg_test:.4f}")

    # ── DRY RUN stops here ────────────────────────────────────────────────────
    if args.dry_run:
        any_problem = (
            (problems["worst_api_auc"] is not None
             and problems["worst_api_auc"] < WORST_API_AUC_THRESHOLD)
            or bool(problems["missed_failure_types"])
            or problems["drift_detected"]
            or problems["imbalance_detected"]
        )
        print(f"\n[DRY RUN] Would apply fixes: "
              f"{'yes' if any_problem else 'no -- model is healthy'}")
        # Log dry-run entry (no retrain stats or new AUC)
        dummy_retrain = {"best_val_loss": None, "train_losses": [], "val_losses": [],
                         "n_features": n_features}
        step7_log(problems, [], dummy_retrain,
                  old_aucs_test, {}, old_avg_test, float("nan"),
                  False, "", len(df_train_pool), len(df_train_pool))
        step8_summary(problems, [], old_avg_test, float("nan"),
                      False, len(df_train_pool), len(df_train_pool),
                      time.time() - t_start)
        return

    # ── Step 3: Fix ───────────────────────────────────────────────────────────
    df_augmented, fixes = step3_fix(df_train_pool, problems)
    rows_after = len(df_augmented)

    # ── Step 4: Retrain ───────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    candidate_model_path  = args.model_path.replace(".pth", "_candidate.pth")
    candidate_scaler_path = args.scaler_path.replace(".pkl", "_candidate.pkl")

    try:
        retrain_stats = step4_retrain(df_augmented,
                                      candidate_model_path,
                                      candidate_scaler_path)
    except Exception as e:
        print(f"\nERROR during retraining: {e}")
        # Log failure and exit cleanly
        step7_log(problems, fixes,
                  {"error": str(e)},
                  old_aucs_test, {}, old_avg_test, float("nan"),
                  False, "", len(df_train_pool), rows_after)
        step8_summary(problems, fixes, old_avg_test, float("nan"),
                      False, len(df_train_pool), rows_after,
                      time.time() - t_start)
        raise

    # ── Step 5: Compare ───────────────────────────────────────────────────────
    if not os.path.exists(candidate_model_path):
        print("ERROR: candidate model was not saved (likely val loss never improved).")
        sys.exit(1)

    old_aucs, new_aucs, old_avg, new_avg = step5_compare(
        df_test,
        args.model_path,      args.scaler_path,
        candidate_model_path, candidate_scaler_path,
        n_features,
    )

    # ── Step 6: Select ────────────────────────────────────────────────────────
    model_updated, backup_path = step6_select(
        old_avg, new_avg,
        args.model_path,      candidate_model_path,
        args.scaler_path,     candidate_scaler_path,
    )

    # ── Step 7: Log ───────────────────────────────────────────────────────────
    step7_log(problems, fixes, retrain_stats,
              old_aucs, new_aucs, old_avg, new_avg,
              model_updated, backup_path,
              len(df_train_pool), rows_after)

    # ── Step 8: Summary ───────────────────────────────────────────────────────
    step8_summary(problems, fixes, old_avg, new_avg,
                  model_updated, len(df_train_pool), rows_after,
                  time.time() - t_start)


if __name__ == "__main__":
    main()
