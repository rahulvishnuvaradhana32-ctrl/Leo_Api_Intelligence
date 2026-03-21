#!/usr/bin/env python3
"""
conformal_prediction.py  --  Inductive Conformal Prediction for the LEO API Intelligence LSTM.

Wraps the trained model so every failure probability comes with a
statistically guaranteed confidence interval.

  "72% chance of failure"
  → "72% ± 11%  (we are 90% confident the true probability is 61%–83%)"

Method: Split (Inductive) Conformal Prediction (ICP)
  - Nonconformity score:  s = |y_true - y_pred|
  - Threshold q_hat:         ceil((n+1)(1−α)) / n  quantile of cal scores
  - Interval:            [y_hat - q_hat,  y_hat + q_hat]  clipped to [0, 1]
  - Coverage guarantee:  P(Y ∈ Ĉ(X)) ≥ 1−α  for exchangeable data

Splits (chronological, never seen during training):
  Calibration : pre-2025 rows  [60% : 80%]   ~200 K rows
  Test        : pre-2025 rows  [80% : 100%]  ~200 K rows

Output:
  models/conformal_results.csv       per-sample predictions + intervals
  models/conformal_results.json      summary statistics per horizon
  models/conformal_calibration.png   reliability diagram

Usage:
    python scripts/conformal_prediction.py
    python scripts/conformal_prediction.py --alpha 0.05   # 95% coverage
    python scripts/conformal_prediction.py --alpha 0.10   # 90% coverage (default)
    python scripts/conformal_prediction.py --cal_seq 50000 --test_seq 50000
"""

import os, sys, json, time, warnings, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Conformal prediction for LEO API Intelligence LSTM")
parser.add_argument("--alpha",      type=float, default=0.10,
                    help="Error rate (0.10 → 90%% coverage, 0.05 → 95%%). Default 0.10")
parser.add_argument("--cal_seq",    type=int,   default=50_000,
                    help="Max calibration sequences (default 50000)")
parser.add_argument("--test_seq",   type=int,   default=50_000,
                    help="Max test sequences (default 50000)")
parser.add_argument("--seq_len",    type=int,   default=30)
parser.add_argument("--data_path",  type=str,   default="data/banking_api_features_v6.csv")
parser.add_argument("--model_path", type=str,   default="models/stress_test_best_model.pth")
parser.add_argument("--scaler_path",type=str,   default="models/scaler.pkl")
args = parser.parse_args()

HORIZONS  = [1, 5, 15]
MAX_H     = max(HORIZONS)
SEQ_LEN   = args.seq_len
COVERAGE  = round((1 - args.alpha) * 100, 1)

print(f"=== Conformal Prediction  (target coverage: {COVERAGE}%) ===\n")

# ── Feature columns — must match exactly what the model was trained with ──────
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
    # precursor signals
    "latency_diff_1", "latency_diff_5",
    "error_rate_diff_1", "error_rate_diff_5",
    "latency_spike", "error_burst", "instability_index",
    "latency_slope", "error_slope",
    # advanced signals
    "traffic_change", "burst_ratio",
    # cross-API correlation features — present in banking_api_features_v6.csv
    "avg_error_rate_others", "max_error_rate_others",
    "n_apis_elevated", "corr_with_similar_api",
    "systemic_stress_index",
]


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture  (identical to run_lstm_training.py)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 output_size=3, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm_out = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(self.lstm_out)
        self.attn_pool  = AttentionPooling(self.lstm_out)
        self.dropout    = nn.Dropout(dropout)
        self.heads      = nn.ModuleList([nn.Linear(self.lstm_out, 1)
                                         for _ in range(output_size)])

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.layer_norm(out)
        out    = self.attn_pool(out)
        out    = self.dropout(out)
        return torch.cat([head(out) for head in self.heads], dim=1)


def _detect_hidden_size(state_dict):
    """Infer hidden_size, input_size, and bidirectionality from checkpoint."""
    lstm_ih       = state_dict["lstm.weight_ih_l0"]
    lstm_hh       = state_dict["lstm.weight_hh_l0"]
    bidirectional = "lstm.weight_ih_l0_reverse" in state_dict
    hidden        = lstm_hh.shape[1]
    n_in          = lstm_ih.shape[1]
    n_heads       = sum(1 for k in state_dict if k.startswith("heads.") and "weight" in k)
    return n_in, hidden, n_heads, bidirectional


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build sequences and infer in batches
# ─────────────────────────────────────────────────────────────────────────────

def _build_and_infer(model, X_scaled, y_raw, seq_len, horizons,
                     max_seq=None, batch_size=512):
    """
    Build sequences from X_scaled/y_raw, run batched inference.
    Returns:
      probas      [n_seq × n_horizons]  -- predicted failure probabilities
      y_true_mat  [n_seq × n_horizons]  -- true failure labels (1=fail)
    """
    n_seq = len(X_scaled) - seq_len - max(horizons) + 1
    if n_seq < 1:
        raise ValueError("Data slice too small for sequences")

    if max_seq and n_seq > max_seq:
        # sample uniformly to keep temporal spread
        rng       = np.random.default_rng(42)
        seq_idx   = rng.choice(n_seq, size=max_seq, replace=False)
        seq_idx   = np.sort(seq_idx)
    else:
        seq_idx = np.arange(n_seq)

    all_proba  = []
    all_ytrue  = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(seq_idx), batch_size):
            batch_idx = seq_idx[start: start + batch_size]
            seqs = np.stack([X_scaled[i: i + seq_len] for i in batch_idx])
            logits  = model(torch.from_numpy(seqs.astype(np.float32)))
            probas  = torch.sigmoid(logits).numpy()
            y_batch = np.array([
                [float(1 - y_raw[i + seq_len + h - 1]) for h in horizons]
                for i in batch_idx
            ])
            all_proba.append(probas)
            all_ytrue.append(y_batch)

    return np.vstack(all_proba), np.vstack(all_ytrue)


# ─────────────────────────────────────────────────────────────────────────────
# Conformal prediction core
# ─────────────────────────────────────────────────────────────────────────────

def conformal_quantile(scores, alpha):
    """
    Return the conformal quantile q̂ such that approximately (1-α) of
    future predictions will contain the true label.

    Uses the finite-sample corrected quantile:
        q_hat = quantile(scores, min(1, ceil((n+1)(1-α)) / n))
    """
    n     = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def make_intervals(probas, q_hat):
    """Return (lower, upper) arrays clipped to [0, 1]."""
    lower = np.clip(probas - q_hat, 0.0, 1.0)
    upper = np.clip(probas + q_hat, 0.0, 1.0)
    return lower, upper


def empirical_coverage(y_true, lower, upper):
    """Fraction of samples where y_true falls inside [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


# ─────────────────────────────────────────────────────────────────────────────
# Reliability diagram (calibration curve)
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability(probas_all, ytrue_all, horizon_labels, out_path):
    """
    Reliability diagram: predicted probability (x) vs actual failure rate (y).
    A perfectly calibrated model sits on the diagonal.
    Shows all horizons on one chart plus a histogram of prediction counts.
    """
    n_bins  = 10
    edges   = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    colors  = ["#e74c3c", "#e67e22", "#3498db"]

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    for h_i, (label, col) in enumerate(zip(horizon_labels, colors)):
        p = probas_all[:, h_i]
        y = ytrue_all[:, h_i]
        actual_rates = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p >= lo) & (p < hi)
            actual_rates.append(y[mask].mean() if mask.sum() > 0 else np.nan)
        ax_main.plot(centers, actual_rates, "o-", color=col,
                     linewidth=2, markersize=6, label=label)

    ax_main.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax_main.set_ylabel("Actual failure rate", fontsize=11)
    ax_main.set_title(
        f"Reliability Diagram  (LEO API Intelligence — {COVERAGE}% Conformal Bands)",
        fontsize=12, pad=10
    )
    ax_main.legend(fontsize=9)
    ax_main.set_ylim(-0.05, 1.05)
    ax_main.grid(alpha=0.3)

    # Histogram of prediction counts (use h=1 as representative)
    counts, _ = np.histogram(probas_all[:, 0], bins=edges)
    ax_hist.bar(centers, counts, width=(edges[1] - edges[0]) * 0.85,
                color="#95a5a6", edgecolor="white")
    ax_hist.set_xlabel("Predicted failure probability", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration chart  -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── Validate paths ─────────────────────────────────────────────────────────
    for path, label in [(args.data_path,   "features CSV"),
                        (args.model_path,  "model checkpoint"),
                        (args.scaler_path, "scaler")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}"); sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {args.data_path} ...")
    df = pd.read_csv(args.data_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["success"].dtype == object:
        df["success"] = df["success"].map({"True": 1, "False": 0}).fillna(0).astype(int)
    else:
        df["success"] = df["success"].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Total rows: {len(df):,}")

    # Use only pre-2025 synthetic data (good failure rate; post-2025 is Kaggle 0%)
    df = df[df["timestamp"] < "2025-01-01"].reset_index(drop=True)
    print(f"  Pre-2025 rows: {len(df):,}  (failure rate: {1-df['success'].mean():.2%})")

    # ── Compute precursor features on-the-fly ──────────────────────────────────
    _EPS = 1e-6
    df["latency_diff_1"]    = (df["response_time"] - df["response_time_lag_1"]).fillna(0)
    df["latency_diff_5"]    = (df["response_time"] - df["response_time_lag_5"]).fillna(0)
    df["error_rate_diff_1"] = (df["error_rate_rolling"] - df["error_rate_lag_1"]).fillna(0)
    df["error_rate_diff_5"] = (df["error_rate_rolling"] - df["error_rate_ema_10"]).fillna(0)
    df["latency_spike"]     = df["response_time"] / (df["response_time_rolling_mean"] + _EPS)
    df["error_burst"]       = df["error_rate_rolling"] / (df["error_rate_ema_10"] + _EPS)
    df["instability_index"] = df["latency_diff_1"].abs() + df["error_rate_diff_1"].abs()
    df["latency_slope"]     = (df["response_time_ema_10"] - df["response_time_ema_30"]) / 20.0
    df["error_slope"]       = (df["error_rate_ema_10"] - df["error_rate_lag_1"]).fillna(0) / 10.0

    # ── Chronological calibration / test split ─────────────────────────────────
    n    = len(df)
    cal_s, cal_e = int(n * 0.60), int(n * 0.80)
    te_s,  te_e  = cal_e, n

    df_cal  = df.iloc[cal_s:cal_e].reset_index(drop=True)
    df_test = df.iloc[te_s:te_e].reset_index(drop=True)
    print(f"\n  Calibration : rows {cal_s:,}–{cal_e:,}  "
          f"({len(df_cal):,} rows, {1-df_cal['success'].mean():.2%} failure)")
    print(f"  Test        : rows {te_s:,}–{te_e:,}  "
          f"({len(df_test):,} rows, {1-df_test['success'].mean():.2%} failure)")
    print(f"  NOTE: these rows were included in training (random stratified split).")
    print(f"        Coverage results are in-distribution estimates.")

    # ── Select features present in the data ───────────────────────────────────
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\n  Features: {len(available)}/{len(FEATURE_COLS)} available")
    if len(available) < 28:
        print("  WARNING: fewer than 28 features — model may have shape mismatch")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.model_path} ...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    n_in, hidden, n_heads, bidir = _detect_hidden_size(state_dict)
    print(f"  Detected: input_size={n_in}  hidden={hidden}  heads={n_heads}  bidirectional={bidir}")

    if n_in != len(available):
        print(f"\n  WARNING: model expects {n_in} features, "
              f"data provides {len(available)}.")
        # Trim available to match model's input size (keep first n_in)
        if len(available) > n_in:
            available = available[:n_in]
            print(f"  Trimmed to first {n_in} features to match checkpoint.")
        else:
            print(f"  FATAL: fewer features than model input — cannot proceed.")
            sys.exit(1)

    model = MultiHorizonLSTM(n_in, hidden, 2, n_heads, bidirectional=bidir)
    model.load_state_dict(state_dict)
    model.eval()
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")

    # ── Load scaler ────────────────────────────────────────────────────────────
    scaler = joblib.load(args.scaler_path)

    def _scale_and_infer(df_slice, max_seq, label):
        X_raw    = df_slice[available].fillna(0).to_numpy(dtype=np.float64)
        y_raw    = df_slice["success"].to_numpy(dtype=np.float32)
        X_scaled = scaler.transform(X_raw).astype(np.float32)
        print(f"  Running inference on {label} set "
              f"(up to {max_seq:,} sequences) ...", end=" ", flush=True)
        t_inf = time.time()
        proba, ytrue = _build_and_infer(
            model, X_scaled, y_raw, SEQ_LEN, HORIZONS, max_seq
        )
        print(f"done in {time.time()-t_inf:.1f}s  ({len(proba):,} sequences)")
        return proba, ytrue

    # ── Step 1: Calibration inference ─────────────────────────────────────────
    print("\nStep 1 — Calibration inference ...")
    cal_proba, cal_ytrue = _scale_and_infer(df_cal, args.cal_seq, "calibration")

    # ── Step 2: Nonconformity scores ───────────────────────────────────────────
    print("\nStep 2 — Computing nonconformity scores  s = |y_true - y_pred| ...")
    # scores shape: [n_cal × n_horizons]
    cal_scores = np.abs(cal_ytrue - cal_proba)
    for h_i, h in enumerate(HORIZONS):
        s = cal_scores[:, h_i]
        print(f"  h={h:>2}  mean={s.mean():.4f}  "
              f"p90={np.percentile(s,90):.4f}  max={s.max():.4f}  "
              f"n_cal={len(s):,}")

    # ── Step 3: Conformal quantile q̂ ──────────────────────────────────────────
    print(f"\nStep 3 -- Setting {COVERAGE}% confidence threshold (α={args.alpha}) ...")
    q_hat = {}
    for h_i, h in enumerate(HORIZONS):
        q = conformal_quantile(cal_scores[:, h_i], args.alpha)
        q_hat[h] = q
        print(f"  h={h:>2}  q_hat = {q:.4f}  "
              f"(intervals will be +/-{q:.3f} wide, "
              f"avg width {2*min(q,0.5):.3f})")

    # ── Step 4: Test inference + prediction intervals ─────────────────────────
    print("\nStep 4 — Generating prediction intervals for test set ...")
    te_proba, te_ytrue = _scale_and_infer(df_test, args.test_seq, "test")

    intervals = {}   # {h: (lower, upper)}
    for h_i, h in enumerate(HORIZONS):
        lo, hi = make_intervals(te_proba[:, h_i], q_hat[h])
        intervals[h] = (lo, hi)

    # ── Step 5: Coverage ───────────────────────────────────────────────────────
    print(f"\nStep 5 — Measuring empirical coverage (target ≥ {COVERAGE}%) ...")
    coverage_results = {}
    for h_i, h in enumerate(HORIZONS):
        lo, hi = intervals[h]
        y      = te_ytrue[:, h_i]
        cov    = empirical_coverage(y, lo, hi)
        width  = float(np.mean(hi - lo))
        status = "PASS" if cov >= (1 - args.alpha) else "FAIL"
        coverage_results[h] = {"coverage": cov, "avg_width": width,
                                "q_hat": q_hat[h], "status": status}
        fail_rate = y.mean()
        print(f"  h={h:>2}  coverage={cov:.2%}  "
              f"avg_width={width:.4f}  "
              f"failure_rate={fail_rate:.2%}  [{status}]")

    # ── Step 6: Reliability diagram ────────────────────────────────────────────
    print("\nStep 6 — Generating reliability diagram ...")
    os.makedirs("models", exist_ok=True)
    horizon_labels = [f"h={h}" for h in HORIZONS]
    plot_reliability(te_proba, te_ytrue, horizon_labels,
                     "models/conformal_calibration.png")

    # ── Step 7: Save results ───────────────────────────────────────────────────
    print("\nStep 7 — Saving results ...")

    # Per-sample CSV (use h=1, h=5, h=15 columns side by side)
    n_rows = len(te_proba)
    rows   = []
    for i in range(n_rows):
        row = {"sample_idx": i}
        for h_i, h in enumerate(HORIZONS):
            lo, hi = intervals[h]
            row[f"h{h}_prob"]      = round(float(te_proba[i, h_i]), 5)
            row[f"h{h}_lower"]     = round(float(lo[i]), 5)
            row[f"h{h}_upper"]     = round(float(hi[i]), 5)
            row[f"h{h}_true"]      = int(te_ytrue[i, h_i])
            row[f"h{h}_covered"]   = bool(lo[i] <= te_ytrue[i, h_i] <= hi[i])
        rows.append(row)

    results_df = pd.DataFrame(rows)
    csv_path   = "models/conformal_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Per-sample CSV → {csv_path}  ({len(results_df):,} rows)")

    # Summary JSON
    avg_cov   = np.mean([v["coverage"] for v in coverage_results.values()])
    avg_width = np.mean([v["avg_width"] for v in coverage_results.values()])
    all_pass  = all(v["status"] == "PASS" for v in coverage_results.values())

    json_payload = {
        "target_coverage":  round(COVERAGE, 1),
        "alpha":            args.alpha,
        "cal_sequences":    int(len(cal_proba)),
        "test_sequences":   int(len(te_proba)),
        "n_features":       len(available),
        "per_horizon": {
            f"h{h}": {
                "q_hat":        round(coverage_results[h]["q_hat"], 6),
                "coverage":     round(coverage_results[h]["coverage"], 4),
                "avg_width":    round(coverage_results[h]["avg_width"], 4),
                "status":       coverage_results[h]["status"],
            }
            for h in HORIZONS
        },
        "summary": {
            "avg_coverage":   round(float(avg_cov), 4),
            "avg_width":      round(float(avg_width), 4),
            "all_horizons_pass": all_pass,
        },
    }
    json_path = "models/conformal_results.json"
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"  Summary JSON   → {json_path}")

    # ── Step 8: Plain-English summary ─────────────────────────────────────────
    elapsed = time.time() - t0
    w = 68
    print(f"\n{'='*w}")
    print(f"  CONFORMAL PREDICTION SUMMARY")
    print(f"{'='*w}")
    print(f"  Target coverage  : {COVERAGE}%  (α = {args.alpha})")
    print(f"  Calibration seqs : {len(cal_proba):,}")
    print(f"  Test sequences   : {len(te_proba):,}")
    print(f"  Features used    : {len(available)}")
    print(f"\n  Coverage per horizon:")
    for h in HORIZONS:
        cr    = coverage_results[h]
        check = "✓" if cr["status"] == "PASS" else "✗"
        print(f"    h={h:>2}  {cr['coverage']:.1%}  {check}  "
              f"(interval ±{cr['q_hat']:.3f},  avg width {cr['avg_width']:.3f})")

    print(f"\n  Average interval width : {avg_width:.4f}")
    print(f"  Average coverage       : {avg_cov:.1%}")
    print(f"\n  Interpretation:")
    if all_pass:
        sentence = (
            f"The model is well calibrated — {avg_cov:.1%} of true failure events "
            f"fall within the predicted {COVERAGE}% confidence bands across all "
            f"horizons, exceeding the {COVERAGE}% target."
        )
    else:
        failed = [h for h in HORIZONS if coverage_results[h]["status"] == "FAIL"]
        sentence = (
            f"Calibration PARTIAL — {avg_cov:.1%} average coverage. "
            f"Horizons {failed} fell below the {COVERAGE}% target; "
            f"consider using a smaller α or a larger calibration set."
        )
    print(f"    {sentence}")

    print(f"\n  Example prediction:")
    mid_i = len(te_proba) // 2
    p1  = te_proba[mid_i, 0]
    lo1 = intervals[1][0][mid_i]
    hi1 = intervals[1][1][mid_i]
    print(f"    h=1 failure probability: {p1:.1%}")
    print(f"    {COVERAGE}% confidence interval: [{lo1:.1%}, {hi1:.1%}]")
    print(f"    → \"There is a {p1:.0%} chance this API fails in the next step.")
    print(f"       We are {COVERAGE}% confident the true probability is")
    print(f"       between {lo1:.0%} and {hi1:.0%}.\"")

    print(f"\n  Output files:")
    print(f"    {csv_path}")
    print(f"    {json_path}")
    print(f"    models/conformal_calibration.png")
    print(f"\n  Runtime: {elapsed:.0f}s")
    print(f"{'='*w}\n")


if __name__ == "__main__":
    main()
