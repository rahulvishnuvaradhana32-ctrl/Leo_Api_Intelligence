#!/usr/bin/env python3
"""
Task 5 — Agent Simulation: Proactive vs Reactive API Switching

Loads the trained LSTM model, runs 1000 simulated financial transactions
across 5 real APIs, and compares proactive (prediction-driven) switching
against a reactive (fail-then-switch) baseline.

Usage:
    python scripts/agent_simulation.py
    python scripts/agent_simulation.py --n_transactions 2000
    python scripts/agent_simulation.py --seed 99
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import os
import pickle
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Constants ─────────────────────────────────────────────────────────────────
COST_PER_FAILURE   = 50.0    # dollars
TRANSACTIONS_PER_YEAR = 260_000  # ~1000 tx/day × 260 trading days

HIGH_RISK_THRESHOLD  = 0.65
LOW_RISK_THRESHOLD   = 0.35

LATENCY_NORMAL  = 0.12   # seconds — normal path
LATENCY_RETRY   = 0.28   # reduced timeout retry
LATENCY_SWITCH  = 0.45   # backup API switch overhead

SEQ_LEN = 30

FEATURE_COLS = [
    'response_time', 'status_code', 'request_count',
    'hour', 'day_of_week', 'is_weekend', 'is_market_hours',
    'response_time_rolling_mean', 'response_time_rolling_std',
    'error_rate_rolling',
    'response_time_variance', 'error_volatility',
    'response_time_lag_1', 'response_time_lag_5',
    'error_rate_lag_1',
    'response_time_ema_10', 'response_time_ema_30',
    'error_rate_ema_10',
    'hour_sin',
    'traffic_change', 'burst_ratio',
    'latency_diff_1', 'latency_diff_5',
    'error_rate_diff_1', 'error_rate_diff_5',
    'latency_spike', 'error_burst', 'instability_index',
    'latency_slope', 'error_slope',
]

# Backup mapping
BACKUP_API = {
    'stock_price_api':  'market_data_api',
    'crypto_api':       'market_data_api',
    'forex_api':        'market_data_api',
    'market_data_api':  'transaction_api',
    'transaction_api':  'market_data_api',
}


# ── Model definition (must match training) ────────────────────────────────────
class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_horizons, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm        = nn.LSTM(input_size, hidden_size, num_layers,
                                   batch_first=True, bidirectional=False,
                                   dropout=dropout if num_layers > 1 else 0.0)
        self.layer_norm  = nn.LayerNorm(hidden_size)
        self.dropout     = nn.Dropout(dropout)
        self.heads       = nn.ModuleList([nn.Linear(hidden_size, 1)
                                          for _ in range(n_horizons)])

    def forward(self, x):
        h0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out  = out[:, -1, :]
        out  = self.layer_norm(out)
        out  = self.dropout(out)
        return torch.cat([head(out) for head in self.heads], dim=1)


def _detect_dims(state_dict):
    lstm_hh = state_dict['lstm.weight_hh_l0']
    lstm_ih = state_dict['lstm.weight_ih_l0']
    hidden  = lstm_hh.shape[1]
    n_in    = lstm_ih.shape[1]
    n_heads = sum(1 for k in state_dict if k.startswith('heads.') and 'weight' in k)
    return n_in, hidden, n_heads


# ── Load model + scaler ───────────────────────────────────────────────────────
def load_model(model_path: str, scaler_path: str):
    sd        = torch.load(model_path, map_location='cpu')
    n_in, hidden, n_heads = _detect_dims(sd)
    print(f"  Model dims: input={n_in}, hidden={hidden}, heads={n_heads}")
    model = MultiHorizonLSTM(n_in, hidden, num_layers=2, n_horizons=n_heads)
    model.load_state_dict(sd)
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Trim FEATURE_COLS to match model input size
    feat = FEATURE_COLS[:n_in]
    return model, scaler, feat, n_heads


# ── Build sequence pool from CSV ──────────────────────────────────────────────
def build_sequence_pool(df: pd.DataFrame, api_name: str, feature_cols: list,
                        scaler, n_sequences: int, rng: np.random.Generator):
    """Return (sequences [N, SEQ_LEN, F], outcomes [N]) for one API."""
    sub = df[df['api_name'] == api_name].copy().reset_index(drop=True)
    sub[feature_cols] = sub[feature_cols].fillna(0).astype(np.float32)
    sub['success'] = sub['success'].fillna(1).astype(int)

    X_raw = sub[feature_cols].to_numpy(dtype=np.float32)
    y     = sub['success'].to_numpy()
    X     = scaler.transform(X_raw).astype(np.float32)

    max_start = len(X) - SEQ_LEN - 1
    if max_start < n_sequences:
        n_sequences = max_start
    starts  = rng.choice(max_start, size=n_sequences, replace=False)
    seqs    = np.stack([X[s: s + SEQ_LEN] for s in starts])
    outcomes = np.array([int(y[s + SEQ_LEN] == 0) for s in starts])  # 1 = failure
    return seqs, outcomes


# ── Run LSTM inference on a batch of sequences ────────────────────────────────
@torch.no_grad()
def predict_batch(model, seqs: np.ndarray):
    """Returns failure probabilities [N, n_heads] for horizons [h1, h5, h15]."""
    t = torch.from_numpy(seqs)
    logits = model(t)
    probs  = torch.sigmoid(logits).cpu().numpy()
    return probs   # shape [N, 3]


# ── Proactive agent ───────────────────────────────────────────────────────────
def run_proactive(transactions, probs_h1):
    """
    Decision logic per transaction:
      p > 0.65  → switch to backup before sending
      0.35-0.65 → retry with reduced timeout
      p < 0.35  → proceed normally
    Returns stats dict.
    """
    results = []
    for i, (outcome, p) in enumerate(zip(transactions, probs_h1)):
        if p > HIGH_RISK_THRESHOLD:
            # Switch to backup — backup API has lower failure rate
            # Assume backup failure prob = p * 0.25 (backup is healthier)
            backup_fail = np.random.random() < (p * 0.25)
            actual_fail = int(backup_fail)
            latency     = LATENCY_SWITCH
            action      = 'switch'
        elif p > LOW_RISK_THRESHOLD:
            # Retry with reduced timeout — halve the failure probability
            actual_fail = int(outcome and (np.random.random() < 0.5))
            latency     = LATENCY_RETRY
            action      = 'retry'
        else:
            actual_fail = int(outcome)
            latency     = LATENCY_NORMAL
            action      = 'normal'

        results.append({
            'action':      action,
            'failed':      actual_fail,
            'latency':     latency,
            'pred_prob':   float(p),
            'true_outcome': int(outcome),
        })
    return results


# ── Reactive baseline ─────────────────────────────────────────────────────────
def run_reactive(transactions):
    """
    Switch API only AFTER a failure occurs. No prediction used.
    Current request fails → next request goes to backup (1 switch),
    then returns to primary.
    """
    results   = []
    use_backup = False
    for outcome in transactions:
        if use_backup:
            # On backup: lower real failure rate (market_data ~4%)
            actual_fail = int(np.random.random() < 0.04)
            latency     = LATENCY_SWITCH
            action      = 'switch'
            use_backup  = False      # return to primary after 1 backup call
        else:
            actual_fail = int(outcome)
            latency     = LATENCY_NORMAL
            action      = 'normal'
            if actual_fail:
                use_backup = True    # trigger switch on next tx

        results.append({
            'action':      action,
            'failed':      actual_fail,
            'latency':     latency,
        })
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(results: list, label: str):
    n          = len(results)
    failures   = sum(r['failed'] for r in results)
    successes  = n - failures
    switches   = sum(1 for r in results if r['action'] == 'switch')
    avg_lat    = np.mean([r['latency'] for r in results])
    cost       = failures * COST_PER_FAILURE
    annual_cost = (failures / n) * TRANSACTIONS_PER_YEAR * COST_PER_FAILURE
    return {
        'label':           label,
        'n_transactions':  n,
        'successes':       successes,
        'failures':        failures,
        'failure_rate':    failures / n,
        'switches':        switches,
        'avg_latency_sec': round(avg_lat, 4),
        'cost_per_1000':   round(cost, 2),
        'annual_cost_usd': round(annual_cost, 2),
    }


# ── Print comparison table ────────────────────────────────────────────────────
def print_comparison(proactive, reactive):
    fail_reduction = (reactive['failures'] - proactive['failures']) / max(1, reactive['failures'])
    cost_saving    = reactive['annual_cost_usd'] - proactive['annual_cost_usd']
    latency_delta  = proactive['avg_latency_sec'] - reactive['avg_latency_sec']

    W = 54
    print()
    print('=' * W)
    print(' PROACTIVE vs REACTIVE — 1000 TRANSACTION SIMULATION')
    print('=' * W)
    fmt = '{:<28} {:>11} {:>12}'
    hdr = fmt.format('Metric', 'Proactive', 'Reactive')
    print(hdr)
    print('-' * W)

    rows = [
        ('Transactions',       f"{proactive['n_transactions']:,}", f"{reactive['n_transactions']:,}"),
        ('Failures',           f"{proactive['failures']:,}",       f"{reactive['failures']:,}"),
        ('Successes',          f"{proactive['successes']:,}",      f"{reactive['successes']:,}"),
        ('Failure rate',       f"{proactive['failure_rate']:.2%}", f"{reactive['failure_rate']:.2%}"),
        ('API switches',       f"{proactive['switches']:,}",       f"{reactive['switches']:,}"),
        ('Avg latency (s)',    f"{proactive['avg_latency_sec']:.3f}", f"{reactive['avg_latency_sec']:.3f}"),
        ('Cost / 1000 tx',     f"${proactive['cost_per_1000']:,.0f}", f"${reactive['cost_per_1000']:,.0f}"),
        ('Est. annual cost',   f"${proactive['annual_cost_usd']:,.0f}", f"${reactive['annual_cost_usd']:,.0f}"),
    ]
    for label, p, r in rows:
        print(fmt.format(label, p, r))

    print('=' * W)
    print(f"  Failure reduction :  {fail_reduction:+.1%}")
    print(f"  Annual cost saving:  ${cost_saving:,.0f}")
    print(f"  Latency delta     :  {latency_delta:+.3f}s per transaction")
    print('=' * W)
    print()
    return fail_reduction, cost_saving


# ── Bar chart ─────────────────────────────────────────────────────────────────
def save_chart(proactive, reactive, out_path: str):
    categories   = ['Failures\n(per 1000)', 'Avg Latency\n(ms)', 'Annual Cost\n($K)']
    pro_vals     = [
        proactive['failures'],
        proactive['avg_latency_sec'] * 1000,
        proactive['annual_cost_usd'] / 1000,
    ]
    react_vals   = [
        reactive['failures'],
        reactive['avg_latency_sec'] * 1000,
        reactive['annual_cost_usd'] / 1000,
    ]

    x   = np.arange(len(categories))
    w   = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_r = ax.bar(x - w/2, react_vals,  w, label='Reactive (baseline)',  color='#e05c5c', alpha=0.9)
    bars_p = ax.bar(x + w/2, pro_vals,    w, label='Proactive (LSTM)',     color='#4c9be8', alpha=0.9)

    for bar in list(bars_r) + list(bars_p):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * max(react_vals + pro_vals),
                f'{h:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title('Proactive LSTM Agent vs Reactive Baseline\n(1000 Financial Transactions)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylabel('Value')
    ax.set_ylim(0, max(react_vals) * 1.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Chart saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    # Load artefacts
    print('\nStep 1 — Loading model and scaler ...')
    model, scaler, feature_cols, n_heads = load_model(
        'models/stress_test_best_model.pth', 'models/scaler.pkl'
    )
    print(f"  Features used: {len(feature_cols)}")

    # Load CSV (pre-2025 only — real failure rates)
    print('Step 1 — Loading CSV (pre-2025 rows) ...')
    csv_path = 'data/banking_api_features.csv'
    df = pd.read_csv(csv_path, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['timestamp'] < '2025-01-01'].copy()
    print(f"  {len(df):,} rows available")

    apis = ['stock_price_api', 'crypto_api', 'forex_api',
            'market_data_api', 'transaction_api']

    # Per-API failure rates from data
    fail_rates = {}
    for api in apis:
        sub = df[df['api_name'] == api]
        fr  = 1 - sub['success'].fillna(1).mean()
        fail_rates[api] = fr
        backup = BACKUP_API[api]
        b_sub  = df[df['api_name'] == backup]
        b_fr   = 1 - b_sub['success'].fillna(1).mean()
        print(f"  {api:<22}  fail={fr:.3f}   backup={backup} ({b_fr:.3f})")

    # Step 2 — Build sequence pool: n_per_api sequences per API
    print(f'\nStep 2 — Building {args.n_transactions} transaction sequence pool ...')
    n_per_api = args.n_transactions // len(apis)
    all_seqs, all_outcomes, all_api = [], [], []
    for api in apis:
        seqs, outcomes = build_sequence_pool(
            df, api, feature_cols, scaler, n_per_api, rng
        )
        all_seqs.append(seqs)
        all_outcomes.append(outcomes)
        all_api.extend([api] * len(seqs))
        print(f"  {api:<22}  {len(seqs):,} seqs  failure rate in sample: "
              f"{outcomes.mean():.3f}")

    all_seqs     = np.vstack(all_seqs).astype(np.float32)
    all_outcomes = np.concatenate(all_outcomes)

    # Shuffle together
    idx = rng.permutation(len(all_seqs))
    all_seqs     = all_seqs[idx]
    all_outcomes = all_outcomes[idx]
    all_api_shuf = [all_api[i] for i in idx]

    # Trim to exact n_transactions
    N = min(args.n_transactions, len(all_seqs))
    all_seqs     = all_seqs[:N]
    all_outcomes = all_outcomes[:N]
    print(f"\n  Total transactions: {N:,}  "
          f"(overall failure rate: {all_outcomes.mean():.3f})")

    # Step 2 — Get LSTM predictions (h=1 = immediate failure probability)
    print('\nStep 2 — Running LSTM inference ...')
    t0    = time.time()
    probs = predict_batch(model, all_seqs)   # [N, 3]
    dt    = time.time() - t0
    probs_h1 = probs[:, 0]   # horizon=1 for immediate risk
    print(f"  Inference on {N:,} sequences: {dt:.2f}s")
    print(f"  Predicted failure prob — mean={probs_h1.mean():.3f}  "
          f"max={probs_h1.max():.3f}  std={probs_h1.std():.3f}")

    # Step 3 & 4 — Run both approaches
    print('\nStep 3 — Running proactive agent ...')
    proactive_results = run_proactive(all_outcomes, probs_h1)

    print('Step 4 — Running reactive baseline ...')
    reactive_results  = run_reactive(all_outcomes)

    # Step 5 — Compute metrics
    print('\nStep 5 — Computing metrics ...')
    pro_metrics  = compute_metrics(proactive_results, 'Proactive (LSTM)')
    react_metrics = compute_metrics(reactive_results,  'Reactive (baseline)')

    # Action distribution
    pro_actions   = pd.Series([r['action'] for r in proactive_results]).value_counts().to_dict()
    react_actions = pd.Series([r['action'] for r in reactive_results]).value_counts().to_dict()
    print(f"  Proactive actions : {pro_actions}")
    print(f"  Reactive actions  : {react_actions}")

    # Step 6 — Print comparison table
    fail_reduction, cost_saving = print_comparison(pro_metrics, react_metrics)

    # Step 7 — Save
    print('Step 7 — Saving results ...')
    os.makedirs('models', exist_ok=True)
    chart_path  = 'models/agent_simulation_chart.png'
    save_chart(pro_metrics, react_metrics, chart_path)

    output = {
        'n_transactions': N,
        'seed': args.seed,
        'proactive': {
            **pro_metrics,
            'action_distribution': pro_actions,
        },
        'reactive': {
            **react_metrics,
            'action_distribution': react_actions,
        },
        'comparison': {
            'failure_reduction_pct': round(fail_reduction * 100, 2),
            'annual_cost_saving_usd': round(cost_saving, 2),
            'annual_failures_avoided': round(
                (react_metrics['failure_rate'] - pro_metrics['failure_rate'])
                * TRANSACTIONS_PER_YEAR, 0
            ),
        },
        'assumptions': {
            'cost_per_failure_usd':     COST_PER_FAILURE,
            'transactions_per_year':    TRANSACTIONS_PER_YEAR,
            'high_risk_threshold':      HIGH_RISK_THRESHOLD,
            'low_risk_threshold':       LOW_RISK_THRESHOLD,
            'latency_normal_sec':       LATENCY_NORMAL,
            'latency_retry_sec':        LATENCY_RETRY,
            'latency_switch_sec':       LATENCY_SWITCH,
            'backup_fail_multiplier':   0.25,
        },
        'artifact_paths': {
            'chart': chart_path,
            'json':  'models/agent_simulation_results.json',
        }
    }
    json_path = 'models/agent_simulation_results.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  JSON  saved -> {json_path}")
    print('\nSimulation complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_transactions', type=int, default=1000,
                        help='Number of transactions to simulate (default: 1000)')
    parser.add_argument('--seed',           type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args)
