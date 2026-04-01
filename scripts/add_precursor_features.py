#!/usr/bin/env python3
"""
add_precursor_features.py  --  Enrich the banking API features CSV with
trend-based failure precursor signals and instability indicators.

Adds 11 new columns to data/banking_api_features.csv (in-place):

  Core precursors (Steps 1-2):
    latency_diff_1     -- response_time acceleration over 1 step
    latency_diff_5     -- response_time acceleration over 5 steps
    error_rate_diff_1  -- error rate change over 1 step
    error_rate_diff_5  -- error rate change over 5 steps (EMA-10 proxy)
    latency_spike      -- current latency vs rolling mean ratio
    error_burst        -- current error rate vs EMA-10 baseline ratio
    instability_index  -- combined short-term system volatility

  Advanced stability signals (Step 3):
    latency_slope      -- EMA-based latency trend direction
    error_slope        -- EMA-based error rate trend direction
    traffic_change     -- request count delta vs previous step (per API)
    burst_ratio        -- request count vs rolling-60 mean (per API)

Already-present columns are skipped -- safe to re-run.

Usage:
    python scripts/add_precursor_features.py
    python scripts/add_precursor_features.py --data_path data/custom.csv
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data/banking_api_features_v7.csv")
parser.add_argument("--no_backup", action="store_true",
                    help="Skip creating a .bak copy before writing")
args = parser.parse_args()

EPS = 1e-6

CORE_FEATURES = [
    "latency_diff_1", "latency_diff_5",
    "error_rate_diff_1", "error_rate_diff_5",
    "latency_spike", "error_burst", "instability_index",
]
ADVANCED_FEATURES = [
    "latency_slope", "error_slope",
    "traffic_change", "burst_ratio",
]
ALL_NEW = CORE_FEATURES + ADVANCED_FEATURES


def main():
    t0 = time.time()
    print("=== Add Precursor Features ===\n")

    if not os.path.exists(args.data_path):
        print(f"ERROR: {args.data_path} not found"); sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {args.data_path} ...")
    df = pd.read_csv(args.data_path, low_memory=False)
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

    # Check which features already exist
    already   = [f for f in ALL_NEW if f in df.columns]
    to_add    = [f for f in ALL_NEW if f not in df.columns]
    if already:
        print(f"  Already present (will skip): {already}")
    if not to_add:
        print("  All precursor features already in CSV. Nothing to do.")
        return
    print(f"  Will add: {to_add}\n")

    # ── Backup ────────────────────────────────────────────────────────────────
    if not args.no_backup:
        bak = args.data_path + ".bak"
        if not os.path.exists(bak):
            print(f"Backing up -> {bak}  (pass --no_backup to skip)")
            import shutil
            shutil.copy2(args.data_path, bak)
            print(f"  Backup created ({os.path.getsize(bak)/1e6:.0f} MB)")
        else:
            print(f"  Backup already exists at {bak}, skipping.")

    # ── Core precursors (row-wise from existing precomputed columns) ───────────
    # These are correct without per-API groupby because the lag/rolling columns
    # were already computed per-API during data generation and stored in the CSV.
    print("Computing core precursor features ...")

    if "latency_diff_1" not in df.columns:
        df["latency_diff_1"] = (df["response_time"] - df["response_time_lag_1"]).fillna(0)
        print("  + latency_diff_1")

    if "latency_diff_5" not in df.columns:
        df["latency_diff_5"] = (df["response_time"] - df["response_time_lag_5"]).fillna(0)
        print("  + latency_diff_5")

    if "error_rate_diff_1" not in df.columns:
        df["error_rate_diff_1"] = (df["error_rate_rolling"] - df["error_rate_lag_1"]).fillna(0)
        print("  + error_rate_diff_1")

    if "error_rate_diff_5" not in df.columns:
        # No error_rate_lag_5 in CSV; use EMA-10 as the 5-step-ahead baseline proxy
        df["error_rate_diff_5"] = (df["error_rate_rolling"] - df["error_rate_ema_10"]).fillna(0)
        print("  + error_rate_diff_5  (proxy: rolling - ema_10)")

    if "latency_spike" not in df.columns:
        df["latency_spike"] = df["response_time"] / (df["response_time_rolling_mean"] + EPS)
        print("  + latency_spike")

    if "error_burst" not in df.columns:
        df["error_burst"] = df["error_rate_rolling"] / (df["error_rate_ema_10"] + EPS)
        print("  + error_burst")

    if "instability_index" not in df.columns:
        df["instability_index"] = (
            df["latency_diff_1"].abs() + df["error_rate_diff_1"].abs()
        )
        print("  + instability_index")

    # ── Advanced stability signals (per-API groupby required) ─────────────────
    print("\nComputing advanced stability signals ...")

    if "latency_slope" not in df.columns:
        # EMA slope proxy: faster EMA minus slower EMA, scaled to per-step units
        df["latency_slope"] = (
            (df["response_time_ema_10"] - df["response_time_ema_30"]) / 20.0
        )
        print("  + latency_slope  (proxy: (ema_10 - ema_30) / 20)")

    if "error_slope" not in df.columns:
        df["error_slope"] = (
            (df["error_rate_ema_10"] - df["error_rate_lag_1"]) / 10.0
        ).fillna(0)
        print("  + error_slope  (proxy: (ema_10 - lag_1) / 10)")

    # traffic_change and burst_ratio require request_count shift within each API
    if "traffic_change" not in df.columns or "burst_ratio" not in df.columns:
        print("  Computing traffic_change and burst_ratio (per-API groupby) ...")
        grp = df.groupby("api_name", sort=False)

        if "traffic_change" not in df.columns:
            df["traffic_change"] = grp["request_count"].transform(
                lambda x: (x - x.shift(1)).fillna(0)
            )
            print("  + traffic_change")

        if "burst_ratio" not in df.columns:
            rolling_mean = grp["request_count"].transform(
                lambda x: x.rolling(60, min_periods=1).mean()
            )
            df["burst_ratio"] = df["request_count"] / (rolling_mean + EPS)
            print("  + burst_ratio")

    # ── Validate ──────────────────────────────────────────────────────────────
    added_cols = [f for f in ALL_NEW if f in df.columns]
    nan_counts = {f: int(df[f].isna().sum()) for f in added_cols}
    nan_cols   = {k: v for k, v in nan_counts.items() if v > 0}
    if nan_cols:
        print(f"\n  [!] NaN values in new columns (will be 0 after fillna in training):")
        for col, n in nan_cols.items():
            print(f"      {col}: {n:,} NaNs")
        for col in nan_cols:
            df[col] = df[col].fillna(0)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\nNew feature statistics:")
    for col in added_cols:
        s = df[col]
        print(f"  {col:<22}  mean={s.mean():>8.4f}  std={s.std():>8.4f}  "
              f"min={s.min():>8.2f}  max={s.max():>8.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving updated CSV ({len(df.columns)} columns) -> {args.data_path} ...")
    t_save = time.time()
    df.to_csv(args.data_path, index=False)
    print(f"  Done in {time.time()-t_save:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n=== Summary ===")
    print(f"  Added {len([f for f in ALL_NEW if f in to_add])} new features "
          f"to {len(df):,} rows")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Runtime: {elapsed:.0f}s")
    print(f"  Output: {args.data_path}")
    print(f"\nNext step: retrain with all features:")
    print(f"  python scripts/run_lstm_training.py --epochs 30\n")


if __name__ == "__main__":
    main()
