#!/usr/bin/env python3
"""
add_cross_api_features.py
Add 5 cross-API correlation features to banking_api_features_clean.csv.
Output: data/banking_api_features_v6.csv

Why these features exist:
  The v5 model was learning per-API patterns in isolation. Banking APIs are
  interconnected -- when market_data_api spikes, crypto_api often follows.
  These 5 features capture systemic stress signals across APIs that a single-
  API view cannot see, giving the LSTM genuine cross-system early-warning signals.

The 5 new features (per row / per API per minute):
  avg_error_rate_others  -- mean error rate of the other 4 APIs at this timestamp
  max_error_rate_others  -- max error rate of the other 4 APIs at this timestamp
  n_apis_elevated        -- count of other APIs with error_rate_rolling > 0.05
  corr_with_similar_api  -- error rate of a closely-paired API (see SIMILAR_API map)
  systemic_stress_index  -- avg_error_rate_others * n_apis_elevated (compound signal)

NaN handling:
  market_data and transaction APIs have sparse timestamps (53-58% NaN in pivot).
  We use ffill().bfill() on the pivot so each API carries its last known error
  rate forward into gaps rather than leaving zeros everywhere.

Usage:
    python scripts/add_cross_api_features.py
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# ── Tee stdout -> console + report file ───────────────────────────────────────
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


os.makedirs("models", exist_ok=True)
_log = open("models/add_cross_api_features_report.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log)

print(f"=== add_cross_api_features.py  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===\n")

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_PATH  = os.path.join("data", "banking_api_features_clean.csv")
OUTPUT_PATH = os.path.join("data", "banking_api_features_v6.csv")
TIME_WINDOW = "1min"   # bucket size for timestamp aggregation

ALL_APIS = [
    "stock_price_api",
    "forex_api",
    "crypto_api",
    "market_data_api",
    "transaction_api",
]

# Each API's "closest sibling" for corr_with_similar_api feature
SIMILAR_API = {
    "stock_price_api": "forex_api",
    "forex_api":       "stock_price_api",
    "crypto_api":      "market_data_api",
    "market_data_api": "crypto_api",
    "transaction_api": "market_data_api",
}

NEW_FEATURES = [
    "avg_error_rate_others",
    "max_error_rate_others",
    "n_apis_elevated",
    "corr_with_similar_api",
    "systemic_stress_index",
]

# ── Step 1: Load ──────────────────────────────────────────────────────────────
print(f"Step 1 -- Loading {INPUT_PATH} ...")
t0 = time.time()
df = pd.read_csv(INPUT_PATH)
print(f"  Loaded {len(df):,} rows  ({len(df.columns)} columns)  in {time.time()-t0:.1f}s")
found_apis = sorted(df["api_name"].unique())
print(f"  APIs found: {found_apis}")
missing = [a for a in ALL_APIS if a not in found_apis]
if missing:
    print(f"  WARNING: APIs not found in data: {missing}")

# ── Step 2: Parse timestamps and create 1-min buckets ─────────────────────────
print("\nStep 2 -- Parsing timestamps and creating 1-min buckets ...")
t0 = time.time()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["ts_bucket"] = df["timestamp"].dt.floor(TIME_WINDOW)
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"  Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
print(f"  Unique raw timestamps: {df['timestamp'].nunique():,}")
print(f"  Unique 1-min buckets : {df['ts_bucket'].nunique():,}")
print(f"  Done in {time.time()-t0:.1f}s")

# ── Step 3: Pivot error_rate_rolling -> wide by (ts_bucket, api_name) ─────────
print("\nStep 3 -- Aggregating and pivoting error_rate_rolling by 1-min bucket ...")
t0 = time.time()

# Aggregate duplicate (ts_bucket, api_name) pairs using mean
agg = (
    df.groupby(["ts_bucket", "api_name"], sort=False)["error_rate_rolling"]
    .mean()
    .reset_index()
)
pivot = agg.pivot(index="ts_bucket", columns="api_name", values="error_rate_rolling")

# Ensure all 5 API columns are present
for api in ALL_APIS:
    if api not in pivot.columns:
        print(f"  NOTE: {api} not found in pivot -- filling with NaN")
        pivot[api] = np.nan
pivot = pivot[ALL_APIS]  # enforce column order

print(f"  Pivot shape: {pivot.shape}  ({time.time()-t0:.1f}s)")
print(f"  Unique buckets: {len(pivot):,}")
nan_rates_before = "  ".join(
    f"{a.split('_')[0]}={pivot[a].isna().mean()*100:.1f}%" for a in ALL_APIS
)
print(f"  NaN rate before fill: {nan_rates_before}")

# Fill gaps -- carry each API's last known error rate forward (then backward for leading NaN)
# This is critical: market_data (53.6% NaN) and transaction (57.9% NaN) have sparse timestamps.
# Without this fill, cross-API features would mostly be 0 for those APIs.
pivot = pivot.ffill().bfill()
nan_rates_after = "  ".join(
    f"{a.split('_')[0]}={pivot[a].isna().mean()*100:.1f}%" for a in ALL_APIS
)
print(f"  NaN rate after fill : {nan_rates_after}")

# ── Step 4: Compute 5 cross-API features -- vectorised, no row loops ──────────
print("\nStep 4 -- Computing cross-API features (vectorised) ...")
t0 = time.time()

cross_parts = []
for api in ALL_APIS:
    others = [a for a in ALL_APIS if a != api]
    others_df = pivot[others]                             # shape: (n_buckets, 4)

    avg_others   = others_df.mean(axis=1)                # mean of 4 others (skipna)
    max_others   = others_df.max(axis=1)                 # max of 4 others (skipna)
    n_elev       = (others_df > 0.05).sum(axis=1)        # count others > 5% error threshold
    corr_partner = pivot[SIMILAR_API[api]]               # similar API's error rate
    stress       = avg_others * n_elev                   # compound systemic stress signal

    part = pd.DataFrame({
        "ts_bucket":              pivot.index,
        "api_name":               api,
        "avg_error_rate_others":  avg_others.to_numpy(),
        "max_error_rate_others":  max_others.to_numpy(),
        "n_apis_elevated":        n_elev.to_numpy().astype(np.float32),
        "corr_with_similar_api":  corr_partner.to_numpy(),
        "systemic_stress_index":  stress.to_numpy(),
    })
    cross_parts.append(part)
    print(f"  {api:<22} done")

cross_df = pd.concat(cross_parts, ignore_index=True)
print(f"  Cross feature df: {cross_df.shape}  ({time.time()-t0:.1f}s)")

# ── Step 5: Merge cross-API features back onto original dataframe ─────────────
print("\nStep 5 -- Merging cross-API features onto original dataframe ...")
t0 = time.time()
df = df.merge(cross_df, on=["ts_bucket", "api_name"], how="left")
print(f"  After merge: {df.shape}  ({time.time()-t0:.1f}s)")

# Drop the helper column
df = df.drop(columns=["ts_bucket"])

# ── Step 6: Fill any remaining NaN in new features ────────────────────────────
print("\nStep 6 -- Final NaN check ...")
for feat in NEW_FEATURES:
    n_na = df[feat].isna().sum()
    if n_na:
        print(f"  {feat}: {n_na:,} NaN -> 0")
    df[feat] = df[feat].fillna(0)

# ── Step 7: Save ──────────────────────────────────────────────────────────────
print(f"\nStep 7 -- Saving to {OUTPUT_PATH} ...")
t0 = time.time()
df.to_csv(OUTPUT_PATH, index=False)
size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
print(f"  Saved {len(df):,} rows  {len(df.columns)} columns  {size_mb:.1f} MB  ({time.time()-t0:.1f}s)")

# ── Validation summary ────────────────────────────────────────────────────────
print("\n=== Cross-API Feature Summary ===")
print(f"Total rows            : {len(df):,}")
print(f"New features added    : {len(NEW_FEATURES)}")

for feat in NEW_FEATURES:
    col       = df[feat]
    mean_v    = col.mean()
    max_v     = col.max()
    nonzero_p = (col != 0).mean() * 100
    if feat == "n_apis_elevated":
        print(f"{feat:<28}-- mean: {mean_v:.4f}  max: {int(max_v)}        non-zero: {nonzero_p:.1f}%")
    else:
        print(f"{feat:<28}-- mean: {mean_v:.4f}  max: {max_v:.4f}  non-zero: {nonzero_p:.1f}%")
    if nonzero_p < 10.0:
        print(f"  WARNING: {feat} has very low non-zero rate ({nonzero_p:.1f}%) -- timestamp pivot may have failed")

print(f"\nOutput saved -> {OUTPUT_PATH}")
print(f"\nNext step: python scripts/run_lstm_training.py --data {OUTPUT_PATH}")

_log.close()
