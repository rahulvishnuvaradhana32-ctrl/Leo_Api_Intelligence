#!/usr/bin/env python3
"""
add_cross_api_features.py
Add 5 cross-API correlation features to banking_api_features_clean.csv.
Output: data/banking_api_features_v6.csv
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
            s.write(data)
            s.flush()

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

# APIs that fire within this window are treated as co-occurring.
# "1min" means all requests within the same minute share cross-API context.
TIME_WINDOW = "1min"

ALL_APIS = [
    "stock_price_api",
    "forex_api",
    "crypto_api",
    "market_data_api",
    "transaction_api",
]

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
print(f"Step 1 — Loading {INPUT_PATH} ...")
t0 = time.time()
df = pd.read_csv(INPUT_PATH)
print(f"  Loaded {len(df):,} rows  ({len(df.columns)} columns)  in {time.time()-t0:.1f}s")
found_apis = sorted(df["api_name"].unique())
print(f"  APIs found: {found_apis}")
missing = [a for a in ALL_APIS if a not in found_apis]
if missing:
    print(f"  WARNING: APIs not found in data: {missing}")

# ── Step 2: Sort by timestamp and bucket to TIME_WINDOW ──────────────────────
print(f"\nStep 2 — Parsing, sorting, and bucketing timestamps to {TIME_WINDOW} ...")
t0 = time.time()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
df["ts_bucket"] = df["timestamp"].dt.floor(TIME_WINDOW)
n_buckets = df["ts_bucket"].nunique()
print(f"  Date range : {df['timestamp'].min()} -> {df['timestamp'].max()}")
print(f"  Time buckets ({TIME_WINDOW}): {n_buckets:,}  (vs {df['timestamp'].nunique():,} exact timestamps)")
print(f"  Done in {time.time()-t0:.1f}s")

# ── Step 3: Pivot error_rate_rolling -> wide by (ts_bucket, api_name) ──────────
print(f"\nStep 3 — Aggregating and pivoting error_rate_rolling by {TIME_WINDOW} buckets ...")
t0 = time.time()

# Aggregate all rows within the same (bucket, api_name) to a single value
agg = (
    df.groupby(["ts_bucket", "api_name"], sort=False)["error_rate_rolling"]
    .mean()
    .reset_index()
)
pivot = agg.pivot(index="ts_bucket", columns="api_name", values="error_rate_rolling")

# Ensure all 5 API columns are present (fill missing API columns with NaN)
for api in ALL_APIS:
    if api not in pivot.columns:
        print(f"  NOTE: {api} not found in pivot — filling with NaN")
        pivot[api] = np.nan
pivot = pivot[ALL_APIS]  # enforce column order

# Before forward-fill: report raw NaN rates (timestamp gaps)
nan_before = "  ".join(
    f"{a.split('_')[0]}={pivot[a].isna().mean()*100:.1f}%" for a in ALL_APIS
)
print(f"  NaN rate before fill: {nan_before}")

# Forward-fill then backward-fill: carry each API's last known error rate
# into gaps where that API has no observation.
# Semantics: "what was this API's most recent error rate?"
pivot = pivot.ffill().bfill()

nan_after = "  ".join(
    f"{a.split('_')[0]}={pivot[a].isna().mean()*100:.1f}%" for a in ALL_APIS
)
print(f"  NaN rate after  fill: {nan_after}")
print(f"  Pivot shape: {pivot.shape}  ({time.time()-t0:.1f}s)")
print(f"  Unique buckets: {len(pivot):,}")

# ── Step 4: Compute 5 cross-API features — vectorised, no row loops ───────────
print("\nStep 4 — Computing cross-API features (vectorised) ...")
t0 = time.time()

cross_parts = []
for api in ALL_APIS:
    others = [a for a in ALL_APIS if a != api]
    others_df = pivot[others]                             # shape: (n_ts, 4)

    avg_others   = others_df.mean(axis=1)                # mean of 4 others (skipna)
    max_others   = others_df.max(axis=1)                 # max of 4 others (skipna)
    n_elev       = (others_df > 0.05).sum(axis=1)        # count others > threshold
    corr_partner = pivot[SIMILAR_API[api]]               # similar API's error rate
    stress       = avg_others * n_elev                   # systemic stress index

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
print("\nStep 5 — Merging cross-API features onto original dataframe ...")
t0 = time.time()
df = df.merge(cross_df, on=["ts_bucket", "api_name"], how="left")
df = df.drop(columns=["ts_bucket"])   # remove join key — not a training feature
print(f"  After merge: {df.shape}  ({time.time()-t0:.1f}s)")

# ── Step 6: Fill NaN ──────────────────────────────────────────────────────────
print("\nStep 6 — Filling NaN with 0 ...")
for feat in NEW_FEATURES:
    n_na = df[feat].isna().sum()
    if n_na:
        print(f"  {feat}: {n_na:,} NaN -> 0")
    df[feat] = df[feat].fillna(0)

# ── Step 7: Save ──────────────────────────────────────────────────────────────
print(f"\nStep 7 — Saving to {OUTPUT_PATH} ...")
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
        print(f"{feat:<28}— mean: {mean_v:.4f}  max: {int(max_v)}        non-zero: {nonzero_p:.1f}%")
    else:
        print(f"{feat:<28}— mean: {mean_v:.4f}  max: {max_v:.4f}  non-zero: {nonzero_p:.1f}%")
    if nonzero_p < 10.0:
        print(f"  ⚠ WARNING: {feat} has very low non-zero rate ({nonzero_p:.1f}%) — timestamp pivot may have failed")

print(f"Output saved -> {OUTPUT_PATH}")

_log.close()
