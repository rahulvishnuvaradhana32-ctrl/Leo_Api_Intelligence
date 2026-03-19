#!/usr/bin/env python3
"""LEO API — Dataset Fix Script
Fixes two critical issues before the next LSTM training run:
  1. Merges rows present in SQLite but missing from the training CSV
  2. Fills null error_type values with meaningful labels
  3. Downsamples dominant transaction_api to ≤25% of dataset
  4. Saves data/banking_api_features_clean.csv

Column notes (actual schema differs from design spec):
  - Neither DB nor CSV has data_source / failure_event / is_failure columns.
  - is_failure is derived from success == 0.
  - New rows from DB are labelled data_source='db_extended';
    existing CSV rows are labelled data_source='synthetic'.
  - traffic_change and burst_ratio are absent from DB rows; set to 0 for them.

Usage:
    python scripts/fix_dataset.py
"""
import os
import sqlite3
import sys
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CSV_PATH      = os.path.join("data", "banking_api_features.csv")
DB_PATH       = os.path.join("data", "banking_api_telemetry.db")
OUT_CSV_PATH  = os.path.join("data", "banking_api_features_clean.csv")
REPORT_PATH   = os.path.join("models", "fix_dataset_report.txt")

EPS = 1e-6

# Failure type mapping per api_name (most statistically common from audit)
FAILURE_MAP = {
    "transaction_api": "cascading_failure",
    "market_data_api": "vendor_upstream_failure",
    "stock_price_api": "market_volatility_overload",
    "crypto_api":      "market_volatility_overload",
    "forex_api":       "regulatory_load_spike",
}

# ── Tee output ────────────────────────────────────────────────────────────────
_buf = StringIO()

def out(msg: str = "") -> None:
    print(msg, flush=True)
    _buf.write(msg + "\n")

def section(title: str) -> None:
    line = "=" * 72
    out()
    out(line)
    out(f"  {title}")
    out(line)

def warn(msg: str) -> None:
    out(f"  ⚠ WARNING: {msg}")


# ── On-the-fly feature computation ───────────────────────────────────────────
def compute_precursor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features for rows that came from the DB (no precomputed features)."""
    def _col(name: str, fill: float = 0.0) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(fill, index=df.index)

    rt   = df["response_time"]
    rl1  = _col("response_time_lag_1", 0.0)
    rl5  = _col("response_time_lag_5", 0.0)
    er   = _col("error_rate_rolling",  0.0)
    el1  = _col("error_rate_lag_1",    0.0)
    rm   = _col("response_time_rolling_mean", 1.0)
    e10  = _col("error_rate_ema_10",   0.0)
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
    # traffic_change / burst_ratio are not computable from base columns — zero-fill
    if "traffic_change" not in df.columns:
        df["traffic_change"] = 0.0
    if "burst_ratio" not in df.columns:
        df["burst_ratio"] = 0.0
    return df


# ══════════════════════════════════════════════════════════════════════════════
out("=" * 72)
out(f"  LEO API — Dataset Fix Script  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
out("=" * 72)
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load CSV and DB, identify missing rows
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1 — Load existing CSV and SQLite DB")

out(f"  Loading CSV: {CSV_PATH} …")
try:
    csv_df = pd.read_csv(CSV_PATH, low_memory=False)
    out(f"  CSV loaded: {len(csv_df):,} rows × {len(csv_df.columns)} columns")
except Exception as e:
    out(f"  FATAL: cannot load CSV — {e}")
    sys.exit(1)

out(f"  Loading DB: {DB_PATH} …")
try:
    con    = sqlite3.connect(DB_PATH)
    db_df  = pd.read_sql("SELECT * FROM api_telemetry", con)
    con.close()
    out(f"  DB loaded:  {len(db_df):,} rows × {len(db_df.columns)} columns")
except Exception as e:
    out(f"  FATAL: cannot load DB — {e}")
    sys.exit(1)

# Normalise timestamps to string for reliable key matching
csv_df["timestamp"] = pd.to_datetime(csv_df["timestamp"], errors="coerce").astype(str)
db_df["timestamp"]  = pd.to_datetime(db_df["timestamp"],  errors="coerce").astype(str)

# Check for data_source in DB — it doesn't exist; warn and proceed
if "data_source" in db_df.columns:
    out(f"  data_source column found in DB — using it to identify Kaggle rows.")
    kaggle_db = db_df[db_df["data_source"] != "synthetic"].copy()
    out(f"  DB Kaggle rows loaded: {len(kaggle_db):,}")
else:
    warn("DB has no 'data_source' column — cannot filter by source.")
    out("  Falling back: treating ALL DB rows not in CSV as 'db_extended'.")
    kaggle_db = db_df.copy()
    out(f"  DB rows loaded for comparison: {len(kaggle_db):,}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1b — Deduplicate DB rows, cross-check against CSV
# ──────────────────────────────────────────────────────────────────────────────
section("STEP 1b — Deduplicate and cross-check")

before_dedup = len(kaggle_db)
out(f"  DB rows before dedup : {before_dedup:,}")

key_cols = ["api_name", "timestamp"]
if "data_source" in kaggle_db.columns:
    key_cols.append("data_source")

kaggle_db = kaggle_db.drop_duplicates(subset=key_cols, keep="first")
after_dedup = len(kaggle_db)
dups_found  = before_dedup - after_dedup
out(f"  Duplicate rows found  : {dups_found:,}")
out(f"  DB rows after dedup  : {after_dedup:,}")

# Cross-check: find rows already in CSV (by api_name + timestamp)
csv_keys    = set(zip(csv_df["api_name"], csv_df["timestamp"]))
mask_new    = ~pd.Series(
    list(zip(kaggle_db["api_name"], kaggle_db["timestamp"]))
).isin(csv_keys).values

skipped  = (~mask_new).sum()
new_rows = mask_new.sum()
out(f"  Kaggle rows skipped (already in CSV): {skipped:,}")
out(f"  Kaggle rows newly added              : {new_rows:,}")

if new_rows == 0:
    warn("No new rows to add — DB rows are fully duplicated in CSV. Proceeding with CSV as-is.")
    merged_df = csv_df.copy()
    skip_merge = True
else:
    skip_merge = False
    new_db_rows = kaggle_db[mask_new].copy()

if not skip_merge:
    # ── Add data_source column ────────────────────────────────────────────────
    if "data_source" not in csv_df.columns:
        csv_df["data_source"] = "synthetic"
        out("  Added data_source='synthetic' to existing CSV rows.")

    src_label = "db_extended"
    if "data_source" in new_db_rows.columns:
        # Use existing values if present
        out("  DB rows have data_source values — preserving them.")
    else:
        new_db_rows["data_source"] = src_label

    # Compute on-the-fly features for new DB rows (missing from DB schema)
    out(f"  Computing precursor features for {new_rows:,} new DB rows …")
    new_db_rows = compute_precursor_features(new_db_rows)

    # Drop DB-only columns not in CSV (id, is_pre_open, is_market_close)
    extra_db_cols = [c for c in new_db_rows.columns if c not in csv_df.columns and c not in ["data_source"]]
    if extra_db_cols:
        out(f"  Dropping DB-only columns not in CSV: {extra_db_cols}")
        new_db_rows = new_db_rows.drop(columns=extra_db_cols, errors="ignore")

    # Add missing CSV columns to new rows (fill with 0)
    for col in csv_df.columns:
        if col not in new_db_rows.columns:
            new_db_rows[col] = 0
            out(f"  Zero-filled missing column in new rows: {col}")

    # Align column order to CSV
    new_db_rows = new_db_rows[[c for c in csv_df.columns if c in new_db_rows.columns]]

    merged_df = pd.concat([csv_df, new_db_rows], ignore_index=True, sort=False)
    out(f"\n  Rows after merge: {len(merged_df):,}")

    out("\n  Rows per data_source after merge:")
    for src, cnt in merged_df["data_source"].value_counts().items():
        out(f"    {src:<20}: {cnt:,}")

else:
    # Still add data_source if not present
    if "data_source" not in merged_df.columns:
        merged_df["data_source"] = "synthetic"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Fix null failure types
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2 — Fix null / unknown error_type")

# Derive is_failure from success column (success=0 → failure)
if "success" in merged_df.columns:
    is_failure_mask = merged_df["success"] == 0
else:
    warn("No 'success' column found — cannot identify failure rows.")
    is_failure_mask = pd.Series(False, index=merged_df.index)

null_mask = (
    merged_df["error_type"].isna()
    | (merged_df["error_type"].astype(str).str.strip() == "")
    | (merged_df["error_type"].astype(str).str.lower() == "unknown")
)

null_before = null_mask.sum()
out(f"  Null error_type before fix  : {null_before:,}")

# Sub-masks
null_failure_mask = null_mask & is_failure_mask
null_success_mask = null_mask & ~is_failure_mask

# (a) Backfill from failure_event if it exists
backfilled = 0
if "failure_event" in merged_df.columns:
    fe_available = null_failure_mask & merged_df["failure_event"].notna()
    merged_df.loc[fe_available, "error_type"] = merged_df.loc[fe_available, "failure_event"]
    backfilled = int(fe_available.sum())
    out(f"  Backfilled from failure_event: {backfilled:,}")
else:
    out("  No 'failure_event' column — skipping backfill step.")

# Re-compute null mask after backfill
null_mask = (
    merged_df["error_type"].isna()
    | (merged_df["error_type"].astype(str).str.strip() == "")
    | (merged_df["error_type"].astype(str).str.lower() == "unknown")
)
null_failure_mask = null_mask & is_failure_mask

# (b) Fill remaining failure nulls from api_name mapping
mapping_filled = 0
for api, label in FAILURE_MAP.items():
    api_null_fail = null_failure_mask & (merged_df["api_name"] == api)
    count = int(api_null_fail.sum())
    if count > 0:
        merged_df.loc[api_null_fail, "error_type"] = label
        mapping_filled += count
        out(f"    {api:<25}: filled {count:,} rows → '{label}'")
out(f"  Filled from api_name mapping : {mapping_filled:,}")

# (c) Fill success-row nulls with 'none'
null_mask = (
    merged_df["error_type"].isna()
    | (merged_df["error_type"].astype(str).str.strip() == "")
    | (merged_df["error_type"].astype(str).str.lower() == "unknown")
)
null_success_now = null_mask & ~is_failure_mask
merged_df.loc[null_success_now, "error_type"] = "none"
out(f"  Healthy rows filled with 'none': {int(null_success_now.sum()):,}")

# Final null count
null_after = (
    merged_df["error_type"].isna()
    | (merged_df["error_type"].astype(str).str.strip() == "")
    | (merged_df["error_type"].astype(str).str.lower() == "unknown")
).sum()
out(f"  Null error_type after fix    : {null_after:,}  (should be 0)")
if null_after > 0:
    warn(f"{null_after:,} null error_type rows remain — check api_name coverage in FAILURE_MAP.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Jointly downsample transaction_api AND market_data_api to ≤25% each
#
# Because the two caps are coupled (removing market_data_api rows changes the
# denominator for transaction_api and vice versa), they must be solved together.
#
# Let R = rows belonging to all OTHER APIs (stock_price, crypto, forex) — fixed.
# Let F_ta, F_ma = failure rows kept for each API (never dropped).
# Let X_ta, X_ma = success rows we KEEP for each API.
#
# Final total  T = R + (F_ta + X_ta) + (F_ma + X_ma)
# Constraint   (F_ta + X_ta) / T = 0.25   →  F_ta + X_ta = 0.25 T
#              (F_ma + X_ma) / T = 0.25   →  F_ma + X_ma = 0.25 T
#
# Solving:   T = R + 0.25T + 0.25T  →  T = 2R
#            X_ta = 0.5R - F_ta     (capped at actual available success rows)
#            X_ma = 0.5R - F_ma
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3 — Joint downsample: transaction_api + market_data_api to ≤25% each")

is_failure_mask = merged_df["success"] == 0

ta_mask         = merged_df["api_name"] == "transaction_api"
ma_mask         = merged_df["api_name"] == "market_data_api"
other_mask      = ~(ta_mask | ma_mask)

ta_fail_mask    = ta_mask & is_failure_mask
ma_fail_mask    = ma_mask & is_failure_mask
ta_success_mask = ta_mask & ~is_failure_mask
ma_success_mask = ma_mask & ~is_failure_mask

before_total  = len(merged_df)
before_ta     = int(ta_mask.sum())
before_ma     = int(ma_mask.sum())
R             = int(other_mask.sum())           # fixed rows, never touched
F_ta          = int(ta_fail_mask.sum())
F_ma          = int(ma_fail_mask.sum())
ta_succ_rows  = int(ta_success_mask.sum())
ma_succ_rows  = int(ma_success_mask.sum())

out(f"  transaction_api before : {before_ta:,}  ({100*before_ta/max(1,before_total):.2f}%)")
out(f"  market_data_api before : {before_ma:,}  ({100*before_ma/max(1,before_total):.2f}%)")
out(f"  Other APIs (fixed)     : {R:,}")
out(f"  Failure rows kept      : transaction_api={F_ta:,}  market_data_api={F_ma:,}")

# Joint solution
target_ta_keep = max(0, int(0.5 * R - F_ta))
target_ma_keep = max(0, int(0.5 * R - F_ma))

# Clamp to available success rows (no need to drop if already within cap)
target_ta_keep = min(target_ta_keep, ta_succ_rows)
target_ma_keep = min(target_ma_keep, ma_succ_rows)

drop_indices = []
rng = np.random.default_rng(42)

if target_ta_keep < ta_succ_rows:
    succ_ta_idx = merged_df[ta_success_mask].index.to_numpy()
    keep_ta_idx = rng.choice(succ_ta_idx, size=target_ta_keep, replace=False)
    drop_indices.extend(np.setdiff1d(succ_ta_idx, keep_ta_idx).tolist())

if target_ma_keep < ma_succ_rows:
    succ_ma_idx = merged_df[ma_success_mask].index.to_numpy()
    keep_ma_idx = rng.choice(succ_ma_idx, size=target_ma_keep, replace=False)
    drop_indices.extend(np.setdiff1d(succ_ma_idx, keep_ma_idx).tolist())

if drop_indices:
    merged_df = merged_df.drop(index=drop_indices).reset_index(drop=True)

after_total = len(merged_df)
after_ta    = int((merged_df["api_name"] == "transaction_api").sum())
after_ma    = int((merged_df["api_name"] == "market_data_api").sum())
after_ta_pct = 100.0 * after_ta / max(1, after_total)
after_ma_pct = 100.0 * after_ma / max(1, after_total)

out()
out(f"  transaction_api after  : {after_ta:,}  ({after_ta_pct:.2f}%)")
out(f"  market_data_api after  : {after_ma:,}  ({after_ma_pct:.2f}%)")
out(f"  Total rows dropped     : {len(drop_indices):,}")

if after_ta_pct > 25.0:
    warn(f"transaction_api still {after_ta_pct:.1f}% — constrained by failure row count.")
if after_ma_pct > 25.0:
    warn(f"market_data_api still {after_ma_pct:.1f}% — constrained by failure row count.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Save and validate
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4 — Save and validate")

out(f"  Saving → {OUT_CSV_PATH} …")
merged_df.to_csv(OUT_CSV_PATH, index=False)
out(f"  Saved {len(merged_df):,} rows.")

# Final summary stats
total_rows   = len(merged_df)
fail_rate    = 100.0 * (merged_df["success"] == 0).sum() / max(1, total_rows)
null_et      = merged_df["error_type"].isna().sum()
ta_pct_final = 100.0 * (merged_df["api_name"] == "transaction_api").sum() / max(1, total_rows)

out()
out("  === Clean Dataset Summary ===")
out(f"  Total rows        : {total_rows:,}")
out(f"  Failure rate      : {fail_rate:.2f}%")
out("  Rows per source   :")
src_col = "data_source" if "data_source" in merged_df.columns else "api_name"
for src, cnt in merged_df[src_col].value_counts().items():
    out(f"    {str(src):<25}: {cnt:,}")
out(f"  Null error_type   : {null_et:,}  (should be 0)")
out(f"  transaction_api % : {ta_pct_final:.2f}%  (should be ≤25%)")

# ── Sequence boundary sanity check ────────────────────────────────────────────
section("Sequence Boundary Sanity Check  (cap=300,000 rows, seq_len=30)")

SEQ_LEN  = 30
ROW_CAP  = 300_000

out(f"  Sorting by (api_name, timestamp) …")
check_df = (
    merged_df[["api_name", src_col, "timestamp"]]
    .sort_values(["api_name", "timestamp"])
    .reset_index(drop=True)
    .iloc[:ROW_CAP]
)

src_arr    = check_df[src_col].to_numpy()
total_seqs = max(0, len(check_df) - SEQ_LEN + 1)
broken     = 0
step_size  = max(1, total_seqs // 10)

out(f"  Checking {len(check_df):,} rows / {total_seqs:,} sequences …")
for i in range(total_seqs):
    if i % step_size == 0:
        out(f"    … {100*i//total_seqs}%")
    if len(set(src_arr[i : i + SEQ_LEN])) > 1:
        broken += 1

broken_pct = 100.0 * broken / max(1, total_seqs)
out()
out(f"  Sequences checked  : {total_seqs:,}")
out(f"  Broken sequences   : {broken:,}  ({broken_pct:.2f}%)")
if broken_pct <= 1.0:
    out("  ✓ Boundary contamination acceptable (<1%)")
else:
    warn(f"{broken_pct:.2f}% boundary contamination — re-sort dataset by (api_name, timestamp) before training.")

# ── Save report ───────────────────────────────────────────────────────────────
out()
out("=" * 72)
out("  Fix complete. Original CSV preserved. Clean CSV ready for training.")
out("=" * 72)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(_buf.getvalue())
out(f"\nReport saved → {REPORT_PATH}")
