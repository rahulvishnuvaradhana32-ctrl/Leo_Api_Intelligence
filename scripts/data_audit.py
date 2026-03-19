#!/usr/bin/env python3
"""LEO API — Data Audit Script
Analyzes data/banking_api_features.csv and data/banking_api_telemetry.db.
Produces a console report and saves it to models/data_audit_report.txt.

NOTE: The CSV does not have a 'data_source' column; 'api_name' is used as
the source identifier (transaction_api, market_data_api, stock_price_api,
crypto_api, forex_api).  'error_type' is used as the failure mode column.

Usage:
    python scripts/data_audit.py
"""
import os
import sqlite3
import sys
from io import StringIO
import argparse
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")



parser = argparse.ArgumentParser(description="FCE Data Audit Script")
parser.add_argument("--input",  type=str, default=os.path.join("data", "banking_api_features.csv"),
                    help="Path to CSV to audit")
parser.add_argument("--output", type=str, default=os.path.join("models", "data_audit_report.txt"),
                    help="Path to save the audit report")
parser.add_argument("--db",     type=str, default=os.path.join("data", "banking_api_telemetry.db"),
                    help="Path to SQLite DB for cross-check")
args = parser.parse_args()

CSV_PATH = args.input
DB_PATH  = args.db
OUT_PATH = args.output

SOURCE_COL  = "api_name"       # plays the role of 'data_source'
FAILURE_COL = "error_type"     # plays the role of 'failure_event'
GROUP_COL   = "api_name"       # plays the role of 'api_type' for temporal check

KEY_FEATURES = [
    "error_rate_boost", "rt_multiplier",
    "latency_spike", "error_burst",
    "instability_index", "error_rate_rolling",
]

SEQ_LEN          = 30
SEQ_CHECK_CAP    = 500_000     # rows, for CPU tractability
SOURCE_LOW_PCT   = 5.0         # flag if below this %
SOURCE_HIGH_PCT  = 40.0        # flag if above this %
COVERAGE_THRESH  = 30.0        # flag source if <30% on 3+ key features
FAILURE_DOM_PCT  = 50.0        # flag if single failure type >50% within a source
FAILURE_MIN_RATE = 5.0         # flag if source failure rate <5%


# ── Output buffer ─────────────────────────────────────────────────────────────
_buf = StringIO()

def out(msg: str = "") -> None:
    print(msg)
    _buf.write(msg + "\n")

def section(title: str) -> None:
    line = "=" * 72
    out()
    out(line)
    out(f"  {title}")
    out(line)


# ── Load CSV ──────────────────────────────────────────────────────────────────
print(f"Loading {CSV_PATH} …", flush=True)
out(f"  Auditing file : {CSV_PATH}")
out(f"  Report output : {OUT_PATH}")
df = pd.read_csv(CSV_PATH, low_memory=False)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
total_rows = len(df)
print(f"  Loaded {total_rows:,} rows × {len(df.columns)} columns", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Source breakdown
# ═══════════════════════════════════════════════════════════════════════════════
section("1. SOURCE BREAKDOWN  (column: api_name)")

src_counts = df[SOURCE_COL].value_counts().sort_values(ascending=False)
src_flags: dict[str, list[str]] = {s: [] for s in src_counts.index}

out(f"  {'Source':<30} {'Rows':>10}  {'%':>7}  Flags")
out(f"  {'-'*30} {'-'*10}  {'-'*7}  {'-'*20}")
for src, cnt in src_counts.items():
    pct = 100.0 * cnt / total_rows
    flags = []
    if pct < SOURCE_LOW_PCT:
        flags.append("LOW (<5%) — potentially droppable")
    if pct > SOURCE_HIGH_PCT:
        flags.append("DOMINANT (>40%)")
    src_flags[src].extend(flags)
    flag_str = "; ".join(flags) if flags else "—"
    out(f"  {src:<30} {cnt:>10,}  {pct:>6.2f}%  {flag_str}")

out()
out(f"  Total rows: {total_rows:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Feature coverage per source
# ═══════════════════════════════════════════════════════════════════════════════
section("2. KEY FEATURE COVERAGE PER SOURCE  (non-zero & non-null %)")

present_feats = [f for f in KEY_FEATURES if f in df.columns]
absent_feats  = [f for f in KEY_FEATURES if f not in df.columns]
if absent_feats:
    out(f"  [!] Features absent from CSV (skipped): {absent_feats}")

out(f"  {'Source':<30} " + "  ".join(f"{f[:12]:>12}" for f in present_feats) + "  Flags")
out(f"  {'-'*30} " + "  ".join("-" * 12 for _ in present_feats) + "  ------")

low_coverage_sources = []
for src in src_counts.index:
    sub = df[df[SOURCE_COL] == src]
    coverage = []
    low_count = 0
    for feat in present_feats:
        if feat in sub.columns:
            nonzero_nonnull = ((sub[feat].notna()) & (sub[feat] != 0)).sum()
            pct = 100.0 * nonzero_nonnull / max(1, len(sub))
            coverage.append(pct)
            if pct < COVERAGE_THRESH:
                low_count += 1
        else:
            coverage.append(float("nan"))

    flag = f"LOW COVERAGE on {low_count} features — likely noise" if low_count >= 3 else "—"
    if low_count >= 3:
        low_coverage_sources.append(src)
        src_flags[src].append(f"Low feature coverage ({low_count}/6 key features <30%)")

    cov_str = "  ".join(f"{c:>11.1f}%" for c in coverage)
    out(f"  {src:<30} {cov_str}  {flag}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Failure mode distribution per source
# ═══════════════════════════════════════════════════════════════════════════════
section("3. FAILURE MODE DISTRIBUTION PER SOURCE  (column: error_type)")

print("  Computing failure distributions …", flush=True)

if FAILURE_COL not in df.columns:
    out(f"  [!] Column '{FAILURE_COL}' not found in CSV — skipping section 3.")
else:
    low_signal_sources = []
    dominant_mode_sources = []

    for src in src_counts.index:
        sub = df[df[SOURCE_COL] == src]
        n_total    = len(sub)
        n_failures = (sub["success"] == 0).sum()
        fail_rate  = 100.0 * n_failures / max(1, n_total)

        out()
        out(f"  [{src}]  total={n_total:,}  failures={n_failures:,}  "
            f"failure_rate={fail_rate:.2f}%")

        if fail_rate < FAILURE_MIN_RATE:
            out(f"    ⚠ FAILURE RATE BELOW {FAILURE_MIN_RATE}% — not contributing failure signal")
            low_signal_sources.append(src)
            src_flags[src].append(f"Low failure rate ({fail_rate:.1f}% < {FAILURE_MIN_RATE}%)")

        fail_sub = sub[sub["success"] == 0]
        if n_failures == 0:
            out("    (no failures)")
            continue

        mode_counts = fail_sub[FAILURE_COL].value_counts(dropna=False)
        for mode, cnt in mode_counts.items():
            mode_pct = 100.0 * cnt / n_failures
            label    = str(mode) if pd.notna(mode) else "(null/unknown)"
            dom_flag = " ⚠ DOMINANT (>50%)" if mode_pct > FAILURE_DOM_PCT else ""
            out(f"    {label:<45} {cnt:>7,}  {mode_pct:>6.2f}%{dom_flag}")
            if mode_pct > FAILURE_DOM_PCT and src not in dominant_mode_sources:
                dominant_mode_sources.append(src)
                src_flags[src].append(
                    f"Single failure mode '{mode}' dominates ({mode_pct:.1f}% of failures)"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Temporal coherence check
# ═══════════════════════════════════════════════════════════════════════════════
section("4. TEMPORAL COHERENCE — LSTM SEQUENCE BOUNDARY CHECK")
out(f"  Sequence length: {SEQ_LEN}  |  Cap: {SEQ_CHECK_CAP:,} rows  |  Group: {GROUP_COL}")

print(f"  Sorting by {GROUP_COL} + timestamp …", flush=True)
df_sorted = (
    df[[SOURCE_COL, "timestamp"]]
    .sort_values([SOURCE_COL, "timestamp"])
    .reset_index(drop=True)
)

cap       = min(SEQ_CHECK_CAP, len(df_sorted))
df_cap    = df_sorted.iloc[:cap].reset_index(drop=True)
src_arr   = df_cap[SOURCE_COL].to_numpy()

print(f"  Checking {cap:,} rows for cross-source sequences …", flush=True)

broken   = 0
total_seqs = cap - SEQ_LEN + 1
step = max(1, total_seqs // 10)

for i in range(total_seqs):
    if i % step == 0:
        print(f"    … {100*i//total_seqs}% complete", flush=True)
    window = src_arr[i : i + SEQ_LEN]
    if len(set(window)) > 1:
        broken += 1

broken_pct = 100.0 * broken / max(1, total_seqs)
out()
out(f"  Rows checked      : {cap:,}")
out(f"  Total sequences   : {total_seqs:,}")
out(f"  Broken sequences  : {broken:,}  ({broken_pct:.2f}%)")

if broken_pct > 10:
    out(f"  ⚠ >10% broken sequences — cross-source boundary contamination is HIGH")
    boundary_flag = True
elif broken_pct > 1:
    out(f"  ⚠ {broken_pct:.1f}% broken sequences — moderate boundary contamination")
    boundary_flag = True
else:
    out(f"  ✓ Boundary contamination is low (<1%)")
    boundary_flag = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SQLite cross-check
# ═══════════════════════════════════════════════════════════════════════════════
section("5. SQLITE CROSS-CHECK  (data/banking_api_telemetry.db)")

try:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM api_telemetry")
    db_total = cur.fetchone()[0]
    cur.execute("SELECT api_name, COUNT(*) FROM api_telemetry GROUP BY api_name ORDER BY 2 DESC")
    db_rows = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM api_telemetry WHERE success=0")
    db_fail = cur.fetchone()[0]
    con.close()

    out(f"  DB total rows : {db_total:,}")
    out(f"  CSV total rows: {total_rows:,}")
    delta = total_rows - db_total
    if abs(delta) > 0:
        out(f"  Delta (CSV-DB): {delta:+,} rows")
    out()
    out(f"  {'API':<30} {'DB rows':>10}")
    out(f"  {'-'*30} {'-'*10}")
    for api, cnt in db_rows:
        out(f"  {str(api):<30} {cnt:>10,}")
    out()
    out(f"  DB failures   : {db_fail:,}  ({100*db_fail/max(1,db_total):.2f}%)")
except Exception as e:
    out(f"  [!] Could not read SQLite DB: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDED ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
section("RECOMMENDED ACTIONS")

actions = []

# Source-level flags
for src, flags in src_flags.items():
    for flag in flags:
        if "LOW (<5%)" in flag:
            actions.append(
                f"DROP source '{src}': rows below 5% threshold — insufficient for stable training."
            )
        if "Low feature coverage" in flag:
            actions.append(
                f"DROP source '{src}': 3+ key features have <30% coverage — rows are noise."
            )
        if "Low failure rate" in flag:
            actions.append(
                f"CONSIDER DROPPING '{src}': <{FAILURE_MIN_RATE}% failure rate adds minimal failure signal."
            )
        if "dominates" in flag:
            actions.append(
                f"REVIEW '{src}': single failure mode >50% — augment with other failure types "
                f"or the model will over-specialise."
            )
        if "DOMINANT (>40%)" in flag:
            actions.append(
                f"DOWNSAMPLE '{src}': represents >40% of all rows — consider capping at "
                f"400k rows to prevent training bias."
            )

# Boundary sequences
if boundary_flag:
    actions.append(
        f"FIX BOUNDARY SEQUENCES: {broken:,} sequences ({broken_pct:.1f}%) cross api_name "
        f"boundaries. Sort by api_name+timestamp and split the dataset per API before "
        f"building LSTM sequences, then concatenate. This alone can improve val AUC by "
        f"removing contaminated training examples."
    )

if not actions:
    actions.append("No critical issues found. Dataset appears clean for training.")

for i, action in enumerate(actions, 1):
    out(f"  {i}. {action}")

out()
out("=" * 72)
out("  Report complete.")
out("=" * 72)


# ── Save to file ──────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(_buf.getvalue())

print(f"\nReport saved → {OUT_PATH}", flush=True)
