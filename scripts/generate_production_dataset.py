#!/usr/bin/env python3
"""
Production-grade synthetic banking API dataset generator.
Generates 1,000,000 rows (~2 years) across 5 API types with:

Real-world failure modes:
  - Market crash events
  - Regulatory reporting windows (quarter-end)
  - Flash crashes
  - DDoS attack simulation
  - Vendor outages
  - Holiday/weekend low-traffic patterns
  - Incorrect API permissions
  - Unsecured endpoints / token expiry
  - Insufficient API testing windows
  - Invalid session management
  - Expiring / deprecated APIs
  - Bad or outdated URLs
  - Overly complex API endpoints
  - APIs exposed on public IPs without protection
  - Poor API design / documentation gaps
  - Dependency failures from external services
  - Lack of monitoring and logging gaps

Storage: SQLite (production-like) + CSV export

Usage:
    python scripts/generate_production_dataset.py
    python scripts/generate_production_dataset.py --n_samples 500000 --seed 99
"""

import os
import time
import argparse
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

print("=== Production-Grade Banking API Dataset Generator ===\n")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, default=1_000_000)
parser.add_argument("--seed",      type=int, default=42)
parser.add_argument("--out_dir",   type=str, default="data")
args = parser.parse_args()

np.random.seed(args.seed)
N   = args.n_samples
OUT = args.out_dir
os.makedirs(OUT, exist_ok=True)

START_DATE = "2023-01-01"
END_DATE   = "2024-12-31"

API_TYPES = [
    "stock_price_api",
    "forex_api",
    "crypto_api",
    "market_data_api",
    "transaction_api",
]

# rows per API (evenly split)
N_PER_API = N // len(API_TYPES)

# ── Known real-world event windows ───────────────────────────────────────────
# Each entry: (label, start, end, error_rate_boost, response_time_mult)
GLOBAL_EVENTS = [
    # Market crashes (high volatility, high error rates)
    ("market_crash",      "2023-03-10", "2023-03-15", 0.45, 2.8),
    ("market_crash",      "2023-08-02", "2023-08-05", 0.38, 2.4),
    ("market_crash",      "2024-04-19", "2024-04-22", 0.42, 2.6),
    # Flash crashes (very short, extreme spikes)
    ("flash_crash",       "2023-05-24", "2023-05-24", 0.70, 4.5),
    ("flash_crash",       "2023-11-08", "2023-11-08", 0.65, 4.0),
    ("flash_crash",       "2024-07-15", "2024-07-15", 0.72, 5.0),
    # Quarter-end regulatory windows (high load, elevated errors)
    ("quarter_end",       "2023-03-29", "2023-03-31", 0.22, 1.8),
    ("quarter_end",       "2023-06-28", "2023-06-30", 0.20, 1.7),
    ("quarter_end",       "2023-09-27", "2023-09-29", 0.21, 1.75),
    ("quarter_end",       "2023-12-27", "2023-12-29", 0.23, 1.9),
    ("quarter_end",       "2024-03-27", "2024-03-29", 0.22, 1.8),
    ("quarter_end",       "2024-06-26", "2024-06-28", 0.20, 1.7),
    ("quarter_end",       "2024-09-25", "2024-09-27", 0.21, 1.75),
    ("quarter_end",       "2024-12-27", "2024-12-31", 0.24, 1.9),
    # DDoS attacks (extreme request spike + high error rate)
    ("ddos_attack",       "2023-07-04", "2023-07-05", 0.60, 3.5),
    ("ddos_attack",       "2024-02-14", "2024-02-15", 0.55, 3.2),
    ("ddos_attack",       "2024-10-31", "2024-11-01", 0.65, 4.0),
    # Vendor outages (specific APIs go down completely)
    ("vendor_outage",     "2023-04-18", "2023-04-19", 0.80, 5.0),
    ("vendor_outage",     "2023-10-03", "2023-10-04", 0.75, 4.5),
    ("vendor_outage",     "2024-01-22", "2024-01-23", 0.82, 5.5),
    ("vendor_outage",     "2024-08-07", "2024-08-08", 0.78, 5.0),
]

# API-specific failure modes (applied on top of global events)
API_SPECIFIC_EVENTS = {
    "stock_price_api": [
        ("deprecated_endpoint",   "2023-06-01", "2023-06-30", 0.15, 1.2),
        ("bad_url",               "2023-09-15", "2023-09-16", 0.50, 1.5),
        ("permission_error",      "2024-03-01", "2024-03-03", 0.35, 1.3),
    ],
    "forex_api": [
        ("session_expiry",        "2023-05-10", "2023-05-12", 0.40, 1.6),
        ("token_expiry",          "2023-11-20", "2023-11-21", 0.45, 1.4),
        ("public_ip_exposure",    "2024-06-10", "2024-06-11", 0.30, 1.2),
    ],
    "crypto_api": [
        ("dependency_failure",    "2023-02-15", "2023-02-17", 0.55, 2.0),
        ("complex_endpoint",      "2023-08-22", "2023-08-24", 0.25, 2.5),
        ("monitoring_gap",        "2024-05-05", "2024-05-07", 0.20, 1.3),
    ],
    "market_data_api": [
        ("poor_documentation",    "2023-03-01", "2023-04-01", 0.12, 1.1),
        ("insufficient_testing",  "2023-07-17", "2023-07-19", 0.30, 1.4),
        ("outdated_url",          "2024-02-28", "2024-03-01", 0.45, 1.5),
    ],
    "transaction_api": [
        ("invalid_session",       "2023-04-05", "2023-04-06", 0.50, 1.8),
        ("unsecured_endpoint",    "2023-12-01", "2023-12-03", 0.35, 1.3),
        ("dependency_failure",    "2024-09-10", "2024-09-12", 0.60, 2.2),
    ],
}

# UK/US public holidays (low traffic, slightly elevated errors)
HOLIDAYS = {
    date(2023, 1, 2), date(2023, 4, 7), date(2023, 4, 10),
    date(2023, 5, 1), date(2023, 5, 29), date(2023, 8, 28),
    date(2023, 11, 23), date(2023, 12, 25), date(2023, 12, 26),
    date(2024, 1, 1), date(2024, 3, 29), date(2024, 4, 1),
    date(2024, 5, 6), date(2024, 5, 27), date(2024, 8, 26),
    date(2024, 11, 28), date(2024, 12, 25), date(2024, 12, 26),
}

# Status code pools
SUCCESS_CODES = [200, 200, 200, 201, 202]
CLIENT_ERRORS = [400, 401, 403, 404, 405, 422, 429]
SERVER_ERRORS = [500, 502, 503, 504]
TIMEOUT_CODE  = 408

# Error type labels matching each failure mode
ERROR_LABELS = {
    "market_crash":        "market_volatility_overload",
    "flash_crash":         "flash_crash_timeout",
    "quarter_end":         "regulatory_load_spike",
    "ddos_attack":         "ddos_rate_limit_exceeded",
    "vendor_outage":       "vendor_upstream_failure",
    "deprecated_endpoint": "410_endpoint_deprecated",
    "bad_url":             "404_bad_or_outdated_url",
    "permission_error":    "403_incorrect_api_permission",
    "session_expiry":      "401_invalid_session",
    "token_expiry":        "401_token_expired",
    "public_ip_exposure":  "security_misconfiguration",
    "dependency_failure":  "503_external_dependency_down",
    "complex_endpoint":    "504_complex_endpoint_timeout",
    "monitoring_gap":      "silent_failure_no_alert",
    "poor_documentation":  "400_malformed_request",
    "insufficient_testing":"500_untested_code_path",
    "outdated_url":        "404_outdated_url",
    "invalid_session":     "401_invalid_session_token",
    "unsecured_endpoint":  "security_breach_attempt",
    "normal":              None,
}


def date_to_ts(d: str) -> pd.Timestamp:
    return pd.Timestamp(d)


def build_event_mask(timestamps: pd.DatetimeIndex, events: list):
    """Return array of (error_boost, rt_mult, event_label) per timestamp."""
    n       = len(timestamps)
    boost   = np.zeros(n)
    rt_mult = np.ones(n)
    labels  = np.array(["normal"] * n, dtype=object)

    for label, start, end, err_b, rt_m in events:
        mask = (timestamps >= date_to_ts(start)) & (timestamps <= date_to_ts(end))
        # later events override earlier (more specific wins)
        boost[mask]   = err_b
        rt_mult[mask] = rt_m
        labels[mask]  = label

    return boost, rt_mult, labels


def generate_api_telemetry(api_name: str, n: int) -> pd.DataFrame:
    print(f"  Generating {n:,} rows for {api_name} ...")
    t0 = time.time()

    timestamps = pd.date_range(START_DATE, END_DATE, periods=n)
    hour  = timestamps.hour.values
    dow   = timestamps.dayofweek.values
    dates = timestamps.date

    # ── Base patterns ─────────────────────────────────────────────────────────
    is_weekend = (dow >= 5)
    is_holiday = np.array([d in HOLIDAYS for d in dates])
    is_low_traffic = is_weekend | is_holiday
    is_peak    = ((hour >= 9) & (hour <= 16) & ~is_low_traffic)
    is_preopen = ((hour >= 7) & (hour < 9))
    is_close   = ((hour >= 16) & (hour <= 18))

    # Response time: base + time-of-day + API-specific complexity
    api_complexity = {
        "stock_price_api":  1.0,
        "forex_api":        1.1,
        "crypto_api":       1.3,   # highest complexity
        "market_data_api":  1.2,
        "transaction_api":  1.15,
    }[api_name]

    base_rt   = np.random.exponential(80 * api_complexity, n)
    hour_eff  = 35 * np.sin(2 * np.pi * hour / 24)
    noise     = np.random.normal(0, 15, n)
    # Random micro-spikes (network jitter)
    jitter    = np.where(np.random.random(n) < 0.02,
                         np.random.exponential(200, n), 0)
    rt_base   = np.maximum(8, base_rt + hour_eff + noise + jitter)

    # Base error probability
    base_err = np.where(is_low_traffic, 0.12,
               np.where(is_peak,        0.07,
               np.where(is_preopen,     0.10, 0.09)))

    # ── Apply global events ───────────────────────────────────────────────────
    g_boost, g_rt_mult, g_labels = build_event_mask(timestamps, GLOBAL_EVENTS)

    # ── Apply API-specific events ─────────────────────────────────────────────
    api_events = API_SPECIFIC_EVENTS.get(api_name, [])
    a_boost, a_rt_mult, a_labels = build_event_mask(timestamps, api_events)

    # Combine: API-specific overrides global where both apply
    combined_boost   = np.maximum(g_boost, a_boost)
    combined_rt_mult = np.maximum(g_rt_mult, a_rt_mult)
    event_labels     = np.where(a_labels != "normal", a_labels, g_labels)

    # Final error probability (clipped 0–0.97)
    err_prob = np.clip(base_err + combined_boost, 0, 0.97)

    # DDoS: inflate request counts massively
    is_ddos = (event_labels == "ddos_attack")
    base_req = np.random.poisson(
        np.where(is_peak, 60, np.where(is_low_traffic, 15, 35)), n
    )
    ddos_req = np.random.poisson(800, n)
    request_count = np.where(is_ddos, ddos_req, base_req)

    # ── Generate status codes ─────────────────────────────────────────────────
    rand = np.random.random(n)
    # Assign error type based on event label
    status_codes = np.empty(n, dtype=int)
    error_types  = np.empty(n, dtype=object)

    for i in range(n):
        if rand[i] < err_prob[i]:
            lbl = event_labels[i]
            # Map event to realistic status code
            if lbl in ("token_expiry", "session_expiry", "invalid_session"):
                sc = 401
            elif lbl in ("permission_error", "unsecured_endpoint",
                         "public_ip_exposure"):
                sc = 403
            elif lbl in ("bad_url", "outdated_url", "deprecated_endpoint"):
                sc = np.random.choice([404, 410])
            elif lbl in ("ddos_attack",):
                sc = 429
            elif lbl in ("poor_documentation",):
                sc = 400
            elif lbl in ("complex_endpoint",):
                sc = 504
            elif lbl in ("vendor_outage", "dependency_failure"):
                sc = 503
            elif lbl in ("flash_crash", "market_crash"):
                sc = np.random.choice([500, 502, 503, 504, 408])
            else:
                sc = np.random.choice(CLIENT_ERRORS + SERVER_ERRORS)
            status_codes[i] = sc
            error_types[i]  = ERROR_LABELS.get(lbl, "unknown_error")
        else:
            status_codes[i] = np.random.choice(SUCCESS_CODES)
            error_types[i]  = None

    success = (status_codes < 400).astype(int)

    # ── Failure clustering (failures beget nearby failures) ───────────────────
    fail_idx = np.where(success == 0)[0]
    for idx in fail_idx[::3]:   # sample every 3rd for speed
        spread = int(np.random.exponential(5))
        for offset in range(1, min(spread + 1, 20)):
            if idx + offset < n and np.random.random() < 0.3:
                success[idx + offset] = 0
                if status_codes[idx + offset] in SUCCESS_CODES:
                    status_codes[idx + offset] = np.random.choice(SERVER_ERRORS)
                    error_types[idx + offset]  = "cascading_failure"

    # ── Response time: apply multipliers + outage penalty ─────────────────────
    rt_final = rt_base * combined_rt_mult
    # Failed requests are slower
    rt_final = np.where(success == 0,
                        rt_final * np.random.uniform(1.5, 3.0, n),
                        rt_final)
    rt_final = np.maximum(5, rt_final)

    # ── Rolling features ──────────────────────────────────────────────────────
    rt_series  = pd.Series(rt_final)
    err_series = pd.Series(1 - success)

    rt_mean_60  = rt_series.rolling(60,  min_periods=1).mean().values
    rt_std_60   = rt_series.rolling(60,  min_periods=1).std().fillna(0).values
    err_rate_60 = err_series.rolling(60, min_periods=1).mean().values
    rt_var      = rt_std_60 ** 2
    err_vol     = err_series.rolling(30, min_periods=1).std().fillna(0).values

    # Lag features
    rt_lag1  = rt_series.shift(1).fillna(rt_series.mean()).values
    rt_lag5  = rt_series.shift(5).fillna(rt_series.mean()).values
    err_lag1 = err_series.shift(1).fillna(0).values

    # EMA features
    rt_ema10  = rt_series.ewm(span=10).mean().values
    rt_ema30  = rt_series.ewm(span=30).mean().values
    err_ema10 = err_series.ewm(span=10).mean().values

    # Cyclical time encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin  = np.sin(2 * np.pi * dow / 7)
    dow_cos  = np.cos(2 * np.pi * dow / 7)

    # API-specific flags
    is_high_freq = 1 if api_name == "crypto_api" else 0
    is_financial_peak = is_peak.astype(int)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        # identifiers
        "timestamp":          timestamps,
        "api_name":           api_name,
        # raw telemetry
        "response_time":      np.round(rt_final, 2),
        "status_code":        status_codes,
        "success":            success,
        "error_type":         error_types,
        "request_count":      request_count,
        # time features
        "hour":               hour,
        "day_of_week":        dow,
        "is_weekend":         is_weekend.astype(int),
        "is_holiday":         is_holiday.astype(int),
        "is_market_hours":    is_peak.astype(int),
        "is_financial_peak":  is_financial_peak,
        "is_pre_open":        is_preopen.astype(int),
        "is_market_close":    is_close.astype(int),
        # rolling features
        "response_time_rolling_mean": np.round(rt_mean_60, 4),
        "response_time_rolling_std":  np.round(rt_std_60,  4),
        "error_rate_rolling":         np.round(err_rate_60, 4),
        "response_time_variance":     np.round(rt_var,     4),
        "error_volatility":           np.round(err_vol,    4),
        # lag features
        "response_time_lag_1":  np.round(rt_lag1, 2),
        "response_time_lag_5":  np.round(rt_lag5, 2),
        "error_rate_lag_1":     np.round(err_lag1, 4),
        # EMA features
        "response_time_ema_10": np.round(rt_ema10,  4),
        "response_time_ema_30": np.round(rt_ema30,  4),
        "error_rate_ema_10":    np.round(err_ema10, 4),
        # cyclical encoding
        "hour_sin": np.round(hour_sin, 6),
        "hour_cos": np.round(hour_cos, 6),
        "dow_sin":  np.round(dow_sin,  6),
        "dow_cos":  np.round(dow_cos,  6),
        # API flags
        "high_frequency_api":  is_high_freq,
        "api_complexity":      api_complexity,
        # event metadata (for analysis / ground truth)
        "event_label":         event_labels,
        "error_rate_boost":    np.round(combined_boost,   4),
        "rt_multiplier":       np.round(combined_rt_mult, 4),
    })

    elapsed = time.time() - t0
    fail_rate = 1 - df["success"].mean()
    print(f"    ✅ Done in {elapsed:.1f}s  |  "
          f"failure rate: {fail_rate:.2%}  |  "
          f"unique events: {df['event_label'].nunique()}")
    return df


# ── Generate all APIs ─────────────────────────────────────────────────────────
print(f"Generating {N:,} rows across {len(API_TYPES)} APIs ...\n")
t_start = time.time()

all_dfs = []
for api in API_TYPES:
    df_api = generate_api_telemetry(api, N_PER_API)
    all_dfs.append(df_api)

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values("timestamp").reset_index(drop=True)

print(f"\n{'='*60}")
print(f"Total rows     : {len(df_all):,}")
print(f"Date range     : {df_all['timestamp'].min()} → {df_all['timestamp'].max()}")
print(f"Overall failure: {1 - df_all['success'].mean():.2%}")
print(f"APIs           : {df_all['api_name'].nunique()}")
print(f"Unique events  : {df_all['event_label'].nunique()}")
print(f"Columns        : {len(df_all.columns)}")
print(f"{'='*60}\n")

# ── Event summary ─────────────────────────────────────────────────────────────
print("Event distribution:")
ev = (df_all[df_all["event_label"] != "normal"]
      .groupby("event_label")
      .agg(rows=("success","count"),
           failure_rate=("success", lambda x: 1 - x.mean()),
           avg_rt=("response_time","mean"))
      .sort_values("rows", ascending=False))
print(ev.to_string())
print()

# ── Save to SQLite ────────────────────────────────────────────────────────────
db_path = os.path.join(OUT, "banking_api_telemetry.db")
print(f"Writing to SQLite: {db_path} ...")
t_db = time.time()

conn = sqlite3.connect(db_path)

# Main telemetry table
df_all.to_sql("api_telemetry", conn, if_exists="replace",
              index=True, index_label="id")

# ── Indexes for production-like query performance ─────────────────────────────
conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON api_telemetry(timestamp)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_api_name  ON api_telemetry(api_name)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_success   ON api_telemetry(success)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_event     ON api_telemetry(event_label)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_api_ts    ON api_telemetry(api_name, timestamp)")

# ── Summary stats view ────────────────────────────────────────────────────────
conn.execute("""
CREATE VIEW IF NOT EXISTS api_daily_summary AS
SELECT
    DATE(timestamp) AS date,
    api_name,
    COUNT(*)                                        AS total_requests,
    SUM(CASE WHEN success=0 THEN 1 ELSE 0 END)     AS total_failures,
    ROUND(AVG(CASE WHEN success=0 THEN 1.0 ELSE 0 END), 4) AS failure_rate,
    ROUND(AVG(response_time), 2)                    AS avg_response_time_ms,
    ROUND(MAX(response_time), 2)                    AS max_response_time_ms,
    event_label
FROM api_telemetry
GROUP BY DATE(timestamp), api_name, event_label
ORDER BY date, api_name
""")

# ── Event impact view ─────────────────────────────────────────────────────────
conn.execute("""
CREATE VIEW IF NOT EXISTS event_impact_summary AS
SELECT
    event_label,
    api_name,
    COUNT(*)                                               AS affected_rows,
    ROUND(AVG(CASE WHEN success=0 THEN 1.0 ELSE 0 END),4) AS failure_rate,
    ROUND(AVG(response_time),2)                            AS avg_response_time_ms,
    ROUND(AVG(error_rate_boost),4)                         AS avg_error_boost,
    MIN(timestamp)                                         AS event_start,
    MAX(timestamp)                                         AS event_end
FROM api_telemetry
WHERE event_label != 'normal'
GROUP BY event_label, api_name
ORDER BY failure_rate DESC
""")

conn.commit()
conn.close()

print(f"  ✅ SQLite saved in {time.time()-t_db:.1f}s")
print(f"  📁 Path: {db_path}")

# ── Save features CSV (LSTM-ready subset) ─────────────────────────────────────
LSTM_COLS = [
    "timestamp", "api_name",
    "response_time", "status_code", "success", "error_type",
    "request_count", "hour", "day_of_week", "is_weekend", "is_holiday",
    "is_market_hours", "is_financial_peak", "high_frequency_api",
    "response_time_rolling_mean", "response_time_rolling_std",
    "error_rate_rolling", "response_time_variance", "error_volatility",
    "response_time_lag_1", "response_time_lag_5", "error_rate_lag_1",
    "response_time_ema_10", "response_time_ema_30", "error_rate_ema_10",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "api_complexity",
    "error_rate_boost", "rt_multiplier",
    "event_label",
]
csv_path = os.path.join(OUT, "banking_api_features.csv")
df_all[LSTM_COLS].to_csv(csv_path, index=False)
print(f"  ✅ LSTM-ready CSV saved: {csv_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
total_elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  Generation complete in {total_elapsed/60:.1f} minutes")
print(f"  SQLite DB   : {db_path}  ({os.path.getsize(db_path)/1e6:.0f} MB)")
print(f"  Features CSV: {csv_path} ({os.path.getsize(csv_path)/1e6:.0f} MB)")
print(f"{'='*60}")
print("""
Next steps:
  1. Retrain LSTM:
       python scripts/run_lstm_training.py --n_samples 1000000 --epochs 20

  2. Evaluate:
       python scripts/evaluate_lstm.py

  3. Run demo notebook:
       jupyter notebook notebooks/api_failure_prediction_demo.ipynb

  4. Query the database directly:
       sqlite3 data/banking_api_telemetry.db
       > SELECT * FROM api_daily_summary LIMIT 20;
       > SELECT * FROM event_impact_summary;
""")