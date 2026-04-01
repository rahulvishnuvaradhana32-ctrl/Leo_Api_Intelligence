#!/usr/bin/env python3
"""LEO API Intelligence — New Kaggle Data Integration (v7)

Downloads real-world Kaggle datasets, maps them to the LEO API v6 schema,
adds synthetic hard-failure rows for the two missed failure types, and
produces data/banking_api_features_v7.csv.

Target: push PR-AUC from 41% toward 70%+ by adding 300k-500k rows covering
        network_attack_detected and fraud_transaction_failure patterns.

Usage:
    # Test without Kaggle download:
    python scripts/integrate_new_kaggle_data.py --synthetic_only

    # Full run (requires ~/.kaggle/kaggle.json):
    python scripts/integrate_new_kaggle_data.py

    # Skip download if files already downloaded:
    python scripts/integrate_new_kaggle_data.py --skip_download
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA       = ROOT / "data"
RAW_DIR    = DATA / "raw_kaggle"
MODELS_DIR = ROOT / "models"

# ── 5 API names used across the project ────────────────────────────────────────
API_NAMES = [
    "market_data_api",
    "crypto_api",
    "transaction_api",
    "forex_api",
    "stock_price_api",
]

# ── Full v6 schema column order ────────────────────────────────────────────────
V6_COLS = [
    "timestamp", "api_name", "response_time", "status_code", "success",
    "error_type", "request_count", "hour", "day_of_week", "is_weekend",
    "is_holiday", "is_market_hours", "is_financial_peak", "high_frequency_api",
    "response_time_rolling_mean", "response_time_rolling_std", "error_rate_rolling",
    "response_time_variance", "error_volatility",
    "response_time_lag_1", "response_time_lag_5", "error_rate_lag_1",
    "response_time_ema_10", "response_time_ema_30", "error_rate_ema_10",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "api_complexity", "error_rate_boost", "rt_multiplier",
    "event_label", "latency_diff_1", "latency_diff_5",
    "error_rate_diff_1", "error_rate_diff_5", "latency_spike", "error_burst",
    "instability_index", "latency_slope", "error_slope",
    "traffic_change", "burst_ratio",
    "data_source",
    "avg_error_rate_others", "max_error_rate_others", "n_apis_elevated",
    "corr_with_similar_api", "systemic_stress_index",
]

# ── Kaggle datasets to download ────────────────────────────────────────────────
KAGGLE_DATASETS = [
    "mlg-ulb/creditcardfraud",
    "hassan06/nslkdd",
    "jsrojas/ip-network-traffic-flows-labeled-with-87-apps",
    "crawford/kddcup99",
]


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Kaggle API setup check
# ══════════════════════════════════════════════════════════════════════════════

def check_kaggle_setup() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: Kaggle API not configured.")
        print("Fix:")
        print("  1. Go to kaggle.com → Settings → API → Create New Token")
        print("  2. Download kaggle.json")
        print(f"  3. Copy to {kaggle_json}")
        print("  4. On Linux/Mac run: chmod 600 ~/.kaggle/kaggle.json")
        print("\nAlternatively run with --synthetic_only to skip Kaggle download")
        sys.exit(1)
    print("Kaggle API configured ✓")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Download Kaggle datasets
# ══════════════════════════════════════════════════════════════════════════════

def download_datasets(skip_if_exists: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ds in KAGGLE_DATASETS:
        slug  = ds.split("/")[1]
        dest  = RAW_DIR / slug
        if skip_if_exists and dest.exists():
            print(f"  [skip] {slug} — already downloaded")
            continue
        print(f"  Downloading {ds} → {dest} ...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", ds,
                 "--unzip", "-p", str(dest)],
                check=True,
            )
            print(f"  ✓ {slug}")
        except subprocess.CalledProcessError as e:
            print(f"  [warn] Failed to download {ds}: {e}")
        except FileNotFoundError:
            print("  [error] kaggle CLI not found — run: pip install kaggle")
            sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all temporal features from the timestamp column."""
    ts = pd.to_datetime(df["timestamp"])
    df["hour"]           = ts.dt.hour.astype(np.int8)
    df["day_of_week"]    = ts.dt.dayofweek.astype(np.int8)
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_holiday"]     = np.int8(0)
    df["is_market_hours"]   = ((df["hour"] >= 9) & (df["hour"] <= 17) &
                               (df["is_weekend"] == 0)).astype(np.int8)
    df["is_financial_peak"] = (df["hour"].isin([9, 10, 14, 15])).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7.0).astype(np.float32)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7.0).astype(np.float32)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling, lag, and EMA features per api_name group."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    rt  = df["response_time"].astype(float)
    suc = df["success"].astype(float)
    err = 1.0 - suc

    # Rolling (window=20)
    df["response_time_rolling_mean"] = rt.rolling(20, min_periods=1).mean().astype(np.float32)
    df["response_time_rolling_std"]  = rt.rolling(20, min_periods=1).std().fillna(0).astype(np.float32)
    df["response_time_variance"]     = df["response_time_rolling_std"] ** 2
    df["error_rate_rolling"]         = err.rolling(20, min_periods=1).mean().astype(np.float32)
    df["error_volatility"]           = err.rolling(20, min_periods=1).std().fillna(0).astype(np.float32)

    # Lag features
    df["response_time_lag_1"] = rt.shift(1).fillna(rt.mean()).astype(np.float32)
    df["response_time_lag_5"] = rt.shift(5).fillna(rt.mean()).astype(np.float32)
    df["error_rate_lag_1"]    = err.shift(1).fillna(0).astype(np.float32)

    # EMA features
    df["response_time_ema_10"] = rt.ewm(span=10, adjust=False).mean().astype(np.float32)
    df["response_time_ema_30"] = rt.ewm(span=30, adjust=False).mean().astype(np.float32)
    df["error_rate_ema_10"]    = err.ewm(span=10, adjust=False).mean().astype(np.float32)

    # Precursor signals
    EPS = 1e-6
    df["latency_diff_1"]    = (rt - df["response_time_lag_1"]).astype(np.float32)
    df["latency_diff_5"]    = (rt - df["response_time_lag_5"]).astype(np.float32)
    df["error_rate_diff_1"] = (df["error_rate_rolling"] - df["error_rate_lag_1"]).astype(np.float32)
    df["error_rate_diff_5"] = (df["error_rate_rolling"] - df["error_rate_ema_10"]).astype(np.float32)
    df["latency_spike"]     = (rt / (df["response_time_rolling_mean"] + EPS)).astype(np.float32)
    df["error_burst"]       = (df["error_rate_rolling"] / (df["error_rate_ema_10"] + EPS)).astype(np.float32)
    df["instability_index"] = (df["latency_diff_1"].abs() + df["error_rate_diff_1"].abs()).astype(np.float32)
    df["latency_slope"]     = ((df["response_time_ema_10"] - df["response_time_ema_30"]) / 20.0).astype(np.float32)
    df["error_slope"]       = ((df["error_rate_ema_10"] - df["error_rate_lag_1"]) / 10.0).astype(np.float32)

    # Traffic change and burst ratio (set defaults; callers can override burst_ratio)
    if "traffic_change" not in df.columns:
        df["traffic_change"] = np.float32(0)
    if "burst_ratio" not in df.columns:
        df["burst_ratio"] = np.float32(0)

    return df


def _set_api_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """Assign api_complexity and high_frequency_api based on api_name."""
    complexity_map = {
        "market_data_api": 0.7,
        "crypto_api":      0.9,
        "transaction_api": 0.8,
        "forex_api":       0.6,
        "stock_price_api": 0.4,
    }
    hf_map = {
        "market_data_api": 1,
        "crypto_api":      1,
        "transaction_api": 0,
        "forex_api":       0,
        "stock_price_api": 1,
    }
    df["api_complexity"]    = df["api_name"].map(complexity_map).fillna(0.5).astype(np.float32)
    df["high_frequency_api"]= df["api_name"].map(hf_map).fillna(0).astype(np.int8)
    return df


def _fill_schema_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any remaining v6 schema columns with safe defaults."""
    defaults = {
        "event_label":            "",
        "error_rate_boost":       np.float32(0),
        "rt_multiplier":          np.float32(1),
        "data_source":            "kaggle",
        "avg_error_rate_others":  np.float32(0),
        "max_error_rate_others":  np.float32(0),
        "n_apis_elevated":        np.int8(1),
        "corr_with_similar_api":  np.float32(0),
        "systemic_stress_index":  np.float32(0),
        "traffic_change":         np.float32(0),
        "burst_ratio":            np.float32(0),
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 2a — Map Credit Card Fraud dataset
# ══════════════════════════════════════════════════════════════════════════════

def map_creditcard_fraud(raw_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = raw_dir / "creditcardfraud" / "creditcard.csv"
    if not csv_path.exists():
        print(f"  [skip] creditcard.csv not found at {csv_path}")
        return None

    print(f"  Loading credit card fraud ({csv_path}) ...")
    raw = pd.read_csv(csv_path)
    print(f"    Raw rows: {len(raw):,}  fraud: {int(raw['Class'].sum()):,}")

    rng = np.random.default_rng(42)
    n   = len(raw)

    # Timestamps: start 2024-01-01, 1-minute intervals
    base_ts = pd.Timestamp("2024-01-01")
    timestamps = pd.date_range(base_ts, periods=n, freq="1min")

    # Hours skewed toward fraud hours for fraud rows; normal hours for others
    fraud_mask    = raw["Class"].values == 1
    hours         = np.where(
        fraud_mask,
        rng.choice([0, 1, 2, 22, 23], size=n),
        rng.integers(6, 22, size=n),
    )
    adjusted_ts = pd.to_datetime([
        ts.replace(hour=int(h)) for ts, h in zip(timestamps, hours)
    ])

    # MinMax scale V1 → burst_ratio (0-5), V3 → systemic_stress_index (0-1)
    v1_scaled = MinMaxScaler(feature_range=(0, 5)).fit_transform(
        raw[["V1"]].values
    ).flatten()
    v3_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        raw[["V3"]].values
    ).flatten()

    df = pd.DataFrame()
    df["timestamp"]           = adjusted_ts
    df["api_name"]            = "transaction_api"
    df["response_time"]       = np.clip(raw["Amount"].values / 100.0, 0.1, 30.0).astype(np.float32)
    df["request_count"]       = rng.integers(10, 500, size=n).astype(np.int32)
    df["success"]             = (1 - raw["Class"].values).astype(np.int8)
    df["error_type"]          = np.where(fraud_mask, "fraud_transaction_failure", None)
    df["status_code"]         = np.where(fraud_mask, 402, 200).astype(np.int16)
    df["is_market_hours"]     = np.int8(0)          # fraud peaks at night
    df["burst_ratio"]         = v1_scaled.astype(np.float32)
    df["systemic_stress_index"] = v3_scaled.astype(np.float32)
    df["n_apis_elevated"]     = np.int8(1)           # isolated to transaction_api
    df["corr_with_similar_api"] = np.where(fraud_mask, 0.15, 0.6).astype(np.float32)
    df["data_source"]         = "kaggle_creditcard"

    # error_rate_rolling from raw Class column (before adding time features)
    df["error_rate_rolling_raw"] = (
        raw["Class"].astype(float).rolling(20, min_periods=1).mean().values
    )

    df = _add_time_features(df)
    # Override is_market_hours after time-feature derivation
    df["is_market_hours"] = np.int8(0)

    df = _add_rolling_features(df)
    # Restore the fraud-specific rolling error rate
    df["error_rate_rolling"] = df["error_rate_rolling_raw"].astype(np.float32)
    df.drop(columns=["error_rate_rolling_raw"], inplace=True)

    df = _set_api_complexity(df)
    df = _fill_schema_defaults(df)

    print(f"    Mapped rows: {len(df):,}  failure rate: "
          f"{(1-df['success'].mean())*100:.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 2b — Map NSL-KDD dataset
# ══════════════════════════════════════════════════════════════════════════════

NSL_KDD_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

ATTACK_CATEGORIES = {
    "normal": ("normal", 1),
    # DoS attacks
    "back": ("dos", 0), "land": ("dos", 0), "neptune": ("dos", 0),
    "pod": ("dos", 0), "smurf": ("dos", 0), "teardrop": ("dos", 0),
    "mailbomb": ("dos", 0), "apache2": ("dos", 0), "processtable": ("dos", 0),
    "udpstorm": ("dos", 0),
    # Probe attacks
    "ipsweep": ("probe", 0), "nmap": ("probe", 0), "portsweep": ("probe", 0),
    "satan": ("probe", 0), "mscan": ("probe", 0), "saint": ("probe", 0),
    # R2L attacks
    "ftp_write": ("r2l", 0), "guess_passwd": ("r2l", 0), "imap": ("r2l", 0),
    "multihop": ("r2l", 0), "phf": ("r2l", 0), "spy": ("r2l", 0),
    "warezclient": ("r2l", 0), "warezmaster": ("r2l", 0),
    # U2R attacks
    "buffer_overflow": ("u2r", 0), "loadmodule": ("u2r", 0),
    "perl": ("u2r", 0), "rootkit": ("u2r", 0),
}


def _map_attack_label(label: str):
    lbl = label.strip().lower().rstrip(".")
    entry = ATTACK_CATEGORIES.get(lbl)
    if entry:
        return entry
    return ("network_attack_detected", 0)


def _load_nsl_kdd(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Try several common filenames for NSL-KDD."""
    candidates = [
        raw_dir / "nslkdd" / "KDDTrain+.txt",
        raw_dir / "nslkdd" / "KDDTrain+.csv",
        raw_dir / "nslkdd" / "KDDTest+.txt",
        raw_dir / "nslkdd" / "KDDTest+.csv",
        raw_dir / "nslkdd" / "NSL-KDD" / "KDDTrain+.txt",
    ]
    for p in candidates:
        if p.exists():
            print(f"  Found NSL-KDD at {p}")
            try:
                return pd.read_csv(p, header=None, names=NSL_KDD_COLS, low_memory=False)
            except Exception as e:
                print(f"    [warn] could not read {p}: {e}")
    return None


def _load_kdd99(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Try KDD Cup 99 CSV."""
    candidates = [
        raw_dir / "kddcup99" / "kddcup.data_10_percent_corrected",
        raw_dir / "kddcup99" / "kddcup.data_10_percent.gz",
        raw_dir / "kddcup99" / "kddcup99.csv",
        raw_dir / "kddcup99" / "kddcup.data.gz",
    ]
    kdd99_cols = NSL_KDD_COLS[:-1]  # no 'difficulty' column
    for p in candidates:
        if p.exists():
            print(f"  Found KDD99 at {p}")
            try:
                if str(p).endswith(".gz"):
                    return pd.read_csv(p, compression="gzip", header=None,
                                       names=kdd99_cols, low_memory=False)
                return pd.read_csv(p, header=None, names=kdd99_cols, low_memory=False)
            except Exception as e:
                print(f"    [warn] could not read {p}: {e}")
    return None


def map_network_intrusion(raw_dir: Path) -> Optional[pd.DataFrame]:
    rng = np.random.default_rng(43)

    # Try NSL-KDD first, then KDD99
    raw = _load_nsl_kdd(raw_dir)
    source_name = "kaggle_nslkdd"
    if raw is None:
        raw = _load_kdd99(raw_dir)
        source_name = "kaggle_kdd99"
    if raw is None:
        print("  [skip] No NSL-KDD or KDD99 data found.")
        return None

    print(f"    Raw rows: {len(raw):,}")

    # Parse labels
    label_col = "label" if "label" in raw.columns else raw.columns[-1]
    labels    = raw[label_col].astype(str).str.strip().str.lower().str.rstrip(".")
    attack_info = labels.map(lambda l: _map_attack_label(l))

    error_types = attack_info.map(lambda x: None if x[0] == "normal" else "network_attack_detected")
    success_vals = attack_info.map(lambda x: x[1]).astype(np.int8)
    attack_mask  = (success_vals == 0).values

    n = len(raw)
    base_ts = pd.Timestamp("2024-02-01")
    timestamps = pd.date_range(base_ts, periods=n, freq="1min")

    # Duration → response_time
    if "duration" in raw.columns:
        rt = np.clip(raw["duration"].astype(float).fillna(1.0), 0.1, 10.0)
    else:
        rt = rng.uniform(0.1, 10.0, size=n)

    # src_bytes → request_count
    if "src_bytes" in raw.columns:
        rc = np.clip(raw["src_bytes"].astype(float).fillna(1000) / 1000, 1, 10000)
    else:
        rc = rng.integers(1, 1000, size=n)

    # num_failed_logins → burst_ratio
    if "num_failed_logins" in raw.columns:
        burst = np.clip(raw["num_failed_logins"].astype(float).fillna(0) * 2, 0, 8)
    else:
        burst = rng.uniform(0, 8, size=n)

    # api_name: attacks spread across all APIs, normal traffic localized
    api_weights_attack = [0.2, 0.2, 0.2, 0.2, 0.2]
    api_weights_normal = [0.3, 0.2, 0.2, 0.2, 0.1]
    api_names = np.where(
        attack_mask,
        rng.choice(API_NAMES, size=n, p=api_weights_attack),
        rng.choice(API_NAMES, size=n, p=api_weights_normal),
    )

    # n_apis_elevated and systemic_stress_index
    n_apis_elev  = np.where(attack_mask, rng.integers(3, 6, size=n), np.int8(1))
    sys_stress   = np.where(
        attack_mask,
        rng.uniform(0.8, 1.0, size=n),
        rng.uniform(0.0, 0.2, size=n),
    )

    df = pd.DataFrame()
    df["timestamp"]             = timestamps
    df["api_name"]              = api_names
    df["response_time"]         = rt.astype(np.float32)
    df["request_count"]         = rc.astype(np.int32)
    df["success"]               = success_vals
    df["error_type"]            = error_types.values
    df["status_code"]           = np.where(attack_mask, 503, 200).astype(np.int16)
    df["burst_ratio"]           = burst.astype(np.float32)
    df["n_apis_elevated"]       = n_apis_elev.astype(np.int8)
    df["systemic_stress_index"] = sys_stress.astype(np.float32)
    df["corr_with_similar_api"] = np.where(attack_mask,
                                           rng.uniform(0.5, 1.0, size=n),
                                           rng.uniform(0.1, 0.4, size=n)).astype(np.float32)
    df["data_source"]           = source_name

    df = _add_time_features(df)
    df = _add_rolling_features(df)
    df = _set_api_complexity(df)
    df = _fill_schema_defaults(df)

    print(f"    Mapped rows: {len(df):,}  attack rate: "
          f"{attack_mask.mean()*100:.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 2c — Map network traffic flow dataset (IP traffic)
# ══════════════════════════════════════════════════════════════════════════════

def map_network_traffic(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Map the IP network traffic flow dataset to project schema."""
    # Find any CSV in the ip-network-traffic directory
    traffic_dir = raw_dir / "ip-network-traffic-flows-labeled-with-87-apps"
    if not traffic_dir.exists():
        print("  [skip] IP network traffic dataset not found")
        return None

    csvs = list(traffic_dir.glob("*.csv"))
    if not csvs:
        print(f"  [skip] No CSVs found in {traffic_dir}")
        return None

    # Use the first (largest) CSV file
    csv_path = sorted(csvs, key=lambda p: p.stat().st_size, reverse=True)[0]
    print(f"  Loading network traffic ({csv_path.name}) ...")
    try:
        raw = pd.read_csv(csv_path, low_memory=False, nrows=200_000)
    except Exception as e:
        print(f"  [warn] Could not read {csv_path}: {e}")
        return None

    print(f"    Raw rows: {len(raw):,}")
    rng = np.random.default_rng(44)
    n   = len(raw)

    # Look for a label column
    label_col = None
    for c in raw.columns:
        if c.lower() in ("label", "class", "category", "app_name", "application"):
            label_col = c
            break

    if label_col:
        labels      = raw[label_col].astype(str).str.lower()
        attack_mask = labels.str.contains("attack|anomal|malici|dos|intrusion",
                                          regex=True, na=False).values
    else:
        # No label column — treat all as normal traffic
        attack_mask = np.zeros(n, dtype=bool)

    # Flow duration → response_time
    dur_col = next((c for c in raw.columns
                    if "duration" in c.lower() or "flow_dur" in c.lower()), None)
    if dur_col:
        rt = np.clip(raw[dur_col].astype(float).fillna(1.0) / 1e6, 0.1, 10.0)
    else:
        rt = rng.uniform(0.1, 5.0, size=n)

    # Packet count → request_count
    pkt_col = next((c for c in raw.columns
                    if "pkt" in c.lower() or "packet" in c.lower()), None)
    if pkt_col:
        rc = np.clip(raw[pkt_col].astype(float).fillna(100), 1, 10000)
    else:
        rc = rng.integers(10, 500, size=n)

    base_ts    = pd.Timestamp("2024-03-01")
    timestamps = pd.date_range(base_ts, periods=n, freq="1min")

    df = pd.DataFrame()
    df["timestamp"]             = timestamps
    df["api_name"]              = rng.choice(API_NAMES, size=n)
    df["response_time"]         = rt.astype(np.float32)
    df["request_count"]         = rc.astype(np.int32)
    df["success"]               = (~attack_mask).astype(np.int8)
    df["error_type"]            = np.where(attack_mask, "network_attack_detected", None)
    df["status_code"]           = np.where(attack_mask, 503, 200).astype(np.int16)
    df["n_apis_elevated"]       = np.where(attack_mask,
                                           rng.integers(3, 6, size=n),
                                           np.int8(1)).astype(np.int8)
    df["systemic_stress_index"] = np.where(attack_mask,
                                           rng.uniform(0.7, 1.0, size=n),
                                           rng.uniform(0.0, 0.3, size=n)).astype(np.float32)
    df["burst_ratio"]           = np.where(attack_mask,
                                           rng.uniform(3.0, 8.0, size=n),
                                           rng.uniform(0.0, 1.0, size=n)).astype(np.float32)
    df["corr_with_similar_api"] = rng.uniform(0, 1, size=n).astype(np.float32)
    df["data_source"]           = "kaggle_network_traffic"

    df = _add_time_features(df)
    df = _add_rolling_features(df)
    df = _set_api_complexity(df)
    df = _fill_schema_defaults(df)

    print(f"    Mapped rows: {len(df):,}  anomaly rate: "
          f"{attack_mask.mean()*100:.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Synthetic hard failure rows
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_failures(n: int = 100_000) -> pd.DataFrame:
    """Generate synthetic rows for the two under-represented failure types."""
    rng = np.random.default_rng(99)

    # Split roughly 50/50 between the two failure types
    n_attack = n // 2
    n_fraud  = n - n_attack

    # ── Network attack rows ────────────────────────────────────────────────
    atk_ts = pd.date_range("2024-04-01", periods=n_attack, freq="1min")
    atk = pd.DataFrame()
    atk["timestamp"]             = atk_ts
    atk["api_name"]              = rng.choice(API_NAMES, size=n_attack)
    atk["response_time"]         = rng.uniform(2.0, 10.0, size=n_attack).astype(np.float32)
    atk["request_count"]         = rng.integers(1001, 5000, size=n_attack).astype(np.int32)
    atk["success"]               = np.int8(0)
    atk["error_type"]            = "network_attack_detected"
    atk["status_code"]           = np.int16(503)
    atk["burst_ratio"]           = rng.uniform(5.1, 10.0, size=n_attack).astype(np.float32)
    atk["n_apis_elevated"]       = rng.integers(3, 6, size=n_attack).astype(np.int8)
    atk["systemic_stress_index"] = rng.uniform(0.76, 1.0, size=n_attack).astype(np.float32)
    atk["error_rate_rolling"]    = rng.uniform(0.61, 1.0, size=n_attack).astype(np.float32)
    atk["corr_with_similar_api"] = rng.uniform(0.6, 1.0, size=n_attack).astype(np.float32)
    atk["avg_error_rate_others"] = rng.uniform(0.4, 0.9, size=n_attack).astype(np.float32)
    atk["max_error_rate_others"] = rng.uniform(0.6, 1.0, size=n_attack).astype(np.float32)
    atk["data_source"]           = "synthetic_attack"

    # ── Fraud rows ─────────────────────────────────────────────────────────
    fraud_hours = rng.choice([0, 1, 2, 3, 22, 23], size=n_fraud)
    base_fraud  = pd.Timestamp("2024-05-01")
    fraud_ts    = pd.to_datetime([
        (base_fraud + pd.Timedelta(minutes=int(i))).replace(hour=int(h))
        for i, h in enumerate(fraud_hours)
    ])
    frd = pd.DataFrame()
    frd["timestamp"]             = fraud_ts
    frd["api_name"]              = "transaction_api"
    frd["response_time"]         = rng.uniform(5.1, 30.0, size=n_fraud).astype(np.float32)
    frd["request_count"]         = rng.integers(10, 200, size=n_fraud).astype(np.int32)
    frd["success"]               = np.int8(0)
    frd["error_type"]            = "fraud_transaction_failure"
    frd["status_code"]           = np.int16(402)
    frd["n_apis_elevated"]       = np.int8(1)
    frd["corr_with_similar_api"] = rng.uniform(0.0, 0.29, size=n_fraud).astype(np.float32)
    frd["systemic_stress_index"] = rng.uniform(0.0, 0.29, size=n_fraud).astype(np.float32)
    frd["burst_ratio"]           = rng.uniform(0.0, 2.0, size=n_fraud).astype(np.float32)
    frd["avg_error_rate_others"] = rng.uniform(0.0, 0.1, size=n_fraud).astype(np.float32)
    frd["max_error_rate_others"] = rng.uniform(0.0, 0.1, size=n_fraud).astype(np.float32)
    frd["data_source"]           = "synthetic_fraud"
    frd["is_market_hours"]       = np.int8(0)

    for df_part in [atk, frd]:
        df_part = _add_time_features(df_part)
        # Re-apply forced values after time feature derivation
        if "is_market_hours" in df_part.columns and df_part["data_source"].iloc[0].startswith("synthetic_fraud"):
            df_part["is_market_hours"] = np.int8(0)
        df_part = _add_rolling_features(df_part)
        df_part = _set_api_complexity(df_part)
        df_part = _fill_schema_defaults(df_part)

    atk = _add_time_features(atk)
    atk = _add_rolling_features(atk)
    atk = _set_api_complexity(atk)
    atk = _fill_schema_defaults(atk)

    frd = _add_time_features(frd)
    frd["is_market_hours"] = np.int8(0)
    frd = _add_rolling_features(frd)
    frd = _set_api_complexity(frd)
    frd = _fill_schema_defaults(frd)
    # Restore fraud-specific rolling error
    frd["error_rate_rolling"] = np.float32(0.9)

    synthetic = pd.concat([atk, frd], ignore_index=True)
    print(f"  Synthetic rows: {len(synthetic):,}  "
          f"(attack={n_attack:,}  fraud={n_fraud:,})")
    return synthetic


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Combine, clean, cap failure rate
# ══════════════════════════════════════════════════════════════════════════════

def combine_and_clean(
    v6_df: pd.DataFrame,
    new_frames: list,
    cap_failure_rate: float = 0.20,
) -> pd.DataFrame:
    parts = [v6_df] + [f for f in new_frames if f is not None]
    combined = pd.concat(parts, ignore_index=True)
    print(f"\nCombined before cleaning: {len(combined):,} rows")

    # Ensure all v6 columns exist
    for col in V6_COLS:
        if col not in combined.columns:
            combined[col] = 0

    # Coerce timestamp to datetime so mixed str/Timestamp sorts correctly
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    failure_rate = 1.0 - combined["success"].fillna(1).mean()
    print(f"Failure rate before cap: {failure_rate * 100:.2f}%")

    if failure_rate > cap_failure_rate:
        success_rows  = combined[combined["success"] == 1]
        failure_rows  = combined[combined["success"] == 0]
        target_fail   = int(len(success_rows) * cap_failure_rate)
        failure_rows  = failure_rows.sample(n=target_fail, random_state=42)
        combined      = pd.concat([success_rows, failure_rows])
        combined      = combined.sort_values("timestamp").reset_index(drop=True)
        failure_rate  = 1.0 - combined["success"].fillna(1).mean()
        print(f"Failure rate after cap:  {failure_rate * 100:.2f}%")

    # Keep only v6 schema columns (in order)
    out_cols = [c for c in V6_COLS if c in combined.columns]
    combined = combined[out_cols]

    print(f"\nTotal rows: {len(combined):,}")
    print(f"Failure rate: {failure_rate * 100:.2f}%")
    print("\nError type distribution:")
    print(combined["error_type"].value_counts(dropna=False).to_string())
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args) -> None:
    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────────────
    if not args.synthetic_only:
        check_kaggle_setup()
        download_datasets(skip_if_exists=args.skip_download)

    # ── Load v6 base ───────────────────────────────────────────────────────
    print(f"\nLoading base dataset: {input_path} ...")
    v6_df = pd.read_csv(input_path, low_memory=False)
    print(f"  v6 rows: {len(v6_df):,}  "
          f"failure rate: {(1 - v6_df['success'].fillna(1).mean())*100:.1f}%")

    # ── Step 2 & 3: Map / generate ────────────────────────────────────────
    new_frames = []

    if not args.synthetic_only:
        print("\n--- Mapping Credit Card Fraud ---")
        cc = map_creditcard_fraud(RAW_DIR)
        if cc is not None:
            new_frames.append(cc)

        print("\n--- Mapping Network Intrusion (NSL-KDD / KDD99) ---")
        net = map_network_intrusion(RAW_DIR)
        if net is not None:
            new_frames.append(net)

        print("\n--- Mapping IP Network Traffic ---")
        traffic = map_network_traffic(RAW_DIR)
        if traffic is not None:
            new_frames.append(traffic)

    print(f"\n--- Generating {args.n_synthetic:,} synthetic hard failure rows ---")
    synthetic = generate_synthetic_failures(n=args.n_synthetic)
    new_frames.append(synthetic)

    # ── Step 4: Combine and clean ─────────────────────────────────────────
    print("\n--- Combining datasets ---")
    combined = combine_and_clean(v6_df, new_frames)

    # ── Step 5: Save ──────────────────────────────────────────────────────
    print(f"\nSaving → {output_path} ...")
    combined.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / 1e6
    print(f"File size: {size_mb:.0f} MB")
    print(f"\nDone. Saved → {output_path}")
    print("\nNext step — retrain on v7:")
    print(f"  python scripts/run_lstm_training.py \\")
    print(f"    --data {output_path} \\")
    print(f"    --epochs 15 --hidden_size 256 --batch_size 128 \\")
    print(f"    --patience 6 --sequence_length 60 --focal_gamma 3.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEO API Intelligence — New Kaggle Data Integration (v7)"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip Kaggle download if files already exist in data/raw_kaggle/",
    )
    parser.add_argument(
        "--synthetic_only", action="store_true",
        help="Only add synthetic hard failure rows, skip Kaggle download",
    )
    parser.add_argument(
        "--n_synthetic", type=int, default=100_000,
        help="Number of synthetic hard failure rows to generate (default 100,000)",
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join("data", "banking_api_features_v6.csv"),
        help="Input CSV to append to",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("data", "banking_api_features_v7.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    main(args)
