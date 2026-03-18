#!/usr/bin/env python3
"""
integrate_kaggle_data.py  --  Real Kaggle data integration for FCE project.

Downloads 5 Kaggle datasets, normalises each to the project schema, engineers
features using identical logic to generate_production_dataset.py, then appends
to data/banking_api_features.csv and data/banking_api_telemetry.db.

Step 1 -- Download via Kaggle CLI (auto-installs kaggle package if missing)
Step 2 -- Normalise to schema: timestamp, api_name, response_time, status_code,
          success, error_type, request_count, error_rate_boost, rt_multiplier
Step 3 -- Engineer features: rolling(60), lags(1,5), EMA(10,30), cyclical
Step 4 -- Append to CSV and SQLite
Step 5 -- Print integration report

Usage:
    python scripts/integrate_kaggle_data.py
    python scripts/integrate_kaggle_data.py --dry_run
    python scripts/integrate_kaggle_data.py --datasets nab creditcard
    python scripts/integrate_kaggle_data.py --skip_download --dry_run
"""

import os, sys, json, time, argparse, pathlib, subprocess, textwrap, warnings
import numpy as np
import pandas as pd
import sqlite3

warnings.filterwarnings("ignore", category=FutureWarning)

print("=== FCE Kaggle Integration Pipeline ===\n")

# -- CLI -----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Download and integrate real Kaggle datasets into the FCE project",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--datasets", nargs="*", default=None,
    choices=["nab", "creditcard", "paysim", "network", "weblog"],
    help="Datasets to process (default: all five)",
)
parser.add_argument(
    "--dry_run", action="store_true",
    help="Parse and report only -- do not write to CSV or DB",
)
parser.add_argument(
    "--skip_download", action="store_true",
    help="Skip download, reuse existing files in data/kaggle_raw/",
)
parser.add_argument(
    "--max_rows", type=int, default=300_000,
    help="Maximum rows per dataset (default 300,000)",
)
parser.add_argument(
    "--out_dir", type=str, default="data",
    help="Project data directory (default: data)",
)
args = parser.parse_args()

DATASETS_TO_RUN = args.datasets or ["nab", "creditcard", "paysim", "network", "weblog"]
OUT_DIR         = pathlib.Path(args.out_dir)
RAW_DIR         = OUT_DIR / "kaggle_raw"
CSV_PATH        = OUT_DIR / "banking_api_features.csv"
DB_PATH         = OUT_DIR / "banking_api_telemetry.db"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# Synthetic start date: 2025-01-01 -- places real data AFTER the 2023-2024
# synthetic range so the training script's global time-sort keeps data clean.
REAL_DATA_START = pd.Timestamp("2025-01-01")

# Map api_name -> api_complexity (mirrors generate_production_dataset.py)
API_COMPLEXITY = {
    "stock_price_api":  1.0,
    "forex_api":        1.1,
    "crypto_api":       1.3,
    "market_data_api":  1.2,
    "transaction_api":  1.15,
}

# Dataset registry
DATASET_REGISTRY = {
    "nab": {
        "slug":        "boltzmannbrain/nab",
        "description": "NAB -- real AWS EC2 server latency with system failures",
        "api_name":    "market_data_api",
    },
    "creditcard": {
        "slug":        "mlg-ulb/creditcardfraud",
        "description": "Credit Card Fraud -- real fraud transaction patterns",
        "api_name":    "transaction_api",
    },
    "paysim": {
        "slug":        "ealaxi/paysim1",
        "description": "PaySim1 -- real financial transaction failures",
        "api_name":    "transaction_api",
    },
    "network": {
        "slug":        "malkasasbeh/network-anomaly-detection-dataset",
        "description": "Network Anomaly Detection -- real DDoS and network attack patterns",
        "api_name":    "stock_price_api",
    },
    "weblog": {
    "slug":        "eliasdabbas/web-server-access-logs",
    "description": "Web Server Access Logs -- real HTTP error patterns",
    "api_name":    "market_data_api",
},
    }

# Exact CSV column order -- must never change
CSV_COLS = [
    "timestamp", "api_name",
    "response_time", "status_code", "success", "error_type", "request_count",
    "hour", "day_of_week", "is_weekend", "is_holiday",
    "is_market_hours", "is_financial_peak", "high_frequency_api",
    "response_time_rolling_mean", "response_time_rolling_std",
    "error_rate_rolling", "response_time_variance", "error_volatility",
    "response_time_lag_1", "response_time_lag_5", "error_rate_lag_1",
    "response_time_ema_10", "response_time_ema_30", "error_rate_ema_10",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "api_complexity", "error_rate_boost", "rt_multiplier", "event_label",
]

# DB columns include two extras not in CSV
DB_EXTRA_COLS = ["is_pre_open", "is_market_close"]


# -----------------------------------------------------------------------------
# STEP 1 -- Prerequisites
# -----------------------------------------------------------------------------

def ensure_kaggle():
    """Ensure the kaggle package is importable; install it if missing."""
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        print("  kaggle package not found -- installing into current environment ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle", "--quiet"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [FAIL] pip install failed:\n{result.stderr}")
            return False
        print("  [OK] kaggle installed successfully")
        return True


def check_kaggle_credentials() -> bool:
    """
    Return True if Kaggle credentials are available via any of these sources
    (checked in priority order):

      1. KAGGLE_USERNAME + KAGGLE_KEY environment variables
      2. KAGGLE_API_TOKEN environment variable (JSON string:
             {"username": "...", "key": "..."}  )
      3. ~/.kaggle/kaggle.json file

    SECURITY NOTE: never hard-code token values in source files or commands.
    Set env vars in your shell session only:
        PowerShell:  $env:KAGGLE_USERNAME = "you"; $env:KAGGLE_KEY = "token"
        bash/zsh:    export KAGGLE_USERNAME=you KAGGLE_KEY=token
    """
    # ── Source 1: KAGGLE_USERNAME + KAGGLE_KEY ────────────────────────────────
    env_user = os.environ.get("KAGGLE_USERNAME", "").strip()
    env_key  = os.environ.get("KAGGLE_KEY",      "").strip()
    if env_user and env_key:
        print(f"  [OK] Kaggle credentials found via env vars  (KAGGLE_USERNAME={env_user})")
        return True

    # ── Source 2: KAGGLE_API_TOKEN (JSON string) ──────────────────────────────
    env_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if env_token:
        try:
            data = json.loads(env_token)
            if "username" in data and "key" in data:
                print(f"  [OK] Kaggle credentials found via KAGGLE_API_TOKEN  "
                      f"(username: {data['username']})")
                # Write to ~/.kaggle/kaggle.json so the kaggle CLI picks it up
                cred = pathlib.Path.home() / ".kaggle" / "kaggle.json"
                cred.parent.mkdir(exist_ok=True)
                cred.write_text(json.dumps(data))
                cred.chmod(0o600)
                return True
            else:
                print("  [FAIL] KAGGLE_API_TOKEN is set but missing 'username' or 'key' fields.")
                print("         Expected format: '{\"username\":\"you\",\"key\":\"token\"}'")
                return False
        except json.JSONDecodeError:
            print("  [FAIL] KAGGLE_API_TOKEN is set but is not valid JSON.")
            print("         Expected format: '{\"username\":\"you\",\"key\":\"token\"}'")
            print("         If you only have a raw key, use KAGGLE_KEY instead:")
            print("           $env:KAGGLE_USERNAME = 'your_username'")
            print("           $env:KAGGLE_KEY      = 'your_token'")
            return False

    # ── Source 3: ~/.kaggle/kaggle.json ───────────────────────────────────────
    cred = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    if cred.exists():
        try:
            data = json.loads(cred.read_text())
            if "username" not in data or "key" not in data:
                print(f"  [FAIL] kaggle.json is malformed (missing username or key): {cred}")
                return False
            print(f"  [OK] Kaggle credentials found via kaggle.json  (username: {data['username']})")
            return True
        except Exception as e:
            print(f"  [FAIL] Could not read kaggle.json: {e}")
            return False

    # ── Nothing found ─────────────────────────────────────────────────────────
    print(textwrap.dedent(f"""
    [FAIL]  No Kaggle credentials found.  Try one of these options:

    Option A -- environment variables (recommended, nothing written to disk):
        PowerShell:
            $env:KAGGLE_USERNAME = "your_username"
            $env:KAGGLE_KEY      = "your_token"
        bash/zsh:
            export KAGGLE_USERNAME=your_username
            export KAGGLE_KEY=your_token

    Option B -- kaggle.json file:
        1. kaggle.com -> Settings -> API -> Create New Token
        2. PowerShell:
               mkdir ~\\.kaggle -Force
               Move-Item $env:USERPROFILE\\Downloads\\kaggle.json ~\\.kaggle\\kaggle.json

    Also accept each dataset's rules at kaggle.com before downloading:
        boltzmannbrain/nab
        mlg-ulb/creditcardfraud
        ealaxi/paysim1
        malkasasbeh/network-anomaly-detection-dataset
        shashwatwork/web-server-access-log
    """))
    return False


# -----------------------------------------------------------------------------
# STEP 1 -- Download
# -----------------------------------------------------------------------------

def download_dataset(slug: str, dest: pathlib.Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {slug}  ->  {dest} ...")
    t0 = time.time()
    try:
        from kaggle import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            slug,
            path=str(dest),
            unzip=True,
            quiet=False,
        )
        elapsed = time.time() - t0
        files   = list(dest.rglob("*.csv"))
        print(f"  [OK] Downloaded in {elapsed:.1f}s  ({len(files)} CSV files)")
        return True
    except Exception as e:
        err = str(e)
        if "403" in err or "Forbidden" in err:
            print(f"  [FAIL] Access denied — accept rules at kaggle.com/datasets/{slug}")
        elif "404" in err or "Not Found" in err:
            print(f"  [FAIL] Dataset not found: {slug}")
        else:
            print(f"  [FAIL] Download failed: {err[:300]}")
        return False


def find_csvs(folder: pathlib.Path) -> list[pathlib.Path]:
    """Return all CSVs in folder, largest first."""
    csvs = sorted(folder.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    return csvs


# -----------------------------------------------------------------------------
# STEP 2 -- Normalisers (one per dataset)
# Each returns (df_normalised, mapping_report_dict)
# df_normalised has: timestamp, api_name, response_time, status_code,
#                    success, error_type, request_count,
#                    error_rate_boost (placeholder), rt_multiplier (placeholder),
#                    event_label
# Features are engineered later in a shared step.
# -----------------------------------------------------------------------------

def _col(df: pd.DataFrame, candidates: list, default=None):
    """Return the first candidate column name found in df, or default."""
    for c in candidates:
        if c in df.columns:
            return c
        # case-insensitive fallback
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return default


def _report(direct: list, derived: list) -> dict:
    return {"direct": direct, "derived": derived}


def normalise_nab(folder: pathlib.Path, api_name: str, max_rows: int):
    """
    NAB (Numenta Anomaly Benchmark) -- real AWS CloudWatch time-series.

    Expected files: multiple CSVs each with [timestamp, value] columns.
    Priority: files containing 'ec2' or 'latency' in their name.
    Labels in labels/combined_labels.json (if present).
    """
    csvs = find_csvs(folder)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {folder}")

    # Prefer EC2 / latency files
    priority = [f for f in csvs if any(k in f.name.lower() for k in ("ec2", "latency", "cloudwatch"))]
    selected = priority if priority else csvs

    # Load anomaly labels if available
    labels_json = folder / "labels" / "combined_labels.json"
    anomaly_windows: dict = {}
    if labels_json.exists():
        raw = json.loads(labels_json.read_text())
        anomaly_windows = raw  # {filename: [timestamp, ...]}

    frames = []
    for csv_path in selected[:10]:   # cap at 10 files
        try:
            raw = pd.read_csv(csv_path, nrows=max_rows // max(len(selected[:10]), 1))
        except Exception:
            continue
        ts_col = _col(raw, ["timestamp", "Timestamp", "time", "Time", "datetime"])
        val_col = _col(raw, ["value", "Value", "metric", "latency", "response"])
        if ts_col is None or val_col is None:
            continue

        raw["_ts"]  = pd.to_datetime(raw[ts_col], errors="coerce")
        raw["_val"] = pd.to_numeric(raw[val_col],  errors="coerce").fillna(100.0)

        # Anomaly flag from label file
        fname_key = next((k for k in anomaly_windows if csv_path.name in k), None)
        anomaly_ts = set(anomaly_windows.get(fname_key, []) if fname_key else [])
        raw["_anomaly"] = raw[ts_col].astype(str).isin(anomaly_ts).astype(int)

        frames.append(raw[["_ts", "_val", "_anomaly"]].rename(
            columns={"_ts": "timestamp", "_val": "response_time", "_anomaly": "is_anomaly"}
        ))

    if not frames:
        raise ValueError("NAB: could not parse any CSV files (missing timestamp/value columns)")

    df = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").head(max_rows)

    # Derive target columns
    df["api_name"]     = api_name
    df["success"]      = (df["is_anomaly"] == 0).astype(int)
    df["status_code"]  = df["is_anomaly"].map({0: 200, 1: 503})
    df["error_type"]   = df["is_anomaly"].map({0: None, 1: "system_latency_anomaly"})
    df["request_count"] = np.random.default_rng(42).poisson(30, len(df))
    df["event_label"]  = df["is_anomaly"].map({0: "normal", 1: "system_latency_anomaly"})
    # Placeholders -- computed after feature engineering
    df["error_rate_boost"] = 0.0
    df["rt_multiplier"]    = 1.0

    mapping = _report(
        direct=["timestamp -> timestamp", "value -> response_time", "anomaly_label -> success/event_label"],
        derived=["status_code", "error_type", "request_count", "error_rate_boost", "rt_multiplier"],
    )
    return df[["timestamp","api_name","response_time","status_code","success",
               "error_type","request_count","error_rate_boost","rt_multiplier","event_label"]], mapping


def normalise_creditcard(folder: pathlib.Path, api_name: str, max_rows: int):
    """
    Credit Card Fraud (mlg-ulb/creditcardfraud).

    Columns: Time (seconds), V1..V28 (PCA), Amount, Class (0=normal, 1=fraud).
    Time is seconds elapsed from first transaction in the dataset.
    """
    csvs = find_csvs(folder)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {folder}")
    df = pd.read_csv(csvs[0], nrows=max_rows)

    time_col   = _col(df, ["Time", "time"])
    amount_col = _col(df, ["Amount", "amount"])
    class_col  = _col(df, ["Class", "class", "label", "fraud"])

    missing = [n for n, c in [("Time", time_col), ("Amount", amount_col), ("Class", class_col)] if c is None]
    if missing:
        raise ValueError(f"creditcardfraud: missing expected columns {missing} in {csvs[0].name}")

    # Timestamps: offset seconds from 2025-01-01 00:00:00
    df["timestamp"] = REAL_DATA_START + pd.to_timedelta(df[time_col], unit="s")

    # Class=1 -> fraud -> API failure (success=0); Class=0 -> normal (success=1)
    df["success"]    = (df[class_col] == 0).astype(int)
    df["status_code"] = df[class_col].map({0: 200, 1: 500})
    df["error_type"]  = df[class_col].map({0: None, 1: "fraud_transaction_failure"})
    df["event_label"] = df[class_col].map({0: "normal", 1: "fraud_transaction"})

    # Response time: base 120ms + noise; fraud transactions run 3-8x slower
    # (elevated latency = fraud scoring / blocking overhead)
    rng = np.random.default_rng(42)
    base_rt = rng.exponential(80, len(df)) + 60
    fraud_mult = np.where(df[class_col] == 1, rng.uniform(3.0, 8.0, len(df)), 1.0)
    df["response_time"] = np.maximum(8, np.round(base_rt * fraud_mult, 2))

    # Request count: transaction volume proxy (higher amount -> higher load)
    df["request_count"] = np.clip(
        np.round(np.log1p(df[amount_col].fillna(0) + 1) * 3).astype(int), 1, 500
    )

    df["api_name"]         = api_name
    df["error_rate_boost"] = 0.0
    df["rt_multiplier"]    = 1.0

    mapping = _report(
        direct=["Time -> timestamp (offset from 2025-01-01)", "Class -> success/error_type/event_label",
                "Amount -> request_count (log-scaled)"],
        derived=["response_time (synthetic base + fraud multiplier)", "status_code",
                 "error_rate_boost", "rt_multiplier"],
    )
    return df[["timestamp","api_name","response_time","status_code","success",
               "error_type","request_count","error_rate_boost","rt_multiplier","event_label"]], mapping


def normalise_paysim(folder: pathlib.Path, api_name: str, max_rows: int):
    """
    PaySim1 (ealaxi/paysim1).

    Columns: step (hour), type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
             nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud.
    6.3M rows -- sampled to max_rows with stratification on isFraud.
    """
    csvs = find_csvs(folder)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {folder}")

    # PaySim is large; read in chunks and stratify-sample
    print(f"    Reading PaySim (large file, may take a moment) ...")
    try:
        df = pd.read_csv(csvs[0])
    except Exception as e:
        raise ValueError(f"paysim: could not read {csvs[0].name}: {e}")

    step_col   = _col(df, ["step"])
    type_col   = _col(df, ["type"])
    amount_col = _col(df, ["amount", "Amount"])
    fraud_col  = _col(df, ["isFraud", "is_fraud", "fraud", "label", "Class"])

    missing = [n for n, c in [("step", step_col), ("type", type_col),
                               ("amount", amount_col), ("isFraud", fraud_col)] if c is None]
    if missing:
        raise ValueError(f"paysim: missing expected columns {missing} in {csvs[0].name}")

    # Stratified sample: keep fraud ratio
    fraud_rows   = df[df[fraud_col] == 1]
    normal_rows  = df[df[fraud_col] == 0]
    fraud_rate   = len(fraud_rows) / max(len(df), 1)
    n_fraud      = min(len(fraud_rows),  int(max_rows * fraud_rate))
    n_normal     = min(len(normal_rows), max_rows - n_fraud)
    rng = np.random.default_rng(42)
    df = pd.concat([
        fraud_rows.sample(n=n_fraud,  random_state=42),
        normal_rows.sample(n=n_normal, random_state=42),
    ], ignore_index=True).sample(frac=1, random_state=42)

    # Timestamps: step = 1 hour unit; 1 step = 1 hour from REAL_DATA_START
    df["timestamp"] = REAL_DATA_START + pd.to_timedelta(df[step_col], unit="h")

    df["success"]    = (df[fraud_col] == 0).astype(int)
    df["status_code"] = df[fraud_col].map({0: 200, 1: 500})

    # Map transaction type -> error type for fraud, None for normal
    type_error_map = {
        "TRANSFER":  "fraudulent_transfer_detected",
        "CASH_OUT":  "fraudulent_cashout_detected",
        "PAYMENT":   "fraudulent_payment_detected",
        "CASH_IN":   "fraudulent_cashin_detected",
        "DEBIT":     "fraudulent_debit_detected",
    }
    df["error_type"] = df.apply(
        lambda r: type_error_map.get(str(r[type_col]).upper()) if r[fraud_col] == 1 else None,
        axis=1,
    )
    df["event_label"] = df[fraud_col].map({0: "normal", 1: "fraudulent_payment"})

    # Response time: base varies by transaction type; fraud = elevated
    type_base_rt = {"TRANSFER": 250, "CASH_OUT": 180, "PAYMENT": 120,
                    "CASH_IN": 100, "DEBIT": 90}
    base_rt_vec = df[type_col].str.upper().map(type_base_rt).fillna(150).values
    fraud_mult  = np.where(df[fraud_col] == 1, rng.uniform(2.5, 6.0, len(df)), 1.0)
    noise       = rng.normal(0, 20, len(df))
    df["response_time"] = np.maximum(8, np.round(base_rt_vec * fraud_mult + noise, 2))

    # Request count: amount proxy (log-scaled, clipped)
    df["request_count"] = np.clip(
        np.round(np.log1p(df[amount_col].fillna(0).abs()) * 2).astype(int), 1, 500
    )

    df["api_name"]         = api_name
    df["error_rate_boost"] = 0.0
    df["rt_multiplier"]    = 1.0

    mapping = _report(
        direct=["step -> timestamp (hours from 2025-01-01)", "isFraud -> success/event_label",
                "type -> error_type (fraud only)", "amount -> request_count (log-scaled)"],
        derived=["response_time (type-based base + fraud multiplier)", "status_code",
                 "error_rate_boost", "rt_multiplier"],
    )
    return df[["timestamp","api_name","response_time","status_code","success",
               "error_type","request_count","error_rate_boost","rt_multiplier","event_label"]], mapping


def normalise_network(folder: pathlib.Path, api_name: str, max_rows: int):
    """
    Network Anomaly Detection (malkasasbeh).

    Flexible parser -- handles UNSW-NB15, KDDCup99, NSL-KDD, and generic
    traffic datasets. Tries multiple column naming conventions.
    """
    csvs = find_csvs(folder)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {folder}")

    frames = []
    for csv_path in csvs[:5]:
        try:
            raw = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
            frames.append(raw)
        except Exception:
            continue
    if not frames:
        raise ValueError(f"network: could not read any CSV files under {folder}")
    df = pd.concat(frames, ignore_index=True).head(max_rows)

    # -- Detect label column ----------------------------------------------------
    label_col = _col(df, ["label", "Label", "attack_cat", "attack", "class",
                           "Class", "category", "type", "outcome"])
    # -- Detect duration / response time ---------------------------------------
    dur_col = _col(df, ["dur", "duration", "Duration", "flow_duration",
                         "fwd_iat_mean", "response_time"])
    # -- Detect byte / size columns --------------------------------------------
    bytes_col = _col(df, ["sbytes", "src_bytes", "totlen_fwd_pkts",
                           "tot_fwd_pkts", "total_length_of_fwd_packets"])
    # -- Detect timestamp ------------------------------------------------------
    ts_col = _col(df, ["timestamp", "Timestamp", "time", "Time",
                        "flow_start_time", "stime"])

    # Build normalised columns
    # Timestamp
    if ts_col is not None:
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        df["timestamp"] = df["timestamp"].fillna(
            REAL_DATA_START + pd.to_timedelta(df.index.astype(int), unit="s")
        )
    else:
        # Synthetic: one row per second from REAL_DATA_START
        df["timestamp"] = REAL_DATA_START + pd.to_timedelta(df.index.astype(int) * 60, unit="s")

    # Response time from duration (seconds -> ms)
    if dur_col is not None:
        rt_raw = pd.to_numeric(df[dur_col], errors="coerce").fillna(0)
        df["response_time"] = np.maximum(8, np.round(rt_raw * 1000, 2))
        # If values look like they're already in ms (median > 50), keep as-is
        if df["response_time"].median() > 50_000:
            df["response_time"] = np.round(rt_raw, 2)
    else:
        rng = np.random.default_rng(42)
        df["response_time"] = np.maximum(8, rng.exponential(150, len(df)))

    # Request count from bytes
    if bytes_col is not None:
        df["request_count"] = np.clip(
            pd.to_numeric(df[bytes_col], errors="coerce").fillna(1)
              .apply(lambda x: max(1, int(np.log1p(abs(x))))), 1, 1000
        )
    else:
        df["request_count"] = 1

    # Label -> success / error_type / event_label
    ATTACK_MAP = {
        # KDDCup-style
        "normal": ("normal", 1, 200, None),
        # DoS attacks
        "dos":         ("ddos_attack",        0, 429, "network_dos_overload"),
        "neptune":     ("ddos_attack",        0, 429, "network_dos_overload"),
        "smurf":       ("ddos_attack",        0, 429, "network_dos_overload"),
        "pod":         ("ddos_attack",        0, 429, "network_dos_overload"),
        "teardrop":    ("ddos_attack",        0, 429, "network_dos_overload"),
        "land":        ("ddos_attack",        0, 429, "network_dos_overload"),
        "back":        ("ddos_attack",        0, 429, "network_dos_overload"),
        "apache2":     ("ddos_attack",        0, 429, "network_dos_overload"),
        "udpstorm":    ("ddos_attack",        0, 429, "network_dos_overload"),
        "mailbomb":    ("ddos_attack",        0, 429, "network_dos_overload"),
        # Probe / Scan
        "portsweep":   ("network_probe",      0, 403, "security_probe_detected"),
        "nmap":        ("network_probe",      0, 403, "security_probe_detected"),
        "satan":       ("network_probe",      0, 403, "security_probe_detected"),
        "ipsweep":     ("network_probe",      0, 403, "security_probe_detected"),
        # Remote/User exploits
        "r2l":         ("security_breach_attempt", 0, 403, "security_breach_attempt"),
        "u2r":         ("security_breach_attempt", 0, 403, "security_breach_attempt"),
        "warezmaster": ("security_breach_attempt", 0, 403, "security_breach_attempt"),
        "ftp_write":   ("security_breach_attempt", 0, 403, "security_breach_attempt"),
        # UNSW-NB15 / generic
        "exploits":    ("dependency_failure", 0, 503, "503_external_dependency_down"),
        "fuzzers":     ("ddos_attack",        0, 429, "network_dos_overload"),
        "generic":     ("ddos_attack",        0, 429, "network_dos_overload"),
        "reconnaissance": ("network_probe",   0, 403, "security_probe_detected"),
        "shellcode":   ("security_breach_attempt", 0, 403, "security_breach_attempt"),
        "backdoor":    ("security_breach_attempt", 0, 403, "security_breach_attempt"),
        "worms":       ("dependency_failure", 0, 503, "503_external_dependency_down"),
        "analysis":    ("network_probe",      0, 403, "security_probe_detected"),
        # Binary label fallbacks
        "0": ("normal",      1, 200, None),
        "1": ("ddos_attack", 0, 429, "network_attack_detected"),
        "attack": ("ddos_attack", 0, 429, "network_attack_detected"),
    }

    if label_col is not None:
        labels_raw = df[label_col].astype(str).str.strip().str.lower()
        # Map; anything not in map -> treat as normal if numeric 0, else attack
        def _map_label(lbl):
            if lbl in ATTACK_MAP:
                return ATTACK_MAP[lbl]
            if lbl.startswith("0") or lbl == "normal":
                return ATTACK_MAP["normal"]
            return ATTACK_MAP["1"]  # unknown label -> treat as attack
        mapped = labels_raw.apply(_map_label)
        df["event_label"]  = mapped.apply(lambda x: x[0])
        df["success"]      = mapped.apply(lambda x: x[1])
        df["status_code"]  = mapped.apply(lambda x: x[2])
        df["error_type"]   = mapped.apply(lambda x: x[3])
    else:
        df["event_label"] = "normal"
        df["success"]     = 1
        df["status_code"] = 200
        df["error_type"]  = None

    # Elevate response_time for attacks
    attack_mask = (df["success"] == 0)
    df.loc[attack_mask, "response_time"] *= np.random.default_rng(1).uniform(
        2.0, 5.0, attack_mask.sum()
    )
    df["response_time"] = np.maximum(8, df["response_time"]).round(2)

    df["api_name"]         = api_name
    df["error_rate_boost"] = 0.0
    df["rt_multiplier"]    = 1.0

    direct  = [f"{label_col} -> success/event_label/error_type"] if label_col else []
    direct += [f"{dur_col} -> response_time (?1000 s->ms)"]   if dur_col else []
    direct += [f"{bytes_col} -> request_count (log-scaled)"]  if bytes_col else []
    if ts_col:
        direct.append(f"{ts_col} -> timestamp")
    derived = [c for c in ["timestamp","response_time","request_count",
                            "status_code","error_rate_boost","rt_multiplier"]
               if not any(c in d for d in direct)]

    mapping = _report(direct=direct, derived=derived)
    return df[["timestamp","api_name","response_time","status_code","success",
               "error_type","request_count","error_rate_boost","rt_multiplier","event_label"]], mapping


def normalise_weblog(folder: pathlib.Path, api_name: str, max_rows: int):
    """
    Web Server Access Logs (shashwatwork/web-server-access-log).

    Handles: Apache/Nginx combined log format, SEC EDGAR logs, and
    any CSV with at least a status code column.
    """
    csvs = find_csvs(folder)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {folder}")

    df = None
    for csv_path in csvs[:3]:
        try:
            raw = pd.read_csv(csv_path, nrows=max_rows, low_memory=False,
                               on_bad_lines="skip")
            if len(raw.columns) >= 3:
                df = raw
                break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"weblog: could not read any usable CSV files under {folder}")
    df = df.head(max_rows)

    # -- Detect columns ---------------------------------------------------------
    ts_col   = _col(df, ["timestamp", "Timestamp", "datetime", "date_time",
                          "time", "Time", "date", "Date", "log_time",
                          # SEC EDGAR format
                          "date", "ip_ts"])
    code_col = _col(df, ["status", "status_code", "code", "http_status",
                          "response_code", "sc-status",
                          # SEC EDGAR
                          "code"])
    size_col = _col(df, ["size", "bytes", "content_length", "sc-bytes",
                          "cs-bytes", "body_bytes_sent", "bytes_sent",
                          # SEC EDGAR
                          "size"])
    rt_col   = _col(df, ["response_time", "time_taken", "request_time",
                          "sc-time-taken", "duration_ms", "duration"])
    url_col  = _col(df, ["url", "request", "uri", "path", "endpoint",
                          "accession", "cik"])

    # -- Timestamps -------------------------------------------------------------
    if ts_col is not None:
        parsed_ts = pd.to_datetime(df[ts_col], errors="coerce", infer_datetime_format=True)
        n_valid = parsed_ts.notna().sum()
        if n_valid < len(df) * 0.5:
            # Try combining date + time columns if available
            date_col2 = _col(df, ["date", "Date"])
            time_col2 = _col(df, ["time", "Time"])
            if date_col2 and time_col2 and date_col2 != time_col2:
                parsed_ts = pd.to_datetime(
                    df[date_col2].astype(str) + " " + df[time_col2].astype(str),
                    errors="coerce",
                )
        # Shift to 2025+ window regardless of original year
        valid_ts = parsed_ts.dropna()
        if len(valid_ts) > 0:
            ts_min = valid_ts.min()
            parsed_ts = parsed_ts + (REAL_DATA_START - ts_min.normalize())
        parsed_ts = parsed_ts.fillna(
            REAL_DATA_START + pd.to_timedelta(df.index * 60, unit="s")
        )
        df["timestamp"] = parsed_ts
    else:
        df["timestamp"] = REAL_DATA_START + pd.to_timedelta(df.index * 60, unit="s")

    # -- Status code ------------------------------------------------------------
    if code_col is not None:
        df["status_code"] = pd.to_numeric(df[code_col], errors="coerce").fillna(200).astype(int)
    else:
        df["status_code"] = 200

    df["success"] = (df["status_code"] < 400).astype(int)

    # -- Error type from status code --------------------------------------------
    def _http_error_type(sc):
        if sc < 400:   return None
        if sc == 400:  return "400_malformed_request"
        if sc == 401:  return "401_unauthorized"
        if sc == 403:  return "403_incorrect_api_permission"
        if sc == 404:  return "404_bad_or_outdated_url"
        if sc == 408:  return "408_request_timeout"
        if sc == 429:  return "ddos_rate_limit_exceeded"
        if sc == 500:  return "500_internal_server_error"
        if sc == 502:  return "502_bad_gateway"
        if sc == 503:  return "503_external_dependency_down"
        if sc == 504:  return "504_complex_endpoint_timeout"
        if 400 <= sc < 500: return f"{sc}_client_error"
        if sc >= 500:       return f"{sc}_server_error"
        return None

    df["error_type"]  = df["status_code"].apply(_http_error_type)

    # event_label: group 4xx/5xx into meaningful labels
    def _event_label(sc):
        if sc < 400:        return "normal"
        if sc == 429:       return "ddos_attack"
        if sc in (401,403): return "permission_error"
        if sc == 404:       return "bad_url"
        if sc >= 500:       return "vendor_outage"
        return "normal"

    df["event_label"] = df["status_code"].apply(_event_label)

    # -- Response time ----------------------------------------------------------
    if rt_col is not None:
        df["response_time"] = pd.to_numeric(df[rt_col], errors="coerce").fillna(100)
        # Normalise units: if median < 1, assume seconds -> convert to ms
        if df["response_time"].median() < 1:
            df["response_time"] *= 1000
    elif size_col is not None:
        # Bytes -> latency proxy: assume 1 MB/s throughput floor
        bytes_vals = pd.to_numeric(df[size_col], errors="coerce").fillna(0).clip(0)
        df["response_time"] = np.maximum(8, np.round(bytes_vals / 1000 + 50, 2))
    else:
        rng = np.random.default_rng(42)
        df["response_time"] = np.maximum(8, rng.exponential(100, len(df)))

    # Elevate RT for error responses
    err_mask = df["success"] == 0
    df.loc[err_mask, "response_time"] *= np.random.default_rng(7).uniform(
        1.5, 4.0, err_mask.sum()
    )
    df["response_time"] = df["response_time"].clip(lower=8).round(2)

    # -- Request count ----------------------------------------------------------
    df["request_count"] = 1   # each log row = 1 request

    # -- API name from URL (optional refinement) --------------------------------
    if url_col is not None:
        url_str = df[url_col].astype(str).str.lower()
        api_map_mask = url_str.str.contains("trade|transaction|payment", na=False)
        df["api_name"] = api_name
        df.loc[api_map_mask, "api_name"] = "transaction_api"
    else:
        df["api_name"] = api_name

    df["error_rate_boost"] = 0.0
    df["rt_multiplier"]    = 1.0

    direct  = []
    if ts_col:   direct.append(f"{ts_col} -> timestamp (shifted to 2025+)")
    if code_col: direct.append(f"{code_col} -> status_code + success + error_type + event_label")
    if rt_col:   direct.append(f"{rt_col} -> response_time")
    elif size_col: direct.append(f"{size_col} -> response_time (bytes/1000 + 50ms floor)")
    derived = [c for c in ["timestamp","response_time","request_count",
                            "api_name","error_rate_boost","rt_multiplier"]
               if not any(c in d for d in direct)]

    mapping = _report(direct=direct, derived=derived)
    return df[["timestamp","api_name","response_time","status_code","success",
               "error_type","request_count","error_rate_boost","rt_multiplier","event_label"]], mapping


# -----------------------------------------------------------------------------
# STEP 3 -- Feature engineering (mirrors generate_production_dataset.py exactly)
# -----------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rolling stats, lag features, EMA, and cyclical encoding per api_name.
    Then derive time-of-day flags, api_complexity, error_rate_boost, rt_multiplier.
    Returns a DataFrame with all CSV_COLS populated.
    """
    rng = np.random.default_rng(99)
    results = []

    for api_name, grp in df.groupby("api_name", sort=False):
        g = grp.copy().sort_values("timestamp").reset_index(drop=True)
        n = len(g)

        ts    = pd.DatetimeIndex(g["timestamp"])
        hour  = ts.hour.values
        dow   = ts.dayofweek.values

        # Time flags (mirrors generate_production_dataset.py exactly)
        is_weekend = (dow >= 5).astype(int)
        is_holiday = np.zeros(n, dtype=int)          # conservative default for external data
        is_low     = (is_weekend | is_holiday).astype(bool)
        is_peak    = ((hour >= 9) & (hour <= 16) & ~is_low).astype(int)
        is_preopen = ((hour >= 7) & (hour < 9)).astype(int)
        is_close   = ((hour >= 16) & (hour <= 18)).astype(int)

        rt_series  = pd.Series(g["response_time"].values, dtype=float)
        err_series = pd.Series(1 - g["success"].values,   dtype=float)

        # Rolling (window=60, min_periods=1) -- exact match to generate script
        rt_mean_60  = rt_series.rolling(60,  min_periods=1).mean().values
        rt_std_60   = rt_series.rolling(60,  min_periods=1).std().fillna(0).values
        err_rate_60 = err_series.rolling(60, min_periods=1).mean().values
        rt_var      = rt_std_60 ** 2
        err_vol     = err_series.rolling(30, min_periods=1).std().fillna(0).values

        # Lag features
        rt_mean_val = rt_series.mean()
        rt_lag1  = rt_series.shift(1).fillna(rt_mean_val).values
        rt_lag5  = rt_series.shift(5).fillna(rt_mean_val).values
        err_lag1 = err_series.shift(1).fillna(0).values

        # EMA
        rt_ema10  = rt_series.ewm(span=10).mean().values
        rt_ema30  = rt_series.ewm(span=30).mean().values
        err_ema10 = err_series.ewm(span=10).mean().values

        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin  = np.sin(2 * np.pi * dow / 7)
        dow_cos  = np.cos(2 * np.pi * dow / 7)

        # API metadata
        complexity = API_COMPLEXITY.get(api_name, 1.0)
        high_freq  = 1 if api_name == "crypto_api" else 0

        # Derive error_rate_boost from observed rolling error rate
        # (how much above the 7% baseline this API is currently running)
        BASELINE_ERROR_RATE = 0.07
        error_rate_boost = np.maximum(0.0, err_rate_60 - BASELINE_ERROR_RATE).round(4)

        # rt_multiplier: current RT relative to the 30-span EMA baseline
        rt_multiplier = np.clip(
            np.where(rt_ema30 > 0, g["response_time"].values / rt_ema30, 1.0),
            1.0, 10.0,
        ).round(4)

        out = pd.DataFrame({
            "timestamp":                  g["timestamp"],
            "api_name":                   api_name,
            "response_time":              g["response_time"].round(2),
            "status_code":                g["status_code"],
            "success":                    g["success"],
            "error_type":                 g["error_type"],
            "request_count":              g["request_count"],
            "hour":                       hour,
            "day_of_week":                dow,
            "is_weekend":                 is_weekend,
            "is_holiday":                 is_holiday,
            "is_market_hours":            is_peak,
            "is_financial_peak":          is_peak,
            "high_frequency_api":         high_freq,
            "response_time_rolling_mean": np.round(rt_mean_60,  4),
            "response_time_rolling_std":  np.round(rt_std_60,   4),
            "error_rate_rolling":         np.round(err_rate_60, 4),
            "response_time_variance":     np.round(rt_var,      4),
            "error_volatility":           np.round(err_vol,     4),
            "response_time_lag_1":        np.round(rt_lag1,     2),
            "response_time_lag_5":        np.round(rt_lag5,     2),
            "error_rate_lag_1":           np.round(err_lag1,    4),
            "response_time_ema_10":       np.round(rt_ema10,    4),
            "response_time_ema_30":       np.round(rt_ema30,    4),
            "error_rate_ema_10":          np.round(err_ema10,   4),
            "hour_sin":                   np.round(hour_sin,    6),
            "hour_cos":                   np.round(hour_cos,    6),
            "dow_sin":                    np.round(dow_sin,     6),
            "dow_cos":                    np.round(dow_cos,     6),
            "api_complexity":             complexity,
            "error_rate_boost":           error_rate_boost,
            "rt_multiplier":              rt_multiplier,
            "event_label":                g["event_label"],
            # DB-only extras
            "is_pre_open":                is_preopen,
            "is_market_close":            is_close,
        })
        results.append(out)

    return pd.concat(results, ignore_index=True)


# -----------------------------------------------------------------------------
# STEP 4 -- Write to CSV and SQLite
# -----------------------------------------------------------------------------

def append_to_csv(df: pd.DataFrame):
    """Append new rows to banking_api_features.csv, preserving column order."""
    df_csv = df[CSV_COLS].copy()
    # Convert timestamp to string matching existing format
    df_csv["timestamp"] = df_csv["timestamp"].astype(str)
    df_csv.to_csv(CSV_PATH, mode="a", header=False, index=False)
    print(f"  [OK] CSV updated: {CSV_PATH}  (+{len(df_csv):,} rows)")


def insert_into_db(df: pd.DataFrame):
    """Insert new rows into the api_telemetry table."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur  = conn.cursor()
        max_id = cur.execute("SELECT COALESCE(MAX(id),0) FROM api_telemetry").fetchone()[0]

        # Build DB frame -- include all columns the table has
        db_cols = [
            "timestamp","api_name","response_time","status_code","success","error_type",
            "request_count","hour","day_of_week","is_weekend","is_holiday",
            "is_market_hours","is_financial_peak","is_pre_open","is_market_close",
            "response_time_rolling_mean","response_time_rolling_std","error_rate_rolling",
            "response_time_variance","error_volatility",
            "response_time_lag_1","response_time_lag_5","error_rate_lag_1",
            "response_time_ema_10","response_time_ema_30","error_rate_ema_10",
            "hour_sin","hour_cos","dow_sin","dow_cos",
            "high_frequency_api","api_complexity","event_label",
            "error_rate_boost","rt_multiplier",
        ]
        df_db = df[[c for c in db_cols if c in df.columns]].copy()
        df_db["id"] = range(max_id + 1, max_id + 1 + len(df_db))
        df_db.to_sql("api_telemetry", conn, if_exists="append", index=False)
        conn.commit()
        print(f"  [OK] SQLite updated: {DB_PATH}  (+{len(df_db):,} rows, ids {max_id+1}..{max_id+len(df_db)})")
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# STEP 5 -- Report
# -----------------------------------------------------------------------------

def print_report(results: list, rows_before: int, rows_after: int):
    w = 72
    print(f"\n{'='*w}")
    print(f"  INTEGRATION REPORT")
    print(f"{'='*w}")
    print(f"  Rows before : {rows_before:>10,}")
    print(f"  Rows added  : {rows_after - rows_before:>10,}")
    print(f"  Rows after  : {rows_after:>10,}")
    print(f"{'-'*w}")

    total_added  = 0
    total_failed = 0
    for r in results:
        status = r["status"]
        name   = r["key"]
        desc   = DATASET_REGISTRY[name]["description"]
        print(f"\n  [{status}] {name.upper()}  --  {desc}")

        if status == "[OK]":
            n      = r["rows_added"]
            fr     = r["failure_rate"]
            total_added += n
            print(f"       Rows added    : {n:,}")
            print(f"       Failure rate  : {fr:.2%}")
            print(f"       API name(s)   : {r['api_names']}")
            print(f"       Direct mapped : {', '.join(r['mapping']['direct']) or '--'}")
            print(f"       Derived       : {', '.join(r['mapping']['derived']) or '--'}")
            if r.get("elapsed"):
                print(f"       Time          : {r['elapsed']:.1f}s")
        else:
            total_failed += 1
            print(f"       Error: {r.get('error','unknown')}")

    print(f"\n{'-'*w}")
    print(f"  Datasets succeeded : {len(results) - total_failed}/{len(results)}")
    print(f"  Total rows added   : {total_added:,}")
    if args.dry_run:
        print(f"\n  [WARN]  DRY RUN -- no files were modified")
    print(f"{'='*w}\n")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

NORMALISER_MAP = {
    "nab":        normalise_nab,
    "creditcard": normalise_creditcard,
    "paysim":     normalise_paysim,
    "network":    normalise_network,
    "weblog":     normalise_weblog,
}


def main():
    # -- Check CSV and DB exist ------------------------------------------------
    if not CSV_PATH.exists():
        print(f"[FAIL] CSV not found: {CSV_PATH}")
        print("  Run generate_production_dataset.py first.")
        sys.exit(1)
    if not DB_PATH.exists():
        print(f"[FAIL] SQLite DB not found: {DB_PATH}")
        print("  Run generate_production_dataset.py first.")
        sys.exit(1)

    rows_before = sum(1 for _ in open(CSV_PATH)) - 1   # fast line count
    print(f"Existing CSV rows : {rows_before:,}")
    print(f"Datasets to run   : {DATASETS_TO_RUN}\n")

    # -- Prerequisites ---------------------------------------------------------
    if not args.skip_download:
        print("--- Step 1: Checking prerequisites -------------------------------")
        if not ensure_kaggle():
            print("[FAIL] Could not install kaggle package. Aborting.")
            sys.exit(1)
        if not check_kaggle_credentials():
            sys.exit(1)
        print()

    # -- Process each dataset --------------------------------------------------
    all_results   = []
    all_frames    = []

    for key in DATASETS_TO_RUN:
        meta     = DATASET_REGISTRY[key]
        dest_dir = RAW_DIR / key
        result   = {"key": key, "status": "[FAIL]"}

        print(f"--- {key.upper()}  ({meta['description']}) " + "-"*max(1, 50-len(key)))

        try:
            # Step 1 -- Download
            if not args.skip_download:
                ok = download_dataset(meta["slug"], dest_dir)
                if not ok:
                    result["error"] = f"Download failed for {meta['slug']}"
                    all_results.append(result)
                    print()
                    continue
            else:
                if not dest_dir.exists():
                    result["error"] = f"--skip_download set but {dest_dir} not found"
                    all_results.append(result)
                    print()
                    continue
                print(f"  (skipping download, using {dest_dir})")

            # Step 2 -- Normalise
            print(f"  Normalising ...")
            t0 = time.time()
            normaliser = NORMALISER_MAP[key]
            df_norm, mapping = normaliser(dest_dir, meta["api_name"], args.max_rows)
            print(f"  [OK] Normalised: {len(df_norm):,} rows  "
                  f"(failure rate: {1 - df_norm['success'].mean():.2%})")

            # Step 3 -- Feature engineering
            print(f"  Engineering features ...")
            df_feat = engineer_features(df_norm)
            elapsed = time.time() - t0

            result.update({
                "status":       "[OK]",
                "rows_added":   len(df_feat),
                "failure_rate": 1 - df_feat["success"].mean(),
                "api_names":    df_feat["api_name"].unique().tolist(),
                "mapping":      mapping,
                "elapsed":      elapsed,
            })
            all_frames.append(df_feat)
            print(f"  [OK] Features engineered in {elapsed:.1f}s")

        except Exception as e:
            import traceback
            result["error"] = str(e)
            print(f"  [FAIL] Failed: {e}")
            if os.environ.get("FCE_DEBUG"):
                traceback.print_exc()

        all_results.append(result)
        print()

    # -- Step 4 -- Write --------------------------------------------------------
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        # Sort by (api_name, timestamp) so sequences within each API are coherent
        combined = combined.sort_values(["api_name", "timestamp"]).reset_index(drop=True)

        if not args.dry_run:
            print("--- Step 4: Writing to CSV and SQLite ----------------------------")
            append_to_csv(combined)
            insert_into_db(combined)
            rows_after = rows_before + len(combined)
        else:
            rows_after = rows_before + len(combined)
            print("--- Step 4: DRY RUN -- skipping write -----------------------------")
            print(f"  Would append {len(combined):,} rows")
    else:
        rows_after = rows_before
        print("--- Step 4: Nothing to write (all datasets failed) ---------------")

    # -- Step 5 -- Report -------------------------------------------------------
    print_report(all_results, rows_before, rows_after)


if __name__ == "__main__":
    main()
