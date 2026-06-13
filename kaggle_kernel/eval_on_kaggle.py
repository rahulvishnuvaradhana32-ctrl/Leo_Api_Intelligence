#!/usr/bin/env python3
"""
eval_on_kaggle.py — LEO API Intelligence: Full Evaluation Suite on Kaggle GPU

Runs all 4 evaluations in one notebook session:
  1. LSTM vs Baselines (LR, RF, XGBoost)
  2. Ablation Study (feature group importance)
  3. Conformal Prediction (uncertainty bands)
  4. Agent Simulation (proactive vs reactive switching)

Auto-detects all file paths. Upload the trained model as a Kaggle dataset
named "leo-api-models" containing: stress_test_best_model.pth, scaler.pkl

Outputs saved to /kaggle/working/models/:
  evaluation_results.json
  ablation_results.json + ablation_results.png
  conformal_results.json + conformal_calibration.png + conformal_results.csv
  agent_simulation_results.json + agent_simulation_chart.png
"""

import os, sys, json, time, warnings, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
warnings.filterwarnings("ignore")

# ── Output directory ───────────────────────────────────────────────────────────
OUT_DIR = "/kaggle/working/models"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared constants ───────────────────────────────────────────────────────────
HORIZONS     = [1, 5, 15]
SEQ_LEN      = 30
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
    "latency_diff_1", "latency_diff_5",
    "error_rate_diff_1", "error_rate_diff_5",
    "latency_spike", "error_burst", "instability_index",
    "latency_slope", "error_slope",
    "traffic_change", "burst_ratio",
    "avg_error_rate_others", "max_error_rate_others",
    "n_apis_elevated", "corr_with_similar_api",
    "systemic_stress_index",
]

# ── Path auto-detection ────────────────────────────────────────────────────────
def _find_csv():
    for p in Path("/kaggle/input").rglob("banking_api_features_v7.csv"):
        return str(p)
    csvs = list(Path("/kaggle/input").rglob("*.csv"))
    if csvs:
        return str(max(csvs, key=lambda x: x.stat().st_size))
    raise FileNotFoundError("No CSV found in /kaggle/input")

def _find_file(name):
    # same session as training — check working dir first
    wp = Path(OUT_DIR) / name
    if wp.exists():
        return str(wp)
    for p in Path("/kaggle/input").rglob(name):
        return str(p)
    raise FileNotFoundError(f"{name} not found in /kaggle/working/models or /kaggle/input")

CSV_PATH    = _find_csv()
MODEL_PATH  = _find_file("stress_test_best_model.pth")
SCALER_PATH = _find_file("scaler.pkl")

print("=" * 60)
print("LEO API Intelligence — Full Evaluation Suite")
print("=" * 60)
print(f"  CSV   : {CSV_PATH}")
print(f"  Model : {MODEL_PATH}")
print(f"  Scaler: {SCALER_PATH}")
print()

# ── Shared model architecture (must match run_lstm_training.py exactly) ────────
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return (w.unsqueeze(-1) * x).sum(dim=1)

class MultiHorizonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 n_horizons, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm_out = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.layer_norm = nn.LayerNorm(self.lstm_out)
        self.attn_pool  = AttentionPooling(self.lstm_out)
        self.dropout    = nn.Dropout(dropout)
        self.heads      = nn.ModuleList([nn.Linear(self.lstm_out, 1)
                                         for _ in range(n_horizons)])
    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.layer_norm(out)
        out    = self.attn_pool(out)
        out    = self.dropout(out)
        return torch.cat([h(out) for h in self.heads], dim=1)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    def forward(self, logits, targets):
        pw  = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
        p_t = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return ((1 - p_t) ** self.gamma * bce).mean()

def _detect_dims(sd):
    hidden = sd["lstm.weight_hh_l0"].shape[1]
    n_in   = sd["lstm.weight_ih_l0"].shape[1]
    bidir  = "lstm.weight_ih_l0_reverse" in sd
    n_h    = sum(1 for k in sd if k.startswith("heads.") and k.endswith(".weight"))
    return n_in, hidden, n_h, bidir

def load_model_and_scaler():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd     = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    n_in, hidden, n_h, bidir = _detect_dims(sd)
    model  = MultiHorizonLSTM(n_in, hidden, 2, n_h, bidirectional=bidir).to(device)
    model.load_state_dict(sd)
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    print(f"  Model: input={n_in} hidden={hidden} horizons={n_h} bidir={bidir} [{device}]")
    return model, scaler, device

# ── Shared dataset ─────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len, horizons, feat_cols):
        self.seq_len  = seq_len
        self.horizons = horizons
        self.data     = df[feat_cols].to_numpy(dtype=np.float32)
        self.targets  = df["success"].to_numpy(dtype=np.float32)
        self.max_start = len(self.data) - seq_len - max(horizons) + 1
    def __len__(self):
        return max(0, self.max_start)
    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_len].copy()
        tgt = [float(1 - self.targets[idx + self.seq_len + h - 1]) for h in self.horizons]
        return {"sequence": torch.from_numpy(seq),
                "targets":  torch.tensor(tgt, dtype=torch.float32)}

def load_dataframe(pre2025=False):
    print(f"  Loading CSV ...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if pre2025:
        df = df[df["timestamp"] < "2025-01-01"].copy()
    if df["success"].dtype == object:
        df["success"] = df["success"].map({"True": 1, "False": 0}).fillna(0).astype(int)
    else:
        df["success"] = df["success"].astype(int)
    avail = [c for c in FEATURE_COLS if c in df.columns]
    df[avail] = df[avail].fillna(0).astype(np.float32)
    print(f"  {len(df):,} rows  failure_rate={1-df['success'].mean():.3f}  features={len(avail)}")
    return df, avail


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LSTM vs Baselines
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluate():
    print("\n" + "=" * 60)
    print("SECTION 1 — LSTM vs Baselines")
    print("=" * 60)

    df, avail = load_dataframe()
    X = df[avail].values.astype(np.float32)
    y = (1 - df["success"]).values.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    results = {"baselines": {}, "lstm": {}, "comparison": None}

    print("  Training Logistic Regression ...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_tr, y_tr)
    lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    results["baselines"]["LogisticRegression"] = {"auc": float(lr_auc)}
    print(f"    AUC: {lr_auc:.4f}")

    print("  Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_auc = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
    results["baselines"]["RandomForest"] = {"auc": float(rf_auc)}
    print(f"    AUC: {rf_auc:.4f}")

    print("  Training XGBoost ...")
    xgb_m = xgb.XGBClassifier(n_estimators=100, random_state=42,
                               eval_metric="logloss", device="cuda",
                               tree_method="hist")
    xgb_m.fit(X_tr, y_tr, verbose=False)
    xgb_auc = roc_auc_score(y_te, xgb_m.predict_proba(X_te)[:, 1])
    results["baselines"]["XGBoost"] = {"auc": float(xgb_auc)}
    print(f"    AUC: {xgb_auc:.4f}")

    print("  Evaluating LSTM ...")
    model, scaler, device = load_model_and_scaler()
    df_sc = df[avail + ["success"]].copy()
    df_sc[avail] = scaler.transform(df_sc[avail]).astype(np.float32)
    dataset = TimeSeriesDataset(df_sc, SEQ_LEN, HORIZONS, avail)
    n = len(dataset)
    te_n = int(0.15 * n)
    _, te_ds = random_split(dataset, [n - te_n, te_n],
                            generator=torch.Generator().manual_seed(42))
    loader = DataLoader(te_ds, batch_size=512, shuffle=False,
                        num_workers=2, pin_memory=True)
    all_t, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            out  = model(batch["sequence"].to(device))
            prob = torch.sigmoid(out)
            all_t.append(batch["targets"].numpy())
            all_p.append(prob.cpu().numpy())
    all_t = np.vstack(all_t); all_p = np.vstack(all_p)
    per_h, avg = {}, 0.0
    for i, h in enumerate(HORIZONS):
        a = roc_auc_score(all_t[:, i], all_p[:, i])
        per_h[f"horizon_{h}"] = float(a); avg += a
        print(f"    h={h} AUC: {a:.4f}")
    avg /= len(HORIZONS)
    results["lstm"] = {"avg_auc": float(avg), "per_horizon": per_h}
    best_bl = max(v["auc"] for v in results["baselines"].values())
    impr    = (avg - best_bl) / best_bl * 100
    results["comparison"] = {"best_baseline_auc": float(best_bl),
                              "lstm_avg_auc": float(avg),
                              "improvement_percent": float(impr)}
    print(f"\n  LSTM avg AUC: {avg:.4f}  ({impr:+.1f}% over best baseline)")
    out_path = f"{OUT_DIR}/evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_path}")
    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Ablation Study
# ══════════════════════════════════════════════════════════════════════════════
ABLATION_GROUPS = [
    ("No Event Signals",     ["error_rate_boost", "rt_multiplier"]),
    ("No Rolling Stats",     ["response_time_rolling_mean","response_time_rolling_std",
                              "error_rate_rolling","response_time_variance","error_volatility"]),
    ("No Lag Features",      ["response_time_lag_1","response_time_lag_5","error_rate_lag_1"]),
    ("No EMA Features",      ["response_time_ema_10","response_time_ema_30","error_rate_ema_10"]),
    ("No Cyclical Enc.",     ["hour_sin","hour_cos","dow_sin","dow_cos"]),
    ("No API Flags",         ["high_frequency_api","api_complexity"]),
    ("No Precursor Signals", ["latency_diff_1","latency_diff_5","error_rate_diff_1",
                              "error_rate_diff_5","latency_spike","error_burst",
                              "instability_index","latency_slope","error_slope"]),
    ("No Cross-API",         ["avg_error_rate_others","max_error_rate_others",
                              "n_apis_elevated","corr_with_similar_api","systemic_stress_index"]),
]

def _abl_train(tr_ds, vl_ds, feat_cols, device, epochs=5, bs=128):
    model = MultiHorizonLSTM(len(feat_cols), 128, 2, len(HORIZONS),
                             dropout=0.3, bidirectional=True).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    crit  = FocalLoss(gamma=2.0)
    for _ in range(epochs):
        model.train()
        for b in DataLoader(tr_ds, batch_size=bs, shuffle=True,
                            num_workers=2, pin_memory=True):
            opt.zero_grad()
            crit(model(b["sequence"].to(device)), b["targets"].to(device)).backward()
            opt.step()
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for b in DataLoader(vl_ds, batch_size=512, shuffle=False,
                            num_workers=2, pin_memory=True):
            p = torch.sigmoid(model(b["sequence"].to(device)))
            all_t.append(b["targets"].numpy()); all_p.append(p.cpu().numpy())
    all_t = np.vstack(all_t); all_p = np.vstack(all_p)
    aucs  = [roc_auc_score(all_t[:, i], all_p[:, i]) for i in range(len(HORIZONS))]
    del model; gc.collect(); torch.cuda.empty_cache()
    return float(np.mean(aucs)), aucs

def run_ablation():
    print("\n" + "=" * 60)
    print("SECTION 2 — Ablation Study")
    print("=" * 60)

    df, avail = load_dataframe(pre2025=True)
    df_sub    = df.tail(300_000).copy()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_ds(feats):
        cols  = [c for c in feats if c in avail]
        df2   = df_sub[cols + ["success"]].copy()
        n     = int(0.8 * len(df2))
        sc    = StandardScaler()
        df2[cols] = sc.fit_transform(df2[cols].values)
        tr = Subset(TimeSeriesDataset(df2.iloc[:n], SEQ_LEN, HORIZONS, cols),
                    range(min(50_000, len(TimeSeriesDataset(df2.iloc[:n], SEQ_LEN, HORIZONS, cols)))))
        vl = Subset(TimeSeriesDataset(df2.iloc[n:], SEQ_LEN, HORIZONS, cols),
                    range(min(10_000, len(TimeSeriesDataset(df2.iloc[n:], SEQ_LEN, HORIZONS, cols)))))
        return tr, vl, cols

    print("  Training baseline (all features) ...")
    tr, vl, cols = make_ds(avail)
    base_auc, _  = _abl_train(tr, vl, cols, device)
    print(f"    Baseline AUC: {base_auc:.4f}")

    experiments = []
    for name, remove in ABLATION_GROUPS:
        feat_sub = [f for f in avail if f not in remove]
        print(f"  {name} (-{len(avail)-len(feat_sub)} feats) ...")
        tr, vl, cols = make_ds(feat_sub)
        auc, h_aucs  = _abl_train(tr, vl, cols, device)
        drop  = base_auc - auc
        label = "Critical" if drop > 0.02 else "Significant" if drop > 0.005 else "Minor"
        print(f"    AUC: {auc:.4f}  drop: {drop:+.4f}  [{label}]")
        experiments.append({"name": name, "auc": auc, "auc_drop": round(drop, 6),
                             "importance": label,
                             "per_horizon_auc": {f"h{HORIZONS[i]}": h_aucs[i]
                                                  for i in range(len(HORIZONS))}})

    exps_sorted = sorted(experiments, key=lambda x: x["auc_drop"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#ef4444" if e["importance"]=="Critical" else
               "#f97316" if e["importance"]=="Significant" else "#22c55e"
               for e in exps_sorted]
    ax.barh([e["name"] for e in exps_sorted],
            [e["auc_drop"] for e in exps_sorted], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("AUC Drop (higher = more important)")
    ax.set_title("LEO API Intelligence — Feature Group Ablation")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ablation_results.png", dpi=120); plt.close(fig)

    result = {"baseline_auc": base_auc, "experiments": exps_sorted}
    with open(f"{OUT_DIR}/ablation_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {OUT_DIR}/ablation_results.json + ablation_results.png")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Conformal Prediction
# ══════════════════════════════════════════════════════════════════════════════
def run_conformal(alpha=0.10):
    cov_target = round((1 - alpha) * 100, 1)
    print("\n" + "=" * 60)
    print(f"SECTION 3 — Conformal Prediction (target: {cov_target}%)")
    print("=" * 60)

    df, avail  = load_dataframe(pre2025=True)
    model, scaler, device = load_model_and_scaler()
    df_sc      = df[avail + ["success"]].copy()
    df_sc[avail] = scaler.transform(df_sc[avail]).astype(np.float32)
    n   = len(df_sc)
    s60 = int(0.60 * n); s80 = int(0.80 * n)

    def get_preds(ds):
        loader = DataLoader(ds, batch_size=512, shuffle=False,
                            num_workers=2, pin_memory=True)
        all_t, all_p = [], []
        with torch.no_grad():
            for b in loader:
                p = torch.sigmoid(model(b["sequence"].to(device)))
                all_t.append(b["targets"].numpy()); all_p.append(p.cpu().numpy())
        return np.vstack(all_t), np.vstack(all_p)

    cal_ds_raw  = TimeSeriesDataset(df_sc.iloc[s60:s80], SEQ_LEN, HORIZONS, avail)
    test_ds_raw = TimeSeriesDataset(df_sc.iloc[s80:],    SEQ_LEN, HORIZONS, avail)
    cal_ds  = Subset(cal_ds_raw,  range(min(50_000, len(cal_ds_raw))))
    test_ds = Subset(test_ds_raw, range(min(50_000, len(test_ds_raw))))

    print("  Calibration inference ..."); cal_t,  cal_p  = get_preds(cal_ds)
    print("  Test inference ...");        test_t, test_p = get_preds(test_ds)

    per_horizon = {}
    fig, axes   = plt.subplots(1, 3, figsize=(15, 5))
    all_rows    = []

    for i, h in enumerate(HORIZONS):
        nc    = np.abs(cal_t[:, i] - cal_p[:, i])
        n_cal = len(nc)
        qlvl  = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_hat = float(np.quantile(nc, min(qlvl, 1.0)))
        lo    = np.clip(test_p[:, i] - q_hat, 0, 1)
        hi    = np.clip(test_p[:, i] + q_hat, 0, 1)
        cov   = float(np.mean((test_t[:, i] >= lo) & (test_t[:, i] <= hi)))
        wid   = float(np.mean(hi - lo))
        passed = bool(abs(cov - (1 - alpha)) < 0.05)
        per_horizon[f"horizon_{h}"] = {"q_hat": round(q_hat, 6),
                                        "empirical_coverage": round(cov, 4),
                                        "avg_interval_width": round(wid, 4),
                                        "pass": passed}
        print(f"  h={h:2d}: q_hat={q_hat:.4f}  coverage={cov:.3f}  "
              f"width={wid:.4f}  {'PASS' if passed else 'FAIL'}")

        ax   = axes[i]
        bins = np.linspace(0, 1, 11)
        bm, bc = [], []
        for j in range(len(bins)-1):
            mask = (test_p[:, i] >= bins[j]) & (test_p[:, i] < bins[j+1])
            if mask.sum() > 10:
                bm.append((bins[j]+bins[j+1])/2); bc.append(test_t[mask, i].mean())
        ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect")
        ax.plot(bm, bc, "o-", color="#3b82f6", label=f"h={h}")
        ax.axhline(1-alpha, color="#ef4444", ls=":", label=f"Target {cov_target}%")
        ax.set_xlabel("Predicted prob"); ax.set_ylabel("Empirical cov")
        ax.set_title(f"Horizon {h}  cov={cov:.3f}"); ax.legend(fontsize=8)
        ax.set_xlim(0,1); ax.set_ylim(0,1)

        for k in range(len(test_p)):
            all_rows.append({"horizon": h, "y_true": float(test_t[k,i]),
                             "y_pred": float(test_p[k,i]),
                             "lo": float(lo[k]), "hi": float(hi[k]),
                             "covered": bool((test_t[k,i]>=lo[k])&(test_t[k,i]<=hi[k]))})

    fig.suptitle(f"LEO API — Conformal Reliability (target {cov_target}%)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/conformal_calibration.png", dpi=120); plt.close(fig)
    summary = {"alpha": alpha, "coverage_target": cov_target, "per_horizon": per_horizon}
    with open(f"{OUT_DIR}/conformal_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(all_rows).to_csv(f"{OUT_DIR}/conformal_results.csv", index=False)
    print(f"  Saved → conformal_results.json + .png + .csv")
    del model; gc.collect(); torch.cuda.empty_cache()
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Agent Simulation
# ══════════════════════════════════════════════════════════════════════════════
COST_PER_FAILURE      = 50.0
TRANSACTIONS_PER_YEAR = 260_000
HIGH_RISK_THRESHOLD   = 0.65
LOW_RISK_THRESHOLD    = 0.35
LATENCY_NORMAL        = 0.12
LATENCY_RETRY         = 0.28
LATENCY_SWITCH        = 0.45
BACKUP_API = {
    "stock_price_api": "market_data_api",
    "crypto_api":      "market_data_api",
    "forex_api":       "market_data_api",
    "market_data_api": "transaction_api",
    "transaction_api": "market_data_api",
}

def run_agent_simulation(n_transactions=1000, seed=42):
    print("\n" + "=" * 60)
    print("SECTION 4 — Agent Simulation")
    print("=" * 60)

    rng   = np.random.default_rng(seed)
    model, scaler, device = load_model_and_scaler()
    df, avail = load_dataframe(pre2025=True)
    apis = ["stock_price_api","crypto_api","forex_api","market_data_api","transaction_api"]
    n_per = n_transactions // len(apis)

    all_seqs, all_outcomes = [], []
    for api in apis:
        sub = df[df["api_name"] == api].copy()
        sub[avail] = scaler.transform(sub[avail].fillna(0)).astype(np.float32)
        pool = np.arange(len(sub) - SEQ_LEN - max(HORIZONS))
        chosen = rng.choice(pool, size=min(n_per, len(pool)), replace=False)
        vals = sub[avail].values; tgts = sub["success"].values
        for idx in chosen:
            all_seqs.append(vals[idx:idx+SEQ_LEN])
            all_outcomes.append(float(1 - tgts[idx+SEQ_LEN]))
        print(f"  {api:<22} {len(chosen):,} seqs  failure={1-sub['success'].mean():.3f}")

    seqs     = np.array(all_seqs, dtype=np.float32)
    outcomes = np.array(all_outcomes)
    perm     = rng.permutation(len(seqs))
    seqs     = seqs[perm][:n_transactions]
    outcomes = outcomes[perm][:n_transactions]
    N        = len(seqs)

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(seqs).to(device))).cpu().numpy()
    probs_h1 = probs[:, 0]

    def sim_proactive(outcomes, probs):
        res = []
        for i in range(len(outcomes)):
            p = probs[i]; o = outcomes[i]
            if p > HIGH_RISK_THRESHOLD:
                res.append({"action":"switch",  "failure": float(rng.random() < o*0.25),
                            "latency": LATENCY_SWITCH,  "cost": float(rng.random()<o*0.25)*COST_PER_FAILURE})
            elif p > LOW_RISK_THRESHOLD:
                res.append({"action":"retry",   "failure": float(rng.random() < o*0.50),
                            "latency": LATENCY_RETRY,   "cost": float(rng.random()<o*0.50)*COST_PER_FAILURE})
            else:
                res.append({"action":"normal",  "failure": float(o),
                            "latency": LATENCY_NORMAL,  "cost": float(o)*COST_PER_FAILURE})
        return res

    def sim_reactive(outcomes):
        return [{"action":"post-fail-switch" if o else "normal",
                 "failure": float(o), "latency": LATENCY_SWITCH if o else LATENCY_NORMAL,
                 "cost": float(o)*COST_PER_FAILURE} for o in outcomes]

    pro_r  = sim_proactive(outcomes, probs_h1)
    reac_r = sim_reactive(outcomes)

    def metrics(res, label):
        fails   = sum(r["failure"] for r in res)
        fr      = fails / len(res)
        avg_lat = float(np.mean([r["latency"] for r in res]))
        ann_c   = fr * TRANSACTIONS_PER_YEAR * COST_PER_FAILURE
        print(f"  [{label}] failures={fails:.0f}  rate={fr:.3f}  "
              f"lat={avg_lat:.3f}s  annual=${ann_c:,.0f}")
        return {"failures": int(fails), "failure_rate": round(fr,4),
                "avg_latency": round(avg_lat,4), "annual_cost_usd": round(ann_c,2)}

    print()
    pro_m  = metrics(pro_r,  "Proactive")
    reac_m = metrics(reac_r, "Reactive ")
    fail_red  = (reac_m["failure_rate"] - pro_m["failure_rate"]) / max(reac_m["failure_rate"],1e-9)
    cost_save = reac_m["annual_cost_usd"] - pro_m["annual_cost_usd"]
    print(f"\n  Failure reduction: {fail_red*100:.1f}%  |  Annual saving: ${cost_save:,.0f}")

    cats   = ["Failures", "Annual Cost ($k)", "Avg Latency (ms)"]
    pro_v  = [pro_m["failures"],  pro_m["annual_cost_usd"]/1000,  pro_m["avg_latency"]*1000]
    reac_v = [reac_m["failures"], reac_m["annual_cost_usd"]/1000, reac_m["avg_latency"]*1000]
    x = np.arange(len(cats)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w/2, reac_v, w, label="Reactive",         color="#ef4444", alpha=0.85)
    ax.bar(x+w/2, pro_v,  w, label="Proactive (LSTM)", color="#22c55e", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_title("LEO API — Proactive vs Reactive Agent Simulation")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/agent_simulation_chart.png", dpi=130); plt.close(fig)

    output = {"n_transactions": N, "seed": seed,
              "proactive": pro_m, "reactive": reac_m,
              "comparison": {"failure_reduction_pct": round(fail_red*100,2),
                             "annual_cost_saving_usd": round(cost_save,2)},
              "assumptions": {"cost_per_failure_usd": COST_PER_FAILURE,
                              "high_risk_threshold": HIGH_RISK_THRESHOLD}}
    with open(f"{OUT_DIR}/agent_simulation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → agent_simulation_results.json + agent_simulation_chart.png")
    del model; gc.collect(); torch.cuda.empty_cache()
    return output


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()

    r1 = run_evaluate()
    r2 = run_ablation()
    r3 = run_conformal(alpha=0.10)
    r4 = run_agent_simulation(n_transactions=1000)

    print("\n" + "=" * 60)
    print("ALL SECTIONS COMPLETE")
    print("=" * 60)
    print(f"  LSTM avg AUC         : {r1['lstm']['avg_auc']:.4f}")
    print(f"  vs best baseline     : {r1['comparison']['improvement_percent']:+.1f}%")
    print(f"  Ablation baseline    : {r2['baseline_auc']:.4f}")
    print(f"  Conformal h=1 cov    : {r3['per_horizon']['horizon_1']['empirical_coverage']:.3f}")
    print(f"  Failure reduction    : {r4['comparison']['failure_reduction_pct']:.1f}%")
    print(f"  Annual saving        : ${r4['comparison']['annual_cost_saving_usd']:,.0f}")
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"\nAll outputs saved to: {OUT_DIR}/")

    # ── List all output files for the Kaggle Output tab ────────────────────
    print("\n" + "=" * 50)
    print("OUTPUT FILES READY FOR DOWNLOAD")
    print("=" * 50)
    files = sorted(Path(OUT_DIR).rglob("*"))
    total = 0
    for f in files:
        if f.is_file():
            kb = f.stat().st_size / 1024
            total += kb
            print(f"  {f.name:<45} {kb:>8.1f} KB")
    print(f"\n  Total: {total/1024:.2f} MB  |  {len([f for f in files if f.is_file()])} files")
    print("=" * 50)
    print("Go to Output tab (right panel) → download each file")
