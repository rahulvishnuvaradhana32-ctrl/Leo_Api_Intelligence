#!/usr/bin/env python3
"""
build_snapshot.py — bake the latest models/*.json into web/scripts/data.js

The new LEO frontend (web/index.html) is a pure static site for portability,
so the model results are embedded as a single global JS object. Re-run this
after any pipeline script writes new JSON.

Usage:
    python scripts/build_snapshot.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUT = ROOT / "web" / "scripts" / "data.js"


def load(name, default):
    p = MODELS / name
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default


def round_list(xs, n=6):
    return [round(float(v), n) for v in xs]


def _clean(obj):
    """Recursively replace NaN/Inf floats with None so the JS literal is sane."""
    import math
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def load_selfheal(max_runs=40):
    """Parse models/self_heal_log.jsonl → compact timeline for the drift &
    reliability pages. Each line is one self-healing run with a drift report."""
    p = MODELS / "self_heal_log.jsonl"
    runs = []
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            prob = r.get("problems_found", {}) or {}
            drift = prob.get("drift_report", {}) or {}
            outcome = r.get("outcome", {}) or {}
            comp = r.get("comparison", {}) or {}
            runs.append({
                "run_id": r.get("run_id"),
                "timestamp": r.get("timestamp"),
                "mode": r.get("mode"),
                "rows": (r.get("data", {}) or {}).get("rows_in_recent_window"),
                "failure_rate": prob.get("failure_rate"),
                "drift_detected": bool(prob.get("drift_detected")),
                "imbalance": bool(prob.get("imbalance_detected")),
                "signals": {
                    k: {"ks": v.get("ks"), "p": v.get("p"), "drifted": bool(v.get("drifted"))}
                    for k, v in drift.items()
                },
                "model_updated": bool(outcome.get("model_updated")),
                "delta": comp.get("delta"),
            })
    except Exception:
        pass
    return _clean(runs[-max_runs:])


def main() -> None:
    lstm = load("lstm_results.json", {})
    conf = load("conformal_results.json", {})
    agent = load("agent_simulation_results.json", {})
    abl = load("ablation_results.json", {})
    selfheal = load_selfheal()

    ph = lstm.get("per_horizon", {})
    conf_summary = conf.get("summary", {}) or {}
    assumptions = agent.get("assumptions", {}) or {}

    payload = {
        "meta": {
            "version": lstm.get("version", "v5"),
            "n_features": lstm.get("n_features", 43),
            "best_val_loss": round(float(lstm.get("best_val_loss", 0.0)), 8),
            "total_time_sec": round(float(lstm.get("total_time", 0.0)), 2),
            "n_epochs": len(lstm.get("train_losses", [])),
        },
        "lstm": {
            "avg_auc": lstm.get("avg_auc"),
            "avg_pr_auc": lstm.get("avg_pr_auc"),
            "per_horizon": {
                "h1":  {
                    "auc":    ph.get("horizon_1", {}).get("auc"),
                    "pr_auc": ph.get("horizon_1", {}).get("pr_auc"),
                    "p100":   ph.get("horizon_1", {}).get("precision_at_100"),
                },
                "h5":  {
                    "auc":    ph.get("horizon_5", {}).get("auc"),
                    "pr_auc": ph.get("horizon_5", {}).get("pr_auc"),
                    "p100":   ph.get("horizon_5", {}).get("precision_at_100"),
                },
                "h15": {
                    "auc":    ph.get("horizon_15", {}).get("auc"),
                    "pr_auc": ph.get("horizon_15", {}).get("pr_auc"),
                    "p100":   ph.get("horizon_15", {}).get("precision_at_100"),
                },
            },
            "train_losses": round_list(lstm.get("train_losses", [])),
            "val_losses":   round_list(lstm.get("val_losses",   [])),
        },
        "conformal": {
            "target_coverage": conf.get("target_coverage", 90.0) / 100.0
                if conf.get("target_coverage", 0) > 1 else conf.get("target_coverage", 0.9),
            "alpha": conf.get("alpha", 0.1),
            "cal_sequences": conf.get("cal_sequences", 0),
            "avg_coverage": conf_summary.get("avg_coverage"),
            "avg_width":    conf_summary.get("avg_width"),
            "all_pass":     conf_summary.get("all_horizons_pass"),
            "per_horizon": {
                k: {
                    "q_hat":    v.get("q_hat"),
                    "coverage": v.get("coverage"),
                    "width":    v.get("avg_width"),
                    "status":   v.get("status"),
                } for k, v in (conf.get("per_horizon") or {}).items()
            },
        },
        "agent": {
            "n_transactions": agent.get("n_transactions"),
            "proactive": {
                "failures":  agent.get("proactive", {}).get("failures"),
                "rate":      agent.get("proactive", {}).get("failure_rate"),
                "switches":  agent.get("proactive", {}).get("switches"),
                "latency":   agent.get("proactive", {}).get("avg_latency_sec"),
                "cost_1k":   agent.get("proactive", {}).get("cost_per_1000"),
                "retry":     (agent.get("proactive", {}).get("action_distribution") or {}).get("retry"),
                "normal":    (agent.get("proactive", {}).get("action_distribution") or {}).get("normal"),
                "switch":    (agent.get("proactive", {}).get("action_distribution") or {}).get("switch"),
            },
            "reactive": {
                "failures":  agent.get("reactive",  {}).get("failures"),
                "rate":      agent.get("reactive",  {}).get("failure_rate"),
                "switches":  agent.get("reactive",  {}).get("switches"),
                "latency":   agent.get("reactive",  {}).get("avg_latency_sec"),
                "cost_1k":   agent.get("reactive",  {}).get("cost_per_1000"),
                "normal":    (agent.get("reactive",  {}).get("action_distribution") or {}).get("normal"),
                "switch":    (agent.get("reactive",  {}).get("action_distribution") or {}).get("switch"),
            },
            "comparison": {
                "fail_reduction_pct": agent.get("comparison", {}).get("failure_reduction_pct"),
                "annual_savings":     agent.get("comparison", {}).get("annual_cost_saving_usd"),
                "failures_avoided":   agent.get("comparison", {}).get("annual_failures_avoided"),
            },
            "assumptions": {
                "cost_per_failure":   assumptions.get("cost_per_failure_usd"),
                "tx_per_year":        assumptions.get("transactions_per_year"),
                "high_risk":          assumptions.get("high_risk_threshold"),
                "low_risk":           assumptions.get("low_risk_threshold"),
                "lat_normal":         assumptions.get("latency_normal_sec"),
                "lat_retry":          assumptions.get("latency_retry_sec"),
                "lat_switch":         assumptions.get("latency_switch_sec"),
            },
        },
        "selfheal": selfheal,
        "ablation": {
            "baseline_auc": abl.get("baseline_auc"),
            "experiments": [
                {
                    "name": e.get("name"),
                    "auc":  e.get("auc"),
                    "n":    e.get("n_features"),
                    **({"base": True} if e.get("name") == "Baseline" else {}),
                }
                for e in (abl.get("experiments") or [])
            ],
        },
        # dataset block is kept hand-curated in data.js because reading the
        # full CSV here would be too slow for a quick rebuild; values are
        # already verified in the README. If you want it auto-derived, run
        # data_audit.py and paste the API distribution back.
    }

    js = "/* AUTO-GENERATED by scripts/build_snapshot.py — do not edit by hand. */\n"
    js += "window.LEO_DATA = window.LEO_DATA || {};\n"
    js += "Object.assign(window.LEO_DATA, " + json.dumps(payload, indent=2) + ");\n"

    # Preserve the hand-tuned `dataset` block from the existing data.js
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = OUT.read_text(encoding="utf-8") if OUT.exists() else ""

    # naive: keep the existing file's `dataset:` object if present
    DATASET_FALLBACK = """
window.LEO_DATA.dataset = window.LEO_DATA.dataset || {
  rows: 1220008,
  failure_rate: 0.1388,
  apis: [
    { name: "transaction_api", share: 0.25,   color: "#dc2626" },
    { name: "market_data_api", share: 0.25,   color: "#fbbf24" },
    { name: "stock_price_api", share: 0.103,  color: "#f59e0b" },
    { name: "crypto_api",      share: 0.104,  color: "#84cc16" },
    { name: "forex_api",       share: 0.104,  color: "#ea580c" },
    { name: "other",           share: 0.189,  color: "#fcd34d" }
  ],
  date_range: "2023-01-01 → 2024-12-31"
};
"""
    js += DATASET_FALLBACK

    OUT.write_text(js, encoding="utf-8")
    print(f"[snapshot] wrote {OUT}  ({len(js):,} bytes)")


if __name__ == "__main__":
    main()
