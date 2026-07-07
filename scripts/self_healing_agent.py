#!/usr/bin/env python3
"""
self_healing_agent.py — Full detect → reroute → diagnose → fix → restore loop.

Extends agent_simulation.py's proactive approach with:
  3. Diagnose  — DiagnosticEngine classifies root cause from raw features
  4. Fix       — RemediationEngine applies targeted remedy; risk decays each tick
  5. Restore   — RouteEngine FAIL-BACK fires once fix brings primary risk < LO
                 for COOLDOWN consecutive ticks

Usage:
    python scripts/self_healing_agent.py
    python scripts/self_healing_agent.py --n_transactions 2000 --seed 99
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse, json, os, time
import numpy as np
import pandas as pd
import torch

# ── Import from existing scripts (build on top, don't duplicate) ──────────────
import agent_simulation as A
from route_engine       import RouteEngine
from diagnostic_engine  import DiagnosticEngine
from remediation_engine import RemediationEngine

SEQ_LEN  = 30
HI       = 0.65   # reroute threshold (matches agent_simulation.py)
LO       = 0.35   # fail-back / pre-warm threshold
COOLDOWN = 3      # consecutive healthy ticks before restore


# ── Core simulation loop ──────────────────────────────────────────────────────

def run_self_healing(all_outcomes, probs_h1, all_api, all_features_raw,
                     fail_rates, feature_cols, rng):
    """
    Process N transactions through the full 5-step loop.

    Returns:
        results  — per-transaction dict (action, failed, latency, etc.)
        events   — heal-cycle audit log (reroute / diagnose / fix / restore)
    """
    apis      = ["stock_price_api", "crypto_api", "forex_api",
                 "market_data_api", "transaction_api"]
    engines   = {api: RouteEngine(apis, primary=api, hi=HI, lo=LO, cooldown=COOLDOWN)
                 for api in apis}
    diagnostic   = DiagnosticEngine()
    remediation  = RemediationEngine()

    # Per-API risk scores fed into RouteEngine each tick
    api_risks = {api: float(fail_rates.get(api, 0.10)) for api in apis}
    # Track which APIs were rerouted (for restore detection)
    rerouted_apis = set()

    results = []
    events  = []

    for i, (outcome, p) in enumerate(zip(all_outcomes, probs_h1)):
        primary = all_api[i]

        # ── Update LSTM risk for this API ─────────────────────────────────────
        api_risks[primary] = float(p)

        # ── STEP 4 tick: active fixes decay risk each transaction ─────────────
        api_risks = remediation.tick_all(api_risks)

        # ── Build risk map and step RouteEngine ───────────────────────────────
        eng      = engines[primary]
        decision = eng.step(dict(api_risks))
        state    = decision["state"]
        active   = decision["active"]

        # ── Determine outcome and action ──────────────────────────────────────
        if state == "REROUTED" and active != primary:
            # ── STEP 2: Rerouted to backup ────────────────────────────────────
            actual_fail = int(rng.random() < fail_rates.get(active, 0.04))
            latency     = A.LATENCY_SWITCH
            action      = "switch"

            # ── STEP 3: Diagnose root cause ───────────────────────────────────
            feat_vec   = _to_feature_dict(all_features_raw[i], feature_cols)
            root_cause = diagnostic.diagnose(feat_vec)

            # ── STEP 4: Apply fix to primary (idempotent — no thrash) ─────────
            if not remediation.is_fixing(primary):
                remedy = remediation.apply_fix(primary, root_cause.category)
                rerouted_apis.add(primary)
                events.append({
                    "tx_index":       i,
                    "api":            primary,
                    "phase":          "reroute+diagnose+fix",
                    "root_cause":     root_cause.category,
                    "confidence":     root_cause.confidence,
                    "evidence":       root_cause.evidence,
                    "remedy_action":  remedy.action,
                    "remedy_params":  remedy.params,
                    "recovery_ticks": remedy.recovery_ticks,
                    "explain":        diagnostic.explain(root_cause),
                })

        elif state == "PREWARM":
            # Primary degrading but not critical yet — serve primary, arm standby
            actual_fail = int(outcome)
            latency     = A.LATENCY_NORMAL
            action      = "prewarm"

        elif state == "NORMAL" and active == primary and primary in rerouted_apis:
            # ── STEP 5: FAIL-BACK complete — primary restored ─────────────────
            actual_fail = int(outcome)
            latency     = A.LATENCY_NORMAL
            action      = "restore"
            rerouted_apis.discard(primary)
            events.append({
                "tx_index": i,
                "api":      primary,
                "phase":    "restore",
                "action":   "FAIL-BACK complete — primary restored",
            })

        elif state == "SYSTEMIC":
            # All routes degraded — hold and alert
            actual_fail = int(outcome)
            latency     = A.LATENCY_RETRY
            action      = "hold"

        else:
            # Normal operation or pre-warm retry
            if p > LO:
                actual_fail = int(outcome and (rng.random() < 0.5))
                latency     = A.LATENCY_RETRY
                action      = "retry"
            else:
                actual_fail = int(outcome)
                latency     = A.LATENCY_NORMAL
                action      = "normal"

        results.append({
            "action":       action,
            "failed":       actual_fail,
            "latency":      latency,
            "pred_prob":    float(p),
            "true_outcome": int(outcome),
            "route_state":  state,
        })

    return results, events


def _to_feature_dict(raw_vec, feature_cols: list) -> dict:
    return {col: float(raw_vec[j]) for j, col in enumerate(feature_cols)
            if j < len(raw_vec)}


# ── Print comparison table ────────────────────────────────────────────────────

def print_summary(metrics, events, n):
    diagnose_events = [e for e in events if e["phase"] == "reroute+diagnose+fix"]
    restore_events  = [e for e in events if e["phase"] == "restore"]
    root_causes     = pd.Series([e["root_cause"] for e in diagnose_events]).value_counts().to_dict()
    remedies        = pd.Series([e["remedy_action"] for e in diagnose_events]).value_counts().to_dict()

    W = 60
    print()
    print("=" * W)
    print("  LEO SELF-HEALING AGENT — FULL LOOP RESULTS")
    print("=" * W)
    fmt = "  {:<32} {}"
    print(fmt.format("Transactions",          f"{n:,}"))
    print(fmt.format("Failures",              f"{metrics['failures']:,} ({metrics['failure_rate']:.2%})"))
    print(fmt.format("API switches (reroute)", f"{metrics['switches']:,}"))
    print(fmt.format("Avg latency",           f"{metrics['avg_latency_sec']:.3f}s"))
    print(fmt.format("Est. annual cost",      f"${metrics['annual_cost_usd']:,.0f}"))
    print()
    print("  HEAL CYCLE")
    print(fmt.format("Reroutes + diagnoses",  len(diagnose_events)))
    print(fmt.format("Restores completed",    len(restore_events)))
    print(fmt.format("Root causes detected",  root_causes))
    print(fmt.format("Remedies applied",      remedies))
    print()

    if diagnose_events:
        print("  SAMPLE EVENTS (first 5 heal cycles)")
        print("-" * W)
        for e in diagnose_events[:5]:
            print(f"  tx={e['tx_index']:<5} api={e['api']}")
            print(f"         {e['explain']}")
            print(f"         fix={e['remedy_action']}  "
                  f"recovery_ticks={e['recovery_ticks']}")

    print("=" * W)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    print("\n=== LEO: Detect → Reroute → Diagnose → Fix → Restore ===\n")

    print("Loading model and scaler ...")
    model, scaler, feature_cols, n_heads = A.load_model(
        "models/stress_test_best_model.pth", "models/scaler.pkl"
    )
    print(f"  {len(feature_cols)} features, {n_heads} horizon heads")

    print(f"\nLoading data from {args.data} ...")
    df = pd.read_csv(args.data, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"] < "2025-01-01"].copy()
    print(f"  {len(df):,} rows available")

    apis = ["stock_price_api", "crypto_api", "forex_api",
            "market_data_api", "transaction_api"]

    fail_rates = {}
    for api in apis:
        sub = df[df["api_name"] == api]
        fr  = 1 - sub["success"].fillna(1).mean()
        fail_rates[api] = fr
        print(f"  {api:<22}  fail_rate={fr:.3f}")

    # Build sequence pool and keep raw (unscaled) last-step features
    print(f"\nBuilding {args.n_transactions} transaction sequences ...")
    n_per_api = args.n_transactions // len(apis)
    all_seqs, all_outcomes, all_api, all_features_raw = [], [], [], []

    for api in apis:
        sub = df[df["api_name"] == api].copy().reset_index(drop=True)
        sub[feature_cols] = sub[feature_cols].fillna(0).astype(np.float32)
        raw_X = sub[feature_cols].to_numpy(dtype=np.float32)

        seqs, outcomes = A.build_sequence_pool(df, api, feature_cols, scaler, n_per_api, rng)
        all_seqs.append(seqs)
        all_outcomes.append(outcomes)
        all_api.extend([api] * len(seqs))

        # Keep the unscaled last-step feature vector for diagnosis
        max_start = len(raw_X) - SEQ_LEN - 1
        starts    = rng.choice(max_start, size=len(seqs), replace=False)
        all_features_raw.extend([raw_X[s + SEQ_LEN - 1] for s in starts])

        print(f"  {api:<22}  {len(seqs):,} seqs  fail_in_sample={outcomes.mean():.3f}")

    all_seqs         = np.vstack(all_seqs).astype(np.float32)
    all_outcomes     = np.concatenate(all_outcomes)

    # Shuffle all together
    idx              = rng.permutation(len(all_seqs))
    all_seqs         = all_seqs[idx]
    all_outcomes     = all_outcomes[idx]
    all_api_shuf     = [all_api[i] for i in idx]
    all_features_raw = [all_features_raw[i] for i in idx]

    N                = min(args.n_transactions, len(all_seqs))
    all_seqs         = all_seqs[:N]
    all_outcomes     = all_outcomes[:N]
    all_features_raw = all_features_raw[:N]

    print(f"\n  Total transactions: {N:,}  "
          f"(overall fail rate: {all_outcomes.mean():.3f})")

    # LSTM inference
    print("\nRunning LSTM inference ...")
    t0       = time.time()
    probs    = A.predict_batch(model, all_seqs)
    probs_h1 = probs[:, 0]
    print(f"  {N:,} sequences in {time.time()-t0:.2f}s  "
          f"mean_p={probs_h1.mean():.3f}")

    # Self-healing simulation
    print("\nRunning self-healing simulation ...")
    results, events = run_self_healing(
        all_outcomes, probs_h1,
        all_api_shuf[:N], all_features_raw,
        fail_rates, feature_cols, rng,
    )

    metrics = A.compute_metrics(results, "Self-Healing (LEO)")
    print_summary(metrics, events, N)

    # Save
    os.makedirs("models", exist_ok=True)

    diagnose_events = [e for e in events if e["phase"] == "reroute+diagnose+fix"]
    restore_events  = [e for e in events if e["phase"] == "restore"]
    root_causes     = pd.Series([e["root_cause"] for e in diagnose_events]).value_counts().to_dict()
    remedies        = pd.Series([e["remedy_action"] for e in diagnose_events]).value_counts().to_dict()

    out = {
        "n_transactions":          N,
        "seed":                    args.seed,
        "metrics":                 metrics,
        "heal_cycle": {
            "reroutes_with_diagnosis": len(diagnose_events),
            "restores_completed":      len(restore_events),
            "root_cause_distribution": root_causes,
            "remedy_distribution":     remedies,
        },
        "sample_events": events[:100],
    }
    out_path = "models/self_healing_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default=os.path.join("data", "banking_api_features_v7.csv"),
                        help="Path to features CSV")
    parser.add_argument("--n_transactions", type=int, default=1000,
                        help="Number of transactions to simulate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    main(args)
