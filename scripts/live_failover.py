#!/usr/bin/env python3
"""
live_failover.py — WATCH the full flow live in your terminal:

  primary API healthy → telemetry degrades (injected incident) → the REAL
  model's risk climbs → crosses 0.55 → LEO reroutes to a warm standby of the
  SAME service → standby returns the IDENTICAL record (checksum match).

Real: the trained model + scaler score a real 30-step window from the dataset
at every tick. Modelled: the incident ramp (so it always trips) and the
primary/standby split over one shared source of truth.

Run:
    python scripts/live_failover.py
    python scripts/live_failover.py --api crypto_api --speed 0.5
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_simulation as A

THRESH = 0.55
MODEL_PATH, SCALER_PATH = "models/stress_test_best_model.pth", "models/scaler.pkl"
# raw telemetry features we ramp to simulate a degrading endpoint
STRESS = ["response_time", "response_time_rolling_mean", "response_time_rolling_std",
          "response_time_variance", "error_rate_rolling", "error_volatility",
          "error_rate_boost", "rt_multiplier", "latency_spike", "error_burst",
          "instability_index"]

C = {"g": "\x1b[32m", "y": "\x1b[33m", "r": "\x1b[31m", "b": "\x1b[36m",
     "d": "\x1b[90m", "w": "\x1b[97m", "x": "\x1b[0m", "bold": "\x1b[1m"}


def color(s, c): return f"{C[c]}{s}{C['x']}"
def bar(p, n=20):
    f = int(round(p * n)); c = "r" if p >= THRESH else "y" if p >= 0.35 else "g"
    return color("█" * f, c) + color("░" * (n - f), "d")
def checksum(rec): return "sha256:" + hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
def line(s): sys.stdout.write(s + "\n"); sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="transaction_api")
    ap.add_argument("--data_path", default="data/banking_api_features_v7.csv")
    ap.add_argument("--nrows", type=int, default=500_000)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--speed", type=float, default=0.7, help="seconds between ticks")
    a = ap.parse_args()

    line(color("\n  LEO · LIVE FAILOVER  ", "bold"))
    line(color("  loading real model + scaler …", "d"))
    model, scaler, feat, _ = A.load_model(MODEL_PATH, SCALER_PATH)

    df = pd.read_csv(a.data_path, usecols=lambda c: c in set(feat + ["api_name", "success"]), nrows=a.nrows)
    sub = df[df["api_name"] == a.api].copy().reset_index(drop=True)
    need = A.SEQ_LEN + a.steps
    if len(sub) < need:
        line(color(f"  not enough rows for {a.api}; raise --nrows", "r")); return
    raw = sub[feat].fillna(0).astype(np.float32).to_numpy()[-need:]
    stress_idx = [feat.index(c) for c in STRESS if c in feat]

    # source of truth — one record, two read paths (primary / region-b standby)
    idem = f"req-{a.api}-{int(sub.iloc[-1].name)}"
    record = {"idempotency_key": idem, "api": a.api, "source": "core-ledger",
              "account": "acct_8841", "balance": "12480.55", "ccy": "USD"}
    SOURCE = {idem: record}
    incident_start = 4
    line(f"  primary = {color(a.api + ' · region-a', 'w')}   standby = {color(a.api + ' · region-b', 'w')}")
    line(color(f"  streaming live — incident injected at t+{incident_start}\n", "d"))
    line(f"  {'tick':<6}{'risk':<26}{'p(h+1)':<9}{'route':<26}status")
    line(color("  " + "-" * 74, "d"))

    rerouted = False; trip_tick = None
    for t in range(a.steps):
        win = raw[t:t + A.SEQ_LEN].copy()
        if not rerouted and t >= incident_start:           # ramp the degrading endpoint
            g = 1.0 + 0.9 * (t - incident_start + 1)
            for j in stress_idx:
                win[-min(6, A.SEQ_LEN):, j] *= g
        p = float(A.predict_batch(model, scaler.transform(win)[None, ...].astype(np.float32))[0][0])

        if not rerouted and p >= THRESH:                   # ── the reroute event ──
            trip_tick = t
            line(f"  t+{t:<4}{bar(p)}  {p:5.3f}   {color(a.api+' · region-a','r'):<35}{color('⚠ HIGH — primary failing','r')}")
            time.sleep(a.speed)
            line("")
            line(color("  ┌──────────────────────────────────────────────────────────┐", "b"))
            line(color("  │  LEO DECISION: risk ≥ 0.55  →  REROUTE to warm standby     │", "b"))
            line(color("  └──────────────────────────────────────────────────────────┘", "b"))
            time.sleep(a.speed)
            rec_p = SOURCE[idem]; rec_s = dict(SOURCE[idem])    # both read same source
            cp, cs = checksum(rec_p), checksum(rec_s)
            line(f"    primary  {a.api}·region-a   {color('DOWN','r')}")
            line(f"    standby  {a.api}·region-b   {color('ACTIVE · serving','g')}")
            line(f"    record   {json.dumps(rec_s, separators=(',',':'))}")
            line(f"    checksum primary {color(cp,'b')}")
            line(f"    checksum standby {color(cs,'b')}")
            ok = cp == cs
            line("    " + (color("✓ IDENTICAL — customer data intact, zero divergence", "g") if ok
                           else color("✗ DIVERGED", "r")))
            line("")
            rerouted = True
            continue

        route = f"{a.api}·region-b" if rerouted else f"{a.api}·region-a"
        node = "standby" if rerouted else "primary"
        status = color("● healthy", "g") if p < 0.35 else color("◐ degrading", "y") if p < THRESH else color("⚠ high", "r")
        if rerouted: status = color("● healthy (rerouted)", "g")
        line(f"  t+{t:<4}{bar(p)}  {p:5.3f}   {color(route,'w'):<35}{status}")
        time.sleep(a.speed)

    line(color("\n  " + "-" * 74, "d"))
    if trip_tick is not None:
        lead = (a.steps - trip_tick)
        line(color(f"  ✓ DETECTED at t+{trip_tick} · rerouted before failure · "
                   f"~{lead} ticks of runway · data parity held", "g"))
    else:
        line(color("  primary stayed healthy this run — raise --steps or try --api crypto_api", "y"))
    line("")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
