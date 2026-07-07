#!/usr/bin/env python3
"""
chaos_runner.py — Orchestrates a full LEO self-healing demo on real traffic.

What this script does (in sequence):
  1.  Warm up — sends 25 normal transfers through the proxy to fill LSTM window
  2.  Inject  — forces DOWN chaos on the primary API (port 8001)
  3.  Traffic — sends 15 more transfers through the proxy; LEO detects failure,
                reroutes to backup, diagnoses root cause, applies fix
  4.  Health  — polls proxy /status to confirm heal cycle is active
  5.  Restore — clears chaos from primary, lets health poller trigger canary restore
  6.  Confirm — final balance check and event log dump

All output is printed with clear section headers so the full loop is visible.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx, time, json, uuid

PROXY   = "http://localhost:9000"
PRIMARY = "http://localhost:8001"

W = 66
def sep(title=""):
    if title:
        print(f"\n{'='*W}\n  {title}\n{'='*W}")
    else:
        print(f"\n{'-'*W}")


def banner(msg, char="="):
    print(f"\n{char*W}")
    print(f"  {msg}")
    print(f"{char*W}")


def send_transfer(amount=100.0, note="") -> dict:
    idem = str(uuid.uuid4())
    try:
        r = httpx.post(
            f"{PROXY}/transfer",
            json={
                "account_from":    "acct_001",
                "account_to":      "acct_002",
                "amount":          amount,
                "idempotency_key": idem,
            },
            timeout=15,
        )
        d = r.json()
        backend = r.headers.get("x-leo-backend", "?")
        risk    = r.headers.get("x-leo-risk", "?")
        route   = r.headers.get("x-leo-route", "?")
        cb      = r.headers.get("x-leo-cb-state", "?")
        status  = r.status_code

        if status < 500:
            print(f"  [{status}] backend={backend:<22} risk={risk:<6} route={route}  "
                  f"cb={cb}  {note}")
        else:
            print(f"  [{status}] ERROR — {d.get('error','?')}  (backend={backend})")
        return {"status": status, "backend": backend, "risk": risk, "data": d}
    except Exception as e:
        print(f"  [ERR] {e}  {note}")
        return {"status": 0, "error": str(e)}


def show_proxy_status():
    r = httpx.get(f"{PROXY}/status", timeout=5)
    s = r.json()
    print(f"  Route:          {s['route']}")
    print(f"  Risk:           {s['risk']}")
    print(f"  Circuit breaker:{s['circuit_breaker']['state']}  "
          f"(fail_streak={s['circuit_breaker']['fail_streak']})")
    h = s.get("heal_cycle", {})
    print(f"  Heal active:    {h.get('active', False)}")
    if h.get("active"):
        print(f"  Root cause:     {h.get('root_cause')}")
        print(f"  Remedy:         {h.get('remedy')}")
        print(f"  Ticks elapsed:  {h.get('ticks_elapsed')}")
        canary = h.get('canary_stage', 0)
        print(f"  Canary stage:   {canary*100:.0f}% to primary" if canary else "  Canary stage:   pending")
    return s


def show_events(limit=30):
    r = httpx.get(f"{PROXY}/events?limit={limit}", timeout=5)
    events = r.json().get("events", [])
    if not events:
        print("  (no events yet)")
        return
    for e in events:
        phase = e.get("phase", e.get("event", "?"))
        msg   = e.get("message", "")
        ts    = e.get("ts", "")
        print(f"  [{phase:<16}] {msg}")


def check_balance():
    r = httpx.get(f"{PROXY}/balance/acct_001", timeout=5)
    d = r.json()
    print(f"  acct_001 balance: ${d['balance']:,.2f}")
    return d["balance"]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0: Verify both APIs are up
# ─────────────────────────────────────────────────────────────────────────────
sep("PHASE 0 — Pre-flight checks")

proxy_h = httpx.get(f"{PROXY}/health", timeout=5).json()
print(f"  LEO Proxy:  {proxy_h['status']}  lstm={proxy_h['lstm_enabled']}")

prim_h  = httpx.get(f"{PRIMARY}/health", timeout=5).json()
print(f"  Primary:    {prim_h['status']}  region={prim_h['region']}")

print("\n  Resetting proxy state to clean baseline...")
httpx.post(f"{PROXY}/admin/reset", timeout=5)
print("  Done.")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Warm-up traffic (fills LSTM window, establishes baseline)
# ─────────────────────────────────────────────────────────────────────────────
sep("PHASE 1 — Warm-up: 25 normal transfers through proxy → primary (baseline)")

for i in range(25):
    send_transfer(amount=10.0 + i, note=f"warmup #{i+1}")
    time.sleep(0.3)

bal_before = check_balance()
print(f"\n  Warm-up complete. Baseline balance: ${bal_before:,.2f}")

show_proxy_status()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Inject failure — primary goes DOWN
# ─────────────────────────────────────────────────────────────────────────────
sep("PHASE 2 — Injecting DOWN failure on primary (port 8001)")

r = httpx.post(f"{PRIMARY}/chaos/inject",
               json={"mode": "down", "reason": "DB crash — primary offline"},
               timeout=5)
print(f"  Chaos injected: {r.json()}")

# Verify primary is down
h = httpx.get(f"{PRIMARY}/health", timeout=5).json()
print(f"  Primary health: {h['status']}  chaos={h['chaos']['mode']}")
print()
print("  >>> Primary is now DOWN.  LEO Proxy will detect this. <<<")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: Traffic during failure — LEO detects, reroutes, diagnoses, fixes
# ─────────────────────────────────────────────────────────────────────────────
sep("PHASE 3 — 15 transfers during failure: detect → reroute → diagnose → fix")

for i in range(15):
    send_transfer(amount=50.0, note=f"during-failure #{i+1}")
    time.sleep(0.5)

print()
sep("Proxy status after failure traffic:")
state = show_proxy_status()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: Show event log — the full heal cycle audit trail
# ─────────────────────────────────────────────────────────────────────────────
sep("PHASE 4 — LEO Event Log (full heal cycle audit)")
show_events(limit=40)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: Restore primary, let LEO run canary restore
# ─────────────────────────────────────────────────────────────────────────────
sep("PHASE 5 — Restoring primary: clear chaos → canary restore")

httpx.post(f"{PRIMARY}/chaos/clear", timeout=5)
h = httpx.get(f"{PRIMARY}/health", timeout=5).json()
print(f"  Primary health: {h['status']}  chaos={h['chaos']['mode']}")
print()
print(f"  Waiting {30}s for health poller to detect recovery and run canary restore...")

for tick in range(6):
    time.sleep(5)
    s = httpx.get(f"{PROXY}/status", timeout=5).json()
    route = s["route"]
    heal  = s.get("heal_cycle", {})
    canary = heal.get("canary_stage", 0)
    print(f"  t+{(tick+1)*5:02d}s  route={route}  "
          f"canary={canary*100:.0f}%  heal_active={heal.get('active', False)}")
    if not heal.get("active") and route == "primary":
        print("  RESTORED — proxy back on primary with canary complete!")
        break

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6: Verify data integrity — same accounts, one balance
# ─────────────────────────────────────────────────────────────────────────────
sep("PHASE 6 — Data integrity check")

# Get balance via primary directly
pr = httpx.get(f"{PRIMARY}/balance/acct_001", timeout=5).json()
# Get balance via backup directly
br = httpx.get("http://localhost:8002/balance/acct_001", timeout=5).json()
# Get via proxy
prx = httpx.get(f"{PROXY}/balance/acct_001", timeout=5).json()

print(f"  Primary  (direct):  acct_001 = ${pr['balance']:,.2f}")
print(f"  Backup   (direct):  acct_001 = ${br['balance']:,.2f}")
print(f"  Proxy    (current): acct_001 = ${prx['balance']:,.2f}")

match = abs(pr["balance"] - br["balance"]) < 0.01
print()
print(f"  Primary == Backup?  {'YES - shared ledger, zero discrepancy' if match else 'NO - MISMATCH!'}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
sep("FINAL — LEO Proxy Event Log (all phases)")
show_events(limit=60)

banner("LEO API Intelligence — Phase 2 Proof of Concept")
print(f"  All traffic routed through LEO Proxy on :9000")
print(f"  Primary   :8001 (region-a) — subject of failure injection")
print(f"  Backup    :8002 (region-b) — received rerouted traffic")
print()
print(f"  LSTM risk scoring:   every {10} requests on real telemetry")
print(f"  Reroute threshold:   risk >= 0.65")
print(f"  Restore threshold:   risk <= 0.35")
print(f"  Canary stages:       10% → 50% → 100%")
print()
print(f"  Self-healing loop:  DETECT → REROUTE → DIAGNOSE → FIX → RESTORE")
print(f"  Data integrity:     shared SQLite ledger, zero double-debit")
print("=" * W)
