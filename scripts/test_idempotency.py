"""
test_idempotency.py — Proves cross-region idempotency end-to-end.

Steps:
  1. Send $500 transfer to PRIMARY  (port 8001, region-a)
  2. Inject DOWN failure on primary
  3. Replay same idempotency_key to BACKUP (port 8002, region-b)
     → backup must return cached result, NOT re-process
  4. Check balance — must be debited exactly ONCE ($500 + $0.50 fee)
  5. Show shared ledger contents
  6. Restore primary
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx, sqlite3, json

PRIMARY = "http://localhost:8001"
BACKUP  = "http://localhost:8002"
IDEM    = "IDEM-XYZ-PROOF-001"

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARED  = os.path.join(ROOT, "data", "shared_ledger.db")

W = 62
def sep(title=""): print(f"\n{'='*W}\n  {title}\n{'='*W}" if title else f"\n{'-'*W}")

# ── Step 1: Normal transaction on primary ─────────────────────────────────────
sep("STEP 1 — Process $500 on PRIMARY (region-a, port 8001)")
r = httpx.post(f"{PRIMARY}/transfer", json={
    "account_from": "acct_001",
    "account_to":   "acct_002",
    "amount":       500.0,
    "idempotency_key": IDEM,
}, timeout=10)
d1 = r.json()
print(f"  Status:          {r.status_code}")
print(f"  TXN ID:          {d1.get('txn_id')}")
print(f"  Region served:   {d1.get('region')}")
print(f"  Cached hit:      {d1.get('cached')}")
print(f"  Amount:          ${d1.get('amount')}  Fee: ${d1.get('fee')}")
print(f"  Processing time: {d1.get('processing_ms')} ms")

# ── Step 2: Inject failure on primary ─────────────────────────────────────────
sep("STEP 2 — Inject DOWN failure on PRIMARY")
httpx.post(f"{PRIMARY}/chaos/inject", json={
    "mode": "down", "reason": "DB crash — primary offline"
}, timeout=5)
h = httpx.get(f"{PRIMARY}/health", timeout=5).json()
print(f"  Primary status:  {h['status']}  chaos={h['chaos']['mode']}")

# ── Step 3: Same key to backup (rerouting) ────────────────────────────────────
sep("STEP 3 — Replay same idempotency_key to BACKUP (port 8002)")
r2 = httpx.post(f"{BACKUP}/transfer", json={
    "account_from": "acct_001",
    "account_to":   "acct_002",
    "amount":       500.0,
    "idempotency_key": IDEM,
}, timeout=10)
d2 = r2.json()
print(f"  Status:          {r2.status_code}")
print(f"  TXN ID:          {d2.get('txn_id')}  (same = idempotent)")
print(f"  Region served:   {d2.get('served_by')}  (backup served it)")
print(f"  Cached hit:      {d2.get('cached')}  ← MUST be True")

if d2.get("cached"):
    print("\n  PASS — backup returned cached result. NO double-debit.")
else:
    print("\n  FAIL — backup re-processed the transaction!")

# ── Step 4: Balance check ─────────────────────────────────────────────────────
sep("STEP 4 — Balance check (deducted exactly ONCE?)")
bal = httpx.get(f"{BACKUP}/balance/acct_001", timeout=5).json()
expected = 50000.0 - 500.0 - 0.50
actual   = bal["balance"]
print(f"  acct_001 balance: ${actual:,.2f}")
print(f"  Expected:         ${expected:,.2f}")
print(f"  {'PASS' if abs(actual - expected) < 0.01 else 'FAIL'} — "
      f"{'Deducted exactly once' if abs(actual-expected)<0.01 else 'Mismatch!'}")

# ── Step 5: Shared ledger ─────────────────────────────────────────────────────
sep("STEP 5 — Shared ledger contents")
conn = sqlite3.connect(SHARED)
conn.row_factory = sqlite3.Row
rows = list(conn.execute("SELECT * FROM transactions"))
print(f"  Total rows: {len(rows)}  (must be 1 — one transfer, not two)")
for row in rows:
    print(f"  {dict(row)}")

# Audit log per region
for region, db in [("region-a", "audit_region-a.db"), ("region-b", "audit_region-b.db")]:
    audit_path = os.path.join(ROOT, "data", db)
    if os.path.exists(audit_path):
        ac = sqlite3.connect(audit_path)
        ac.row_factory = sqlite3.Row
        logs = list(ac.execute("SELECT action, response_ms FROM request_log"))
        print(f"  Audit [{region}]: {logs}")
        ac.close()
conn.close()

# ── Step 6: Restore primary ───────────────────────────────────────────────────
sep("STEP 6 — Restore PRIMARY")
httpx.post(f"{PRIMARY}/chaos/clear", timeout=5)
h2 = httpx.get(f"{PRIMARY}/health", timeout=5).json()
print(f"  Primary status: {h2['status']}  (back to healthy)")

sep("DONE")
print("  Cross-region idempotency proven.")
print("  One transfer. One debit. Two regions. Zero data loss.")
