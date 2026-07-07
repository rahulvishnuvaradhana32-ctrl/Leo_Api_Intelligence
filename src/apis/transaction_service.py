"""
transaction_service.py — Real banking transaction API.

What makes this REAL (not simulated):
  - SQLite stores actual account balances and transaction ledger
  - Idempotency keys: if the proxy reroutes the same request to backup,
    the backup detects the duplicate key and returns the cached result —
    no double debit, no double credit, guaranteed exactly-once semantics
  - Chaos endpoints let LEO Proxy inject real failures
  - /health returns real computed metrics (error_rate, avg_response_time)

Run primary:  uvicorn src.apis.transaction_service:app --port 8001
Run backup:   REGION=region-b uvicorn src.apis.transaction_service:app --port 8002
"""

import os
import time
import uuid
import sqlite3
import asyncio
import random
from contextlib import contextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.apis.base_service import build_base_app

# ── Config ────────────────────────────────────────────────────────────────────

REGION      = os.getenv("REGION", "region-a")
PORT        = int(os.getenv("PORT", 8001))

# Absolute paths so both region processes always resolve to the same files
# regardless of the working directory uvicorn was launched from.
_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHARED_DB   = os.path.join(_ROOT, "data", "shared_ledger.db")
AUDIT_DB    = os.path.join(_ROOT, "data", f"audit_{REGION}.db")

# ── SQLite helpers ────────────────────────────────────────────────────────────

def _conn(path: str) -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    c = sqlite3.connect(path, check_same_thread=False, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")   # allows concurrent readers
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("PRAGMA busy_timeout=5000")  # wait up to 5 s if locked
    return c


def _init_db():
    # ── Shared ledger (both regions read/write this) ───────────────────────
    s = _conn(SHARED_DB)
    s.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id  TEXT PRIMARY KEY,
            balance     REAL NOT NULL DEFAULT 0,
            currency    TEXT NOT NULL DEFAULT 'USD',
            created_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS transactions (
            txn_id          TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE,   -- UNIQUE enforces exactly-once
            account_from    TEXT NOT NULL,
            account_to      TEXT NOT NULL,
            amount          REAL NOT NULL,
            currency        TEXT NOT NULL,
            status          TEXT NOT NULL,
            failure_reason  TEXT,
            fee             REAL,
            region          TEXT NOT NULL, -- which region committed this
            created_at      TEXT NOT NULL
        );
        INSERT OR IGNORE INTO accounts VALUES
            ('acct_001', 50000.00, 'USD', datetime('now')),
            ('acct_002', 25000.00, 'USD', datetime('now')),
            ('acct_003', 75000.00, 'USD', datetime('now')),
            ('acct_004', 10000.00, 'USD', datetime('now')),
            ('acct_005', 100000.00,'USD', datetime('now'));
    """)
    s.commit()
    s.close()

    # ── Per-region audit log (lightweight, region-specific) ───────────────
    a = _conn(AUDIT_DB)
    a.executescript("""
        CREATE TABLE IF NOT EXISTS request_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            txn_id       TEXT,
            idempotency_key TEXT,
            action       TEXT,  -- 'processed' | 'idempotent_hit' | 'error'
            response_ms  REAL,
            created_at   TEXT NOT NULL
        );
    """)
    a.commit()
    a.close()


_init_db()

# ── App setup ─────────────────────────────────────────────────────────────────

app, telemetry, chaos = build_base_app(
    title=f"Transaction API ({REGION})",
    region=REGION,
)


# ── Request / Response models ─────────────────────────────────────────────────

class TransferRequest(BaseModel):
    account_from:    str
    account_to:      str
    amount:          float
    currency:        str = "USD"
    idempotency_key: str = ""      # caller must supply; proxy auto-injects if missing


class TransactionResponse(BaseModel):
    txn_id:          str
    idempotency_key: str
    status:          str
    account_from:    str
    account_to:      str
    amount:          float
    fee:             float
    net_amount:      float
    currency:        str
    region:          str           # which region served this request
    processing_ms:   float
    created_at:      str


# ── Core business logic ───────────────────────────────────────────────────────

def _process_transfer(req: TransferRequest, idem_key: str) -> dict:
    """
    Atomic transfer against the SHARED ledger with cross-region idempotency.

    Flow:
      1. Check shared_ledger.transactions for idempotency_key
         → if found, return cached result immediately (no double debit)
      2. Validate accounts + balance in shared_ledger.accounts
      3. BEGIN IMMEDIATE on shared ledger (serialised write across regions)
      4. Debit sender, credit receiver, insert transaction row
      5. Log to per-region audit DB
    """
    t0   = time.perf_counter()
    conn = _conn(SHARED_DB)
    try:
        # ── 1. Cross-region idempotency check ─────────────────────────────
        existing = conn.execute(
            "SELECT * FROM transactions WHERE idempotency_key = ?", (idem_key,)
        ).fetchone()
        if existing:
            row = dict(existing)
            _audit(row["txn_id"], idem_key, "idempotent_hit",
                   (time.perf_counter() - t0) * 1000)
            return row | {"cached": True, "served_by": REGION}

        # ── 2. Validate accounts ───────────────────────────────────────────
        from_acc = conn.execute(
            "SELECT * FROM accounts WHERE account_id = ?", (req.account_from,)
        ).fetchone()
        to_acc = conn.execute(
            "SELECT * FROM accounts WHERE account_id = ?", (req.account_to,)
        ).fetchone()

        if not from_acc:
            raise ValueError(f"Account not found: {req.account_from}")
        if not to_acc:
            raise ValueError(f"Account not found: {req.account_to}")
        if req.amount <= 0:
            raise ValueError("Amount must be positive")
        if from_acc["balance"] < req.amount:
            raise ValueError(
                f"Insufficient funds — balance=${from_acc['balance']:,.2f}, "
                f"requested=${req.amount:,.2f}"
            )

        # ── 3. Atomic debit / credit on shared ledger ─────────────────────
        fee    = max(0.25, round(req.amount * 0.001, 2))
        txn_id = f"TXN-{uuid.uuid4().hex[:12].upper()}"
        now    = datetime.utcnow().isoformat()

        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE accounts SET balance = balance - ? WHERE account_id = ?",
            (req.amount + fee, req.account_from),
        )
        conn.execute(
            "UPDATE accounts SET balance = balance + ? WHERE account_id = ?",
            (req.amount, req.account_to),
        )
        conn.execute(
            """INSERT INTO transactions
               (txn_id, idempotency_key, account_from, account_to,
                amount, currency, status, fee, region, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (txn_id, idem_key, req.account_from, req.account_to,
             req.amount, req.currency, "completed", fee, REGION, now),
        )
        conn.commit()

        result = {
            "txn_id": txn_id, "idempotency_key": idem_key,
            "account_from": req.account_from, "account_to": req.account_to,
            "amount": req.amount, "fee": fee, "currency": req.currency,
            "status": "completed", "region": REGION,
            "created_at": now, "cached": False, "served_by": REGION,
        }
        _audit(txn_id, idem_key, "processed", (time.perf_counter() - t0) * 1000)
        return result

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _audit(txn_id: str, idem_key: str, action: str, ms: float):
    try:
        a = _conn(AUDIT_DB)
        a.execute(
            "INSERT INTO request_log (txn_id, idempotency_key, action, response_ms, created_at) "
            "VALUES (?,?,?,?,datetime('now'))",
            (txn_id, idem_key, action, round(ms, 2)),
        )
        a.commit()
        a.close()
    except Exception:
        pass


# ── POST /transfer ────────────────────────────────────────────────────────────

@app.post("/transfer", response_model=TransactionResponse)
async def transfer(req: TransferRequest):
    t0 = time.perf_counter()

    # Auto-generate idempotency key if caller didn't supply one
    idem_key = req.idempotency_key or str(uuid.uuid4())

    # ── Apply chaos ───────────────────────────────────────────────────────────
    if chaos.mode == "down":
        telemetry.record(time.perf_counter() - t0, False)
        raise HTTPException(503, "Service unavailable — planned outage")

    if chaos.mode == "timeout":
        await asyncio.sleep(chaos.latency_multiplier * 2.0)
        telemetry.record(time.perf_counter() - t0, False)
        raise HTTPException(504, "Gateway timeout — upstream overloaded")

    if chaos.mode in ("error_surge", "overload") and random.random() < chaos.error_rate:
        telemetry.record(time.perf_counter() - t0, False)
        code = 503 if chaos.mode == "overload" else 500
        raise HTTPException(code, f"Chaos mode active: {chaos.mode}")

    # ── Real processing delay (realistic for banking: 200–600 ms) ─────────────
    base_delay = 0.25
    if chaos.mode == "overload":
        base_delay *= chaos.latency_multiplier
    await asyncio.sleep(random.uniform(base_delay * 0.8, base_delay * 1.3))

    # ── Business logic ────────────────────────────────────────────────────────
    try:
        result = _process_transfer(req, idem_key)
    except ValueError as e:
        elapsed = time.perf_counter() - t0
        telemetry.record(elapsed, False)
        raise HTTPException(422, str(e))

    elapsed = time.perf_counter() - t0
    telemetry.record(elapsed, True)

    return {
        **result,
        "net_amount":    result["amount"] - result["fee"],
        "processing_ms": round(elapsed * 1000, 2),
    }


# ── GET /transfer/{txn_id} ────────────────────────────────────────────────────

@app.get("/transfer/{txn_id}")
async def get_transfer(txn_id: str):
    conn = _conn(SHARED_DB)
    row  = conn.execute(
        "SELECT * FROM transactions WHERE txn_id = ?", (txn_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, f"Transaction not found: {txn_id}")
    return dict(row)


# ── GET /balance/{account_id} ─────────────────────────────────────────────────

@app.get("/balance/{account_id}")
async def get_balance(account_id: str):
    conn = _conn(SHARED_DB)
    row  = conn.execute(
        "SELECT account_id, balance, currency FROM accounts WHERE account_id = ?",
        (account_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, f"Account not found: {account_id}")
    return dict(row)


# ── Startup banner ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print(f"\n  Transaction API [{REGION}] ready")
    print(f"  Shared DB: {SHARED_DB}")
    print(f"  Idempotency: enabled (safe rerouting guaranteed)\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.apis.transaction_service:app",
                host="0.0.0.0", port=PORT, reload=False)
