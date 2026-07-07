#!/usr/bin/env python3
"""
leo_proxy.py — LEO API Intelligence Proxy  (Phase 2)

Real HTTP reverse proxy that implements the full self-healing loop on live traffic:

    DETECT → REROUTE → DIAGNOSE → FIX → RESTORE

Traffic flow:
  Client → :9000 (this proxy) → :8001 (primary) or :8002 (backup)

Key capabilities:
  - Real async HTTP forwarding via httpx.AsyncClient
  - LSTM risk scoring every SCORE_EVERY requests (real feature computation)
  - RouteEngine state machine decides primary vs backup per API
  - DiagnosticEngine classifies root cause from real telemetry
  - RemediationEngine applies fix (circuit_break / backoff / throttle / etc.)
  - In-proxy circuit breaker prevents hammering a failed primary
  - Health polling every HEALTH_POLL_SEC confirms primary recovery
  - Canary restore: 10% → 50% → 100% traffic before full switchback
  - WebSocket /ws/events streams every heal-cycle event in real time

Usage:
    cd FCE_project
    python scripts/leo_proxy.py

    # In another terminal — all API calls go through :9000
    curl -s http://localhost:9000/health
    curl -s -X POST http://localhost:9000/transfer -H 'Content-Type: application/json' \
         -d '{"account_from":"acct_001","account_to":"acct_002","amount":100}'
    # Watch events live:
    wscat -c ws://localhost:9000/ws/events
"""

import asyncio
import json
import os
import sys
import time
import random
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── sys.path so we can import from scripts/ ──────────────────────────────────
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_SCRIPTS, ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from telemetry_collector import ApiTelemetry, FEATURE_COLS
from diagnostic_engine   import DiagnosticEngine
from remediation_engine  import RemediationEngine

# ── Config ────────────────────────────────────────────────────────────────────
PRIMARY_URL     = os.getenv("PRIMARY_URL",  "http://localhost:8001")
BACKUP_URL      = os.getenv("BACKUP_URL",   "http://localhost:8002")
PROXY_PORT      = int(os.getenv("PROXY_PORT", "9000"))
MODEL_PATH      = os.path.join(_ROOT, "models", "stress_test_best_model.pth")
SCALER_PATH     = os.path.join(_ROOT, "models", "scaler.pkl")

RISK_HI         = 0.65   # reroute threshold
RISK_LO         = 0.35   # restore threshold
SCORE_EVERY     = 10     # run LSTM every N requests (lower = more responsive)
HEALTH_POLL_SEC = 5.0    # health-check interval
CANARY_STEPS    = [0.10, 0.50, 1.00]   # fraction of traffic to primary during restore
CANARY_HOLD_SEC = 10.0   # seconds at each canary stage before advancing
CB_FAIL_THRESH  = 3      # consecutive failures to open circuit breaker
CB_PROBE_SEC    = 15.0   # seconds before circuit breaker allows a probe


# ── Circuit breaker ───────────────────────────────────────────────────────────
class CBState(Enum):
    CLOSED    = "CLOSED"      # normal, requests flow through
    OPEN      = "OPEN"        # primary is down, block all primary requests
    HALF_OPEN = "HALF_OPEN"   # probing: 1 request allowed to test recovery


@dataclass
class CircuitBreaker:
    fail_thresh:  int   = CB_FAIL_THRESH
    probe_sec:    float = CB_PROBE_SEC

    state:         CBState = CBState.CLOSED
    fail_streak:   int     = 0
    opened_at:     float   = 0.0
    probe_sent:    bool    = False

    def record_success(self):
        self.fail_streak = 0
        if self.state == CBState.HALF_OPEN:
            self.state = CBState.CLOSED
            return "closed"
        return None

    def record_failure(self) -> Optional[str]:
        self.fail_streak += 1
        if self.state == CBState.HALF_OPEN:
            self.state    = CBState.OPEN
            self.opened_at = time.time()
            return "reopened"
        if self.state == CBState.CLOSED and self.fail_streak >= self.fail_thresh:
            self.state    = CBState.OPEN
            self.opened_at = time.time()
            return "opened"
        return None

    def allow_request(self) -> bool:
        if self.state == CBState.CLOSED:
            return True
        if self.state == CBState.OPEN:
            if time.time() - self.opened_at >= self.probe_sec:
                self.state      = CBState.HALF_OPEN
                self.probe_sent = False
                return True  # let the first probe through
            return False
        if self.state == CBState.HALF_OPEN:
            if not self.probe_sent:
                self.probe_sent = True
                return True
            return False
        return True


# ── Proxy state ───────────────────────────────────────────────────────────────
class RouteTarget(Enum):
    PRIMARY = "primary"
    BACKUP  = "backup"


@dataclass
class HealState:
    """Tracks one full detect→reroute→diagnose→fix→restore cycle."""
    triggered_at:   float = 0.0
    root_cause:     str   = ""
    remedy_action:  str   = ""
    recovery_ticks: int   = 0
    ticks_elapsed:  int   = 0
    canary_stage:   int   = 0         # index into CANARY_STEPS
    canary_since:   float = 0.0
    restored:       bool  = False


@dataclass
class ProxyState:
    route:          RouteTarget   = RouteTarget.PRIMARY
    risk:           float         = 0.0
    primary_health: bool          = True
    backup_health:  bool          = True
    request_count:  int           = 0
    heal:           Optional[HealState] = None
    lstm_loaded:    bool          = False


# ── LSTM loader ───────────────────────────────────────────────────────────────
def _try_load_model():
    try:
        import torch
        import sys, os
        if _SCRIPTS not in sys.path:
            sys.path.insert(0, _SCRIPTS)
        import agent_simulation as A

        if not os.path.exists(MODEL_PATH):
            print(f"  [LSTM] model not found at {MODEL_PATH} — risk scoring disabled")
            return None, None, None

        model, scaler, feat_cols, _ = A.load_model(MODEL_PATH, SCALER_PATH)
        print(f"  [LSTM] loaded — {len(feat_cols)} features")
        return model, scaler, feat_cols
    except Exception as e:
        print(f"  [LSTM] load failed ({e}) — risk scoring disabled")
        return None, None, None


def _lstm_infer(model, scaler, feat_cols, seq: np.ndarray) -> float:
    """Run LSTM inference on a (seq_len, n_features) array, return h=1 risk."""
    try:
        import torch
        # Trim features to what model expects
        n_in = len(feat_cols)
        x    = seq[:, :n_in]
        x    = scaler.transform(x.reshape(-1, n_in)).reshape(1, -1, n_in)
        tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = model(tensor)          # (1, n_horizons)
            prob   = torch.sigmoid(logits[0, 0]).item()
        return float(prob)
    except Exception as e:
        print(f"  [LSTM] infer error: {e}")
        return 0.0


# ── Global objects (initialised on startup) ───────────────────────────────────
state      = ProxyState()
telemetry  = ApiTelemetry("transaction_api")
diagnostic = DiagnosticEngine()
remediation = RemediationEngine()
circuit    = CircuitBreaker()

model_obj  = None
scaler_obj = None
feat_cols  = None

ws_clients: list[WebSocket] = []
event_log:  list[dict]      = []   # in-memory audit (last 500 events)
MAX_LOG     = 500


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="LEO API Intelligence Proxy",
    description="Real self-healing reverse proxy — detect → reroute → diagnose → fix → restore",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

http_client: httpx.AsyncClient = None   # created in startup


# ── Event broadcast ───────────────────────────────────────────────────────────
async def _broadcast(event: dict):
    event["ts"] = round(time.time(), 3)
    event_log.append(event)
    if len(event_log) > MAX_LOG:
        del event_log[:MAX_LOG // 2]

    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)

    # Also print to terminal for visibility
    phase = event.get("phase", event.get("event", "event"))
    api   = event.get("api", "")
    msg   = event.get("message", "")
    print(f"  [LEO:{phase}] {api}  {msg}")


# ── Core: pick target URL ──────────────────────────────────────────────────────
def _pick_target() -> tuple[str, str]:
    """Returns (url, label) based on current route and canary stage."""
    if state.route == RouteTarget.PRIMARY:
        return PRIMARY_URL, "primary"

    # During canary restore, route some traffic back to primary
    if state.heal and state.heal.canary_stage > 0:
        fraction = CANARY_STEPS[state.heal.canary_stage - 1]
        if random.random() < fraction and circuit.allow_request():
            return PRIMARY_URL, "primary(canary)"

    return BACKUP_URL, "backup"


# ── Core: health check polling ─────────────────────────────────────────────────
async def _health_poll_loop():
    # Debounce: only act on a health change after 2 consecutive consistent readings
    _consecutive: dict = {"state": None, "count": 0}
    while True:
        await asyncio.sleep(HEALTH_POLL_SEC)
        try:
            r = await http_client.get(f"{PRIMARY_URL}/health", timeout=3.0)
            healthy = r.status_code == 200 and r.json().get("status") == "healthy"
        except Exception:
            healthy = False

        # Debounce: require 2 consistent readings before treating as real state change
        if healthy == _consecutive["state"]:
            _consecutive["count"] += 1
        else:
            _consecutive["state"] = healthy
            _consecutive["count"] = 1

        if _consecutive["count"] != 2:
            continue   # not yet confirmed

        prev = state.primary_health
        state.primary_health = healthy

        if prev != healthy:
            await _broadcast({
                "phase":   "health_check",
                "api":     "transaction_api",
                "target":  "primary",
                "healthy": healthy,
                "message": f"primary health confirmed: {'UP' if healthy else 'DOWN'}",
            })

        # If primary recovered and we're in a heal cycle, advance canary
        if healthy and state.heal and not state.heal.restored:
            await _try_advance_canary()


# ── Core: canary restore ───────────────────────────────────────────────────────
async def _try_advance_canary():
    h = state.heal
    if h is None or h.restored:
        return

    now = time.time()

    if h.canary_stage == 0:
        # Start canary at 10%
        h.canary_stage = 1
        h.canary_since = now
        await _broadcast({
            "phase":   "restore",
            "step":    "canary_10pct",
            "api":     "transaction_api",
            "message": "primary recovered — canary restore started at 10%",
        })
        return

    elapsed = now - h.canary_since
    if elapsed < CANARY_HOLD_SEC:
        return  # not yet time to advance

    if h.canary_stage < len(CANARY_STEPS):
        h.canary_stage += 1
        h.canary_since  = now
        pct = int(CANARY_STEPS[h.canary_stage - 1] * 100)
        await _broadcast({
            "phase":   "restore",
            "step":    f"canary_{pct}pct",
            "api":     "transaction_api",
            "message": f"canary advanced to {pct}% — primary holding steady",
        })

    if h.canary_stage >= len(CANARY_STEPS):
        # Full restore
        state.route     = RouteTarget.PRIMARY
        state.risk      = 0.0
        h.restored      = True
        circuit.state   = CBState.CLOSED
        circuit.fail_streak = 0
        await _broadcast({
            "phase":   "restore",
            "step":    "full_restore",
            "api":     "transaction_api",
            "message": "RESTORED — primary fully restored, all traffic back to primary",
        })
        state.heal = None


# ── Core: LSTM risk scoring ────────────────────────────────────────────────────
async def _run_lstm_check():
    if not state.lstm_loaded or model_obj is None:
        # Fallback: derive risk from real telemetry (no model)
        m = telemetry.current_metrics()
        risk = min(1.0, m["error_rate"] * 3.0 + max(0.0, m["avg_rt_s"] - 0.5) * 2.0)
        state.risk = risk
    else:
        seq = telemetry.get_feature_sequence(seq_len=30)
        if seq is None:
            return
        risk = _lstm_infer(model_obj, scaler_obj, feat_cols, seq)
        state.risk = risk

    await _broadcast({
        "phase":   "detect",
        "api":     "transaction_api",
        "risk":    round(state.risk, 4),
        "message": f"LSTM risk score: {state.risk:.4f}",
    })

    # Trigger reroute if risk above threshold and we're still on primary
    if state.risk >= RISK_HI and state.route == RouteTarget.PRIMARY and state.heal is None:
        await _trigger_reroute()


# ── Core: reroute → diagnose → fix ────────────────────────────────────────────
async def _trigger_reroute():
    state.route = RouteTarget.BACKUP
    heal = HealState(triggered_at=time.time())

    await _broadcast({
        "phase":   "reroute",
        "api":     "transaction_api",
        "risk":    round(state.risk, 4),
        "target":  "backup",
        "message": f"risk={state.risk:.3f} >= {RISK_HI} — rerouting to backup:8002",
    })

    # Diagnose root cause from real telemetry features
    m      = telemetry.current_metrics()
    feat_d = _telemetry_to_diag_features(m, state.risk)
    rc     = diagnostic.diagnose(feat_d)

    heal.root_cause = rc.category

    await _broadcast({
        "phase":        "diagnose",
        "api":          "transaction_api",
        "root_cause":   rc.category,
        "confidence":   round(rc.confidence, 3),
        "evidence":     rc.evidence,
        "explain":      diagnostic.explain(rc),
        "message":      diagnostic.explain(rc),
    })

    # Apply fix
    remedy = remediation.apply_fix("transaction_api", rc.category)
    heal.remedy_action  = remedy.action
    heal.recovery_ticks = remedy.recovery_ticks

    await _broadcast({
        "phase":          "fix",
        "api":            "transaction_api",
        "action":         remedy.action,
        "recovery_ticks": remedy.recovery_ticks,
        "params":         remedy.params,
        "message": (
            f"fix applied: {remedy.action} — "
            f"risk will decay {remedy.risk_reduction:.2f}/tick over "
            f"{remedy.recovery_ticks} ticks"
        ),
    })

    state.heal = heal


# ── Core: translate live metrics → DiagnosticEngine input ─────────────────────
def _telemetry_to_diag_features(m: dict, lstm_risk: float) -> dict:
    """Build the feature dict DiagnosticEngine.diagnose() expects."""
    rt  = m.get("avg_rt_s", 0.3)
    err = m.get("error_rate", 0.0)
    return {
        "systemic_stress_index": lstm_risk * 0.7,
        "n_apis_elevated":       1 if lstm_risk > 0.6 else 0,
        "burst_ratio":           max(1.0, lstm_risk * 3.0),
        "rt_multiplier":         rt / 0.3 if rt > 0.3 else 1.0,
        "error_burst":           1.0 if err > 0.5 else 0.0,
        # pass raw error_rate as well so cascade heuristic can use it
        "error_rate_rolling":    err,
    }


# ── Proxy handler (all routes) ────────────────────────────────────────────────
async def _proxy(request: Request, path: str) -> Response:
    state.request_count += 1
    body = await request.body()

    target_url, label = _pick_target()

    # Respect circuit breaker for primary requests
    if "primary" in label and not circuit.allow_request():
        # CB is OPEN — route to backup without even trying primary
        target_url = BACKUP_URL
        label      = "backup(cb-open)"

    # Forward request
    url     = f"{target_url}/{path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}

    t0 = time.perf_counter()
    try:
        resp = await http_client.request(
            method  = request.method,
            url     = url,
            headers = headers,
            content = body,
            params  = dict(request.query_params),
            timeout = 10.0,
        )
        elapsed = time.perf_counter() - t0
        success = resp.status_code < 500

        # Update circuit breaker
        if "primary" in label:
            if success:
                cb_ev = circuit.record_success()
                if cb_ev == "closed":
                    asyncio.create_task(_broadcast({
                        "phase":   "circuit_breaker",
                        "state":   "CLOSED",
                        "message": "circuit breaker closed — primary stable",
                    }))
            else:
                cb_ev = circuit.record_failure()
                if cb_ev in ("opened", "reopened"):
                    asyncio.create_task(_broadcast({
                        "phase":   "circuit_breaker",
                        "state":   "OPEN",
                        "message": f"circuit breaker {cb_ev} after {CB_FAIL_THRESH} failures",
                    }))
                    # CB opening is a real detection signal — trigger full heal cycle
                    if state.route == RouteTarget.PRIMARY and state.heal is None:
                        state.risk = max(state.risk, RISK_HI + 0.05)
                        asyncio.create_task(_trigger_reroute())

        # Record real telemetry
        telemetry.record(elapsed, success, resp.status_code)

        # Tick remediation fix (decays risk each request)
        if remediation.is_fixing("transaction_api"):
            updated, done = remediation.tick("transaction_api", state.risk)
            state.risk = updated
            state.heal and setattr(state.heal, "ticks_elapsed",
                                   getattr(state.heal, "ticks_elapsed", 0) + 1)
            if done:
                asyncio.create_task(_broadcast({
                    "phase":   "fix_complete",
                    "api":     "transaction_api",
                    "message": "fix timer complete — monitoring primary for restore",
                }))

        # Periodic LSTM risk check
        if state.request_count % SCORE_EVERY == 0:
            asyncio.create_task(_run_lstm_check())

        # Add LEO header so caller can see which backend served the request
        resp_headers = dict(resp.headers)
        resp_headers["X-LEO-Backend"]   = label
        resp_headers["X-LEO-Risk"]      = str(round(state.risk, 4))
        resp_headers["X-LEO-Route"]     = state.route.value
        resp_headers["X-LEO-CB-State"]  = circuit.state.value
        # Remove headers that conflict with FastAPI response handling
        for h in ("content-encoding", "transfer-encoding", "content-length"):
            resp_headers.pop(h, None)

        return Response(
            content    = resp.content,
            status_code= resp.status_code,
            headers    = resp_headers,
            media_type = resp.headers.get("content-type", "application/json"),
        )

    except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
        elapsed = time.perf_counter() - t0
        telemetry.record(elapsed, False, 503)

        if "primary" in label:
            cb_ev = circuit.record_failure()
            if cb_ev in ("opened", "reopened"):
                asyncio.create_task(_broadcast({
                    "phase":   "circuit_breaker",
                    "state":   "OPEN",
                    "message": f"circuit breaker opened — {exc}",
                }))

        # If we were trying primary and it failed hard, force reroute now
        if "primary" in label and state.route == RouteTarget.PRIMARY and state.heal is None:
            state.risk = 1.0   # treat connection error as maximum risk
            asyncio.create_task(_trigger_reroute())
            # Retry immediately on backup
            try:
                r2 = await http_client.request(
                    method=request.method, url=f"{BACKUP_URL}/{path}",
                    headers=headers, content=body,
                    params=dict(request.query_params), timeout=10.0,
                )
                return Response(
                    content=r2.content, status_code=r2.status_code,
                    headers={"X-LEO-Backend": "backup(failover)",
                             "X-LEO-Risk": "1.0",
                             "X-LEO-Route": "backup",
                             "X-LEO-CB-State": circuit.state.value},
                    media_type=r2.headers.get("content-type", "application/json"),
                )
            except Exception as e2:
                return JSONResponse({"error": f"all backends failed: {e2}"}, status_code=503)

        return JSONResponse(
            {"error": f"upstream error: {exc}", "backend": label},
            status_code=503,
        )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def proxy_health():
    m = telemetry.current_metrics()
    return {
        "status":         "healthy",
        "proxy":          "LEO API Intelligence Proxy",
        "version":        "2.0.0",
        "route":          state.route.value,
        "risk":           round(state.risk, 4),
        "circuit_breaker":circuit.state.value,
        "primary_health": state.primary_health,
        "backup_health":  state.backup_health,
        "lstm_enabled":   state.lstm_loaded,
        "requests_served":state.request_count,
        "heal_active":    state.heal is not None,
        "telemetry":      m,
    }


@app.get("/status")
async def proxy_status():
    """Detailed proxy status including active heal cycle."""
    fix_status = remediation.fix_status("transaction_api")
    return {
        "route":           state.route.value,
        "risk":            round(state.risk, 4),
        "circuit_breaker": {
            "state":       circuit.state.value,
            "fail_streak": circuit.fail_streak,
        },
        "heal_cycle": {
            "active":          state.heal is not None,
            "root_cause":      state.heal.root_cause if state.heal else None,
            "remedy":          state.heal.remedy_action if state.heal else None,
            "ticks_elapsed":   state.heal.ticks_elapsed if state.heal else 0,
            "canary_stage":    (CANARY_STEPS[state.heal.canary_stage - 1]
                                if state.heal and state.heal.canary_stage > 0 else 0),
            "restored":        state.heal.restored if state.heal else False,
        } if state.heal else {"active": False},
        "fix_status":      fix_status,
        "primary_url":     PRIMARY_URL,
        "backup_url":      BACKUP_URL,
        "requests_served": state.request_count,
        "lstm_enabled":    state.lstm_loaded,
    }


@app.get("/events")
async def get_events(limit: int = 50):
    """Return last N events from the in-memory audit log."""
    return {"events": event_log[-limit:], "total": len(event_log)}


@app.post("/admin/inject-risk")
async def inject_risk(payload: dict):
    """Manually set risk to test the heal loop (e.g. risk=0.9 triggers reroute)."""
    risk = float(payload.get("risk", 0.5))
    state.risk = min(1.0, max(0.0, risk))
    if state.risk >= RISK_HI and state.route == RouteTarget.PRIMARY and state.heal is None:
        await _trigger_reroute()
    return {"risk": state.risk, "route": state.route.value}


@app.post("/admin/reset")
async def reset():
    """Reset proxy to clean state (for testing)."""
    state.route = RouteTarget.PRIMARY
    state.risk  = 0.0
    state.heal  = None
    circuit.state       = CBState.CLOSED
    circuit.fail_streak = 0
    remediation._active.clear()
    event_log.clear()
    return {"status": "reset", "route": state.route.value}


# ── WebSocket event feed ───────────────────────────────────────────────────────
@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        # Send last 20 events on connect so client has context
        for ev in event_log[-20:]:
            await ws.send_json(ev)
        await ws.send_json({"phase": "connected", "message": "LEO event feed live"})
        while True:
            await ws.receive_text()   # keep-alive
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ── Catch-all proxy routes ────────────────────────────────────────────────────
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    # Don't proxy our own admin routes
    if path in ("health", "status", "events", "ws/events") or path.startswith("admin/"):
        raise Exception("Internal route — should not be proxied")
    return await _proxy(request, path)


# ── Startup / shutdown ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global http_client, model_obj, scaler_obj, feat_cols

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=3.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )

    print("\n" + "=" * 62)
    print("  LEO API Intelligence Proxy v2.0  —  starting up")
    print("=" * 62)
    print(f"  Primary:  {PRIMARY_URL}")
    print(f"  Backup:   {BACKUP_URL}")
    print(f"  Proxy:    http://localhost:{PROXY_PORT}")
    print(f"  Events:   ws://localhost:{PROXY_PORT}/ws/events")
    print(f"  Reroute threshold:  risk >= {RISK_HI}")
    print(f"  Restore threshold:  risk <= {RISK_LO}")
    print(f"  LSTM scoring every: {SCORE_EVERY} requests")

    model_obj, scaler_obj, feat_cols = _try_load_model()
    state.lstm_loaded = model_obj is not None

    # Start background tasks
    asyncio.create_task(_health_poll_loop())

    await _broadcast({
        "phase":   "startup",
        "message": f"LEO Proxy ready — LSTM {'enabled' if state.lstm_loaded else 'disabled (telemetry fallback)'}",
    })
    print("=" * 62 + "\n")


@app.on_event("shutdown")
async def shutdown():
    if http_client:
        await http_client.aclose()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "leo_proxy:app",
        host="0.0.0.0",
        port=PROXY_PORT,
        reload=False,
        log_level="warning",   # suppress uvicorn access log; LEO prints its own
    )
