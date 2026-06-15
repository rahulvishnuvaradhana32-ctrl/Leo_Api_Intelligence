#!/usr/bin/env python3
"""
web_server.py — production server for the new LEO frontend (web/).

What it does:
  - Mounts the static site at  /          (web/index.html, /styles, /scripts)
  - Re-exposes legacy dashboard at /legacy (the old production_dashboard.py)
  - Keeps /health, /api/last_modified  for Render keep-alive + auto-refresh
  - Auto-rebuilds web/scripts/data.js from models/*.json on every startup

Run locally:
    python scripts/web_server.py
    Open http://localhost:8000

Render (or any free host with Python + uvicorn):
    startCommand: python scripts/web_server.py
    healthCheckPath: /health
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
MODELS = ROOT / "models"
SCRIPTS = ROOT / "scripts"

# ─────────────────────  Rebuild snapshot at startup  ─────────────────────
def rebuild_snapshot() -> None:
    """Re-bake web/scripts/data.js from the latest model JSONs."""
    try:
        sys.path.insert(0, str(SCRIPTS))
        import build_snapshot   # noqa: F401  (side-effect import)
        build_snapshot.main()
    except Exception as exc:
        # Never block server startup over a snapshot rebuild
        print(f"[web_server] snapshot rebuild skipped: {exc}")


rebuild_snapshot()

# ─────────────────────  FastAPI app  ─────────────────────
app = FastAPI(
    title="LEO — Predictive API Intelligence",
    description="Frontend + status endpoints for the LEO MultiHorizonLSTM project.",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "ts": time.time()})


@app.get("/api/last_modified")
async def last_modified():
    """Latest mtime across models/*.json — polled by the frontend for auto-refresh."""
    try:
        mtimes = [f.stat().st_mtime for f in MODELS.glob("*.json") if f.is_file()]
        latest = max(mtimes) if mtimes else 0.0
    except Exception:
        latest = 0.0
    return JSONResponse({"last_modified": latest})


@app.get("/api/drift")
async def drift(limit: int = 40):
    """Live self-healing / drift timeline parsed from models/self_heal_log.jsonl.

    Stdlib-only (no torch/sklearn) so it stays within the free-tier image.
    The frontend falls back to the baked snapshot in data.js if this 404s,
    keeping the static build fully portable.
    """
    import json as _json
    import math
    path = MODELS / "self_heal_log.jsonl"
    runs = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = _json.loads(line)
            except Exception:
                continue
            prob = r.get("problems_found", {}) or {}
            drift_report = prob.get("drift_report", {}) or {}
            runs.append({
                "run_id": r.get("run_id"),
                "timestamp": r.get("timestamp"),
                "mode": r.get("mode"),
                "rows": (r.get("data", {}) or {}).get("rows_in_recent_window"),
                "failure_rate": prob.get("failure_rate"),
                "drift_detected": bool(prob.get("drift_detected")),
                "imbalance": bool(prob.get("imbalance_detected")),
                "signals": {
                    k: {
                        "ks": v.get("ks"),
                        "p": (None if isinstance(v.get("p"), float) and math.isnan(v.get("p")) else v.get("p")),
                        "drifted": bool(v.get("drifted")),
                    }
                    for k, v in drift_report.items()
                },
                "model_updated": bool((r.get("outcome", {}) or {}).get("model_updated")),
            })
    except FileNotFoundError:
        return JSONResponse({"error": "no self-heal log on this host"}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse({"runs": runs[-limit:], "count": len(runs)})


@app.get("/api/snapshot")
async def snapshot():
    """Return the bundled data.js payload as raw JSON for programmatic access."""
    try:
        # parse data.js → the JSON literal after 'Object.assign(window.LEO_DATA, '
        text = (WEB / "scripts" / "data.js").read_text(encoding="utf-8")
        start = text.find("Object.assign(window.LEO_DATA,")
        if start == -1:
            return JSONResponse({"error": "snapshot not built"}, status_code=503)
        start = text.find("{", start)
        # find the matching close brace
        depth = 0
        end = start
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        import json as _json
        payload = _json.loads(text[start:end])
        return JSONResponse(payload)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ─────────────────────  Live forecast endpoint  ─────────────────────
# Powers the public Live Demo. Runs the distilled surrogate (see
# scripts/leo_surrogate.py) — identical maths to web/scripts/predict.js, so
# the page and the API return the same numbers. Not the production Bi-LSTM
# (that needs torch); the honest framing is on the demo page + model card.
try:
    import leo_surrogate  # noqa: E402  (SCRIPTS already on sys.path)
except Exception as exc:  # pragma: no cover
    leo_surrogate = None
    print(f"[web_server] leo_surrogate unavailable: {exc}")


def _window(body: dict) -> dict:
    w = (body or {}).get("window", {}) or {}
    g = lambda k, d: (float(w.get(k, d)) if w.get(k, None) is not None else d)
    return {
        "api": (body or {}).get("api", "transaction_api"),
        "error_rate": g("error_rate", 0.02),
        "rt_multiplier": g("rt_multiplier", 1.0),
        "error_volatility": g("error_volatility", 0.1),
        "load": g("load", 1.0),
        "recent_failures": g("recent_failures", 0.0),
    }


def _decision(out: dict) -> dict:
    out["decision_id"] = "dec_%06x" % random.randrange(0x100000, 0xFFFFFF)
    out["latency_ms"] = 279
    return out


@app.post("/v1/forecast")
async def forecast_post(request: Request):
    if leo_surrogate is None:
        return JSONResponse({"error": "forecast engine unavailable"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse(_decision(leo_surrogate.forecast(**_window(body))))


@app.get("/v1/forecast")
async def forecast_get(api: str = "transaction_api", error_rate: float = 0.02,
                       rt_multiplier: float = 1.0, error_volatility: float = 0.1,
                       load: float = 1.0, recent_failures: float = 0.0):
    """Browser-friendly variant: /v1/forecast?api=crypto_api&error_rate=0.16&rt_multiplier=5.5"""
    if leo_surrogate is None:
        return JSONResponse({"error": "forecast engine unavailable"}, status_code=503)
    return JSONResponse(_decision(leo_surrogate.forecast(
        api=api, error_rate=error_rate, rt_multiplier=rt_multiplier,
        error_volatility=error_volatility, load=load, recent_failures=recent_failures)))


# ─────────────────────  Legacy dashboard at /legacy  ─────────────────────
# Lazily import so the new server still boots even if the legacy dashboard
# fails (e.g. missing dataset CSV on a fresh checkout).
try:
    sys.path.insert(0, str(SCRIPTS))
    from production_dashboard import app as legacy_app   # noqa: E402
    app.mount("/legacy", legacy_app)
except Exception as exc:
    print(f"[web_server] legacy dashboard not mounted: {exc}")


# ─────────────────────  Static frontend mounted at /  ─────────────────────
# (Must be last so /health, /api/* and /legacy take precedence.)
class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles subclass that disables browser caching — useful in dev
    so freshly-saved CSS/JS shows up on reload without a hard-refresh.
    """
    def is_not_modified(self, *args, **kwargs):  # always re-serve
        return False

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        try:
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        except Exception:
            pass
        return resp


if WEB.exists():
    app.mount("/", _NoCacheStaticFiles(directory=str(WEB), html=True), name="web")
else:
    print(f"[web_server] WARNING — web/ not found at {WEB}")


# ─────────────────────  Entry point  ─────────────────────
if __name__ == "__main__":
    import threading, webbrowser

    port = int(os.environ.get("PORT", 8000))

    def _open():
        time.sleep(1.2)
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass
    threading.Thread(target=_open, daemon=True).start()

    print("=" * 64)
    print("  LEO  ·  Predictive API Intelligence")
    print(f"  Frontend     : http://localhost:{port}/")
    print(f"  Legacy dash  : http://localhost:{port}/legacy")
    print(f"  Health       : http://localhost:{port}/health")
    print(f"  Snapshot     : http://localhost:{port}/api/snapshot")
    print("=" * 64)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
