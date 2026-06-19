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
import secrets
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders

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


# ─────────────────────  Security hardening  ─────────────────────
# Cloudflare / Render protect the network layer (L3/L4 DDoS, TLS); these
# guard the L7 application: security headers + a per-request-nonce CSP on the
# static frontend, plus IP-based rate limiting on the dynamic endpoints.
#
# Frontend currently lives on the same origin as the API. When it moves to
# Cloudflare Pages, set ALLOWED_ORIGINS (comma-separated) to the Pages origin —
# that value extends both CORS and the CSP connect-src directive automatically.
_EXTRA_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
# HSTS only matters over HTTPS (Cloudflare/Render terminate TLS). On by default;
# set ENABLE_HSTS=0 to disable (e.g. plain-HTTP local testing). No includeSubDomains
# / preload here — onrender.com is a shared parent domain; add those on a custom domain.
_ENABLE_HSTS = os.environ.get("ENABLE_HSTS", "1") != "0"

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-XSS-Protection": "0",  # modern guidance: disable the legacy, buggy XSS auditor
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": (
        "accelerometer=(), autoplay=(), camera=(), display-capture=(), "
        "geolocation=(), gyroscope=(), magnetometer=(), microphone=(), "
        "payment=(), usb=()"
    ),
}


def _csp(nonce: str, relaxed: bool = False) -> str:
    # relaxed = the legacy dashboard at /legacy, which relies on inline event
    # handlers (onclick=...). Those can't carry a nonce, so it gets unsafe-inline.
    # The main static site is nonce-locked.
    script_src = "'self' 'unsafe-inline'" if relaxed else f"'self' 'nonce-{nonce}'"
    connect = " ".join(["'self'", *_EXTRA_ORIGINS])
    return "; ".join([
        "default-src 'self'",
        f"script-src {script_src}",
        # style-src needs unsafe-inline: a handful of inline style="" attributes +
        # the Google Fonts stylesheet. Style injection is low-risk vs script.
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "img-src 'self' data:",
        f"connect-src {connect}",
        "object-src 'none'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
        "form-action 'self'",
        "upgrade-insecure-requests",
    ])


def _apply_headers(headers: MutableHeaders, nonce: str, relaxed: bool) -> None:
    headers["Content-Security-Policy"] = _csp(nonce, relaxed)
    for k, v in _SECURITY_HEADERS.items():
        headers[k] = v
    if _ENABLE_HSTS:
        headers["Strict-Transport-Security"] = "max-age=31536000"


class SecurityHeadersMiddleware:
    """Pure-ASGI middleware (not BaseHTTPMiddleware, which buffers and would
    break the legacy dashboard's SSE streams). Sets security headers on every
    response and rewrites static HTML to inject the per-request CSP nonce into
    <script> tags — StaticFiles serves files verbatim, so the body is patched here.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        relaxed = path.startswith("/legacy")
        nonce = secrets.token_urlsafe(16)
        scope.setdefault("state", {})["csp_nonce"] = nonce

        deferred = {}      # holds the start message while we buffer HTML
        chunks: list[bytes] = []

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                _apply_headers(headers, nonce, relaxed)
                # Only buffer + rewrite real HTML on the nonce-locked site.
                if headers.get("content-type", "").startswith("text/html") and not relaxed:
                    deferred["start"] = message
                    deferred["headers"] = headers
                    return
                await send(message)

            elif message["type"] == "http.response.body" and "start" in deferred:
                chunks.append(message.get("body", b""))
                if message.get("more_body"):
                    return
                body = b"".join(chunks).replace(
                    b"<script", b'<script nonce="' + nonce.encode() + b'"'
                )
                deferred["headers"]["content-length"] = str(len(body))
                await send(deferred["start"])
                await send({"type": "http.response.body", "body": body, "more_body": False})

            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(SecurityHeadersMiddleware)

# CORS only when the frontend is served from a different origin (Cloudflare Pages).
# Same-origin today needs no CORS, so this stays inert unless ALLOWED_ORIGINS is set.
if _EXTRA_ORIGINS:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_EXTRA_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,  # no cookies/sessions — token-free public API
        max_age=600,
    )


# ─────────────────────  Rate limiting (in-memory; Render free has no Redis)  ──
# slowapi is optional: if the dependency is missing in the deploy env, fall
# back to a no-op limiter so the app still boots (rate limiting simply off).
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    _SLOWAPI = True
except Exception as exc:  # pragma: no cover
    print(f"[web_server] slowapi unavailable ({exc}); rate limiting disabled")
    _SLOWAPI = False

    def get_remote_address(request):
        return request.client.host if request.client else "0.0.0.0"

    class Limiter:                       # no-op shim with the same surface
        def __init__(self, *a, **k): pass
        def limit(self, *a, **k):
            def deco(fn): return fn
            return deco

    RateLimitExceeded = None


def _client_ip(request: Request) -> str:
    # Behind Cloudflare → Render the socket peer is a proxy. Trust the real
    # client IP from the proxy headers (Cloudflare sets CF-Connecting-IP).
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip, headers_enabled=True)
app.state.limiter = limiter
if _SLOWAPI:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
async def health():
    # No limit: Render keep-alive pings this and it must always answer.
    return JSONResponse({"status": "ok", "ts": time.time()})


@app.get("/api/last_modified")
@limiter.limit("60/minute")  # frontend polls every 20s; allow several open tabs
async def last_modified(request: Request):
    """Latest mtime across models/*.json — polled by the frontend for auto-refresh."""
    try:
        mtimes = [f.stat().st_mtime for f in MODELS.glob("*.json") if f.is_file()]
        latest = max(mtimes) if mtimes else 0.0
    except Exception:
        latest = 0.0
    return JSONResponse({"last_modified": latest})


@app.get("/api/drift")
@limiter.limit("30/minute")
async def drift(request: Request, limit: int = 40):
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
@limiter.limit("30/minute")
async def snapshot(request: Request):
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
@limiter.limit("30/minute")  # the compute endpoint — the most abuse-prone
async def forecast_post(request: Request):
    if leo_surrogate is None:
        return JSONResponse({"error": "forecast engine unavailable"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse(_decision(leo_surrogate.forecast(**_window(body))))


@app.get("/v1/forecast")
@limiter.limit("30/minute")
async def forecast_get(request: Request,
                       api: str = "transaction_api", error_rate: float = 0.02,
                       rt_multiplier: float = 1.0, error_volatility: float = 0.1,
                       load: float = 1.0, recent_failures: float = 0.0):
    """Browser-friendly variant: /v1/forecast?api=crypto_api&error_rate=0.16&rt_multiplier=5.5"""
    if leo_surrogate is None:
        return JSONResponse({"error": "forecast engine unavailable"}, status_code=503)
    return JSONResponse(_decision(leo_surrogate.forecast(
        api=api, error_rate=error_rate, rt_multiplier=rt_multiplier,
        error_volatility=error_volatility, load=load, recent_failures=recent_failures)))


# ─────────────────────  Routing endpoint (real failover state machine)  ─────
# Powers the live failover widget. The engine (scripts/route_engine.py) is
# stateful — hysteresis, cooldown, fail-back — so we keep one instance per
# session_id and advance exactly one tick per call.
try:
    import route_engine  # noqa: E402
except Exception as exc:  # pragma: no cover
    route_engine = None
    print(f"[web_server] route_engine unavailable: {exc}")

_ROUTERS: dict = {}          # session_id -> RouteEngine
_ROUTERS_MAX = 500


@app.post("/v1/route")
@limiter.limit("120/minute")  # one call per UI tick — looser than the compute endpoint
async def route_post(request: Request):
    if route_engine is None:
        return JSONResponse({"error": "route engine unavailable"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    routes = (body or {}).get("routes") or {}
    if not isinstance(routes, dict) or not routes:
        return JSONResponse({"error": "body.routes must be {route: risk, ...}"}, status_code=400)
    try:
        risks = {str(k): float(v) for k, v in routes.items()}
    except Exception:
        return JSONResponse({"error": "route risks must be numbers"}, status_code=400)

    names = list(risks.keys())
    primary = (body or {}).get("primary") or names[0]
    sid = str((body or {}).get("session_id", "default"))[:64]

    eng = _ROUTERS.get(sid)
    if eng is None or eng.routes != names or eng.primary != primary:
        if len(_ROUTERS) >= _ROUTERS_MAX:
            _ROUTERS.clear()                       # crude bound; sessions are cheap
        eng = route_engine.RouteEngine(names, primary)
        _ROUTERS[sid] = eng

    return JSONResponse(eng.step(risks))


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
