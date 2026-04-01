#!/usr/bin/env python3
"""
dashboard_server.py — LEO API Predictive Reliability Dashboard (Auto-Refresh)

Wraps production_dashboard.py and adds:
  - /health              — keep-alive for Render / UptimeRobot
  - /api/last_modified   — returns max mtime of all models/*.json files
  - AutoRefreshMiddleware — injects JS into "/" that polls every 20s and
                            reloads the page the moment any result file changes

Usage (local):
    python scripts/dashboard_server.py
    Open http://localhost:8000

Render (production):
    startCommand: python scripts/dashboard_server.py
    healthCheckPath: /health

UptimeRobot (keep Render free tier alive):
    Monitor URL: https://<your-render-url>/health
    Interval: every 5 minutes
"""
import sys, os, time
from pathlib import Path

# ── Make sure production_dashboard is importable ──────────────────────────────
_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))

# Import the FastAPI app (with all existing routes) and needed helpers
from production_dashboard import app, MODELS       # noqa: E402

from fastapi import Request                         # noqa: E402
from fastapi.responses import JSONResponse, Response # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
import uvicorn                                      # noqa: E402


# ── Auto-refresh JS — injected into every "/" HTML response ───────────────────
_REFRESH_JS = r"""
<script>
/* LEO API Dashboard — auto-refresh (dashboard_server.py) */
(function () {
    'use strict';

    /* ── floating badge ─────────────────────────────────────────────────── */
    var badge = document.createElement('div');
    badge.style.cssText = [
        'position:fixed', 'bottom:18px', 'right:18px',
        'background:#0f1623', 'border:1px solid #1e293b',
        'border-radius:8px', 'padding:7px 14px',
        'font-size:11px', 'color:#4b5563',
        'z-index:9999', 'font-family:monospace',
        'letter-spacing:0.3px', 'transition:color 0.4s',
        'cursor:default', 'user-select:none'
    ].join(';');
    badge.title = 'Reloads automatically when model results change';
    document.body.appendChild(badge);

    var lastMod  = null;
    var INTERVAL = 20000;   /* ms — poll every 20 seconds */

    function setBadge(msg, color) {
        badge.style.color = color || '#4b5563';
        badge.innerHTML   = msg;
    }

    function checkUpdates() {
        fetch('/api/last_modified', { cache: 'no-store' })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var ts = new Date().toLocaleTimeString();
                if (lastMod === null) {
                    lastMod = data.last_modified;
                    setBadge('&#9711; Live &nbsp;&middot;&nbsp; ' + ts);
                } else if (data.last_modified > lastMod) {
                    setBadge('&#8635; Results updated &mdash; reloading&hellip;', '#22c55e');
                    lastMod = data.last_modified;
                    setTimeout(function () { window.location.reload(); }, 900);
                } else {
                    setBadge('&#9711; Live &nbsp;&middot;&nbsp; ' + ts);
                }
            })
            .catch(function () { setBadge('&#9711; Watching&hellip;'); });
    }

    checkUpdates();                         /* run once immediately     */
    setInterval(checkUpdates, INTERVAL);    /* then every 20 seconds    */
})();
</script>
"""


# ── Middleware — intercept "/" responses and inject the refresh script ─────────
class AutoRefreshMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        ct = response.headers.get("content-type", "")

        # Only touch the main dashboard page HTML — leave API/stream routes alone
        if request.url.path == "/" and "text/html" in ct:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            html = body.decode("utf-8", errors="replace")

            # Inject JS right before </body> so it runs after the page is loaded
            html = html.replace("</body>", _REFRESH_JS + "\n</body>", 1)

            return Response(
                content=html,
                status_code=response.status_code,
                media_type="text/html",
                headers={k: v for k, v in response.headers.items()
                         if k.lower() not in ("content-length",)},
            )
        return response


app.add_middleware(AutoRefreshMiddleware)


# ── /health — keep-alive for Render free tier + UptimeRobot ───────────────────
@app.get("/health")
async def health():
    """
    Returns {"status": "ok"}.
    Point UptimeRobot at this URL with a 5-minute interval to prevent
    Render free-tier cold starts (service spins down after 15 min idle).
    """
    return JSONResponse({"status": "ok", "ts": time.time()})


# ── /api/last_modified — file-change detection for the JS poller ──────────────
@app.get("/api/last_modified")
async def last_modified():
    """
    Returns the latest modification timestamp across all models/*.json files.
    The frontend polls this; when the value increases it triggers a page reload.
    This means the dashboard updates automatically after any pipeline script
    (evaluate_lstm, ablation_study, conformal_prediction, agent_simulation,
    self_improving_pipeline) finishes writing its results JSON.
    """
    try:
        mtimes = [f.stat().st_mtime for f in MODELS.glob("*.json") if f.is_file()]
        latest = max(mtimes) if mtimes else 0.0
    except Exception:
        latest = 0.0
    return JSONResponse({"last_modified": latest})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading, webbrowser

    def _open_browser():
        time.sleep(1.3)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=_open_browser, daemon=True).start()

    print("=" * 60)
    print("  LEO API Dashboard  (Auto-Refresh Edition)")
    print("  http://localhost:8000")
    print("-" * 60)
    print("  /health           — Render keep-alive endpoint")
    print("  /api/last_modified — polled every 20s by the browser")
    print("  Page auto-reloads when any models/*.json file changes")
    print("=" * 60)

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
