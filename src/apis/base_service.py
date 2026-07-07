"""
base_service.py — Shared foundation for every real LEO banking API service.

Provides:
  - In-memory telemetry ring buffer (response_time, error_rate, request_count)
  - Chaos state manager (inject timeout / error_surge / overload / service_down)
  - /health endpoint with real computed metrics
  - /chaos/inject and /chaos/clear endpoints
  - ServiceDown exception raised when chaos mode is active

Every real API service inherits ChaosMiddleware and mounts these routes.
"""

import time
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# ── Telemetry ring buffer ─────────────────────────────────────────────────────

class TelemetryBuffer:
    """Tracks last N requests: response_time, success/failure."""

    def __init__(self, window: int = 100):
        self._rt    = deque(maxlen=window)
        self._ok    = deque(maxlen=window)
        self._start = time.time()
        self._total = 0

    def record(self, response_time: float, success: bool):
        self._rt.append(response_time)
        self._ok.append(1 if success else 0)
        self._total += 1

    def metrics(self) -> dict:
        if not self._rt:
            return {"avg_response_time_ms": 0, "error_rate": 0,
                    "request_count": 0, "uptime_sec": round(time.time() - self._start, 1)}
        rt_list = list(self._rt)
        ok_list = list(self._ok)
        return {
            "avg_response_time_ms": round(sum(rt_list) / len(rt_list) * 1000, 2),
            "p95_response_time_ms": round(sorted(rt_list)[int(len(rt_list) * 0.95)] * 1000, 2),
            "error_rate":           round(1 - sum(ok_list) / len(ok_list), 4),
            "request_count":        self._total,
            "window_size":          len(rt_list),
            "uptime_sec":           round(time.time() - self._start, 1),
        }


# ── Chaos state ───────────────────────────────────────────────────────────────

@dataclass
class ChaosState:
    mode: str = "healthy"          # healthy | timeout | error_surge | overload | down
    error_rate: float = 0.0        # injected error probability (0–1)
    latency_multiplier: float = 1.0
    injected_at: Optional[float] = None
    reason: str = ""

    def is_active(self) -> bool:
        return self.mode != "healthy"

    def to_dict(self) -> dict:
        return {
            "mode":               self.mode,
            "error_rate":         self.error_rate,
            "latency_multiplier": self.latency_multiplier,
            "injected_at":        self.injected_at,
            "reason":             self.reason,
        }


# ── Middleware — records every request into telemetry ─────────────────────────

class TelemetryMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, telemetry: TelemetryBuffer, chaos: ChaosState):
        super().__init__(app)
        self.telemetry = telemetry
        self.chaos     = chaos

    async def dispatch(self, request: Request, call_next):
        # Skip telemetry for internal routes
        if request.url.path in ("/health", "/chaos/inject", "/chaos/clear", "/metrics"):
            return await call_next(request)

        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0

        success = response.status_code < 500
        self.telemetry.record(elapsed, success)
        return response


# ── Factory — builds a base FastAPI app with shared routes ───────────────────

def build_base_app(title: str, region: str) -> tuple:
    """
    Returns (app, telemetry, chaos) — mount your API routes onto app.
    """
    telemetry = TelemetryBuffer(window=200)
    chaos     = ChaosState()
    app       = FastAPI(title=title, description=f"LEO Real Banking Service — {region}")

    app.add_middleware(TelemetryMiddleware, telemetry=telemetry, chaos=chaos)

    # ── /health ───────────────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        m = telemetry.metrics()
        status = "degraded" if chaos.is_active() else (
            "unhealthy" if m["error_rate"] > 0.3 else "healthy"
        )
        return {
            "status":  status,
            "service": title,
            "region":  region,
            "chaos":   chaos.to_dict(),
            **m,
        }

    # ── /metrics (Prometheus-style text, LEO Proxy reads this) ───────────────
    @app.get("/metrics")
    async def metrics():
        m = telemetry.metrics()
        return {**m, "chaos": chaos.to_dict()}

    # ── /chaos/inject ─────────────────────────────────────────────────────────
    @app.post("/chaos/inject")
    async def inject_chaos(payload: dict):
        """
        Inject a failure mode. Payload:
          { "mode": "timeout|error_surge|overload|down", "reason": "..." }
        """
        mode = payload.get("mode", "error_surge")
        if mode not in ("timeout", "error_surge", "overload", "down"):
            return JSONResponse({"error": "unknown mode"}, status_code=400)

        chaos.mode               = mode
        chaos.injected_at        = time.time()
        chaos.reason             = payload.get("reason", "manual injection")
        chaos.error_rate         = float(payload.get("error_rate", 0.8))
        chaos.latency_multiplier = float(payload.get("latency_multiplier", 1.0))

        if mode == "timeout":
            chaos.latency_multiplier = float(payload.get("latency_multiplier", 8.0))
            chaos.error_rate         = 0.0
        elif mode == "overload":
            chaos.latency_multiplier = float(payload.get("latency_multiplier", 3.0))
            chaos.error_rate         = float(payload.get("error_rate", 0.4))
        elif mode == "down":
            chaos.error_rate         = 1.0
            chaos.latency_multiplier = 1.0

        return {"injected": chaos.to_dict()}

    # ── /chaos/clear ──────────────────────────────────────────────────────────
    @app.post("/chaos/clear")
    async def clear_chaos():
        chaos.mode               = "healthy"
        chaos.error_rate         = 0.0
        chaos.latency_multiplier = 1.0
        chaos.injected_at        = None
        chaos.reason             = ""
        return {"status": "healthy", "chaos": chaos.to_dict()}

    return app, telemetry, chaos
