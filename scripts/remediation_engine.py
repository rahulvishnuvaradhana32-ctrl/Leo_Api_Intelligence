#!/usr/bin/env python3
"""
remediation_engine.py — Maps a DiagnosticEngine RootCause to a concrete fix
action and simulates recovery by decaying the primary API's risk score.

Each tick after apply_fix() is called, the primary API's risk drops by
`risk_reduction` per tick. This feeds directly into RouteEngine's risk map,
enabling its existing FAIL-BACK logic to trigger restore once the primary
risk falls back below LO for COOLDOWN consecutive ticks.

Fix actions:
  backoff             — exponential retry interval, halved timeout
  circuit_break       — pause requests for N ticks, then probe-and-resume
  throttle            — shed X% of load, queue the rest
  isolate_dependency  — decouple from failing upstream, serve from cache
  retry_jitter        — retry with random jitter to break correlated spikes
"""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Remedy:
    action: str           # fix action label
    recovery_ticks: int   # estimated ticks until primary risk < LO
    risk_reduction: float # risk score decrement applied each tick
    params: dict          # action-specific parameters


# One remedy per root cause category
_REMEDY_MAP: Dict[str, Remedy] = {
    "timeout": Remedy(
        action="backoff",
        recovery_ticks=4,
        risk_reduction=0.12,
        params={"timeout_factor": 0.5, "retry_interval_multiplier": 2},
    ),
    "error_surge": Remedy(
        action="circuit_break",
        recovery_ticks=6,
        risk_reduction=0.09,
        params={"pause_ticks": 3, "probe_interval_ticks": 2},
    ),
    "overload": Remedy(
        action="throttle",
        recovery_ticks=5,
        risk_reduction=0.10,
        params={"shed_fraction": 0.30, "queue_limit_requests": 100},
    ),
    "cascade": Remedy(
        action="isolate_dependency",
        recovery_ticks=8,
        risk_reduction=0.08,
        params={"use_cache": True, "decouple_timeout_ticks": 5},
    ),
    "intermittent": Remedy(
        action="retry_jitter",
        recovery_ticks=3,
        risk_reduction=0.15,
        params={"jitter_ms": 200, "max_retries": 3},
    ),
}


class RemediationEngine:
    """Apply fixes and tick-decay risk scores until primary is restored."""

    def __init__(self):
        # api_name → active fix state
        self._active: Dict[str, dict] = {}

    def apply_fix(self, api: str, root_cause_category: str) -> Remedy:
        """
        Start a fix for `api`. If a fix is already active, this is a no-op
        and returns the existing remedy (avoids thrashing).
        """
        if api in self._active:
            return self._active[api]["remedy"]

        remedy = _REMEDY_MAP.get(root_cause_category, _REMEDY_MAP["intermittent"])
        self._active[api] = {
            "remedy":          remedy,
            "ticks_remaining": remedy.recovery_ticks,
            "ticks_elapsed":   0,
            "root_cause":      root_cause_category,
        }
        return remedy

    def tick(self, api: str, current_risk: float) -> tuple:
        """
        Advance one tick for an active fix on `api`.
        Returns (updated_risk: float, fix_complete: bool).
        If no fix is active, returns (current_risk, False) unchanged.
        """
        if api not in self._active:
            return current_risk, False

        state = self._active[api]
        state["ticks_remaining"] -= 1
        state["ticks_elapsed"]   += 1
        updated_risk = max(0.0, current_risk - state["remedy"].risk_reduction)

        complete = state["ticks_remaining"] <= 0
        if complete:
            del self._active[api]

        return updated_risk, complete

    def tick_all(self, api_risks: Dict[str, float]) -> Dict[str, float]:
        """
        Tick every active fix and return the updated risk map.
        Convenience wrapper for the simulation loop.
        """
        updated = dict(api_risks)
        for api in list(self._active.keys()):
            updated[api], _ = self.tick(api, updated.get(api, 0.0))
        return updated

    def is_fixing(self, api: str) -> bool:
        return api in self._active

    def fix_status(self, api: str) -> Optional[dict]:
        return self._active.get(api)

    def active_fixes(self) -> Dict[str, dict]:
        return dict(self._active)
