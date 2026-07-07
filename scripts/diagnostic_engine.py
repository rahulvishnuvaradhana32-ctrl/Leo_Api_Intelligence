#!/usr/bin/env python3
"""
diagnostic_engine.py — Rule-based root cause analysis from API feature vectors.

Operates on the same 43 raw (unscaled) features already used by the LSTM
(see agent_simulation.py:FEATURE_COLS).

Root cause categories (evaluated in priority order):
  cascade      — cross-API systemic stress is driving the failure
  overload     — traffic burst / request surge overwhelming the endpoint
  timeout      — latency multiplier / spike is the dominant signal
  error_surge  — error rate burst / boost is the dominant signal
  intermittent — elevated instability without a single dominant cause
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class RootCause:
    category: str          # cascade | overload | timeout | error_surge | intermittent
    confidence: float      # 0.0 – 1.0
    evidence: Dict[str, float]   # feature → value that triggered this diagnosis
    remediation_hint: str  # passed to RemediationEngine


# ── Thresholds (raw feature space) ───────────────────────────────────────────
_SYSTEMIC_STRESS_HI = 0.6
_N_APIS_ELEVATED    = 2
_BURST_RATIO_HI     = 2.0
_TRAFFIC_CHANGE_HI  = 1.5
_RT_MULTIPLIER_HI   = 2.0
_LATENCY_SPIKE_HI   = 1.0
_ERROR_BURST_HI     = 1.0
_ERROR_BOOST_HI     = 0.5
_INSTABILITY_HI     = 0.5


class DiagnosticEngine:
    """Classify failure root cause from a raw (unscaled) feature dict."""

    def diagnose(self, features: dict) -> RootCause:
        f = features

        # 1. Cascade — cross-API systemic stress
        systemic      = float(f.get("systemic_stress_index", 0))
        n_elev        = float(f.get("n_apis_elevated", 0))
        avg_err_other = float(f.get("avg_error_rate_others", 0))
        max_err_other = float(f.get("max_error_rate_others", 0))
        if systemic >= _SYSTEMIC_STRESS_HI or n_elev >= _N_APIS_ELEVATED:
            conf = min(1.0, (systemic + max_err_other) / 1.2)
            return RootCause(
                category="cascade",
                confidence=round(conf, 3),
                evidence={
                    "systemic_stress_index":  systemic,
                    "n_apis_elevated":        n_elev,
                    "avg_error_rate_others":  avg_err_other,
                },
                remediation_hint="isolate_dependency",
            )

        # 2. Overload — traffic burst
        burst = float(f.get("burst_ratio", 0))
        traf  = float(f.get("traffic_change", 0))
        if burst >= _BURST_RATIO_HI or traf >= _TRAFFIC_CHANGE_HI:
            conf = min(1.0, max(burst / 4.0, traf / 3.0))
            return RootCause(
                category="overload",
                confidence=round(conf, 3),
                evidence={"burst_ratio": burst, "traffic_change": traf},
                remediation_hint="throttle",
            )

        # 3. Timeout — latency dominant
        rt_mult  = float(f.get("rt_multiplier", 0))
        lat_spk  = float(f.get("latency_spike", 0))
        lat_slp  = float(f.get("latency_slope", 0))
        if rt_mult >= _RT_MULTIPLIER_HI or lat_spk >= _LATENCY_SPIKE_HI:
            conf = min(1.0, max(rt_mult / 4.0, lat_spk))
            return RootCause(
                category="timeout",
                confidence=round(conf, 3),
                evidence={
                    "rt_multiplier":  rt_mult,
                    "latency_spike":  lat_spk,
                    "latency_slope":  lat_slp,
                },
                remediation_hint="backoff",
            )

        # 4. Error surge
        err_burst = float(f.get("error_burst", 0))
        err_boost = float(f.get("error_rate_boost", 0))
        err_rate  = float(f.get("error_rate_rolling", 0))
        if err_burst >= _ERROR_BURST_HI or err_boost >= _ERROR_BOOST_HI:
            conf = min(1.0, max(err_burst, err_boost + err_rate))
            return RootCause(
                category="error_surge",
                confidence=round(conf, 3),
                evidence={"error_burst": err_burst, "error_rate_boost": err_boost},
                remediation_hint="circuit_break",
            )

        # 5. Intermittent — catch-all
        instab = float(f.get("instability_index", 0))
        conf   = min(1.0, instab + 0.2)
        return RootCause(
            category="intermittent",
            confidence=round(conf, 3),
            evidence={"instability_index": instab},
            remediation_hint="retry_jitter",
        )

    def explain(self, root_cause: RootCause) -> str:
        descriptions = {
            "cascade":     "Multiple APIs are degrading simultaneously — systemic upstream stress.",
            "overload":    "Traffic burst exceeded endpoint capacity — request surge detected.",
            "timeout":     "Response latency spike — endpoint is slow to respond.",
            "error_surge": "Error rate burst — endpoint is returning errors at elevated rate.",
            "intermittent":"Unstable signal without dominant cause — likely transient fault.",
        }
        base = descriptions.get(root_cause.category, "Unknown failure pattern.")
        return f"[{root_cause.category.upper()} {root_cause.confidence:.0%}] {base}"
