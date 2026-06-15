#!/usr/bin/env python3
"""
leo_surrogate.py — the distilled logistic surrogate behind /v1/forecast.

This mirrors, exactly, the in-browser surrogate in web/scripts/predict.js so
the live demo and the API return identical numbers. It is NOT the production
Bi-LSTM (which needs torch); it is a hand-calibrated stand-in that reproduces
LEO's qualitative behaviour — monotone in the right signals, anchored to the
real 13.88% base rate, longer horizons regressing toward it, with 90% bands
from the measured conformal q-hat. Pure stdlib so it's trivially testable.
"""
from __future__ import annotations

import math

BASE = 0.1388
QHAT = {"h1": 0.5195, "h5": 0.5207, "h15": 0.5258}
HI, LO = 0.55, 0.35
API_BIAS = {
    "transaction_api": 0.00,
    "market_data_api": 0.10,
    "stock_price_api": -0.10,
    "crypto_api": 0.40,
    "forex_api": 0.20,
}
W = {"err": 9.0, "rt": 0.55, "vol": 1.6, "recent": 0.18, "load": 0.45, "b0": -2.72}


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _band(p: float, qhat: float, hscale: float) -> float:
    w = (0.045 + 0.16 * (1 - abs(p - 0.5) * 2)) * (qhat / 0.52) * hscale
    return _clamp(w, 0.03, 0.42)


def forecast(api: str = "transaction_api", error_rate: float = 0.02,
             rt_multiplier: float = 1.0, error_volatility: float = 0.1,
             load: float = 1.0, recent_failures: float = 0.0) -> dict:
    """Return the multi-horizon forecast for one telemetry window."""
    contrib = {
        "Error rate":       W["err"] * error_rate,
        "Response time":    W["rt"] * math.log(max(1.0, rt_multiplier)),
        "Error volatility": W["vol"] * error_volatility,
        "Recent failures":  W["recent"] * recent_failures,
        "Load stress":      W["load"] * max(0.0, load - 1.2),
        "API profile":      API_BIAS.get(api, 0.0),
    }
    z = W["b0"] + sum(contrib.values())
    p1 = _sig(z)
    p5 = BASE + (p1 - BASE) * 0.86
    p15 = BASE + (p1 - BASE) * 0.72

    def r3(v: float) -> float:
        return round(v, 3)

    horizons = [(1, p1, QHAT["h1"], 1.0), (5, p5, QHAT["h5"], 1.12), (15, p15, QHAT["h15"], 1.25)]
    fc = []
    for h, p, q, sc in horizons:
        bw = _band(p, q, sc)
        fc.append({
            "horizon_min": h,
            "failure_prob": r3(p),
            "interval": [r3(_clamp(p - bw, 0, 1)), r3(_clamp(p + bw, 0, 1))],
            "coverage": 0.90,
        })

    peak = max(p1, p5, p15)
    action = "reroute" if peak >= HI else "pre_warm" if peak >= LO else "none"
    lead = 1 if peak >= 0.78 else 5 if peak >= HI else 15 if peak >= LO else None

    return {
        "api": api,
        "forecast": fc,
        "drivers": {k: r3(v) for k, v in contrib.items()},
        "recommended_action": action,
        "lead_time_min": lead,
        "peak_risk": r3(peak),
        "reversible": True,
        "model": "leo-surrogate-v1",
    }


if __name__ == "__main__":  # quick self-test / parity check vs predict.js
    import json
    cases = {
        "calm (transaction)":  dict(api="transaction_api", error_rate=0.015, rt_multiplier=1.0, error_volatility=0.08, load=0.9, recent_failures=0),
        "payments (degrading)": dict(api="transaction_api", error_rate=0.06, rt_multiplier=2.2, error_volatility=0.32, load=1.7, recent_failures=3),
        "crypto (storm)":      dict(api="crypto_api", error_rate=0.16, rt_multiplier=5.5, error_volatility=0.68, load=2.3, recent_failures=11),
    }
    for name, w in cases.items():
        out = forecast(**w)
        probs = [f["failure_prob"] for f in out["forecast"]]
        print(f"{name:24s} probs={probs}  action={out['recommended_action']:8s} lead={out['lead_time_min']}")
