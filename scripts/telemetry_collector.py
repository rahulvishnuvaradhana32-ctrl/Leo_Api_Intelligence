#!/usr/bin/env python3
"""
telemetry_collector.py — Converts real HTTP traffic into LSTM feature sequences.

Maintains a rolling buffer of real requests per API endpoint and computes all
43 FEATURE_COLS from actual measured response_time / success / timestamp data.
The output is a (seq_len, 43) numpy array ready for LSTM inference.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import numpy as np

# Matches agent_simulation.py FEATURE_COLS exactly (order matters)
FEATURE_COLS = [
    'response_time', 'request_count',
    'hour', 'day_of_week', 'is_market_hours', 'is_financial_peak',
    'is_weekend', 'is_holiday',
    'response_time_rolling_mean', 'response_time_rolling_std',
    'error_rate_rolling', 'response_time_variance', 'error_volatility',
    'response_time_lag_1', 'response_time_lag_5', 'error_rate_lag_1',
    'response_time_ema_10', 'response_time_ema_30', 'error_rate_ema_10',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'high_frequency_api', 'api_complexity',
    'error_rate_boost', 'rt_multiplier',
    'latency_diff_1', 'latency_diff_5',
    'error_rate_diff_1', 'error_rate_diff_5',
    'latency_spike', 'error_burst', 'instability_index',
    'latency_slope', 'error_slope',
    'traffic_change', 'burst_ratio',
    'avg_error_rate_others', 'max_error_rate_others',
    'n_apis_elevated', 'corr_with_similar_api',
    'systemic_stress_index',
]

# Static API-type flags (matching training data encoding)
_API_FLAGS = {
    "transaction_api":  {"high_frequency_api": 0.0, "api_complexity": 0.8},
    "stock_price_api":  {"high_frequency_api": 1.0, "api_complexity": 0.5},
    "crypto_api":       {"high_frequency_api": 1.0, "api_complexity": 0.6},
    "forex_api":        {"high_frequency_api": 1.0, "api_complexity": 0.5},
    "market_data_api":  {"high_frequency_api": 0.5, "api_complexity": 0.6},
}

_US_HOLIDAYS_2026 = {
    (1, 1), (1, 19), (2, 16), (5, 25), (6, 19),
    (7, 4), (9, 7), (11, 26), (12, 25),
}


@dataclass
class RequestRecord:
    timestamp: float        # time.time()
    response_time: float    # seconds
    success: bool
    status_code: int
    request_num: int        # cumulative count


class ApiTelemetry:
    """
    Rolling buffer that converts real HTTP traffic into LSTM feature sequences.

    Usage:
        tel = ApiTelemetry("transaction_api")
        tel.record(0.312, True, 200)
        seq = tel.get_feature_sequence()   # None until 30 records in buffer
        if seq is not None:
            risk = lstm_infer(seq)
    """

    # Keep extra history beyond the sequence window for lag/rolling computation
    _HISTORY = 250

    def __init__(self, api_name: str = "transaction_api"):
        self.api_name   = api_name
        self._buf: deque[RequestRecord] = deque(maxlen=self._HISTORY)
        self._total     = 0
        self._flags     = _API_FLAGS.get(api_name, _API_FLAGS["transaction_api"])

        # Running EMAs (updated on every record call)
        self._ema10_rt:  float = 0.0
        self._ema30_rt:  float = 0.0
        self._ema10_err: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(self, response_time: float, success: bool, status_code: int = 200):
        self._total += 1
        rec = RequestRecord(
            timestamp=time.time(),
            response_time=response_time,
            success=success,
            status_code=status_code,
            request_num=self._total,
        )
        self._buf.append(rec)

        err = 0.0 if success else 1.0
        alpha10 = 2 / (10 + 1)
        alpha30 = 2 / (30 + 1)
        self._ema10_rt  = alpha10 * response_time + (1 - alpha10) * self._ema10_rt
        self._ema30_rt  = alpha30 * response_time + (1 - alpha30) * self._ema30_rt
        self._ema10_err = alpha10 * err           + (1 - alpha10) * self._ema10_err

    def get_feature_sequence(
        self,
        seq_len: int = 30,
        cross_api_stats: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Returns (seq_len, n_features) float32 array, or None if not enough data.

        cross_api_stats: dict with keys avg_error_rate_others, max_error_rate_others,
                         n_apis_elevated, corr_with_similar_api, systemic_stress_index.
                         If None, these features are zeroed.
        """
        buf = list(self._buf)
        if len(buf) < seq_len:
            return None

        # Use the last seq_len records as the sequence window
        window = buf[-seq_len:]
        # Use up to 50 extra preceding records for rolling/lag context
        context_start = max(0, len(buf) - seq_len - 50)
        context = buf[context_start:]

        rows = []
        for i, rec in enumerate(window):
            feat = self._compute_features(rec, i, window, context, cross_api_stats)
            rows.append(feat)

        return np.array(rows, dtype=np.float32)

    def current_metrics(self) -> dict:
        if not self._buf:
            return {"avg_rt_s": 0.0, "error_rate": 0.0, "total_requests": 0}
        recent = list(self._buf)[-50:]
        rts    = [r.response_time for r in recent]
        errs   = [0 if r.success else 1 for r in recent]
        return {
            "avg_rt_s":       round(sum(rts) / len(rts), 4),
            "p95_rt_s":       round(sorted(rts)[int(len(rts) * 0.95)], 4),
            "error_rate":     round(sum(errs) / len(errs), 4),
            "total_requests": self._total,
            "window":         len(recent),
        }

    # ── Feature computation ────────────────────────────────────────────────────

    def _compute_features(
        self,
        rec: RequestRecord,
        pos_in_window: int,
        window: list,
        context: list,
        cross_api_stats: Optional[dict],
    ) -> list:
        dt = datetime.fromtimestamp(rec.timestamp, tz=timezone.utc)
        h  = dt.hour
        dow = dt.weekday()   # 0=Mon … 6=Sun
        is_weekend = float(dow >= 5)
        is_market  = float(9 <= h < 17)
        is_peak    = float((9 <= h < 10) or (14 <= h < 15))
        is_holiday = float((dt.month, dt.day) in _US_HOLIDAYS_2026)

        # Rolling window: up to 20 records before this one in context
        ctx_idx = next(
            (j for j, r in enumerate(context) if r is rec), len(context) - 1
        )
        roll_slice = [context[j] for j in range(max(0, ctx_idx - 20), ctx_idx + 1)]
        rt_roll    = [r.response_time for r in roll_slice]
        err_roll   = [0.0 if r.success else 1.0 for r in roll_slice]

        rt_mean  = float(np.mean(rt_roll))       if rt_roll else rec.response_time
        rt_std   = float(np.std(rt_roll))        if len(rt_roll) > 1 else 0.0
        rt_var   = rt_std ** 2
        err_rate = float(np.mean(err_roll))       if err_roll else 0.0
        err_vol  = float(np.std(err_roll))        if len(err_roll) > 1 else 0.0

        # Lag features (relative to current position in window)
        lag1_rt   = window[pos_in_window - 1].response_time if pos_in_window >= 1 else rec.response_time
        lag5_rt   = window[pos_in_window - 5].response_time if pos_in_window >= 5 else rec.response_time
        lag1_err  = 0.0 if (window[pos_in_window - 1].success if pos_in_window >= 1 else rec.success) else 1.0

        # EMA: use running state at current record (approximate: use context EMAs)
        ema10_rt  = self._running_ema(rt_roll, alpha=2/(10+1))
        ema30_rt  = self._running_ema(rt_roll, alpha=2/(30+1))
        ema10_err = self._running_ema(err_roll, alpha=2/(10+1))

        # Cyclical encoding
        h_sin  = math.sin(2 * math.pi * h / 24)
        h_cos  = math.cos(2 * math.pi * h / 24)
        d_sin  = math.sin(2 * math.pi * dow / 7)
        d_cos  = math.cos(2 * math.pi * dow / 7)

        # Stress signals (relative to rolling baseline)
        baseline_rt  = rt_mean if rt_mean > 0 else 0.3
        rt_mult      = rec.response_time / baseline_rt
        err_boost    = max(0.0, err_rate - 0.05)   # excess above 5% baseline

        # Diff features
        diff1_rt  = rec.response_time - lag1_rt
        diff5_rt  = rec.response_time - lag5_rt
        lag1_err_prev = 0.0 if (pos_in_window >= 2 and window[pos_in_window-2].success) else 1.0 if pos_in_window >= 2 else 0.0
        diff1_err = lag1_err - lag1_err_prev
        diff5_err = lag1_err - (0.0 if (pos_in_window >= 6 and window[pos_in_window-6].success) else lag1_err)

        # Spike / burst / instability
        lat_spike   = float(rec.response_time > rt_mean + 2 * rt_std) if rt_std > 0 else 0.0
        err_burst   = float(err_rate > 0.5)
        instability = min(1.0, (rt_std / baseline_rt) + err_vol)

        # Slope (linear trend over roll_slice)
        lat_slope = self._slope([r.response_time for r in roll_slice])
        err_slope = self._slope(err_roll)

        # Traffic change / burst ratio
        n_prev  = max(1, len(roll_slice) // 2)
        cnt_now = len(roll_slice) - n_prev
        cnt_old = n_prev
        traffic_change = (cnt_now - cnt_old) / max(cnt_old, 1)
        burst_ratio    = cnt_now / max(cnt_old, 1)

        # Cross-API features
        if cross_api_stats:
            avg_err_others      = float(cross_api_stats.get("avg_error_rate_others", 0.0))
            max_err_others      = float(cross_api_stats.get("max_error_rate_others", 0.0))
            n_apis_elevated     = float(cross_api_stats.get("n_apis_elevated", 0.0))
            corr_similar        = float(cross_api_stats.get("corr_with_similar_api", 0.0))
            systemic_stress     = float(cross_api_stats.get("systemic_stress_index", 0.0))
        else:
            avg_err_others = max_err_others = n_apis_elevated = 0.0
            corr_similar   = systemic_stress = 0.0

        return [
            rec.response_time,          # response_time
            float(rec.request_num),     # request_count
            float(h),                   # hour
            float(dow),                 # day_of_week
            is_market,                  # is_market_hours
            is_peak,                    # is_financial_peak
            is_weekend,                 # is_weekend
            is_holiday,                 # is_holiday
            rt_mean,                    # response_time_rolling_mean
            rt_std,                     # response_time_rolling_std
            err_rate,                   # error_rate_rolling
            rt_var,                     # response_time_variance
            err_vol,                    # error_volatility
            lag1_rt,                    # response_time_lag_1
            lag5_rt,                    # response_time_lag_5
            lag1_err,                   # error_rate_lag_1
            ema10_rt,                   # response_time_ema_10
            ema30_rt,                   # response_time_ema_30
            ema10_err,                  # error_rate_ema_10
            h_sin,                      # hour_sin
            h_cos,                      # hour_cos
            d_sin,                      # dow_sin
            d_cos,                      # dow_cos
            self._flags["high_frequency_api"],  # high_frequency_api
            self._flags["api_complexity"],       # api_complexity
            err_boost,                  # error_rate_boost
            rt_mult,                    # rt_multiplier
            diff1_rt,                   # latency_diff_1
            diff5_rt,                   # latency_diff_5
            diff1_err,                  # error_rate_diff_1
            diff5_err,                  # error_rate_diff_5
            lat_spike,                  # latency_spike
            err_burst,                  # error_burst
            instability,                # instability_index
            lat_slope,                  # latency_slope
            err_slope,                  # error_slope
            traffic_change,             # traffic_change
            burst_ratio,                # burst_ratio
            avg_err_others,             # avg_error_rate_others
            max_err_others,             # max_error_rate_others
            n_apis_elevated,            # n_apis_elevated
            corr_similar,               # corr_with_similar_api
            systemic_stress,            # systemic_stress_index
        ]

    @staticmethod
    def _running_ema(series: list, alpha: float) -> float:
        if not series:
            return 0.0
        val = series[0]
        for x in series[1:]:
            val = alpha * x + (1 - alpha) * val
        return val

    @staticmethod
    def _slope(series: list) -> float:
        n = len(series)
        if n < 2:
            return 0.0
        x = list(range(n))
        xm = sum(x) / n
        ym = sum(series) / n
        num = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, series))
        den = sum((xi - xm) ** 2 for xi in x)
        return num / den if den > 0 else 0.0
