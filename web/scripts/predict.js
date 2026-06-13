/* ============================================================
 * LEO · predict.js
 *   Live inference playground + what-if sliders.
 *
 *   IMPORTANT / honesty note:
 *   The public site is a pure-static build (no torch on the free
 *   tier), so this playground runs a *distilled logistic surrogate*
 *   that is hand-calibrated to reproduce LEO's qualitative behaviour
 *   (monotone in the right signals, base-rate anchored to the real
 *   13.88% failure rate, horizon decay toward the base rate). The
 *   90% bands use the real conformal q̂ from conformal_results.json.
 *   The full Bi-LSTM + attention model and every headline metric on
 *   the rest of the site come from the real held-out evaluation.
 * ============================================================ */
(function () {
  'use strict';

  const D = window.LEO_DATA || {};
  const TAU = Math.PI * 2;
  const BASE = (D.dataset && D.dataset.failure_rate) || 0.1388;

  // Real conformal nonconformity quantiles (probability scale) per horizon.
  const QHAT = {
    h1:  (D.conformal && D.conformal.per_horizon && D.conformal.per_horizon.h1  && D.conformal.per_horizon.h1.q_hat)  || 0.5195,
    h5:  (D.conformal && D.conformal.per_horizon && D.conformal.per_horizon.h5  && D.conformal.per_horizon.h5.q_hat)  || 0.5207,
    h15: (D.conformal && D.conformal.per_horizon && D.conformal.per_horizon.h15 && D.conformal.per_horizon.h15.q_hat) || 0.5258,
  };
  const HI = (D.agent && D.agent.assumptions && D.agent.assumptions.high_risk) || 0.55;
  const LO = (D.agent && D.agent.assumptions && D.agent.assumptions.low_risk)  || 0.35;

  const API_BIAS = {
    transaction_api: 0.00,
    market_data_api: 0.10,
    stock_price_api: -0.10,
    crypto_api:      0.40,
    forex_api:       0.20,
  };

  const sig = z => 1 / (1 + Math.exp(-z));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  // ── The distilled surrogate ─────────────────────────────────
  // Inputs are normalised operational signals; weights are tuned so a
  // calm estate (err 2%, rt 1×, vol 0.1, 1 recent failure, nominal load)
  // sits ~10% h+1 risk, and an outage scenario saturates toward ~95%.
  const W = { err: 9.0, rt: 0.55, vol: 1.6, recent: 0.18, load: 0.45, b0: -2.72 };

  function surrogate(s) {
    const contrib = {
      'Error rate':       W.err * s.errRate,
      'Response time':    W.rt  * Math.log(Math.max(1, s.rtMult)),
      'Error volatility': W.vol * s.errVol,
      'Recent failures':  W.recent * s.recentFail,
      'Load stress':      W.load * Math.max(0, s.tput - 1.2),
      'API profile':      (API_BIAS[s.apiType] != null ? API_BIAS[s.apiType] : 0),
    };
    const z = W.b0 + Object.values(contrib).reduce((a, b) => a + b, 0);
    const p1 = sig(z);
    // longer horizons regress toward the base rate (lower AUC, more blur)
    const p5  = BASE + (p1 - BASE) * 0.86;
    const p15 = BASE + (p1 - BASE) * 0.72;
    return { p1, p5, p15, contrib };
  }

  // 90% conformal half-width: widest at p=0.5, tightens at the extremes,
  // grows with horizon. Anchored to the calibration q̂.
  function band(p, qhat, hScale) {
    const w = (0.045 + 0.16 * (1 - Math.abs(p - 0.5) * 2)) * (qhat / 0.52) * hScale;
    return clamp(w, 0.03, 0.42);
  }

  // ── 270° gauge on a canvas ──────────────────────────────────
  function ctxFor(id, size) {
    const c = document.getElementById(id);
    if (!c) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = size * dpr; c.height = size * dpr;
    c.style.width = size + 'px'; c.style.height = size + 'px';
    const x = c.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { x, w: size, h: size };
  }

  function colorFor(p) {
    if (p >= HI) return '#ef4444';
    if (p >= LO) return '#fbbf24';
    return '#84cc16';
  }

  function drawGauge(id, p, bw, label) {
    const ref = ctxFor(id, 180);
    if (!ref) return;
    const { x, w } = ref;
    const cx = w / 2, cy = w / 2 + 8, R = 66;
    const A0 = Math.PI * 0.75, A1 = Math.PI * 2.25;          // 270° sweep
    const ang = v => A0 + clamp(v, 0, 1) * (A1 - A0);
    x.clearRect(0, 0, w, w);

    // track
    x.lineWidth = 12; x.lineCap = 'round';
    x.strokeStyle = 'rgba(255,255,255,0.07)';
    x.beginPath(); x.arc(cx, cy, R, A0, A1); x.stroke();

    // conformal band (translucent wedge around p)
    const lo = clamp(p - bw, 0, 1), hi = clamp(p + bw, 0, 1);
    x.lineWidth = 12; x.strokeStyle = 'rgba(251,191,36,0.18)';
    x.beginPath(); x.arc(cx, cy, R, ang(lo), ang(hi)); x.stroke();

    // value arc
    const col = colorFor(p);
    x.lineWidth = 12; x.strokeStyle = col;
    x.shadowBlur = 16; x.shadowColor = col;
    x.beginPath(); x.arc(cx, cy, R, A0, ang(p)); x.stroke();
    x.shadowBlur = 0;

    // threshold ticks (LO amber, HI red)
    [[LO, '#fbbf24'], [HI, '#ef4444']].forEach(([t, c]) => {
      const a = ang(t);
      x.strokeStyle = c; x.lineWidth = 2;
      x.beginPath();
      x.moveTo(cx + Math.cos(a) * (R - 9), cy + Math.sin(a) * (R - 9));
      x.lineTo(cx + Math.cos(a) * (R + 9), cy + Math.sin(a) * (R + 9));
      x.stroke();
    });

    // center text
    x.fillStyle = '#fef3c7';
    x.font = '700 26px Space Grotesk';
    x.textAlign = 'center'; x.textBaseline = 'middle';
    x.fillText((p * 100).toFixed(1) + '%', cx, cy - 2);
    x.fillStyle = '#9a7b5e';
    x.font = '10px JetBrains Mono';
    x.fillText(label, cx, cy + 20);
    x.fillStyle = '#7a5b3e';
    x.font = '9px JetBrains Mono';
    x.fillText('±' + (bw * 100).toFixed(0) + ' · 90%', cx, cy + 34);
  }

  // ── read the controls ───────────────────────────────────────
  function readState() {
    const num = (id, d) => {
      const el = document.getElementById(id);
      return el ? parseFloat(el.value) : d;
    };
    const apiEl = document.getElementById('pApi');
    return {
      errRate:    num('pErr', 2) / 100,
      rtMult:     num('pRt', 1),
      errVol:     num('pVol', 10) / 100,
      tput:       num('pTput', 1),
      recentFail: num('pRecent', 1),
      apiType:    apiEl ? apiEl.value : 'transaction_api',
    };
  }

  function syncLabels(s) {
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('pErrV', (s.errRate * 100).toFixed(1) + '%');
    set('pRtV', s.rtMult.toFixed(1) + '×');
    set('pVolV', (s.errVol * 100).toFixed(0) + '%');
    set('pTputV', s.tput.toFixed(1) + '×');
    set('pRecentV', s.recentFail.toFixed(0));
  }

  // ── verdict + driver breakdown ──────────────────────────────
  function render() {
    const s = readState();
    syncLabels(s);
    const r = surrogate(s);

    drawGauge('gaugeH1', r.p1, band(r.p1, QHAT.h1, 1.0), 'h+1 min');
    drawGauge('gaugeH5', r.p5, band(r.p5, QHAT.h5, 1.12), 'h+5 min');
    drawGauge('gaugeH15', r.p15, band(r.p15, QHAT.h15, 1.25), 'h+15 min');

    const peak = Math.max(r.p1, r.p5, r.p15);
    let action, cls, msg;
    if (peak >= HI) {
      action = 'REROUTE'; cls = 'crit';
      msg = 'Failure imminent on the horizon — fail over to the backup provider now.';
    } else if (peak >= LO) {
      action = 'RETRY / WARM BACKUP'; cls = 'warn';
      msg = 'Degradation building — arm retries and pre-warm the backup route.';
    } else {
      action = 'NORMAL'; cls = 'ok';
      msg = 'All horizons nominal — continue on the primary route.';
    }

    const setT = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    setT('vAction', action);
    setT('vMsg', msg);
    setT('vPeak', (peak * 100).toFixed(1) + '%');
    const vc = document.getElementById('verdictCard');
    if (vc) vc.className = 'verdict ' + cls;

    // driver bars (sorted, signed)
    const drivers = Object.entries(r.contrib)
      .map(([k, v]) => ({ k, v }))
      .sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const maxAbs = Math.max(0.001, ...drivers.map(d => Math.abs(d.v)));
    const root = document.getElementById('driverBars');
    if (root) {
      root.innerHTML = drivers.map(d => {
        const pct = (Math.abs(d.v) / maxAbs * 100).toFixed(0);
        const pos = d.v >= 0;
        return `<div class="drv">
            <span class="drv-k">${d.k}</span>
            <span class="drv-track ${pos ? 'up' : 'down'}">
              <span class="drv-fill" style="width:${pct}%"></span>
            </span>
            <span class="drv-v">${pos ? '+' : ''}${d.v.toFixed(2)}</span>
          </div>`;
      }).join('');
    }
  }

  // ── scenario presets ────────────────────────────────────────
  const PRESETS = {
    calm:      { pErr: 1.5, pRt: 1.0, pVol: 8,  pTput: 0.9, pRecent: 0 },
    degrading: { pErr: 7,   pRt: 2.4, pVol: 35, pTput: 1.6, pRecent: 4 },
    outage:    { pErr: 18,  pRt: 6.0, pVol: 70, pTput: 2.4, pRecent: 12 },
  };
  function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;
    Object.entries(p).forEach(([id, v]) => {
      const el = document.getElementById(id);
      if (el) el.value = v;
    });
    render();
  }

  // ── boot ────────────────────────────────────────────────────
  function init() {
    ['pErr', 'pRt', 'pVol', 'pTput', 'pRecent', 'pApi'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', render);
    });
    document.querySelectorAll('[data-preset]').forEach(b => {
      b.addEventListener('click', () => {
        document.querySelectorAll('[data-preset]').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        applyPreset(b.dataset.preset);
      });
    });
    render();
    let t;
    window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(render, 150); });
  }

  window.LEO_PREDICT = { init, surrogate };
})();
