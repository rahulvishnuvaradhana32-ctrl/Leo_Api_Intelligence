/* ============================================================
 * LEO · reliability.js
 *   SRE view of the model:
 *     - Coverage SLO cards + error-budget consumption (real conformal)
 *     - 30-day error-budget burn-down (illustrative, anchored to
 *       the measured per-horizon coverage)
 *     - Incident timeline (real self-healing runs from data.js)
 *     - Proactive vs reactive agent comparison (real simulation)
 * ============================================================ */
(function () {
  'use strict';

  const D = window.LEO_DATA || {};
  const TAU = Math.PI * 2;
  const TARGET = (D.conformal && D.conformal.target_coverage) || 0.90;
  const PH = (D.conformal && D.conformal.per_horizon) || {};

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const pct = (v, d = 2) => (v * 100).toFixed(d) + '%';

  const HORIZONS = [
    { k: 'h1', label: 'h+1', color: '#4D8DFF' },
    { k: 'h5', label: 'h+5', color: '#8B5CF6' },
    { k: 'h15', label: 'h+15', color: '#A78BFA' },
  ];

  function ctxFor(id) {
    const c = document.getElementById(id);
    if (!c) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = c.getBoundingClientRect();
    const w = rect.width || c.clientWidth;
    const h = parseInt(c.getAttribute('height')) || 240;
    c.width = w * dpr; c.height = h * dpr;
    c.style.width = w + 'px'; c.style.height = h + 'px';
    const x = c.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { x, w, h };
  }

  // ── SLO summary cards ───────────────────────────────────────
  function renderSlo() {
    const root = document.getElementById('sloCards');
    if (!root) return;
    const budget = 1 - TARGET; // allowed miscoverage
    root.innerHTML = HORIZONS.map(({ k, label, color }) => {
      const cov = (PH[k] && PH[k].coverage) || 0;
      const miss = 1 - cov;
      const consumed = clamp(miss / budget, 0, 1.25);
      const pass = (PH[k] && PH[k].status) === 'PASS';
      const remain = (1 - consumed) * 100;
      return `<div class="slo-card ${pass ? 'pass' : 'fail'}">
        <div class="slo-top"><span class="slo-h" style="color:${color}">${label}</span>
          <span class="slo-badge ${pass ? 'ok' : 'bad'}">${pass ? 'PASS' : 'BREACH'}</span></div>
        <div class="slo-cov">${pct(cov)}</div>
        <div class="slo-sub">coverage · target ${pct(TARGET, 0)}</div>
        <div class="slo-budget">
          <div class="sb-label"><span>error budget</span><b>${remain >= 0 ? remain.toFixed(0) : '0'}% left</b></div>
          <div class="sb-track"><span class="sb-fill ${consumed >= 1 ? 'over' : ''}" style="width:${Math.min(100, consumed * 100).toFixed(0)}%"></span></div>
        </div>
        <div class="slo-q">q̂ ${(PH[k] && PH[k].q_hat || 0).toFixed(3)} · width ${(PH[k] && PH[k].width || 0).toFixed(3)}</div>
      </div>`;
    }).join('');
  }

  // ── Error-budget burn-down (30-day, illustrative) ───────────
  function drawBurn() {
    const ref = ctxFor('burnChart');
    if (!ref) return;
    const { x, w, h } = ref;
    const padL = 44, padR = 14, padT = 22, padB = 28;
    const budget = 1 - TARGET;
    const N = 30;

    function rnd(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

    const sx = i => padL + (i / (N - 1)) * (w - padL - padR);
    const sy = v => padT + (1 - clamp(v, 0, 1.1) / 1.1) * (h - padT - padB);

    x.clearRect(0, 0, w, h);
    // grid + y labels (remaining budget %)
    x.strokeStyle = 'rgba(124,148,210,0.10)'; x.lineWidth = 1;
    [0, 0.25, 0.5, 0.75, 1].forEach(v => {
      const y = sy(v);
      x.beginPath(); x.moveTo(padL, y); x.lineTo(w - padR, y); x.stroke();
      x.fillStyle = '#6F7C9C'; x.font = '10px IBM Plex Mono'; x.textAlign = 'right';
      x.fillText((v * 100).toFixed(0) + '%', padL - 6, y + 3);
    });
    // breach line at 0
    x.strokeStyle = 'rgba(239,68,68,0.5)'; x.setLineDash([4, 4]);
    x.beginPath(); x.moveTo(padL, sy(0)); x.lineTo(w - padR, sy(0)); x.stroke(); x.setLineDash([]);
    x.fillStyle = '#ef4444'; x.font = '10px IBM Plex Mono'; x.textAlign = 'left';
    x.fillText('budget exhausted', padL + 6, sy(0) - 5);

    HORIZONS.forEach(({ k, color }, idx) => {
      const cov = (PH[k] && PH[k].coverage) || 0;
      const consumptionRate = (1 - cov) / budget; // ~1 means exactly on budget
      const r = rnd(1000 + idx * 7);
      x.strokeStyle = color; x.lineWidth = 2.4; x.beginPath();
      let breachX = null;
      for (let i = 0; i < N; i++) {
        const frac = i / (N - 1);
        const noise = (r() - 0.5) * 0.05;
        const remain = 1 - consumptionRate * frac + noise;
        if (remain <= 0 && breachX == null) breachX = sx(i);
        i === 0 ? x.moveTo(sx(i), sy(remain)) : x.lineTo(sx(i), sy(remain));
      }
      x.stroke();
      if (breachX != null) {
        x.fillStyle = color; x.beginPath(); x.arc(breachX, sy(0), 4, 0, TAU); x.fill();
      }
    });

    // x label
    x.fillStyle = '#6F7C9C'; x.font = '10px IBM Plex Mono'; x.textAlign = 'center';
    ['day 1', 'day 15', 'day 30'].forEach((t, i) => x.fillText(t, sx(i * (N - 1) / 2), h - 9));

    // legend
    x.font = '11px IBM Plex Mono'; x.textAlign = 'left';
    let lx = padL + 6;
    HORIZONS.forEach(({ label, color }) => {
      x.fillStyle = color; x.fillRect(lx, padT - 12, 10, 3);
      x.fillStyle = '#A6B2D0'; x.fillText(label, lx + 14, padT - 8);
      lx += 56;
    });
  }

  // ── Incident timeline (real self-healing runs) ──────────────
  function buildTimeline() {
    const root = document.getElementById('timeline');
    if (!root) return;
    const runs = (D.selfheal || []).slice().reverse(); // newest first
    if (!runs.length) { root.innerHTML = '<p class="muted">No self-healing runs logged yet.</p>'; return; }
    root.innerHTML = runs.map(r => {
      const drift = r.drift_detected;
      const drifted = Object.entries(r.signals || {}).filter(([, v]) => v.drifted).map(([k]) => k);
      const cls = drift ? 'incident' : (r.model_updated ? 'update' : 'clear');
      const dot = drift ? 'bad' : (r.model_updated ? 'upd' : 'ok');
      const when = (r.timestamp || '').replace('T', ' ').slice(0, 16);
      const sigBadges = Object.entries(r.signals || {}).map(([k, v]) =>
        `<span class="sig ${v.drifted ? 'on' : ''}" title="KS ${v.ks}, p ${v.p}">${k.replace(/_/g, ' ')}</span>`
      ).join('');
      return `<div class="tl-row ${cls}">
        <span class="tl-dot ${dot}"></span>
        <div class="tl-body">
          <div class="tl-head">
            <b>${when}</b>
            <span class="tl-mode">${r.mode || ''}</span>
            ${drift ? '<span class="tl-tag bad">DRIFT</span>' : '<span class="tl-tag ok">stable</span>'}
            ${r.model_updated ? '<span class="tl-tag upd">model updated</span>' : ''}
          </div>
          <div class="tl-meta">${(r.rows || 0).toLocaleString()} rows · failure-rate ${((r.failure_rate || 0) * 100).toFixed(2)}%${drifted.length ? ' · drift: ' + drifted.join(', ').replace(/_/g, ' ') : ''}</div>
          <div class="tl-sigs">${sigBadges}</div>
        </div>
      </div>`;
    }).join('');
  }

  // ── Proactive vs reactive comparison ────────────────────────
  function buildComparison() {
    const root = document.getElementById('compareBars');
    if (!root) return;
    const a = D.agent || {}; const pro = a.proactive || {}; const rea = a.reactive || {};
    const rows = [
      { k: 'Failure rate', p: pro.rate, r: rea.rate, fmt: v => (v * 100).toFixed(2) + '%', lowerBetter: true },
      { k: 'Failures / 10k', p: pro.failures, r: rea.failures, fmt: v => Math.round(v).toLocaleString(), lowerBetter: true },
      { k: 'Provider switches', p: pro.switches, r: rea.switches, fmt: v => Math.round(v).toLocaleString(), lowerBetter: true },
      { k: 'Cost / 1k tx', p: pro.cost_1k, r: rea.cost_1k, fmt: v => '$' + Math.round(v).toLocaleString(), lowerBetter: true },
      { k: 'Avg latency', p: pro.latency, r: rea.latency, fmt: v => (v).toFixed(3) + 's', lowerBetter: true },
    ];
    root.innerHTML = rows.map(row => {
      const max = Math.max(row.p, row.r) || 1;
      const proWin = row.lowerBetter ? row.p <= row.r : row.p >= row.r;
      return `<div class="cmp">
        <div class="cmp-k">${row.k}</div>
        <div class="cmp-bars">
          <div class="cmp-bar pro ${proWin ? 'win' : ''}"><span style="width:${(row.p / max * 100).toFixed(0)}%"></span><i>${row.fmt(row.p)}</i></div>
          <div class="cmp-bar rea ${!proWin ? 'win' : ''}"><span style="width:${(row.r / max * 100).toFixed(0)}%"></span><i>${row.fmt(row.r)}</i></div>
        </div>
      </div>`;
    }).join('');
  }

  function init() {
    renderSlo();
    drawBurn();
    buildTimeline();
    buildComparison();
    let t;
    window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(drawBurn, 160); });
  }

  window.LEO_RELIABILITY = { init };
})();
