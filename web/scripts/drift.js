/* ============================================================
 * LEO · drift.js
 *   Data-drift monitor. Pulls the live self-healing log from
 *   /api/drift when served by web_server.py, and falls back to the
 *   baked snapshot (window.LEO_DATA.selfheal) for the static build.
 *
 *   - Per-signal KS-statistic trend across runs (feature drift)
 *   - Latest-run signal grid (KS / p-value / drifted flag)
 *   - Summary tiles (runs, signals drifting, failure-rate trend)
 * ============================================================ */
(function () {
  'use strict';

  const D = window.LEO_DATA || {};
  const TAU = Math.PI * 2;

  // KS p<0.05 alone isn't enough — the pipeline flags drift only on a
  // large effect size, so we surface KS against an effect-size guide.
  const KS_GUIDE = 0.10;

  const SIGNALS = [
    { k: 'error_rate_rolling',         label: 'error rate (rolling)',  color: '#fbbf24' },
    { k: 'response_time_rolling_mean', label: 'response time (mean)',  color: '#dc2626' },
    { k: 'rt_multiplier',              label: 'rt multiplier',         color: '#f59e0b' },
    { k: 'error_rate_boost',           label: 'error-rate boost',      color: '#84cc16' },
    { k: 'error_volatility',           label: 'error volatility',      color: '#ea580c' },
  ];

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function ctxFor(id) {
    const c = document.getElementById(id);
    if (!c) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = c.getBoundingClientRect();
    const w = rect.width || c.clientWidth;
    const h = parseInt(c.getAttribute('height')) || 260;
    c.width = w * dpr; c.height = h * dpr;
    c.style.width = w + 'px'; c.style.height = h + 'px';
    const x = c.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { x, w, h };
  }

  async function loadRuns() {
    try {
      const res = await fetch('/api/drift?limit=40', { cache: 'no-store' });
      if (res.ok) {
        const j = await res.json();
        if (j && Array.isArray(j.runs) && j.runs.length) {
          markSource('live · /api/drift');
          return j.runs;
        }
      }
    } catch (e) { /* static build — fall through */ }
    markSource('snapshot · data.js');
    return D.selfheal || [];
  }

  function markSource(txt) {
    const el = document.getElementById('driftSource');
    if (el) el.textContent = txt;
  }

  let RUNS = [];

  function drawDriftChart() {
    const ref = ctxFor('driftChart');
    if (!ref || !RUNS.length) return;
    const { x, w, h } = ref;
    const padL = 44, padR = 14, padT = 22, padB = 30;
    const N = RUNS.length;
    let maxKS = KS_GUIDE * 1.4;
    RUNS.forEach(r => SIGNALS.forEach(s => {
      const v = r.signals && r.signals[s.k] && r.signals[s.k].ks;
      if (v != null) maxKS = Math.max(maxKS, v);
    }));

    const sx = i => padL + (N === 1 ? 0.5 : i / (N - 1)) * (w - padL - padR);
    const sy = v => padT + (1 - clamp(v / maxKS, 0, 1)) * (h - padT - padB);

    x.clearRect(0, 0, w, h);
    x.strokeStyle = 'rgba(220,140,100,0.08)'; x.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const v = maxKS * g / 4, y = sy(v);
      x.beginPath(); x.moveTo(padL, y); x.lineTo(w - padR, y); x.stroke();
      x.fillStyle = '#7a5b3e'; x.font = '10px JetBrains Mono'; x.textAlign = 'right';
      x.fillText(v.toFixed(2), padL - 6, y + 3);
    }
    // effect-size guide
    x.strokeStyle = 'rgba(239,68,68,0.45)'; x.setLineDash([5, 4]);
    x.beginPath(); x.moveTo(padL, sy(KS_GUIDE)); x.lineTo(w - padR, sy(KS_GUIDE)); x.stroke(); x.setLineDash([]);
    x.fillStyle = '#ef4444'; x.font = '10px JetBrains Mono'; x.textAlign = 'left';
    x.fillText('drift guide · KS ' + KS_GUIDE.toFixed(2), padL + 6, sy(KS_GUIDE) - 5);

    SIGNALS.forEach(s => {
      x.strokeStyle = s.color; x.lineWidth = 2; x.beginPath();
      let started = false;
      RUNS.forEach((r, i) => {
        const v = r.signals && r.signals[s.k] && r.signals[s.k].ks;
        if (v == null) return;
        started ? x.lineTo(sx(i), sy(v)) : x.moveTo(sx(i), sy(v));
        started = true;
      });
      x.stroke();
      // dot on drifted points
      RUNS.forEach((r, i) => {
        const cell = r.signals && r.signals[s.k];
        if (cell && cell.drifted) {
          x.fillStyle = s.color; x.beginPath(); x.arc(sx(i), sy(cell.ks), 3.5, 0, TAU); x.fill();
        }
      });
    });

    x.fillStyle = '#7a5b3e'; x.font = '10px JetBrains Mono'; x.textAlign = 'center';
    x.fillText('run 1', sx(0), h - 9);
    x.fillText('latest', sx(N - 1), h - 9);
  }

  function buildLegend() {
    const el = document.getElementById('driftLegend');
    if (el) el.innerHTML = SIGNALS.map(s =>
      `<span class="dl"><span class="sw" style="background:${s.color}"></span>${s.label}</span>`).join('');
  }

  function buildGrid() {
    const root = document.getElementById('signalGrid');
    if (!root || !RUNS.length) return;
    const latest = RUNS[RUNS.length - 1];
    root.innerHTML = SIGNALS.map(s => {
      const cell = (latest.signals && latest.signals[s.k]) || {};
      const ks = cell.ks != null ? cell.ks : 0;
      const drifted = !!cell.drifted;
      const ratio = clamp(ks / (KS_GUIDE * 1.6), 0, 1);
      return `<div class="sg-card ${drifted ? 'drift' : ''}">
        <div class="sg-top"><span class="sg-dot" style="background:${s.color}"></span><span class="sg-name">${s.label}</span>
          <span class="sg-badge ${drifted ? 'bad' : 'ok'}">${drifted ? 'DRIFT' : 'stable'}</span></div>
        <div class="sg-ks">KS ${ks.toFixed(4)}</div>
        <div class="sg-track"><span style="width:${(ratio * 100).toFixed(0)}%;background:${s.color}"></span></div>
        <div class="sg-p">p-value ${cell.p != null ? Number(cell.p).toFixed(3) : '—'}</div>
      </div>`;
    }).join('');
  }

  function buildSummary() {
    const root = document.getElementById('driftSummary');
    if (!root || !RUNS.length) return;
    const latest = RUNS[RUNS.length - 1];
    const driftingNow = Object.values(latest.signals || {}).filter(v => v.drifted).length;
    const everDrifted = RUNS.some(r => r.drift_detected);
    const fr = ((latest.failure_rate || 0) * 100).toFixed(2);
    const updated = RUNS.filter(r => r.model_updated).length;
    root.innerHTML = `
      <div class="ds-tile"><b>${RUNS.length}</b><span>self-healing runs</span></div>
      <div class="ds-tile ${driftingNow ? 'warn' : 'ok'}"><b>${driftingNow}/${SIGNALS.length}</b><span>signals drifting now</span></div>
      <div class="ds-tile ${everDrifted ? 'warn' : 'ok'}"><b>${everDrifted ? 'yes' : 'no'}</b><span>drift ever triggered</span></div>
      <div class="ds-tile"><b>${updated}</b><span>auto model updates</span></div>`;
  }

  async function init() {
    buildLegend();
    RUNS = await loadRuns();
    drawDriftChart();
    buildGrid();
    buildSummary();
    let t;
    window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(drawDriftChart, 160); });
  }

  window.LEO_DRIFT = { init };
})();
