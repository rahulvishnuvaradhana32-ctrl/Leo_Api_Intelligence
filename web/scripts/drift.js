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
    { k: 'error_rate_rolling',         label: 'error rate (rolling)',  color: '#4D8DFF' },
    { k: 'response_time_rolling_mean', label: 'response time (mean)',  color: '#8B5CF6' },
    { k: 'rt_multiplier',              label: 'rt multiplier',         color: '#38BDF8' },
    { k: 'error_rate_boost',           label: 'error-rate boost',      color: '#34D399' },
    { k: 'error_volatility',           label: 'error volatility',      color: '#A78BFA' },
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
  let SELECTED = null;   // signal key to isolate in the trend chart (null = all)

  function drawDriftChart() {
    const el = document.getElementById('driftChart');
    if (!el || !window.echarts || !RUNS.length) return;
    const inst = echarts.getInstanceByDom(el) || echarts.init(el, null, { renderer: 'canvas' });
    const N = RUNS.length;
    const idx = RUNS.map((_, i) => i + 1);
    const shown = SELECTED ? SIGNALS.filter(s => s.k === SELECTED) : SIGNALS;

    const series = shown.map(s => ({
      name: s.label, type: 'line', smooth: true, symbol: 'circle', symbolSize: 5,
      data: RUNS.map(r => { const c = r.signals && r.signals[s.k]; return (c && c.ks != null) ? +c.ks.toFixed(4) : null; }),
      color: s.color, itemStyle: { color: s.color },
      lineStyle: { width: 2, color: s.color, shadowColor: s.color, shadowBlur: 8 },
      // glow the points that actually flagged drift
      markPoint: {
        symbol: 'circle', symbolSize: 9,
        data: RUNS.map((r, i) => { const c = r.signals && r.signals[s.k]; return (c && c.drifted) ? { coord: [i + 1, +c.ks.toFixed(4)] } : null; }).filter(Boolean),
        itemStyle: { color: s.color, shadowColor: s.color, shadowBlur: 14 }, label: { show: false },
      },
    }));

    inst.setOption({
      backgroundColor: 'transparent', animationDuration: 900, animationEasing: 'cubicOut',
      grid: { left: 46, right: 16, top: 16, bottom: 26 },
      tooltip: {
        trigger: 'axis', backgroundColor: 'rgba(14,20,36,.96)', borderColor: 'rgba(124,148,210,.25)', borderWidth: 1,
        textStyle: { color: '#EAF1FF', fontFamily: 'IBM Plex Mono', fontSize: 11 },
      },
      xAxis: {
        type: 'category', data: idx, boundaryGap: false, axisTick: { show: false },
        axisLine: { lineStyle: { color: 'rgba(124,148,210,.2)' } },
        axisLabel: { color: '#6F7C9C', fontFamily: 'IBM Plex Mono', fontSize: 10, interval: i => (i === 0 || i === N - 1), formatter: v => (v === 1 ? 'run 1' : v === N ? 'latest' : '') },
      },
      yAxis: {
        type: 'value', min: 0, splitLine: { lineStyle: { color: 'rgba(124,148,210,.07)' } },
        axisLabel: { color: '#6F7C9C', fontFamily: 'IBM Plex Mono', fontSize: 10, formatter: v => v.toFixed(2) },
      },
      series: series.concat([{
        type: 'line', data: [],
        markLine: { silent: true, symbol: 'none', data: [{ yAxis: KS_GUIDE }],
          label: { show: true, formatter: 'drift trigger · KS ' + KS_GUIDE.toFixed(2), position: 'insideStartTop', color: '#F87171', fontFamily: 'IBM Plex Mono', fontSize: 9 },
          lineStyle: { color: 'rgba(248,113,113,.5)', type: 'dashed', width: 1 } },
      }]),
    }, true);
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
      const active = SELECTED === s.k ? ' active' : '';
      return `<div class="sg-card${drifted ? ' drift' : ''}${active}" data-sig="${s.k}" role="button" tabindex="0" title="Isolate this signal in the trend">
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

    // click a signal card → isolate its line in the trend (click again to reset)
    const grid = document.getElementById('signalGrid');
    if (grid) {
      const toggle = el => {
        const card = el.closest('[data-sig]'); if (!card) return;
        SELECTED = (SELECTED === card.dataset.sig) ? null : card.dataset.sig;
        grid.querySelectorAll('.sg-card').forEach(c => c.classList.toggle('active', c.dataset.sig === SELECTED));
        drawDriftChart();
      };
      grid.addEventListener('click', e => toggle(e.target));
      grid.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(e.target); } });
    }

    let t;
    window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(drawDriftChart, 160); });
  }

  window.LEO_DRIFT = { init };
})();
