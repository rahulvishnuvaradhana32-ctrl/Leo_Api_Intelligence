/* ============================================================
 *  LEO · site.js — clean chrome + restrained interactions for the
 *  "Institutional Trust" rebuild. Scroll-reveal, number count-up,
 *  one subtle hero sparkline. No particles / glitch / scanlines.
 * ============================================================ */
(function () {
  'use strict';

  // ── brand lion: use the supplied image file (web/lion.png),
  //    fall back to the bundled SVG until the file is added ──
  const LION = `<img class="lion-img" src="/lion.png" alt="LEO" onerror="this.onerror=null;this.src='/lion.svg'">`;
  const D = window.LEO_DATA || {};

  // ── scroll reveal ──
  function initReveal() {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach(e => e.classList.add('in')); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { threshold: 0.18, rootMargin: '0px 0px -8% 0px' });
    els.forEach(e => io.observe(e));
  }

  // ── number count-up (data-count="3.10" data-suffix="M" data-prefix="$" data-dec="2") ──
  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    const dec = parseInt(el.dataset.dec || '0', 10);
    const pre = el.dataset.prefix || '';
    const suf = el.dataset.suffix || '';
    const dur = 1100; let t0 = null;
    function frame(ts) {
      if (t0 == null) t0 = ts;
      const p = Math.min(1, (ts - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      const v = (target * eased).toFixed(dec);
      el.textContent = pre + Number(v).toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suf;
      if (p < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  function initCounts() {
    const els = document.querySelectorAll('[data-count]');
    if (!('IntersectionObserver' in window)) { els.forEach(countUp); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { countUp(e.target); io.unobserve(e.target); } });
    }, { threshold: 0.6 });
    els.forEach(e => io.observe(e));
  }

  // ── LEO live-forecast card: an ECharts HUD that shows a genuine FORWARD
  //    forecast — x-axis is minutes ahead (now → 15 min), the line is the
  //    predicted failure probability over that horizon, and "time to failure"
  //    is read straight off the curve's threshold crossing, so the number can
  //    never disagree with the graph. The scenario loops the lead time from
  //    off-chart (calm) → 15 → 5 → 1 min (imminent) → back. ──
  const STEP_MS = 90;          // morph refresh
  const CYCLE = 17000;         // full scenario loop length
  const HORIZON = 15;          // minutes shown on the x-axis
  const XN = 49;               // forecast resolution
  const THRESH = 0.55;         // high-risk / auto-reroute threshold
  let _chart = null, _lead = 26, _t0 = null, _liveTimer = null, _els = null;

  function cacheEls() {
    _els = {
      chip: document.getElementById('liveChip'),
      conf: document.getElementById('liveConf'),
      action: document.getElementById('liveAction'),
      lat: document.getElementById('liveLat'),
      ttf: document.getElementById('liveTtf'),
    };
  }

  // scenario: lead time (minutes until failure) over the loop. 26 = off-chart.
  function targetLead(p) {
    if (p < 0.28) return 26;                                  // calm
    if (p < 0.52) return 15 - (p - 0.28) / 0.24 * 10;         // degrading 15→5
    if (p < 0.68) return 5 - (p - 0.52) / 0.16 * 4;           // worsening 5→1
    if (p < 0.80) return 1;                                   // imminent
    return 26;                                                // recovered
  }

  // predicted failure-probability curve: rises to a sigmoid crossing 0.55 at x≈lead
  function buildCurve(lead) {
    const pts = [];
    for (let i = 0; i < XN; i++) {
      const x = i / (XN - 1) * HORIZON;
      const risk = 0.07 + 0.87 / (1 + Math.exp(-(x - lead) * 0.9));
      pts.push([+x.toFixed(3), +Math.max(0.03, Math.min(0.98, risk)).toFixed(4)]);
    }
    return pts;
  }

  // the x (minutes) where the curve first crosses the threshold — null if never
  function crossOf(curve) {
    for (let i = 1; i < curve.length; i++) {
      if (curve[i][1] >= THRESH) {
        const [x0, y0] = curve[i - 1], [x1, y1] = curve[i];
        const t = (THRESH - y0) / (y1 - y0 || 1);
        return x0 + t * (x1 - x0);
      }
    }
    return null;
  }

  function setText(el, v) { if (el && el.textContent !== v) el.textContent = v; }
  function setChip(el, txt, st) {
    if (!el) return;
    setText(el.querySelector('.chip-tx') || el, txt);
    if (el.dataset.st !== st) { el.dataset.st = st; el.className = 'pc-chip ' + st; }
  }

  function palette(st) {
    if (st === 'crit') return { end: '#F87171', glow: 'rgba(248,113,113,.95)', node: '#F87171', area: 'rgba(248,113,113,.30)', ripple: 4 };
    if (st === 'warn') return { end: '#C084FC', glow: 'rgba(192,132,252,.85)', node: '#FBBF24', area: 'rgba(139,92,246,.26)', ripple: 3 };
    return { end: '#8B5CF6', glow: 'rgba(77,141,255,.85)', node: '#34D399', area: 'rgba(77,141,255,.26)', ripple: 2.4 };
  }

  // everything below is DERIVED from the same crossX → readout & graph agree
  function applyState(curve, crossX) {
    const peak = curve[curve.length - 1][1];
    const ttfMin = crossX == null ? null : Math.max(1, Math.round(crossX));
    const st = ttfMin == null ? 'ok' : ttfMin <= 5 ? 'crit' : 'warn';
    const pal = palette(st);

    if (_els) {
      if (st === 'crit') {
        setChip(_els.chip, 'outage imminent · auto-rerouting', 'crit');
        setText(_els.action, 'auto-reroute'); if (_els.action) _els.action.className = 'v crit';
      } else if (st === 'warn') {
        setChip(_els.chip, 'degradation forming · pre-warming backup', 'warn');
        setText(_els.action, 'pre-warm'); if (_els.action) _els.action.className = 'v warn';
      } else {
        setChip(_els.chip, 'all clear · auto-failover armed', 'ok');
        setText(_els.action, 'standby'); if (_els.action) _els.action.className = 'v pos';
      }
      setText(_els.ttf, ttfMin == null ? 'monitored' : ttfMin + ' min');
      const conf = Math.min(98.5, 85 + peak * 13.5 + Math.sin(performance.now() / 900) * 0.3);
      setText(_els.conf, conf.toFixed(1) + '%');
      setText(_els.lat, Math.round(268 + (1 - (crossX == null ? 1 : crossX / HORIZON)) * 22 + Math.random() * 3) + ' ms');
    }
    return { st, pal, ttfMin };
  }

  function render(curve, crossX, animate) {
    const { pal, ttfMin } = applyState(curve, crossX);
    const markData = [{ yAxis: THRESH }];
    if (crossX != null) {
      markData.push({
        xAxis: +crossX.toFixed(3),
        label: { show: true, formatter: ttfMin + 'm', position: 'insideEndTop', color: pal.node, fontFamily: 'IBM Plex Mono', fontSize: 9 },
        lineStyle: { color: pal.glow },
      });
    }
    _chart.setOption({
      series: [
        {
          data: curve,
          lineStyle: { width: 2.6, color: { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: '#4D8DFF' }, { offset: 1, color: pal.end }] }, shadowColor: pal.glow, shadowBlur: 16 },
          areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: pal.area }, { offset: 1, color: 'rgba(139,92,246,0)' }] } },
          markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: 'rgba(248,113,113,.4)', type: 'dashed', width: 1 }, data: markData },
        },
        {
          data: crossX != null ? [[+crossX.toFixed(3), THRESH]] : [],
          itemStyle: { color: pal.node, shadowColor: pal.glow, shadowBlur: 12 },
          rippleEffect: { brushType: 'stroke', scale: pal.ripple, period: 2.6 },
        },
      ],
      animation: animate,
    });
  }

  function baseOption() {
    return {
      animation: true, animationDurationUpdate: STEP_MS, animationEasingUpdate: 'linear',
      backgroundColor: 'transparent',
      grid: { left: 6, right: 12, top: 12, bottom: 20 },
      xAxis: {
        type: 'value', min: 0, max: HORIZON,
        axisLine: { show: false }, axisTick: { show: false },
        splitLine: { show: true, lineStyle: { color: 'rgba(124,148,210,0.06)' } },
        axisLabel: { show: true, interval: 0, color: '#5b678a', fontFamily: 'IBM Plex Mono', fontSize: 9,
          formatter: v => (v === 0 ? 'now' : (v === 5 || v === 10 || v === 15) ? v + 'm' : ''), },
      },
      yAxis: { type: 'value', show: false, min: 0, max: 1 },
      series: [
        { type: 'line', smooth: 0.45, symbol: 'none', z: 3, data: [] },
        { type: 'effectScatter', symbolSize: 9, showEffectOn: 'render', z: 5, data: [] },
      ],
    };
  }

  function tick() {
    const p = ((performance.now() - _t0) % CYCLE) / CYCLE;
    _lead += (targetLead(p) - _lead) * 0.2;        // smooth morph
    const curve = buildCurve(_lead);
    render(curve, crossOf(curve), true);
  }

  function drawSpark() {
    const el = document.getElementById('heroSpark');
    if (!el || !window.echarts) return;
    if (!_chart) { _chart = echarts.init(el, null, { renderer: 'canvas' }); cacheEls(); }
    _chart.setOption(baseOption(), true);

    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {                                   // static snapshot: failure ~5 min out
      const curve = buildCurve(5);
      render(curve, crossOf(curve), false);
      return;
    }
    _lead = 26;
    if (_t0 == null) _t0 = performance.now();
    if (_liveTimer) clearInterval(_liveTimer);
    _liveTimer = setInterval(tick, STEP_MS);
  }

  function resizeSpark() { if (_chart) _chart.resize(); }

  // ── shared chrome: inject nav + footer on every page ──
  const NAV = [
    { href: '/architecture.html', label: 'Platform' },
    { href: '/reliability.html',  label: 'Reliability' },
    { href: '/predict.html',      label: 'Live demo' },
    { href: '/modelcard.html',    label: 'Model card' },
    { href: '/cite.html',         label: 'Docs' },
  ];

  function currentPath() {
    let p = location.pathname;
    if (p.endsWith('/index.html') || p.endsWith('/home.html')) p = '/';
    return p || '/';
  }

  function buildNav() {
    const cur = currentPath();
    const links = NAV.map(n => {
      const active = (cur === n.href) ? ' aria-current="page"' : '';
      return `<a href="${n.href}"${active}>${n.label}</a>`;
    }).join('');
    return `<div class="wrap nav-inner">
      <a class="brand" href="/"><span class="mark">${LION}</span><span class="brand-tx">LEO<span class="brand-api">.api</span></span></a>
      <nav class="nav-links" aria-label="Primary">${links}</nav>
      <div class="nav-cta">
        <span class="nav-status"><span class="dot"></span> Alive</span>
        <a class="btn btn-primary pulse-cta" href="/predict.html">Book a demo</a>
      </div>
    </div>`;
  }

  function buildFooter() {
    const yr = new Date().getFullYear();
    return `<div class="wrap">
      <div class="foot-grid">
        <div class="foot-brand">
          <a class="brand" href="/"><span class="mark">${LION}</span><span class="brand-tx">LEO<span class="brand-api">.api</span></span></a>
          <p>We turn ‘the API went down’ into ‘the API was about to.’</p>
        </div>
        <div class="foot-col"><h4>Platform</h4>
          <a href="/architecture.html">Architecture</a><a href="/forecasting.html">Forecasting</a>
          <a href="/predict.html">Live demo</a><a href="/agent.html">Proactive agent</a></div>
        <div class="foot-col"><h4>Reliability</h4>
          <a href="/reliability.html">SLO &amp; error budget</a><a href="/confidence.html">Confidence bands</a>
          <a href="/drift.html">Drift monitor</a><a href="/pipeline.html">Self-healing pipeline</a></div>
        <div class="foot-col"><h4>Resources</h4>
          <a href="/modelcard.html">Model card</a><a href="/dataset.html">Dataset</a>
          <a href="/cite.html">Documentation</a><a href="/results.html">Benchmarks</a></div>
      </div>
      <div class="foot-base">
        <span>© ${yr} LEO · Predictive API Intelligence</span>
        <span class="mono">made for fintech reliability teams</span>
      </div>
    </div>`;
  }

  function injectChrome() {
    let nav = document.querySelector('header.nav');
    if (!nav) { nav = document.createElement('header'); nav.className = 'nav'; document.body.insertBefore(nav, document.body.firstChild); }
    nav.innerHTML = buildNav();

    let foot = document.querySelector('footer.foot');
    if (!foot) { foot = document.createElement('footer'); foot.className = 'foot'; document.body.appendChild(foot); }
    foot.innerHTML = buildFooter();

    const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  // ── render legacy .leo-says[data-msg] bubbles as .explainer blocks ──
  function initExplainers() {
    document.querySelectorAll('.leo-says[data-msg]').forEach(el => {
      const html = String(el.dataset.msg || '').replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
      el.classList.add('explainer');
      el.innerHTML = `<div><div class="who">LEO explains</div><div class="msg">${html}</div></div>`;
    });
  }

  function boot() {
    injectChrome();
    initExplainers();
    initReveal();
    initCounts();
    drawSpark();
    let t; window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(resizeSpark, 160); });
  }
  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
