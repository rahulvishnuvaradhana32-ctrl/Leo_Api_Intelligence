/* ============================================================
 *  LEO · site.js — clean chrome + restrained interactions for the
 *  "Institutional Trust" rebuild. Scroll-reveal, number count-up,
 *  one subtle hero sparkline. No particles / glitch / scanlines.
 * ============================================================ */
(function () {
  'use strict';
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

  // ── hero sparkline: calm risk curve with one averted spike,
  //    drawn progressively on load + a softly pulsing marker ──
  let _sparkPts = null, _sparkGeom = null, _sparkRAF = null;
  function buildSpark() {
    const c = document.getElementById('heroSpark');
    if (!c) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = c.getBoundingClientRect();
    const w = rect.width || 360, h = parseInt(c.getAttribute('height')) || 90;
    c.width = w * dpr; c.height = h * dpr;
    c.style.width = w + 'px'; c.style.height = h + 'px';
    const x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0);
    const N = 64, pts = [];
    let seed = 7;
    const rnd = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
    for (let i = 0; i < N; i++) {
      const bump = Math.exp(-Math.pow((i - 40) / 7, 2)) * 0.7;
      pts.push(0.18 + bump + (rnd() - 0.5) * 0.05);
    }
    _sparkPts = pts; _sparkGeom = { x, w, h, N };
    return _sparkGeom;
  }
  function paintSpark(progress, pulse) {
    if (!_sparkGeom) return;
    const { x, w, h, N } = _sparkGeom, pts = _sparkPts;
    const sx = i => (i / (N - 1)) * w;
    const sy = v => h - 6 - Math.max(0, Math.min(1, v)) * (h - 12);
    const drawN = Math.max(2, Math.floor(N * progress));
    x.clearRect(0, 0, w, h);
    // area
    const g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, 'rgba(99,102,241,.20)');
    g.addColorStop(1, 'rgba(139,92,246,0)');
    x.beginPath(); x.moveTo(0, h);
    for (let i = 0; i < drawN; i++) x.lineTo(sx(i), sy(pts[i]));
    x.lineTo(sx(drawN - 1), h); x.closePath(); x.fillStyle = g; x.fill();
    // gradient line
    const lg = x.createLinearGradient(0, 0, w, 0);
    lg.addColorStop(0, '#6366F1'); lg.addColorStop(1, '#8B5CF6');
    x.beginPath();
    for (let i = 0; i < drawN; i++) i ? x.lineTo(sx(i), sy(pts[i])) : x.moveTo(sx(i), sy(pts[i]));
    x.strokeStyle = lg; x.lineWidth = 2.2; x.lineJoin = 'round'; x.stroke();
    // pulsing marker at the peak once revealed
    const pk = 40;
    if (drawN > pk) {
      const r = 3.4 + (pulse || 0) * 2.2;
      x.beginPath(); x.arc(sx(pk), sy(pts[pk]), r + 3, 0, Math.PI * 2);
      x.fillStyle = 'rgba(124,58,237,' + (0.18 - (pulse || 0) * 0.12) + ')'; x.fill();
      x.beginPath(); x.arc(sx(pk), sy(pts[pk]), 3.4, 0, Math.PI * 2);
      x.fillStyle = '#4F46E5'; x.fill();
      x.strokeStyle = '#fff'; x.lineWidth = 2; x.stroke();
    }
  }
  function drawSpark() {
    if (!buildSpark()) return;
    if (_sparkRAF) cancelAnimationFrame(_sparkRAF);
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) { paintSpark(1, 0); return; }
    let t0 = null;
    function frame(ts) {
      if (t0 == null) t0 = ts;
      const el = ts - t0;
      const draw = Math.min(1, el / 1200);                 // 1.2s reveal
      const pulse = (Math.sin(el / 700) + 1) / 2;           // gentle breathing
      paintSpark(draw, draw >= 1 ? pulse : 0);
      _sparkRAF = requestAnimationFrame(frame);
    }
    _sparkRAF = requestAnimationFrame(frame);
  }

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
      <a class="brand" href="/"><span class="mark">L</span> LEO</a>
      <nav class="nav-links" aria-label="Primary">${links}</nav>
      <div class="nav-cta">
        <span class="nav-status"><span class="dot"></span> all systems operational</span>
        <a class="btn btn-primary" href="/predict.html">Book a demo</a>
      </div>
    </div>`;
  }

  function buildFooter() {
    const yr = new Date().getFullYear();
    return `<div class="wrap">
      <div class="foot-grid">
        <div class="foot-brand">
          <a class="brand" href="/"><span class="mark">L</span> LEO</a>
          <p>Predictive reliability for banking &amp; fintech APIs. Call the alarm before the failure lands.</p>
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
        <span class="mono">conformal-calibrated · α = 0.10</span>
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
    let t; window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(drawSpark, 160); });
  }
  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
