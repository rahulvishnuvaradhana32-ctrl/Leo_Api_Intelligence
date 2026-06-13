/* ============================================================
 * LEO · chrome.js
 *   - Injects the animated crimson-lion logo
 *   - Builds shared nav + footer on every page
 *   - Installs ambient layers (particles canvas, cursor glow, grain)
 *   - Marks the active nav item via aria-current="page"
 * ============================================================ */

(function () {
  'use strict';

  // ──────────────────────────────────────────────────────────────
  //  LION LOGO  ·  Brand identity bumper
  //  Style: refined heraldic anime, modelled on the motion language
  //  of production logos (Stripe / Linear / anime.js demos).
  //  Plays a 1.2s intro reveal on first paint, then enters a calm
  //  idle micro-motion. Hover triggers a snappy "roar" with sparks.
  // ──────────────────────────────────────────────────────────────
  const LION_SVG = `
  <svg viewBox="0 0 100 100" class="lion-svg" aria-hidden="true">
    <defs>
      <!-- Radial soul aura -->
      <radialGradient id="lionAura" cx="50%" cy="50%" r="50%">
        <stop offset="0%"  stop-color="#fde047" stop-opacity="0.55"/>
        <stop offset="55%" stop-color="#dc2626" stop-opacity="0.22"/>
        <stop offset="100%" stop-color="#7c1d1d" stop-opacity="0"/>
      </radialGradient>

      <!-- Mane gradients — each layer slightly different temperature -->
      <radialGradient id="maneOuter" cx="50%" cy="42%" r="62%">
        <stop offset="0%"   stop-color="#dc2626"/>
        <stop offset="65%"  stop-color="#991b1b"/>
        <stop offset="100%" stop-color="#4a0e0e"/>
      </radialGradient>
      <radialGradient id="maneMid" cx="50%" cy="40%" r="55%">
        <stop offset="0%"   stop-color="#fb923c"/>
        <stop offset="60%"  stop-color="#dc2626"/>
        <stop offset="100%" stop-color="#991b1b"/>
      </radialGradient>
      <radialGradient id="maneInner" cx="50%" cy="38%" r="48%">
        <stop offset="0%"   stop-color="#fde047"/>
        <stop offset="55%"  stop-color="#f59e0b"/>
        <stop offset="100%" stop-color="#c2410c"/>
      </radialGradient>

      <!-- Face — warm radial with shadow at jaw -->
      <radialGradient id="lionFace" cx="50%" cy="38%" r="60%">
        <stop offset="0%"  stop-color="#fde047"/>
        <stop offset="45%" stop-color="#f59e0b"/>
        <stop offset="100%" stop-color="#9a3412"/>
      </radialGradient>

      <!-- Eyes -->
      <linearGradient id="lionEye" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#fef3c7"/>
        <stop offset="100%" stop-color="#f59e0b"/>
      </linearGradient>

      <!-- Soft inner-glow filter on the eyes -->
      <filter id="lionGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="0.8" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>

      <!-- Strong drop-shadow used during the hover ROAR -->
      <filter id="roarGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2.4" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>

      <!-- Eye-flare sweep gradient -->
      <linearGradient id="eyeFlare" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%"  stop-color="#fff" stop-opacity="0"/>
        <stop offset="50%" stop-color="#fff" stop-opacity="1"/>
        <stop offset="100%" stop-color="#fff" stop-opacity="0"/>
      </linearGradient>

      <!-- Lens flare for the roar -->
      <radialGradient id="flare" cx="50%" cy="50%" r="50%">
        <stop offset="0%"  stop-color="#fff" stop-opacity="0.9"/>
        <stop offset="40%" stop-color="#fde047" stop-opacity="0.6"/>
        <stop offset="100%" stop-color="#fde047" stop-opacity="0"/>
      </radialGradient>
    </defs>

    <!-- AURA -->
    <circle class="lion-aura" cx="50" cy="50" r="48" fill="url(#lionAura)"/>

    <!-- ROAR FLARE — only visible during hover -->
    <circle class="lion-flare" cx="50" cy="50" r="55" fill="url(#flare)" opacity="0"/>

    <!-- ORBIT RING — thin gold ring rotating slowly -->
    <circle class="lion-orbit" cx="50" cy="50" r="45"
      fill="none" stroke="rgba(251,191,36,0.35)" stroke-width="0.4"
      stroke-dasharray="2 5"/>

    <!-- MANE OUTER  ·  big flame petals -->
    <g class="mane-outer-g">
      <path class="mane-stroke" fill="url(#maneOuter)" stroke="#4a0e0e" stroke-width="0.5" stroke-linejoin="round" d="
        M50 3
        C 46 6 44 11 41 8
        C 36 3  29 6  27 13
        C 21 10 14 14 13 23
        C  6 23  3 32  8 38
        C  2 43  2 52  9 58
        C  4 64  6 73 15 75
        C 15 84 24 87 31 83
        C 33 91 43 93 48 87
        C 50 93 57 91 57 91
        C 64 93 70 91 73 83
        C 80 87 89 84 89 75
        C 95 73 96 64 92 58
        C 98 52 98 43 92 38
        C 96 32 94 23 87 23
        C 87 14 79 10 73 13
        C 70 6  64 3  59 8
        C 56 11 54 6  50 3 Z"/>
    </g>

    <!-- MANE MID  ·  inset orange layer -->
    <g class="mane-mid-g">
      <path fill="url(#maneMid)" d="
        M50 12
        C 45 14 42 19 36 16
        C 29 14 23 20 24 28
        C 16 30 14 38 19 44
        C 13 50 16 60 23 62
        C 23 72 33 74 39 70
        C 42 77 50 78 50 78
        C 58 78 62 76 62 70
        C 69 74 78 72 78 62
        C 85 60 87 50 81 44
        C 87 38 85 30 77 28
        C 78 20 71 14 64 16
        C 58 19 55 14 50 12 Z"/>
    </g>

    <!-- MANE INNER  ·  bright gold halo -->
    <g class="mane-inner-g">
      <path fill="url(#maneInner)" d="
        M50 18
        C 42 20 36 22 32 28
        C 26 32 27 41 31 46
        C 26 52 29 60 36 62
        C 38 70 46 71 50 68
        C 54 71 62 70 64 62
        C 71 60 74 52 69 46
        C 73 41 74 32 68 28
        C 64 22 58 20 50 18 Z"/>
    </g>

    <!-- TOP-CENTER FLAME TUFT — anime signature point -->
    <path class="lion-tuft" fill="url(#maneInner)" d="
      M50 4
      C 47 9 47 14 46 18
      C 49 16 51 16 54 18
      C 53 14 53 9 50 4 Z"/>

    <!-- EARS -->
    <g class="lion-ears">
      <path d="M30 32 L34 22 L40 32 Z" fill="#dc2626" stroke="#7c1d1d" stroke-width="0.55" stroke-linejoin="round"/>
      <path d="M70 32 L66 22 L60 32 Z" fill="#dc2626" stroke="#7c1d1d" stroke-width="0.55" stroke-linejoin="round"/>
      <path d="M33 30 L34 25 L37 30 Z" fill="#fbbf24" opacity="0.9"/>
      <path d="M67 30 L66 25 L63 30 Z" fill="#fbbf24" opacity="0.9"/>
    </g>

    <!-- FACE SHAPE — refined heraldic with subtle cheek planes -->
    <path class="lion-face" fill="url(#lionFace)" stroke="#7c2d12" stroke-width="0.55" stroke-linejoin="round" d="
      M50 30
      C 42 30 36 33 34 39
      C 32 45 33 52 38 56
      C 36 60 38 65 44 66
      C 44 70 47 72 50 70
      C 53 72 56 70 56 66
      C 62 65 64 60 62 56
      C 67 52 68 45 66 39
      C 64 33 58 30 50 30 Z"/>

    <!-- Cheek planes (subtle shadow / volume hint) -->
    <path d="M37 50 Q 41 56 41 60 L 38 57 Q 36 53 37 50 Z" fill="#9a3412" opacity="0.35"/>
    <path d="M63 50 Q 59 56 59 60 L 62 57 Q 64 53 63 50 Z" fill="#9a3412" opacity="0.35"/>

    <!-- Forehead V — bold anime frown -->
    <path fill="#7c2d12" d="M44 36 L50 43 L56 36 L56 39.5 L50 45.5 L44 39.5 Z"/>

    <!-- Bridge highlight between brows -->
    <path d="M48.5 40 Q 50 42 51.5 40 L 51.5 49 Q 50 50 48.5 49 Z" fill="#fde047" opacity="0.45"/>

    <!-- White muzzle -->
    <path class="lion-muzzle" fill="#fff5cf" stroke="#7c2d12" stroke-width="0.4" stroke-linejoin="round" d="
      M43 59
      Q 50 57 57 59
      Q 57 65 50 68
      Q 43 65 43 59 Z"/>

    <!-- Nose -->
    <path d="M50 53.5 L46 60 L54 60 Z" fill="#1a0405" stroke="#fde047" stroke-width="0.3" stroke-linejoin="round"/>
    <path d="M48.5 56.5 Q 50 57.5 51.5 56.5" fill="#fff" opacity="0.15"/>

    <!-- Mouth -->
    <path d="M50 60 L50 63.5" stroke="#1a0405" stroke-width="0.9" stroke-linecap="round"/>
    <path d="M50 63.5 Q47 66 45 64" stroke="#1a0405" stroke-width="0.9" stroke-linecap="round" fill="none"/>
    <path d="M50 63.5 Q53 66 55 64" stroke="#1a0405" stroke-width="0.9" stroke-linecap="round" fill="none"/>

    <!-- Cheek flame tufts -->
    <path class="cheek-tuft" d="M34 57 L29 61 L33 61 L31 65 L36 61 Z" fill="#f59e0b"/>
    <path class="cheek-tuft" d="M66 57 L71 61 L67 61 L69 65 L64 61 Z" fill="#f59e0b"/>

    <!-- EYES — sharp almond, vertical slit pupils, fierce -->
    <g class="lion-eyes" filter="url(#lionGlow)">
      <path class="eye-l" fill="url(#lionEye)" stroke="#5c1f0a" stroke-width="0.45" stroke-linejoin="round" d="
        M37 47 Q 41 43 46 46 Q 45 51 39 50 Z"/>
      <path class="eye-r" fill="url(#lionEye)" stroke="#5c1f0a" stroke-width="0.45" stroke-linejoin="round" d="
        M63 47 Q 59 43 54 46 Q 55 51 61 50 Z"/>
      <!-- slit pupils -->
      <path d="M41.5 45 Q 42.2 47.5 41.5 49.6 Q 40.8 47.5 41.5 45 Z" fill="#1a0405"/>
      <path d="M58.5 45 Q 59.2 47.5 58.5 49.6 Q 57.8 47.5 58.5 45 Z" fill="#1a0405"/>
      <!-- highlight catch -->
      <circle cx="40.8" cy="46" r="0.55" fill="#ffffff"/>
      <circle cx="57.8" cy="46" r="0.55" fill="#ffffff"/>
    </g>

    <!-- EYE-FLARE  ·  sweeping light bars across both eyes on hover/idle -->
    <g class="eye-flare-g">
      <rect class="eye-flare" x="36" y="44.5" width="11" height="6" fill="url(#eyeFlare)" opacity="0"/>
      <rect class="eye-flare" x="53" y="44.5" width="11" height="6" fill="url(#eyeFlare)" opacity="0"/>
    </g>

    <!-- SPARKS (hover only) -->
    <g class="lion-sparks" opacity="0">
      <circle class="spark s1" cx="50" cy="50" r="1.2" fill="#fde047"/>
      <circle class="spark s2" cx="50" cy="50" r="1"   fill="#fbbf24"/>
      <circle class="spark s3" cx="50" cy="50" r="0.9" fill="#f59e0b"/>
      <circle class="spark s4" cx="50" cy="50" r="1.1" fill="#fde047"/>
      <circle class="spark s5" cx="50" cy="50" r="0.8" fill="#fbbf24"/>
      <circle class="spark s6" cx="50" cy="50" r="1"   fill="#f59e0b"/>
      <circle class="spark s7" cx="50" cy="50" r="0.9" fill="#fde047"/>
      <circle class="spark s8" cx="50" cy="50" r="1.1" fill="#fbbf24"/>
    </g>
  </svg>`;

  window.LEO_LION = LION_SVG;

  // ──────────────────────────────────────────────────────────────
  //  Shared NAV
  // ──────────────────────────────────────────────────────────────
  const PAGES = [
    { href: '/',                  label: 'Home' },
    { href: '/architecture.html', label: 'Architecture' },
    { href: '/results.html',      label: 'Results' },
    { href: '/forecasting.html',  label: 'Forecasting' },
    { href: '/confidence.html',   label: 'Confidence' },
    { href: '/agent.html',        label: 'Agent' },
    { href: '/pipeline.html',     label: 'Pipeline' },
    { href: '/dataset.html',      label: 'Dataset' },
    { href: '/cite.html',         label: 'Cite' },
  ];

  function currentPath() {
    let p = window.location.pathname;
    if (p.endsWith('/index.html')) p = '/';
    if (p === '') p = '/';
    return p;
  }

  function buildNav() {
    const cur = currentPath();
    const links = PAGES.map(p => {
      const active = (p.href === cur) ? ' aria-current="page"' : '';
      return `<a href="${p.href}"${active}>${p.label}</a>`;
    }).join('');

    return `
      <a class="brand" href="/" aria-label="LEO — home">
        <span class="brand-mark" aria-hidden="true"></span>
        <span class="brand-text">LEO<span class="dot">·</span><i>api</i></span>
      </a>
      <nav class="primary-nav" aria-label="Primary">
        ${links}
      </nav>
      <div class="nav-meta">
        <span class="status-dot"></span>
        <span class="status-text">live</span>
        <a class="nav-cta" href="/cite.html">Cite <span aria-hidden="true">→</span></a>
      </div>
    `;
  }

  function buildFooter() {
    const yr = new Date().getFullYear();
    return `
      <div class="foot-left">
        <div class="foot-brand">LEO · API Intelligence</div>
        <div class="foot-line">
          Predictive reliability modelling for banking-API systems.
          A multi-horizon attention-LSTM that calls the alarm before the failure lands.
        </div>
      </div>
      <div class="foot-right">
        <div class="foot-col">
          <div class="fc-k">Pages</div>
          <a href="/architecture.html">Architecture</a>
          <a href="/results.html">Results</a>
          <a href="/forecasting.html">Forecasting</a>
          <a href="/agent.html">Agent</a>
        </div>
        <div class="foot-col">
          <div class="fc-k">More</div>
          <a href="/confidence.html">Confidence bands</a>
          <a href="/pipeline.html">Self-healing pipeline</a>
          <a href="/dataset.html">Dataset</a>
          <a href="/cite.html">Cite the work</a>
        </div>
        <div class="foot-col">
          <div class="fc-k">Status</div>
          <a href="/health">Health endpoint</a>
          <a href="/api/snapshot">Snapshot API</a>
          <span class="muted">© ${yr} · MIT licence</span>
        </div>
      </div>
    `;
  }

  // ──────────────────────────────────────────────────────────────
  //  Ambient layers (auto-inject)
  // ──────────────────────────────────────────────────────────────
  function ensureAmbient() {
    if (!document.getElementById('bg-particles')) {
      const c = document.createElement('canvas');
      c.id = 'bg-particles';
      c.setAttribute('aria-hidden', 'true');
      document.body.insertBefore(c, document.body.firstChild);
    }
    if (!document.getElementById('cursor-glow')) {
      const d = document.createElement('div');
      d.id = 'cursor-glow';
      d.className = 'cursor-glow';
      d.setAttribute('aria-hidden', 'true');
      document.body.appendChild(d);
    }
    if (!document.querySelector('.grain')) {
      const d = document.createElement('div');
      d.className = 'grain'; d.setAttribute('aria-hidden', 'true');
      document.body.appendChild(d);
    }
    if (!document.querySelector('.scanline')) {
      const d = document.createElement('div');
      d.className = 'scanline'; d.setAttribute('aria-hidden', 'true');
      document.body.appendChild(d);
    }
    if (!document.getElementById('scroll-progress')) {
      const d = document.createElement('div');
      d.id = 'scroll-progress'; d.className = 'scroll-progress';
      d.setAttribute('aria-hidden', 'true');
      document.body.appendChild(d);
    }
  }

  function injectChrome() {
    ensureAmbient();

    let nav = document.getElementById('site-nav');
    if (!nav) {
      nav = document.createElement('header');
      nav.id = 'site-nav';
      nav.className = 'site-nav';
      document.body.insertBefore(nav, document.body.firstChild);
    }
    nav.innerHTML = buildNav();

    let foot = document.querySelector('.site-foot');
    if (!foot) {
      foot = document.createElement('footer');
      foot.className = 'site-foot';
      document.body.appendChild(foot);
    }
    foot.innerHTML = buildFooter();
  }

  // ──────────────────────────────────────────────────────────────
  //  LEO Explains  ·  small chat-bubble brief, type-on entry
  //  Use:  <div class="leo-says" data-msg="Plain-English summary..."></div>
  //  Or call window.LEO_SAYS(target, "msg")
  // ──────────────────────────────────────────────────────────────
  function buildLeoSays(target, text) {
    target.innerHTML = `
      <div class="avatar">${LION_SVG}</div>
      <div class="body">
        <div class="who">LEO explains</div>
        <div class="msg"></div>
      </div>`;

    const msgEl = target.querySelector('.msg');
    // Render markdown-lite — **bold** → <b>
    const html = String(text || '').replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');

    // Type-on once the bubble enters the viewport
    function typeIn() {
      let i = 0;
      const total = html.length;
      const speed = Math.max(8, Math.min(22, 1400 / Math.max(20, total)));
      msgEl.innerHTML = '<span class="typing-cursor"></span>';
      const cursor = msgEl.querySelector('.typing-cursor');
      function tick() {
        i = Math.min(total, i + 2);
        msgEl.innerHTML = html.slice(0, i) + '<span class="typing-cursor"></span>';
        if (i < total) setTimeout(tick, speed);
      }
      tick();
    }

    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((es) => {
        es.forEach(e => {
          if (e.isIntersecting) { typeIn(); io.disconnect(); }
        });
      }, { threshold: 0.35 });
      io.observe(target);
    } else {
      msgEl.innerHTML = html;
    }
  }

  window.LEO_SAYS = buildLeoSays;

  // ──────────────────────────────────────────────────────────────
  //  Boot
  // ──────────────────────────────────────────────────────────────
  function boot() {
    injectChrome();
    document.querySelectorAll('.leo-says[data-msg]').forEach(el => {
      buildLeoSays(el, el.dataset.msg);
    });
  }

  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
