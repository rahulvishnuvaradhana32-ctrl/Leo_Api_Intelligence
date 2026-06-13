/* ============================================================
 * LEO · metrics.js
 *   Advanced server-themed metric reveal.
 *   - Odometer digit reels (digits scroll into place)
 *   - Sonar ping ring around metric icon
 *   - Request packet animation under each card
 *   - Status LEDs synchronised with the reveal
 *   - Live "API request" ticker tape under the rack
 * ============================================================ */

(function () {
  'use strict';

  const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ────────────────────────────────────────────────────────────
  //  Odometer  ·  builds a stack of digit reels.
  //  data-odo="82.02" data-suffix="%" data-prefix="$"
  // ────────────────────────────────────────────────────────────
  function buildOdometer(el) {
    if (el.dataset.odoBuilt) return;
    el.dataset.odoBuilt = '1';

    const target = parseFloat(el.dataset.odo);
    if (isNaN(target)) return;

    const decimals = parseInt(el.dataset.decimals || '0', 10);
    const prefix = el.dataset.prefix || '';
    const suffix = el.dataset.suffix || '';
    const thousands = el.dataset.thousands === '1';

    // Format final string
    let s;
    if (thousands) {
      s = target.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      });
    } else {
      s = target.toFixed(decimals);
    }

    el.innerHTML = '';

    if (prefix) {
      const p = document.createElement('span');
      p.className = 'lead'; p.textContent = prefix;
      el.appendChild(p);
    }

    // For each character: if digit → reel, else literal
    [...s].forEach((ch, i) => {
      if (/\d/.test(ch)) {
        const odo = document.createElement('span');
        odo.className = 'odo';
        const reel = document.createElement('span');
        reel.className = 'reel';
        // build 0..ch column (so it reveals by sliding up)
        const finalDigit = parseInt(ch, 10);
        // we render digits from 9 → ch so transform: translateY(-(9-ch)*1em) lands on ch
        // simpler: render the full 0..9 cycle plus the target at the end
        const sequence = [];
        for (let r = 0; r < 1; r++) for (let d = 0; d <= 9; d++) sequence.push(d);
        sequence.push(finalDigit);
        sequence.forEach(d => {
          const span = document.createElement('span');
          span.textContent = d;
          reel.appendChild(span);
        });
        odo.appendChild(reel);
        el.appendChild(odo);
        // stash final index so we know the offset
        reel.dataset.finalIndex = String(sequence.length - 1);
      } else {
        const lit = document.createElement('span');
        lit.className = 'lit';
        lit.textContent = ch;
        el.appendChild(lit);
      }
    });

    if (suffix) {
      const sfx = document.createElement('span');
      sfx.className = 'suffix'; sfx.textContent = suffix;
      el.appendChild(sfx);
    }
  }

  function runOdometer(el, delay = 0) {
    const reels = el.querySelectorAll('.odo .reel');
    reels.forEach((reel, idx) => {
      const finalIndex = parseInt(reel.dataset.finalIndex || '0', 10);
      // each child is 1em tall; we translate -finalIndex em
      setTimeout(() => {
        reel.style.transform = `translateY(-${finalIndex}em)`;
      }, delay + idx * 80);
    });
  }

  // ────────────────────────────────────────────────────────────
  //  Init all metrics on the page
  // ────────────────────────────────────────────────────────────
  function initMetrics() {
    const cards = document.querySelectorAll('.metric');
    if (!cards.length) return;

    cards.forEach(card => {
      // build odometer from .metric-num span[data-odo]
      const numEl = card.querySelector('[data-odo]');
      if (numEl) buildOdometer(numEl);

      // make sure flow has 4 packets
      const flow = card.querySelector('.metric-flow');
      if (flow && !flow.querySelector('.pkt')) {
        const lane = document.createElement('span'); lane.className = 'lane'; flow.appendChild(lane);
        for (let i = 0; i < 4; i++) {
          const p = document.createElement('span'); p.className = 'pkt'; flow.appendChild(p);
        }
      }
    });

    if (REDUCED) {
      cards.forEach(c => {
        c.querySelectorAll('.odo .reel').forEach(reel => {
          const idx = parseInt(reel.dataset.finalIndex || '0', 10);
          reel.style.transform = `translateY(-${idx}em)`;
        });
      });
      return;
    }

    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries) => {
        entries.forEach(en => {
          if (en.isIntersecting) {
            const numEl = en.target.querySelector('[data-odo]');
            if (numEl) runOdometer(numEl, 250);
            io.unobserve(en.target);
          }
        });
      }, { threshold: 0.45 });
      cards.forEach(c => io.observe(c));
    } else {
      cards.forEach(c => {
        const numEl = c.querySelector('[data-odo]');
        if (numEl) runOdometer(numEl);
      });
    }
  }

  // ────────────────────────────────────────────────────────────
  //  Request ticker  ·  synthetic live API log scrolling beneath rack
  // ────────────────────────────────────────────────────────────
  function buildTicker(host) {
    if (!host || host.dataset.tickerBuilt) return;
    host.dataset.tickerBuilt = '1';

    const VERBS = ['GET', 'POST', 'PUT', 'PATCH'];
    const APIS = [
      '/v1/transaction', '/v1/market-data', '/v1/stock-price',
      '/v1/crypto/tick',  '/v1/forex/quote',  '/v1/risk/score',
      '/v1/conformal/q',  '/v1/predict',      '/v1/agent/decide',
      '/v1/heartbeat',    '/v1/scaler/fit',   '/v1/healthcheck',
    ];
    const STATUS = [
      ['200 ok',    'ok',   '14 ms'],
      ['200 ok',    'ok',   '22 ms'],
      ['200 ok',    'ok',   '38 ms'],
      ['200 ok',    'ok',   '11 ms'],
      ['200 ok',    'ok',   '47 ms'],
      ['429 retry', 'warn', '180 ms'],
      ['200 ok',    'ok',   '19 ms'],
      ['504 gateway', 'bad', '2200 ms'],
      ['200 ok',    'ok',   '26 ms'],
      ['200 ok',    'ok',   '33 ms'],
      ['LEO predict h+5 risk=0.31', 'warn', '8 ms'],
      ['LEO predict h+1 risk=0.07', 'ok',   '6 ms'],
      ['LEO predict h+15 risk=0.62 → SWITCH', 'bad', '9 ms'],
    ];

    function row() {
      const v = VERBS[Math.floor(Math.random() * VERBS.length)];
      const a = APIS[Math.floor(Math.random() * APIS.length)];
      const [stxt, cls, lat] = STATUS[Math.floor(Math.random() * STATUS.length)];
      return `<span class="item">
        <span class="verb">${v}</span>
        <span class="ep">${a}</span>
        <span class="${cls}">${stxt}</span>
        <span class="lat">${lat}</span>
      </span>`;
    }

    // Build enough rows for an apparent infinite loop (ticker animation translates -50%)
    const items = Array.from({ length: 16 }, row).join('');
    host.innerHTML = `<div class="track">${items}${items}</div>`;
  }

  // ────────────────────────────────────────────────────────────
  //  Boot
  // ────────────────────────────────────────────────────────────
  function boot() {
    initMetrics();
    buildTicker(document.querySelector('.req-ticker'));
  }

  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
