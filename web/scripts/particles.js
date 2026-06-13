/* ============================================================
 * LEO · particles.js
 *  - Ambient particle network on <canvas id="bg-particles">
 *  - Cursor-following glow
 *  - Hero SVG network nodes
 * ============================================================ */

function _leoStartParticles() {
  'use strict';

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return;
  }

  // ────────────────────────── 1.  Background particle network
  const canvas = document.getElementById('bg-particles');
  if (!canvas) return;     // chrome.js hasn't injected yet — skip silently
  const ctx = canvas.getContext('2d');

  let W, H, DPR;
  const particles = [];
  const COUNT = 70;
  const MAX_DIST = 140;

  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width  = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  function spawn() {
    particles.length = 0;
    for (let i = 0; i < COUNT; i++) {
      particles.push({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        r: Math.random() * 1.4 + 0.4,
        hue: Math.random() < 0.5 ? '#fbbf24' : '#dc2626',
      });
    }
  }

  let mx = -9999, my = -9999;
  window.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });

  function step() {
    ctx.clearRect(0, 0, W, H);

    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;

      // mouse repel
      const dx = p.x - mx, dy = p.y - my;
      const d2 = dx * dx + dy * dy;
      if (d2 < 12000) {
        const f = (12000 - d2) / 12000 * 0.4;
        p.x += dx / Math.sqrt(d2) * f;
        p.y += dy / Math.sqrt(d2) * f;
      }

      ctx.beginPath();
      ctx.fillStyle = p.hue;
      ctx.globalAlpha = 0.55;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    // edges
    ctx.lineWidth = 1;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i], b = particles[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < MAX_DIST) {
          const o = 1 - d / MAX_DIST;
          ctx.strokeStyle = `rgba(220,140,100,${o * 0.18})`;
          ctx.globalAlpha = o * 0.5;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    ctx.globalAlpha = 1;
    requestAnimationFrame(step);
  }

  window.addEventListener('resize', () => { resize(); spawn(); });
  resize();
  spawn();
  step();

  // ────────────────────────── 2. Cursor glow follower
  const glow = document.getElementById('cursor-glow');
  let gx = -9999, gy = -9999, cx = -9999, cy = -9999;

  window.addEventListener('mousemove', e => { gx = e.clientX; gy = e.clientY; });
  function followCursor() {
    cx += (gx - cx) * 0.16;
    cy += (gy - cy) * 0.16;
    if (glow) {
      glow.style.transform = `translate(${cx}px, ${cy}px) translate(-50%, -50%)`;
    }
    requestAnimationFrame(followCursor);
  }
  followCursor();

  document.addEventListener('mouseleave', () => { if (glow) glow.style.opacity = 0; });
  document.addEventListener('mouseenter', () => { if (glow) glow.style.opacity = 1; });

  // ────────────────────────── 3. Hero network nodes (inside SVG) — only if present
  const heroNet = document.getElementById('hero-network');
  if (heroNet) {
    const cxC = 240, cyC = 240;
    const rings = [
      { r: 120, n: 8,  size: 4,   color: '#fbbf24' },
      { r: 180, n: 12, size: 3,   color: '#f59e0b' },
      { r: 220, n: 16, size: 2.2, color: '#dc2626' },
    ];

    const svgns = 'http://www.w3.org/2000/svg';

    rings.forEach((ring, ri) => {
      for (let i = 0; i < ring.n; i++) {
        const angle = (i / ring.n) * Math.PI * 2 + ri * 0.18;
        const x = cxC + Math.cos(angle) * ring.r;
        const y = cyC + Math.sin(angle) * ring.r;

        // line from center
        const line = document.createElementNS(svgns, 'line');
        line.setAttribute('x1', cxC);
        line.setAttribute('y1', cyC);
        line.setAttribute('x2', x);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', ring.color);
        line.setAttribute('stroke-opacity', '0.12');
        line.setAttribute('stroke-width', '1');
        heroNet.appendChild(line);

        // node
        const dot = document.createElementNS(svgns, 'circle');
        dot.setAttribute('cx', x);
        dot.setAttribute('cy', y);
        dot.setAttribute('r', ring.size);
        dot.setAttribute('fill', ring.color);
        dot.setAttribute('opacity', 0.6 + Math.random() * 0.4);
        dot.style.filter = 'drop-shadow(0 0 6px ' + ring.color + ')';
        dot.style.animation = `pulse-dot ${1.4 + Math.random() * 2}s ease-in-out infinite`;
        dot.style.animationDelay = `${Math.random() * 2}s`;
        dot.style.transformOrigin = `${x}px ${y}px`;
        heroNet.appendChild(dot);
      }
    });

    // animated traveling pulses on connectors
    const path1 = document.createElementNS(svgns, 'circle');
    path1.setAttribute('r', '3');
    path1.setAttribute('fill', '#fbbf24');
    path1.style.filter = 'drop-shadow(0 0 8px #fbbf24)';
    heroNet.appendChild(path1);

    const path2 = document.createElementNS(svgns, 'circle');
    path2.setAttribute('r', '3');
    path2.setAttribute('fill', '#dc2626');
    path2.style.filter = 'drop-shadow(0 0 8px #dc2626)';
    heroNet.appendChild(path2);

    let t = 0;
    function animateTravelers() {
      t += 0.012;
      const ang1 = t;
      const ang2 = -t * 0.7 + 1.2;
      path1.setAttribute('cx', cxC + Math.cos(ang1) * 120);
      path1.setAttribute('cy', cyC + Math.sin(ang1) * 120);
      path2.setAttribute('cx', cxC + Math.cos(ang2) * 180);
      path2.setAttribute('cy', cyC + Math.sin(ang2) * 180);
      requestAnimationFrame(animateTravelers);
    }
    animateTravelers();
  }
}

if (document.readyState !== 'loading') _leoStartParticles();
else document.addEventListener('DOMContentLoaded', _leoStartParticles);

