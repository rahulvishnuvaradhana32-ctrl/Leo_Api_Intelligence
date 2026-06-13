/* ============================================================
 * LEO · animations.js
 *  - IntersectionObserver scroll reveals
 *  - Animated counters (data-counter)
 *  - AUC ring progress trigger
 *  - Nav shrink + scroll progress bar
 * ============================================================ */

(function () {
  'use strict';

  // ──────────────  data-p → --p CSS var (avoids inline-style lint warnings)
  document.querySelectorAll('[data-p]').forEach(el => {
    el.style.setProperty('--p', el.dataset.p + '%');
  });
  document.querySelectorAll('[data-t]').forEach(el => {
    el.style.setProperty('--t', el.dataset.t + '%');
  });

  // ──────────────  Scroll reveal
  const revealEls = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(en => {
        if (en.isIntersecting) {
          en.target.classList.add('in-view');
          // fire reveal-once handlers
          if (en.target.dataset.onReveal) {
            const fn = window[en.target.dataset.onReveal];
            if (typeof fn === 'function') fn();
          }
          io.unobserve(en.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -60px 0px' });
    revealEls.forEach(el => io.observe(el));
  } else {
    revealEls.forEach(el => el.classList.add('in-view'));
  }

  // ──────────────  Counters
  function animateCounter(el) {
    const target = parseFloat(el.dataset.counter);
    if (isNaN(target)) return;
    const decimals = parseInt(el.dataset.decimals || '0', 10);
    const prefix = el.dataset.prefix || '';
    const suffix = el.dataset.suffix || '';
    const duration = 1600;
    const t0 = performance.now();

    const fmt = v => {
      if (target >= 1000 && decimals === 0) {
        return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
      }
      return v.toFixed(decimals);
    };

    function tick(now) {
      const t = Math.min(1, (now - t0) / duration);
      // easeOutCubic
      const e = 1 - Math.pow(1 - t, 3);
      el.textContent = prefix + fmt(target * e) + suffix;
      if (t < 1) requestAnimationFrame(tick);
      else      el.textContent = prefix + fmt(target) + suffix;
    }
    requestAnimationFrame(tick);
  }

  const counterObs = new IntersectionObserver((entries) => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        animateCounter(en.target);
        counterObs.unobserve(en.target);
      }
    });
  }, { threshold: 0.4 });
  document.querySelectorAll('[data-counter]').forEach(el => counterObs.observe(el));

  // ──────────────  Ring progress
  const ring = document.querySelector('.ring-progress');
  if (ring) {
    const ringObs = new IntersectionObserver((entries) => {
      entries.forEach(en => {
        if (en.isIntersecting) {
          const r = parseFloat(ring.getAttribute('r')) || 92;
          const C = 2 * Math.PI * r;
          const t = parseFloat(ring.dataset.target) || 0.82;
          const offset = C - C * t;
          ring.style.transition = 'stroke-dashoffset 1.6s cubic-bezier(.16,.84,.34,1)';
          ring.style.strokeDashoffset = offset;
          ringObs.unobserve(ring);
        }
      });
    }, { threshold: 0.4 });
    ringObs.observe(ring);
  }

  // ──────────────  Nav shrink + scroll progress
  const nav = document.getElementById('site-nav');
  const prog = document.getElementById('scroll-progress');
  function onScroll() {
    const y = window.scrollY || 0;
    if (nav) nav.classList.toggle('shrink', y > 40);
    if (prog) {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const pct = max > 0 ? (y / max) * 100 : 0;
      prog.style.width = pct.toFixed(2) + '%';
    }
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // ──────────────  Subtle parallax on hero viz
  const heroViz = document.querySelector('.hero-viz');
  const heroTitle = document.querySelector('.hero-title');
  function onParallax(e) {
    const mx = (e.clientX / window.innerWidth - 0.5);
    const my = (e.clientY / window.innerHeight - 0.5);
    if (heroViz) heroViz.style.transform = `translate(${-mx * 10}px, ${-my * 10}px)`;
    if (heroTitle) heroTitle.style.transform = `translate(${mx * 4}px, ${my * 4}px)`;
  }
  if (window.innerWidth > 980) {
    window.addEventListener('mousemove', onParallax);
  }

  // ──────────────  Smooth-scroll for anchor links (extra polish)
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href');
      if (!id || id === '#') return;
      const t = document.querySelector(id);
      if (!t) return;
      e.preventDefault();
      const navH = (nav && nav.offsetHeight) || 0;
      const y = t.getBoundingClientRect().top + window.scrollY - navH - 12;
      window.scrollTo({ top: y, behavior: 'smooth' });
    });
  });
})();
