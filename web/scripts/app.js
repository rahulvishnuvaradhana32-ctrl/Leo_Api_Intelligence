/* ============================================================
 * LEO · app.js
 *  Boots the page once DOM is ready and triggers each subsystem.
 * ============================================================ */

(function () {
  'use strict';

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  ready(() => {
    if (!window.LEO_CHARTS) return;
    const C = window.LEO_CHARTS;

    // Loss chart: redraw on resize too
    C.drawLossChart();
    C.drawApiDonut();
    C.buildAblationBars();

    // Forecast chart: kick off the animation
    C.drawForecastChart();

    // Redraw responsive canvases on resize
    let rT = null;
    window.addEventListener('resize', () => {
      clearTimeout(rT);
      rT = setTimeout(() => {
        C.drawLossChart();
        C.drawApiDonut();
      }, 180);
    });

    // Optional: live refresh signal (when served via FastAPI)
    // Polls /api/last_modified — reloads on change.
    if (location.hostname && location.hostname !== '' && location.protocol.startsWith('http')) {
      let lastMod = null;
      setInterval(async () => {
        try {
          const r = await fetch('/api/last_modified', { cache: 'no-store' });
          if (!r.ok) return;
          const j = await r.json();
          if (lastMod === null) lastMod = j.last_modified;
          else if (j.last_modified > lastMod) location.reload();
        } catch (_) {}
      }, 20000);
    }
  });
})();
