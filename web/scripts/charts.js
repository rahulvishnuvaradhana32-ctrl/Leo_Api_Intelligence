/* ============================================================
 * LEO · charts.js
 *  Custom canvas charts — no external dependencies.
 *
 *  Exported on window.LEO_CHARTS:
 *   - drawLossChart()         → results section
 *   - drawForecastChart()     → animated multi-horizon timeline
 *   - drawApiDonut()          → dataset section
 *   - buildAblationBars()     → ablation row
 * ============================================================ */

(function () {
  'use strict';

  const D = window.LEO_DATA;
  const TAU = Math.PI * 2;

  const COLORS = {
    bg:      '#060912',
    grid:    'rgba(124,148,210,0.10)',
    text:    '#A6B2D0',
    mute:    '#6F7C9C',
    gold:    '#4D8DFF',   // primary line (electric blue)
    ember:   '#8B5CF6',   // secondary (violet)
    crimson: '#A78BFA',   // tertiary (light violet)
    blood:   '#6D5BE0',
    green:   '#34D399',
    amber:   '#4D8DFF',
    red:     '#F87171',
    ivory:   '#EAF1FF',
    // legacy aliases kept so old chart code still resolves
    cyan:    '#38BDF8',   // val line (cyan)
    violet:  '#8B5CF6',
    magenta: '#A78BFA',
  };

  function ctxFor(id) {
    const c = document.getElementById(id);
    if (!c) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = c.getBoundingClientRect();
    const w = rect.width || c.clientWidth;
    const h = parseInt(c.getAttribute('height')) || rect.height;
    c.width  = w * dpr;
    c.height = h * dpr;
    c.style.width  = w + 'px';
    c.style.height = h + 'px';
    const x = c.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { c, x, w, h };
  }

  // ────────────────────────────────────────────────────────────
  //  1.  Training loss chart
  // ────────────────────────────────────────────────────────────
  function drawLossChart() {
    const ref = ctxFor('lossChart');
    if (!ref) return;
    const { c, x, w, h } = ref;

    const train = D.lstm.train_losses;
    const val   = D.lstm.val_losses;
    const all = train.concat(val);
    const minV = Math.min(...all) - 0.0008;
    const maxV = Math.max(...all) + 0.0008;
    const padL = 48, padR = 20, padT = 18, padB = 32;

    const sx = i => padL + (i / (train.length - 1)) * (w - padL - padR);
    const sy = v => h - padB - ((v - minV) / (maxV - minV)) * (h - padT - padB);

    // grid
    x.strokeStyle = COLORS.grid;
    x.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = padT + g * (h - padT - padB) / 4;
      x.beginPath(); x.moveTo(padL, y); x.lineTo(w - padR, y); x.stroke();
      x.fillStyle = COLORS.mute;
      x.font = '10px JetBrains Mono';
      x.textAlign = 'right';
      const val = maxV - (g / 4) * (maxV - minV);
      x.fillText(val.toFixed(4), padL - 6, y + 3);
    }

    // x labels
    x.fillStyle = COLORS.mute;
    x.font = '10px JetBrains Mono';
    x.textAlign = 'center';
    [0, 7, 14, 21, 29].forEach(i => {
      x.fillText('epoch ' + (i + 1), sx(i), h - 10);
    });

    // train area (subtle)
    const grad = x.createLinearGradient(0, padT, 0, h - padB);
    grad.addColorStop(0, 'rgba(77,141,255,0.25)');
    grad.addColorStop(1, 'rgba(77,141,255,0)');
    x.fillStyle = grad;
    x.beginPath();
    x.moveTo(sx(0), h - padB);
    train.forEach((v, i) => x.lineTo(sx(i), sy(v)));
    x.lineTo(sx(train.length - 1), h - padB);
    x.closePath();
    x.fill();

    // train line
    x.strokeStyle = COLORS.amber;
    x.lineWidth = 2;
    x.beginPath();
    train.forEach((v, i) => i === 0 ? x.moveTo(sx(i), sy(v)) : x.lineTo(sx(i), sy(v)));
    x.stroke();

    // val area
    const grad2 = x.createLinearGradient(0, padT, 0, h - padB);
    grad2.addColorStop(0, 'rgba(56,189,248,0.30)');
    grad2.addColorStop(1, 'rgba(56,189,248,0)');
    x.fillStyle = grad2;
    x.beginPath();
    x.moveTo(sx(0), h - padB);
    val.forEach((v, i) => x.lineTo(sx(i), sy(v)));
    x.lineTo(sx(val.length - 1), h - padB);
    x.closePath();
    x.fill();

    // val line
    x.strokeStyle = COLORS.cyan;
    x.lineWidth = 2.5;
    x.beginPath();
    val.forEach((v, i) => i === 0 ? x.moveTo(sx(i), sy(v)) : x.lineTo(sx(i), sy(v)));
    x.stroke();

    // dots on best val
    const minIdx = val.indexOf(Math.min(...val));
    x.fillStyle = COLORS.cyan;
    x.beginPath();
    x.arc(sx(minIdx), sy(val[minIdx]), 5, 0, TAU);
    x.fill();
    x.strokeStyle = '#fff';
    x.lineWidth = 1.5;
    x.stroke();

    // label best
    x.fillStyle = COLORS.text;
    x.font = '11px JetBrains Mono';
    x.textAlign = 'left';
    x.fillText('best · ' + val[minIdx].toFixed(4), sx(minIdx) + 8, sy(val[minIdx]) - 6);
  }

  // ────────────────────────────────────────────────────────────
  //  2.  Forecast multi-horizon timeline (animated playback)
  // ────────────────────────────────────────────────────────────
  function drawForecastChart() {
    const ref = ctxFor('forecastChart');
    if (!ref) return;
    const { c, x, w, h } = ref;

    // Synthesize a plausible 120-step time series with an emerging failure event
    const N = 120;
    const seed = 42;
    let rnd = mulberry32(seed);
    function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

    // Probability tracks for h+1, h+5, h+15
    const p1 = [], p5 = [], p15 = [];
    let base = 0.18 + rnd()*0.05;
    for (let i = 0; i < N; i++) {
      // event around t=60..90 ramps probabilities up
      const event = Math.max(0, Math.min(1, (i - 55) / 25));
      const noise = (rnd() - 0.5) * 0.06;
      const lvl  = base + event * 0.65 + noise;
      p1.push(  clamp(lvl + (rnd()-0.5)*0.04 + (i>80? -0.15:0), 0.02, 0.99));
      p5.push(  clamp(lvl*0.9 + (rnd()-0.5)*0.05, 0.02, 0.99));
      p15.push( clamp(lvl*0.75 + (rnd()-0.5)*0.05, 0.02, 0.99));
    }

    function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

    const padL = 44, padR = 16, padT = 28, padB = 36;

    function paint(progress) {
      x.clearRect(0, 0, w, h);

      const drawN = Math.floor(N * progress);

      const sx = i => padL + (i / (N - 1)) * (w - padL - padR);
      const sy = v => h - padB - v * (h - padT - padB);

      // grid + y labels
      x.strokeStyle = COLORS.grid;
      x.lineWidth = 1;
      [0, 0.25, 0.5, 0.75, 1].forEach(v => {
        const y = sy(v);
        x.beginPath(); x.moveTo(padL, y); x.lineTo(w - padR, y); x.stroke();
        x.fillStyle = COLORS.mute;
        x.font = '10px JetBrains Mono';
        x.textAlign = 'right';
        x.fillText((v * 100).toFixed(0) + '%', padL - 8, y + 3);
      });

      // alert threshold band
      x.fillStyle = 'rgba(77,141,255,0.06)';
      x.fillRect(padL, sy(0.55), w - padL - padR, sy(0.35) - sy(0.55));
      x.strokeStyle = 'rgba(77,141,255,0.35)';
      x.setLineDash([4, 4]);
      x.beginPath(); x.moveTo(padL, sy(0.55)); x.lineTo(w - padR, sy(0.55)); x.stroke();
      x.setLineDash([]);

      x.fillStyle = COLORS.amber;
      x.font = '10px JetBrains Mono';
      x.textAlign = 'left';
      x.fillText('high-risk · 0.55', padL + 6, sy(0.55) - 4);

      // x ticks
      x.fillStyle = COLORS.mute;
      x.textAlign = 'center';
      for (let i = 0; i < N; i += 20) {
        x.fillText('t+' + i, sx(i), h - 12);
      }

      // helper drawer
      function drawLine(arr, color, fill) {
        if (drawN < 2) return;
        if (fill) {
          const g = x.createLinearGradient(0, padT, 0, h - padB);
          g.addColorStop(0, fill);
          g.addColorStop(1, 'transparent');
          x.fillStyle = g;
          x.beginPath();
          x.moveTo(sx(0), h - padB);
          for (let i = 0; i < drawN; i++) x.lineTo(sx(i), sy(arr[i]));
          x.lineTo(sx(drawN - 1), h - padB);
          x.closePath();
          x.fill();
        }
        x.strokeStyle = color;
        x.lineWidth = 2;
        x.beginPath();
        for (let i = 0; i < drawN; i++) {
          if (i === 0) x.moveTo(sx(i), sy(arr[i]));
          else x.lineTo(sx(i), sy(arr[i]));
        }
        x.stroke();
      }

      drawLine(p15, COLORS.crimson, 'rgba(167,139,250,0.18)');
      drawLine(p5,  COLORS.ember,   'rgba(139,92,246,0.18)');
      drawLine(p1,  COLORS.gold,    'rgba(77,141,255,0.20)');

      // current marker
      if (drawN > 0) {
        const i = drawN - 1;
        x.strokeStyle = 'rgba(255,255,255,0.35)';
        x.lineWidth = 1;
        x.setLineDash([3, 4]);
        x.beginPath(); x.moveTo(sx(i), padT); x.lineTo(sx(i), h - padB); x.stroke();
        x.setLineDash([]);

        [['#fbbf24', p1[i]], ['#f59e0b', p5[i]], ['#dc2626', p15[i]]].forEach(([col, v]) => {
          x.fillStyle = col;
          x.beginPath();
          x.arc(sx(i), sy(v), 4.5, 0, TAU);
          x.fill();
          x.strokeStyle = '#060912';
          x.lineWidth = 1.5;
          x.stroke();
        });

        // readout panel update
        const setT = document.getElementById('fcT');
        const set1 = document.getElementById('fc1');
        const set5 = document.getElementById('fc5');
        const setF = document.getElementById('fc15');
        const setA = document.getElementById('fcAlert');
        if (setT) setT.textContent = 't+' + i;
        if (set1) set1.textContent = (p1[i]  * 100).toFixed(1) + '%';
        if (set5) set5.textContent = (p5[i]  * 100).toFixed(1) + '%';
        if (setF) setF.textContent = (p15[i] * 100).toFixed(1) + '%';
        if (setA) {
          const max = Math.max(p1[i], p5[i], p15[i]);
          if (max >= 0.55)      { setA.textContent = 'switch'; setA.style.color = '#f87171'; }
          else if (max >= 0.35) { setA.textContent = 'retry';  setA.style.color = '#fbbf24'; }
          else                  { setA.textContent = 'normal'; setA.style.color = '#34d399'; }
        }
      }

      // legend
      x.font = '11px JetBrains Mono';
      x.textAlign = 'left';
      let lx = padL + 8;
      const ly = padT - 14;
      [['h+1', COLORS.gold], ['h+5', COLORS.ember], ['h+15', COLORS.crimson]].forEach(([k, col]) => {
        x.fillStyle = col;
        x.fillRect(lx, ly - 4, 10, 3);
        x.fillStyle = COLORS.text;
        x.fillText(k, lx + 14, ly);
        lx += 60;
      });
    }

    // animate once on first reveal, then loop slowly forever
    let t0 = null;
    let phase = 'in'; // 'in' → 'idle' → 'loop'

    function frame(ts) {
      if (t0 == null) t0 = ts;
      const elapsed = ts - t0;
      if (phase === 'in') {
        const p = Math.min(1, elapsed / 2400);
        paint(p);
        if (p >= 1) { phase = 'idle'; t0 = ts; }
      } else if (phase === 'idle') {
        if (elapsed > 1400) { phase = 'loop'; t0 = ts; }
      } else {
        const p = Math.min(1, elapsed / 4500);
        paint(p);
        if (p >= 1) { phase = 'idle'; t0 = ts; }
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // ────────────────────────────────────────────────────────────
  //  3.  API donut chart
  // ────────────────────────────────────────────────────────────
  function drawApiDonut() {
    const c = document.getElementById('apiDonut');
    if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const SIZE = 320;
    c.width  = SIZE * dpr; c.height = SIZE * dpr;
    c.style.width = SIZE + 'px'; c.style.height = SIZE + 'px';
    const x = c.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);

    const cx = SIZE / 2, cy = SIZE / 2;
    const Ro = 130, Ri = 78;
    const items = D.dataset.apis;
    const total = items.reduce((s, it) => s + it.share, 0);

    // animate sweep
    let progress = 0;
    function paint() {
      x.clearRect(0, 0, SIZE, SIZE);

      let start = -Math.PI / 2;
      items.forEach((it, idx) => {
        const sweep = (it.share / total) * TAU * progress;
        x.beginPath();
        x.fillStyle = it.color;
        x.shadowBlur = 22;
        x.shadowColor = it.color;
        x.globalAlpha = 0.92;
        x.moveTo(cx, cy);
        x.arc(cx, cy, Ro, start, start + sweep);
        x.closePath();
        x.fill();
        x.shadowBlur = 0;
        start += sweep;
      });
      x.globalAlpha = 1;

      // inner hole
      x.fillStyle = '#060912';
      x.beginPath();
      x.arc(cx, cy, Ri, 0, TAU);
      x.fill();

      // center text
      x.fillStyle = '#eef1ff';
      x.font = '600 22px Space Grotesk';
      x.textAlign = 'center';
      x.textBaseline = 'middle';
      x.fillText('5 + 1', cx, cy - 8);
      x.font = '11px JetBrains Mono';
      x.fillStyle = '#aab2d4';
      x.fillText('APIs · ' + Math.round(total * 100) + '%', cx, cy + 14);
    }

    function step() {
      progress = Math.min(1, progress + 0.025);
      paint();
      if (progress < 1) requestAnimationFrame(step);
    }
    step();

    // legend
    const leg = document.getElementById('apiLegend');
    if (leg) {
      leg.innerHTML = items.map(it =>
        `<div class="lg"><span class="sw" style="background:${it.color}"></span>
         <span>${it.name}</span><b>${(it.share * 100).toFixed(1)}%</b></div>`
      ).join('');
    }
  }

  // ────────────────────────────────────────────────────────────
  //  4.  Ablation bars
  // ────────────────────────────────────────────────────────────
  function buildAblationBars() {
    const root = document.getElementById('ablationBars');
    if (!root) return;
    const exps = D.ablation.experiments;
    const min = Math.min(...exps.map(e => e.auc));
    const max = Math.max(...exps.map(e => e.auc));
    const span = max - min || 1;
    root.innerHTML = exps.map(e => {
      const t = (e.auc - min) / span;
      const w = (20 + t * 80).toFixed(1);   // 20–100% range for visual punch
      const cls = e.base ? ' base' : '';
      return `
        <div class="abl-bar${cls}">
          <span class="name">${e.name}</span>
          <span class="track"><span class="fill" style="transform:scaleX(${(w/100).toFixed(3)})"></span></span>
          <span class="num">${(e.auc * 100).toFixed(2)}%</span>
        </div>`;
    }).join('');
  }

  // ────────────────────────────────────────────────────────────
  //  Exports
  // ────────────────────────────────────────────────────────────
  window.LEO_CHARTS = {
    drawLossChart,
    drawForecastChart,
    drawApiDonut,
    buildAblationBars,
  };
})();
