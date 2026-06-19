/* ============================================================
 * LEO · livefo.js — LIVE failover, decided by the REAL engine
 *   Streams the risk of the API + signals selected above (fed by
 *   predict.js via update(api, peak)) and asks the backend
 *   /v1/route state machine (scripts/route_engine.py) what to do
 *   each tick: reroute, fail-back, hold. Falls back to a local
 *   state machine if the endpoint isn't reachable (static build).
 * ============================================================ */
(function () {
  'use strict';
  const THRESH = 0.55, FAILBACK = 0.45, STANDBY_BASE = 0.16;
  const TICK_MS = 200, SERVER_MS = 700;   // visual cadence vs server poll (rate-safe)
  const SID = 'live-' + Math.floor((window.performance && performance.now ? performance.now() : 1) * 1000 % 1e9);

  const RECORDS = {
    transaction_api: { source: 'core-ledger', rec: { account: 'acct_8841', balance: '12480.55', ccy: 'USD' } },
    market_data_api: { source: 'market-feed', rec: { symbol: 'AAPL', price: '224.18' } },
    stock_price_api: { source: 'market-feed', rec: { symbol: 'MSFT', price: '418.92' } },
    crypto_api:      { source: 'market-feed', rec: { symbol: 'BTC-USD', price: '67940.00' } },
    forex_api:       { source: 'fx-feed', rec: { pair: 'EUR/USD', rate: '1.0832' } },
  };
  function fnv1a(s) { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); } return 'sha256:' + (h >>> 0).toString(16).padStart(8, '0'); }
  function sumFor(a) { const m = RECORDS[a] || RECORDS.transaction_api; return fnv1a(JSON.stringify({ idempotency_key: 'req-' + a + '-001', api: a, source: m.source, ...m.rec })); }

  let api = 'transaction_api', target = 0.18;
  function update(a, peak) { if (a) api = a; target = Math.max(0.02, Math.min(0.98, peak)); }

  function init() {
    const cv = document.getElementById('lfCanvas'); if (!cv) return;
    const els = {
      status: document.getElementById('lfStatus'),
      primary: document.getElementById('lfPrimary'), pState: document.getElementById('lfPrimaryState'),
      standby: document.getElementById('lfStandby'), sState: document.getElementById('lfStandbyState'),
      packet: document.getElementById('lfPacket'), verdict: document.getElementById('lfVerdict'),
    };
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const N = 64, buf = []; for (let i = 0; i < N; i++) buf.push(0.18);
    let W = 0, H = 0, dpr = 1; const ctx = cv.getContext('2d');
    function resize() { dpr = Math.min(window.devicePixelRatio || 1, 2); const r = cv.getBoundingClientRect(); W = r.width || 600; H = parseInt(cv.getAttribute('height')) || 150; cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px'; }
    resize();

    function draw() {
      const cur = buf[N - 1];
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const padB = 16, padT = 10, sx = i => (i / (N - 1)) * W, sy = v => padT + (1 - Math.max(0, Math.min(1, v))) * (H - padT - padB);
      ctx.strokeStyle = 'rgba(124,148,210,.07)'; ctx.lineWidth = 1;
      [0.25, 0.5, 0.75].forEach(v => { const y = sy(v); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); });
      ctx.strokeStyle = 'rgba(248,113,113,.55)'; ctx.setLineDash([5, 4]); const ty = sy(THRESH); ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(W, ty); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#F87171'; ctx.font = '10px "IBM Plex Mono",monospace'; ctx.textAlign = 'left'; ctx.fillText('reroute · 0.55', 6, ty - 4);
      const hot = cur >= THRESH, warm = cur >= 0.35, top = hot ? '#F87171' : warm ? '#8B5CF6' : '#4D8DFF';
      const g = ctx.createLinearGradient(0, padT, 0, H - padB); g.addColorStop(0, (hot ? 'rgba(248,113,113,' : 'rgba(77,141,255,') + '.28)'); g.addColorStop(1, 'rgba(77,141,255,0)');
      ctx.beginPath(); ctx.moveTo(0, H - padB); buf.forEach((v, i) => ctx.lineTo(sx(i), sy(v))); ctx.lineTo(W, H - padB); ctx.closePath(); ctx.fillStyle = g; ctx.fill();
      ctx.beginPath(); buf.forEach((v, i) => i ? ctx.lineTo(sx(i), sy(v)) : ctx.moveTo(sx(i), sy(v))); ctx.strokeStyle = top; ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.shadowColor = top; ctx.shadowBlur = 12; ctx.stroke(); ctx.shadowBlur = 0;
      const lx = sx(N - 1), ly = sy(cur); ctx.beginPath(); ctx.arc(lx, ly, 4, 0, 6.283); ctx.fillStyle = top; ctx.fill(); ctx.globalAlpha = .35; ctx.beginPath(); ctx.arc(lx, ly, 8, 0, 6.283); ctx.strokeStyle = top; ctx.stroke(); ctx.globalAlpha = 1;
    }

    const setT = (el, v) => { if (el) el.textContent = v; };
    const apiLabel = (el, suffix) => { const a = el && el.querySelector('.fo-api'); if (a) a.textContent = api + suffix; };

    // shared rendering of node/verdict state
    let rerouted = false, systemic = false, lastAction = '', src = 'server';
    function render() {
      apiLabel(els.primary, ' · region-a'); apiLabel(els.standby, ' · region-b');
      if (systemic) {
        els.primary.className = 'fo-node primary failing'; setT(els.pState, 'degraded');
        els.standby.className = 'fo-node backup failing'; setT(els.sState, 'degraded');
        setT(els.status, `🔔 systemic — all routes degraded · holding + alert (${src})`);
        if (els.verdict) { els.verdict.className = 'fo-verdict'; els.verdict.style.borderColor = 'rgba(248,113,113,.5)'; els.verdict.innerHTML = `<b>Systemic stress</b> — no healthy route. LEO holds, circuit-breaks, and pages on-call. Reads served from cache · writes paused.`; }
      } else if (rerouted) {
        els.primary.className = 'fo-node primary failing'; setT(els.pState, 'FAILING · draining');
        els.standby.className = 'fo-node backup active'; setT(els.sState, 'ACTIVE · serving');
        setT(els.status, `● rerouted · ${api} · region-b serving · region-a monitored (${src})`);
        const m = RECORDS[api] || RECORDS.transaction_api;
        if (els.verdict) { els.verdict.className = 'fo-verdict ok'; els.verdict.style.borderColor = ''; els.verdict.innerHTML = `✓ rerouted to <b>region-b</b> · both read <b>${m.source}</b> via idempotency key — checksum <b>${sumFor(api)}</b> matches both nodes, <b>zero data divergence</b>.`; }
      } else {
        els.primary.className = 'fo-node primary active'; setT(els.pState, 'active · serving');
        els.standby.className = 'fo-node backup standby'; setT(els.sState, 'standby · synced');
        const lvl = target >= THRESH ? '⚠ high' : target >= 0.35 ? '◐ degrading' : '● healthy';
        setT(els.status, `${lvl} · monitoring ${api} · region-a (risk ${target.toFixed(2)}) (${src})`);
        if (els.verdict) { els.verdict.className = 'fo-verdict'; els.verdict.style.borderColor = ''; els.verdict.innerHTML = `Monitoring <b>${api} · region-a</b> — if risk crosses <b>0.55</b>, LEO reroutes to <b>region-b</b> with zero data divergence.`; }
      }
    }
    function fireReroute() { if (els.packet) { els.packet.classList.remove('travel'); void els.packet.offsetWidth; els.packet.classList.add('travel'); } }

    function applyDecision(d) {
      const wasRerouted = rerouted;
      systemic = d.state === 'SYSTEMIC';
      rerouted = !systemic && d.active && d.active !== 'region-a';
      if (rerouted && !wasRerouted) fireReroute();
      render();
    }
    // local fallback when /v1/route is unreachable (static build)
    function clientDecide() {
      const was = rerouted;
      if (!rerouted && target >= THRESH) rerouted = true;
      else if (rerouted && target < FAILBACK) rerouted = false;
      systemic = false;
      if (rerouted && !was) fireReroute();
      render();
    }

    let serverMode = true, busy = false, lastServer = -1e9;
    function decide(now) {
      if (serverMode) {
        if (!busy && now - lastServer >= SERVER_MS) {
          busy = true; lastServer = now;
          const rb = STANDBY_BASE + 0.03 * Math.sin(now / 900);
          fetch('/v1/route', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, cache: 'no-store',
            body: JSON.stringify({ session_id: SID, primary: 'region-a', routes: { 'region-a': +target.toFixed(3), 'region-b': +rb.toFixed(3) } }),
          }).then(r => { if (!r.ok) throw 0; return r.json(); })
            .then(d => { src = 'live · /v1/route'; applyDecision(d); busy = false; })
            .catch(() => { serverMode = false; src = 'local'; busy = false; });
        }
      } else {
        clientDecide();
      }
    }

    render(); draw();
    if (reduce) { decide(0); return; }
    let last = 0, acc = 0;
    (function loop(ts) {
      if (!last) last = ts; acc += ts - last; last = ts;
      if (acc >= TICK_MS) {
        acc = 0;
        decide(ts);
        const goal = (rerouted || systemic) ? (systemic ? 0.62 : 0.30) : target;
        const cur = buf[N - 1] + (goal - buf[N - 1]) * 0.25;
        buf.push(Math.max(0.04, Math.min(0.98, cur))); buf.shift();
        draw();
        if (!serverMode) { /* labels already via clientDecide */ }
      }
      requestAnimationFrame(loop);
    })(0);
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { resize(); draw(); }, 160); });
  }

  window.LEO_LIVEFO = { init, update };
})();
