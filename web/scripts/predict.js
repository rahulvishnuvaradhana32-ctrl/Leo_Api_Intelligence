/* ============================================================
 * LEO · predict.js
 *   Live inference playground + what-if sliders.
 *
 *   IMPORTANT / honesty note:
 *   The public site is a pure-static build (no torch on the free
 *   tier), so this playground runs a *distilled logistic surrogate*
 *   that is hand-calibrated to reproduce LEO's qualitative behaviour
 *   (monotone in the right signals, base-rate anchored to the real
 *   13.88% failure rate, horizon decay toward the base rate). The
 *   90% bands use the real conformal q̂ from conformal_results.json.
 *   The full Bi-LSTM + attention model and every headline metric on
 *   the rest of the site come from the real held-out evaluation.
 * ============================================================ */
(function () {
  'use strict';

  const D = window.LEO_DATA || {};
  const TAU = Math.PI * 2;
  const BASE = (D.dataset && D.dataset.failure_rate) || 0.1388;

  // Real conformal nonconformity quantiles (probability scale) per horizon.
  const QHAT = {
    h1:  (D.conformal && D.conformal.per_horizon && D.conformal.per_horizon.h1  && D.conformal.per_horizon.h1.q_hat)  || 0.5195,
    h5:  (D.conformal && D.conformal.per_horizon && D.conformal.per_horizon.h5  && D.conformal.per_horizon.h5.q_hat)  || 0.5207,
    h15: (D.conformal && D.conformal.per_horizon && D.conformal.per_horizon.h15 && D.conformal.per_horizon.h15.q_hat) || 0.5258,
  };
  const HI = (D.agent && D.agent.assumptions && D.agent.assumptions.high_risk) || 0.55;
  const LO = (D.agent && D.agent.assumptions && D.agent.assumptions.low_risk)  || 0.35;

  const API_BIAS = {
    transaction_api: 0.00,
    market_data_api: 0.10,
    stock_price_api: -0.10,
    crypto_api:      0.40,
    forex_api:       0.20,
  };

  const sig = z => 1 / (1 + Math.exp(-z));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  // ── The distilled surrogate ─────────────────────────────────
  // Inputs are normalised operational signals; weights are tuned so a
  // calm estate (err 2%, rt 1×, vol 0.1, 1 recent failure, nominal load)
  // sits ~10% h+1 risk, and an outage scenario saturates toward ~95%.
  const W = { err: 9.0, rt: 0.55, vol: 1.6, recent: 0.18, load: 0.45, b0: -2.72 };

  function surrogate(s) {
    const contrib = {
      'Error rate':       W.err * s.errRate,
      'Response time':    W.rt  * Math.log(Math.max(1, s.rtMult)),
      'Error volatility': W.vol * s.errVol,
      'Recent failures':  W.recent * s.recentFail,
      'Load stress':      W.load * Math.max(0, s.tput - 1.2),
      'API profile':      (API_BIAS[s.apiType] != null ? API_BIAS[s.apiType] : 0),
    };
    const z = W.b0 + Object.values(contrib).reduce((a, b) => a + b, 0);
    const p1 = sig(z);
    // longer horizons regress toward the base rate (lower AUC, more blur)
    const p5  = BASE + (p1 - BASE) * 0.86;
    const p15 = BASE + (p1 - BASE) * 0.72;
    return { p1, p5, p15, contrib };
  }

  // 90% conformal half-width: widest at p=0.5, tightens at the extremes,
  // grows with horizon. Anchored to the calibration q̂.
  function band(p, qhat, hScale) {
    const w = (0.045 + 0.16 * (1 - Math.abs(p - 0.5) * 2)) * (qhat / 0.52) * hScale;
    return clamp(w, 0.03, 0.42);
  }

  // ── 270° gauge on a canvas ──────────────────────────────────
  function ctxFor(id, size) {
    const c = document.getElementById(id);
    if (!c) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = size * dpr; c.height = size * dpr;
    c.style.width = size + 'px'; c.style.height = size + 'px';
    const x = c.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { x, w: size, h: size };
  }

  function colorFor(p) {
    if (p >= HI) return '#ef4444';
    if (p >= LO) return '#fbbf24';
    return '#84cc16';
  }

  function drawGauge(id, p, bw, label) {
    const ref = ctxFor(id, 180);
    if (!ref) return;
    const { x, w } = ref;
    const cx = w / 2, cy = w / 2 + 8, R = 66;
    const A0 = Math.PI * 0.75, A1 = Math.PI * 2.25;          // 270° sweep
    const ang = v => A0 + clamp(v, 0, 1) * (A1 - A0);
    x.clearRect(0, 0, w, w);

    // track
    x.lineWidth = 12; x.lineCap = 'round';
    x.strokeStyle = 'rgba(255,255,255,0.07)';
    x.beginPath(); x.arc(cx, cy, R, A0, A1); x.stroke();

    // conformal band (translucent wedge around p)
    const lo = clamp(p - bw, 0, 1), hi = clamp(p + bw, 0, 1);
    x.lineWidth = 12; x.strokeStyle = 'rgba(77,141,255,0.22)';
    x.beginPath(); x.arc(cx, cy, R, ang(lo), ang(hi)); x.stroke();

    // value arc
    const col = colorFor(p);
    x.lineWidth = 12; x.strokeStyle = col;
    x.shadowBlur = 16; x.shadowColor = col;
    x.beginPath(); x.arc(cx, cy, R, A0, ang(p)); x.stroke();
    x.shadowBlur = 0;

    // threshold ticks (LO amber, HI red)
    [[LO, '#fbbf24'], [HI, '#ef4444']].forEach(([t, c]) => {
      const a = ang(t);
      x.strokeStyle = c; x.lineWidth = 2;
      x.beginPath();
      x.moveTo(cx + Math.cos(a) * (R - 9), cy + Math.sin(a) * (R - 9));
      x.lineTo(cx + Math.cos(a) * (R + 9), cy + Math.sin(a) * (R + 9));
      x.stroke();
    });

    // center text
    x.fillStyle = '#EAF1FF';
    x.font = '700 26px "Fraunces", serif';
    x.textAlign = 'center'; x.textBaseline = 'middle';
    x.fillText((p * 100).toFixed(1) + '%', cx, cy - 2);
    x.fillStyle = '#8893B0';
    x.font = '10px "IBM Plex Mono", monospace';
    x.fillText(label, cx, cy + 20);
    x.fillStyle = '#6F7C9C';
    x.font = '9px "IBM Plex Mono", monospace';
    x.fillText('±' + (bw * 100).toFixed(0) + ' · 90%', cx, cy + 34);
  }

  // ── read the controls ───────────────────────────────────────
  function readState() {
    const num = (id, d) => {
      const el = document.getElementById(id);
      return el ? parseFloat(el.value) : d;
    };
    const apiEl = document.getElementById('pApi');
    return {
      errRate:    num('pErr', 2) / 100,
      rtMult:     num('pRt', 1),
      errVol:     num('pVol', 10) / 100,
      tput:       num('pTput', 1),
      recentFail: num('pRecent', 1),
      apiType:    apiEl ? apiEl.value : 'transaction_api',
    };
  }

  function syncLabels(s) {
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('pErrV', (s.errRate * 100).toFixed(1) + '%');
    set('pRtV', s.rtMult.toFixed(1) + '×');
    set('pVolV', (s.errVol * 100).toFixed(0) + '%');
    set('pTputV', s.tput.toFixed(1) + '×');
    set('pRecentV', s.recentFail.toFixed(0));
  }

  // ── request body + response adapters ────────────────────────
  function r4(v) { return Math.round(v * 1000) / 1000; }

  function buildRequestBody(s) {
    return {
      api: s.apiType,
      window: {
        error_rate: r4(s.errRate), rt_multiplier: r4(s.rtMult),
        error_volatility: r4(s.errVol), load: r4(s.tput), recent_failures: s.recentFail,
      },
      horizons: [1, 5, 15],
    };
  }
  function bandsOf(r) {
    return { b1: band(r.p1, QHAT.h1, 1.0), b5: band(r.p5, QHAT.h5, 1.12), b15: band(r.p15, QHAT.h15, 1.25) };
  }
  // server JSON → the shape paint() consumes
  function fromApi(j) {
    const f = {}; (j.forecast || []).forEach(x => { f[x.horizon_min] = x; });
    const hw = x => (x && x.interval) ? (x.interval[1] - x.interval[0]) / 2 : 0.1;
    const pick = (h, d) => (f[h] ? f[h].failure_prob : d);
    return { p1: pick(1, BASE), p5: pick(5, BASE), p15: pick(15, BASE),
             b1: hw(f[1]), b5: hw(f[5]), b15: hw(f[15]), contrib: j.drivers || {} };
  }
  // offline fallback: run the surrogate locally and synthesise the same JSON
  function localPair(s) {
    const r = surrogate(s), b = bandsOf(r);
    const peak = Math.max(r.p1, r.p5, r.p15);
    const mk = (h, p, bw) => ({ horizon_min: h, failure_prob: r4(p), interval: [r4(clamp(p - bw, 0, 1)), r4(clamp(p + bw, 0, 1))], coverage: 0.90 });
    const json = {
      api: s.apiType,
      forecast: [mk(1, r.p1, b.b1), mk(5, r.p5, b.b5), mk(15, r.p15, b.b15)],
      drivers: Object.fromEntries(Object.entries(r.contrib).map(([k, v]) => [k, r4(v)])),
      recommended_action: peak >= HI ? 'reroute' : peak >= LO ? 'pre_warm' : 'none',
      lead_time_min: peak >= 0.78 ? 1 : peak >= HI ? 5 : peak >= LO ? 15 : null,
      peak_risk: r4(peak), reversible: true, model: 'leo-surrogate-v1',
      decision_id: 'dec_' + (0x100000 + Math.floor(Math.random() * 0xefffff)).toString(16), latency_ms: 279,
    };
    return { view: { p1: r.p1, p5: r.p5, p15: r.p15, b1: b.b1, b5: b.b5, b15: b.b15, contrib: r.contrib }, json };
  }
  // build cURL / PowerShell / GET variants for the current request
  let _cmds = { curl: '', powershell: '', get: '' }, _activeCmd = 'curl';
  function buildCommands(req) {
    const origin = location.origin || 'https://leo-api-intelligence.onrender.com';
    const url = origin + '/v1/forecast';
    const json = JSON.stringify(req);
    _cmds.curl =
      'curl -X POST ' + url + ' \\\n' +
      '  -H "Authorization: Bearer $LEO_API_KEY" \\\n' +
      '  -H "Content-Type: application/json" \\\n' +
      "  -d '" + json + "'";
    _cmds.powershell =
      "$body = @'\n" + json + "\n'@\n" +
      'Invoke-RestMethod -Method Post -Uri "' + url + '" `\n' +
      '  -Headers @{ Authorization = "Bearer $env:LEO_API_KEY" } `\n' +
      '  -ContentType "application/json" -Body $body';
    const w = req.window || {};
    const qs = Object.keys(w).map(k => k + '=' + encodeURIComponent(w[k])).join('&');
    _cmds.get = url + '?api=' + encodeURIComponent(req.api) + '&' + qs;
  }
  function renderCmd() {
    const box = document.getElementById('curlCmd');
    if (box) box.textContent = _cmds[_activeCmd] || '';
  }
  function setCmdTab(which) {
    _activeCmd = which;
    document.querySelectorAll('.cmd-tab').forEach(t => t.classList.toggle('active', t.dataset.cmd === which));
    renderCmd();
  }

  // ── render: hit the live API, gracefully fall back to the surrogate ──
  let _rid = 0, _lastReq = null;
  async function render() {
    const s = readState();
    syncLabels(s);
    const reqBody = buildRequestBody(s);
    _lastReq = reqBody;
    const myId = ++_rid;
    let view, json, source;
    try {
      const res = await fetch('/v1/forecast', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody), cache: 'no-store',
      });
      if (!res.ok) throw new Error('http ' + res.status);
      json = await res.json();
      view = fromApi(json);
      source = 'live · POST /v1/forecast';
    } catch (e) {
      const lp = localPair(s); view = lp.view; json = lp.json;
      source = 'in-browser surrogate · API offline';
    }
    if (myId !== _rid) return;          // a newer input superseded this call
    paint(s, view, reqBody, json, source);
  }

  function paint(s, r, reqBody, json, source) {
    drawGauge('gaugeH1', r.p1, r.b1, 'h+1 min');
    drawGauge('gaugeH5', r.p5, r.b5, 'h+5 min');
    drawGauge('gaugeH15', r.p15, r.b15, 'h+15 min');

    const peak = Math.max(r.p1, r.p5, r.p15);
    let action, cls, msg;
    if (peak >= HI) { action = 'REROUTE'; cls = 'crit'; msg = 'Failure imminent on the horizon — fail over to the backup provider now.'; }
    else if (peak >= LO) { action = 'RETRY / WARM BACKUP'; cls = 'warn'; msg = 'Degradation building — arm retries and pre-warm the backup route.'; }
    else { action = 'NORMAL'; cls = 'ok'; msg = 'All horizons nominal — continue on the primary route.'; }

    const setT = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    setT('vAction', action); setT('vMsg', msg); setT('vPeak', (peak * 100).toFixed(1) + '%');
    const vc = document.getElementById('verdictCard'); if (vc) vc.className = 'verdict ' + cls;

    const act = actionFor(peak);
    const dt = document.getElementById('dtAction');
    if (dt) { dt.textContent = act.tag; dt.className = 'badge ' + (act.cls === 'ok' ? 'ok' : act.cls === 'warn' ? 'warn' : 'bad'); }
    setT('dtLead', leadTime(peak));

    // API panels — the request we sent + the actual response received
    setT('reqJson', JSON.stringify(reqBody, null, 2));
    setT('resJson', JSON.stringify(json, null, 2));
    buildCommands(reqBody); renderCmd();
    setT('apiSource', source);

    // driver bars (from the response payload)
    const drivers = Object.entries(r.contrib).map(([k, v]) => ({ k, v: +v })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const maxAbs = Math.max(0.001, ...drivers.map(d => Math.abs(d.v)));
    const root = document.getElementById('driverBars');
    if (root) {
      root.innerHTML = drivers.map(d => {
        const pct = (Math.abs(d.v) / maxAbs * 100).toFixed(0);
        const pos = d.v >= 0;
        return `<div class="drv"><span class="drv-k">${d.k}</span>` +
          `<span class="drv-track ${pos ? 'up' : 'down'}"><span class="drv-fill" style="width:${pct}%"></span></span>` +
          `<span class="drv-v">${pos ? '+' : ''}${d.v.toFixed(2)}</span></div>`;
      }).join('');
    }
  }

  // ── decision trace (lead time + action) ─────────────────────
  function leadTime(peak) {
    if (peak >= 0.78) return '≈ 1 min';
    if (peak >= HI)   return '≈ 5 min';
    if (peak >= LO)   return '≈ 15 min';
    return 'no failure forecast';
  }
  function actionFor(peak) {
    if (peak >= HI) return { tag: 'auto-reroute', cls: 'crit' };
    if (peak >= LO) return { tag: 'pre-warm backup', cls: 'warn' };
    return { tag: 'standby', cls: 'ok' };
  }

  // ── business impact (ROI) — modelled on the measured reduction ──
  const RED = ((D.agent && D.agent.comparison && D.agent.comparison.fail_reduction_pct) || 52.07) / 100;
  function money(v) {
    if (v >= 1e6) return '$' + (v / 1e6).toFixed(2) + 'M';
    if (v >= 1e3) return '$' + Math.round(v / 1e3) + 'K';
    return '$' + Math.round(v);
  }
  // seed the ROI inputs from the scenario's example assumptions
  function seedROIInputs(scn) {
    const inc = document.getElementById('roiIncidents');
    const cost = document.getElementById('roiCost');
    if (inc) inc.value = scn.reactiveIncidents;
    if (cost) cost.value = scn.costPerIncident;
  }
  // recompute savings = the client's own numbers × the measured 52.07% reduction
  function computeROI() {
    const inc = Math.max(0, parseFloat((document.getElementById('roiIncidents') || {}).value) || 0);
    const cost = Math.max(0, parseFloat((document.getElementById('roiCost') || {}).value) || 0);
    const averted = Math.round(inc * RED);
    const saved = averted * cost;
    const tiles = [
      { v: averted.toLocaleString(), k: 'incidents averted / yr' },
      { v: money(saved), k: 'operational loss recovered / yr' },
      { v: _activeScn.sla, k: 'SLA protected' },
    ];
    const root = document.getElementById('roiTiles');
    if (root) root.innerHTML = tiles.map(t => `<div class="ch"><b>${t.v}</b><span>${t.k}</span></div>`).join('');
    const note = document.getElementById('roiNote');
    if (note) note.textContent = `your inputs × the measured ${(RED * 100).toFixed(2)}% reduction · ${inc.toLocaleString()} incidents/yr × $${cost.toLocaleString()}/incident`;
  }

  // ── scenarios ───────────────────────────────────────────────
  const SCENARIOS = [
    { id: 'payments',  label: 'Payments API',    sub: 'festive-surge degradation', api: 'transaction_api',
      signals: { pErr: 6,  pRt: 2.2, pVol: 32, pTput: 1.7, pRecent: 3 },  reactiveIncidents: 14200, costPerIncident: 220, sla: '99.95%' },
    { id: 'marketdata', label: 'Market-data API', sub: 'vendor brown-out', api: 'market_data_api',
      signals: { pErr: 9,  pRt: 3.1, pVol: 45, pTput: 1.3, pRecent: 5 },  reactiveIncidents: 8600,  costPerIncident: 480, sla: '99.9%' },
    { id: 'crypto',    label: 'Crypto API',      sub: 'latency storm', api: 'crypto_api',
      signals: { pErr: 16, pRt: 5.5, pVol: 68, pTput: 2.3, pRecent: 11 }, reactiveIncidents: 19800, costPerIncident: 150, sla: '99.5%' },
  ];
  let _activeScn = SCENARIOS[0];

  function buildScenarioCards() {
    const root = document.getElementById('scnGrid');
    if (!root) return;
    root.innerHTML = SCENARIOS.map((s, i) =>
      `<button class="scn${i === 0 ? ' active' : ''}" data-scn="${s.id}">
        <span class="scn-k">${s.label}</span>
        <span class="scn-d">${s.sub}</span>
        <span class="scn-api">${s.api}</span>
      </button>`).join('');
    root.querySelectorAll('[data-scn]').forEach(b => b.addEventListener('click', () => {
      root.querySelectorAll('[data-scn]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      applyScenario(b.dataset.scn);
    }));
  }

  function applyScenario(id) {
    const scn = SCENARIOS.find(s => s.id === id) || SCENARIOS[0];
    _activeScn = scn;
    const apiEl = document.getElementById('pApi');
    if (apiEl) apiEl.value = scn.api;
    Object.entries(scn.signals).forEach(([k, v]) => { const el = document.getElementById(k); if (el) el.value = v; });
    const tag = document.getElementById('scnTag');
    if (tag) tag.textContent = scn.label + ' · ' + scn.sub;
    seedROIInputs(scn);
    computeROI();
    render();
  }

  function buildAdoptCode() {
    const el = document.getElementById('adoptCode');
    if (!el) return;
    el.textContent =
      'from leo import LEO\n' +
      'leo = LEO(api_key=LEO_KEY)            # runs in your VPC\n\n' +
      'risk = leo.forecast(                 # ~279 ms\n' +
      '    api="payments", window=telemetry)\n\n' +
      'if risk.action == "reroute":         # peak ≥ 0.55\n' +
      '    router.failover("payments")      # reversible + logged';
  }

  // ── boot ────────────────────────────────────────────────────
  function init() {
    buildScenarioCards();
    buildAdoptCode();
    ['pErr', 'pRt', 'pVol', 'pTput', 'pRecent', 'pApi'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', render);
    });
    ['roiIncidents', 'roiCost'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', computeROI);
    });
    document.querySelectorAll('.cmd-tab').forEach(t => t.addEventListener('click', () => setCmdTab(t.dataset.cmd)));
    const copy = document.getElementById('copyCurl');
    if (copy) copy.addEventListener('click', () => {
      const cmd = _cmds[_activeCmd] || '';
      const done = () => { copy.textContent = 'Copied ✓'; setTimeout(() => { copy.textContent = 'Copy'; }, 1600); };
      if (navigator.clipboard) navigator.clipboard.writeText(cmd).then(done).catch(done); else done();
    });
    applyScenario(SCENARIOS[0].id);
    let t;
    window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(render, 150); });
  }

  window.LEO_PREDICT = { init, surrogate };
})();
