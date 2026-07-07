/* live.js — LEO Live Self-Healing Dashboard
 * Polls /api/proxy/status and /api/proxy/events via the web_server bridge.
 * Inject buttons fire a real traffic burst so LEO detects and self-heals.
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  var offline   = $('offlineBanner');
  var routeEl   = $('statusRoute');
  var cbEl      = $('statusCB');
  var primEl    = $('statusPrimary');
  var healEl    = $('statusHeal');
  var riskBar   = $('riskBar');
  var riskVal   = $('riskVal');
  var rootEl    = $('healRootCause');
  var remedyEl  = $('healRemedy');
  var canaryEl  = $('healCanary');
  var reqsEl    = $('metricReqs');
  var rtEl      = $('metricRT');
  var errEl     = $('metricErr');
  var fsEl      = $('metricFailStreak');
  var feedEl    = $('eventFeed');
  var countEl   = $('eventCount');
  var fbEl      = $('actionFeedback');
  var burstEl   = $('burstStatus');

  var steps = {
    detect:  $('step-detect'),
    reroute: $('step-reroute'),
    diagnose:$('step-diagnose'),
    fix:     $('step-fix'),
    restore: $('step-restore'),
  };
  var canSegs = { 10: $('can10'), 50: $('can50'), 100: $('can100') };

  var lastEventTs = 0;
  var seenPhases  = {};
  var isOnline    = null;
  var burstTimer  = null;   // interval for polling during a burst

  // ── Colour helpers ────────────────────────────────────────────────────────
  function riskColor(r) {
    if (r >= 0.65) return '#ef4444';
    if (r >= 0.40) return '#f97316';
    return '#22c55e';
  }

  // ── Status poll ───────────────────────────────────────────────────────────
  function pollStatus() {
    fetch('/api/proxy/status')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.offline) { setOffline(true); return; }
        setOffline(false);

        var route = d.route || '—';
        routeEl.textContent = route;
        routeEl.className   = 'status-value ' + (route === 'primary' ? 'ok' : 'warn');

        var cb = (d.circuit_breaker || {}).state || '—';
        cbEl.textContent = cb;
        cbEl.className   = 'status-value ' + (cb === 'CLOSED' ? 'ok' : cb === 'HALF_OPEN' ? 'warn' : 'error');

        var ph = d.primary_health;
        primEl.textContent = ph ? 'healthy' : 'DOWN';
        primEl.className   = 'status-value ' + (ph ? 'ok' : 'error');

        var ha = d.heal_cycle && d.heal_cycle.active;
        healEl.textContent = ha ? 'YES' : 'no';
        healEl.className   = 'status-value ' + (ha ? 'warn' : 'muted');

        var risk = typeof d.risk === 'number' ? d.risk : 0;
        riskBar.style.width      = (risk * 100).toFixed(1) + '%';
        riskBar.style.background = riskColor(risk);
        riskVal.textContent      = risk.toFixed(4);

        var hc = d.heal_cycle || {};
        rootEl.textContent   = hc.root_cause || '—';
        remedyEl.textContent = hc.remedy     || '—';

        var canary = hc.canary_stage || 0;
        canaryEl.textContent       = canary ? (canary * 100).toFixed(0) + '%' : '—';
        canSegs[10].className  = 'canary-seg' + (canary >= 0.10 ? ' active' : '');
        canSegs[50].className  = 'canary-seg' + (canary >= 0.50 ? ' active' : '');
        canSegs[100].className = 'canary-seg' + (canary >= 1.00 ? ' active' : '');

        reqsEl.textContent = (d.requests_served || 0).toLocaleString();
        fsEl.textContent   = ((d.circuit_breaker || {}).fail_streak || 0);

        var tel = d.telemetry || {};
        rtEl.textContent  = tel.avg_rt_s  != null ? (tel.avg_rt_s * 1000).toFixed(0) + ' ms' : '—';
        errEl.textContent = tel.error_rate != null ? (tel.error_rate * 100).toFixed(1) + '%'  : '—';
        errEl.style.color = tel.error_rate > 0.3 ? '#ef4444' : tel.error_rate > 0.05 ? '#f97316' : '#22c55e';

        updateHealSteps(hc, route);
      })
      .catch(function () { setOffline(true); });
  }

  function updateHealSteps(hc, route) {
    Object.keys(steps).forEach(function (k) { steps[k].className = 'heal-step'; });
    if (seenPhases.detect)   steps.detect.className   = 'heal-step done';
    if (seenPhases.reroute)  steps.reroute.className  = 'heal-step done';
    if (seenPhases.diagnose) steps.diagnose.className = 'heal-step done';
    if (seenPhases.fix)      steps.fix.className      = 'heal-step done';
    if (seenPhases.restore)  steps.restore.className  = 'heal-step done';
    if (hc.active) {
      if      (!seenPhases.restore && seenPhases.fix)      steps.restore.className  = 'heal-step active';
      else if (!seenPhases.fix     && seenPhases.diagnose) steps.fix.className      = 'heal-step active';
      else if (!seenPhases.diagnose && seenPhases.reroute) steps.diagnose.className = 'heal-step active';
      else if (!seenPhases.reroute  && seenPhases.detect)  steps.reroute.className  = 'heal-step active';
      else if (!seenPhases.detect)                         steps.detect.className   = 'heal-step active';
    }
  }

  // ── Event feed poll ───────────────────────────────────────────────────────
  function pollEvents() {
    fetch('/api/proxy/events?limit=80')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var events = (d.events || []);
        if (!events.length) return;
        countEl.textContent = events.length + ' events';

        var fresh = events.filter(function (e) { return (e.ts || 0) > lastEventTs; });
        if (!fresh.length) return;

        fresh.forEach(function (e) {
          if (e.ts > lastEventTs) lastEventTs = e.ts;
          var ph = e.phase || '';
          if (ph === 'detect')   seenPhases.detect   = true;
          if (ph === 'reroute')  seenPhases.reroute  = true;
          if (ph === 'diagnose') seenPhases.diagnose = true;
          if (ph === 'fix')      seenPhases.fix      = true;
          if (ph === 'restore' && (e.message || '').indexOf('RESTORED') >= 0) {
            seenPhases.restore = true;
            setTimeout(function () { seenPhases = {}; }, 5000);
          }
        });

        // Clear placeholder
        if (feedEl.children.length === 1 && feedEl.querySelector('[data-placeholder]')) {
          feedEl.innerHTML = '';
        }

        fresh.forEach(function (e) {
          var row = document.createElement('div');
          row.className = 'event-row';
          var ts  = e.ts ? new Date(e.ts * 1000).toLocaleTimeString() : '';
          var ph  = (e.phase || '?');
          var msg = e.message || JSON.stringify(e);
          row.innerHTML =
            '<span class="event-time">'  + escHtml(ts)  + '</span>' +
            '<span class="event-phase '  + escHtml(ph)  + '">' + escHtml(ph) + '</span>' +
            '<span class="event-msg">'   + escHtml(msg) + '</span>';
          feedEl.appendChild(row);
        });

        while (feedEl.children.length > 100) feedEl.removeChild(feedEl.firstChild);
        feedEl.scrollTop = feedEl.scrollHeight;
      })
      .catch(function () {});
  }

  // ── Traffic burst — fires real requests through proxy ─────────────────────
  function sendBurst(count, label) {
    if (burstTimer) clearInterval(burstTimer);
    var sent = 0;

    burstEl.textContent = '⚡ firing ' + count + ' real requests through proxy…';
    burstEl.style.color = '#60a5fa';

    // Poll faster during burst so changes show up immediately
    burstTimer = setInterval(function () {
      pollStatus();
      pollEvents();
    }, 600);

    fetch('/api/proxy/burst', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ count: count }),
    })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        clearInterval(burstTimer);
        burstTimer = null;
        var results = d.results || [];
        var errors  = results.filter(function (r) { return r.status >= 500 || r.status === 0; }).length;
        var backups  = results.filter(function (r) { return (r.backend || '').indexOf('backup') >= 0; }).length;
        burstEl.textContent =
          '✓ ' + d.sent + ' requests sent — ' +
          errors + ' errors · ' + backups + ' served by backup';
        burstEl.style.color = errors > 0 ? '#f97316' : '#22c55e';
        pollStatus();
        pollEvents();
      })
      .catch(function (e) {
        clearInterval(burstTimer);
        burstTimer = null;
        burstEl.textContent = '✗ burst failed: ' + e;
        burstEl.style.color = '#ef4444';
      });
  }

  // ── Action + burst ────────────────────────────────────────────────────────
  function injectAndBurst(action, label, burstCount) {
    fbEl.textContent = '⏳ injecting ' + label + '…';
    fbEl.className   = 'action-feedback';

    fetch('/api/proxy/action', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ action: action }),
    })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.offline || d.error) {
          fbEl.textContent = '✗ ' + (d.error || 'proxy offline');
          fbEl.className   = 'action-feedback fail';
          return;
        }
        fbEl.textContent = '✓ ' + label + ' injected — sending traffic…';
        fbEl.className   = 'action-feedback ok';
        // Fire real traffic through proxy so LEO can detect
        sendBurst(burstCount || 18, label);
        setTimeout(function () { fbEl.textContent = ''; fbEl.className = 'action-feedback'; }, 8000);
      })
      .catch(function (e) {
        fbEl.textContent = '✗ ' + e;
        fbEl.className   = 'action-feedback fail';
      });
  }

  function sendAction(action, label) {
    fbEl.textContent = '⏳ ' + label + '…';
    fbEl.className   = 'action-feedback';
    fetch('/api/proxy/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body:   JSON.stringify({ action: action }),
    })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        fbEl.textContent = d.offline ? '✗ proxy offline' : '✓ ' + label;
        fbEl.className   = 'action-feedback ' + (d.offline ? 'fail' : 'ok');
        setTimeout(function () { fbEl.textContent = ''; fbEl.className = 'action-feedback'; }, 4000);
        pollStatus();
        pollEvents();
      })
      .catch(function (e) { fbEl.textContent = '✗ ' + e; fbEl.className = 'action-feedback fail'; });
  }

  // ── Offline ───────────────────────────────────────────────────────────────
  function setOffline(flag) {
    if (flag === isOnline) return;
    isOnline = !flag;
    if (flag) offline.classList.add('show'); else offline.classList.remove('show');
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', function () {
    // Inject buttons → inject chaos THEN fire burst so LEO detects it
    $('btnDown').addEventListener('click', function () {
      injectAndBurst('inject_down', 'DB crash', 18);
    });
    $('btnTimeout').addEventListener('click', function () {
      injectAndBurst('inject_timeout', 'timeout', 18);
    });
    $('btnErrorSurge').addEventListener('click', function () {
      injectAndBurst('inject_error_surge', 'error surge', 18);
    });
    $('btnOverload').addEventListener('click', function () {
      injectAndBurst('inject_overload', 'overload', 20);
    });

    // Restore / reset — no burst needed
    $('btnRestore').addEventListener('click', function () {
      sendAction('restore', 'primary restored');
      burstEl.textContent = '';
    });
    $('btnReset').addEventListener('click', function () {
      seenPhases  = {};
      lastEventTs = 0;
      feedEl.innerHTML = '<div data-placeholder style="color:#334155;font-style:italic;font-size:12px">Waiting for events…</div>';
      burstEl.textContent = '';
      sendAction('reset', 'proxy reset');
    });

    // Start polling
    pollStatus();
    pollEvents();
    setInterval(pollStatus, 2000);
    setInterval(pollEvents, 3000);
  });

}());
