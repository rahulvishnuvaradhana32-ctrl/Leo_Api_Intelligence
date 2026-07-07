/* live.js — LEO Live Self-Healing Dashboard
 * Polls /api/proxy/status and /api/proxy/events via the web_server bridge.
 * No WebSocket needed — 2s status poll + 3s event poll covers real-time feel.
 */
(function () {
  'use strict';

  // ── DOM refs ──────────────────────────────────────────────────────────────
  var $ = function (id) { return document.getElementById(id); };

  var offline    = $('offlineBanner');
  var routeEl    = $('statusRoute');
  var cbEl       = $('statusCB');
  var primEl     = $('statusPrimary');
  var healEl     = $('statusHeal');
  var riskBar    = $('riskBar');
  var riskVal    = $('riskVal');
  var rootEl     = $('healRootCause');
  var remedyEl   = $('healRemedy');
  var canaryEl   = $('healCanary');
  var reqsEl     = $('metricReqs');
  var rtEl       = $('metricRT');
  var errEl      = $('metricErr');
  var fsEl       = $('metricFailStreak');
  var feedEl     = $('eventFeed');
  var countEl    = $('eventCount');
  var fbEl       = $('actionFeedback');

  // Heal step elements
  var steps = {
    detect:   $('step-detect'),
    reroute:  $('step-reroute'),
    diagnose: $('step-diagnose'),
    fix:      $('step-fix'),
    restore:  $('step-restore'),
  };

  // Canary segments
  var canSegs = { 10: $('can10'), 50: $('can50'), 100: $('can100') };

  // ── State ─────────────────────────────────────────────────────────────────
  var lastEventTs    = 0;   // highest ts seen so far (avoid duplicate rows)
  var seenPhases     = {};  // which heal phases we've seen in this cycle
  var isOnline       = null;

  // ── Colour helpers ────────────────────────────────────────────────────────
  function riskColor(r) {
    if (r >= 0.65) return '#ef4444';
    if (r >= 0.40) return '#f97316';
    return '#22c55e';
  }

  function colorClass(val, okVal, warnVal) {
    if (val === okVal)   return 'ok';
    if (val === warnVal) return 'warn';
    return 'error';
  }

  // ── Status polling ────────────────────────────────────────────────────────
  function pollStatus() {
    fetch('/api/proxy/status')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.offline) { setOffline(true); return; }
        setOffline(false);

        // Route
        var route = d.route || '—';
        routeEl.textContent = route;
        routeEl.className   = 'status-value ' + (route === 'primary' ? 'ok' : 'warn');

        // Circuit breaker
        var cb = (d.circuit_breaker || {}).state || '—';
        cbEl.textContent = cb;
        cbEl.className   = 'status-value ' + (cb === 'CLOSED' ? 'ok' : cb === 'HALF_OPEN' ? 'warn' : 'error');

        // Primary health
        var ph = d.primary_health;
        primEl.textContent = ph ? 'healthy' : 'down';
        primEl.className   = 'status-value ' + (ph ? 'ok' : 'error');

        // Heal active
        var ha = d.heal_cycle && d.heal_cycle.active;
        healEl.textContent = ha ? 'YES' : 'no';
        healEl.className   = 'status-value ' + (ha ? 'warn' : 'muted');

        // Risk bar
        var risk = typeof d.risk === 'number' ? d.risk : 0;
        riskBar.style.width      = (risk * 100).toFixed(1) + '%';
        riskBar.style.background = riskColor(risk);
        riskVal.textContent      = risk.toFixed(4);

        // Heal cycle details
        var hc = d.heal_cycle || {};
        rootEl.textContent   = hc.root_cause || '—';
        remedyEl.textContent = hc.remedy      || '—';

        var canary = hc.canary_stage || 0;
        canaryEl.textContent = canary ? (canary * 100).toFixed(0) + '%' : '—';
        canSegs[10].className  = 'canary-seg' + (canary >= 0.10 ? ' active' : '');
        canSegs[50].className  = 'canary-seg' + (canary >= 0.50 ? ' active' : '');
        canSegs[100].className = 'canary-seg' + (canary >= 1.00 ? ' active' : '');

        // Metrics
        reqsEl.textContent = (d.requests_served || 0).toLocaleString();
        fsEl.textContent   = ((d.circuit_breaker || {}).fail_streak || 0);

        var tel = d.telemetry || {};
        rtEl.textContent  = tel.avg_rt_s  != null ? (tel.avg_rt_s * 1000).toFixed(0) + ' ms' : '—';
        errEl.textContent = tel.error_rate != null ? (tel.error_rate * 100).toFixed(1) + '%'  : '—';
        errEl.style.color = tel.error_rate > 0.3 ? '#ef4444' : tel.error_rate > 0.05 ? '#f97316' : '#22c55e';

        // Heal pipeline step highlighting
        updateHealSteps(hc, route);
      })
      .catch(function () { setOffline(true); });
  }

  function updateHealSteps(hc, route) {
    // Clear all
    Object.values(steps).forEach(function (el) {
      el.className = 'heal-step';
    });

    if (!hc.active && route === 'primary' && hc.restored === false) {
      // idle
      return;
    }

    if (seenPhases.detect)   steps.detect.className  = 'heal-step done';
    if (seenPhases.reroute)  steps.reroute.className = 'heal-step done';
    if (seenPhases.diagnose) steps.diagnose.className= 'heal-step done';
    if (seenPhases.fix)      steps.fix.className     = 'heal-step done';
    if (seenPhases.restore)  steps.restore.className = 'heal-step done';

    // Highlight active step
    if (hc.active) {
      if (!seenPhases.restore && seenPhases.fix) {
        steps.restore.className = 'heal-step active';
      } else if (!seenPhases.fix && seenPhases.diagnose) {
        steps.fix.className = 'heal-step active';
      } else if (!seenPhases.diagnose && seenPhases.reroute) {
        steps.diagnose.className = 'heal-step active';
      } else if (!seenPhases.reroute && seenPhases.detect) {
        steps.reroute.className = 'heal-step active';
      } else if (!seenPhases.detect) {
        steps.detect.className = 'heal-step active';
      }
    }
  }

  // ── Event feed polling ────────────────────────────────────────────────────
  function pollEvents() {
    fetch('/api/proxy/events?limit=60')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var events = d.events || [];
        if (!events.length) return;

        countEl.textContent = events.length + ' events';

        // Filter to only new events
        var fresh = events.filter(function (e) { return (e.ts || 0) > lastEventTs; });
        if (!fresh.length) return;

        // Track highest ts
        fresh.forEach(function (e) {
          if (e.ts > lastEventTs) lastEventTs = e.ts;
          // Track which heal phases we've seen
          var ph = e.phase || '';
          if (ph === 'detect')    seenPhases.detect   = true;
          if (ph === 'reroute')   seenPhases.reroute  = true;
          if (ph === 'diagnose')  seenPhases.diagnose = true;
          if (ph === 'fix')       seenPhases.fix      = true;
          if (ph === 'restore' && (e.message || '').indexOf('RESTORED') >= 0) {
            seenPhases.restore = true;
          }
          if (ph === 'restore' && (e.message || '').indexOf('RESTORED') >= 0) {
            // Full restore — reset for next cycle
            setTimeout(function () { seenPhases = {}; }, 3000);
          }
        });

        // Clear placeholder
        if (feedEl.children.length === 1 &&
            feedEl.children[0].style.fontStyle === 'italic') {
          feedEl.innerHTML = '';
        }

        // Append rows (newest at bottom)
        fresh.forEach(function (e) {
          var row = document.createElement('div');
          row.className = 'event-row';

          var ts  = e.ts ? new Date(e.ts * 1000).toLocaleTimeString() : '';
          var ph  = e.phase || '?';
          var msg = e.message || JSON.stringify(e);

          row.innerHTML =
            '<span class="event-time">' + escHtml(ts) + '</span>' +
            '<span class="event-phase ' + escHtml(ph) + '">' + escHtml(ph) + '</span>' +
            '<span class="event-msg">' + escHtml(msg) + '</span>';

          feedEl.appendChild(row);
        });

        // Keep at most 80 rows
        while (feedEl.children.length > 80) {
          feedEl.removeChild(feedEl.firstChild);
        }

        // Scroll to bottom
        feedEl.scrollTop = feedEl.scrollHeight;
      })
      .catch(function () { /* silently ignore — offline banner handles it */ });
  }

  // ── Offline state ─────────────────────────────────────────────────────────
  function setOffline(flag) {
    if (flag === isOnline) return;   // no change
    isOnline = !flag;
    if (flag) {
      offline.classList.add('show');
    } else {
      offline.classList.remove('show');
    }
  }

  // ── Action buttons ────────────────────────────────────────────────────────
  function sendAction(action, label) {
    fbEl.textContent  = '⏳ sending ' + label + '…';
    fbEl.className    = 'action-feedback';

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
        } else {
          fbEl.textContent = '✓ ' + label + ' applied';
          fbEl.className   = 'action-feedback ok';
        }
        setTimeout(function () {
          fbEl.textContent = '';
          fbEl.className   = 'action-feedback';
        }, 4000);
        pollStatus();
      })
      .catch(function (err) {
        fbEl.textContent = '✗ ' + String(err);
        fbEl.className   = 'action-feedback fail';
      });
  }

  // ── Utility ───────────────────────────────────────────────────────────────
  function escHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', function () {
    // Wire buttons
    $('btnDown').addEventListener('click', function () { sendAction('inject_down', 'DB crash'); });
    $('btnTimeout').addEventListener('click', function () { sendAction('inject_timeout', 'timeout'); });
    $('btnErrorSurge').addEventListener('click', function () { sendAction('inject_error_surge', 'error surge'); });
    $('btnOverload').addEventListener('click', function () { sendAction('inject_overload', 'overload'); });
    $('btnRestore').addEventListener('click', function () { sendAction('restore', 'restore primary'); });
    $('btnReset').addEventListener('click', function () {
      seenPhases = {};
      lastEventTs = 0;
      feedEl.innerHTML = '<div style="color:#334155;font-style:italic;font-size:12px">Waiting for events…</div>';
      sendAction('reset', 'proxy reset');
    });

    // Start polling
    pollStatus();
    pollEvents();
    setInterval(pollStatus, 2000);
    setInterval(pollEvents, 3000);
  });

}());
