#!/usr/bin/env python3
"""
route_engine.py — LEO's real failover state machine.

This is the actual routing logic the live demo describes, now implemented as a
pure, testable engine. It consumes a per-route risk map at each tick (in
production these come from the model's per-endpoint forecast) and decides the
active route, applying:

  • least-risk backup selection   (route to the healthiest endpoint, not a fixed map)
  • pre-warm band                 (LO–HI: arm the standby, keep serving primary)
  • reroute                       (>= HI: switch to the least-risk healthy route)
  • fail-back with hysteresis     (return to primary only after it stays < LO for COOLDOWN ticks)
  • re-hop                        (if the active backup itself degrades, move to the next healthy route)
  • systemic hold + alert         (if ALL routes are >= HI, stop thrashing — hold + page on-call)

Run the built-in scenario:
    python scripts/route_engine.py
"""
from __future__ import annotations

HI = 0.55          # reroute threshold
LO = 0.35          # pre-warm / fail-back threshold
COOLDOWN = 3       # consecutive healthy ticks on primary before fail-back


class RouteEngine:
    def __init__(self, routes, primary, hi=HI, lo=LO, cooldown=COOLDOWN):
        self.routes = list(routes)
        self.primary = primary
        self.active = primary
        self.hi, self.lo, self.cooldown_n = hi, lo, cooldown
        self.state = "NORMAL"            # NORMAL · PREWARM · REROUTED · SYSTEMIC
        self._cool = cooldown
        self.alert = False

    def _healthiest(self, risks, exclude):
        """Least-risk route below HI, excluding `exclude`."""
        cands = [(r, risks[r]) for r in self.routes if r != exclude and risks.get(r, 1.0) < self.hi]
        cands.sort(key=lambda kv: kv[1])
        return cands[0][0] if cands else None

    def step(self, risks: dict) -> dict:
        self.alert = False
        prewarm = None
        all_bad = all(risks.get(r, 1.0) >= self.hi for r in self.routes)

        if all_bad:
            # nothing healthy to route to — don't flap; hold + alert
            self.state = "SYSTEMIC"; self.alert = True
            action = "hold + alert on-call (systemic — all routes degraded)"

        elif self.active == self.primary:
            ra = risks.get(self.active, 0.0)
            if ra >= self.hi:
                tgt = self._healthiest(risks, exclude=self.active)
                if tgt:
                    self.active = tgt; self.state = "REROUTED"; self._cool = self.cooldown_n
                    action = f"REROUTE → {tgt} (least-risk healthy)"
                else:
                    self.state = "SYSTEMIC"; self.alert = True
                    action = "hold + alert (no healthy backup)"
            elif ra >= self.lo:
                self.state = "PREWARM"; prewarm = self._healthiest(risks, exclude=self.active)
                action = f"pre-warm {prewarm} (armed, still serving {self.active})"
            else:
                self.state = "NORMAL"
                action = f"serve {self.active}"

        else:  # currently on a backup
            if risks.get(self.active, 0.0) >= self.hi:
                tgt = self._healthiest(risks, exclude=self.active)
                if tgt:
                    self.active = tgt; self._cool = self.cooldown_n
                    action = f"re-hop → {tgt} (backup degraded)"
                else:
                    self.state = "SYSTEMIC"; self.alert = True
                    action = "hold + alert (backup degraded, none healthy)"
            elif risks.get(self.primary, 1.0) < self.lo:
                self._cool -= 1                       # primary healthy — count down
                if self._cool <= 0:
                    self.active = self.primary; self.state = "NORMAL"
                    action = f"FAIL-BACK → {self.primary} (recovered, cooldown clear)"
                else:
                    action = f"{self.primary} recovering — fail-back in {self._cool}"
            else:
                self._cool = self.cooldown_n          # primary not healthy yet — reset cooldown
                action = f"serving {self.active} (primary still elevated)"

        return {"active": self.active, "state": self.state, "alert": self.alert,
                "prewarm": prewarm, "action": action, "risks": dict(risks)}


# ── built-in scenario that exercises every transition ────────────────────────
def _demo():
    routes = ["txn·region-a", "txn·region-b", "txn·region-c"]
    eng = RouteEngine(routes, primary="txn·region-a")
    A, B, C = routes
    # scripted per-route risk timeline (in prod these come from the model)
    timeline = [
        {A: .12, B: .14, C: .20},   # all calm
        {A: .22, B: .15, C: .19},
        {A: .41, B: .16, C: .21},   # primary enters pre-warm
        {A: .58, B: .17, C: .22},   # primary crosses HI → reroute to B (healthiest)
        {A: .67, B: .19, C: .25},   # serving B, primary still bad
        {A: .63, B: .61, C: .28},   # B degrades too → re-hop to C
        {A: .40, B: .58, C: .24},   # primary recovering (<? not yet <LO)
        {A: .30, B: .50, C: .26},   # primary < LO → cooldown 3
        {A: .28, B: .44, C: .27},   # cooldown 2
        {A: .31, B: .40, C: .25},   # cooldown 1
        {A: .29, B: .38, C: .23},   # cooldown 0 → FAIL-BACK to A
        {A: .61, B: .59, C: .57},   # ALL bad → systemic hold + alert
        {A: .58, B: .56, C: .60},   # still systemic
        {A: .30, B: .33, C: .40},   # recovered → serve primary
    ]
    print(f"{'t':>3} | {'A':>5} {'B':>5} {'C':>5} | {'active':<13} {'state':<9} action")
    print("-" * 78)
    for t, risks in enumerate(timeline):
        d = eng.step(risks)
        flag = "  🔔" if d["alert"] else ""
        print(f"{t:>3} | {risks[A]:>5.2f} {risks[B]:>5.2f} {risks[C]:>5.2f} | "
              f"{d['active']:<13} {d['state']:<9} {d['action']}{flag}")


if __name__ == "__main__":
    try:
        import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _demo()
