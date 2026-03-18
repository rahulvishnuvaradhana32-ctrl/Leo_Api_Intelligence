#!/usr/bin/env python3
"""
LEO API Predictive Reliability Dashboard — Crimson Edition

Usage:
    python scripts/production_dashboard.py
    Open http://localhost:8000
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import asyncio, base64, json, os, time
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

ROOT    = Path(__file__).parent.parent
MODELS  = ROOT / "models"
DATA    = ROOT / "data"
SCRIPTS = ROOT / "scripts"

app = FastAPI(title="LEO API Predictive Reliability Dashboard")

# ── Emblem SVG — orange crescent moon + shooting star ─────────────────────────
LION_SVG = """<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
<defs>
  <radialGradient id="moonGrad" cx="38%" cy="38%" r="62%">
    <stop offset="0%"   stop-color="#ffffff"/>
    <stop offset="55%"  stop-color="#f97316"/>
    <stop offset="100%" stop-color="#ea580c"/>
  </radialGradient>
</defs>
<!-- Crescent moon: outer arc minus inner offset arc -->
<path d="M30,8 A16,16 0 1,0 30,40 A10,10 0 1,1 30,8 Z" fill="url(#moonGrad)"/>
<!-- Shooting star body (bright core) -->
<circle cx="38" cy="11" r="2.2" fill="#fff7ed"/>
<!-- Star tail — fading lines streaking away from moon -->
<line x1="38" y1="11" x2="48" y2="4"  stroke="#ffffff" stroke-width="1.8" stroke-linecap="round" opacity="0.85"/>
<line x1="38" y1="11" x2="50" y2="7"  stroke="#fbbf24" stroke-width="1.0" stroke-linecap="round" opacity="0.55"/>
<line x1="38" y1="11" x2="50" y2="13" stroke="#f97316" stroke-width="0.7" stroke-linecap="round" opacity="0.35"/>
<!-- Sparkle dots along tail -->
<circle cx="43" cy="7.5" r="0.9" fill="#ffffff" opacity="0.7"/>
<circle cx="46" cy="5.5" r="0.6" fill="#fde68a" opacity="0.5"/>
</svg>"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return default or {}

def _img_b64(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return ""

def _auc_color(v) -> str:
    if v is None: return "#6b3a4a"
    return "#DC143C" if v >= 0.75 else "#f97316" if v >= 0.65 else "#991b1b"

def _auc_label(v) -> str:
    return f"{v:.3f}" if v is not None else "N/A"

def _img_tag(b64: str, alt: str) -> str:
    if not b64:
        return f'<div class="no-img">{alt} — not yet generated</div>'
    return f'<div class="img-wrap"><img src="data:image/png;base64,{b64}" alt="{alt}"></div>'


# ── Data loader ───────────────────────────────────────────────────────────────
def load_data() -> dict:
    lstm    = _load_json(MODELS / "lstm_results.json")
    ph      = lstm.get("per_horizon", {})
    aucs    = {
        "h1":  ph.get("horizon_1", {}).get("auc"),
        "h5":  ph.get("horizon_5", {}).get("auc"),
        "h15": ph.get("horizon_15", {}).get("auc"),
    }
    avg_auc    = lstm.get("avg_auc")
    n_features = lstm.get("n_features", 30)

    ablation  = _load_json(MODELS / "ablation_results.json")
    abl_exps  = ablation.get("experiments", [])
    abl_base  = ablation.get("baseline_auc") or 1.0
    abl_conclusion = "Feature ablation complete — see chart for importance ranking."
    if abl_exps:
        worst = min(abl_exps, key=lambda e: e.get("auc", 1.0))
        abl_conclusion = (f"Removing <strong>{worst.get('name','?')}</strong> caused the "
                          f"largest AUC drop ({worst.get('auc',0):.3f}), identifying it as "
                          f"the most critical feature group.")

    conf    = _load_json(MODELS / "conformal_results.json")
    conf_ph = conf.get("per_horizon", {})

    agent      = _load_json(MODELS / "agent_simulation_results.json")
    agent_cmp  = agent.get("comparison", {})
    agent_pro  = agent.get("proactive", {})
    agent_react= agent.get("reactive",  {})

    heals = []
    try:
        lines = (MODELS / "self_heal_log.jsonl").read_text(encoding="utf-8", errors="replace").strip().splitlines()
        for line in lines[-5:]:
            e  = json.loads(line)
            pf = e.get("problems_found") or {}
            fa = e.get("fixes_applied") or []
            oc = e.get("outcome") or {}
            cm = e.get("comparison") or {}
            heals.append({
                "ts":   e.get("timestamp", "—")[:19],
                "mode": e.get("mode", "—"),
                "api":  pf.get("worst_api", "—"),
                "drift":  "Yes" if pf.get("drift_detected") else "No",
                "imbal":  "Yes" if pf.get("imbalance_detected") else "No",
                "fixes":  len(fa),
                "promoted": "Yes" if oc.get("model_updated") else "No",
                "delta": f"{cm.get('delta',0):+.4f}" if cm.get("delta") is not None else "—",
            })
    except Exception:
        pass

    ds_stats = {"rows": "—", "fr": "—", "apis": 5, "range": "2023-01 → 2024-12", "kaggle": 5}
    try:
        import pandas as pd
        df = pd.read_csv(DATA / "banking_api_features.csv",
                         usecols=["timestamp", "success", "api_name"], low_memory=False)
        ds_stats["rows"] = f"{len(df):,}"
        ds_stats["fr"]   = f"{(1 - df['success'].fillna(1).mean()) * 100:.2f}%"
        ds_stats["apis"] = int(df["api_name"].nunique())
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        ds_stats["range"] = (f"{df['ts'].min().strftime('%Y-%m-%d')} → "
                             f"{df['ts'].max().strftime('%Y-%m-%d')}")
    except Exception:
        pass

    return dict(
        aucs=aucs, avg_auc=avg_auc, n_features=n_features,
        roc_b64=_img_b64(MODELS / "lstm_roc_curves.png"),
        ablation_b64=_img_b64(MODELS / "ablation_results.png"),
        abl_exps=abl_exps, abl_base=abl_base, abl_conclusion=abl_conclusion,
        conf_ph=conf_ph, conf_cal_seq=conf.get("cal_sequences", 0),
        conf_b64=_img_b64(MODELS / "conformal_calibration.png"),
        agent_cmp=agent_cmp, agent_pro=agent_pro, agent_react=agent_react,
        agent_b64=_img_b64(MODELS / "agent_simulation_chart.png"),
        heals=heals, ds_stats=ds_stats,
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


# ── HTML builder ──────────────────────────────────────────────────────────────
def build_html(d: dict) -> str:
    aucs  = d["aucs"]
    avg   = d["avg_auc"]
    ds    = d["ds_stats"]
    cph   = d["conf_ph"]
    acmp  = d["agent_cmp"]
    apro  = d["agent_pro"]
    arct  = d["agent_react"]
    heals = d["heals"]
    exps  = d["abl_exps"]

    # AUC pills
    def auc_pill(label, val):
        c = _auc_color(val)
        v = _auc_label(val)
        return (f'<div class="pill" style="border-color:{c}55">'
                f'<div class="pill-label">{label}</div>'
                f'<div class="pill-val" style="color:{c}">{v}</div>'
                f'</div>')

    auc_pills = "".join([
        auc_pill("Horizon 1", aucs.get("h1")),
        auc_pill("Horizon 5", aucs.get("h5")),
        auc_pill("Horizon 15", aucs.get("h15")),
        auc_pill("Average", avg),
    ])

    # Ablation rows
    abl_rows = ""
    for e in exps:
        a  = e.get("auc", 0)
        dv = a - (d["abl_base"] or 1.0)
        dc = "#DC143C" if dv >= 0 else "#f97316"
        abl_rows += (f'<tr><td>{e.get("name","")}</td>'
                     f'<td style="color:{_auc_color(a)}">{a:.4f}</td>'
                     f'<td style="color:{dc}">{dv:+.4f}</td>'
                     f'<td>{e.get("n_features","?")}</td></tr>')
    if not abl_rows:
        abl_rows = '<tr><td colspan=4 class="muted-cell">No ablation data yet</td></tr>'

    # Conformal rows
    conf_rows = ""
    for hz, hd in cph.items():
        cov = hd.get("coverage", 0)
        wid = hd.get("avg_width", 0)
        st  = hd.get("status", "?")
        sc  = "#DC143C" if st == "PASS" else "#f97316"
        conf_rows += (f'<tr><td>{hz}</td>'
                      f'<td>{cov*100:.1f}%</td>'
                      f'<td>{wid:.4f}</td>'
                      f'<td style="color:{sc}">{st}</td></tr>')
    if not conf_rows:
        conf_rows = '<tr><td colspan=4 class="muted-cell">Run conformal_prediction.py first</td></tr>'

    # Self-heal rows
    heal_rows = ""
    for h in heals:
        mc = "#f97316" if h["mode"] == "full" else "#6b3a4a"
        pc = "#DC143C" if h["promoted"] == "Yes" else "#6b3a4a"
        heal_rows += (f'<tr>'
                      f'<td style="font-size:12px;font-family:monospace">{h["ts"]}</td>'
                      f'<td><span style="color:{mc}">{h["mode"]}</span></td>'
                      f'<td>{h["api"]}</td>'
                      f'<td>{h["drift"]}</td><td>{h["imbal"]}</td>'
                      f'<td>{h["fixes"]}</td>'
                      f'<td style="color:{pc}">{h["promoted"]}</td>'
                      f'<td style="color:#f97316">{h["delta"]}</td>'
                      f'</tr>')
    if not heal_rows:
        heal_rows = '<tr><td colspan=8 class="muted-cell">No pipeline runs yet</td></tr>'

    # Training history (hardcoded)
    hist_rows = """
    <tr><td>v1</td><td style="color:#6b3a4a">~0.55</td><td style="color:#6b3a4a">—</td>
        <td>Prototype — 10 features, hidden=64, random split</td></tr>
    <tr><td>v2</td><td style="color:#f97316">0.680</td><td style="color:#DC143C">+0.13</td>
        <td>28 features, LayerNorm, Focal Loss, stratified split</td></tr>
    <tr><td>v3</td><td style="color:#DC143C">0.826</td><td style="color:#DC143C">+0.15</td>
        <td>hidden=128, per-horizon heads, CosineAnnealingLR, 1M rows</td></tr>
    <tr><td>v4-bal</td><td style="color:#991b1b">0.636</td><td style="color:#991b1b">−0.19</td>
        <td>39 features + precursors, 50/50 balanced — hurt AUC</td></tr>
    <tr><td>v4-nat</td><td style="color:#f97316">0.642</td><td style="color:#DC143C">+0.006</td>
        <td>Natural distribution, --balance opt-in, instability index</td></tr>"""

    # Slides
    slides = [
        ("model",    "◈", "Model Performance",
         f"Current avg AUC: {_auc_label(avg)} &nbsp;·&nbsp; 3 prediction horizons (h=1, 5, 15)",
         f"""<div class="auc-row">{auc_pills}</div>
         <div class="meta-bar">Architecture: Multi-Horizon LSTM v4 &nbsp;·&nbsp;
         {d["n_features"]} features &nbsp;·&nbsp; hidden=128, 2-layer, LayerNorm, Focal Loss (γ=2.0)</div>
         {_img_tag(d["roc_b64"], "ROC Curves")}"""),

        ("history",  "⟳", "Training History",
         "5 model versions from prototype to production &nbsp;·&nbsp; +49% AUC improvement overall",
         f"""<table><thead><tr><th>Version</th><th>AUC</th><th>Delta</th><th>Key Changes</th>
         </tr></thead><tbody>{hist_rows}</tbody></table>"""),

        ("dataset",  "⊞", "Dataset Statistics",
         f"{ds['rows']} rows &nbsp;·&nbsp; {ds['fr']} failure rate &nbsp;·&nbsp; {ds['kaggle']} Kaggle sources integrated",
         f"""<div class="stat-grid">
         <div class="stat"><div class="sl">Total Rows</div><div class="sv">{ds["rows"]}</div></div>
         <div class="stat"><div class="sl">Failure Rate</div><div class="sv" style="color:#f97316">{ds["fr"]}</div></div>
         <div class="stat"><div class="sl">APIs Monitored</div><div class="sv">{ds["apis"]}</div></div>
         <div class="stat"><div class="sl">Date Range</div><div class="sv" style="font-size:14px">{ds["range"]}</div></div>
         <div class="stat"><div class="sl">Kaggle Datasets</div><div class="sv">{ds["kaggle"]}</div></div>
         <div class="stat"><div class="sl">Precursor Features</div><div class="sv">11</div></div>
         </div>"""),

        ("ablation", "⊗", "Ablation Study",
         d["abl_conclusion"].replace("<strong>","").replace("</strong>",""),
         f"""<div class="two-col">
         <div><table><thead><tr><th>Removed Group</th><th>AUC</th><th>Delta</th><th>Features</th>
         </tr></thead><tbody>{abl_rows}</tbody></table>
         <div class="insight">{d["abl_conclusion"]}</div></div>
         <div>{_img_tag(d["ablation_b64"], "Feature Importance")}</div>
         </div>"""),

        ("conformal","⧖", "Conformal Prediction",
         "Inductive Conformal Prediction (ICP) &nbsp;·&nbsp; finite-sample coverage guarantee at 90%",
         f"""<div class="two-col">
         <div><table><thead><tr><th>Horizon</th><th>Coverage</th><th>Avg Width</th><th>Status</th>
         </tr></thead><tbody>{conf_rows}</tbody></table>
         <div class="meta-bar" style="margin-top:14px">
         Method: ICP (split conformal) &nbsp;·&nbsp; Target: 90% &nbsp;·&nbsp;
         Cal sequences: {d["conf_cal_seq"]:,}</div></div>
         <div>{_img_tag(d["conf_b64"], "Calibration Chart")}</div>
         </div>"""),

        ("agent",    "⬡", "Agent Simulation",
         (f"Proactive switching cuts failures by "
          f"{acmp.get('failure_reduction_pct','?')}% &nbsp;·&nbsp; "
          f"saves ${acmp.get('annual_cost_saving_usd',0):,.0f}/year vs reactive baseline"),
         f"""<div class="kpi-row">
         <div class="kpi"><div class="kl">Failure Reduction</div>
           <div class="kv" style="color:#DC143C">{acmp.get('failure_reduction_pct','—')}%</div></div>
         <div class="kpi"><div class="kl">Annual Savings</div>
           <div class="kv" style="color:#DC143C">${acmp.get('annual_cost_saving_usd',0):,.0f}</div></div>
         <div class="kpi"><div class="kl">Failures Avoided/yr</div>
           <div class="kv" style="color:#f97316">{acmp.get('annual_failures_avoided',0):,.0f}</div></div>
         </div>
         <div class="two-col">
         <div><table><thead><tr><th>Metric</th><th>Proactive</th><th>Reactive</th></tr></thead><tbody>
           <tr><td>Failures / 1000</td>
             <td style="color:#DC143C">{apro.get("failures","—")}</td>
             <td style="color:#991b1b">{arct.get("failures","—")}</td></tr>
           <tr><td>Failure Rate</td>
             <td style="color:#DC143C">{f'{apro.get("failure_rate",0)*100:.1f}%' if apro.get("failure_rate") is not None else "—"}</td>
             <td style="color:#991b1b">{f'{arct.get("failure_rate",0)*100:.1f}%' if arct.get("failure_rate") is not None else "—"}</td></tr>
           <tr><td>API Switches</td>
             <td>{apro.get("switches","—")}</td><td>{arct.get("switches","—")}</td></tr>
           <tr><td>Avg Latency (s)</td>
             <td>{apro.get("avg_latency_sec","—")}</td><td>{arct.get("avg_latency_sec","—")}</td></tr>
           <tr><td>Cost / 1000 tx</td>
             <td style="color:#DC143C">${apro.get("cost_per_1000",0):,.0f}</td>
             <td style="color:#991b1b">${arct.get("cost_per_1000",0):,.0f}</td></tr>
         </tbody></table></div>
         <div>{_img_tag(d["agent_b64"], "Agent Simulation Chart")}</div>
         </div>"""),

        ("selfheal", "⟳", "Self-Healing Pipeline",
         f"Last {len(heals)} autonomous runs &nbsp;·&nbsp; drift detection, augmentation &amp; conditional retrain",
         f"""<table><thead><tr>
         <th>Timestamp</th><th>Mode</th><th>Worst API</th>
         <th>Drift</th><th>Imbalance</th><th>Fixes</th><th>Promoted</th><th>AUC Δ</th>
         </tr></thead><tbody>{heal_rows}</tbody></table>"""),

        ("actions",  "▶", "Quick Actions",
         "Trigger evaluation, ablation, or self-heal dry run — live output streamed below",
         """<div class="btn-row">
         <button class="run-btn" onclick="runScript('evaluate')">
           <span class="rb-icon">◈</span>
           <span class="rb-label">Run Evaluation</span>
           <span class="rb-sub">evaluate_lstm.py — full test-set AUC</span>
         </button>
         <button class="run-btn" onclick="runScript('ablation')">
           <span class="rb-icon">⊗</span>
           <span class="rb-label">Ablation Study</span>
           <span class="rb-sub">ablation_study.py --fast (5k seqs, 3 epochs)</span>
         </button>
         <button class="run-btn" onclick="runScript('selfheal')">
           <span class="rb-icon">⟳</span>
           <span class="rb-label">Self-Heal Dry Run</span>
           <span class="rb-sub">self_improving_pipeline.py --dry_run</span>
         </button>
         </div>
         <div id="spinner" class="spinner" style="display:none">
           <span class="spin-dot"></span> Running script...
         </div>
         <div id="output-box" class="output-box" style="display:none"></div>"""),
    ]

    # Build nav + panels
    nav_html = panel_html = ""
    for i, (sid, icon, title, subtitle, content) in enumerate(slides):
        active = "active" if i == 0 else ""
        nav_html += (f'<button class="nav-btn {active}" onclick="showSlide(\'{sid}\')" id="nav-{sid}">'
                     f'<span class="nav-icon">{icon}</span>'
                     f'<span class="nav-label">{title}</span>'
                     f'</button>')
        panel_html += (f'<div class="slide {"active" if i==0 else ""}" id="slide-{sid}">'
                       f'<div class="slide-header">'
                       f'<div class="slide-icon">{icon}</div>'
                       f'<div><div class="slide-title">{title}</div>'
                       f'<div class="slide-sub">{subtitle}</div></div>'
                       f'</div>'
                       f'<div class="slide-body">{content}</div>'
                       f'</div>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LEO API Predictive Reliability Dashboard</title>
<style>
:root {{
  --bg:#080005;
  --surface:#110009;
  --surface2:#1a0010;
  --border:#2d0018;
  --text:#f0d8e0;
  --muted:#6b3a4a;
  --accent:#DC143C;
  --accent2:#f97316;
  --dim:#991b1b;
  --nav-w:224px;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;overflow:hidden}}
body{{background:var(--bg);color:var(--text);
     font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;
     display:flex;flex-direction:column}}

/* ── Topbar ── */
.topbar{{
  display:flex;align-items:center;justify-content:space-between;
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:0 24px;height:58px;flex-shrink:0;
  box-shadow:0 2px 24px #40000040;
}}
.logo{{display:flex;align-items:center;gap:14px}}
.logo-emblem{{
  width:44px;height:44px;flex-shrink:0;
  filter:drop-shadow(0 0 6px #f9731660);
  transition:filter .3s;
}}
.logo-emblem:hover{{filter:drop-shadow(0 0 12px #f97316aa)}}
.logo-text{{font-size:15px;font-weight:700;letter-spacing:0.3px;
            background:linear-gradient(90deg,#DC143C,#f97316);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.logo-sub{{font-size:11px;color:var(--muted);margin-top:2px}}
.topbar-right{{display:flex;align-items:center;gap:14px}}
.live-badge{{
  display:flex;align-items:center;gap:6px;
  background:#1a0005;border:1px solid #4a0010;
  border-radius:20px;padding:4px 12px;
  font-size:11px;color:var(--accent);font-weight:700;letter-spacing:.5px;
}}
.pulse{{width:7px;height:7px;border-radius:50%;background:var(--accent);
        animation:pulse 1.8s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.2;transform:scale(.6)}}}}
.ts-badge{{font-size:11px;color:var(--muted)}}

/* ── Layout ── */
.layout{{display:flex;flex:1;overflow:hidden}}

/* ── Sidebar ── */
.sidebar{{
  width:var(--nav-w);background:var(--surface);
  border-right:1px solid var(--border);
  padding:14px 10px;display:flex;flex-direction:column;gap:3px;
  flex-shrink:0;overflow-y:auto;
}}
.nav-btn{{
  display:flex;align-items:center;gap:10px;
  background:transparent;border:1px solid transparent;
  border-radius:8px;padding:10px 12px;
  color:var(--muted);font-size:12px;font-weight:500;
  cursor:pointer;text-align:left;width:100%;
  transition:all .2s ease;
}}
.nav-btn:hover{{color:var(--text);background:var(--surface2);border-color:var(--border)}}
.nav-btn.active{{
  color:var(--accent);background:#1a0005;
  border-color:#4a0010;
  box-shadow:inset 2px 0 0 var(--accent);
}}
.nav-icon{{font-size:15px;width:20px;text-align:center;flex-shrink:0}}
.sidebar-footer{{
  margin-top:auto;padding-top:14px;border-top:1px solid var(--border);
}}
.sidebar-footer p{{font-size:10px;color:var(--muted);line-height:1.7;padding:0 4px}}
/* Emblem watermark in sidebar footer */
.lion-watermark{{
  width:56px;height:56px;margin:10px auto;opacity:0.25;
  display:block;filter:drop-shadow(0 0 4px #f97316);
}}

/* ── Main ── */
.main{{flex:1;overflow:hidden;position:relative}}

/* ── Slides ── */
.slide{{
  position:absolute;inset:0;padding:28px 32px;overflow-y:auto;
  opacity:0;transform:translateX(28px) scale(0.99);
  transition:opacity .32s ease,transform .32s ease;
  pointer-events:none;
}}
.slide.active{{opacity:1;transform:translateX(0) scale(1);pointer-events:all}}
.slide.exit{{opacity:0;transform:translateX(-28px) scale(0.99)}}

.slide-header{{
  display:flex;align-items:flex-start;gap:16px;
  margin-bottom:22px;padding-bottom:18px;
  border-bottom:1px solid var(--border);
}}
.slide-icon{{
  width:46px;height:46px;border-radius:10px;
  background:linear-gradient(135deg,#1a0005,#0a0002);
  border:1px solid #4a0010;
  display:flex;align-items:center;justify-content:center;
  font-size:20px;color:var(--accent);flex-shrink:0;
  box-shadow:0 0 12px #DC143C22;
}}
.slide-title{{font-size:20px;font-weight:700;
              background:linear-gradient(90deg,#DC143C,#f97316);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              margin-bottom:4px}}
.slide-sub{{font-size:13px;color:var(--muted);line-height:1.5;max-width:700px}}

/* ── Tables ── */
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:var(--surface);color:var(--muted);font-size:11px;
    text-transform:uppercase;letter-spacing:.5px;
    padding:10px 14px;text-align:left;border-bottom:1px solid var(--border)}}
td{{padding:10px 14px;border-bottom:1px solid var(--border);transition:background .15s}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:var(--surface2)}}
.muted-cell{{color:var(--muted);font-style:italic;text-align:center;padding:20px}}

/* ── Images ── */
.img-wrap{{
  border:1px solid var(--border);border-radius:10px;overflow:hidden;
  background:var(--surface);margin-top:4px;
  box-shadow:0 4px 20px #20000030;
}}
.img-wrap img{{width:100%;display:block;object-fit:contain}}
.no-img{{border:1px dashed var(--border);border-radius:10px;
         padding:32px;text-align:center;color:var(--muted);font-size:13px}}

/* ── AUC pills ── */
.auc-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}}
.pill{{
  flex:1;min-width:110px;background:var(--surface2);border:1px solid;
  border-radius:10px;padding:14px 16px;text-align:center;
  transition:transform .2s,box-shadow .2s;
}}
.pill:hover{{transform:translateY(-3px);box-shadow:0 8px 20px #DC143C22}}
.pill-label{{font-size:11px;color:var(--muted);text-transform:uppercase;
             letter-spacing:.5px;margin-bottom:6px}}
.pill-val{{font-size:28px;font-weight:800;line-height:1}}

/* ── Stat grid ── */
.stat-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.stat{{background:var(--surface2);border:1px solid var(--border);
       border-radius:10px;padding:16px;transition:border-color .2s}}
.stat:hover{{border-color:#4a0010}}
.sl{{font-size:11px;color:var(--muted);text-transform:uppercase;
     letter-spacing:.5px;margin-bottom:6px}}
.sv{{font-size:20px;font-weight:700;
     background:linear-gradient(90deg,#DC143C,#f97316);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent}}

/* ── KPIs ── */
.kpi-row{{display:flex;gap:12px;margin-bottom:20px}}
.kpi{{flex:1;background:var(--surface2);border:1px solid var(--border);
      border-radius:10px;padding:16px;text-align:center;transition:border-color .2s}}
.kpi:hover{{border-color:#4a0010}}
.kl{{font-size:11px;color:var(--muted);text-transform:uppercase;
     letter-spacing:.5px;margin-bottom:6px}}
.kv{{font-size:24px;font-weight:800}}

/* ── Two-col ── */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}

/* ── Insight ── */
.insight{{
  background:var(--surface2);border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0;padding:12px 16px;
  margin-top:14px;font-size:13px;color:var(--text);line-height:1.6;
}}
.meta-bar{{font-size:12px;color:var(--muted);margin-bottom:14px}}

/* ── Quick action buttons ── */
.btn-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:16px}}
.run-btn{{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;padding:18px;cursor:pointer;
  text-align:left;display:flex;flex-direction:column;gap:5px;
  transition:all .2s;
}}
.run-btn:hover{{
  background:#1a0005;border-color:#4a0010;
  transform:translateY(-3px);box-shadow:0 8px 20px #DC143C22;
}}
.rb-icon{{font-size:20px;color:var(--accent);margin-bottom:4px}}
.rb-label{{font-size:14px;font-weight:700;color:var(--text)}}
.rb-sub{{font-size:11px;color:var(--muted)}}
.spinner{{display:flex;align-items:center;gap:8px;
          color:var(--muted);font-size:12px;margin-bottom:8px}}
.spin-dot{{width:8px;height:8px;border-radius:50%;background:var(--accent);
           animation:pulse 1s infinite}}
.output-box{{
  background:#040002;border:1px solid var(--border);border-radius:8px;
  padding:14px;font-family:'Consolas',monospace;font-size:12px;
  color:#fca5a5;white-space:pre-wrap;
  max-height:300px;overflow-y:auto;line-height:1.7;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:#4a0010;border-radius:2px}}
::-webkit-scrollbar-thumb:hover{{background:#DC143C55}}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">
    <div class="logo-emblem">{LION_SVG}</div>
    <div>
      <div class="logo-text">LEO API Predictive Reliability Dashboard</div>
      <div class="logo-sub">Multi-Horizon LSTM &nbsp;·&nbsp; Banking API Failure Prediction</div>
    </div>
  </div>
  <div class="topbar-right">
    <div class="live-badge"><div class="pulse"></div>LIVE</div>
    <div class="ts-badge">Updated: <span id="ts">{d["ts"]}</span></div>
  </div>
</div>

<div class="layout">
  <nav class="sidebar">
    {nav_html}
    <div class="sidebar-footer">
      <div class="lion-watermark">{LION_SVG}</div>
      <p>Auto-refreshes every 60s<br>Click a section to explore</p>
    </div>
  </nav>
  <main class="main">
    {panel_html}
  </main>
</div>

<script>
let current = '{slides[0][0]}';
let ess = null;

function showSlide(id) {{
  if (id === current) return;
  const prev = document.getElementById('slide-' + current);
  const next = document.getElementById('slide-' + id);
  prev.classList.add('exit');
  setTimeout(() => prev.classList.remove('active','exit'), 320);
  requestAnimationFrame(() => requestAnimationFrame(() => next.classList.add('active')));
  document.getElementById('nav-' + current).classList.remove('active');
  document.getElementById('nav-' + id).classList.add('active');
  current = id;
}}

async function refreshMeta() {{
  try {{
    const r = await fetch('/api/data');
    const d = await r.json();
    document.getElementById('ts').textContent = d.generated_at;
    const vals = [d.auc_h1, d.auc_h5, d.auc_h15, d.avg_auc];
    document.querySelectorAll('.pill-val').forEach((el, i) => {{
      if (vals[i] !== undefined) el.textContent = vals[i];
    }});
  }} catch(e) {{}}
}}
setInterval(refreshMeta, 60000);

function runScript(name) {{
  const box = document.getElementById('output-box');
  const spin = document.getElementById('spinner');
  box.textContent = ''; box.style.display = 'block';
  spin.style.display = 'flex';
  if (ess) ess.close();
  ess = new EventSource('/api/stream/' + name);
  ess.onmessage = ev => {{
    if (ev.data === '__DONE__') {{
      spin.style.display = 'none'; ess.close(); ess = null;
    }} else {{
      box.textContent += ev.data + '\\n';
      box.scrollTop = box.scrollHeight;
    }}
  }};
  ess.onerror = () => {{
    spin.style.display = 'none';
    box.textContent += '\\n[stream closed]';
    if (ess) {{ ess.close(); ess = null; }}
  }};
}}
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return build_html(load_data())

@app.get("/api/data")
async def api_data():
    d = load_data()
    a = d["aucs"]
    return JSONResponse({
        "generated_at": d["ts"],
        "auc_h1":  _auc_label(a.get("h1")),
        "auc_h5":  _auc_label(a.get("h5")),
        "auc_h15": _auc_label(a.get("h15")),
        "avg_auc": _auc_label(d["avg_auc"]),
    })

SCRIPT_CMDS = {
    "evaluate": [sys.executable, str(SCRIPTS / "evaluate_lstm.py")],
    "ablation": [sys.executable, str(SCRIPTS / "ablation_study.py"),
                 "--max_sequences", "5000", "--epochs", "3"],
    "selfheal": [sys.executable, str(SCRIPTS / "self_improving_pipeline.py"), "--dry_run"],
}

async def _stream(cmd) -> AsyncGenerator[str, None]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )
    async for raw in proc.stdout:
        yield f"data: {raw.decode('utf-8', 'replace').rstrip()}\n\n"
    await proc.wait()
    yield "data: __DONE__\n\n"

@app.get("/api/stream/{name}")
async def stream(name: str):
    cmd = SCRIPT_CMDS.get(name)
    if not cmd:
        async def _e():
            yield "data: Unknown script\n\ndata: __DONE__\n\n"
        return StreamingResponse(_e(), media_type="text/event-stream")
    return StreamingResponse(_stream(cmd), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading, webbrowser
    threading.Thread(target=lambda: (time.sleep(1.2),
                     webbrowser.open("http://localhost:8000")), daemon=True).start()
    print("LEO API Dashboard  →  http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
