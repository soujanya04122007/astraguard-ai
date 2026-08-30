"""
app.py
──────────────────────────────────────────────────────────────────────────────
AstraGuard AI — Streamlit Mission Control Dashboard
IBM AI Builders Challenge, August 2026: Advance Space Exploration with AI

Run with:
    streamlit run app.py

DISCLAIMER: This is an AI prototype built on SIMULATED spacecraft telemetry.
It is NOT an official NASA, ESA, or agency system and must NOT be used for
real mission operations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from anomaly_detector import load_model, score_dataframe, load_telemetry, FEATURE_COLS
from risk_engine       import assess_row, assess_dataframe, batch_summary, level_color, ENVELOPES
from ai_explanation    import ExplanationInput, generate_explanation

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AstraGuard AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — dark aerospace theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a !important;
    color: #e2e8f0 !important;
}
[data-testid="stHeader"] { background-color: #0a0e1a !important; }
section[data-testid="stSidebar"] { background-color: #0d1224 !important; }

/* ── Typography ── */
h1, h2, h3, h4 { color: #e2e8f0 !important; letter-spacing: 0.04em; }
p, li, label, span { color: #cbd5e1 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"]  { color: #94a3b8 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"]  { color: #e2e8f0 !important; font-size: 1.6rem !important; font-weight: 700; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem !important; }

/* ── Section headers ── */
.section-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3b82f6 !important;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin-bottom: 14px;
    margin-top: 8px;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.risk-LOW      { background:#14532d; color:#86efac; border:1px solid #22c55e; }
.risk-MEDIUM   { background:#451a03; color:#fbbf24; border:1px solid #f59e0b; }
.risk-HIGH     { background:#450a0a; color:#fca5a5; border:1px solid #ef4444; }
.risk-CRITICAL { background:#2e1065; color:#c4b5fd; border:1px solid #7c3aed; }

/* ── Alert boxes ── */
.alert-box {
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.alert-HIGH     { background:#1c0505; border-left:4px solid #ef4444; }
.alert-MEDIUM   { background:#1c1005; border-left:4px solid #f59e0b; }
.alert-CRITICAL { background:#140020; border-left:4px solid #7c3aed; }

/* ── Analysis card ── */
.analysis-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 18px 20px;
    font-size: 0.92rem;
    line-height: 1.75;
    color: #cbd5e1 !important;
}

/* ── Data tables ── */
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 6px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 100%);
    color: #e2e8f0;
    border: 1px solid #3b82f6;
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.2rem;
    width: 100%;
}
.stButton > button:hover { background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%); }

/* ── Emergency button ── */
.emergency-btn > button {
    background: linear-gradient(135deg, #450a0a 0%, #991b1b 100%) !important;
    border: 1px solid #ef4444 !important;
    color: #fca5a5 !important;
}

/* ── Text input ── */
.stTextInput input {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 6px;
}

/* ── Divider ── */
hr { border-color: #1e3a5f !important; }

/* ── Disclaimer banner ── */
.disclaimer {
    background: #0d1224;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.72rem;
    color: #475569 !important;
    text-align: center;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — cached so it only runs once per session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AstraGuard AI model...")
def _load_pipeline():
    return load_model()

@st.cache_data(show_spinner="Processing telemetry...")
def _load_and_score():
    df       = load_telemetry()
    pipeline = _load_pipeline()
    scored   = score_dataframe(pipeline, df)
    risk_df  = assess_dataframe(scored)
    return risk_df

pipeline = _load_pipeline()
risk_df  = _load_and_score()

# ─────────────────────────────────────────────────────────────────────────────
# Session state — emergency simulation flag + chat history
# ─────────────────────────────────────────────────────────────────────────────
if "emergency_active" not in st.session_state:
    st.session_state.emergency_active = False
if "emergency_row" not in st.session_state:
    st.session_state.emergency_row = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────────────────────────────────────
# Emergency simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_emergency_row() -> pd.Series:
    """Craft a worst-case simultaneous multi-fault telemetry row."""
    return pd.Series({
        "timestamp":         "SIMULATED EVENT",
        "temperature":       112.4,
        "battery_voltage":   19.8,
        "power_consumption": 331.0,
        "radiation_level":   11.2,
        "signal_strength":   -121.0,
        "fuel_level":        85.5,
        "solar_output":      8.7,
        "anomaly_flag":      -1,
        "anomaly_score":     -0.18,
    })

def _score_emergency(row: pd.Series):
    assessment = assess_row(row, int(row["anomaly_flag"]), float(row["anomaly_score"]))
    inp        = ExplanationInput.from_assessment(row, assessment)
    text       = generate_explanation(inp, backend="template")
    return assessment, text

# ─────────────────────────────────────────────────────────────────────────────
# Determine the "current" row — last record (or emergency override)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.emergency_active and st.session_state.emergency_row is not None:
    current_row        = st.session_state.emergency_row
    current_assessment, _  = _score_emergency(current_row)
    display_df         = risk_df  # keep chart data as normal mission data
else:
    current_row        = risk_df.iloc[-1]
    current_assessment = assess_row(
        current_row,
        int(current_row["anomaly_flag"]),
        float(current_row["anomaly_score"]),
    )
    display_df = risk_df

# Pre-compute top anomalies
anomaly_df    = risk_df[risk_df["label"] == "Anomaly"].sort_values("risk_score", ascending=False)
worst_anomaly = anomaly_df.iloc[0] if len(anomaly_df) > 0 else risk_df.iloc[-1]
worst_assess  = assess_row(worst_anomaly, int(worst_anomaly["anomaly_flag"]), float(worst_anomaly["anomaly_score"]))
worst_inp     = ExplanationInput.from_assessment(worst_anomaly, worst_assess)
worst_text    = generate_explanation(worst_inp, backend="template")

summary       = batch_summary(display_df)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart helper
# ─────────────────────────────────────────────────────────────────────────────

CHART_BG   = "#0a0e1a"
GRID_COLOR = "#1e3a5f"
LINE_NORM  = "#3b82f6"
LINE_ANOM  = "#ef4444"

def _make_telemetry_chart(df: pd.DataFrame, col: str, label: str, unit: str, env) -> go.Figure:
    """Build a Plotly time-series with anomaly markers and normal-range band."""
    normal = df[df["label"] == "Normal"]
    anom   = df[df["label"] == "Anomaly"]

    fig = go.Figure()

    # Normal-range shaded band
    fig.add_hrect(
        y0=env.min_normal, y1=env.max_normal,
        fillcolor="rgba(59,130,246,0.07)",
        line_width=0,
        annotation_text="normal", annotation_position="top left",
        annotation_font_size=9, annotation_font_color="#475569",
    )

    # Normal trace
    fig.add_trace(go.Scatter(
        x=normal["timestamp"], y=normal[col],
        mode="lines",
        line=dict(color=LINE_NORM, width=1.4),
        name="Normal",
        hovertemplate=f"%{{x}}<br>{label}: %{{y:.2f}} {unit}<extra></extra>",
    ))

    # Anomaly markers
    if len(anom) > 0:
        fig.add_trace(go.Scatter(
            x=anom["timestamp"], y=anom[col],
            mode="markers",
            marker=dict(color=LINE_ANOM, size=7, symbol="circle",
                        line=dict(color="#fca5a5", width=1)),
            name="Anomaly",
            hovertemplate=f"%{{x}}<br>{label}: %{{y:.2f}} {unit} ⚠<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color="#94a3b8", size=11),
        margin=dict(l=0, r=0, t=28, b=0),
        height=200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="#475569", tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, color="#475569",
                   tickfont=dict(size=9), title=unit),
        title=dict(text=label, font=dict(size=12, color="#94a3b8"), x=0, xanchor="left"),
    )
    return fig

def _gauge(score: int, level: str) -> go.Figure:
    """Semicircular gauge for the Mission Risk Index."""
    color = level_color(level)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(color=color, size=48), suffix=""),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#475569",
                      tickfont=dict(color="#475569", size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=CHART_BG,
            borderwidth=0,
            steps=[
                dict(range=[0,  30], color="#0f2a1a"),
                dict(range=[30, 61], color="#1c1505"),
                dict(range=[61, 81], color="#1c0505"),
                dict(range=[81,100], color="#140020"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=score),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=220, margin=dict(l=20, r=20, t=30, b=10),
        font=dict(color="#e2e8f0"),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Q&A engine
# ─────────────────────────────────────────────────────────────────────────────

def _answer(question: str, df: pd.DataFrame, s: dict) -> str:
    """Rule-based Q&A using computed telemetry and risk data."""
    q = question.lower().strip()

    anom_rows = df[df["label"] == "Anomaly"]
    n_anom    = len(anom_rows)
    top_anom  = anom_rows.sort_values("risk_score", ascending=False).head(3)

    # ── What anomalies? ──────────────────────────────────────────────────────
    if any(k in q for k in ["anomal", "detect", "found", "flagged"]):
        if n_anom == 0:
            return "No anomalies have been detected in the current telemetry dataset. All channels are within normal operating ranges."
        lines = [f"AstraGuard AI detected **{n_anom} anomalous telemetry records** across the mission timeline.\n\nTop anomaly events:"]
        for _, r in top_anom.iterrows():
            lines.append(f"- **{r['timestamp']}** — Risk {r['risk_score']}/100 ({r['risk_level']}) | {r['top_factors']}")
        return "\n".join(lines)

    # ── Why is the spacecraft at risk? ───────────────────────────────────────
    if any(k in q for k in ["risk", "danger", "threat", "why", "concern"]):
        worst = top_anom.iloc[0] if n_anom > 0 else None
        if worst is None:
            return f"Current Mission Risk Index is **{s['current_risk_score']}/100 ({s['current_risk_level']})**. All telemetry channels are within safe operating envelopes. No active risk factors identified."
        factors = worst["top_factors"]
        return (
            f"The spacecraft's peak risk score is **{s['max_risk_score']}/100 ({s['current_risk_level']} — current)**.\n\n"
            f"The primary risk driver is: **{factors}**\n\n"
            f"This was recorded at **{worst['timestamp']}** with a risk score of {worst['risk_score']}/100. "
            f"The IsolationForest model confirmed this as a statistically anomalous reading. "
            f"Please see the Mission Risk and AI Analysis sections for full details."
        )

    # ── What to investigate? ─────────────────────────────────────────────────
    if any(k in q for k in ["investigat", "check", "inspect", "look", "next step", "action", "do"]):
        if n_anom == 0:
            return "No anomalies require investigation at this time. Continue standard telemetry monitoring."
        worst = top_anom.iloc[0]
        assessment = assess_row(worst, int(worst["anomaly_flag"]), float(worst["anomaly_score"]))
        contributing = [c for c in assessment.contributions if c.is_contributing]
        if contributing:
            from ai_explanation import INVESTIGATION_ACTIONS
            steps = [INVESTIGATION_ACTIONS.get(c.channel, f"Review {c.channel} subsystem.") for c in contributing[:3]]
            numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
            return f"Based on the most critical anomaly ({worst['timestamp']}, Risk {worst['risk_score']}/100), recommended investigation steps:\n\n{numbered}"
        return "Review all telemetry channels and compare against mission baseline data."

    # ── Status / health ──────────────────────────────────────────────────────
    if any(k in q for k in ["status", "health", "nominal", "ok", "fine", "good"]):
        health = max(0, 100 - s["max_risk_score"])
        return (
            f"**Spacecraft Health: {health}%**\n\n"
            f"- Current MRI: {s['current_risk_score']}/100 ({s['current_risk_level']})\n"
            f"- Peak MRI recorded: {s['max_risk_score']}/100\n"
            f"- Total anomalies detected: {s['anomaly_count']}\n"
            f"- Normal records: {s['normal_count']}/{s['total_records']}\n\n"
            f"{'⚠️ Anomalies require operator attention.' if n_anom > 0 else '✅ All systems nominal.'}"
        )

    # ── Temperature ──────────────────────────────────────────────────────────
    if "temp" in q:
        temp_anom = anom_rows[anom_rows["top_factors"].str.contains("Thermal", case=False)] if n_anom > 0 else pd.DataFrame()
        if len(temp_anom):
            r = temp_anom.sort_values("risk_score", ascending=False).iloc[0]
            return f"A **thermal anomaly** was detected at {r['timestamp']}. Temperature reached **{r['temperature']:.1f} °C** (normal range: -5 to 30 °C). This may indicate heater runaway, solar overexposure, or cooling system failure. Verify thermal control system and spacecraft attitude."
        return f"Temperature telemetry appears nominal. Last reading: {df.iloc[-1]['temperature']:.1f} °C."

    # ── Battery ──────────────────────────────────────────────────────────────
    if any(k in q for k in ["battery", "voltage", "power bus"]):
        batt_anom = anom_rows[anom_rows["top_factors"].str.contains("Battery", case=False)] if n_anom > 0 else pd.DataFrame()
        if len(batt_anom):
            r = batt_anom.sort_values("risk_score", ascending=False).iloc[0]
            return f"A **battery voltage anomaly** was detected at {r['timestamp']}. Voltage dropped to **{r['battery_voltage']:.2f} V** (nominal: 27.0–29.5 V). This may indicate cell failure or deep discharge. Review battery state-of-charge and power bus load shedding."
        return f"Battery voltage is nominal. Last reading: {df.iloc[-1]['battery_voltage']:.2f} V."

    # ── Radiation ────────────────────────────────────────────────────────────
    if any(k in q for k in ["radiation", "rad", "particle"]):
        rad_anom = anom_rows[anom_rows["top_factors"].str.contains("Radiation", case=False)] if n_anom > 0 else pd.DataFrame()
        if len(rad_anom):
            r = rad_anom.sort_values("radiation_level", ascending=False).iloc[0]
            return f"A **radiation burst** was detected at {r['timestamp']}. Radiation reached **{r['radiation_level']:.2f} mSv/h** (normal: 0.1–1.3 mSv/h). This may indicate a solar particle event or radiation belt crossing. Consider enabling instrument safe-mode if levels persist."
        return f"Radiation levels are nominal. Last reading: {df.iloc[-1]['radiation_level']:.4f} mSv/h."

    # ── Signal / comms ───────────────────────────────────────────────────────
    if any(k in q for k in ["signal", "comm", "antenna", "link", "contact"]):
        sig_anom = anom_rows[anom_rows["top_factors"].str.contains("signal", case=False)] if n_anom > 0 else pd.DataFrame()
        if len(sig_anom):
            r = sig_anom.sort_values("signal_strength").iloc[0]
            return f"A **communication signal anomaly** was detected at {r['timestamp']}. Signal strength dropped to **{r['signal_strength']:.1f} dBm** (normal: -82 to -55 dBm). This may indicate antenna misalignment or link fade. Verify antenna pointing angles and transponder health."
        return f"Signal strength is nominal. Last reading: {df.iloc[-1]['signal_strength']:.1f} dBm."

    # ── Solar ────────────────────────────────────────────────────────────────
    if any(k in q for k in ["solar", "panel", "sun", "power gen"]):
        sol_anom = anom_rows[anom_rows["top_factors"].str.contains("Solar", case=False)] if n_anom > 0 else pd.DataFrame()
        if len(sol_anom):
            r = sol_anom.sort_values("solar_output").iloc[0]
            return f"A **solar output anomaly** was detected at {r['timestamp']}. Solar output dropped to **{r['solar_output']:.1f} W** (normal: 55–125 W). This may indicate panel occlusion or attitude drift. Confirm solar array deployment and pointing."
        return f"Solar output is nominal. Last reading: {df.iloc[-1]['solar_output']:.1f} W."

    # ── Fuel ─────────────────────────────────────────────────────────────────
    if "fuel" in q:
        last_fuel = df.iloc[-1]["fuel_level"]
        return f"Current fuel level is **{last_fuel:.1f}%**. Fuel depletes gradually throughout the mission. No anomalous depletion rate detected."

    # ── Fallback ─────────────────────────────────────────────────────────────
    return (
        f"I can answer questions about spacecraft anomalies, risk factors, telemetry channels, "
        f"and recommended investigation steps.\n\n"
        f"Try asking:\n"
        f"- *\"What anomalies were detected?\"*\n"
        f"- *\"Why is the spacecraft at risk?\"*\n"
        f"- *\"What should the operator investigate?\"*\n"
        f"- *\"What is the battery status?\"*\n"
        f"- *\"What is the radiation level?\"*"
    )


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title, col_time = st.columns([0.08, 0.72, 0.20])
with col_logo:
    st.markdown("# 🛰️")
with col_title:
    st.markdown("""
    <div style="padding-top:4px;">
      <div style="font-size:1.5rem;font-weight:900;letter-spacing:0.12em;color:#e2e8f0;">ASTRAGUARD AI</div>
      <div style="font-size:0.78rem;letter-spacing:0.18em;color:#3b82f6;text-transform:uppercase;margin-top:-2px;">Space Mission Risk &amp; Anomaly Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
with col_time:
    latest_ts = risk_df["timestamp"].iloc[-1]
    st.markdown(f"""
    <div style="text-align:right;padding-top:10px;">
      <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;">Last Telemetry</div>
      <div style="font-size:0.85rem;color:#94a3b8;font-family:monospace;">{str(latest_ts)[:16]}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="disclaimer">⚠️ PROTOTYPE — SIMULATED TELEMETRY ONLY — NOT AN OFFICIAL AGENCY SYSTEM — NOT FOR OPERATIONAL USE</div>', unsafe_allow_html=True)

# Emergency banner
if st.session_state.emergency_active:
    st.markdown("""
    <div style="background:#1c0000;border:2px solid #ef4444;border-radius:6px;padding:10px 16px;margin:10px 0;
                text-align:center;font-size:0.9rem;font-weight:700;letter-spacing:0.12em;color:#fca5a5;">
        ⚡ EMERGENCY SIMULATION ACTIVE — MULTI-FAULT EVENT INJECTED ⚡
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Mission Overview
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">01 &nbsp;&nbsp; Mission Overview</div>', unsafe_allow_html=True)

active_src     = current_assessment if st.session_state.emergency_active else None
current_score  = current_assessment.risk_score
current_level  = current_assessment.risk_level
health_pct     = max(0, 100 - summary["max_risk_score"])
n_active_anom = 1
mission_status = "EMERGENCY" if st.session_state.emergency_active else (
    "CRITICAL" if summary["critical_count"] > 0 else
    "ALERT"    if summary["high_count"]     > 0 else
    "CAUTION"  if summary["medium_count"]   > 0 else "NOMINAL"
)
status_color = {"NOMINAL":"#22c55e","CAUTION":"#f59e0b","ALERT":"#ef4444",
                "CRITICAL":"#7c3aed","EMERGENCY":"#ef4444"}.get(mission_status,"#94a3b8")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Mission Status", mission_status)
    st.markdown(f'<div style="height:3px;background:{status_color};border-radius:2px;margin-top:-14px;"></div>', unsafe_allow_html=True)
with m2:
    st.metric("Spacecraft Health", f"{health_pct}%",
              delta=f"{health_pct-100:.0f}%" if health_pct < 100 else "100%")
with m3:
    st.metric("Mission Risk Score", f"{summary['max_risk_score']} / 100")
with m4:
    level_disp = current_assessment.risk_level if st.session_state.emergency_active else (
        "CRITICAL" if summary["critical_count"] > 0 else
        "HIGH"     if summary["high_count"]     > 0 else
        "MEDIUM"   if summary["medium_count"]   > 0 else "LOW"
    )
    st.metric("Risk Level", level_disp)
with m5:
    st.metric("Active Anomalies", n_active_anom,
              delta=f"+{n_active_anom}" if n_active_anom > 0 else "0",
              delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Telemetry Monitoring
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">02 &nbsp;&nbsp; Telemetry Monitoring</div>', unsafe_allow_html=True)

CHARTS = [
    ("temperature",       "Temperature",          "°C",    ENVELOPES["temperature"]),
    ("battery_voltage",   "Battery Voltage",      "V",     ENVELOPES["battery_voltage"]),
    ("power_consumption", "Power Consumption",    "W",     ENVELOPES["power_consumption"]),
    ("radiation_level",   "Radiation Level",      "mSv/h", ENVELOPES["radiation_level"]),
    ("signal_strength",   "Signal Strength",      "dBm",   ENVELOPES["signal_strength"]),
    ("solar_output",      "Solar Output",         "W",     ENVELOPES["solar_output"]),
]

for row_pair in [CHARTS[:3], CHARTS[3:]]:
    cols = st.columns(3)
    for (col, label, unit, env), c in zip(row_pair, cols):
        with c:
            fig = _make_telemetry_chart(display_df, col, label, unit, env)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Active Anomalies
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">03 &nbsp;&nbsp; Active Anomalies</div>', unsafe_allow_html=True)

if len(anomaly_df) == 0:
    st.success("No anomalies detected. All telemetry channels within normal operating envelopes.")
else:
    if st.session_state.emergency_active:
        emerg_display = pd.DataFrame([{
            "timestamp": "SIMULATED EVENT",
            "temperature": 112.4, "battery_voltage": 19.8,
            "power_consumption": 331.0, "radiation_level": 11.2,
            "signal_strength": -121.0, "solar_output": 8.7,
            "risk_score": current_assessment.risk_score,
            "risk_level": current_assessment.risk_level,
            "top_factors": " | ".join(current_assessment.top_factors),
        }])
        display_anom = pd.concat([emerg_display, anomaly_df.head(9)[
            ["timestamp","temperature","battery_voltage","power_consumption",
             "radiation_level","signal_strength","solar_output","risk_score","risk_level","top_factors"]
        ]], ignore_index=True)
    else:
        display_anom = anomaly_df.head(10)[[
            "timestamp","temperature","battery_voltage","power_consumption",
            "radiation_level","signal_strength","solar_output","risk_score","risk_level","top_factors"
        ]]

    st.dataframe(
        display_anom.style
            .background_gradient(subset=["risk_score"], cmap="Reds")
            .set_properties(**{"color": "#e2e8f0", "background-color": "#111827"}),
        use_container_width=True, hide_index=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Mission Risk  +  Section 5 — AI Analysis  (side by side)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">04 &nbsp;&nbsp; Mission Risk &nbsp;&nbsp;&nbsp; 05 &nbsp;&nbsp; AI Mission Analysis</div>', unsafe_allow_html=True)

col_gauge, col_analysis = st.columns([0.30, 0.70])

with col_gauge:
    score_for_gauge = current_assessment.risk_score if st.session_state.emergency_active else summary["max_risk_score"]
    level_for_gauge = current_assessment.risk_level if st.session_state.emergency_active else (
        "CRITICAL" if summary["critical_count"] > 0 else
        "HIGH"     if summary["high_count"]     > 0 else
        "MEDIUM"   if summary["medium_count"]   > 0 else "LOW"
    )
    st.plotly_chart(_gauge(score_for_gauge, level_for_gauge),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<div style="text-align:center;margin-top:-16px;">'
        f'<span class="risk-badge risk-{level_for_gauge}">{level_for_gauge}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Breakdown bar chart
    level_counts = {
        "LOW":      summary["low_count"],
        "MEDIUM":   summary["medium_count"],
        "HIGH":     summary["high_count"],
        "CRITICAL": summary["critical_count"],
    }
    bar_colors = {"LOW":"#22c55e","MEDIUM":"#f59e0b","HIGH":"#ef4444","CRITICAL":"#7c3aed"}
    bar_fig = go.Figure(go.Bar(
        x=list(level_counts.keys()),
        y=list(level_counts.values()),
        marker_color=[bar_colors[k] for k in level_counts],
        text=list(level_counts.values()),
        textposition="outside",
        textfont=dict(color="#94a3b8", size=10),
    ))
    bar_fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=160, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(color="#475569", tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, visible=False),
        showlegend=False,
    )
    st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

with col_analysis:
    # Determine which analysis to show
    if st.session_state.emergency_active:
        analysis_assessment = current_assessment
        analysis_text       = generate_explanation(
            ExplanationInput.from_assessment(st.session_state.emergency_row, current_assessment),
            backend="template",
        )
        analysis_ts         = "SIMULATED EMERGENCY EVENT"
    else:
        analysis_assessment = worst_assess
        analysis_text       = worst_text
        analysis_ts         = str(worst_anomaly["timestamp"])

    level_cls = analysis_assessment.risk_level
    st.markdown(
        f'<div class="alert-box alert-{level_cls}">'
        f'<strong style="color:#fca5a5;">⚠ Anomaly Event Detected</strong> &nbsp;|&nbsp; '
        f'<span style="color:#94a3b8;font-size:0.82rem;">{analysis_ts}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Contributing factors
    if analysis_assessment.contributions:
        facs = [c for c in analysis_assessment.contributions if c.is_contributing]
        if facs:
            fcols = st.columns(min(len(facs), 3))
            for fc, cc in zip(facs[:3], fcols):
                with cc:
                    sev_color = ("#ef4444" if fc.threshold_score >= 80
                                 else "#f59e0b" if fc.threshold_score >= 50
                                 else "#3b82f6")
                    st.markdown(
                        f'<div style="background:#111827;border:1px solid #1e3a5f;border-left:3px solid {sev_color};'
                        f'border-radius:6px;padding:10px 12px;margin-bottom:8px;">'
                        f'<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;">{fc.channel.replace("_"," ")}</div>'
                        f'<div style="font-size:1.2rem;font-weight:700;color:{sev_color};">{fc.value} <span style="font-size:0.75rem;color:#475569;">{fc.unit}</span></div>'
                        f'<div style="font-size:0.72rem;color:#475569;margin-top:2px;">Exceedance: {fc.threshold_score:.0f}/100</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown('<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin:10px 0 6px;">AI Mission Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="analysis-card">{analysis_text}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Emergency Simulation
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">06 &nbsp;&nbsp; Emergency Simulation</div>', unsafe_allow_html=True)

col_btn, col_desc = st.columns([0.25, 0.75])
with col_btn:
    if not st.session_state.emergency_active:
        st.markdown('<div class="emergency-btn">', unsafe_allow_html=True)
        if st.button("⚡  Simulate Emergency Event", key="emerg_btn"):
            st.session_state.emergency_active = True
            st.session_state.emergency_row    = _make_emergency_row()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        if st.button("✖  Clear Emergency Simulation", key="clear_btn"):
            st.session_state.emergency_active = False
            st.session_state.emergency_row    = None
            st.rerun()

with col_desc:
    if st.session_state.emergency_active:
        ea = current_assessment
        st.markdown(
            f'<div class="alert-box alert-{ea.risk_level}">'
            f'<strong>Simulated Multi-Fault Event</strong> — '
            f'Temperature 112.4 °C | Battery 19.8 V | Power 331 W | '
            f'Radiation 11.2 mSv/h | Signal −121 dBm | Solar 8.7 W<br>'
            f'<strong>Result:</strong> MRI {ea.risk_score}/100 &nbsp;'
            f'<span class="risk-badge risk-{ea.risk_level}">{ea.risk_level}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Show pipeline: NORMAL → ANOMALY → HIGH RISK → AI ANALYSIS
        stages = [
            ("NORMAL",     "#22c55e", "Baseline telemetry within\nall normal envelopes"),
            ("ANOMALY",    "#f59e0b", "IsolationForest flags\nmulti-fault deviation"),
            ("HIGH RISK",  "#ef4444", f"MRI {ea.risk_score}/100\n{ea.risk_level} level activated"),
            ("AI ANALYSIS","#3b82f6", "Explanation & investigation\nsteps generated"),
        ]
        sc = st.columns(4)
        for (stage, clr, desc), c in zip(stages, sc):
            with c:
                st.markdown(
                    f'<div style="border:1px solid {clr};border-radius:6px;padding:10px;text-align:center;background:#0a0e1a;">'
                    f'<div style="font-size:0.75rem;font-weight:700;color:{clr};letter-spacing:0.1em;">{stage}</div>'
                    f'<div style="font-size:0.72rem;color:#64748b;margin-top:4px;white-space:pre-line;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div class="analysis-card" style="font-size:0.85rem;">'
            'Click <strong>Simulate Emergency Event</strong> to inject a simultaneous '
            'multi-fault telemetry record (thermal spike + battery drop + power surge + '
            'radiation burst + comms fade + solar drop) and watch the dashboard update '
            'through the full pipeline: '
            '<strong>NORMAL → ANOMALY → HIGH RISK → AI ANALYSIS</strong>.'
            '</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Ask AstraGuard
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">07 &nbsp;&nbsp; Ask AstraGuard</div>', unsafe_allow_html=True)

col_qa_input, col_qa_btn = st.columns([0.82, 0.18])
with col_qa_input:
    user_q = st.text_input(
        label="Ask AstraGuard",
        placeholder='e.g. "Why is the spacecraft at risk?" | "What anomalies were detected?" | "What should the operator investigate?"',
        label_visibility="collapsed",
        key="qa_input",
    )
with col_qa_btn:
    ask_clicked = st.button("Ask  ▶", key="qa_ask")

# Quick-prompt buttons
qp_cols = st.columns(4)
quick_prompts = [
    "What anomalies were detected?",
    "Why is the spacecraft at risk?",
    "What should the operator investigate?",
    "What is the battery status?",
]
for qp, c in zip(quick_prompts, qp_cols):
    with c:
        if st.button(qp, key=f"qp_{qp[:10]}"):
            st.session_state.chat_history.append({
                "q": qp,
                "a": _answer(qp, display_df, summary),
            })
            st.rerun()

if ask_clicked and user_q.strip():
    st.session_state.chat_history.append({
        "q": user_q.strip(),
        "a": _answer(user_q.strip(), display_df, summary),
    })
    st.rerun()

# Render chat history (newest first)
if st.session_state.chat_history:
    for entry in reversed(st.session_state.chat_history[-6:]):
        st.markdown(
            f'<div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:8px;'
            f'padding:12px 16px;margin-bottom:10px;">'
            f'<div style="font-size:0.8rem;color:#3b82f6;margin-bottom:4px;">▶ {entry["q"]}</div>'
            f'<div style="font-size:0.88rem;color:#cbd5e1;line-height:1.7;">{entry["a"].replace(chr(10), "<br>").replace("**", "<b>", 1).replace("**", "</b>", 1)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
else:
    st.markdown(
        '<div style="color:#475569;font-size:0.85rem;padding:10px 0;">Ask a question above or click a quick-prompt button to start.</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="disclaimer">'
    'AstraGuard AI &nbsp;|&nbsp; IBM AI Builders Challenge, August 2026 &nbsp;|&nbsp; '
    'Advance Space Exploration with AI &nbsp;|&nbsp; '
    'All telemetry is SIMULATED. This is NOT an official agency system. &nbsp;|&nbsp; '
    'Anomaly detection: IsolationForest (scikit-learn) &nbsp;|&nbsp; '
    'Risk scoring: Mission Risk Index prototype engine'
    '</div>',
    unsafe_allow_html=True,
)
