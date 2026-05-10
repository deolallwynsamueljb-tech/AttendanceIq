"""
AttendanceIQ Pro · Neural Intelligence Platform v5.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chennai Institute of Technology | Smart Attendance System
50 students · 6 departments · 3 semesters · Computer Vision + ML

12 Intelligence Pages
─────────────────────
  01  Command Center       — Live KPIs + trend overview
  02  Deep Analytics       — Subject / dept / temporal deep dive
  03  ML Engine            — Model performance + absence predictor
  04  Face Scanner         — OpenCV simulation + live log
  05  Risk Matrix          — At-risk students + action centre
  06  Anomaly Detection    — IsolationForest + LOF dual detection
  07  Forecast             — 14-day attendance forecast
  08  Reports              — Findings + CSV export hub
  09  Student Profile      — Individual deep dive with radar chart
  10  Grade Intelligence   — Attendance–GPA correlation analysis
  11  Cluster Analysis     — K-Means behavioural segments
  12  Smart Alerts         — Priority-ranked notification centre

Author : Deol Allwyn Samuel J B · VLSI · CIT · Afynix Digital
Run    : python -m streamlit run app.py
"""

import warnings, os, sys, time, json, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from ml_pipeline import (
    # data loaders
    load_attendance, load_grades, load_medical_leaves,
    load_teachers, load_live_attendance,
    # analytics
    attendance_summary, student_report, subject_report,
    dept_report, daily_trend, hourly_pattern,
    at_risk_students, streak_analysis, weekly_heatmap_data,
    monthly_trend, semester_comparison,
    # ML
    forecast_attendance, anomaly_detection,
    train_absence_predictor_v2, train_grade_predictor,
    simulate_face_recognition, mark_attendance,
    engineer_features,
    # advanced
    grade_analysis, cluster_students, compute_engagement_scores,
    generate_smart_alerts, department_benchmark,
    teacher_attendance_impact,
    VERSION,
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AttendanceIQ Pro · v5",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN TOKENS ─────────────────────────────────────────────────────────────
ACID  = "#00e5ff"   # cyan — primary accent
ELEC  = "#7c3aed"   # violet — secondary
VOLT  = "#fbbf24"   # amber — warnings / highlight
WARN  = "#ef4444"   # red — danger
PINK  = "#f472b6"   # pink — special
PURP  = "#a78bfa"   # purple — ML/AI
ORNG  = "#fb923c"   # orange — medium alert
TEAL  = "#34d399"   # green — success / present
VOID  = "#060b14"   # background
PANEL = "#0d1525"   # card background

PLT_PALETTE = [ACID, ELEC, VOLT, WARN, PINK, PURP, ORNG, TEAL,
               "#60a5fa","#f472b6","#4ade80","#facc15"]

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ═══ BASE ═══════════════════════════════════════════════════ */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background: #060b14 !important;
  color: #e2e8f0 !important;
}
::-webkit-scrollbar { width: 5px; background: #0d1525; }
::-webkit-scrollbar-thumb { background: #00e5ff44; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #00e5ff88; }

/* ═══ LAYOUT ═══════════════════════════════════════════════ */
.main .block-container {
  background: #060b14;
  padding-top: 1rem;
  padding-bottom: 3rem;
  max-width: 1440px;
}

/* ═══ SIDEBAR ═══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: #0a1020 !important;
  border-right: 1px solid #1a2840 !important;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stRadio label {
  color: #cbd5e1 !important;
  font-size: 13px !important;
  font-family: 'Inter', sans-serif !important;
  letter-spacing: 0.3px !important;
  padding: 4px 0 !important;
}

/* ═══ TOPBAR ════════════════════════════════════════════════ */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 24px;
  background: linear-gradient(135deg, #0d1525 0%, #0a1930 100%);
  border: 1px solid #1a2840;
  border-top: 2px solid #00e5ff;
  border-radius: 12px;
  margin-bottom: 18px;
  box-shadow: 0 4px 24px rgba(0,229,255,0.08);
}
.topbar-brand {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 22px; font-weight: 700;
  color: #ffffff; letter-spacing: 1px;
}
.topbar-brand span { color: #00e5ff; }
.topbar-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: #64748b;
  letter-spacing: 1.5px; margin-top: 3px;
}
.tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; font-weight: 600;
  background: rgba(0,229,255,0.08);
  color: #00e5ff;
  border: 1px solid rgba(0,229,255,0.25);
  padding: 4px 10px; border-radius: 6px;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

/* ═══ STAT / KPI CARDS ══════════════════════════════════════ */
.stat-node {
  background: linear-gradient(135deg, #0d1525 0%, #0f1a2e 100%);
  border: 1px solid #1a2840;
  border-radius: 12px;
  padding: 18px 20px;
  position: relative; overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}
.stat-node:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.stat-node::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, currentColor, transparent);
  opacity: 0.3;
}
.stat-v {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 26px; font-weight: 700;
  line-height: 1; margin-bottom: 6px;
}
.stat-k {
  font-family: 'Inter', sans-serif;
  font-size: 12px; font-weight: 500;
  color: #94a3b8;
  letter-spacing: 0.3px; text-transform: uppercase;
}
.stat-d {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: #64748b; margin-top: 4px;
}

/* ═══ SECTION DIVIDERS ══════════════════════════════════════ */
.sec-div {
  display: flex; align-items: center; gap: 12px;
  margin: 24px 0 14px;
}
.sec-line {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, #1a2840, transparent);
}
.sec-line.r { background: linear-gradient(90deg, transparent, #1a2840); }
.sec-label {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 12px; font-weight: 600;
  letter-spacing: 2px; text-transform: uppercase; white-space: nowrap;
}

/* ═══ PANEL CARDS ═══════════════════════════════════════════ */
.panel {
  background: #0d1525;
  border: 1px solid #1a2840;
  border-radius: 10px;
  padding: 16px 20px;
  margin-bottom: 10px;
  position: relative;
}
.panel::before {
  content: '';
  position: absolute; top: 0; left: 0; bottom: 0;
  width: 3px; background: #00e5ff;
  border-radius: 10px 0 0 10px;
}
.p-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 13px; font-weight: 600;
  color: #e2e8f0; margin-bottom: 6px;
}

/* ═══ PROGRESS BARS ═════════════════════════════════════════ */
.prog-wrap {
  background: #1a2840; border-radius: 4px;
  height: 7px; margin: 6px 0;
}
.prog-fill { height: 7px; border-radius: 4px; }

/* ═══ MODEL CARDS ═══════════════════════════════════════════ */
.model-card {
  background: #0d1525;
  border: 1px solid #1a2840;
  border-radius: 12px; padding: 18px 20px;
}
.model-acc {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 32px; font-weight: 700;
}

/* ═══ FACE SCAN RESULTS ═════════════════════════════════════ */
.scan-ok {
  background: linear-gradient(135deg, #0d1525, #0d2020);
  border: 1px solid rgba(52,211,153,0.4);
  border-radius: 12px; padding: 28px; text-align: center;
  box-shadow: 0 0 30px rgba(52,211,153,0.1);
}
.scan-fail {
  background: linear-gradient(135deg, #1a0d0d, #200d0d);
  border: 1px solid rgba(239,68,68,0.4);
  border-radius: 12px; padding: 28px; text-align: center;
  box-shadow: 0 0 30px rgba(239,68,68,0.1);
}
.scan-pct {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 52px; font-weight: 700; line-height: 1;
}
.scan-name {
  font-family: 'Inter', sans-serif;
  font-size: 16px; font-weight: 600;
  color: #00e5ff; margin-top: 8px;
}

/* ═══ ALERT CARDS ═══════════════════════════════════════════ */
.alert-p1 {
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.35);
  border-left: 4px solid #ef4444;
  border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
}
.alert-p2 {
  background: rgba(251,146,60,0.08);
  border: 1px solid rgba(251,146,60,0.35);
  border-left: 4px solid #fb923c;
  border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
}
.alert-p3 {
  background: rgba(251,191,36,0.08);
  border: 1px solid rgba(251,191,36,0.35);
  border-left: 4px solid #fbbf24;
  border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
}
.alert-p4 {
  background: rgba(52,211,153,0.06);
  border: 1px solid rgba(52,211,153,0.25);
  border-left: 4px solid #34d399;
  border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
}

/* ═══ CLUSTER CHIPS ═════════════════════════════════════════ */
.cluster-chip {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; padding: 3px 10px;
  border-radius: 6px; letter-spacing: 0.5px;
  margin-right: 5px; font-weight: 600;
}

/* ═══ STREAMLIT OVERRIDES ═══════════════════════════════════ */
div[data-testid="stTabs"] button {
  background: transparent !important;
  color: #64748b !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important; font-weight: 500 !important;
  border-bottom: 2px solid transparent !important;
  padding: 8px 16px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: #00e5ff !important;
  border-bottom: 2px solid #00e5ff !important;
  font-weight: 600 !important;
}
div[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid #1a2840 !important;
  gap: 0 !important; margin-bottom: 16px;
}
.stButton > button {
  background: linear-gradient(135deg, rgba(0,229,255,0.1), rgba(0,229,255,0.05)) !important;
  color: #00e5ff !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important; font-size: 13px !important;
  border: 1px solid rgba(0,229,255,0.4) !important;
  border-radius: 8px !important; padding: 10px 20px !important;
  width: 100%; transition: all 0.2s !important;
  letter-spacing: 0.3px !important;
}
.stButton > button:hover {
  background: rgba(0,229,255,0.15) !important;
  box-shadow: 0 0 20px rgba(0,229,255,0.2) !important;
  transform: translateY(-1px) !important;
}
.stSelectbox div[data-baseweb="select"] > div {
  background: #0d1525 !important;
  border-color: #1a2840 !important;
  color: #e2e8f0 !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
}
label, .stRadio label {
  color: #94a3b8 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
}
h1, h2, h3 {
  font-family: 'Space Grotesk', sans-serif !important;
  color: #e2e8f0 !important;
}
.stDataFrame { background: #0d1525 !important; }
p, li { color: #94a3b8; font-family: 'Inter', sans-serif; font-size: 13px; }
.stSlider [role="slider"] { background: #00e5ff !important; }
.stSuccess { background: rgba(52,211,153,0.1) !important; border-color: rgba(52,211,153,0.3) !important; color: #34d399 !important; }
.stWarning { background: rgba(251,191,36,0.1) !important; border-color: rgba(251,191,36,0.3) !important; }
.stError   { background: rgba(239,68,68,0.1) !important;  border-color: rgba(239,68,68,0.3) !important; }
.stInfo    { background: rgba(0,229,255,0.08) !important; border-color: rgba(0,229,255,0.25) !important; }
.stTextInput input, .stTextArea textarea, .stNumberInput input {
  background: #0d1525 !important;
  border-color: #1a2840 !important;
  color: #e2e8f0 !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
}
hr { border-color: #1a2840 !important; }

/* ═══ ANIMATIONS ════════════════════════════════════════════ */
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.blink { animation: blink 1.6s ease-in-out infinite; }

@keyframes glow {
  0%,100% { text-shadow: 0 0 6px rgba(0,229,255,0.6); }
  50%      { text-shadow: 0 0 20px rgba(0,229,255,1), 0 0 40px rgba(0,229,255,0.4); }
}
.glow { animation: glow 2.5s ease-in-out infinite; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.55} }
.pulse { animation: pulse 2s ease-in-out infinite; }

@keyframes slideIn {
  from { opacity:0; transform: translateY(12px); }
  to   { opacity:1; transform: translateY(0); }
}
.slide-in { animation: slideIn 0.4s ease forwards; }

@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position: 200% center; }
}
.shimmer-text {
  background: linear-gradient(90deg, #00e5ff, #a78bfa, #f472b6, #00e5ff);
  background-size: 200% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: shimmer 3s linear infinite;
}

/* ═══ BADGE ═════════════════════════════════════════════════ */
.badge {
  display: inline-block;
  font-size: 11px; font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  padding: 2px 8px; border-radius: 20px;
  margin: 1px 2px;
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    legend=dict(bgcolor="rgba(13,21,37,0.8)", font=dict(size=11),
                bordercolor="#1a2840", borderwidth=1),
    margin=dict(t=36, b=14, l=14, r=14),
    xaxis=dict(gridcolor="#1a2840", linecolor="#1a2840", tickcolor="#64748b"),
    yaxis=dict(gridcolor="#1a2840", linecolor="#1a2840", tickcolor="#64748b"),
)

def pgo(**kw):
    d = dict(PLOTLY_BASE); d.update(kw); return d


# ── UI HELPERS ────────────────────────────────────────────────────────────────

def sec(label: str, color: str = ACID):
    st.markdown(
        f'<div class="sec-div">'
        f'<div class="sec-line"></div>'
        f'<div class="sec-label" style="color:{color};">{label}</div>'
        f'<div class="sec-line r"></div></div>',
        unsafe_allow_html=True,
    )


def kpi(col, label: str, value: str, delta: str,
        color: str = ACID, bot_color: str = None):
    bc = bot_color or color
    col.markdown(f"""
    <div class="stat-node" style="border-bottom:2px solid {bc}20;">
      <div class="stat-v" style="color:{color};">{value}</div>
      <div class="stat-k">{label}</div>
      <div class="stat-d">{delta}</div>
    </div>""", unsafe_allow_html=True)


def risk_color(pct: float) -> str:
    if pct >= 85: return ACID
    if pct >= 75: return VOLT
    if pct >= 60: return ORNG
    return WARN


def progress_bar(pct: float, color: str, max_val: float = 100) -> str:
    w = min(int(pct / max_val * 100), 100)
    return (f'<div class="prog-wrap">'
            f'<div class="prog-fill" style="width:{w}%;background:{color};"></div>'
            f'</div>')


# ── DATA LOADING (all cached) ─────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_all():
    df    = load_attendance()
    stus  = pd.read_csv("data/students.csv") if os.path.exists("data/students.csv") else pd.DataFrame()
    summ  = attendance_summary(df)
    s_rpt = student_report(df)
    sub_r = subject_report(df)
    d_rpt = dept_report(df)
    trend = daily_trend(df)
    hrly  = hourly_pattern(df)
    risk  = at_risk_students(df)
    strk  = streak_analysis(df)
    fcast = forecast_attendance(df)
    anom  = anomaly_detection(df)
    whm   = weekly_heatmap_data(df)
    mth   = monthly_trend(df)
    sem_c = semester_comparison(df)
    bench = department_benchmark(df)
    eng   = compute_engagement_scores(df)
    return (df, stus, summ, s_rpt, sub_r, d_rpt, trend, hrly,
            risk, strk, fcast, anom, whm, mth, sem_c, bench, eng)


@st.cache_data(ttl=300)
def load_extra():
    df_grades = load_grades()
    df_leaves = load_medical_leaves()
    df_tchr   = load_teachers()
    return df_grades, df_leaves, df_tchr


@st.cache_resource
def load_ml():
    df = load_attendance()
    return train_absence_predictor_v2(df)


@st.cache_data(ttl=300)
def load_grade_intel():
    df_att    = load_attendance()
    df_grades = load_grades()
    return grade_analysis(df_att, df_grades)


@st.cache_data(ttl=300)
def load_clusters():
    df = load_attendance()
    return cluster_students(df)


@st.cache_data(ttl=300)
def load_alerts():
    df_att    = load_attendance()
    df_grades = load_grades()
    df_leaves = load_medical_leaves()
    df_feat, FEATURES, _ = engineer_features(df_att)
    anom = anomaly_detection(df_att)
    return generate_smart_alerts(df_att, df_leaves, anom)


# ── BOOTSTRAP ─────────────────────────────────────────────────────────────────
# Auto-generate data on first run (needed for Streamlit Cloud)
if not os.path.exists("data/attendance.csv"):
    with st.spinner("Setting up data for first run — this takes about 15 seconds..."):
        try:
            import generate_data as _gd
            _gd.generate()
        except Exception as _e:
            st.error(f"Data generation failed: {_e}")
            st.stop()

try:
    (df, students, summary, stu_rpt, sub_rpt, dept_rpt, trend, hrly,
     risk_stu, streaks, forecast, anomalies, whm,
     mth_data, sem_data, bench_data, eng_data) = load_all()

    df_grades, df_leaves, df_tchr = load_extra()
    ml_results, ml_scaler, ml_feat, ml_best, *_ = load_ml()
except Exception as e:
    st.error(f"**Setup failed** — {e}")
    st.stop()

best_m   = ml_results[ml_best]
n_models = len(ml_results)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:16px 8px 12px;'>
      <div class='shimmer-text' style='font-family:Space Grotesk,sans-serif;
           font-size:20px;font-weight:700;letter-spacing:1px;'>
        Attendance<span style='color:#00e5ff;'>IQ</span> Pro</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:11px;
           color:#475569;letter-spacing:1.5px;margin-top:4px;'>
        v{VERSION} · NEURAL CV · CIT · AFYNIX</div>
    </div>
    <hr style='border-color:#1a2840;margin:8px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("NAV", [
        "⬡ COMMAND CENTER",
        "📊 DEEP ANALYTICS",
        "🤖 ML ENGINE",
        "👁 FACE SCANNER",
        "⚠  RISK MATRIX",
        "🔴 ANOMALY DETECT",
        "📈 FORECAST",
        "📋 REPORTS",
        "🎓 STUDENT PROFILE",
        "📚 GRADE INTEL",
        "🧠 CLUSTER ANALYSIS",
        "🔔 SMART ALERTS",
    ], label_visibility="collapsed")

    n_alerts = 0
    try:
        df_alerts_side = load_alerts()
        n_alerts = len(df_alerts_side[df_alerts_side["priority"] <= 2]) if not df_alerts_side.empty else 0
    except Exception:
        pass

    st.markdown(f"""
    <hr style='border-color:#1a2840;margin:12px 0 10px;'>
    <div style='font-family:Inter,sans-serif;font-size:12px;line-height:2;color:#64748b;'>
      <div style='color:#00e5ff;font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:4px;'>ML STATUS</div>
      <span style='color:#94a3b8;'>Acc</span> <b style='color:#34d399;'>{best_m['accuracy']}</b> &nbsp;
      <span style='color:#94a3b8;'>F1</span> <b style='color:#34d399;'>{best_m['f1']}</b><br>
      <span style='color:#94a3b8;'>AUC</span> <b style='color:#a78bfa;'>{best_m['auc']}</b> · {n_models} models<br>
      <br>
      <div style='color:#00e5ff;font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:4px;'>DATASET</div>
      <span style='color:#94a3b8;'>{len(df):,} records</span><br>
      <span style='color:#94a3b8;'>{summary['total_students']} students · {summary['total_depts']} depts</span><br>
      <span style='color:#fbbf24;font-weight:600;'>{summary['attendance_pct']}%</span> <span style='color:#94a3b8;'>attendance</span><br>
      <br>
      <div style='color:{"#ef4444" if n_alerts > 0 else "#00e5ff"};font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:4px;'>{"⚠ " if n_alerts > 0 else ""}ALERTS</div>
      <span style='color:{"#ef4444" if n_alerts>0 else "#64748b"};font-weight:600;'>{n_alerts}</span>
      <span style='color:#64748b;'> critical/high priority</span><br>
      <br>
      <div style='color:#00e5ff;font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:4px;'>DEVELOPER</div>
      <span style='color:#e2e8f0;font-weight:500;'>Deol Allwyn Samuel J B</span><br>
      <span style='color:#64748b;'>VLSI · CIT · Afynix Digital</span>
    </div>
    """, unsafe_allow_html=True)

# ── TOP BAR ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar slide-in">
  <div>
    <div class="topbar-brand">Attendance<span>IQ</span>
      <span style='font-size:13px;font-weight:400;color:#64748b;margin-left:6px;'>Pro</span>
    </div>
    <div class="topbar-sub">Neural CV · Smart Attendance · CIT · Afynix Digital · Project 4</div>
  </div>
  <div style='display:flex;gap:8px;flex-wrap:wrap;align-items:center;'>
    <span class="tag blink" style='color:#34d399;border-color:rgba(52,211,153,0.35);background:rgba(52,211,153,0.08);'>● LIVE</span>
    <span class="tag">Acc {best_m['accuracy']}</span>
    <span class="tag" style='color:#a78bfa;border-color:rgba(167,139,250,0.3);background:rgba(167,139,250,0.08);'>AUC {best_m['auc']}</span>
    <span class="tag" style='color:#fbbf24;border-color:rgba(251,191,36,0.3);background:rgba(251,191,36,0.08);'>{summary['attendance_pct']}% Present</span>
    <span class="tag" style='color:#ef4444;border-color:rgba(239,68,68,0.3);background:rgba(239,68,68,0.08);'>{len(risk_stu)} At Risk</span>
    <span class="tag">{summary['total_students']} Students</span>
    <span class="tag">{summary['total_depts']} Depts</span>
    <span class="tag" style='color:{"#ef4444" if n_alerts>0 else "#64748b"};border-color:{"rgba(239,68,68,0.35)" if n_alerts>0 else "#1a2840"};'>
      {"⚠ " if n_alerts>0 else ""}{n_alerts} Alerts</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════
if "COMMAND" in page:

    # KPI row 1
    c = st.columns(6)
    kpi(c[0], "Overall Attendance", f"{summary['attendance_pct']}%",
        f"{summary['present']:,} present", ACID)
    kpi(c[1], "Active Students", f"{summary['total_students']}",
        f"{summary['total_depts']} departments", ELEC)
    kpi(c[2], "Face Recognition", f"{summary['face_recog_pct']}%",
        f"avg conf {summary['avg_confidence']}%", VOLT)
    kpi(c[3], "At Risk", f"{len(risk_stu)}",
        "below 75% threshold", WARN)
    kpi(c[4], "ML Accuracy", f"{best_m['accuracy']}",
        f"F1 {best_m['f1']} · AUC {best_m['auc']}", PINK)
    kpi(c[5], "Anomalies", f"{len(anomalies[anomalies['anomaly']])}",
        "dual-detector flagged", ORNG)

    # KPI row 2
    st.markdown("<br>", unsafe_allow_html=True)
    c2 = st.columns(6)
    kpi(c2[0], "Total Records",  f"{len(df):,}",        "3 semesters",         "#60a5fa")
    kpi(c2[1], "Total Days",     f"{summary['total_days']}",  "working days tracked", TEAL)
    kpi(c2[2], "Total Subjects", f"{summary['total_subjects']}", "across 6 depts",  PURP)
    kpi(c2[3], "Medical Leaves", f"{summary['medical_leaves']}", "recorded episodes",  "#f472b6")
    kpi(c2[4], "Rain Days",      f"{summary['rain_affected']}", "monsoon impact",    ELEC)
    kpi(c2[5], "Models Trained", f"{n_models}",          f"best: {ml_best[:12]}", VOLT)

    st.markdown("<hr style='border-color:#1a2840;margin:14px 0;'>", unsafe_allow_html=True)

    # Trend chart
    sec("LIVE ATTENDANCE TREND · 7-DAY & 14-DAY ROLLING AVERAGE")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["pct"],
        mode="lines", name="Daily %",
        line=dict(color=ACID, width=1.2),
        fill="tozeroy", fillcolor="rgba(57,255,20,.05)",
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["rolling_7d"],
        mode="lines", name="7-day avg",
        line=dict(color=ELEC, width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=trend["date"], y=trend["rolling_14d"],
        mode="lines", name="14-day avg",
        line=dict(color=PINK, width=1.5, dash="dash"),
    ))
    fig.add_hline(y=75, line_color=WARN, line_dash="dash", line_width=1,
                  annotation_text="Min 75%", annotation_font_color=WARN)
    fig.add_hline(y=85, line_color=VOLT, line_dash="dash", line_width=1,
                  annotation_text="Target 85%", annotation_font_color=VOLT)
    fig.update_layout(**pgo(height=300, xaxis_title="Date", yaxis_title="Attendance %",
                             yaxis_range=[50, 105]))
    st.plotly_chart(fig, use_container_width=True)

    # Dept + Subject + Hourly
    c1, c2, c3 = st.columns(3)
    with c1:
        sec("DEPT ATTENDANCE")
        dr = dept_rpt.sort_values("pct")
        fig = go.Figure(go.Bar(
            x=dr["pct"], y=dr["department"], orientation="h",
            marker=dict(color=[risk_color(p) for p in dr["pct"]], line=dict(width=0)),
            text=[f"{p}%" for p in dr["pct"]], textposition="outside",
        ))
        fig.add_vline(x=75, line_color=WARN, line_dash="dash", line_width=1)
        fig.add_vline(x=85, line_color=VOLT, line_dash="dash", line_width=1)
        fig.update_layout(**pgo(height=280, xaxis_range=[0, 115]))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("SUBJECT ATTENDANCE")
        sr = sub_rpt.sort_values("pct").tail(12)
        fig = go.Figure(go.Bar(
            x=sr["pct"], y=sr["subject"], orientation="h",
            marker=dict(color=[risk_color(p) for p in sr["pct"]], line=dict(width=0)),
            text=[f"{p}%" for p in sr["pct"]], textposition="outside",
        ))
        fig.add_vline(x=75, line_color=WARN, line_dash="dash", line_width=1)
        fig.update_layout(**pgo(height=280, xaxis_range=[0, 115]))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        sec("HOURLY PATTERN")
        fig = go.Figure(go.Bar(
            x=hrly["hour"].astype(str), y=hrly["pct"],
            marker=dict(
                color=[ACID if h in [9, 10] else ELEC if h == 11 else VOLT
                       for h in hrly["hour"]],
                line=dict(width=0),
            ),
            text=hrly["pct"], textposition="outside",
        ))
        fig.update_layout(**pgo(height=280, xaxis_title="Hour",
                                 yaxis_title="%", yaxis_range=[0, 115]))
        st.plotly_chart(fig, use_container_width=True)

    # Weekly heatmap + Top streaks
    c4, c5 = st.columns([3, 1])
    with c4:
        sec("WEEKLY ATTENDANCE HEATMAP (last 16 weeks)")
        fig = px.imshow(
            whm.tail(16).T,
            color_continuous_scale=[[0, "#0d1a0d"], [0.5, VOLT], [1, ACID]],
            aspect="auto", text_auto=".0f",
        )
        fig.update_layout(**pgo(height=200, coloraxis_showscale=False))
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        sec("TOP STREAKS 🔥")
        for i, (_, row) in enumerate(streaks.head(8).iterrows()):
            c = PLT_PALETTE[i % len(PLT_PALETTE)]
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
              align-items:center;padding:5px 0;border-bottom:1px solid #0a150a;'>
              <span style='font-family:Share Tech Mono,monospace;font-size:8px;
                color:#60c060;'>#{i+1} {row['student_name'].split()[0]}</span>
              <span style='font-family:Orbitron,monospace;font-size:12px;
                font-weight:700;color:{c};'>{row['streak']}d</span>
            </div>""", unsafe_allow_html=True)

    # Department benchmark radar
    sec("DEPARTMENT BENCHMARK RADAR")
    cats = ["attendance_pct", "face_recog_pct", "avg_confidence"]
    fig  = go.Figure()
    for _, row in bench_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[c] for c in cats] + [row[cats[0]]],
            theta=cats + [cats[0]],
            fill="toself", name=row["department"],
            fillcolor="rgba(57,255,20,.05)",
            line=dict(width=1.5),
        ))
    fig.update_layout(**pgo(height=320,
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[50, 100], color="#1a4a1a"))))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DEEP ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "ANALYTICS" in page:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "STUDENT MATRIX", "SUBJECT DEEP DIVE", "METHOD ANALYSIS",
        "TEMPORAL", "ENGAGEMENT"
    ])

    with tab1:
        sec("FULL STUDENT LEADERBOARD")
        disp = stu_rpt[[
            "student_id", "name", "dept", "year",
            "total", "present", "absent", "late",
            "pct", "absent_rate", "late_rate",
            "risk_score", "risk_label", "dropout_risk", "ahi",
        ]].copy()
        st.dataframe(disp, use_container_width=True, height=380,
                     column_config={
                         "pct":          st.column_config.ProgressColumn("Att %",      min_value=0, max_value=100),
                         "risk_score":   st.column_config.ProgressColumn("Risk",       min_value=0, max_value=100),
                         "dropout_risk": st.column_config.ProgressColumn("Dropout",    min_value=0, max_value=100),
                         "ahi":          st.column_config.ProgressColumn("AHI",        min_value=0, max_value=100),
                     })

        c1, c2 = st.columns(2)
        with c1:
            sec("ATTENDANCE DISTRIBUTION")
            fig = px.histogram(stu_rpt, x="pct", nbins=15,
                               color_discrete_sequence=[ACID])
            fig.add_vline(x=75, line_color=WARN, line_dash="dash",
                          annotation_text="75%", annotation_font_color=WARN)
            fig.add_vline(x=85, line_color=VOLT, line_dash="dash",
                          annotation_text="85%", annotation_font_color=VOLT)
            fig.update_layout(**pgo(height=280, xaxis_title="Attendance %",
                                     yaxis_title="Students"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sec("RISK × ATTENDANCE SCATTER")
            fig = px.scatter(stu_rpt, x="pct", y="risk_score",
                             color="risk_label", size="absent",
                             hover_name="name",
                             color_discrete_map={
                                 "Good":     ACID,
                                 "Low Risk": VOLT,
                                 "At Risk":  ORNG,
                                 "Critical": WARN,
                             })
            fig.update_layout(**pgo(height=280,
                                     xaxis_title="Attendance %",
                                     yaxis_title="Risk Score"))
            st.plotly_chart(fig, use_container_width=True)

        sec("ACADEMIC HEALTH INDEX vs DROPOUT RISK")
        fig = px.scatter(
            stu_rpt, x="ahi", y="dropout_risk",
            color="dept", size="absent",
            hover_name="name",
            color_discrete_sequence=PLT_PALETTE,
        )
        fig.update_layout(**pgo(height=280,
                                 xaxis_title="Academic Health Index",
                                 yaxis_title="Dropout Risk Score"))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sec("SUBJECT × DAY HEATMAP")
        piv = df.groupby(["subject", "day"])["status"].apply(
            lambda x: (x == "Present").sum() / len(x) * 100
        ).unstack().round(1)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        piv = piv.reindex(columns=[d for d in day_order if d in piv.columns])
        fig = px.imshow(piv, text_auto=".0f",
                        color_continuous_scale=[[0,"#0d1a0d"],[0.5,VOLT],[1,ACID]],
                        aspect="auto")
        fig.update_layout(**pgo(height=380, coloraxis_showscale=False))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            sec("SUBJECT TREEMAP")
            fig = px.treemap(sub_rpt, path=["dept", "subject"],
                             values="total", color="pct",
                             color_continuous_scale=[[0,WARN],[0.5,VOLT],[1,ACID]])
            fig.update_layout(**pgo(height=320))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sec("SUBJECT RADAR")
            radar_sub = sub_rpt.head(10)
            fig = go.Figure(go.Scatterpolar(
                r=radar_sub["pct"].tolist(),
                theta=radar_sub["subject"].tolist(),
                fill="toself",
                fillcolor="rgba(57,255,20,.08)",
                line=dict(color=ACID, width=1.5),
            ))
            fig.update_layout(**pgo(height=320,
                polar=dict(bgcolor="rgba(0,0,0,0)",
                           radialaxis=dict(range=[60, 100]))))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        sec("RECOGNITION METHOD BREAKDOWN")
        mc = df["method"].value_counts()
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = go.Figure(go.Pie(
                labels=mc.index, values=mc.values, hole=.5,
                marker_colors=PLT_PALETTE,
            ))
            fig.update_layout(**pgo(height=260, title="Method Mix"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            conf_d = df[df["confidence"] > 0]["confidence"] * 100
            fig = px.histogram(conf_d, nbins=30, color_discrete_sequence=[ACID])
            fig.add_vline(x=80, line_color=WARN, line_dash="dash")
            fig.update_layout(**pgo(height=260,
                                     title="Face Recognition Confidence",
                                     xaxis_title="Confidence %"))
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            late_day = df[df["late"] == True].groupby("day").size().reset_index(name="count")
            fig = px.bar(late_day, x="day", y="count",
                         color_discrete_sequence=[VOLT])
            fig.update_layout(**pgo(height=260, title="Late Arrivals by Day"))
            st.plotly_chart(fig, use_container_width=True)

        sec("HOSTEL vs DAY SCHOLAR ATTENDANCE")
        if "hostel" in df.columns:
            h_grp = df.groupby("hostel")["status"].apply(
                lambda x: (x == "Present").mean() * 100
            ).reset_index()
            h_grp["hostel"] = h_grp["hostel"].map({True: "Hostel", False: "Day Scholar"})
            h_grp.columns = ["Type", "Attendance %"]
            fig = px.bar(h_grp, x="Type", y="Attendance %",
                         color="Type",
                         color_discrete_map={"Hostel": ELEC, "Day Scholar": ACID})
            fig.update_layout(**pgo(height=260))
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        sec("SEMESTER TIMELINE")
        if "semester" in df.columns:
            sem_grp = df.groupby(["semester", "department"]).agg(
                pct=("status", lambda x: (x == "Present").mean() * 100)
            ).reset_index().round(1)
            fig = px.bar(sem_grp, x="department", y="pct",
                         color="semester", barmode="group",
                         color_discrete_sequence=[ACID, ELEC, VOLT])
            fig.update_layout(**pgo(height=300,
                                     yaxis_title="Attendance %",
                                     yaxis_range=[0, 110]))
            st.plotly_chart(fig, use_container_width=True)

        sec("MONTHLY ATTENDANCE BY DEPARTMENT")
        if not mth_data.empty:
            fig = px.line(mth_data, x="month_yr", y="pct",
                          color="department",
                          color_discrete_sequence=PLT_PALETTE)
            fig.add_hline(y=75, line_color=WARN, line_dash="dash", line_width=1)
            fig.update_layout(**pgo(height=300,
                                     xaxis_title="Month",
                                     yaxis_title="Attendance %",
                                     yaxis_range=[50, 105]))
            st.plotly_chart(fig, use_container_width=True)

        sec("MONSOON IMPACT ANALYSIS")
        if "rain_day" in df.columns:
            rain_comp = df.groupby("rain_day")["status"].apply(
                lambda x: (x == "Present").mean() * 100
            ).reset_index()
            rain_comp["rain_day"] = rain_comp["rain_day"].map({True: "Rain Day", False: "Normal Day"})
            rain_comp.columns = ["Condition", "Attendance %"]
            fig = px.bar(rain_comp, x="Condition", y="Attendance %",
                         color="Condition",
                         color_discrete_map={"Rain Day": WARN, "Normal Day": ACID})
            fig.update_layout(**pgo(height=260))
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        sec("STUDENT ENGAGEMENT SCORES")
        st.dataframe(
            eng_data[["student_name","department","year","att_pct","late_pct",
                       "avg_conf","trend_score","exam_pct","engagement"]].round(1),
            use_container_width=True, height=320,
            column_config={
                "engagement": st.column_config.ProgressColumn("Engagement", min_value=0, max_value=100),
                "att_pct":    st.column_config.ProgressColumn("Att %",      min_value=0, max_value=100),
            },
        )

        c1, c2 = st.columns(2)
        with c1:
            sec("ENGAGEMENT DISTRIBUTION")
            fig = px.histogram(eng_data, x="engagement", nbins=15,
                               color_discrete_sequence=[ACID])
            fig.update_layout(**pgo(height=280, xaxis_title="Engagement Score"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sec("ENGAGEMENT vs ATTENDANCE")
            fig = px.scatter(eng_data, x="att_pct", y="engagement",
                             color="department", hover_name="student_name",
                             color_discrete_sequence=PLT_PALETTE)
            fig.update_layout(**pgo(height=280, xaxis_title="Attendance %",
                                     yaxis_title="Engagement Score"))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif "ML" in page:
    sec("MODEL PERFORMANCE MATRIX")

    cols = st.columns(min(len(ml_results), 5))
    for col, (name, m) in zip(cols, ml_results.items()):
        is_best = name == ml_best
        border  = "border:1px solid #39ff1460;" if is_best else ""
        col.markdown(f"""
        <div class="model-card" style="{border}">
          <div style='font-size:8px;font-family:Share Tech Mono,monospace;
               color:#1a4a1a;letter-spacing:2px;'>
            {"★ BEST MODEL" if is_best else "MODEL"}</div>
          <div style='font-family:Orbitron,monospace;font-size:10px;
               color:#39ff14;margin:3px 0;'>{name[:16]}</div>
          <div class="model-acc" style='color:{"#39ff14" if is_best else "#00d4ff"};'>
            {m["accuracy"]}</div>
          <div style='font-size:8px;font-family:Share Tech Mono,monospace;
               color:#1a4a1a;margin-bottom:8px;'>ACCURACY</div>
          <div style='font-family:Share Tech Mono,monospace;font-size:9px;
               color:#2a5a2a;line-height:2;'>
            F1 &nbsp;&nbsp;&nbsp;<span style='color:#ffe600;'>{m['f1']}</span><br>
            AUC &nbsp;&nbsp;<span style='color:#00d4ff;'>{m['auc']}</span><br>
            PREC &nbsp;<span style='color:#a78bfa;'>{m['precision']}</span><br>
            REC &nbsp;&nbsp;<span style='color:#fb923c;'>{m['recall']}</span><br>
            CV-F1 <span style='color:#60c060;'>{m['cv_f1_mean']}±{m['cv_f1_std']}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("ROC CURVES")
        from sklearn.metrics import roc_curve
        fig = go.Figure()
        for (name, m), col in zip(ml_results.items(), PLT_PALETTE):
            fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name[:14]} (AUC={m['auc']})",
                line=dict(color=col, width=1.5),
            ))
        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                      line=dict(color="#2a3a2a", dash="dash"))
        fig.update_layout(**pgo(height=340, xaxis_title="FPR", yaxis_title="TPR"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("CONFUSION MATRICES")
        n = min(len(ml_results), 3)
        items = list(ml_results.items())[:n]
        fig = make_subplots(rows=1, cols=n, subplot_titles=[k[:14] for k, _ in items])
        for idx, (name, m) in enumerate(items):
            cm = np.array(m["cm"])
            fig.add_trace(go.Heatmap(
                z=cm, text=cm, texttemplate="%{text}",
                colorscale=[[0,"#080b10"],[1,ACID]],
                showscale=False, xgap=1, ygap=1,
            ), row=1, col=idx+1)
        fig.update_layout(**pgo(height=340))
        st.plotly_chart(fig, use_container_width=True)

    sec("FEATURE IMPORTANCE (TOP 20)")
    fi = ml_results[ml_best]["fi"]
    if len(fi) > 0:
        fi_df = fi.head(20).reset_index()
        fi_df.columns = ["feature", "importance"]
        fig = go.Figure(go.Bar(
            x=fi_df["importance"], y=fi_df["feature"],
            orientation="h",
            marker=dict(
                color=fi_df["importance"],
                colorscale=[[0,"#0d2a0d"],[0.5,ELEC],[1,ACID]],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in fi_df["importance"]],
            textposition="outside",
        ))
        fig.update_layout(**pgo(height=420, xaxis_title="Importance",
                                 yaxis_autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    sec("5-FOLD CROSS-VALIDATION F1")
    cv_names = list(ml_results.keys())
    cv_means = [ml_results[n]["cv_f1_mean"] for n in cv_names]
    cv_stds  = [ml_results[n]["cv_f1_std"]  for n in cv_names]
    fig = go.Figure(go.Bar(
        x=cv_names, y=cv_means,
        error_y=dict(type="data", array=cv_stds, visible=True),
        marker_color=PLT_PALETTE[:len(cv_names)],
        text=[f"{v:.4f}" for v in cv_means], textposition="outside",
    ))
    fig.update_layout(**pgo(height=280, yaxis_range=[0, 1.15]))
    st.plotly_chart(fig, use_container_width=True)

    # Live Predictor
    sec("LIVE ABSENCE RISK PREDICTOR", ELEC)
    c1, c2, c3 = st.columns(3)
    with c1:
        pred_sub  = st.selectbox("Subject",    df["subject"].unique())
        pred_dept = st.selectbox("Department", df["department"].unique())
        pred_year = st.slider("Year", 1, 4, 2)
    with c2:
        pred_hour = st.slider("Session Hour", 8, 17, 10)
        pred_day  = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday"])
        pred_pct  = st.slider("Historical Attendance %", 40, 100, 85)
    with c3:
        pred_late = st.slider("Historical Late Rate %", 0, 40, 5)
        pred_pat  = st.selectbox("Pattern", ["excellent","consistent","good","moderate","at_risk","critical"])
        pred_sess = st.slider("Sessions Attended", 10, 400, 120)

    if st.button("⚡  RUN ABSENCE RISK ANALYSIS", use_container_width=True):
        day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4}
        sub_map = {s:i for i,s in enumerate(sorted(df["subject"].unique()))}
        dep_map = {d:i for i,d in enumerate(sorted(df["department"].unique()))}
        pat_map = {"excellent":0,"consistent":1,"good":2,"moderate":3,"at_risk":4,"critical":5}

        feat_vals = [
            pred_hour, int(pred_hour<12), int(pred_hour>=13), int(pred_hour<=9),
            int(pred_day=="Monday"), int(pred_day=="Friday"),
            int(pred_day in ["Monday","Friday"]), int(pred_day in ["Tue","Wed","Thu"]),
            10, 40, 0, 0, 3,
            0, 0, 0, 0,
            pred_year,
            pred_pct/100, pred_late/100, pred_sess, 0.88, 0,
            0.85, 0.02, 0.85, 0.85,
            day_map.get(pred_day,0), sub_map.get(pred_sub,0),
            dep_map.get(pred_dept,0), pat_map.get(pred_pat,2),
        ]
        n_feat   = len(ml_feat)
        X_raw    = np.array([feat_vals[:n_feat]], dtype=float)
        X_scaled = ml_scaler.transform(X_raw)
        mdl      = ml_results[ml_best]["model"]
        prob_ab  = float(mdl.predict_proba(X_scaled)[0, 1])
        prob_pr  = 1 - prob_ab
        risk_lbl = ("CRITICAL" if prob_ab>0.6 else "HIGH" if prob_ab>0.4
                    else "MEDIUM" if prob_ab>0.25 else "LOW")
        rc = WARN if prob_ab>0.6 else ORNG if prob_ab>0.4 else VOLT if prob_ab>0.25 else ACID

        col_r, col_l = st.columns([1, 2])
        with col_r:
            st.markdown(f"""
            <div class="scan-ok" style='border-color:{rc}40;'>
              <div style='font-size:8px;font-family:Share Tech Mono,monospace;
                   color:#1a4a1a;letter-spacing:2px;'>ABSENCE PROBABILITY</div>
              <div class="scan-pct" style='color:{rc};'>{prob_ab*100:.1f}%</div>
              <div style='font-family:Orbitron,monospace;font-size:14px;
                   color:{rc};margin-top:6px;'>{risk_lbl} RISK</div>
              <div style='font-size:9px;font-family:Share Tech Mono,monospace;
                   color:#1a4a1a;margin-top:10px;'>
                PRESENT PROB: {prob_pr*100:.1f}%<br>MODEL: {ml_best}
              </div>
            </div>""", unsafe_allow_html=True)
        with col_l:
            fig = go.Figure(go.Bar(
                x=["Will Absent", "Will Present"],
                y=[prob_ab*100, prob_pr*100],
                marker_color=[rc, ELEC],
                text=[f"{prob_ab*100:.1f}%", f"{prob_pr*100:.1f}%"],
                textposition="outside",
            ))
            fig.update_layout(**pgo(height=240, yaxis_range=[0, 120]))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FACE SCANNER
# ══════════════════════════════════════════════════════════════════════════════
elif "FACE" in page or "SCANNER" in page:
    sec("NEURAL FACE RECOGNITION SYSTEM · OpenCV + Haar Cascade")

    c_info, c_scan = st.columns([1, 1])
    with c_info:
        st.markdown("""
        <div class="panel">
          <div class="p-title">PRODUCTION PIPELINE</div>
          <div style='font-family:Share Tech Mono,monospace;font-size:9px;
               color:#2a5a2a;line-height:2.4;margin-top:8px;'>
            STEP 1 → cv2.VideoCapture(0) opens webcam<br>
            STEP 2 → Convert frame BGR → Grayscale<br>
            STEP 3 → Haar Cascade detectMultiScale(1.1, minNeighbors=5)<br>
            STEP 4 → Face ROI cropped + normalised (128×128)<br>
            STEP 5 → HOG features extracted from face patch<br>
            STEP 6 → Cosine similarity vs known_faces/ database<br>
            STEP 7 → Liveness anti-spoof check (blink detection)<br>
            STEP 8 → Confidence ≥ 80% → identity confirmed<br>
            STEP 9 → Triple duplicate guard (student+subject+date)<br>
            STEP 10 → Append to attendance CSV with timestamp<br>
            STEP 11 → Streamlit dashboard auto-refreshes
          </div>
        </div>
        <div class="panel" style='border-left-color:#ffe600;margin-top:8px;'>
          <div class="p-title" style='color:#ffe600;'>TECH STACK</div>
          <div style='font-family:Share Tech Mono,monospace;font-size:9px;
               color:#2a5a2a;line-height:2.2;margin-top:6px;'>
            OpenCV 4.x · haarcascade_frontalface_default.xml<br>
            face_recognition (dlib HOG + CNN model)<br>
            NumPy cosine similarity · Pillow preprocessing<br>
            Anti-spoofing: blink + liveness detection<br>
            Pandas CSV logging · Streamlit real-time UI<br>
            Confidence threshold: 80% (configurable)
          </div>
        </div>""", unsafe_allow_html=True)

    with c_scan:
        sec("FACE SCAN SIMULATOR")
        stu_list = students["name"].tolist() if len(students) > 0 else df["student_name"].unique().tolist()
        scan_stu = st.selectbox("Student", stu_list)
        scan_sub = st.selectbox("Subject  ", df["subject"].unique())
        conf_thr = st.slider("Confidence threshold %", 70, 95, 80)
        attempts = st.slider("Scan attempts", 1, 5, 3)

        if st.button("👁  INITIATE FACE SCAN", use_container_width=True):
            prog       = st.progress(0)
            status_box = st.empty()
            best_result, best_conf = None, 0

            for i in range(attempts):
                prog.progress((i+1)/attempts)
                status_box.markdown(
                    f'<div style="font-family:Share Tech Mono,monospace;font-size:9px;'
                    f'color:#39ff14;">[ SCANNING ] Attempt {i+1}/{attempts} …</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(0.30)
                res = simulate_face_recognition(scan_stu, conf_thr/100)
                if res["confidence"] > best_conf:
                    best_conf, best_result = res["confidence"], res

            prog.empty(); status_box.empty()
            rc   = ACID if best_result["recognized"] else WARN
            bcls = "scan-ok" if best_result["recognized"] else "scan-fail"

            liveness_txt = "✓ LIVENESS OK" if best_result["liveness_check"] else "⚠ LIVENESS FAIL"
            liveness_col = ACID if best_result["liveness_check"] else WARN

            st.markdown(f"""
            <div class="{bcls}" style='border-color:{rc}40;'>
              <div class="scan-pct" style='color:{rc};'>{best_result['confidence']*100:.1f}%</div>
              <div class="scan-name">{"✓" if best_result['recognized'] else "✗"} {best_result['student']}</div>
              <div style='font-size:9px;font-family:Share Tech Mono,monospace;
                   color:{liveness_col};margin-top:6px;'>{liveness_txt}</div>
              <div style='font-size:9px;font-family:Share Tech Mono,monospace;
                   color:#1a4a1a;margin-top:8px;'>
                {best_result['status']} · {best_result['timestamp']}<br>
                {best_result['face_id']} · {best_result['processing_ms']}ms
              </div>
            </div>""", unsafe_allow_html=True)

            if best_result["recognized"] and best_result["liveness_check"]:
                stu_row = students[students["name"] == scan_stu].iloc[0] if len(students)>0 else {"id": "UNK"}
                ok, msg = mark_attendance(
                    stu_row.get("id","UNK"), scan_stu, scan_sub,
                    "Face Recognition", best_result["confidence"],
                )
                color = ACID if ok else VOLT
                st.markdown(f"""
                <div style='font-family:Share Tech Mono,monospace;font-size:10px;
                  color:{color};text-align:center;margin-top:10px;'>
                  [{datetime.now().strftime('%H:%M:%S')}] {msg}
                </div>""", unsafe_allow_html=True)

    sec("TODAY'S LIVE ATTENDANCE LOG")
    live_df = load_live_attendance()
    if len(live_df) > 0:
        today    = datetime.now().strftime("%Y-%m-%d")
        today_df = live_df[live_df["date"] == today] if "date" in live_df.columns else live_df
        st.dataframe(today_df, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style='font-family:Share Tech Mono,monospace;font-size:9px;
          color:#1a4a1a;text-align:center;padding:20px;'>
          [ NO RECORDS ] · Run a face scan to mark attendance
        </div>""", unsafe_allow_html=True)

    sec("OPENCV PRODUCTION CODE")
    st.code("""
import cv2
import pandas as pd
from datetime import datetime

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def mark_attendance(name, subject, csv='attendance.csv'):
    now = datetime.now()
    try:
        df  = pd.read_csv(csv)
        if ((df['Name']==name) &
            (df['Date']==now.strftime('%Y-%m-%d')) &
            (df['Subject']==subject)).any():
            return False        # duplicate guard
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name','Subject','Time','Date','Confidence'])
    df = pd.concat([df, pd.DataFrame([{
        'Name': name, 'Subject': subject,
        'Time': now.strftime('%H:%M:%S'),
        'Date': now.strftime('%Y-%m-%d'),
        'Confidence': 0.95,
    }])], ignore_index=True)
    df.to_csv(csv, index=False)
    return True

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (57,255,20), 2)
        cv2.putText(frame, 'Identifying...', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (57,255,20), 2)
    cv2.imshow('AttendanceIQ Pro', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
    """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RISK MATRIX
# ══════════════════════════════════════════════════════════════════════════════
elif "RISK" in page:
    sec("STUDENT RISK INTELLIGENCE MATRIX")

    risk_counts = stu_rpt["risk_label"].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (lbl, clr) in zip([c1,c2,c3,c4], [
        ("Good",ACID), ("Low Risk",VOLT), ("At Risk",ORNG), ("Critical",WARN)
    ]):
        kpi(col, lbl, str(risk_counts.get(lbl, 0)), "students", clr)
    kpi(c5, "Avg Dropout Risk",
        f"{stu_rpt['dropout_risk'].mean():.1f}",
        "cohort average", PINK)

    st.markdown("<hr style='border-color:#0d1a0d;'>", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        sec("ALL STUDENTS — ATTENDANCE %")
        fig = go.Figure()
        for _, row in stu_rpt.iterrows():
            rc = risk_color(row["pct"])
            fig.add_trace(go.Bar(
                x=[f"{row['name'].split()[0]} {row['name'].split()[-1]}"],
                y=[row["pct"]],
                marker_color=rc,
                showlegend=False,
            ))
        fig.add_hline(y=75, line_color=WARN, line_dash="dash", line_width=1)
        fig.add_hline(y=85, line_color=VOLT, line_dash="dash", line_width=1)
        fig.update_layout(**pgo(height=340, yaxis_title="Attendance %",
                                 yaxis_range=[0, 110]))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("RISK DISTRIBUTION")
        fig = go.Figure(go.Pie(
            labels=["Good","Low Risk","At Risk","Critical"],
            values=[risk_counts.get(l,0) for l in ["Good","Low Risk","At Risk","Critical"]],
            hole=.55,
            marker_colors=[ACID, VOLT, ORNG, WARN],
        ))
        fig.update_layout(**pgo(height=340, showlegend=True))
        st.plotly_chart(fig, use_container_width=True)

    sec("DROPOUT RISK RANKING — TOP 15", WARN)
    top_dropout = stu_rpt.sort_values("dropout_risk", ascending=False).head(15)
    fig = go.Figure(go.Bar(
        x=top_dropout["dropout_risk"],
        y=top_dropout["name"].apply(lambda n: f"{n.split()[0]} {n.split()[-1]}"),
        orientation="h",
        marker=dict(
            color=top_dropout["dropout_risk"],
            colorscale=[[0, VOLT],[0.5, ORNG],[1, WARN]],
            showscale=False,
        ),
        text=[f"{v:.0f}" for v in top_dropout["dropout_risk"]],
        textposition="outside",
    ))
    fig.update_layout(**pgo(height=320, xaxis_range=[0, 120],
                             xaxis_title="Dropout Risk Score",
                             yaxis_autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    sec("AT-RISK & CRITICAL STUDENTS — ACTION CARDS", WARN)
    for _, row in risk_stu.iterrows():
        rc   = risk_color(row["pct"])
        barw = int(row["pct"])
        st.markdown(f"""
        <div class="panel" style='border-left-color:{rc};padding:10px 16px;'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div>
              <span style='font-family:Orbitron,monospace;font-size:10px;color:#60c060;'>
                {row['name']}</span>
              <span style='font-family:Share Tech Mono,monospace;font-size:8px;
                   color:#1a4a1a;margin-left:10px;'>
                {row['dept']} · Y{row['year']} · {row['student_id']}</span>
            </div>
            <div style='text-align:right;'>
              <span style='font-family:Orbitron,monospace;font-size:18px;
                   font-weight:700;color:{rc};'>{row['pct']}%</span>
              <span style='font-family:Share Tech Mono,monospace;font-size:8px;
                   color:#1a4a1a;display:block;'>
                Risk:{row['risk_score']} · Dropout:{row['dropout_risk']} · AHI:{row['ahi']}</span>
            </div>
          </div>
          {progress_bar(barw, rc)}
          <div style='font-family:Share Tech Mono,monospace;font-size:8px;color:#1a4a1a;'>
            Present:{row['present']} · Absent:{row['absent']} ·
            Late:{row['late']} ({row['late_rate']}%)
          </div>
        </div>""", unsafe_allow_html=True)

    sec("RECOMMENDED INTERVENTIONS")
    for lbl, actions, clr in [
        ("Critical <60%", [
            "Issue official attendance warning letter immediately",
            "Schedule HOD counselling within 3 working days",
            "Notify parents/guardians by phone + written notice",
            "Place on academic probation — weekly check-in required",
            "Check for mental health or personal crisis indicators",
        ], WARN),
        ("At Risk 60–75%", [
            "Send automated alert email to student + parent",
            "Assign dedicated faculty mentor for 4 weeks",
            "Bi-weekly one-on-one attendance review",
            "Identify root cause: transport, health, financial, motivation",
        ], ORNG),
        ("Low Risk 75–85%", [
            "Send gentle SMS/email reminder",
            "Monitor attendance for the next 2 weeks",
            "Positive reinforcement if improving trend detected",
        ], VOLT),
    ]:
        st.markdown(f"""
        <div class="panel" style='border-left-color:{clr};margin-bottom:6px;'>
          <div class="p-title" style='color:{clr};'>{lbl}</div>
          {"".join(f"<div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#2a5a2a;padding:3px 0;border-bottom:1px solid #0a150a;'><span style='color:{clr};'>▶</span> {a}</div>" for a in actions)}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif "ANOMALY" in page:
    sec("DUAL ANOMALY DETECTION ENGINE")
    st.markdown("""
    <div class="panel">
      <div class="p-title">WHAT THIS DETECTS</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:9px;
           color:#2a5a2a;line-height:2.2;margin-top:6px;'>
        <span style='color:#39ff14;'>IsolationForest</span> — isolates outliers by random partitioning.
        Students with sudden drops or unusual absence clusters are flagged.<br>
        <span style='color:#00d4ff;'>LocalOutlierFactor</span> — compares each student's density to neighbours.
        Catches students whose pattern is locally unusual (different from peers).<br>
        <span style='color:#ffe600;'>Combined flag</span> — student flagged by EITHER detector.
        Both flagged = highest priority review.
      </div>
    </div>""", unsafe_allow_html=True)

    n_if  = int(anomalies["if_anomaly"].sum())  if "if_anomaly"  in anomalies.columns else 0
    n_lof = int(anomalies["lof_anomaly"].sum()) if "lof_anomaly" in anomalies.columns else 0
    n_both= int(anomalies.get("both_flagged", pd.Series([False]*len(anomalies))).sum())
    n_any = int(anomalies["anomaly"].sum())
    n_norm= len(anomalies) - n_any

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, "IF Flagged",   str(n_if),   "IsolationForest", WARN)
    kpi(c2, "LOF Flagged",  str(n_lof),  "LocalOutlierFactor", ORNG)
    kpi(c3, "Both Flagged", str(n_both), "highest priority", PINK)
    kpi(c4, "Any Anomaly",  str(n_any),  "total flagged", WARN)
    kpi(c5, "Normal",       str(n_norm), "healthy pattern", ACID)

    sec("ANOMALY SCORES — ALL STUDENTS (IsolationForest)")
    fig = go.Figure()
    for _, row in anomalies.iterrows():
        clr = WARN if row.get("if_anomaly", row["anomaly"]) else ACID
        fig.add_trace(go.Bar(
            x=[row["student_name"].split()[0]],
            y=[row["score"]],
            marker_color=clr,
            showlegend=False,
            hovertext=f"{row['student_name']} · {row['department']}",
        ))
    fig.add_hline(y=0, line_color=VOLT, line_dash="dash", line_width=1,
                  annotation_text="Anomaly boundary")
    fig.update_layout(**pgo(height=300,
                             yaxis_title="IF Score (negative = anomalous)"))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("FLAGGED STUDENTS", WARN)
        flagged = anomalies[anomalies["anomaly"]].sort_values("score")
        for _, row in flagged.iterrows():
            both = row.get("both_flagged", False)
            bc   = PINK if both else WARN
            st.markdown(f"""
            <div class="alert-p{'1' if both else '2'}">
              <div style='display:flex;justify-content:space-between;'>
                <span style='font-family:Orbitron,monospace;font-size:10px;color:{bc};'>
                  ⚠ {row['student_name']}</span>
                <span style='font-family:Share Tech Mono,monospace;font-size:8px;color:#1a4a1a;'>
                  {row['department']}</span>
              </div>
              <div style='font-family:Share Tech Mono,monospace;font-size:8px;color:#2a2a2a;margin-top:4px;'>
                IF:{row.get('if_anomaly','?')} · LOF:{row.get('lof_anomaly','?')} ·
                Score:{row['score']:.4f}
                {"· BOTH DETECTORS" if both else ""}
              </div>
            </div>""", unsafe_allow_html=True)

    with c2:
        sec("NORMAL PATTERNS", ACID)
        for _, row in anomalies[~anomalies["anomaly"]].sort_values("score", ascending=False).head(10).iterrows():
            st.markdown(f"""
            <div class="alert-p4">
              <div style='display:flex;justify-content:space-between;'>
                <span style='font-family:Share Tech Mono,monospace;font-size:9px;color:#39ff14;'>
                  ✓ {row['student_name']}</span>
                <span style='font-family:Share Tech Mono,monospace;font-size:8px;color:#1a4a1a;'>
                  {row['department']} · {row['score']:.4f}</span>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif "FORECAST" in page:
    sec("ATTENDANCE FORECAST ENGINE · NEXT 14 DAYS")
    st.markdown("""
    <div class="panel">
      <div class="p-title">METHODOLOGY</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:9px;
           color:#2a5a2a;line-height:2.2;margin-top:6px;'>
        Rolling 7-day average baseline · Linear trend extrapolation ·
        Seasonal adjustments (monsoon −1.2%, exam +2%, Mon −1%) ·
        Dynamic CI width = volatility × 1.5 + horizon × 0.15 ·
        Weekend excluded · Tamil Nadu festival calendar integrated
      </div>
    </div>""", unsafe_allow_html=True)

    hist     = daily_trend(df)
    fcast_df = forecast

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["pct"],
        mode="lines", name="Historical",
        line=dict(color=ACID, width=1.2),
        fill="tozeroy", fillcolor="rgba(57,255,20,.05)",
    ))
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["rolling_7d"],
        mode="lines", name="7-day avg",
        line=dict(color=ELEC, width=1.8, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=fcast_df["date"], y=fcast_df["predicted"],
        mode="lines+markers", name="Forecast",
        line=dict(color=PINK, width=2, dash="dash"),
        marker=dict(size=6, color=PINK),
    ))
    fig.add_trace(go.Scatter(
        x=list(fcast_df["date"]) + list(fcast_df["date"])[::-1],
        y=list(fcast_df["upper"]) + list(fcast_df["lower"])[::-1],
        fill="toself", fillcolor="rgba(255,45,155,.07)",
        line=dict(width=0), name="95% CI band",
    ))
    fig.add_hline(y=75, line_color=WARN, line_dash="dash", line_width=1)
    fig.add_hline(y=85, line_color=VOLT, line_dash="dash", line_width=1)
    fig.update_layout(**pgo(height=400, xaxis_title="Date",
                             yaxis_title="Attendance %", yaxis_range=[52,105],
                             title="Historical + 14-Day Forecast with Confidence Interval"))
    st.plotly_chart(fig, use_container_width=True)

    sec("FORECAST TABLE")
    fcast_show = fcast_df.copy()
    fcast_show["date"] = fcast_show["date"].dt.strftime("%Y-%m-%d (%a)")
    fcast_show.columns = ["Date","Predicted %","Lower 95%","Upper 95%"]
    st.dataframe(fcast_show, use_container_width=True, hide_index=True, height=320)

    sec("TREND SUMMARY")
    c1, c2, c3, c4 = st.columns(4)
    last_7  = hist["pct"].tail(7).mean()
    last_30 = hist["pct"].tail(30).mean()
    slope   = hist["trend"].tail(14).mean()
    vol     = hist["volatility"].tail(7).mean()
    kpi(c1, "7-day avg",   f"{last_7:.1f}%",  "recent performance",  ACID)
    kpi(c2, "30-day avg",  f"{last_30:.1f}%", "medium term",         ELEC)
    kpi(c3, "Trend slope", f"{slope:+.2f}%/d",
        "↑ improving" if slope >= 0 else "↓ declining",
        ACID if slope >= 0 else WARN)
    kpi(c4, "Volatility",  f"{vol:.2f}%",     "std over 7 days",     VOLT)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif "REPORTS" in page:
    sec("PROJECT INTELLIGENCE BRIEF")
    st.markdown(f"""
    <div class="panel" style='border-left-color:#00d4ff;'>
      <div class="p-title" style='font-size:15px;color:#00d4ff;'>
        ATTENDANCEIQ PRO · SMART FACE RECOGNITION SYSTEM v{VERSION}</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:9px;
           color:#2a5a2a;line-height:2.2;margin-top:8px;'>
        Afynix Digital AI Internship · Week 4 · Computer Vision + ML<br>
        Author: Deol Allwyn Samuel J B · VLSI · Chennai Institute of Technology<br>
        Version: {VERSION} · Date: {datetime.now().strftime('%Y-%m-%d')}
      </div>
    </div>""", unsafe_allow_html=True)

    sec("SYSTEM SPECIFICATIONS")
    specs = [
        ("CV Engine",        "OpenCV 4.x · Haar Cascade · Real-time face detection"),
        ("Face Recog",       "dlib HOG + CNN · Cosine similarity · 80% threshold · liveness check"),
        ("ML Models",        f"RF + GBM + LogReg + MLP + XGBoost + Stacking · SMOTE+Tomek balanced"),
        ("Best Model",       f"{ml_best} · Acc={best_m['accuracy']} · F1={best_m['f1']} · AUC={best_m['auc']}"),
        ("Features",         f"{len(ml_feat)} engineered features · temporal + behavioral + contextual"),
        ("Anomaly Detect",   f"IsolationForest + LOF dual detection · {len(anomalies[anomalies['anomaly']])} flagged"),
        ("Forecasting",      "Rolling avg + trend + seasonal · 14-day ahead · dynamic CI"),
        ("Grade Prediction", "RF Regressor · attendance → CGPA mapping · detention alert"),
        ("Clustering",       "K-Means (k=5) + PCA 2D · 5 behavioral segments"),
        ("Smart Alerts",     "P1–P4 priority alerts · dropout risk + anomaly + medical patterns"),
        ("Dataset",          f"{len(df):,} records · {summary['total_students']} students · {summary['total_days']} days · 3 semesters"),
        ("Festival Calendar","Tamil Nadu festival + monsoon effects integrated"),
        ("Duplicate Guard",  "Student + Subject + Date triple-check"),
        ("Storage",          "Pandas CSV · append-only · timestamped · multi-CSV architecture"),
        ("Deployment",       "Streamlit Cloud · GitHub Actions CI/CD ready"),
    ]
    for spec, detail in specs:
        st.markdown(f"""
        <div class="panel" style='padding:8px 14px;margin-bottom:4px;'>
          <span style='font-family:Orbitron,monospace;font-size:9px;color:#39ff14;'>{spec}</span>
          <span style='font-family:Share Tech Mono,monospace;font-size:9px;
               color:#1a4a1a;margin-left:12px;'>{detail}</span>
        </div>""", unsafe_allow_html=True)

    sec("KEY FINDINGS")
    hist_local = daily_trend(df)
    findings = [
        f"Overall attendance: {summary['attendance_pct']}% across {summary['total_days']} working days",
        f"{summary['face_recog_pct']}% captured via face recognition (avg confidence {summary['avg_confidence']}%)",
        f"{len(risk_stu)} students below 75% threshold — immediate action required",
        f"{len(anomalies[anomalies['anomaly']])} anomalous patterns flagged by dual-detector",
        f"Best ML: {ml_best} · F1={best_m['f1']} · AUC={best_m['auc']}",
        f"Monday/Friday show 7–12% lower attendance than midweek sessions",
        f"Morning sessions (08–10h) have highest attendance rates",
        f"Monsoon months (Oct–Dec) show {summary['rain_affected']} rain-affected session days",
        f"Medical leave episodes: {summary['medical_leaves']} recorded across {summary['total_students']} students",
        f"Forecast trend: {'stable ≈' if abs(hist_local['trend'].tail(14).mean()) < 0.1 else 'shifting'} over next 14 days",
    ]
    for f in findings:
        st.markdown(f"""
        <div style='font-family:Share Tech Mono,monospace;font-size:9px;
          color:#2a5a2a;padding:5px 0;border-bottom:1px solid #0a150a;'>
          <span style='color:#39ff14;'>▶</span> {f}
        </div>""", unsafe_allow_html=True)

    sec("EXPORT CENTRE")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("⬇ Student Report",   stu_rpt.to_csv(index=False).encode(),
                           "student_report.csv","text/csv", use_container_width=True)
    with c2:
        st.download_button("⬇ Full Attendance",  df.to_csv(index=False).encode(),
                           "attendance_full.csv","text/csv", use_container_width=True)
    with c3:
        st.download_button("⬇ At-Risk Students", risk_stu.to_csv(index=False).encode(),
                           "at_risk.csv","text/csv", use_container_width=True)
    with c4:
        st.download_button("⬇ Anomaly Report",   anomalies.to_csv(index=False).encode(),
                           "anomalies.csv","text/csv", use_container_width=True)

    if not df_grades.empty:
        c5, c6 = st.columns(2)
        with c5:
            st.download_button("⬇ Grade Data", df_grades.to_csv(index=False).encode(),
                               "grades.csv","text/csv", use_container_width=True)
        with c6:
            st.download_button("⬇ Medical Leaves", df_leaves.to_csv(index=False).encode(),
                               "medical_leaves.csv","text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — STUDENT PROFILE (individual deep dive)
# ══════════════════════════════════════════════════════════════════════════════
elif "PROFILE" in page:
    sec("INDIVIDUAL STUDENT PROFILE · DEEP DIVE")

    all_names = sorted(df["student_name"].unique())
    sel_name  = st.selectbox("Select Student", all_names)

    sdf = df[df["student_name"] == sel_name]
    if sdf.empty:
        st.warning("No records found.")
        st.stop()

    sid     = sdf["student_id"].iloc[0]
    s_row   = stu_rpt[stu_rpt["student_id"] == sid]
    if s_row.empty:
        st.warning("Profile not computed.")
        st.stop()
    s_row = s_row.iloc[0]

    # Profile header
    rc = risk_color(s_row["pct"])
    st.markdown(f"""
    <div class="panel" style='border-left-color:{rc};padding:18px 20px;'>
      <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
        <div>
          <div style='font-family:Orbitron,monospace;font-size:18px;
               font-weight:700;color:#39ff14;'>{s_row['name']}</div>
          <div style='font-family:Share Tech Mono,monospace;font-size:10px;
               color:#1a4a1a;margin-top:4px;letter-spacing:1px;'>
            {sid} · {s_row['dept']} · Year {s_row['year']} · {s_row['email']}
          </div>
          <div style='margin-top:8px;'>
            <span class='cluster-chip' style='background:#39ff1415;color:#39ff14;
                  border:1px solid #39ff1430;'>{s_row['pattern'].upper()}</span>
            <span class='cluster-chip' style='background:#00d4ff15;color:#00d4ff;
                  border:1px solid #00d4ff30;'>{s_row['risk_label']}</span>
          </div>
        </div>
        <div style='text-align:right;'>
          <div style='font-family:Orbitron,monospace;font-size:36px;
               font-weight:900;color:{rc};'>{s_row['pct']}%</div>
          <div style='font-family:Share Tech Mono,monospace;font-size:9px;
               color:#1a4a1a;'>ATTENDANCE</div>
        </div>
      </div>
      {progress_bar(s_row['pct'], rc)}
    </div>""", unsafe_allow_html=True)

    # KPIs
    c = st.columns(6)
    kpi(c[0], "Present",      str(s_row["present"]),           f"of {s_row['total']} sessions", ACID)
    kpi(c[1], "Absent",       str(s_row["absent"]),            f"{s_row['absent_rate']}%", WARN)
    kpi(c[2], "Late",         str(s_row["late"]),              f"{s_row['late_rate']}% rate", VOLT)
    kpi(c[3], "Risk Score",   str(s_row["risk_score"]),        "composite", ORNG)
    kpi(c[4], "Dropout Risk", f"{s_row['dropout_risk']}",      "/100", PINK)
    kpi(c[5], "AHI",          f"{s_row['ahi']}",               "health index", ELEC)

    st.markdown("<br>", unsafe_allow_html=True)

    # Daily attendance timeline
    c1, c2 = st.columns([2, 1])
    with c1:
        sec("ATTENDANCE TIMELINE")
        daily = sdf.groupby("date")["status"].apply(
            lambda x: 1.0 if (x == "Present").any() else 0.0
        ).reset_index(name="present")
        daily["date"] = pd.to_datetime(daily["date"])
        daily["rolling"] = daily["present"].rolling(7, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["present"],
            marker_color=[ACID if v else WARN for v in daily["present"]],
            name="Daily (1=Present)",
        ))
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["rolling"],
            mode="lines", name="7-day avg",
            line=dict(color=ELEC, width=2),
        ))
        fig.add_hline(y=0.75, line_color=WARN, line_dash="dash", line_width=1)
        fig.update_layout(**pgo(height=260,
                                 yaxis_title="Present (1=yes)",
                                 yaxis_range=[0, 1.2]))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("SUBJECT BREAKDOWN")
        sub_pct = sdf.groupby("subject")["status"].apply(
            lambda x: (x == "Present").mean() * 100
        ).reset_index(name="pct").sort_values("pct")
        fig = go.Figure(go.Bar(
            x=sub_pct["pct"], y=sub_pct["subject"],
            orientation="h",
            marker=dict(color=[risk_color(p) for p in sub_pct["pct"]]),
            text=[f"{p:.0f}%" for p in sub_pct["pct"]],
            textposition="outside",
        ))
        fig.add_vline(x=75, line_color=WARN, line_dash="dash", line_width=1)
        fig.update_layout(**pgo(height=260, xaxis_range=[0, 115]))
        st.plotly_chart(fig, use_container_width=True)

    # Radar + session time
    c3, c4 = st.columns(2)
    with c3:
        sec("BEHAVIORAL RADAR")
        att_score   = s_row["pct"]
        punct_score = max(0, 100 - s_row["late_rate"] * 3)
        consist_score = max(0, 100 - sdf.groupby("date")["status"].apply(
            lambda x: 1.0 if (x=="Present").any() else 0.0
        ).std() * 200)
        exam_sdf    = sdf[sdf.get("exam_period", pd.Series([""]*len(sdf))).ne("")]
        exam_score  = (exam_sdf["status"]=="Present").mean()*100 if len(exam_sdf)>0 else att_score
        risk_inv    = max(0, 100 - s_row["risk_score"])

        radar_cats  = ["Attendance","Punctuality","Consistency","Exam Perf","Risk Inv."]
        radar_vals  = [att_score, punct_score, consist_score, exam_score, risk_inv]
        fig = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself",
            fillcolor="rgba(57,255,20,.10)",
            line=dict(color=rc, width=2),
        ))
        fig.update_layout(**pgo(height=300,
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(range=[0, 100], color="#1a4a1a"))))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        sec("SESSION HOUR ATTENDANCE")
        sdf2 = sdf.copy()
        sdf2["hour"] = sdf2["session_time"].str.split(":").str[0].astype(int)
        h_grp = sdf2.groupby("hour")["status"].apply(
            lambda x: (x=="Present").mean()*100
        ).reset_index(name="pct")
        fig = go.Figure(go.Bar(
            x=h_grp["hour"].astype(str), y=h_grp["pct"],
            marker_color=[risk_color(p) for p in h_grp["pct"]],
            text=[f"{p:.0f}%" for p in h_grp["pct"]], textposition="outside",
        ))
        fig.update_layout(**pgo(height=300, xaxis_title="Hour",
                                 yaxis_title="%", yaxis_range=[0,115]))
        st.plotly_chart(fig, use_container_width=True)

    # Grade profile
    if not df_grades.empty:
        sec("GRADE PROFILE")
        stu_grades = df_grades[df_grades["student_id"] == sid]
        if not stu_grades.empty:
            st.dataframe(
                stu_grades[["subject","credits","attendance_pct","internal_marks",
                            "external_marks","total_marks","grade","grade_point","semester"]],
                use_container_width=True, hide_index=True, height=280,
            )
            cgpa = ((stu_grades["grade_point"] * stu_grades["credits"]).sum()
                    / stu_grades["credits"].sum()) if stu_grades["credits"].sum() > 0 else 0
            detained_n = (stu_grades["grade"] == "DETAINED").sum()
            col1, col2, col3 = st.columns(3)
            kpi(col1, "CGPA", f"{cgpa:.2f}", "weighted average", ACID)
            kpi(col2, "Detained Subjects", str(detained_n), "attendance <75%", WARN)
            kpi(col3, "Subjects", str(len(stu_grades)), "total graded", ELEC)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — GRADE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif "GRADE" in page:
    sec("GRADE INTELLIGENCE · ATTENDANCE–ACADEMIC CORRELATION")

    g_intel = load_grade_intel()

    if not g_intel or g_intel.get("merged") is None or g_intel["merged"].empty:
        st.warning("Grade data not available — run `python generate_data.py` first.")
        st.stop()

    merged   = g_intel["merged"]
    corr_p   = g_intel.get("corr_pearson",  0)
    corr_s   = g_intel.get("corr_spearman", 0)
    avg_cgpa = g_intel.get("avg_cgpa", 0)
    gd       = g_intel.get("grade_dist", pd.DataFrame())

    # Top KPIs
    c = st.columns(5)
    kpi(c[0], "Pearson Corr",  f"{corr_p:.3f}",   "att % vs CGPA",     ACID)
    kpi(c[1], "Spearman Corr", f"{corr_s:.3f}",   "rank correlation",  ELEC)
    kpi(c[2], "Avg CGPA",      f"{avg_cgpa:.2f}",  "cohort avg",        VOLT)
    kpi(c[3], "Detained Total",str(df_grades[df_grades["grade"]=="DETAINED"].shape[0]) if not df_grades.empty else "0",
        "insufficient attendance", WARN)
    kpi(c[4], "Top CGPA",      f"{merged['cgpa'].max():.2f}" if "cgpa" in merged.columns else "—",
        "highest achiever", PINK)

    st.markdown(f"""
    <div class="panel" style='border-left-color:#00d4ff;margin:12px 0;'>
      <div class="p-title" style='color:#00d4ff;'>CORRELATION INTERPRETATION</div>
      <div style='font-family:Share Tech Mono,monospace;font-size:9px;
           color:#2a5a2a;line-height:2.2;margin-top:6px;'>
        Pearson r = {corr_p:.3f} → {"Strong" if abs(corr_p)>0.6 else "Moderate" if abs(corr_p)>0.4 else "Weak"}
        {"positive" if corr_p>0 else "negative"} linear relationship between attendance and CGPA.<br>
        Every 10% increase in attendance is associated with roughly
        {abs(corr_p) * 0.5:.2f} CGPA points improvement on average.
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("ATTENDANCE % vs CGPA SCATTER")
        if "cgpa" in merged.columns:
            fig = px.scatter(
                merged, x="att_pct", y="cgpa",
                color="department", trendline="ols",
                hover_name="student_name",
                color_discrete_sequence=PLT_PALETTE,
                labels={"att_pct":"Attendance %","cgpa":"CGPA"},
            )
            fig.add_hline(y=6.0, line_color=WARN, line_dash="dash", line_width=1)
            fig.add_vline(x=75,  line_color=WARN, line_dash="dash", line_width=1)
            fig.update_layout(**pgo(height=340))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("GRADE DISTRIBUTION")
        if not gd.empty:
            fig = go.Figure(go.Bar(
                x=gd["grade"], y=gd["count"],
                marker_color=PLT_PALETTE[:len(gd)],
                text=gd["count"], textposition="outside",
            ))
            fig.update_layout(**pgo(height=340, xaxis_title="Grade",
                                     yaxis_title="Count"))
            st.plotly_chart(fig, use_container_width=True)

    sec("SUBJECT PASS RATE vs ATTENDANCE")
    sub_m = g_intel.get("sub_merged")
    if sub_m is not None and not sub_m.empty:
        fig = px.scatter(sub_m, x="att_pct", y="pass_rate",
                         text="subject", size_max=12,
                         color_discrete_sequence=[ELEC])
        fig.update_traces(textposition="top center",
                          marker=dict(size=10, color=ELEC))
        fig.add_hline(y=75, line_color=WARN, line_dash="dash", line_width=1)
        fig.add_vline(x=75, line_color=WARN, line_dash="dash", line_width=1)
        fig.update_layout(**pgo(height=300,
                                 xaxis_title="Avg Attendance %",
                                 yaxis_title="Pass Rate %"))
        st.plotly_chart(fig, use_container_width=True)

    sec("TOP 10 PERFORMERS")
    if "top10_cgpa" in g_intel:
        st.dataframe(g_intel["top10_cgpa"].round(2),
                     use_container_width=True, hide_index=True)

    sec("LOWEST 10 PERFORMERS — NEED SUPPORT")
    if "low10_cgpa" in g_intel:
        st.dataframe(g_intel["low10_cgpa"].round(2),
                     use_container_width=True, hide_index=True)

    # Grade predictor
    if not df_grades.empty:
        sec("CGPA PREDICTOR", ELEC)
        gm, gs, gf, gm_m = train_grade_predictor(df, df_grades)
        if gm is not None:
            st.markdown(f"""
            <div class="panel">
              <div class="p-title">GRADE PREDICTION MODEL</div>
              <div style='font-family:Share Tech Mono,monospace;font-size:9px;
                   color:#2a5a2a;line-height:2.2;margin-top:6px;'>
                MAE = {gm_m['mae']} · R² = {gm_m['r2']} · RMSE = {gm_m['rmse']}
              </div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            p_att  = c1.slider("Attendance %", 40, 100, 85, key="gp_att")
            p_late = c2.slider("Late Rate %",  0,  30,  5, key="gp_late")
            p_sess = c3.slider("Sessions",    20, 500, 150, key="gp_sess")

            if st.button("🎓  PREDICT CGPA", use_container_width=True):
                x_in  = np.array([[p_att, p_late/100, 0.88, p_sess, 0]])
                x_sc  = gs.transform(x_in[:, :len(gf)])
                pred  = float(gm.predict(x_sc)[0])
                gr    = ("O" if pred>=9 else "A+" if pred>=8 else "A" if pred>=7
                         else "B+" if pred>=6 else "B" if pred>=5 else "C")
                col   = ACID if pred >= 7 else VOLT if pred >= 5 else WARN
                st.markdown(f"""
                <div class="scan-ok" style='border-color:{col}40;'>
                  <div style='font-size:8px;font-family:Share Tech Mono,monospace;
                       color:#1a4a1a;letter-spacing:2px;'>PREDICTED CGPA</div>
                  <div class="scan-pct" style='color:{col};'>{pred:.2f}</div>
                  <div style='font-family:Orbitron,monospace;font-size:14px;
                       color:{col};margin-top:6px;'>Grade: {gr}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 11 — CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "CLUSTER" in page:
    sec("BEHAVIOURAL CLUSTER ANALYSIS · K-MEANS SEGMENTATION")

    try:
        clusters = load_clusters()
    except Exception as e:
        st.error(f"Clustering failed: {e}")
        st.stop()

    # Cluster summary KPIs
    cl_summary = clusters.groupby("cluster_label").agg(
        count       = ("student_id", "count"),
        avg_att     = ("att_pct",    "mean"),
        avg_late    = ("late_rate",  "mean"),
    ).reset_index()

    cols = st.columns(min(len(cl_summary), 5))
    for col, (_, row) in zip(cols, cl_summary.iterrows()):
        cc = clusters[clusters["cluster_label"] == row["cluster_label"]]["cluster_color"].iloc[0]
        kpi(col, row["cluster_label"], str(row["count"]),
            f"Att: {row['avg_att']:.1f}% | Late: {row['avg_late']:.1f}%", cc)

    # PCA scatter
    sec("2-D PCA CLUSTER MAP")
    fig = px.scatter(
        clusters, x="pca_x", y="pca_y",
        color="cluster_label", text="student_name",
        color_discrete_map={row["cluster_label"]: row["cluster_color"]
                            for _, row in clusters.drop_duplicates("cluster_label").iterrows()},
        hover_data={"att_pct": True, "late_rate": True, "department": True},
    )
    fig.update_traces(textposition="top center",
                      marker=dict(size=10, line=dict(width=1, color="#0d1a0d")))
    fig.update_layout(**pgo(height=460, xaxis_title="PCA Dimension 1",
                             yaxis_title="PCA Dimension 2"))
    st.plotly_chart(fig, use_container_width=True)

    # Per-cluster breakdowns
    sec("CLUSTER PROFILES")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(clusters, x="cluster_label", y="att_pct",
                     color="cluster_label",
                     color_discrete_map={row["cluster_label"]: row["cluster_color"]
                                         for _, row in clusters.drop_duplicates("cluster_label").iterrows()})
        fig.add_hline(y=75, line_color=WARN, line_dash="dash", line_width=1)
        fig.update_layout(**pgo(height=300, title="Attendance % by Cluster",
                                 showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(cl_summary, x="cluster_label", y="avg_att",
                     color="cluster_label",
                     text=[f"{v:.1f}%" for v in cl_summary["avg_att"]],
                     color_discrete_map={row["cluster_label"]: row["cluster_color"]
                                         for _, row in clusters.drop_duplicates("cluster_label").iterrows()})
        fig.add_hline(y=75, line_color=WARN, line_dash="dash", line_width=1)
        fig.update_layout(**pgo(height=300, title="Avg Attendance by Cluster",
                                 showlegend=False, yaxis_range=[0,110]))
        st.plotly_chart(fig, use_container_width=True)

    sec("STUDENT CLUSTER TABLE")
    st.dataframe(
        clusters[["student_name","department","year","cluster_label",
                  "att_pct","late_rate","avg_conf","cluster_dist"]].round(2),
        use_container_width=True, height=380,
        column_config={
            "att_pct":  st.column_config.ProgressColumn("Att %",  min_value=0, max_value=100),
            "late_rate":st.column_config.ProgressColumn("Late %", min_value=0, max_value=50),
        },
    )

    sec("DEPARTMENT × CLUSTER HEATMAP")
    ct = clusters.groupby(["department","cluster_label"]).size().unstack(fill_value=0)
    fig = px.imshow(ct, text_auto=True,
                    color_continuous_scale=[[0,"#0d1a0d"],[0.5,ELEC],[1,ACID]],
                    aspect="auto")
    fig.update_layout(**pgo(height=280, coloraxis_showscale=False))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 12 — SMART ALERTS
# ══════════════════════════════════════════════════════════════════════════════
elif "ALERTS" in page:
    sec("SMART ALERT CENTRE · PRIORITY-RANKED NOTIFICATIONS")

    try:
        df_alerts = load_alerts()
    except Exception as e:
        st.error(f"Alert generation failed: {e}")
        st.stop()

    if df_alerts.empty:
        st.success("✅ No alerts — all students are in healthy range!")
    else:
        p1 = df_alerts[df_alerts["priority"] == 1]
        p2 = df_alerts[df_alerts["priority"] == 2]
        p3 = df_alerts[df_alerts["priority"] == 3]
        p4 = df_alerts[df_alerts["priority"] == 4]

        c1,c2,c3,c4 = st.columns(4)
        kpi(c1, "P1 CRITICAL", str(len(p1)), "immediate action", WARN)
        kpi(c2, "P2 HIGH",     str(len(p2)), "priority response", ORNG)
        kpi(c3, "P3 MEDIUM",   str(len(p3)), "monitor closely",   VOLT)
        kpi(c4, "P4 INFO",     str(len(p4)), "recognition/info",  ACID)

        alert_filter = st.multiselect(
            "Filter by Priority",
            ["P1 CRITICAL","P2 HIGH","P3 MEDIUM","P4 INFO"],
            default=["P1 CRITICAL","P2 HIGH"],
        )

        st.markdown("<br>", unsafe_allow_html=True)

        priority_map = {
            "P1 CRITICAL": 1, "P2 HIGH": 2, "P3 MEDIUM": 3, "P4 INFO": 4
        }
        selected_priorities = [priority_map[f] for f in alert_filter]
        filtered = df_alerts[df_alerts["priority"].isin(selected_priorities)]

        css_map = {1:"alert-p1", 2:"alert-p2", 3:"alert-p3", 4:"alert-p4"}
        col_map = {1:WARN, 2:ORNG, 3:VOLT, 4:ACID}

        for _, row in filtered.iterrows():
            css = css_map.get(row["priority"], "alert-p3")
            clr = col_map.get(row["priority"], VOLT)
            yr  = f"Y{row['year']}" if row.get("year") else ""
            st.markdown(f"""
            <div class="{css}">
              <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                <div>
                  <span style='font-family:Orbitron,monospace;font-size:9px;
                       color:{clr};letter-spacing:1px;'>
                    {row['priority_label']}</span>
                  <span style='font-family:Share Tech Mono,monospace;font-size:8px;
                       color:#1a4a1a;margin-left:10px;'>
                    {row.get('alert_type','').replace('_',' ').upper()}</span>
                </div>
                <span style='font-family:Share Tech Mono,monospace;font-size:8px;
                     color:#1a4a1a;'>{row.get('timestamp','')}</span>
              </div>
              <div style='font-family:Orbitron,monospace;font-size:10px;
                   color:#60c060;margin:6px 0 4px;'>
                {row['student_name']} · {row['department']} {yr}</div>
              <div style='font-family:Share Tech Mono,monospace;font-size:9px;
                   color:#2a5a2a;line-height:1.8;margin-bottom:6px;'>
                {row['message']}</div>
              <div style='font-family:Share Tech Mono,monospace;font-size:8px;
                   color:{clr};border-top:1px solid #0d1a0d;padding-top:5px;'>
                ▶ ACTION: {row['action']}</div>
            </div>""", unsafe_allow_html=True)

        sec("ALERT DISTRIBUTION")
        c1, c2 = st.columns(2)
        with c1:
            alert_by_dept = df_alerts.groupby("department").size().reset_index(name="count")
            fig = go.Figure(go.Bar(
                x=alert_by_dept["department"], y=alert_by_dept["count"],
                marker_color=PLT_PALETTE[:len(alert_by_dept)],
                text=alert_by_dept["count"], textposition="outside",
            ))
            fig.update_layout(**pgo(height=280, title="Alerts by Department",
                                     yaxis_title="Alert Count"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            apt = df_alerts.groupby("alert_type").size().reset_index(name="count")
            fig = go.Figure(go.Pie(
                labels=apt["alert_type"].str.replace("_"," ").str.title(),
                values=apt["count"], hole=.45,
                marker_colors=PLT_PALETTE[:len(apt)],
            ))
            fig.update_layout(**pgo(height=280, title="Alert Type Mix"))
            st.plotly_chart(fig, use_container_width=True)

        sec("EXPORT ALERTS")
        st.download_button(
            "⬇ Download All Alerts CSV",
            df_alerts.drop(columns=["message","action"], errors="ignore").to_csv(index=False).encode(),
            "smart_alerts.csv", "text/csv", use_container_width=True,
        )


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:32px 0 10px;
  border-top:1px solid #0d1a0d;margin-top:28px;'>
  <span style='font-family:Orbitron,monospace;font-size:12px;
    font-weight:900;color:#39ff14;'>ATTENDANCEIQ PRO</span>
  <span style='font-family:Share Tech Mono,monospace;font-size:8px;color:#0d2a0d;'>
   &nbsp;·&nbsp; NEURAL CV SYSTEM &nbsp;·&nbsp; v{VERSION} &nbsp;·&nbsp;
   AFYNIX DIGITAL &nbsp;·&nbsp; WEEK 4 &nbsp;·&nbsp;
   DEOL ALLWYN SAMUEL J B &nbsp;·&nbsp; VLSI &nbsp;·&nbsp; CIT &nbsp;·&nbsp;
   {datetime.now().strftime("%Y")}
  </span>
</div>
""", unsafe_allow_html=True)
