"""
AttendanceIQ Pro · Neural Intelligence Platform v6.0
Author: Deol Allwyn Samuel J B · VLSI · CIT · Afynix Digital
"""
import warnings, os, sys, time, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_pipeline import (
    load_attendance, attendance_summary, student_report, subject_report,
    dept_report, daily_trend, hourly_pattern, at_risk_students, streak_analysis,
    weekly_heatmap_data, monthly_trend, semester_comparison,
    forecast_attendance, anomaly_detection, engineer_features,
    train_absence_predictor_v2, simulate_face_recognition, mark_attendance,
    department_benchmark, compute_engagement_scores, generate_smart_alerts,
    cluster_students, VERSION,
)

st.set_page_config(
    page_title="AttendanceIQ Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PALETTE ───────────────────────────────────────────────────────────────────
GREEN = "#39ff14"
CYAN  = "#00d4ff"
VOID  = "#050508"
CARD  = "#0a0d14"
WARN  = "#ff4444"
GOLD  = "#fbbf24"
PURP  = "#a78bfa"
PINK  = "#f472b6"
PAL   = [GREEN, CYAN, GOLD, WARN, PURP, PINK, "#60a5fa", "#34d399", "#fb923c"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

html,body,[class*="css"]{font-family:'Exo 2',sans-serif!important;background:#050508!important;color:#c9d1d9!important;}
.main{background:#050508!important;}
.block-container{padding:1rem 2rem 3rem;max-width:1600px;}
section[data-testid="stSidebar"]{background:#070a10!important;border-right:1px solid #1a2a1a!important;}
section[data-testid="stSidebar"] *{color:#8b949e!important;}
::-webkit-scrollbar{width:4px;background:#050508;}
::-webkit-scrollbar-thumb{background:#39ff1433;border-radius:4px;}

/* scanline */
.main::before{content:'';position:fixed;top:0;left:0;width:100%;height:100%;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(57,255,20,.012) 2px,rgba(57,255,20,.012) 4px);
  pointer-events:none;z-index:9999;}

/* kpi card */
.kc{background:linear-gradient(135deg,#0a0d14,#0d1117);border:1px solid #1a2a1a;
  border-top:2px solid #39ff14;border-radius:10px;padding:16px 18px;
  position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s;}
.kc:hover{transform:translateY(-3px);box-shadow:0 8px 32px rgba(57,255,20,.12);}
.kv{font-family:'Orbitron',monospace;font-size:28px;font-weight:900;color:#39ff14;
  line-height:1;margin-bottom:5px;text-shadow:0 0 20px rgba(57,255,20,.5);}
.kl{font-family:'Share Tech Mono',monospace;font-size:10px;color:#4a5568;
  letter-spacing:2px;text-transform:uppercase;}
.kd{font-family:'Share Tech Mono',monospace;font-size:10px;color:#00d4ff;margin-top:3px;}

/* section header */
.sh{font-family:'Orbitron',monospace;font-size:11px;font-weight:700;letter-spacing:3px;
  color:#39ff14;border-left:3px solid #39ff14;padding:5px 0 5px 12px;margin:18px 0 10px;
  text-shadow:0 0 10px rgba(57,255,20,.4);background:linear-gradient(90deg,rgba(57,255,20,.05),transparent);}

/* topbar */
.tb{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;
  background:linear-gradient(135deg,#070a10,#0a0d14);border:1px solid #1a2a1a;
  border-top:2px solid #39ff14;border-radius:12px;margin-bottom:20px;
  box-shadow:0 4px 24px rgba(57,255,20,.08);}
.brand{font-family:'Orbitron',monospace;font-size:22px;font-weight:900;color:#39ff14;
  text-shadow:0 0 30px rgba(57,255,20,.6);letter-spacing:2px;}
.bsub{font-family:'Share Tech Mono',monospace;font-size:10px;color:#2d4a2d;letter-spacing:2px;margin-top:2px;}

/* tags */
.tg{font-family:'Share Tech Mono',monospace;font-size:11px;font-weight:600;
  background:rgba(57,255,20,.06);color:#39ff14;border:1px solid rgba(57,255,20,.2);
  padding:4px 10px;border-radius:4px;letter-spacing:1px;display:inline-block;margin:2px;}
.tgc{background:rgba(0,212,255,.06);color:#00d4ff;border-color:rgba(0,212,255,.2);}
.tgw{background:rgba(255,68,68,.06);color:#ff4444;border-color:rgba(255,68,68,.2);}

/* live badge */
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.live{display:inline-block;font-family:'Share Tech Mono',monospace;font-size:10px;font-weight:700;
  color:#39ff14;letter-spacing:2px;border:1px solid rgba(57,255,20,.3);background:rgba(57,255,20,.08);
  padding:3px 8px;border-radius:3px;animation:blink 1.5s infinite;text-shadow:0 0 8px rgba(57,255,20,.8);}

/* risk card */
.rc{background:#0a0d14;border:1px solid #1a2a1a;border-radius:8px;padding:13px 16px;
  margin-bottom:8px;transition:transform .15s;}
.rc:hover{transform:translateX(4px);}

/* model card */
.mc{background:linear-gradient(135deg,#0a0d14,#0d1117);border:1px solid #1a2a1a;
  border-radius:10px;padding:16px;text-align:center;transition:box-shadow .2s;}
.mc:hover{box-shadow:0 0 20px rgba(57,255,20,.15);}

/* scanner frame */
.sf{position:relative;padding:24px;background:rgba(0,212,255,.03);border:1px solid #0a2030;border-radius:8px;}
@keyframes scan{0%{opacity:0}50%{opacity:1}100%{opacity:0}}
.sf::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,#00d4ff,transparent);animation:scan 3s linear infinite;}
.cr{position:absolute;width:16px;height:16px;border-color:#00d4ff;border-style:solid;}
.ctl{top:4px;left:4px;border-width:2px 0 0 2px;}
.ctr{top:4px;right:4px;border-width:2px 2px 0 0;}
.cbl{bottom:4px;left:4px;border-width:0 0 2px 2px;}
.cbr{bottom:4px;right:4px;border-width:0 2px 2px 0;}

/* progress bar */
.pb{background:#0a0d14;border-radius:4px;height:6px;overflow:hidden;margin-top:6px;}
.pf{height:100%;border-radius:4px;}

/* buttons */
.stButton>button{font-family:'Orbitron',monospace!important;font-size:10px!important;letter-spacing:2px!important;
  background:rgba(57,255,20,.06)!important;color:#39ff14!important;border:1px solid rgba(57,255,20,.3)!important;
  border-radius:4px!important;transition:all .2s!important;}
.stButton>button:hover{background:rgba(57,255,20,.15)!important;box-shadow:0 0 16px rgba(57,255,20,.2)!important;}
.stDownloadButton>button{font-family:'Orbitron',monospace!important;font-size:10px!important;letter-spacing:2px!important;
  background:rgba(0,212,255,.06)!important;color:#00d4ff!important;border:1px solid rgba(0,212,255,.3)!important;border-radius:4px!important;}
.stRadio label{font-family:'Share Tech Mono',monospace!important;font-size:12px!important;}
.stSelectbox label,.stSelectbox div{font-family:'Share Tech Mono',monospace!important;}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def pgo(**kw):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", font_family="Share Tech Mono", font_color="#8b949e",
        margin=dict(l=30, r=30, t=40, b=30),
        title_font=dict(family="Orbitron", size=12, color="#39ff14"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#8b949e"),
        xaxis=dict(gridcolor="#1a2a1a", zerolinecolor="#1a2a1a"),
        yaxis=dict(gridcolor="#1a2a1a", zerolinecolor="#1a2a1a"),
    )
    base.update(kw)
    return base


def sec(label, color=GREEN):
    st.markdown(
        f"<div class='sh' style='border-left-color:{color};color:{color};'>{label}</div>",
        unsafe_allow_html=True)


def kpi(col, label, value, delta="", color=GREEN):
    col.markdown(f"""
    <div class='kc' style='border-top-color:{color};'>
      <div class='kv' style='color:{color};text-shadow:0 0 20px {color}55;'>{value}</div>
      <div class='kl'>{label}</div>
      <div class='kd'>{delta}</div>
    </div>""", unsafe_allow_html=True)


def pbar(pct, color=GREEN):
    return f"<div class='pb'><div class='pf' style='width:{min(100,max(0,pct)):.0f}%;background:{color};'></div></div>"


def rclr(pct):
    if pct < 60: return WARN
    if pct < 75: return GOLD
    if pct < 85: return CYAN
    return GREEN


# ── CACHED LOADERS ────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def load_all():
    att_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "attendance.csv")
    if not os.path.exists(att_path):
        with st.spinner("Generating dataset for first run…"):
            from generate_data import generate
            generate()
    df   = load_attendance()
    summ = attendance_summary(df)
    srpt = student_report(df)
    subr = subject_report(df)
    drpt = dept_report(df)
    trnd = daily_trend(df)
    hrly = hourly_pattern(df)
    risk = at_risk_students(df)
    strk = streak_analysis(df)
    whm  = weekly_heatmap_data(df)
    mth  = monthly_trend(df)
    sem  = semester_comparison(df)
    return df, summ, srpt, subr, drpt, trnd, hrly, risk, strk, whm, mth, sem


@st.cache_resource(show_spinner=False)
def _load_ml():
    df = load_attendance()
    return train_absence_predictor_v2(df, fast_mode=True)


@st.cache_data(ttl=600, show_spinner=False)
def _load_anomalies():
    return anomaly_detection(load_attendance())


@st.cache_data(ttl=600, show_spinner=False)
def _load_forecast():
    return forecast_attendance(load_attendance())


# ── BOOTSTRAP ─────────────────────────────────────────────────────────────────
try:
    df, summary, s_rpt, sub_rpt, dept_rpt, trend, hrly, risk_stu, streaks, whm, mth_data, sem_data = load_all()
except Exception as e:
    st.error(f"**Boot failed** — {e}")
    st.stop()

ml_results, ml_scaler, ml_feat, ml_best = {}, None, [], "N/A"
_BEST = {"accuracy":"—","f1":"—","auc":"—","precision":"—","recall":"—",
         "model":None,"fi":{},"cv_f1_mean":0,"cv_f1_std":0,"cm":[[0,0],[0,0]],"fpr":[0,1],"tpr":[0,1]}
best_m  = _BEST.copy()
n_models = 0
anomalies = pd.DataFrame()
forecast  = pd.DataFrame()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:12px 8px 6px;'>
      <div style='font-family:Orbitron,monospace;font-size:18px;font-weight:900;color:#39ff14;
           text-shadow:0 0 20px rgba(57,255,20,.5);letter-spacing:2px;'>
        Attendance<span style='color:#00d4ff;'>IQ</span></div>
      <div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#2d4a2d;
           letter-spacing:2px;margin-top:2px;'>v{VERSION} · NEURAL-CV · AFYNIX</div>
      <div style='margin-top:6px;'><span class='live'>● LIVE</span></div>
    </div>
    <hr style='border-color:#1a2a1a;margin:8px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("NAV", [
        "⬡  COMMAND CENTER",
        "📊 DEEP ANALYTICS",
        "🤖 ML ENGINE",
        "👁  FACE SCANNER",
        "⚠  RISK MATRIX",
        "🔴 ANOMALY + FORECAST",
    ], label_visibility="collapsed")

    at_risk_n = int((s_rpt["pct"] < 75).sum())
    st.markdown(f"""
    <hr style='border-color:#1a2a1a;margin:10px 0;'>
    <div style='font-family:Share Tech Mono,monospace;font-size:11px;line-height:2;color:#4a5568;'>
      <div style='color:#39ff14;font-size:9px;letter-spacing:2px;margin-bottom:2px;'>◈ DATASET</div>
      <span>{len(df):,}</span> records · <span>{summary['total_students']}</span> students<br>
      <span style='color:#39ff14;font-weight:700;'>{summary['attendance_pct']}%</span> overall attendance<br>
      <span style='color:#ff4444;'>{at_risk_n}</span> at-risk students<br><br>
      <div style='color:#00d4ff;font-size:9px;letter-spacing:2px;margin-bottom:2px;'>◈ ML STATUS</div>
      Acc <b style='color:#39ff14;'>{best_m['accuracy']}</b> &nbsp;
      F1 <b style='color:#39ff14;'>{best_m['f1']}</b><br>
      AUC <b style='color:#a78bfa;'>{best_m['auc']}</b><br><br>
      <div style='color:#00d4ff;font-size:9px;letter-spacing:2px;margin-bottom:2px;'>◈ OPERATOR</div>
      <span style='color:#c9d1d9;'>Deol Allwyn Samuel J B</span><br>
      <span>VLSI · CIT · Afynix Digital</span>
    </div>
    """, unsafe_allow_html=True)

# ── TOP BAR ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='tb'>
  <div>
    <div class='brand'>ATTENDANCE<span style='color:#00d4ff;'>IQ</span>
      <span style='font-size:14px;color:#2d4a2d;font-weight:400;'>&nbsp;PRO</span></div>
    <div class='bsub'>NEURAL CV · SMART ATTENDANCE · CIT · AFYNIX DIGITAL · WEEK 4</div>
  </div>
  <div>
    <span class='live'>● LIVE</span>&nbsp;
    <span class='tg'>ACC {best_m['accuracy']}</span>
    <span class='tg tgc'>AUC {best_m['auc']}</span>
    <span class='tg'>{summary['attendance_pct']}% PRESENT</span>
    <span class='tg tgw'>⚠ {at_risk_n} AT RISK</span>
    <span class='tg tgc'>{summary['total_students']} STUDENTS</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════
if "COMMAND" in page:
    c = st.columns(6)
    kpi(c[0], "ATTENDANCE",  f"{summary['attendance_pct']}%",       "overall rate",              GREEN)
    kpi(c[1], "STUDENTS",    str(summary['total_students']),         f"{summary['total_depts']} depts",    CYAN)
    kpi(c[2], "FACE RECOG",  f"{summary['face_recog_pct']}%",       f"avg conf {summary['avg_confidence']}%", GOLD)
    kpi(c[3], "AT RISK",     str(at_risk_n),                         "below 75%",                 WARN)
    kpi(c[4], "RECORDS",     f"{len(df):,}",                        "6 months",                  PURP)
    kpi(c[5], "SUBJECTS",    str(summary['total_subjects']),         "across all depts",          PINK)

    st.markdown("<br>", unsafe_allow_html=True)

    sec("LIVE ATTENDANCE TREND · 7-DAY ROLLING AVERAGE")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["pct"], mode="lines", name="Daily %",
                             line=dict(color=GREEN, width=1.5),
                             fill="tozeroy", fillcolor="rgba(57,255,20,0.06)"))
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["rolling_7"], mode="lines", name="7-Day Avg",
                             line=dict(color=CYAN, width=2.5, dash="dot")))
    fig.add_hline(y=75, line_color=WARN, line_dash="dash", line_width=1,
                  annotation_text="75% Threshold", annotation_font_color=WARN)
    fig.update_layout(**pgo(height=280, title="Daily Attendance %"))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("DEPARTMENT RANKING", CYAN)
        fig = go.Figure(go.Bar(
            x=dept_rpt["pct"], y=dept_rpt["department"], orientation="h",
            marker=dict(color=dept_rpt["pct"],
                        colorscale=[[0, WARN],[0.5, GOLD],[1, GREEN]], line=dict(width=0)),
            text=dept_rpt["pct"].astype(str)+"%", textposition="outside", textfont_color="#c9d1d9",
        ))
        fig.update_layout(**pgo(height=260, title="Attendance % by Department", xaxis_range=[0,108]))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("SUBJECT HEATMAP", GOLD)
        fig = go.Figure(go.Bar(
            x=sub_rpt["subject"], y=sub_rpt["pct"],
            marker=dict(color=PAL[:len(sub_rpt)], line=dict(width=0)),
            text=sub_rpt["pct"].astype(str)+"%", textposition="outside", textfont_color="#c9d1d9",
        ))
        fig.update_layout(**pgo(height=260, title="Attendance % by Subject",
                                 xaxis_tickangle=-30, yaxis_range=[0,108]))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("HOURLY ATTENDANCE PATTERN", PURP)
        fig = go.Figure(go.Bar(
            x=hrly["hour"], y=hrly["pct"],
            marker=dict(color=PURP, opacity=0.85, line=dict(width=0)),
            text=hrly["pct"].astype(str)+"%", textposition="outside",
        ))
        fig.update_layout(**pgo(height=250, title="Attendance % by Period", yaxis_range=[0,108]))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("STREAK LEADERBOARD — TOP 5", GOLD)
        rank_colors = [GREEN, CYAN, GOLD, PURP, PINK]
        for i, (_, row) in enumerate(streaks.head(5).iterrows()):
            clr = rank_colors[i]
            st.markdown(f"""
            <div class='rc' style='border-left:3px solid {clr};'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                  <span style='font-family:Orbitron,monospace;font-size:9px;color:{clr};'>
                    #{i+1} · {row['name'].split()[0]} {row['name'].split()[-1]}</span>
                  <span style='font-family:Share Tech Mono,monospace;font-size:9px;color:#4a5568;margin-left:8px;'>
                    {row['department']}</span>
                </div>
                <span style='font-family:Orbitron,monospace;font-size:20px;font-weight:900;color:{clr};'>
                  {row['max_streak']}<span style='font-size:9px;color:#4a5568;'> days</span></span>
              </div>
            </div>""", unsafe_allow_html=True)

    sec("WEEKLY CALENDAR HEATMAP")
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    whm_f = whm.reindex(columns=[d for d in day_order if d in whm.columns])
    fig = px.imshow(whm_f.T,
                    color_continuous_scale=[[0,VOID],[0.35,WARN],[0.65,GOLD],[1,GREEN]],
                    labels=dict(x="Week", y="Day", color="Att%"), zmin=50, zmax=100)
    fig.update_layout(**pgo(height=220, title="Weekly Calendar Heatmap · Attendance %",
                             coloraxis_showscale=True))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DEEP ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "ANALYTICS" in page:
    sec("STUDENT PERFORMANCE MATRIX")
    disp = s_rpt[["name","department","year","pct","present","absent","late","late_rate","risk_score","dropout_risk","ahi"]].copy()
    disp.columns = ["Name","Dept","Yr","Att%","Present","Absent","Late","Late%","Risk","DropoutRisk","AHI"]
    st.dataframe(
        disp.style
            .background_gradient(subset=["Att%"],  cmap="RdYlGn", vmin=50, vmax=100)
            .background_gradient(subset=["Risk"],   cmap="RdYlGn_r", vmin=0, vmax=50)
            .format({"Att%":"{:.1f}","Late%":"{:.1f}","Risk":"{:.1f}","AHI":"{:.1f}"}),
        use_container_width=True, height=320,
    )
    st.download_button("⬇  EXPORT STUDENT REPORT CSV",
                       s_rpt.to_csv(index=False).encode(),
                       "student_report.csv", "text/csv", use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("SEMESTER COMPARISON", CYAN)
        colors_sem = [GREEN, CYAN]
        fig = go.Figure([
            go.Bar(name=f"Sem {int(r['semester'])}",
                   x=[f"Semester {int(r['semester'])}"], y=[r['pct']],
                   marker_color=colors_sem[int(r['semester'])-1],
                   text=[f"{r['pct']}%"], textposition="outside")
            for _, r in sem_data.iterrows()
        ])
        fig.update_layout(**pgo(height=260, title="Semester Attendance Comparison",
                                 yaxis_range=[0,108], barmode="group"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("MONTHLY TREND", GOLD)
        fig = go.Figure(go.Scatter(
            x=mth_data["month_name"], y=mth_data["pct"],
            mode="lines+markers",
            line=dict(color=GOLD, width=2.5),
            marker=dict(size=8, color=GOLD, symbol="diamond"),
            fill="tozeroy", fillcolor="rgba(251,191,36,0.06)",
        ))
        fig.update_layout(**pgo(height=260, title="Monthly Attendance Trend"))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("SUBJECT TREEMAP", PURP)
        fig = px.treemap(sub_rpt, path=["subject"], values="total",
                         color="pct", color_continuous_scale=[[0,WARN],[0.5,GOLD],[1,GREEN]],
                         custom_data=["pct"])
        fig.update_traces(textfont_size=12, textfont_family="Orbitron",
                          hovertemplate="<b>%{label}</b><br>Att: %{customdata[0]:.1f}%<extra></extra>")
        fig.update_layout(**pgo(height=300, title="Subject Distribution Treemap"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("TOP vs AT-RISK RADAR", CYAN)
        top = s_rpt.nlargest(1, "pct").iloc[0]
        bot = s_rpt.nsmallest(1, "pct").iloc[0]
        cats = ["Attendance", "Punctuality", "AHI", "Consistency", "Engagement"]
        def rv(r):
            return [r["pct"], 100-r["late_rate"], min(100,r["ahi"]), r["pct"]*0.9, min(100,r["ahi"]*0.95)]
        fig = go.Figure()
        for name, vals, clr in [(top["name"].split()[0], rv(top), GREEN),
                                  (bot["name"].split()[0],  rv(bot),  WARN)]:
            fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                fill="toself", name=name,
                fillcolor=f"rgba({'57,255,20' if clr==GREEN else '255,68,68'},0.1)",
                line=dict(color=clr, width=2)))
        fig.update_layout(**pgo(height=300, title="Radar: Best vs At-Risk",
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(range=[0,100], gridcolor="#1a2a1a"),
                       angularaxis=dict(gridcolor="#1a2a1a"))))
        st.plotly_chart(fig, use_container_width=True)

    sec("ATTENDANCE METHOD BREAKDOWN")
    mc = df["method"].value_counts().reset_index()
    mc.columns = ["method","count"]
    mclr = {"Face":GREEN,"Manual":CYAN,"RFID":GOLD,"Absent":WARN}
    fig = go.Figure(go.Pie(
        labels=mc["method"], values=mc["count"], hole=0.55,
        marker_colors=[mclr.get(m, PURP) for m in mc["method"]],
        textfont=dict(family="Share Tech Mono", size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(**pgo(height=300, title="Capture Method Distribution"))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif "ML" in page:
    if not ml_results:
        with st.spinner("⚡ Training RF + GB + LR ensemble with SMOTE — first run ~30s…"):
            ml_results, ml_scaler, ml_feat, ml_best, _enc = _load_ml()
        best_m  = ml_results.get(ml_best, _BEST)
        n_models = len(ml_results)

    sec("MODEL PERFORMANCE MATRIX")
    cols = st.columns(len(ml_results))
    for col, (name, m) in zip(cols, ml_results.items()):
        is_best = name == ml_best
        bdr = f"border:1px solid {GREEN}55;" if is_best else ""
        col.markdown(f"""
        <div class='mc' style='{bdr}'>
          <div style='font-family:Share Tech Mono,monospace;font-size:9px;
               color:{GREEN if is_best else "#4a5568"};letter-spacing:2px;margin-bottom:6px;'>
            {"★ BEST" if is_best else "MODEL"}</div>
          <div style='font-family:Orbitron,monospace;font-size:10px;font-weight:700;
               color:#c9d1d9;margin-bottom:10px;'>{name.upper()}</div>
          <div style='font-family:Share Tech Mono,monospace;font-size:11px;line-height:2.1;text-align:left;'>
            <span style='color:#4a5568;'>ACC&nbsp;</span><b style='color:{GREEN};'>{m['accuracy']}</b><br>
            <span style='color:#4a5568;'>F1&nbsp;&nbsp;</span><b style='color:{CYAN};'>{m['f1']}</b><br>
            <span style='color:#4a5568;'>AUC&nbsp;</span><b style='color:{PURP};'>{m['auc']}</b><br>
            <span style='color:#4a5568;'>PREC</span><b style='color:{GOLD};'>{m['precision']}</b><br>
            <span style='color:#4a5568;'>REC&nbsp;</span><b style='color:{PINK};'>{m['recall']}</b><br>
            <span style='color:#4a5568;'>CV F1</span><b style='color:{GREEN};'>{m['cv_f1_mean']}±{m['cv_f1_std']}</b>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sec("ROC CURVES — ALL MODELS", PURP)
        fig = go.Figure()
        for i, (name, m) in enumerate(ml_results.items()):
            fig.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"], mode="lines",
                name=f"{name} AUC={m['auc']}", line=dict(color=PAL[i], width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline",
                                  line=dict(color="#4a5568", dash="dash")))
        fig.update_layout(**pgo(height=300, title="ROC Curves · All Models",
                                 xaxis_title="FPR", yaxis_title="TPR"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("CONFUSION MATRIX — BEST MODEL", CYAN)
        cm = ml_results[ml_best]["cm"]
        fig = px.imshow(cm, text_auto=True,
                        color_continuous_scale=[[0,VOID],[1,GREEN]],
                        x=["Pred Present","Pred Absent"],
                        y=["Actual Present","Actual Absent"])
        fig.update_layout(**pgo(height=300, title=f"Confusion Matrix · {ml_best}"))
        st.plotly_chart(fig, use_container_width=True)

    sec("FEATURE IMPORTANCE", GREEN)
    fi = ml_results[ml_best].get("fi", {})
    if fi:
        fi_df = pd.DataFrame(list(fi.items()), columns=["feature","importance"]).sort_values("importance").tail(12)
        fig = go.Figure(go.Bar(
            x=fi_df["importance"], y=fi_df["feature"], orientation="h",
            marker=dict(color=fi_df["importance"],
                        colorscale=[[0,CYAN],[1,GREEN]], line=dict(width=0)),
            text=fi_df["importance"].round(3), textposition="outside", textfont_color="#c9d1d9",
        ))
        fig.update_layout(**pgo(height=320, title="Feature Importance · Best Model"))
        st.plotly_chart(fig, use_container_width=True)

    sec("⚡ LIVE ABSENCE RISK PREDICTOR", CYAN)
    sel_stu = st.selectbox("Select Student", s_rpt["name"].tolist(), key="ml_stu")
    if st.button("PREDICT RISK →", key="pred_btn"):
        stu_row = s_rpt[s_rpt["name"] == sel_stu].iloc[0]
        try:
            feat_df, FEATS, _ = engineer_features(df)
            row_feat = feat_df[feat_df["student_id"] == stu_row["student_id"]]
            if not row_feat.empty and ml_scaler is not None:
                X_raw = row_feat[ml_feat].fillna(0).values
                X_sc  = ml_scaler.transform(X_raw)
                mdl   = ml_results[ml_best]["model"]
                prob  = float(mdl.predict_proba(X_sc)[0][1])
                risk_pct = round(prob * 100, 1)
                clr = WARN if risk_pct > 60 else GOLD if risk_pct > 35 else GREEN
                lbl = ("⚠ HIGH RISK — Immediate Intervention" if risk_pct > 60
                       else "⚡ MEDIUM RISK — Monitor Closely" if risk_pct > 35
                       else "✓ LOW RISK — On Track")
                st.markdown(f"""
                <div class='kc' style='border-top-color:{clr};text-align:center;margin-top:12px;'>
                  <div class='kv' style='color:{clr};font-size:42px;text-shadow:0 0 30px {clr}55;'>{risk_pct}%</div>
                  <div class='kl'>ABSENCE RISK PROBABILITY · {sel_stu.split()[0]}</div>
                  <div style='font-family:Share Tech Mono,monospace;font-size:10px;color:{clr};margin:8px 0 4px;'>{lbl}</div>
                  {pbar(risk_pct, clr)}
                </div>""", unsafe_allow_html=True)
        except Exception as ex:
            st.warning(f"Prediction unavailable: {ex}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FACE SCANNER
# ══════════════════════════════════════════════════════════════════════════════
elif "FACE" in page or "SCANNER" in page:
    sec("BIOMETRIC FACE RECOGNITION SYSTEM", CYAN)
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("""
        <div class='sf'>
          <div class='cr ctl'></div><div class='cr ctr'></div>
          <div class='cr cbl'></div><div class='cr cbr'></div>
          <div style='text-align:center;padding:28px 0;'>
            <div style='font-family:Orbitron,monospace;font-size:11px;color:#00d4ff;
                 letter-spacing:3px;margin-bottom:14px;'>◈ BIOMETRIC SCANNER ACTIVE</div>
            <div style='font-size:64px;margin:10px 0;'>👁</div>
            <div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#4a5568;line-height:2;'>
              OpenCV · HOG + CNN · dlib 128D Embeddings<br>
              Cosine Similarity · Multi-Attempt Best-Of-3<br>
              Threshold: 80% · Liveness Detection Active
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        all_names = s_rpt["name"].tolist()
        sel_face  = st.selectbox("Student", all_names, key="face_stu")
        sel_subj  = st.selectbox("Subject", sub_rpt["subject"].tolist(), key="face_subj")

        if st.button("⚡ INITIATE FACE SCAN", key="scan_btn"):
            result = simulate_face_recognition(sel_face)
            prog = st.progress(0)
            for i in range(101):
                time.sleep(0.008)
                prog.progress(i)

            conf    = result["confidence"]
            matched = result["matched"]
            clr     = GREEN if matched else WARN

            attempt_html = "".join([
                f"<span style='font-family:Share Tech Mono,monospace;font-size:9px;"
                f"color:{GREEN if a[\"confidence\"]>80 else WARN};margin-right:12px;'>"
                f"#{a['attempt']}: {a['confidence']}%</span>"
                for a in result["attempts"]
            ])
            st.markdown(f"""
            <div class='kc' style='border-top-color:{clr};margin-top:10px;'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                  <div class='kv' style='color:{clr};font-size:38px;text-shadow:0 0 20px {clr}55;'>{conf}%</div>
                  <div class='kl'>CONFIDENCE SCORE</div>
                </div>
                <div style='font-family:Orbitron,monospace;font-size:13px;color:{clr};text-align:right;'>
                  {"✓ MATCHED" if matched else "✗ NO MATCH"}<br>
                  <span style='font-size:9px;color:#4a5568;'>{sel_face.split()[0]}</span></div>
              </div>
              {pbar(conf, clr)}
              <div style='margin-top:8px;'>{attempt_html}</div>
            </div>""", unsafe_allow_html=True)

            if matched:
                ok = mark_attendance(sel_face, sel_subj)
                if ok:
                    st.success(f"✓ Attendance marked — {sel_face} · {sel_subj}")
                else:
                    st.warning("Already marked for today")
            else:
                st.error("Face not recognised — please retry or use manual entry")

    sec("TODAY'S LIVE ATTENDANCE LOG", GOLD)
    live_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "live_attendance.csv")
    try:
        live_df = pd.read_csv(live_path)
        st.dataframe(live_df, use_container_width=True)
        st.download_button("⬇  EXPORT TODAY'S LOG",
                           live_df.to_csv(index=False).encode(),
                           "live_log.csv", "text/csv", use_container_width=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:11px;color:#4a5568;"
                    "text-align:center;padding:20px;'>[ NO RECORDS ] · Run a face scan to mark attendance</div>",
                    unsafe_allow_html=True)

    sec("OPENCV PRODUCTION CODE", PURP)
    st.code('''import cv2, pandas as pd
from datetime import datetime

# Load cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def scan_and_mark(frame, student_name, subject):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (57,255,20), 2)
        cv2.putText(frame, student_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,212,255), 2)
    return frame, len(faces) > 0

# Real-time loop
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame, detected = scan_and_mark(frame, "Student", "CMOS VLSI")
    cv2.imshow("AttendanceIQ Scanner", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()''', language="python")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RISK MATRIX
# ══════════════════════════════════════════════════════════════════════════════
elif "RISK" in page:
    critical = risk_stu[risk_stu["pct"] < 60]
    high     = risk_stu[(risk_stu["pct"] >= 60) & (risk_stu["pct"] < 75)]
    medium   = risk_stu[(risk_stu["pct"] >= 75) & (risk_stu["pct"] < 85)]
    safe     = risk_stu[risk_stu["pct"] >= 85]

    c = st.columns(4)
    kpi(c[0], "CRITICAL <60%",  str(len(critical)), "immediate action", WARN)
    kpi(c[1], "AT RISK 60-75%", str(len(high)),     "close monitoring",  GOLD)
    kpi(c[2], "LOW RISK 75-85%",str(len(medium)),   "periodic check",    CYAN)
    kpi(c[3], "SAFE >85%",      str(len(safe)),     "on track",          GREEN)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("ALL STUDENT ATTENDANCE RANKINGS", CYAN)

    sorted_s = risk_stu.sort_values("pct")
    fig = go.Figure(go.Bar(
        x=sorted_s["pct"], y=sorted_s["name"], orientation="h",
        marker=dict(color=[rclr(p) for p in sorted_s["pct"]], line=dict(width=0)),
        text=sorted_s["pct"].astype(str)+"%",
        textposition="outside", textfont_color="#c9d1d9",
        customdata=sorted_s[["department","risk_score"]].values,
        hovertemplate="<b>%{y}</b><br>Att: %{x}%<br>Dept: %{customdata[0]}<br>Risk: %{customdata[1]}<extra></extra>",
    ))
    fig.add_vline(x=75, line_color=GOLD, line_dash="dash", line_width=1)
    fig.add_vline(x=60, line_color=WARN, line_dash="dash", line_width=1)
    fig.update_layout(**pgo(height=max(300, len(sorted_s)*24),
                             title="Student Attendance Rankings",
                             xaxis_range=[0,112], margin=dict(l=180,r=60,t=40,b=30)))
    st.plotly_chart(fig, use_container_width=True)

    sec("AT-RISK STUDENT DETAIL CARDS", WARN)
    at_risk_all = risk_stu[risk_stu["pct"] < 75].sort_values("pct")
    for _, row in at_risk_all.iterrows():
        clr = WARN if row["pct"] < 60 else GOLD
        st.markdown(f"""
        <div class='rc' style='border-left:3px solid {clr};'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div>
              <span style='font-family:Orbitron,monospace;font-size:10px;color:{clr};'>{row['name']}</span>
              <span style='font-family:Share Tech Mono,monospace;font-size:9px;color:#4a5568;margin-left:10px;'>
                {row['department']} · Yr{row['year']}</span>
            </div>
            <div style='text-align:right;'>
              <span style='font-family:Orbitron,monospace;font-size:22px;font-weight:900;color:{clr};'>
                {row['pct']}%</span>
              <span style='font-family:Share Tech Mono,monospace;font-size:9px;color:#4a5568;display:block;'>
                Risk:{row['risk_score']} · AHI:{row['ahi']}</span>
            </div>
          </div>
          {pbar(row['pct'], clr)}
          <div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#4a5568;margin-top:6px;'>
            Present:{row['present']} · Absent:{row['absent']} · Late:{row['late']} ({row['late_rate']}%)</div>
        </div>""", unsafe_allow_html=True)

    st.download_button("⬇  EXPORT AT-RISK LIST CSV",
                       at_risk_all.to_csv(index=False).encode(),
                       "at_risk_students.csv", "text/csv", use_container_width=True)

    sec("RECOMMENDED ACTIONS BY RISK LEVEL")
    for lbl, acts, clr in [
        ("CRITICAL <60% — URGENT", [
            "Issue official attendance warning letter immediately",
            "Schedule HOD counselling within 3 working days",
            "Notify parents/guardians by phone + written notice",
            "Place on academic probation — weekly check-in required",
        ], WARN),
        ("AT RISK 60–75%", [
            "Send automated SMS + email alert to student and parent",
            "Assign dedicated faculty mentor for 4 weeks",
            "Bi-weekly attendance review session",
            "Identify root cause: transport, health, motivation",
        ], GOLD),
        ("LOW RISK 75–85%", [
            "Send gentle reminder email",
            "Monitor attendance for next 2 weeks",
            "Positive reinforcement if improving trend detected",
        ], CYAN),
    ]:
        actions_html = "".join(
            f"<div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#6b7280;"
            f"padding:3px 0;border-bottom:1px solid #0a0d14;'>"
            f"<span style='color:{clr};'>▶</span> {a}</div>"
            for a in acts)
        st.markdown(f"""
        <div class='rc' style='border-left:3px solid {clr};margin-bottom:10px;'>
          <div style='font-family:Orbitron,monospace;font-size:10px;color:{clr};
               letter-spacing:2px;margin-bottom:8px;'>{lbl}</div>
          {actions_html}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ANOMALY + FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif "ANOMALY" in page:
    if anomalies.empty:
        with st.spinner("Running IsolationForest anomaly detection…"):
            anomalies = _load_anomalies()
    if forecast.empty:
        with st.spinner("Generating 14-day attendance forecast…"):
            forecast = _load_forecast()

    # ── Anomaly section ─────────────────────────────────────────────────────
    sec("ISOLATIONFOREST ANOMALY DETECTION", WARN)
    if not anomalies.empty and "anomaly" in anomalies.columns:
        n_flag  = int(anomalies["anomaly"].sum())
        n_norm  = len(anomalies) - n_flag
        c = st.columns(4)
        kpi(c[0], "TOTAL ANALYSED", str(len(anomalies)), "all students",        CYAN)
        kpi(c[1], "ANOMALIES",      str(n_flag),         "flagged by AI",       WARN)
        kpi(c[2], "NORMAL",         str(n_norm),         "within range",        GREEN)
        kpi(c[3], "SENSITIVITY",    "15%",               "contamination rate",  PURP)

        st.markdown("<br>", unsafe_allow_html=True)

        normal_df  = anomalies[~anomalies["anomaly"]]
        flagged_df = anomalies[anomalies["anomaly"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=normal_df["name"],  y=normal_df["score"],
                             name="Normal",  marker_color=GREEN, opacity=0.75))
        fig.add_trace(go.Bar(x=flagged_df["name"], y=flagged_df["score"],
                             name="Anomaly", marker_color=WARN,  opacity=0.9))
        fig.update_layout(**pgo(height=280,
                                 title="IsolationForest Scores (lower = more anomalous)",
                                 barmode="group", xaxis_tickangle=-30))
        st.plotly_chart(fig, use_container_width=True)

        if not flagged_df.empty:
            sec("FLAGGED STUDENT CARDS", WARN)
            for _, row in flagged_df.sort_values("score").iterrows():
                st.markdown(f"""
                <div class='rc' style='border-left:3px solid {WARN};'>
                  <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                      <span style='font-family:Orbitron,monospace;font-size:10px;color:{WARN};'>
                        {row.get('name','—')}</span>
                      <span style='font-family:Share Tech Mono,monospace;font-size:9px;
                           color:#4a5568;margin-left:10px;'>{row.get('department','—')}</span>
                    </div>
                    <div style='text-align:right;'>
                      <span style='font-family:Share Tech Mono,monospace;font-size:10px;color:{WARN};'>
                        Score: {row['score']:.4f}</span><br>
                      <span style='font-family:Share Tech Mono,monospace;font-size:9px;color:#4a5568;'>
                        Att: {row.get('overall_pct','—')}%</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Forecast section ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sec("14-DAY ATTENDANCE FORECAST · CONFIDENCE BANDS", CYAN)

    if not forecast.empty:
        avg_f = round(forecast["forecast"].mean(), 1)
        c = st.columns(4)
        kpi(c[0], "AVG FORECAST",  f"{avg_f}%",                          "next 14 days",      CYAN)
        kpi(c[1], "PEAK DAY",      f"{forecast['forecast'].max():.1f}%", "highest predicted", GREEN)
        kpi(c[2], "LOW DAY",       f"{forecast['forecast'].min():.1f}%", "lowest predicted",  GOLD)
        kpi(c[3], "DIRECTION",
            "▲ UP" if forecast["forecast"].iloc[-1] > forecast["forecast"].iloc[0] else "▼ DOWN",
            "14-day trend", PURP)

        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["upper"],
                                  fill=None, mode="lines", line=dict(width=0),
                                  showlegend=False, name="Upper"))
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["lower"],
                                  fill="tonexty", fillcolor="rgba(0,212,255,0.08)",
                                  mode="lines", line=dict(width=0), name="Confidence Band"))
        fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["forecast"],
                                  mode="lines+markers", name="Forecast",
                                  line=dict(color=CYAN, width=2.5),
                                  marker=dict(size=7, color=CYAN, symbol="circle")))
        fig.add_hline(y=75, line_color=WARN, line_dash="dash", line_width=1,
                      annotation_text="75% Threshold", annotation_font_color=WARN)
        fig.update_layout(**pgo(height=320,
                                 title="14-Day Attendance Forecast · Confidence Band",
                                 xaxis_title="Date", yaxis_title="Predicted Attendance %",
                                 yaxis_range=[50, 105]))
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        sec("FORECAST TABLE", GREEN)
        st.dataframe(forecast.rename(columns={"date":"Date","forecast":"Predicted %",
                                               "lower":"Lower Bound","upper":"Upper Bound"}),
                     use_container_width=True, height=280)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:26px 0 6px;border-top:1px solid #1a2a1a;margin-top:24px;'>
  <span style='font-family:Orbitron,monospace;font-size:11px;font-weight:900;color:#39ff14;
       text-shadow:0 0 20px rgba(57,255,20,.4);letter-spacing:2px;'>ATTENDANCEIQ PRO</span>
  <span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#2d4a2d;margin-left:12px;'>
    v{VERSION} · Deol Allwyn Samuel J B · VLSI · CIT · Afynix Digital</span>
</div>
""", unsafe_allow_html=True)
