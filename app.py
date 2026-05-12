"""
Performix: Intelligent System for Identifying Workforce Readiness
and Performance Stability in Entry-Level Professionals
────────────────────────────────────────────────────────
Premium UI Edition — Dark Glassmorphism Design System
Single-file Streamlit Application (Streamlit Cloud compatible)
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Performix Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    /* ── Root Palette ── */
    :root {
        --bg-base:       #070B14;
        --bg-surface:    #0D1424;
        --bg-card:       #111827;
        --bg-glass:      rgba(17,24,39,0.75);
        --border:        rgba(99,179,237,0.12);
        --border-bright: rgba(99,179,237,0.35);
        --accent-blue:   #3B82F6;
        --accent-cyan:   #06B6D4;
        --accent-violet: #8B5CF6;
        --accent-green:  #10B981;
        --accent-amber:  #F59E0B;
        --accent-red:    #EF4444;
        --text-primary:  #F0F6FF;
        --text-secondary:#94A3B8;
        --text-muted:    #475569;
        --glow-blue:     0 0 40px rgba(59,130,246,0.25);
        --glow-cyan:     0 0 40px rgba(6,182,212,0.20);
        --radius-lg:     16px;
        --radius-md:     10px;
    }

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
    }

    /* ── Streamlit overrides ── */
    .stApp { background: var(--bg-base) !important; }
    .main .block-container {
        padding: 1.5rem 2rem 3rem !important;
        max-width: 1400px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 3px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080D1A 0%, #0B1221 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div { padding-top: 0 !important; }
    [data-testid="stSidebar"] .stRadio label {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
        padding: 0.3rem 0 !important;
        transition: color 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover { color: var(--accent-cyan) !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMarkdown p { color: var(--text-secondary) !important; }
    [data-testid="stSidebar"] h3 { color: var(--text-primary) !important; }

    /* ── Selectbox / Inputs ── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    .stSelectbox > div > div:hover,
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
    }
    [data-baseweb="select"] { background: var(--bg-card) !important; }
    [data-baseweb="menu"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; }

    /* ── Slider ── */
    .stSlider > div > div > div { background: var(--accent-blue) !important; }
    .stSlider [data-testid="stTickBar"] { color: var(--text-muted) !important; }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: var(--radius-lg) !important; overflow: hidden !important; }
    .stDataFrame thead th {
        background: var(--bg-surface) !important;
        color: var(--accent-cyan) !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        border-bottom: 1px solid var(--border-bright) !important;
    }
    .stDataFrame tbody tr { background: var(--bg-card) !important; }
    .stDataFrame tbody tr:hover { background: rgba(59,130,246,0.08) !important; }
    .stDataFrame tbody td { border-color: var(--border) !important; color: var(--text-primary) !important; }

    /* ── Metric Cards ── */
    [data-testid="metric-container"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1rem 1.2rem !important;
        backdrop-filter: blur(12px) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="metric-container"]:hover {
        border-color: var(--border-bright) !important;
        box-shadow: var(--glow-blue) !important;
    }
    [data-testid="metric-container"] label {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ── Buttons ── */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 28px rgba(59,130,246,0.45) !important;
    }

    /* ── Forms ── */
    [data-testid="stForm"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(12px) !important;
    }

    /* ── Alerts / Info ── */
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-surface) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border) !important;
        gap: 4px !important;
        padding: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet)) !important;
        color: white !important;
    }

    /* ── Headings ── */
    h1, h2, h3, h4 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; }

    /* ── Custom Components ── */
    .px-hero {
        background: linear-gradient(135deg, #0A1628 0%, #0D1F3C 50%, #0A1628 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .px-hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .px-hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 40%;
        width: 160px; height: 160px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .px-hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0 0 0.35rem;
        line-height: 1.2;
    }
    .px-hero-sub {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin: 0;
        font-weight: 300;
    }
    .px-hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
        color: white;
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.2rem 0.7rem;
        border-radius: 100px;
        margin-bottom: 0.75rem;
    }

    .px-kpi {
        background: var(--bg-glass);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.2rem 1.4rem;
        backdrop-filter: blur(12px);
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .px-kpi::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 3px;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }
    .px-kpi.blue::before  { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
    .px-kpi.green::before { background: linear-gradient(90deg, var(--accent-green), #34D399); }
    .px-kpi.red::before   { background: linear-gradient(90deg, var(--accent-red), #F87171); }
    .px-kpi.amber::before { background: linear-gradient(90deg, var(--accent-amber), #FCD34D); }
    .px-kpi.violet::before{ background: linear-gradient(90deg, var(--accent-violet), #A78BFA); }
    .px-kpi:hover {
        border-color: var(--border-bright);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .px-kpi-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .px-kpi-label {
        font-size: 0.78rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }
    .px-kpi-icon {
        font-size: 1.5rem;
        position: absolute;
        top: 1rem; right: 1rem;
        opacity: 0.35;
    }

    .px-section {
        background: var(--bg-glass);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }
    .px-section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.03em;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }

    .px-tag {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .px-tag.high   { background: rgba(239,68,68,0.15);  color: #F87171; border: 1px solid rgba(239,68,68,0.3); }
    .px-tag.medium { background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); }
    .px-tag.low    { background: rgba(16,185,129,0.15); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.3); }
    .px-tag.ready  { background: rgba(16,185,129,0.15); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.3); }
    .px-tag.partial{ background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); }
    .px-tag.not    { background: rgba(239,68,68,0.15);  color: #F87171; border: 1px solid rgba(239,68,68,0.3); }

    .px-sidebar-logo {
        background: linear-gradient(135deg, #0F1E38, #1A2E52);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .px-sidebar-logo h2 {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        background: linear-gradient(135deg, #60A5FA, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .px-sidebar-logo p {
        font-size: 0.72rem !important;
        color: var(--text-muted) !important;
        margin: 0.2rem 0 0 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .px-profile-card {
        background: var(--bg-glass);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        backdrop-filter: blur(12px);
    }
    .px-avatar {
        width: 56px; height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet));
        display: flex; align-items: center; justify-content: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.75rem;
    }
    .px-profile-name {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .px-profile-meta {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin-top: 0.15rem;
    }
    .px-detail-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border);
        font-size: 0.85rem;
    }
    .px-detail-key { color: var(--text-muted); }
    .px-detail-val { color: var(--text-primary); font-weight: 500; }

    .px-plan-item {
        background: rgba(59,130,246,0.06);
        border: 1px solid rgba(59,130,246,0.18);
        border-left: 3px solid var(--accent-blue);
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .px-plan-item.red  { border-left-color: var(--accent-red); background: rgba(239,68,68,0.06); border-color: rgba(239,68,68,0.18); }
    .px-plan-item.amber{ border-left-color: var(--accent-amber); background: rgba(245,158,11,0.06); border-color: rgba(245,158,11,0.18); }
    .px-plan-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.9rem; color: var(--text-primary); }
    .px-plan-score { font-size: 0.78rem; color: var(--text-secondary); margin: 0.2rem 0 0.5rem; }
    .px-plan-advice { font-size: 0.83rem; color: var(--text-secondary); line-height: 1.5; }

    .px-chip {
        display: inline-block;
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.25);
        color: #93C5FD;
        border-radius: 100px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.15rem;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(11,18,35,0)",
    plot_bgcolor="rgba(11,18,35,0)",
    font=dict(family="DM Sans, sans-serif", color="#94A3B8", size=12),
    title_font=dict(family="Syne, sans-serif", color="#F0F6FF", size=15, weight="bold"),
    xaxis=dict(gridcolor="rgba(99,179,237,0.08)", zerolinecolor="rgba(99,179,237,0.15)",
               linecolor="rgba(99,179,237,0.15)", tickfont=dict(color="#64748B")),
    yaxis=dict(gridcolor="rgba(99,179,237,0.08)", zerolinecolor="rgba(99,179,237,0.15)",
               linecolor="rgba(99,179,237,0.15)", tickfont=dict(color="#64748B")),
    legend=dict(bgcolor="rgba(11,18,35,0.6)", bordercolor="rgba(99,179,237,0.2)", borderwidth=1),
    margin=dict(l=10, r=10, t=45, b=10),
)


def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


RISK_COLORS = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
READ_COLORS = {"Ready": "#10B981", "Partially Ready": "#F59E0B", "Not Ready": "#EF4444"}
BLUE_SCALE = ["#1E3A5F", "#1D4ED8", "#3B82F6", "#60A5FA", "#93C5FD", "#BAE6FD"]

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "study_hours", "screen_time", "quiz_score", "coding_score",
    "attendance", "task_completion", "feedback_rating",
    "engagement_score", "communication_score", "technical_assessment",
    "learning_progression"
]


def generate_dataset(n=300):
    np.random.seed(42)
    ids = [f"EMP{str(i).zfill(4)}" for i in range(1, n + 1)]
    names = [f"Employee_{i}" for i in range(1, n + 1)]
    depts = np.random.choice(["Backend Dev", "Frontend Dev", "Data Science", "QA Testing", "DevOps"], n)
    roles = np.random.choice(["Intern", "Fresh Graduate"], n)
    batches = np.random.choice(["Batch-2023", "Batch-2024", "Batch-2025"], n)

    study_hours = np.round(np.random.normal(5.5, 2.0, n).clip(0, 12), 1)
    screen_time = np.round(np.random.normal(6.0, 1.5, n).clip(1, 14), 1)
    quiz_score = np.round(np.random.normal(65, 15, n).clip(0, 100), 1)
    coding_score = np.round(np.random.normal(60, 18, n).clip(0, 100), 1)
    attendance = np.round(np.random.normal(78, 12, n).clip(30, 100), 1)
    task_completion = np.round(np.random.normal(72, 15, n).clip(0, 100), 1)
    feedback_rating = np.round(np.random.uniform(1, 5, n), 1)
    engagement_score = np.round(np.random.normal(65, 18, n).clip(0, 100), 1)
    communication_score = np.round(np.random.normal(60, 15, n).clip(0, 100), 1)
    technical_assessment = np.round(np.random.normal(62, 16, n).clip(0, 100), 1)
    learning_progression = np.round(np.random.normal(5, 3, n).clip(-10, 20), 1)

    performance_score = np.round((
            0.20 * quiz_score + 0.20 * coding_score +
            0.15 * attendance + 0.15 * task_completion +
            0.10 * feedback_rating * 20 +
            0.10 * engagement_score + 0.10 * technical_assessment
    ), 1).clip(0, 100)

    df = pd.DataFrame({
        "employee_id": ids, "name": names, "department": depts,
        "role": roles, "batch": batches,
        "study_hours": study_hours, "screen_time": screen_time,
        "quiz_score": quiz_score, "coding_score": coding_score,
        "attendance": attendance, "task_completion": task_completion,
        "feedback_rating": feedback_rating, "engagement_score": engagement_score,
        "communication_score": communication_score,
        "technical_assessment": technical_assessment,
        "learning_progression": learning_progression,
        "performance_score": performance_score,
    })

    def assign_risk(row):
        s = (row.attendance * 0.2 + row.task_completion * 0.2 + row.quiz_score * 0.15
             + row.coding_score * 0.15 + row.engagement_score * 0.15
             + row.feedback_rating * 20 * 0.15)
        return "High" if s < 60 else ("Medium" if s < 75 else "Low")

    df["fatigue_risk"] = df.apply(assign_risk, axis=1)
    df["readiness_level"] = pd.cut(
        df["performance_score"],
        bins=[-1, 54.9, 74.9, 100],
        labels=["Not Ready", "Partially Ready", "Ready"]
    ).astype(str)
    return df


def get_feature_matrix(df):
    return df[FEATURE_COLS].copy(), df["fatigue_risk"], df["readiness_level"]


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def compute_summary(df):
    return {
        "total": len(df),
        "avg_perf": round(float(df["performance_score"].mean()), 2),
        "avg_attend": round(float(df["attendance"].mean()), 2),
        "avg_task": round(float(df["task_completion"].mean()), 2),
        "avg_engage": round(float(df["engagement_score"].mean()), 2),
        "avg_quiz": round(float(df["quiz_score"].mean()), 2),
        "avg_coding": round(float(df["coding_score"].mean()), 2),
        "avg_feedback": round(float(df["feedback_rating"].mean()), 2),
        "high_risk": int((df["fatigue_risk"] == "High").sum()),
        "medium_risk": int((df["fatigue_risk"] == "Medium").sum()),
        "low_risk": int((df["fatigue_risk"] == "Low").sum()),
        "ready": int((df["readiness_level"] == "Ready").sum()),
        "partial": int((df["readiness_level"] == "Partially Ready").sum()),
        "not_ready": int((df["readiness_level"] == "Not Ready").sum()),
    }


def dept_analysis(df):
    return df.groupby("department").agg(
        Employees=("employee_id", "count"),
        Avg_Performance=("performance_score", "mean"),
        Avg_Attendance=("attendance", "mean"),
        Avg_Engagement=("engagement_score", "mean"),
        Avg_Quiz=("quiz_score", "mean"),
        Avg_Coding=("coding_score", "mean"),
        High_Risk=("fatigue_risk", lambda x: (x == "High").sum()),
        Ready_Count=("readiness_level", lambda x: (x == "Ready").sum()),
    ).round(2).reset_index()


def top_performers(df, n=10):
    cols = ["employee_id", "name", "department", "role", "performance_score",
            "attendance", "task_completion", "engagement_score", "readiness_level"]
    return df[cols].sort_values("performance_score", ascending=False).head(n).reset_index(drop=True)


def at_risk_employees(df):
    cols = ["employee_id", "name", "department", "role",
            "performance_score", "attendance", "engagement_score",
            "fatigue_risk", "readiness_level"]
    return df[df["fatigue_risk"] == "High"][cols].sort_values("performance_score").reset_index(drop=True)


def correlation_matrix(df):
    return df[FEATURE_COLS + ["performance_score"]].corr().round(2)


def engagement_segments(df):
    me = df["engagement_score"].median()
    mp = df["performance_score"].median()

    def quad(r):
        if r.engagement_score >= me and r.performance_score >= mp:
            return "High Eng / High Perf"
        elif r.engagement_score >= me:
            return "High Eng / Low Perf"
        elif r.performance_score >= mp:
            return "Low Eng / High Perf"
        else:
            return "Low Eng / Low Perf"

    df = df.copy()
    df["segment"] = df.apply(quad, axis=1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
def train_model(X, y_series):
    le = LabelEncoder()
    y = le.fit_transform(y_series)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    cv = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    report = classification_report(y_te, y_pred, target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    return clf, le, {
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "f1": round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "cm": confusion_matrix(y_te, y_pred),
        "report": report,
        "classes": le.classes_.tolist(),
        "cv_mean": round(float(cv.mean()), 4),
        "cv_std": round(float(cv.std()), 4),
    }


def feature_importance_df(clf):
    fi = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": clf.feature_importances_})
    return fi.sort_values("Importance", ascending=False).reset_index(drop=True)


def predict_employee(clf, le, vals):
    inp = pd.DataFrame([[vals.get(c, 0) for c in FEATURE_COLS]], columns=FEATURE_COLS)
    pred = clf.predict(inp)[0]
    prob = clf.predict_proba(inp)[0]
    label = le.inverse_transform([pred])[0]
    proba = {le.classes_[i]: round(float(prob[i]) * 100, 1) for i in range(len(le.classes_))}
    return label, proba


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — READINESS
# ══════════════════════════════════════════════════════════════════════════════
def suggest_role(row):
    if row.coding_score >= 75 and row.technical_assessment >= 70:
        return "Developer / Engineer"
    elif row.quiz_score >= 70 and row.communication_score >= 65:
        return "Business Analyst"
    elif row.engagement_score >= 70 and row.task_completion >= 70:
        return "Project Coordinator"
    elif row.technical_assessment >= 65 and row.coding_score < 60:
        return "QA / Testing"
    else:
        return "Support / Training"


def skill_gaps_label(row):
    scores = {c: row[c] for c in ["quiz_score", "coding_score", "attendance",
                                  "task_completion", "engagement_score", "communication_score", "technical_assessment"]}
    return ", ".join(k.replace("_", " ").title() for k in sorted(scores, key=scores.get)[:2])


def batch_readiness(df):
    return df.groupby("batch").agg(
        Total=("employee_id", "count"),
        Ready=("readiness_level", lambda x: (x == "Ready").sum()),
        Partial=("readiness_level", lambda x: (x == "Partially Ready").sum()),
        Not_Ready=("readiness_level", lambda x: (x == "Not Ready").sum()),
        Avg_Performance=("performance_score", "mean"),
    ).round(2).reset_index()


def dept_readiness(df):
    return df.groupby("department").agg(
        Total=("employee_id", "count"),
        Ready=("readiness_level", lambda x: (x == "Ready").sum()),
        Partial=("readiness_level", lambda x: (x == "Partially Ready").sum()),
        Not_Ready=("readiness_level", lambda x: (x == "Not Ready").sum()),
        Avg_Performance=("performance_score", "mean"),
        High_Risk=("fatigue_risk", lambda x: (x == "High").sum()),
    ).round(2).reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
THRESHOLDS = {
    "attendance": 70, "quiz_score": 55, "coding_score": 55,
    "task_completion": 60, "engagement_score": 55,
    "communication_score": 55, "technical_assessment": 55,
    "feedback_rating": 3.0, "study_hours": 3.0,
}
TRAINING_CATALOG = {
    "attendance": "📅 Attendance Program: 1-on-1 HR check-ins; identify barriers.",
    "quiz_score": "📚 Knowledge Booster: Weekly quizzes; e-learning (Coursera/Udemy).",
    "coding_score": "💻 Coding Bootcamp: Daily LeetCode; pair programming with seniors.",
    "task_completion": "⏱ Productivity Workshop: Pomodoro technique; Jira/Trello tracking.",
    "engagement_score": "🤝 Engagement Initiative: Team activities, mentorship, recognition.",
    "communication_score": "🗣 Communication Training: Presentation & writing workshops.",
    "technical_assessment": "🔧 Technical Upskilling: Domain certs (AWS, Python, SQL, etc.).",
    "feedback_rating": "🌱 Growth Coaching: Performance reviews; goal-setting with manager.",
    "study_hours": "📖 Study Habit Counseling: Structured daily plans; protected hours.",
}


def get_recommendations(row):
    recs = []
    for metric, threshold in THRESHOLDS.items():
        val = row.get(metric)
        if val is not None and float(val) < threshold:
            recs.append({
                "metric": metric.replace("_", " ").title(),
                "current": round(float(val), 2),
                "target": threshold,
                "advice": TRAINING_CATALOG.get(metric, "General improvement recommended."),
                "gap": round(threshold - float(val), 2),
            })
    return sorted(recs, key=lambda x: x["gap"], reverse=True)


def personalized_plan(row):
    return get_recommendations(row)


def intervention_plan(df):
    rows = []
    for _, r in df[df["fatigue_risk"] == "High"].iterrows():
        recs = get_recommendations(r)
        areas = [x["metric"] for x in recs]
        rows.append({
            "Employee ID": r.employee_id,
            "Name": r["name"],
            "Department": r.department,
            "Performance": r.performance_score,
            "Fatigue Risk": r.fatigue_risk,
            "Readiness": r.readiness_level,
            "Issues Count": len(recs),
            "Priority Areas": ", ".join(areas) if areas else "None",
            "Urgency": "🔴 Immediate" if len(recs) >= 4 else ("🟡 Moderate" if len(recs) >= 2 else "🟢 Low"),
        })
    return pd.DataFrame(rows)


def onboarding_report(df):
    total = len(df)
    if total == 0:
        return {"Onboarding Effectiveness": "N/A", "Attrition Risk": "N/A",
                "Avg Performance": "N/A", "Needs Intervention": 0, "Status": "N/A"}
    ready = (df["readiness_level"] == "Ready").sum()
    high_r = (df["fatigue_risk"] == "High").sum()
    eff = round(ready / total * 100, 1)
    return {
        "Onboarding Effectiveness": f"{eff}%",
        "Attrition Risk": f"{round(high_r / total * 100, 1)}%",
        "Avg Performance": round(float(df["performance_score"].mean()), 2),
        "Needs Intervention": int(high_r),
        "Status": "✅ Good" if eff >= 60 else "⚠️ Needs Improvement",
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
def fig_risk_pie(df):
    c = df["fatigue_risk"].value_counts()
    fig = px.pie(names=c.index, values=c.values, color=c.index,
                 color_discrete_map=RISK_COLORS, hole=0.6)
    fig.update_traces(textfont=dict(family="Syne, sans-serif", size=13),
                      marker=dict(line=dict(color="#070B14", width=3)))
    fig.update_layout(**PLOTLY_LAYOUT, title="Fatigue Risk Distribution",
                      showlegend=True,
                      annotations=[dict(text="<b>Risk</b>", x=0.5, y=0.5,
                                        font=dict(size=14, color="#F0F6FF", family="Syne"),
                                        showarrow=False)])
    return fig


def fig_readiness_donut(df):
    c = df["readiness_level"].value_counts()
    fig = px.pie(names=c.index, values=c.values, color=c.index,
                 color_discrete_map=READ_COLORS, hole=0.6)
    fig.update_traces(textfont=dict(family="Syne, sans-serif", size=13),
                      marker=dict(line=dict(color="#070B14", width=3)))
    fig.update_layout(**PLOTLY_LAYOUT, title="Workforce Readiness",
                      annotations=[dict(text="<b>Ready</b>", x=0.5, y=0.5,
                                        font=dict(size=14, color="#F0F6FF", family="Syne"),
                                        showarrow=False)])
    return fig


def fig_dept_bar(d):
    fig = px.bar(d, x="department", y="Avg_Performance", color="Avg_Performance",
                 color_continuous_scale=BLUE_SCALE, text_auto=".1f",
                 labels={"department": "", "Avg_Performance": "Avg Score"})
    fig.update_traces(marker_line_color="rgba(0,0,0,0)", textfont=dict(color="white"))
    fig.update_layout(**PLOTLY_LAYOUT, title="Avg Performance by Department",
                      coloraxis_showscale=False)
    return fig


def fig_scatter_eng_perf(df):
    fig = px.scatter(df, x="engagement_score", y="performance_score",
                     color="fatigue_risk", color_discrete_map=RISK_COLORS,
                     hover_data=["name", "department", "readiness_level"],
                     opacity=0.8, size_max=10)
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_layout(**PLOTLY_LAYOUT, title="Engagement vs Performance Score")
    return fig


def fig_corr_heatmap(corr):
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0B1223")
    ax.set_facecolor("#0B1223")
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, annot_kws={"size": 8, "color": "white"},
                linecolor="#070B14", ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold",
                 color="#F0F6FF", pad=15, fontfamily="sans-serif")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", color="#94A3B8")
    plt.setp(ax.get_yticklabels(), rotation=0, color="#94A3B8")
    ax.collections[0].colorbar.ax.tick_params(colors="#94A3B8")
    plt.tight_layout()
    return fig


def fig_feat_importance(fi):
    fig = px.bar(fi.head(11), x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale=BLUE_SCALE, text_auto=".3f")
    fig.update_traces(marker_line_color="rgba(0,0,0,0)", textfont=dict(color="white"))
    layout = {**PLOTLY_LAYOUT, "coloraxis_showscale": False}
    layout["yaxis"] = dict(autorange="reversed", **PLOTLY_LAYOUT["yaxis"])
    fig.update_layout(**layout, title="Feature Importance (Random Forest)")
    return fig


def fig_conf_matrix(cm, classes, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0B1223")
    ax.set_facecolor("#0B1223")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor="#070B14",
                annot_kws={"size": 13, "weight": "bold", "color": "white"})
    ax.set_xlabel("Predicted", color="#94A3B8", fontsize=11)
    ax.set_ylabel("Actual", color="#94A3B8", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#F0F6FF", pad=12)
    ax.tick_params(colors="#94A3B8")
    plt.setp(ax.get_xticklabels(), color="#94A3B8")
    plt.setp(ax.get_yticklabels(), color="#94A3B8", rotation=0)
    plt.tight_layout()
    return fig


def fig_attend_task(df):
    fig = px.scatter(df, x="attendance", y="task_completion", color="readiness_level",
                     color_discrete_map=READ_COLORS,
                     hover_data=["name", "department"], opacity=0.8)
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_layout(**PLOTLY_LAYOUT, title="Attendance vs Task Completion")
    return fig


def fig_boxplot(df, metric):
    fig = px.box(df, x="department", y=metric, color="department",
                 points="outliers", color_discrete_sequence=["#3B82F6", "#06B6D4", "#8B5CF6", "#10B981", "#F59E0B"])
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{metric.replace('_', ' ').title()} Distribution by Department",
                      showlegend=False)
    return fig


def fig_radar(row):
    cats = ["Quiz", "Coding", "Attendance", "Task Compl", "Engagement", "Communication", "Technical"]
    vals = [float(row.quiz_score), float(row.coding_score), float(row.attendance),
            float(row.task_completion), float(row.engagement_score),
            float(row.communication_score), float(row.technical_assessment)]
    avg_vals = [65, 60, 78, 72, 65, 60, 62]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_vals + [avg_vals[0]], theta=cats + [cats[0]],
        name="Team Avg", mode="lines",
        line=dict(color="rgba(99,179,237,0.4)", dash="dash", width=1.5),
        fill="toself", fillcolor="rgba(99,179,237,0.04)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
        name=str(row.get("name", "Employee")),
        line=dict(color="#3B82F6", width=2.5),
        fillcolor="rgba(59,130,246,0.18)"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(11,18,35,0.5)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(99,179,237,0.15)",
                            tickfont=dict(color="#64748B", size=9), linecolor="rgba(99,179,237,0.1)"),
            angularaxis=dict(gridcolor="rgba(99,179,237,0.12)", linecolor="rgba(99,179,237,0.15)",
                             tickfont=dict(color="#94A3B8", size=11, family="DM Sans"))
        ),
        legend=dict(bgcolor="rgba(11,18,35,0.6)", bordercolor="rgba(99,179,237,0.2)", borderwidth=1,
                    font=dict(color="#94A3B8")),
        font=dict(family="DM Sans", color="#94A3B8"),
        title=dict(text=f"Skill Profile: {row.get('name', 'Employee')}",
                   font=dict(family="Syne", color="#F0F6FF", size=14)),
        margin=dict(l=10, r=10, t=45, b=10)
    )
    return fig


def fig_risk_by_dept(df):
    g = df.groupby(["department", "fatigue_risk"]).size().reset_index(name="count")
    fig = px.bar(g, x="department", y="count", color="fatigue_risk",
                 color_discrete_map=RISK_COLORS, barmode="group",
                 labels={"department": "", "count": "Count", "fatigue_risk": "Risk"})
    fig.update_traces(marker_line_color="rgba(0,0,0,0)")
    fig.update_layout(**PLOTLY_LAYOUT, title="Fatigue Risk by Department")
    return fig


def fig_study_hist(df):
    fig = px.histogram(df, x="study_hours", color="fatigue_risk",
                       color_discrete_map=RISK_COLORS, nbins=20,
                       barmode="overlay", opacity=0.75,
                       labels={"study_hours": "Study Hours/Day"})
    fig.update_traces(marker_line_color="rgba(0,0,0,0)")
    fig.update_layout(**PLOTLY_LAYOUT, title="Study Hours Distribution by Risk Level")
    return fig


def fig_performance_dist(df):
    fig = px.histogram(df, x="performance_score", color="readiness_level",
                       color_discrete_map=READ_COLORS, nbins=25, barmode="overlay", opacity=0.8)
    fig.update_traces(marker_line_color="rgba(0,0,0,0)")
    fig.update_layout(**PLOTLY_LAYOUT, title="Performance Score Distribution")
    return fig


def fig_batch_readiness(br):
    fig = go.Figure()
    colors = {"Ready": "#10B981", "Partial": "#F59E0B", "Not_Ready": "#EF4444"}
    for col, color in colors.items():
        fig.add_trace(go.Bar(name=col.replace("_", " "), x=br["batch"], y=br[col],
                             marker_color=color, marker_line_width=0))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", title="Readiness by Batch")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_data():
    return generate_dataset(300)


@st.cache_data(show_spinner=False)
def get_models(_df):
    X, y_risk, y_ready = get_feature_matrix(_df)
    risk_clf, risk_le, risk_m = train_model(X, y_risk)
    ready_clf, ready_le, ready_m = train_model(X, y_ready)
    fi = feature_importance_df(risk_clf)
    return risk_clf, risk_le, risk_m, ready_clf, ready_le, ready_m, fi


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def build_sidebar(df_raw):
    st.sidebar.markdown("""
    <div class='px-sidebar-logo'>
        <h2>⚡ Performix</h2>
        <p>Workforce Intelligence Platform</p>
    </div>""", unsafe_allow_html=True)

    pages = [
        "🏠  Dashboard",
        "📊  Performance",
        "⚠️   Risk Prediction",
        "🎯  Readiness",
        "🤖  ML Insights",
        "💡  Recommendations",
        "🔍  Employee Lookup",
    ]
    page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

    st.sidebar.markdown("<hr style='border-color:rgba(99,179,237,0.1);margin:1rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#475569;font-weight:600;'>Filters</p>",
        unsafe_allow_html=True)

    depts = ["All"] + sorted(df_raw["department"].unique().tolist())
    roles = ["All"] + sorted(df_raw["role"].unique().tolist())
    batches = ["All"] + sorted(df_raw["batch"].unique().tolist())

    sel_dept = st.sidebar.selectbox("Department", depts)
    sel_role = st.sidebar.selectbox("Role", roles)
    sel_batch = st.sidebar.selectbox("Batch", batches)

    df = df_raw.copy()
    if sel_dept != "All": df = df[df["department"] == sel_dept]
    if sel_role != "All": df = df[df["role"] == sel_role]
    if sel_batch != "All": df = df[df["batch"] == sel_batch]

    st.sidebar.markdown(f"""
    <div style='background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);
    border-radius:8px;padding:0.6rem 0.9rem;margin:0.75rem 0;'>
        <p style='margin:0;font-size:0.78rem;color:#60A5FA;font-weight:600;'>
            📋 {len(df)} records filtered
        </p>
    </div>""", unsafe_allow_html=True)

    st.sidebar.download_button("⬇️ Export Data", df.to_csv(index=False).encode(),
                               "performix_data.csv", "text/csv")
    return page, df


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
def kpi(cols_list, data_list):
    for col, item in zip(cols_list, data_list):
        with col:
            st.markdown(f"""
            <div class='px-kpi {item.get("color", "blue")}'>
                <span class='px-kpi-icon'>{item.get("icon", "📌")}</span>
                <div class='px-kpi-value'>{item["value"]}</div>
                <div class='px-kpi-label'>{item["label"]}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard(df):
    st.markdown("""
    <div class='px-hero'>
        <div class='px-hero-badge'>⚡ AI-Powered HR Intelligence</div>
        <div class='px-hero-title'>Performix Workforce Intelligence</div>
        <p class='px-hero-sub'>Advanced talent analytics — onboarding readiness, fatigue risk & performance stability for IT organizations</p>
    </div>""", unsafe_allow_html=True)

    s = compute_summary(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi([c1, c2, c3, c4, c5], [
        {"value": s["total"], "label": "Total Employees", "icon": "👥", "color": "blue"},
        {"value": s["avg_perf"], "label": "Avg Performance", "icon": "📈", "color": "green"},
        {"value": s["high_risk"], "label": "High Risk", "icon": "⚠️", "color": "red"},
        {"value": s["ready"], "label": "Workforce Ready", "icon": "✅", "color": "green"},
        {"value": f"{s['avg_attend']}%", "label": "Avg Attendance", "icon": "📅", "color": "amber"},
    ])
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpi([c1, c2, c3, c4], [
        {"value": f"{s['avg_task']}%", "label": "Task Completion", "icon": "✔️", "color": "blue"},
        {"value": s["avg_quiz"], "label": "Avg Quiz Score", "icon": "📝", "color": "violet"},
        {"value": s["avg_coding"], "label": "Avg Coding Score", "icon": "💻", "color": "blue"},
        {"value": f"{s['avg_feedback']}/5", "label": "Avg Feedback", "icon": "⭐", "color": "amber"},
    ])

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_risk_pie(df), use_container_width=True)
    col2.plotly_chart(fig_readiness_donut(df), use_container_width=True)

    st.plotly_chart(fig_scatter_eng_perf(df), use_container_width=True)

    st.markdown(
        "<div class='px-section-title' style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;color:#F0F6FF;padding-bottom:0.5rem;border-bottom:1px solid rgba(99,179,237,0.12);margin-bottom:1rem;'>📋 Employee Preview</div>",
        unsafe_allow_html=True)
    cols = ["employee_id", "name", "department", "role", "performance_score",
            "attendance", "fatigue_risk", "readiness_level"]
    st.dataframe(df[cols].head(20), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def page_performance(df):
    st.markdown("""
    <div class='px-hero'>
        <div class='px-hero-badge'>📊 Analytics</div>
        <div class='px-hero-title'>Performance & Engagement Analysis</div>
        <p class='px-hero-sub'>Deep-dive into departmental metrics, skill distribution, and workforce engagement patterns</p>
    </div>""", unsafe_allow_html=True)

    d = dept_analysis(df)
    st.plotly_chart(fig_dept_bar(d), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_attend_task(df), use_container_width=True)
    col2.plotly_chart(fig_performance_dist(df), use_container_width=True)
    col1.plotly_chart(fig_study_hist(df), use_container_width=True)
    col2.plotly_chart(fig_risk_by_dept(df), use_container_width=True)

    metric = st.selectbox("📦 Metric for Box Plot", [
        "performance_score", "quiz_score", "coding_score",
        "attendance", "task_completion", "engagement_score"])
    st.plotly_chart(fig_boxplot(df, metric), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["🏆 Top Performers", "🏢 Department Summary", "🔗 Correlation Matrix"])
    with tab1:
        st.dataframe(top_performers(df), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(d, use_container_width=True, hide_index=True)
    with tab3:
        st.pyplot(fig_corr_heatmap(correlation_matrix(df)), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def page_risk(df, risk_clf, risk_le, risk_m):
    st.markdown("""
    <div class='px-hero'>
        <div class='px-hero-badge'>⚠️ Risk Intelligence</div>
        <div class='px-hero-title'>Fatigue & Performance Risk Prediction</div>
        <p class='px-hero-sub'>Random Forest classifier for real-time risk scoring with live employee predictor</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Accuracy", f"{risk_m['accuracy'] * 100:.1f}%")
    c2.metric("F1 Score (Weighted)", f"{risk_m['f1']:.4f}")
    c3.metric("5-Fold CV Accuracy", f"{risk_m['cv_mean'] * 100:.1f}% ± {risk_m['cv_std'] * 100:.1f}%")

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_risk_pie(df), use_container_width=True)
    col2.plotly_chart(fig_risk_by_dept(df), use_container_width=True)

    st.markdown(
        "<div class='px-section-title' style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;color:#F0F6FF;padding-bottom:0.5rem;border-bottom:1px solid rgba(99,179,237,0.12);margin-bottom:1rem;'>🔴 High-Risk Employees</div>",
        unsafe_allow_html=True)
    ar = at_risk_employees(df)
    if ar.empty:
        st.info("✅ No high-risk employees in the current filter.")
    else:
        st.dataframe(ar, use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#F0F6FF;margin-bottom:0.5rem;'>Confusion Matrix</div>",
            unsafe_allow_html=True)
        st.pyplot(fig_conf_matrix(risk_m["cm"], risk_m["classes"], "Risk Model"), use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
    color:#F0F6FF;margin-bottom:0.5rem;'>🔮 Live Risk Predictor</div>
    <p style='color:#94A3B8;font-size:0.85rem;margin-bottom:1rem;'>
    Adjust the sliders below to simulate an employee profile and get an instant risk prediction.</p>
    """, unsafe_allow_html=True)

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        study_h = c1.slider("📖 Study Hours/Day", 0.0, 12.0, 5.0, 0.5)
        screen_t = c2.slider("📱 Screen Time/Day", 0.0, 14.0, 6.0, 0.5)
        quiz = c3.slider("📝 Quiz Score", 0, 100, 65)
        coding = c1.slider("💻 Coding Score", 0, 100, 60)
        attend = c2.slider("📅 Attendance (%)", 0, 100, 78)
        task_c = c3.slider("✔️ Task Completion (%)", 0, 100, 72)
        feedback = c1.slider("⭐ Feedback Rating", 1.0, 5.0, 3.5, 0.1)
        engage = c2.slider("🤝 Engagement Score", 0, 100, 65)
        comm = c3.slider("🗣 Communication Score", 0, 100, 60)
        tech = c1.slider("🔧 Technical Assessment", 0, 100, 62)
        learn = c2.slider("📈 Learning Progression", -10.0, 20.0, 5.0, 0.5)
        go_btn = st.form_submit_button("⚡ Predict Risk Now")

    if go_btn:
        vals = {
            "study_hours": study_h, "screen_time": screen_t, "quiz_score": quiz,
            "coding_score": coding, "attendance": attend, "task_completion": task_c,
            "feedback_rating": feedback, "engagement_score": engage,
            "communication_score": comm, "technical_assessment": tech,
            "learning_progression": learn
        }
        label, proba = predict_employee(risk_clf, risk_le, vals)
        icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(label, "⚪")
        color = {"High": "red", "Medium": "amber", "Low": "green"}.get(label, "blue")
        st.markdown(f"""
        <div class='px-kpi {color}' style='margin-top:1rem;'>
            <div class='px-kpi-value'>{icon} {label} Risk</div>
            <div class='px-kpi-label'>Predicted Fatigue Risk Level</div>
        </div>""", unsafe_allow_html=True)
        prob_df = pd.DataFrame.from_dict(proba, orient="index", columns=["Probability (%)"])
        st.dataframe(prob_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — READINESS ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
def page_readiness(df):
    st.markdown("""
    <div class='px-hero'>
        <div class='px-hero-badge'>🎯 Readiness</div>
        <div class='px-hero-title'>Workforce Readiness Assessment</div>
        <p class='px-hero-sub'>Role allocation suggestions, skill gap mapping, and batch/department readiness breakdown</p>
    </div>""", unsafe_allow_html=True)

    df2 = df.copy()
    df2["suggested_role"] = df2.apply(suggest_role, axis=1)
    df2["skill_gaps"] = df2.apply(skill_gaps_label, axis=1)

    br = batch_readiness(df)
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_readiness_donut(df), use_container_width=True)
    col2.plotly_chart(fig_batch_readiness(br), use_container_width=True)

    role_counts = df2["suggested_role"].value_counts().reset_index()
    role_counts.columns = ["Suggested Role", "Count"]
    fig_role = px.bar(role_counts, x="Suggested Role", y="Count",
                      color="Count", color_continuous_scale=BLUE_SCALE, text_auto=True)
    fig_role.update_traces(marker_line_color="rgba(0,0,0,0)")
    fig_role.update_layout(**PLOTLY_LAYOUT, title="Role Allocation Suggestions", coloraxis_showscale=False)
    st.plotly_chart(fig_role, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📦 By Batch", "🏢 By Department", "🔍 Employee Table"])
    with tab1:
        st.dataframe(br, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(dept_readiness(df), use_container_width=True, hide_index=True)
    with tab3:
        cols = ["employee_id", "name", "department", "role", "performance_score",
                "readiness_level", "fatigue_risk", "suggested_role", "skill_gaps"]
        st.dataframe(df2[cols].head(50), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ML INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def page_ml(risk_m, ready_m, fi):
    st.markdown("""
    <div class='px-hero'>
        <div class='px-hero-badge'>🤖 Machine Learning</div>
        <div class='px-hero-title'>ML Model Insights</div>
        <p class='px-hero-sub'>Random Forest classifiers — performance metrics, feature importances, and confusion matrices</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["⚠️ Risk Model", "🎯 Readiness Model"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{risk_m['accuracy'] * 100:.1f}%")
        c2.metric("F1 Score", f"{risk_m['f1']:.4f}")
        c3.metric("CV Accuracy", f"{risk_m['cv_mean'] * 100:.1f}%")
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_feat_importance(fi), use_container_width=True)
        with col2:
            st.pyplot(fig_conf_matrix(risk_m["cm"], risk_m["classes"]), use_container_width=True)
        rows = [{"Class": k, **{kk: round(vv, 3) for kk, vv in v.items()}}
                for k, v in risk_m["report"].items() if isinstance(v, dict)]
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#F0F6FF;margin:1rem 0 0.5rem;'>Classification Report</div>",
            unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab2:
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{ready_m['accuracy'] * 100:.1f}%")
        c2.metric("F1 Score", f"{ready_m['f1']:.4f}")
        st.pyplot(fig_conf_matrix(ready_m["cm"], ready_m["classes"], "Readiness – Confusion Matrix"),
                  use_container_width=True)
        rows2 = [{"Class": k, **{kk: round(vv, 3) for kk, vv in v.items()}}
                 for k, v in ready_m["report"].items() if isinstance(v, dict)]
        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
def page_recommendations(df):
    st.markdown("""
    <div class='px-hero'>
        <div class='px-hero-badge'>💡 Action Plans</div>
        <div class='px-hero-title'>Recommendations & Interventions</div>
        <p class='px-hero-sub'>Personalized improvement plans, intervention urgency mapping, and onboarding effectiveness analysis</p>
    </div>""", unsafe_allow_html=True)

    report = onboarding_report(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Onboarding Effectiveness", report["Onboarding Effectiveness"])
    c2.metric("Attrition Risk", report["Attrition Risk"])
    c3.metric("Avg Performance", report["Avg Performance"])
    c4.metric("Status", report["Status"])

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    plan = intervention_plan(df)
    if plan.empty:
        st.info("✅ No high-risk employees in the current filter.")
    else:
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#F0F6FF;margin-bottom:0.5rem;'>🚨 Intervention Plan — High Risk Employees</div>",
            unsafe_allow_html=True)
        st.dataframe(plan, use_container_width=True, hide_index=True)
        urg = plan["Urgency"].value_counts().reset_index()
        urg.columns = ["Urgency", "Count"]
        fig_urg = px.pie(urg, names="Urgency", values="Count", hole=0.55,
                         color_discrete_sequence=["#EF4444", "#F59E0B", "#10B981"])
        fig_urg.update_layout(**PLOTLY_LAYOUT, title="Intervention Urgency Distribution")
        st.plotly_chart(fig_urg, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#F0F6FF;margin-bottom:0.75rem;'>🎯 Personalized Plan Generator</div>",
        unsafe_allow_html=True)
    emp = st.selectbox("Select Employee", df["name"].tolist())
    row = df[df["name"] == emp].iloc[0]
    recs = personalized_plan(row)
    if not recs:
        st.markdown("""
        <div class='px-plan-item' style='border-left-color:#10B981;background:rgba(16,185,129,0.06);border-color:rgba(16,185,129,0.18);'>
            <div class='px-plan-title'>✅ Performing well across all metrics</div>
            <div class='px-plan-advice'>Continue regular check-ins and maintain current habits.</div>
        </div>""", unsafe_allow_html=True)
    else:
        for i, r in enumerate(recs, 1):
            color = "red" if r["gap"] > 15 else "amber"
            st.markdown(f"""
            <div class='px-plan-item {color}'>
                <div class='px-plan-title'>{i}. {r["metric"]}</div>
                <div class='px-plan-score'>Current: <strong>{r["current"]}</strong> &nbsp;|&nbsp; Target: ≥ <strong>{r["target"]}</strong> &nbsp;|&nbsp; Gap: <strong>−{r["gap"]}</strong></div>
                <div class='px-plan-advice'>{r["advice"]}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — EMPLOYEE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
def page_employee_lookup(df):
    st.markdown("""
    <div class='px-hero'>
        <div class='px-hero-badge'>🔍 Deep Profile</div>
        <div class='px-hero-title'>Employee Lookup & Individual Analysis</div>
        <p class='px-hero-sub'>Search any employee to view their full skill radar, risk profile, and personalized improvement plan</p>
    </div>""", unsafe_allow_html=True)

    search = st.text_input("🔍 Search by ID or Name", placeholder="e.g. EMP0042 or Employee_5")
    filtered = df[
        df["name"].str.contains(search, case=False, na=False) |
        df["employee_id"].str.contains(search, case=False, na=False)
        ] if search else df

    if filtered.empty:
        st.warning("No employees match your search.")
        return

    eid = st.selectbox("Select Employee", filtered["employee_id"].tolist())
    row = df[df["employee_id"] == eid].iloc[0]

    initials = "".join(w[0] for w in row["name"].split("_") if w)[:2].upper()
    risk_tag = {"High": "high", "Medium": "medium", "Low": "low"}.get(row["fatigue_risk"], "low")
    ready_tag = {"Ready": "ready", "Partially Ready": "partial", "Not Ready": "not"}.get(row["readiness_level"], "not")

    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.plotly_chart(fig_radar(row), use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class='px-profile-card'>
            <div class='px-avatar'>{initials}</div>
            <div class='px-profile-name'>{row["name"]}</div>
            <div class='px-profile-meta'>{row["department"]} · {row["role"]} · {row["batch"]}</div>
            <div style='margin:0.75rem 0;'>
                <span class='px-tag {risk_tag}'>{row["fatigue_risk"]} Risk</span>
                <span class='px-tag {ready_tag}' style='margin-left:0.4rem;'>{row["readiness_level"]}</span>
            </div>
            <div class='px-detail-row'><span class='px-detail-key'>Employee ID</span><span class='px-detail-val'>{row.employee_id}</span></div>
            <div class='px-detail-row'><span class='px-detail-key'>Performance Score</span><span class='px-detail-val'>{row.performance_score}</span></div>
            <div class='px-detail-row'><span class='px-detail-key'>Quiz Score</span><span class='px-detail-val'>{row.quiz_score}</span></div>
            <div class='px-detail-row'><span class='px-detail-key'>Coding Score</span><span class='px-detail-val'>{row.coding_score}</span></div>
            <div class='px-detail-row'><span class='px-detail-key'>Attendance</span><span class='px-detail-val'>{row.attendance}%</span></div>
            <div class='px-detail-row'><span class='px-detail-key'>Task Completion</span><span class='px-detail-val'>{row.task_completion}%</span></div>
            <div class='px-detail-row'><span class='px-detail-key'>Engagement</span><span class='px-detail-val'>{row.engagement_score}</span></div>
            <div class='px-detail-row'><span class='px-detail-key'>Communication</span><span class='px-detail-val'>{row.communication_score}</span></div>
            <div class='px-detail-row'><span class='px-detail-key'>Technical Assessment</span><span class='px-detail-val'>{row.technical_assessment}</span></div>
            <div class='px-detail-row'><span class='px-detail-key'>Study Hours/Day</span><span class='px-detail-val'>{row.study_hours}</span></div>
            <div class='px-detail-row'><span class='px-detail-key'>Feedback Rating</span><span class='px-detail-val'>{row.feedback_rating}/5</span></div>
            <div class='px-detail-row' style='border:none;'><span class='px-detail-key'>Learning Progression</span><span class='px-detail-val'>{row.learning_progression}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
    color:#F0F6FF;margin-bottom:0.75rem;'>💡 Personalized Improvement Plan — {row["name"]}</div>
    """, unsafe_allow_html=True)

    recs = personalized_plan(row)
    if not recs:
        st.markdown("""
        <div class='px-plan-item' style='border-left-color:#10B981;background:rgba(16,185,129,0.06);border-color:rgba(16,185,129,0.18);'>
            <div class='px-plan-title'>✅ Performing well across all tracked metrics</div>
            <div class='px-plan-advice'>Continue regular check-ins and maintain current study and work habits.</div>
        </div>""", unsafe_allow_html=True)
    else:
        for i, r in enumerate(recs, 1):
            color = "red" if r["gap"] > 15 else "amber"
            st.markdown(f"""
            <div class='px-plan-item {color}'>
                <div class='px-plan-title'>{i}. {r["metric"]}</div>
                <div class='px-plan-score'>Current: <strong>{r["current"]}</strong> &nbsp;|&nbsp; Target ≥ <strong>{r["target"]}</strong> &nbsp;|&nbsp; Gap: <strong>−{r["gap"]}</strong></div>
                <div class='px-plan-advice'>{r["advice"]}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df_raw = get_data()
    risk_clf, risk_le, risk_m, ready_clf, ready_le, ready_m, fi = get_models(df_raw)
    page, df = build_sidebar(df_raw)

    if "Dashboard" in page:
        page_dashboard(df)
    elif "Performance" in page:
        page_performance(df)
    elif "Risk" in page:
        page_risk(df, risk_clf, risk_le, risk_m)
    elif "Readiness" in page:
        page_readiness(df)
    elif "ML" in page:
        page_ml(risk_m, ready_m, fi)
    elif "Recommendation" in page:
        page_recommendations(df)
    elif "Lookup" in page:
        page_employee_lookup(df)


if __name__ == "__main__":
    main()
