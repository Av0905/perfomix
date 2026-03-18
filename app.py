"""
Intelligent System for Identifying Workforce Readiness
and Performance Stability in Entry-Level Professionals
────────────────────────────────────────────────────────
Single-file Streamlit Application (Streamlit Cloud compatible)
All 7 modules are self-contained — zero relative imports.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Workforce Readiness Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #BDD7EE; margin: 0.4rem 0 0; font-size: 0.95rem; }
    .kpi-card {
        background: white; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 5px solid #2E75B6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 0.5rem;
    }
    .kpi-card.red    { border-left-color: #E74C3C; }
    .kpi-card.green  { border-left-color: #27AE60; }
    .kpi-card.orange { border-left-color: #F39C12; }
    .kpi-card h3 { color: #1F4E79; font-size: 1.6rem; margin: 0; }
    .kpi-card p  { color: #666; font-size: 0.85rem; margin: 0; }
</style>
""", unsafe_allow_html=True)


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
    ids     = [f"EMP{str(i).zfill(4)}" for i in range(1, n+1)]
    names   = [f"Employee_{i}"          for i in range(1, n+1)]
    depts   = np.random.choice(["Backend Dev","Frontend Dev","Data Science","QA Testing","DevOps"], n)
    roles   = np.random.choice(["Intern","Fresh Graduate"], n)
    batches = np.random.choice(["Batch-2023","Batch-2024","Batch-2025"], n)

    study_hours          = np.round(np.random.normal(5.5, 2.0, n).clip(0, 12),  1)
    screen_time          = np.round(np.random.normal(6.0, 1.5, n).clip(1, 14),  1)
    quiz_score           = np.round(np.random.normal(65,  15,   n).clip(0, 100), 1)
    coding_score         = np.round(np.random.normal(60,  18,   n).clip(0, 100), 1)
    attendance           = np.round(np.random.normal(78,  12,   n).clip(30, 100),1)
    task_completion      = np.round(np.random.normal(72,  15,   n).clip(0, 100), 1)
    feedback_rating      = np.round(np.random.uniform(1,  5,    n),               1)
    engagement_score     = np.round(np.random.normal(65,  18,   n).clip(0, 100), 1)
    communication_score  = np.round(np.random.normal(60,  15,   n).clip(0, 100), 1)
    technical_assessment = np.round(np.random.normal(62,  16,   n).clip(0, 100), 1)
    learning_progression = np.round(np.random.normal(5,   3,    n).clip(-10, 20),1)

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
        s = (row.attendance*0.2 + row.task_completion*0.2 + row.quiz_score*0.15
             + row.coding_score*0.15 + row.engagement_score*0.15
             + row.feedback_rating * 20 * 0.15)
        return "High" if s < 50 else ("Medium" if s < 70 else "Low")

    df["fatigue_risk"]    = df.apply(assign_risk, axis=1)
    df["readiness_level"] = pd.cut(
        df["performance_score"],
        bins=[-1, 54.9, 74.9, 100],
        labels=["Not Ready", "Partially Ready", "Ready"]
    ).astype(str)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — PREPROCESSING  (inline)
# ══════════════════════════════════════════════════════════════════════════════
def get_feature_matrix(df):
    X = df[FEATURE_COLS].copy()
    return X, df["fatigue_risk"], df["readiness_level"]


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def compute_summary(df):
    return {
        "total":       len(df),
        "avg_perf":    round(float(df["performance_score"].mean()), 2),
        "avg_attend":  round(float(df["attendance"].mean()), 2),
        "avg_task":    round(float(df["task_completion"].mean()), 2),
        "avg_engage":  round(float(df["engagement_score"].mean()), 2),
        "avg_quiz":    round(float(df["quiz_score"].mean()), 2),
        "avg_coding":  round(float(df["coding_score"].mean()), 2),
        "avg_feedback":round(float(df["feedback_rating"].mean()), 2),
        "high_risk":   int((df["fatigue_risk"]=="High").sum()),
        "medium_risk": int((df["fatigue_risk"]=="Medium").sum()),
        "low_risk":    int((df["fatigue_risk"]=="Low").sum()),
        "ready":       int((df["readiness_level"]=="Ready").sum()),
        "partial":     int((df["readiness_level"]=="Partially Ready").sum()),
        "not_ready":   int((df["readiness_level"]=="Not Ready").sum()),
    }

def dept_analysis(df):
    return df.groupby("department").agg(
        Employees        = ("employee_id", "count"),
        Avg_Performance  = ("performance_score", "mean"),
        Avg_Attendance   = ("attendance", "mean"),
        Avg_Engagement   = ("engagement_score", "mean"),
        Avg_Quiz         = ("quiz_score", "mean"),
        Avg_Coding       = ("coding_score", "mean"),
        High_Risk        = ("fatigue_risk",    lambda x: (x=="High").sum()),
        Ready_Count      = ("readiness_level", lambda x: (x=="Ready").sum()),
    ).round(2).reset_index()

def top_performers(df, n=10):
    cols = ["employee_id","name","department","role","performance_score",
            "attendance","task_completion","engagement_score","readiness_level"]
    return df[cols].sort_values("performance_score", ascending=False).head(n).reset_index(drop=True)

def at_risk_employees(df):
    cols = ["employee_id","name","department","role",
            "performance_score","attendance","engagement_score",
            "fatigue_risk","readiness_level"]
    return df[df["fatigue_risk"]=="High"][cols].sort_values("performance_score").reset_index(drop=True)

def correlation_matrix(df):
    return df[FEATURE_COLS + ["performance_score"]].corr().round(2)

def engagement_segments(df):
    me = df["engagement_score"].median()
    mp = df["performance_score"].median()
    def quad(r):
        if r.engagement_score >= me and r.performance_score >= mp: return "High Eng / High Perf"
        elif r.engagement_score >= me:                              return "High Eng / Low Perf"
        elif r.performance_score >= mp:                             return "Low Eng / High Perf"
        else:                                                       return "Low Eng / Low Perf"
    df = df.copy()
    df["segment"] = df.apply(quad, axis=1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
def train_model(X, y_series):
    le  = LabelEncoder()
    y   = le.fit_transform(y_series)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    cv     = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    report = classification_report(y_te, y_pred, target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    return clf, le, {
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "f1":       round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "cm":       confusion_matrix(y_te, y_pred),
        "report":   report,
        "classes":  le.classes_.tolist(),
        "cv_mean":  round(float(cv.mean()), 4),
        "cv_std":   round(float(cv.std()),  4),
    }

def feature_importance_df(clf):
    fi = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": clf.feature_importances_})
    return fi.sort_values("Importance", ascending=False).reset_index(drop=True)

def predict_employee(clf, le, vals):
    inp   = pd.DataFrame([[vals.get(c, 0) for c in FEATURE_COLS]], columns=FEATURE_COLS)
    pred  = clf.predict(inp)[0]
    prob  = clf.predict_proba(inp)[0]
    label = le.inverse_transform([pred])[0]
    proba = {le.classes_[i]: round(float(prob[i])*100, 1) for i in range(len(le.classes_))}
    return label, proba


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — READINESS ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
def suggest_role(row):
    if row.coding_score >= 75 and row.technical_assessment >= 70:  return "Developer / Engineer"
    elif row.quiz_score >= 70 and row.communication_score >= 65:   return "Business Analyst"
    elif row.engagement_score >= 70 and row.task_completion >= 70: return "Project Coordinator"
    elif row.technical_assessment >= 65 and row.coding_score < 60: return "QA / Testing"
    else:                                                           return "Support / Training"

def skill_gaps_label(row):
    scores = {c: row[c] for c in ["quiz_score","coding_score","attendance",
              "task_completion","engagement_score","communication_score","technical_assessment"]}
    return ", ".join(k.replace("_"," ").title() for k in sorted(scores, key=scores.get)[:2])

def batch_readiness(df):
    return df.groupby("batch").agg(
        Total           = ("employee_id","count"),
        Ready           = ("readiness_level", lambda x: (x=="Ready").sum()),
        Partial         = ("readiness_level", lambda x: (x=="Partially Ready").sum()),
        Not_Ready       = ("readiness_level", lambda x: (x=="Not Ready").sum()),
        Avg_Performance = ("performance_score","mean"),
    ).round(2).reset_index()

def dept_readiness(df):
    return df.groupby("department").agg(
        Total           = ("employee_id","count"),
        Ready           = ("readiness_level", lambda x: (x=="Ready").sum()),
        Partial         = ("readiness_level", lambda x: (x=="Partially Ready").sum()),
        Not_Ready       = ("readiness_level", lambda x: (x=="Not Ready").sum()),
        Avg_Performance = ("performance_score","mean"),
        High_Risk       = ("fatigue_risk",     lambda x: (x=="High").sum()),
    ).round(2).reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
THRESHOLDS = {
    "attendance":70, "quiz_score":55, "coding_score":55,
    "task_completion":60, "engagement_score":55,
    "communication_score":55, "technical_assessment":55,
    "feedback_rating":3.0, "study_hours":3.0,
}
TRAINING_CATALOG = {
    "attendance":           "📅 Attendance Program: 1-on-1 HR check-ins; identify barriers.",
    "quiz_score":           "📚 Knowledge Booster: Weekly quizzes; e-learning (Coursera/Udemy).",
    "coding_score":         "💻 Coding Bootcamp: Daily LeetCode; pair programming with seniors.",
    "task_completion":      "⏱ Productivity Workshop: Pomodoro technique; Jira/Trello tracking.",
    "engagement_score":     "🤝 Engagement Initiative: Team activities, mentorship, recognition.",
    "communication_score":  "🗣 Communication Training: Presentation & writing workshops.",
    "technical_assessment": "🔧 Technical Upskilling: Domain certs (AWS, Python, SQL, etc.).",
    "feedback_rating":      "🌱 Growth Coaching: Performance reviews; goal-setting with manager.",
    "study_hours":          "📖 Study Habit Counseling: Structured daily plans; protected hours.",
}

def get_recommendations(row):
    recs = []
    for metric, threshold in THRESHOLDS.items():
        val = row.get(metric)
        if val is not None and float(val) < threshold:
            recs.append({
                "metric":  metric.replace("_"," ").title(),
                "current": round(float(val), 2),
                "target":  threshold,
                "advice":  TRAINING_CATALOG.get(metric, "General improvement recommended."),
            })
    return recs

def personalized_plan(row):
    recs = get_recommendations(row)
    if not recs:
        return "✅ **Performing well across all metrics.** Continue regular check-ins."
    lines = [f"### 📋 Personalized Plan — {row.get('name','Employee')}\n"]
    for i, r in enumerate(recs, 1):
        lines.append(
            f"**{i}. {r['metric']}** (Current: `{r['current']}` | Target ≥ `{r['target']}`)\n"
            f"> {r['advice']}\n"
        )
    return "\n".join(lines)

def intervention_plan(df):
    rows = []
    for _, r in df[df["fatigue_risk"]=="High"].iterrows():
        recs  = get_recommendations(r)
        areas = [x["metric"] for x in recs]
        rows.append({
            "Employee ID":    r.employee_id,
            "Name":           r["name"],
            "Department":     r.department,
            "Performance":    r.performance_score,
            "Fatigue Risk":   r.fatigue_risk,
            "Readiness":      r.readiness_level,
            "Issues Count":   len(recs),
            "Priority Areas": ", ".join(areas) if areas else "None",
            "Urgency":        "🔴 Immediate" if len(recs)>=4 else ("🟡 Moderate" if len(recs)>=2 else "🟢 Low"),
        })
    return pd.DataFrame(rows)

def onboarding_report(df):
    total = len(df)
    if total == 0:
        return {"Onboarding Effectiveness":"N/A","Attrition Risk":"N/A",
                "Avg Performance":"N/A","Needs Intervention":0,"Status":"N/A"}
    ready  = (df["readiness_level"]=="Ready").sum()
    high_r = (df["fatigue_risk"]=="High").sum()
    eff    = round(ready/total*100, 1)
    return {
        "Onboarding Effectiveness": f"{eff}%",
        "Attrition Risk":           f"{round(high_r/total*100,1)}%",
        "Avg Performance":          round(float(df["performance_score"].mean()), 2),
        "Needs Intervention":       int(high_r),
        "Status":                   "✅ Good" if eff>=60 else "⚠️ Needs Improvement",
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
RISK_COLORS = {"Low":"#2ECC71","Medium":"#F39C12","High":"#E74C3C"}
READ_COLORS = {"Ready":"#27AE60","Partially Ready":"#F1C40F","Not Ready":"#E74C3C"}

def fig_risk_pie(df):
    c = df["fatigue_risk"].value_counts()
    return px.pie(names=c.index, values=c.values, color=c.index,
                  color_discrete_map=RISK_COLORS, title="Fatigue Risk Distribution", hole=0.4)

def fig_readiness_donut(df):
    c = df["readiness_level"].value_counts()
    return px.pie(names=c.index, values=c.values, color=c.index,
                  color_discrete_map=READ_COLORS, title="Workforce Readiness Distribution", hole=0.5)

def fig_dept_bar(d):
    return px.bar(d, x="department", y="Avg_Performance", color="Avg_Performance",
                  color_continuous_scale="Blues", title="Avg Performance by Department",
                  text_auto=".1f", labels={"department":"Dept","Avg_Performance":"Avg Score"})

def fig_scatter_eng_perf(df):
    return px.scatter(df, x="engagement_score", y="performance_score", color="fatigue_risk",
                      color_discrete_map=RISK_COLORS,
                      hover_data=["name","department","readiness_level"],
                      title="Engagement vs Performance", opacity=0.75)

def fig_corr_heatmap(corr):
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, annot_kws={"size":8}, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

def fig_feat_importance(fi):
    return px.bar(fi.head(11), x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Blues",
                  title="Feature Importance (Random Forest)", text_auto=".3f")

def fig_conf_matrix(cm, classes, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

def fig_attend_task(df):
    return px.scatter(df, x="attendance", y="task_completion", color="readiness_level",
                      color_discrete_map=READ_COLORS, hover_data=["name","department"],
                      title="Attendance vs Task Completion", opacity=0.75)

def fig_boxplot(df, metric):
    return px.box(df, x="department", y=metric, color="department",
                  title=f"{metric.replace('_',' ').title()} by Department", points="outliers")

def fig_radar(row):
    cats = ["Quiz","Coding","Attendance","Task Compl","Engagement","Communication","Technical"]
    vals = [float(row.quiz_score), float(row.coding_score), float(row.attendance),
            float(row.task_completion), float(row.engagement_score),
            float(row.communication_score), float(row.technical_assessment)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
        name=str(row.get("name","Employee")),
        line_color="#2E75B6", fillcolor="rgba(46,117,182,0.25)"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        showlegend=False, title=f"Skill Radar: {row.get('name','Employee')}"
    )
    return fig

def fig_risk_by_dept(df):
    g = df.groupby(["department","fatigue_risk"]).size().reset_index(name="count")
    return px.bar(g, x="department", y="count", color="fatigue_risk",
                  color_discrete_map=RISK_COLORS, barmode="group",
                  title="Fatigue Risk by Department")

def fig_study_hist(df):
    return px.histogram(df, x="study_hours", color="fatigue_risk",
                        color_discrete_map=RISK_COLORS, nbins=20,
                        barmode="overlay", opacity=0.7,
                        title="Study Hours Distribution by Risk Level")


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_data():
    return generate_dataset(300)

@st.cache_data(show_spinner=False)
def get_models(_df):
    X, y_risk, y_ready    = get_feature_matrix(_df)
    risk_clf,  risk_le,  risk_m  = train_model(X, y_risk)
    ready_clf, ready_le, ready_m = train_model(X, y_ready)
    fi = feature_importance_df(risk_clf)
    return risk_clf, risk_le, risk_m, ready_clf, ready_le, ready_m, fi


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def build_sidebar(df_raw):
    st.sidebar.markdown("""
    <div style='background:linear-gradient(135deg,#1F4E79,#2E75B6);padding:1rem;
    border-radius:10px;color:white;margin-bottom:1rem;'>
        <h3 style='margin:0;color:white;'>🧠 Workforce IQ</h3>
        <p style='margin:0;font-size:0.8rem;color:#BDD7EE;'>Readiness & Stability Platform</p>
    </div>""", unsafe_allow_html=True)

    page = st.sidebar.radio("📌 Navigate", [
        "🏠 Dashboard Overview",
        "📊 Performance Analysis",
        "⚠️ Risk Prediction",
        "🎯 Readiness Assessment",
        "🤖 ML Model Insights",
        "💡 Recommendations",
        "🔍 Employee Lookup",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Filters")
    depts   = ["All"] + sorted(df_raw["department"].unique().tolist())
    roles   = ["All"] + sorted(df_raw["role"].unique().tolist())
    batches = ["All"] + sorted(df_raw["batch"].unique().tolist())

    sel_dept  = st.sidebar.selectbox("Department", depts)
    sel_role  = st.sidebar.selectbox("Role",       roles)
    sel_batch = st.sidebar.selectbox("Batch",      batches)

    df = df_raw.copy()
    if sel_dept  != "All": df = df[df["department"] == sel_dept]
    if sel_role  != "All": df = df[df["role"]       == sel_role]
    if sel_batch != "All": df = df[df["batch"]      == sel_batch]

    st.sidebar.markdown(f"**Filtered Records:** {len(df)}")
    st.sidebar.download_button("⬇️ Download Data", df.to_csv(index=False).encode(),
                                "workforce_data.csv", "text/csv")
    return page, df


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard(df):
    st.markdown("""
    <div class='main-header'>
        <h1>🧠 Intelligent Workforce Readiness & Performance Stability System</h1>
        <p>Advanced talent analytics for IT organizations — Onboarding Intelligence Platform</p>
    </div>""", unsafe_allow_html=True)

    s = compute_summary(df)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(f"<div class='kpi-card'><h3>{s['total']}</h3><p>Total Employees</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card green'><h3>{s['avg_perf']}</h3><p>Avg Performance</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card red'><h3>{s['high_risk']}</h3><p>High Risk</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-card green'><h3>{s['ready']}</h3><p>Workforce Ready</p></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='kpi-card orange'><h3>{s['avg_attend']}%</h3><p>Avg Attendance</p></div>", unsafe_allow_html=True)

    st.markdown("")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='kpi-card'><h3>{s['avg_task']}%</h3><p>Avg Task Completion</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><h3>{s['avg_quiz']}</h3><p>Avg Quiz Score</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><h3>{s['avg_coding']}</h3><p>Avg Coding Score</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-card'><h3>{s['avg_feedback']}/5</h3><p>Avg Feedback</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_risk_pie(df),        use_container_width=True)
    col2.plotly_chart(fig_readiness_donut(df), use_container_width=True)
    st.plotly_chart(fig_scatter_eng_perf(df),  use_container_width=True)

    st.markdown("### 📋 Employee Table (Preview)")
    cols = ["employee_id","name","department","role","performance_score",
            "attendance","fatigue_risk","readiness_level"]
    st.dataframe(df[cols].head(20), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def page_performance(df):
    st.title("📊 Performance & Engagement Analysis")
    d = dept_analysis(df)
    st.plotly_chart(fig_dept_bar(d), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_attend_task(df), use_container_width=True)
    col2.plotly_chart(fig_study_hist(df),  use_container_width=True)

    metric = st.selectbox("Metric for Box Plot", [
        "performance_score","quiz_score","coding_score",
        "attendance","task_completion","engagement_score"])
    st.plotly_chart(fig_boxplot(df, metric), use_container_width=True)

    st.markdown("### 🏆 Top 10 Performers")
    st.dataframe(top_performers(df), use_container_width=True, hide_index=True)

    st.markdown("### 🏢 Department Summary")
    st.dataframe(d, use_container_width=True, hide_index=True)

    st.markdown("### 🔗 Correlation Heatmap")
    st.pyplot(fig_corr_heatmap(correlation_matrix(df)), use_container_width=True)

    st.markdown("### 🧩 Engagement-Performance Segments")
    seg = engagement_segments(df)["segment"].value_counts().reset_index()
    seg.columns = ["Segment","Count"]
    st.dataframe(seg, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def page_risk(df, risk_clf, risk_le, risk_m):
    st.title("⚠️ Fatigue & Risk Prediction")
    c1,c2,c3 = st.columns(3)
    c1.metric("Model Accuracy",      f"{risk_m['accuracy']*100:.1f}%")
    c2.metric("F1 Score",            f"{risk_m['f1']:.4f}")
    c3.metric("5-Fold CV Accuracy",  f"{risk_m['cv_mean']*100:.1f}% ± {risk_m['cv_std']*100:.1f}%")

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_risk_pie(df),     use_container_width=True)
    col2.plotly_chart(fig_risk_by_dept(df), use_container_width=True)

    st.markdown("### 🔴 At-Risk Employees (High Fatigue Risk)")
    ar = at_risk_employees(df)
    st.dataframe(ar if not ar.empty else pd.DataFrame({"Info":["No high-risk employees in filter."]}),
                 use_container_width=True, hide_index=True)

    st.markdown("### 📉 Confusion Matrix")
    st.pyplot(fig_conf_matrix(risk_m["cm"], risk_m["classes"], "Risk Model – Confusion Matrix"))

    st.markdown("### 🧪 Live Risk Predictor")
    with st.form("risk_form"):
        c1,c2,c3 = st.columns(3)
        study_h  = c1.slider("Study Hours/Day",      0.0, 12.0, 5.0, 0.5)
        screen_t = c2.slider("Screen Time/Day",      0.0, 14.0, 6.0, 0.5)
        quiz     = c3.slider("Quiz Score",           0, 100, 65)
        coding   = c1.slider("Coding Score",         0, 100, 60)
        attend   = c2.slider("Attendance (%)",       0, 100, 78)
        task_c   = c3.slider("Task Completion (%)",  0, 100, 72)
        feedback = c1.slider("Feedback Rating",      1.0, 5.0, 3.5, 0.1)
        engage   = c2.slider("Engagement Score",     0, 100, 65)
        comm     = c3.slider("Communication Score",  0, 100, 60)
        tech     = c1.slider("Technical Assessment", 0, 100, 62)
        learn    = c2.slider("Learning Progression", -10.0, 20.0, 5.0, 0.5)
        go_btn   = st.form_submit_button("🔮 Predict Risk")

    if go_btn:
        vals = {
            "study_hours":study_h,"screen_time":screen_t,"quiz_score":quiz,
            "coding_score":coding,"attendance":attend,"task_completion":task_c,
            "feedback_rating":feedback,"engagement_score":engage,
            "communication_score":comm,"technical_assessment":tech,
            "learning_progression":learn
        }
        label, proba = predict_employee(risk_clf, risk_le, vals)
        icon = {"High":"🔴","Medium":"🟡","Low":"🟢"}.get(label,"⚪")
        st.success(f"**Predicted Fatigue Risk: {icon} {label}**")
        st.table(pd.DataFrame.from_dict(proba, orient="index", columns=["Probability (%)"]))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — READINESS ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
def page_readiness(df):
    st.title("🎯 Workforce Readiness Assessment")
    df2 = df.copy()
    df2["suggested_role"] = df2.apply(suggest_role,     axis=1)
    df2["skill_gaps"]     = df2.apply(skill_gaps_label, axis=1)

    st.markdown("### 📦 Readiness by Batch")
    st.dataframe(batch_readiness(df), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_readiness_donut(df), use_container_width=True)
    role_counts = df2["suggested_role"].value_counts().reset_index()
    role_counts.columns = ["Suggested Role","Count"]
    col2.plotly_chart(
        px.bar(role_counts, x="Suggested Role", y="Count", color="Count",
               color_continuous_scale="Blues", title="Role Allocation Suggestions"),
        use_container_width=True)

    st.markdown("### 🏢 Readiness by Department")
    st.dataframe(dept_readiness(df), use_container_width=True, hide_index=True)

    st.markdown("### 🔍 Employee Readiness Table (with Skill Gaps)")
    cols = ["employee_id","name","department","role","performance_score",
            "readiness_level","fatigue_risk","suggested_role","skill_gaps"]
    st.dataframe(df2[cols].head(50), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ML MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def page_ml(risk_m, ready_m, fi):
    st.title("🤖 Machine Learning Model Insights")

    st.markdown("### 🎯 Fatigue Risk Model (Random Forest)")
    c1,c2,c3 = st.columns(3)
    c1.metric("Accuracy",      f"{risk_m['accuracy']*100:.1f}%")
    c2.metric("F1 Score",      f"{risk_m['f1']:.4f}")
    c3.metric("CV Accuracy",   f"{risk_m['cv_mean']*100:.1f}%")

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_feat_importance(fi), use_container_width=True)
    with col2:
        st.pyplot(fig_conf_matrix(risk_m["cm"], risk_m["classes"]))

    st.markdown("### 📋 Classification Report — Risk")
    rows = [{"Class":k, **{kk:round(vv,3) for kk,vv in v.items()}}
            for k,v in risk_m["report"].items() if isinstance(v, dict)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🎯 Readiness Model (Random Forest)")
    c1,c2 = st.columns(2)
    c1.metric("Accuracy", f"{ready_m['accuracy']*100:.1f}%")
    c2.metric("F1 Score", f"{ready_m['f1']:.4f}")
    st.pyplot(fig_conf_matrix(ready_m["cm"], ready_m["classes"], "Readiness – Confusion Matrix"))

    st.markdown("### 📋 Classification Report — Readiness")
    rows2 = [{"Class":k, **{kk:round(vv,3) for kk,vv in v.items()}}
             for k,v in ready_m["report"].items() if isinstance(v, dict)]
    st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
def page_recommendations(df):
    st.title("💡 Recommendations & Intervention Module")
    report = onboarding_report(df)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Onboarding Effectiveness", report["Onboarding Effectiveness"])
    c2.metric("Attrition Risk",           report["Attrition Risk"])
    c3.metric("Avg Performance",          report["Avg Performance"])
    c4.metric("Status",                   report["Status"])

    st.markdown("### 🚨 Intervention Plan — High Risk Employees")
    plan = intervention_plan(df)
    if plan.empty:
        st.info("No high-risk employees in the current filter.")
    else:
        st.dataframe(plan, use_container_width=True, hide_index=True)
        urg = plan["Urgency"].value_counts().reset_index()
        urg.columns = ["Urgency","Count"]
        st.plotly_chart(px.pie(urg, names="Urgency", values="Count", hole=0.4,
                               title="Intervention Urgency Distribution"),
                        use_container_width=True)

    st.markdown("### 🎯 Personalized Plan Generator")
    emp = st.selectbox("Select Employee", df["name"].tolist())
    row = df[df["name"]==emp].iloc[0]
    st.markdown(personalized_plan(row))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — EMPLOYEE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
def page_employee_lookup(df):
    st.title("🔍 Employee Lookup & Individual Analysis")
    search = st.text_input("Search by ID or Name")
    filtered = df[
        df["name"].str.contains(search, case=False, na=False) |
        df["employee_id"].str.contains(search, case=False, na=False)
    ] if search else df

    if filtered.empty:
        st.warning("No employees match your search.")
        return

    eid = st.selectbox("Select Employee", filtered["employee_id"].tolist())
    row = df[df["employee_id"]==eid].iloc[0]

    c1,c2,c3 = st.columns(3)
    c1.metric("Performance Score", row["performance_score"])
    c2.metric("Fatigue Risk",      row["fatigue_risk"])
    c3.metric("Readiness Level",   row["readiness_level"])

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_radar(row), use_container_width=True)
    with col2:
        st.markdown("### 📋 Profile Details")
        for k, v in {
            "Employee ID":row.employee_id, "Name":row["name"],
            "Department":row.department, "Role":row.role, "Batch":row.batch,
            "Quiz Score":row.quiz_score, "Coding Score":row.coding_score,
            "Attendance":f"{row.attendance}%","Task Completion":f"{row.task_completion}%",
            "Engagement Score":row.engagement_score,
            "Communication Score":row.communication_score,
            "Technical Assessment":row.technical_assessment,
            "Study Hours/Day":row.study_hours,
            "Feedback Rating":row.feedback_rating,
            "Learning Progression":row.learning_progression,
        }.items():
            st.write(f"**{k}:** {v}")

    st.markdown("### 💡 Personalized Improvement Plan")
    st.markdown(personalized_plan(row))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df_raw = get_data()
    risk_clf, risk_le, risk_m, ready_clf, ready_le, ready_m, fi = get_models(df_raw)
    page, df = build_sidebar(df_raw)

    if   page == "🏠 Dashboard Overview":    page_dashboard(df)
    elif page == "📊 Performance Analysis":  page_performance(df)
    elif page == "⚠️ Risk Prediction":        page_risk(df, risk_clf, risk_le, risk_m)
    elif page == "🎯 Readiness Assessment":  page_readiness(df)
    elif page == "🤖 ML Model Insights":     page_ml(risk_m, ready_m, fi)
    elif page == "💡 Recommendations":       page_recommendations(df)
    elif page == "🔍 Employee Lookup":       page_employee_lookup(df)


if __name__ == "__main__":
    main()
