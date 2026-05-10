"""
AttendanceIQ Pro · ML Pipeline v5.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chennai Institute of Technology | Smart Attendance Intelligence

Pipeline Architecture
─────────────────────
  Stage 1  →  Data loading + validation
  Stage 2  →  30+ feature engineering (temporal · behavioral · contextual)
  Stage 3  →  SMOTE class balancing
  Stage 4  →  Ensemble training: RF + GBM + LogReg + XGBoost + MLP
  Stage 5  →  Stacking meta-learner
  Stage 6  →  Grade prediction model (Random Forest Regressor)
  Stage 7  →  Student cluster analysis (K-Means behavioral segments)
  Stage 8  →  Dual anomaly detection (IsolationForest + LocalOutlierFactor)
  Stage 9  →  14-day attendance forecast with confidence intervals
  Stage 10 →  Academic Health Index + Dropout Risk scoring
  Stage 11 →  Smart alert generation (priority-ranked notifications)
  Stage 12 →  SHAP explainability (optional, graceful fallback)

Author : Deol Allwyn Samuel J B · VLSI · CIT · Afynix Digital
Version: 5.0.0
"""

import os
import json
import pickle
import warnings
import time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from sklearn.preprocessing    import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble         import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    IsolationForest,
    StackingClassifier,
)
from sklearn.linear_model     import LogisticRegression
from sklearn.neural_network   import MLPClassifier
from sklearn.neighbors        import LocalOutlierFactor, KNeighborsClassifier
from sklearn.cluster          import KMeans
from sklearn.model_selection  import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    learning_curve,
)
from sklearn.metrics          import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score,
    precision_score, recall_score, mean_absolute_error, r2_score,
)
from sklearn.decomposition    import PCA
from scipy.stats              import pearsonr, spearmanr
from imblearn.over_sampling   import SMOTE, ADASYN
from imblearn.combine         import SMOTETomek

# Optional heavy deps — graceful degradation if not installed
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [INFO] XGBoost not installed — skipping XGB model")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

VERSION      = "5.0.0"
RANDOM_STATE = 42
MODEL_DIR    = "models"
DATA_DIR     = "data"
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_attendance(path: str = "data/attendance.csv") -> pd.DataFrame:
    """Load and type-cast the main attendance CSV."""
    if not os.path.exists(path):
        print("  [WARN] attendance.csv not found — running generator…")
        from generate_data import generate
        generate()
    df = pd.read_csv(path, parse_dates=["date"])
    df["late"]   = df["late"].astype(bool)
    df["hostel"] = df.get("hostel", pd.Series([False] * len(df))).fillna(False).astype(bool)
    return df


def load_grades(path: str = "data/grades.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_medical_leaves(path: str = "data/medical_leaves.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_teachers(path: str = "data/teachers.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_live_attendance(csv_path: str = "data/live_attendance.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["student_id", "student_name", "subject",
                                     "date", "time", "method", "confidence", "status"])
    return pd.read_csv(csv_path)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CORE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def attendance_summary(df: pd.DataFrame) -> Dict:
    """High-level KPIs for the command center."""
    total    = len(df)
    present  = (df["status"] == "Present").sum()
    absent   = (df["status"] == "Absent").sum()
    late     = df["late"].sum()
    by_face  = (df["method"] == "Face Recognition").sum()
    conf_df  = df[df["confidence"] > 0]["confidence"]

    rain_days = df.get("rain_day", pd.Series([False] * len(df))).sum()
    med_leave = df.get("on_medical_leave", pd.Series([False] * len(df))).sum()

    avg_att   = round(present / total * 100, 1)
    at_risk_n = (df.groupby("student_id")["status"]
                   .apply(lambda x: (x == "Present").mean())
                   .lt(0.75).sum())

    return {
        "total_records":   int(total),
        "present":         int(present),
        "absent":          int(absent),
        "late":            int(late),
        "attendance_pct":  avg_att,
        "face_recog_pct":  round(by_face / total * 100, 1),
        "avg_confidence":  round(conf_df.mean() * 100, 1) if len(conf_df) > 0 else 0,
        "total_students":  int(df["student_id"].nunique()),
        "total_days":      int(df["date"].nunique()),
        "total_subjects":  int(df["subject"].nunique()),
        "total_depts":     int(df["department"].nunique()),
        "rain_affected":   int(rain_days),
        "medical_leaves":  int(med_leave),
        "at_risk_count":   int(at_risk_n),
    }


def student_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-student aggregated report with risk scoring."""
    grp = df.groupby("student_id").agg(
        name         = ("student_name", "first"),
        dept         = ("department",   "first"),
        year         = ("year",         "first"),
        email        = ("email",        "first"),
        pattern      = ("pattern",      "first"),
        total        = ("status",       "count"),
        present      = ("status",       lambda x: (x == "Present").sum()),
        absent       = ("status",       lambda x: (x == "Absent").sum()),
        late         = ("late",         "sum"),
        late_mins    = ("late_minutes", "sum"),
        avg_conf     = ("confidence",   "mean"),
        rain_absent  = ("rain_day",     lambda x: (x & (df.loc[x.index, "status"] == "Absent")).sum()
                        if "rain_day" in df.columns else 0),
    ).reset_index()

    grp["pct"]           = (grp["present"] / grp["total"] * 100).round(1)
    grp["absent_rate"]   = (grp["absent"]  / grp["total"] * 100).round(1)
    grp["late_rate"]     = (grp["late"]    / grp["total"] * 100).round(1)
    grp["risk_score"]    = grp.apply(_risk_score, axis=1)
    grp["risk_label"]    = grp["pct"].apply(
        lambda x: "Critical" if x < 60 else "At Risk" if x < 75
        else "Low Risk" if x < 85 else "Good"
    )
    grp["dropout_risk"]  = grp.apply(_dropout_risk_score, axis=1)
    grp["ahi"]           = grp.apply(_academic_health_index, axis=1)

    return grp.sort_values("pct")


def _risk_score(row: pd.Series) -> float:
    """Composite risk 0-100 (higher = more at risk). Multi-factor."""
    pct_risk    = max(0.0, (85.0 - row["pct"]) * 1.25)
    late_risk   = min(float(row.get("late_rate", 0)) * 0.8, 20.0)
    absent_risk = min(float(row.get("absent_rate", 0)) * 0.6, 30.0)
    return float(min(round(pct_risk + late_risk + absent_risk, 1), 100.0))


def _dropout_risk_score(row: pd.Series) -> float:
    """
    Dropout early-warning score 0-100.
    Combines attendance, late rate, and absence trajectory.
    Students below 50% attendance get exponential risk boost.
    """
    pct       = float(row["pct"])
    late_rate = float(row.get("late_rate", 0))
    absent_r  = float(row.get("absent_rate", 0))

    base = max(0.0, (70.0 - pct) * 1.5)
    if pct < 50:
        base *= 1.8    # exponential danger zone
    late_penalty   = late_rate * 1.2
    absent_penalty = absent_r * 0.8
    return float(min(round(base + late_penalty + absent_penalty, 1), 100.0))


def _academic_health_index(row: pd.Series) -> float:
    """
    AHI: composite academic health 0-100.
    Higher = healthier academic trajectory.
    """
    att_score  = row["pct"]
    late_pen   = min(row.get("late_rate", 0) * 2, 20)
    risk_pen   = row.get("risk_score", 0) * 0.2
    ahi = max(0, min(100, att_score - late_pen - risk_pen * 0.1))
    return round(ahi, 1)


def subject_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject aggregated stats including difficulty proxy."""
    grp = df.groupby("subject").agg(
        total   = ("status", "count"),
        present = ("status", lambda x: (x == "Present").sum()),
        absent  = ("status", lambda x: (x == "Absent").sum()),
        dept    = ("department", "first"),
        late    = ("late", "sum"),
        avg_conf= ("confidence", "mean"),
    ).reset_index()
    grp["pct"]       = (grp["present"] / grp["total"] * 100).round(1)
    grp["late_rate"] = (grp["late"]    / grp["total"] * 100).round(1)
    return grp.sort_values("pct")


def dept_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-department aggregated stats."""
    grp = df.groupby("department").agg(
        total    = ("status", "count"),
        present  = ("status", lambda x: (x == "Present").sum()),
        students = ("student_id", "nunique"),
        subjects = ("subject",    "nunique"),
    ).reset_index()
    grp["pct"] = (grp["present"] / grp["total"] * 100).round(1)
    return grp.sort_values("pct", ascending=False)


def daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Daily attendance % with rolling averages and trend."""
    grp = df.groupby("date").agg(
        total   = ("status", "count"),
        present = ("status", lambda x: (x == "Present").sum()),
        absent  = ("status", lambda x: (x == "Absent").sum()),
    ).reset_index()
    grp["pct"]        = (grp["present"] / grp["total"] * 100).round(1)
    grp["rolling_7d"] = grp["pct"].rolling(7,  min_periods=1).mean().round(1)
    grp["rolling_14d"]= grp["pct"].rolling(14, min_periods=1).mean().round(1)
    grp["trend"]      = grp["pct"].diff().fillna(0).round(2)
    grp["volatility"] = grp["pct"].rolling(7, min_periods=1).std().fillna(0).round(2)
    return grp


def hourly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Session-hour attendance patterns."""
    df2 = df.copy()
    df2["hour"] = df2["session_time"].str.split(":").str[0].astype(int)
    grp = df2.groupby("hour").agg(
        total   = ("status", "count"),
        present = ("status", lambda x: (x == "Present").sum()),
        late    = ("late",   "sum"),
    ).reset_index()
    grp["pct"]       = (grp["present"] / grp["total"] * 100).round(1)
    grp["late_rate"] = (grp["late"]    / grp["total"] * 100).round(1)
    return grp


def at_risk_students(df: pd.DataFrame, threshold: float = 75.0) -> pd.DataFrame:
    """Students below the attendance threshold — sorted by severity."""
    return student_report(df).pipe(lambda r: r[r["pct"] < threshold]).sort_values("pct")


def streak_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Current attendance streak for every student."""
    results = []
    for sid in df["student_id"].unique():
        sdf   = df[df["student_id"] == sid].sort_values("date")
        dates = (sdf.groupby("date")["status"]
                 .apply(lambda x: "Present" if (x == "Present").any() else "Absent")
                 .reset_index())
        streak = 0
        for _, row in dates.iloc[::-1].iterrows():
            if row["status"] == "Present":
                streak += 1
            else:
                break
        results.append({
            "student_id":   sid,
            "student_name": sdf["student_name"].iloc[0],
            "dept":         sdf["department"].iloc[0],
            "streak":       streak,
            "total_days":   len(dates),
            "present_days": int((dates["status"] == "Present").sum()),
        })
    return pd.DataFrame(results).sort_values("streak", ascending=False)


def weekly_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Week × Day attendance % pivot for heatmap rendering."""
    df2       = df.copy()
    df2["week"]= df2["date"].dt.isocalendar().week.astype(int)
    piv = df2.groupby(["week", "day"])["status"].apply(
        lambda x: (x == "Present").sum() / len(x) * 100
    ).unstack().round(1)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return piv.reindex(columns=[d for d in day_order if d in piv.columns])


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Month-level aggregation for semester comparison."""
    df2 = df.copy()
    df2["month_yr"] = df2["date"].dt.to_period("M").astype(str)
    return df2.groupby(["month_yr", "department"]).agg(
        pct = ("status", lambda x: (x == "Present").sum() / len(x) * 100)
    ).reset_index().round(1)


def semester_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Semester × department comparison."""
    return df.groupby(["semester", "department"]).agg(
        pct      = ("status", lambda x: (x == "Present").sum() / len(x) * 100),
        students = ("student_id", "nunique"),
    ).reset_index().round(1)


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRADE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def grade_analysis(df_att: pd.DataFrame, df_grades: pd.DataFrame) -> Dict:
    """
    Correlate attendance rates with academic performance.
    Returns data for the Grade Intelligence dashboard.
    """
    if df_grades.empty:
        return {}

    # Student-level: attendance % vs GPA-equivalent
    stu_att = (df_att.groupby("student_id")["status"]
                     .apply(lambda x: (x == "Present").mean() * 100)
                     .reset_index(name="att_pct"))

    stu_gpa = (df_grades.groupby("student_id")
                        .apply(lambda x: (
                            (x["grade_point"] * x["credits"]).sum()
                            / x["credits"].sum()
                            if x["credits"].sum() > 0 else 0
                        ))
                        .reset_index(name="cgpa"))

    merged = stu_att.merge(stu_gpa, on="student_id").merge(
        df_att[["student_id","student_name","department","year","pattern"]].drop_duplicates(),
        on="student_id",
    )

    if len(merged) < 3:
        return {"merged": merged}

    corr_pearson,  p1 = pearsonr(merged["att_pct"],  merged["cgpa"])
    corr_spearman, p2 = spearmanr(merged["att_pct"], merged["cgpa"])

    # Subject-level: pass rate vs average attendance
    sub_att  = (df_att.groupby("subject")["status"]
                      .apply(lambda x: (x == "Present").mean() * 100)
                      .reset_index(name="att_pct"))
    sub_pass = (df_grades.groupby("subject")
                         .apply(lambda x: (x["grade"] != "U").mean() * 100)
                         .reset_index(name="pass_rate"))
    sub_merged = sub_att.merge(sub_pass, on="subject")

    # CGPA category distribution
    merged["cgpa_cat"] = pd.cut(
        merged["cgpa"],
        bins=[0, 5, 6, 7, 8, 9, 10.01],
        labels=["<5","5-6","6-7","7-8","8-9","9-10"],
    )

    # Grade distribution
    grade_dist = df_grades["grade"].value_counts().reset_index()
    grade_dist.columns = ["grade", "count"]

    # Detained students
    detained = df_grades[df_grades["grade"] == "DETAINED"].groupby(
        "student_id"
    ).size().reset_index(name="detained_subjects")

    return {
        "merged":          merged,
        "corr_pearson":    round(corr_pearson,  4),
        "corr_spearman":   round(corr_spearman, 4),
        "p_value_pearson": round(p1, 4),
        "sub_merged":      sub_merged,
        "grade_dist":      grade_dist,
        "detained":        detained,
        "avg_cgpa":        round(merged["cgpa"].mean(), 2),
        "top10_cgpa":      merged.nlargest(10, "cgpa")[["student_name","department","cgpa","att_pct"]],
        "low10_cgpa":      merged.nsmallest(10, "cgpa")[["student_name","department","cgpa","att_pct"]],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING (30+ features)
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame):
    """
    Build 30+ features per attendance record.
    Returns (df_feat, feature_list, encoders_dict).
    """
    df2 = df.copy()

    # ── Temporal ──────────────────────────────────────────────
    df2["hour"]           = df2["session_time"].str.split(":").str[0].astype(int)
    df2["is_morning"]     = (df2["hour"] < 12).astype(int)
    df2["is_afternoon"]   = (df2["hour"] >= 13).astype(int)
    df2["is_early"]       = (df2["hour"] <= 9).astype(int)
    df2["is_monday"]      = (df2["day"] == "Monday").astype(int)
    df2["is_friday"]      = (df2["day"] == "Friday").astype(int)
    df2["is_weekend_adj"] = ((df2["day"].isin(["Monday", "Friday"])).astype(int))
    df2["is_midweek"]     = (df2["day"].isin(["Tuesday", "Wednesday", "Thursday"])).astype(int)
    df2["month"]          = df2["date"].dt.month
    df2["week_no"]        = df2["date"].dt.isocalendar().week.astype(int)
    df2["day_of_year"]    = df2["date"].dt.dayofyear
    df2["is_exam_month"]  = df2["month"].isin([11, 12, 4, 5]).astype(int)
    df2["is_monsoon"]     = df2["month"].isin([10, 11, 12]).astype(int)
    df2["quarter"]        = ((df2["month"] - 1) // 3 + 1)

    # ── Contextual flags ──────────────────────────────────────
    df2["rain_day"]       = df2.get("rain_day",        pd.Series([0]*len(df2))).fillna(0).astype(int)
    df2["on_med_leave"]   = df2.get("on_medical_leave",pd.Series([0]*len(df2))).fillna(0).astype(int)
    df2["hostel"]         = df2.get("hostel",          pd.Series([0]*len(df2))).fillna(0).astype(int)
    df2["has_exam"]       = df2.get("exam_period",     pd.Series([""]  * len(df2))).ne("").astype(int)

    # ── Per-student historical stats ──────────────────────────
    stu_stats = df2.groupby("student_id").agg(
        stu_pct        = ("status",         lambda x: (x == "Present").mean()),
        stu_late_rate  = ("late",           "mean"),
        stu_sessions   = ("status",         "count"),
        stu_avg_conf   = ("confidence",     "mean"),
        stu_absent_run = ("status",         lambda x: _max_consecutive_absent(x)),
    ).reset_index()
    df2 = df2.merge(stu_stats, on="student_id", how="left")

    # ── Per-subject stats ─────────────────────────────────────
    sub_stats = df2.groupby("subject").agg(
        sub_pct        = ("status", lambda x: (x == "Present").mean()),
        sub_late_rate  = ("late",   "mean"),
    ).reset_index()
    df2 = df2.merge(sub_stats, on="subject", how="left")

    # ── Per-department stats ──────────────────────────────────
    dept_stats = df2.groupby("department").agg(
        dept_pct = ("status", lambda x: (x == "Present").mean()),
    ).reset_index()
    df2 = df2.merge(dept_stats, on="department", how="left")

    # ── Per-day-of-week stats ─────────────────────────────────
    day_stats = df2.groupby("day").agg(
        day_pct = ("status", lambda x: (x == "Present").mean()),
    ).reset_index()
    df2 = df2.merge(day_stats, on="day", how="left")

    # ── Encode categoricals ───────────────────────────────────
    le_day  = LabelEncoder(); df2["day_enc"]  = le_day.fit_transform(df2["day"])
    le_sub  = LabelEncoder(); df2["sub_enc"]  = le_sub.fit_transform(df2["subject"])
    le_dept = LabelEncoder(); df2["dept_enc"] = le_dept.fit_transform(df2["department"])
    le_pat  = LabelEncoder(); df2["pat_enc"]  = le_pat.fit_transform(df2["pattern"])
    le_meth = LabelEncoder(); df2["meth_enc"] = le_meth.fit_transform(df2["method"].fillna("Unknown"))

    df2["target"] = (df2["status"] == "Absent").astype(int)

    FEATURES = [
        # temporal
        "hour", "is_morning", "is_afternoon", "is_early",
        "is_monday", "is_friday", "is_weekend_adj", "is_midweek",
        "month", "week_no", "is_exam_month", "is_monsoon", "quarter",
        # contextual
        "rain_day", "on_med_leave", "hostel", "has_exam",
        # year
        "year",
        # student history
        "stu_pct", "stu_late_rate", "stu_sessions", "stu_avg_conf", "stu_absent_run",
        # subject/dept/day aggregates
        "sub_pct", "sub_late_rate", "dept_pct", "day_pct",
        # encoded categoricals
        "day_enc", "sub_enc", "dept_enc", "pat_enc",
    ]

    encoders = {
        "le_day": le_day, "le_sub": le_sub,
        "le_dept": le_dept, "le_pat": le_pat, "le_meth": le_meth,
    }
    return df2, FEATURES, encoders


def _max_consecutive_absent(status_series: pd.Series) -> int:
    """Longest run of consecutive absences for a student."""
    max_run = 0
    cur_run = 0
    for s in status_series:
        if s == "Absent":
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENSEMBLE TRAINING: RF + GBM + LogReg + XGBoost + MLP + Stacking
# ══════════════════════════════════════════════════════════════════════════════

def train_absence_predictor_v2(df: pd.DataFrame, fast_mode: bool = False):
    """
    Train the full absence-prediction ensemble.
    Returns (results_dict, scaler, feature_list, best_model_name, encoders).
    Compatible signature with app.py imports.
    """
    return train_ensemble_v3(df, fast_mode=fast_mode)


def train_ensemble_v3(df: pd.DataFrame, fast_mode: bool = False):
    """
    Full 5-model ensemble with stacking meta-learner.
    Trained with SMOTE balancing and 5-fold cross-validation.
    """
    print("\n" + "-" * 56)
    print("  AttendanceIQ Pro · ML Ensemble v5 Training")
    print("-" * 56)

    df_feat, FEATURES, encoders = engineer_features(df)

    X = df_feat[FEATURES].fillna(0).values
    y = df_feat["target"].values

    print(f"  Samples     : {len(X):,}  |  Features: {len(FEATURES)}")
    print(f"  Class dist  : Present={int((y==0).sum())}  Absent={int((y==1).sum())}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    if fast_mode:
        X_tr_b, y_tr_b = X_tr, y_tr
        print(f"  [FAST MODE] Skipping SMOTE — using {len(X_tr_b)} samples")
    else:
        smt = SMOTETomek(random_state=RANDOM_STATE)
        X_tr_b, y_tr_b = smt.fit_resample(X_tr, y_tr)
        print(f"  SMOTE+Tomek : {len(X_tr)} -> {len(X_tr_b)} training samples")

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_b)
    X_te_sc  = scaler.transform(X_te)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

   # ── Base models ────────────────────────────────────────────────────────────
    if fast_mode:
        print("  [FAST MODE] Using lightweight models for cloud deployment")
        models_def = {
            "Random Forest": RandomForestClassifier(
                n_estimators=50, max_depth=8, min_samples_leaf=4,
                max_features="sqrt", class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "Logistic Reg": LogisticRegression(
                C=1.2, class_weight="balanced",
                max_iter=500, random_state=RANDOM_STATE,
            ),
        }
    else:
        models_def = {
            "Random Forest": RandomForestClassifier(
                n_estimators=350, max_depth=12, min_samples_leaf=2,
                max_features="sqrt", class_weight="balanced",
                oob_score=True, random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "Gradient Boost": GradientBoostingClassifier(
                n_estimators=250, learning_rate=0.07, max_depth=5,
                subsample=0.85, min_samples_leaf=3,
                random_state=RANDOM_STATE,
            ),
            "Logistic Reg": LogisticRegression(
                C=1.2, class_weight="balanced",
                max_iter=1200, random_state=RANDOM_STATE,
            ),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu", solver="adam",
                alpha=0.001, learning_rate="adaptive",
                max_iter=300, random_state=RANDOM_STATE,
                early_stopping=True, validation_fraction=0.1,
            ),
        }
        if HAS_XGB:
            models_def["XGBoost"] = xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.08, max_depth=6,
                subsample=0.85, colsample_bytree=0.85,
                use_label_encoder=False, eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=-1,
            )
    results = {}
    for name, mdl in models_def.items():
        t0 = time.time()
        mdl.fit(X_tr_sc, y_tr_b)
        y_pred = mdl.predict(X_te_sc)
        y_prob = mdl.predict_proba(X_te_sc)[:, 1]

        cv_f1  = cross_val_score(mdl, X_tr_sc, y_tr_b, cv=cv, scoring="f1",       n_jobs=-1)
        cv_acc = cross_val_score(mdl, X_tr_sc, y_tr_b, cv=cv, scoring="accuracy", n_jobs=-1)

        fi = (pd.Series(mdl.feature_importances_, index=FEATURES)
              .sort_values(ascending=False)
              if hasattr(mdl, "feature_importances_") else pd.Series(dtype=float))

        oob_txt = (f"  OOB={mdl.oob_score_:.4f}"
                   if hasattr(mdl, "oob_score_") else "")
        elapsed = time.time() - t0

        results[name] = {
            "model":       mdl,
            "accuracy":    round(accuracy_score(y_te, y_pred),      4),
            "f1":          round(f1_score(y_te,y_pred,zero_division=0), 4),
            "precision":   round(precision_score(y_te,y_pred,zero_division=0), 4),
            "recall":      round(recall_score(y_te,y_pred,zero_division=0), 4),
            "auc":         round(roc_auc_score(y_te, y_prob),        4),
            "cv_f1_mean":  round(cv_f1.mean(),   4),
            "cv_f1_std":   round(cv_f1.std(),    4),
            "cv_acc_mean": round(cv_acc.mean(),  4),
            "cv_acc_std":  round(cv_acc.std(),   4),
            "cm":          confusion_matrix(y_te, y_pred).tolist(),
            "fi":          fi,
            "y_test":      y_te,
            "y_pred":      y_pred,
            "y_prob":      y_prob,
            "train_time":  round(elapsed, 2),
        }
        print(f"  {name:20s}  Acc={results[name]['accuracy']:.4f}  "
              f"F1={results[name]['f1']:.4f}  AUC={results[name]['auc']:.4f}"
              f"{oob_txt}  [{elapsed:.1f}s]")

    # ── Stacking ensemble ──────────────────────────────────────────────────────
    # ── Stacking ensemble (skipped in fast mode) ───────────────────────────────
    if not fast_mode:
        stk_estimators = [
            ("rf",  results["Random Forest"]["model"]),
            ("gb",  results["Gradient Boost"]["model"]),
            ("lr",  results["Logistic Reg"]["model"]),
        ]
        if "XGBoost" in results:
            stk_estimators.append(("xgb", results["XGBoost"]["model"]))

        stacking = StackingClassifier(
            estimators=stk_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=800, random_state=RANDOM_STATE),
            cv=3, n_jobs=-1,
        )
        t0 = time.time()
        stacking.fit(X_tr_sc, y_tr_b)
        y_pred_stk = stacking.predict(X_te_sc)
        y_prob_stk = stacking.predict_proba(X_te_sc)[:, 1]
        cv_f1_stk  = cross_val_score(stacking, X_tr_sc, y_tr_b, cv=cv, scoring="f1", n_jobs=-1)
        elapsed    = time.time() - t0

        results["Stacking Ensemble"] = {
            "model":       stacking,
            "accuracy":    round(accuracy_score(y_te, y_pred_stk), 4),
            "f1":          round(f1_score(y_te, y_pred_stk, zero_division=0), 4),
            "precision":   round(precision_score(y_te, y_pred_stk, zero_division=0), 4),
            "recall":      round(recall_score(y_te, y_pred_stk, zero_division=0), 4),
            "auc":         round(roc_auc_score(y_te, y_prob_stk), 4),
            "cv_f1_mean":  round(cv_f1_stk.mean(), 4),
            "cv_f1_std":   round(cv_f1_stk.std(),  4),
            "cv_acc_mean": round(cv_f1_stk.mean(), 4),
            "cv_acc_std":  round(cv_f1_stk.std(),  4),
            "cm":          confusion_matrix(y_te, y_pred_stk).tolist(),
            "fi":          pd.Series(dtype=float),
            "y_test":      y_te,
            "y_pred":      y_pred_stk,
            "y_prob":      y_prob_stk,
            "train_time":  round(elapsed, 2),
        }
        print(f"  {'Stacking Ensemble':20s}  Acc={results['Stacking Ensemble']['accuracy']:.4f}  "
              f"F1={results['Stacking Ensemble']['f1']:.4f}  "
              f"AUC={results['Stacking Ensemble']['auc']:.4f}  [{elapsed:.1f}s]")
         
    if "XGBoost" in results:
        stk_estimators.append(("xgb", results["XGBoost"]["model"]))

    stacking = StackingClassifier(
        estimators=stk_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=800, random_state=RANDOM_STATE),
        cv=3, n_jobs=-1,
    )
    t0 = time.time()
    stacking.fit(X_tr_sc, y_tr_b)
    y_pred_stk = stacking.predict(X_te_sc)
    y_prob_stk = stacking.predict_proba(X_te_sc)[:, 1]
    cv_f1_stk  = cross_val_score(stacking, X_tr_sc, y_tr_b, cv=cv, scoring="f1", n_jobs=-1)
    elapsed    = time.time() - t0

    results["Stacking Ensemble"] = {
        "model":       stacking,
        "accuracy":    round(accuracy_score(y_te, y_pred_stk), 4),
        "f1":          round(f1_score(y_te, y_pred_stk, zero_division=0), 4),
        "precision":   round(precision_score(y_te, y_pred_stk, zero_division=0), 4),
        "recall":      round(recall_score(y_te, y_pred_stk, zero_division=0), 4),
        "auc":         round(roc_auc_score(y_te, y_prob_stk), 4),
        "cv_f1_mean":  round(cv_f1_stk.mean(), 4),
        "cv_f1_std":   round(cv_f1_stk.std(),  4),
        "cv_acc_mean": round(cv_f1_stk.mean(), 4),
        "cv_acc_std":  round(cv_f1_stk.std(),  4),
        "cm":          confusion_matrix(y_te, y_pred_stk).tolist(),
        "fi":          pd.Series(dtype=float),
        "y_test":      y_te,
        "y_pred":      y_pred_stk,
        "y_prob":      y_prob_stk,
        "train_time":  round(elapsed, 2),
    }
    print(f"  {'Stacking Ensemble':20s}  Acc={results['Stacking Ensemble']['accuracy']:.4f}  "
          f"F1={results['Stacking Ensemble']['f1']:.4f}  "
          f"AUC={results['Stacking Ensemble']['auc']:.4f}  [{elapsed:.1f}s]")

    best = max(results, key=lambda x: results[x]["f1"])
    print(f"\n  [BEST] {best}  F1={results[best]['f1']:.4f}")

    # ── Persist ────────────────────────────────────────────────────────────────
    with open(f"{MODEL_DIR}/classifier.pkl", "wb") as fh:
        pickle.dump({"models": results, "scaler": scaler,
                     "features": FEATURES, "best": best,
                     "encoders": encoders}, fh)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

    _save_meta(results, best, FEATURES)

    return results, scaler, FEATURES, best, encoders


def _save_meta(results: Dict, best: str, features: List[str]):
    """Serialise model metrics to JSON (no sklearn objects)."""
    meta = {}
    for k, v in results.items():
        meta[k] = {
            kk: (vv.tolist() if hasattr(vv, "tolist") else
                 vv.to_dict() if isinstance(vv, pd.Series) else vv)
            for kk, vv in v.items()
            if not callable(getattr(vv, "predict", None))
               and kk not in ("model",)
        }
    with open(f"{MODEL_DIR}/meta.json", "w") as fh:
        json.dump({"results": meta, "best": best, "features": features}, fh, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# 6. GRADE PREDICTION MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_grade_predictor(df_att: pd.DataFrame, df_grades: pd.DataFrame):
    """
    Train a RandomForest regressor to predict a student's CGPA
    from their attendance patterns.
    Returns (model, scaler, feature_names, eval_metrics).
    """
    if df_grades.empty or len(df_grades) < 50:
        return None, None, [], {}

    # Features: per-student attendance stats
    stu_att = df_att.groupby("student_id").agg(
        att_pct        = ("status",      lambda x: (x=="Present").mean() * 100),
        late_rate      = ("late",        "mean"),
        avg_conf       = ("confidence",  "mean"),
        total_sessions = ("status",      "count"),
        absent_run     = ("status",      _max_consecutive_absent),
        mon_pct        = ("status",      lambda x: (
            (x[(df_att.loc[x.index,"day"]=="Monday")=="Monday"] == "Present").mean() * 100
            if (df_att.loc[x.index,"day"]=="Monday").any() else 75
        )),
    ).reset_index()

    stu_gpa = (df_grades.groupby("student_id")
                        .apply(lambda x: (
                            (x["grade_point"] * x["credits"]).sum() / x["credits"].sum()
                            if x["credits"].sum() > 0 else 0
                        ))
                        .reset_index(name="cgpa"))

    merged = stu_att.merge(stu_gpa, on="student_id").dropna()
    if len(merged) < 10:
        return None, None, [], {}

    FEAT = ["att_pct", "late_rate", "avg_conf", "total_sessions", "absent_run"]
    X    = merged[FEAT].fillna(0).values
    y    = merged["cgpa"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr_sc, y_tr)

    y_pred = rf.predict(X_te_sc)
    metrics = {
        "mae":  round(mean_absolute_error(y_te, y_pred), 4),
        "r2":   round(r2_score(y_te, y_pred), 4),
        "rmse": round(float(np.sqrt(((y_te - y_pred)**2).mean())), 4),
    }

    return rf, sc, FEAT, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 7. STUDENT CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

CLUSTER_LABELS = {
    0: ("Star Performers",    "#39ff14"),
    1: ("Consistent Risers",  "#00d4ff"),
    2: ("Moderate Strivers",  "#ffe600"),
    3: ("At-Risk Students",   "#ff8c00"),
    4: ("Critical Zone",      "#ff4500"),
}

def cluster_students(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    K-Means behavioral clustering on student attendance patterns.
    Returns a DataFrame with cluster assignments and PCA coordinates for plotting.
    """
    stu_stats = df.groupby("student_id").agg(
        att_pct        = ("status",      lambda x: (x == "Present").mean() * 100),
        late_rate      = ("late",        lambda x: x.mean() * 100),
        absent_run     = ("status",      _max_consecutive_absent),
        avg_conf       = ("confidence",  "mean"),
        total_sessions = ("status",      "count"),
        mon_pct        = ("status",      lambda x: (
            (df.loc[x.index][df.loc[x.index,"day"]=="Monday"]["status"]=="Present").mean() * 100
            if (df.loc[x.index,"day"]=="Monday").any() else att_pct_fallback(x)
        )),
        fri_pct        = ("status",      lambda x: att_pct_fallback(x)),
    ).reset_index()

    # Merge name / dept
    meta = df[["student_id","student_name","department","year","pattern"]].drop_duplicates("student_id")
    stu_stats = stu_stats.merge(meta, on="student_id")

    FEAT = ["att_pct", "late_rate", "absent_run", "avg_conf"]
    X    = stu_stats[FEAT].fillna(0).values

    sc = MinMaxScaler()
    X_sc = sc.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=15)
    stu_stats["cluster"]       = km.fit_predict(X_sc)
    stu_stats["cluster_dist"]  = km.transform(X_sc).min(axis=1).round(4)

    # PCA for 2-D visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_sc)
    stu_stats["pca_x"] = coords[:, 0].round(4)
    stu_stats["pca_y"] = coords[:, 1].round(4)

    # Sort clusters so cluster 0 = best attendance
    cluster_means = stu_stats.groupby("cluster")["att_pct"].mean().sort_values(ascending=False)
    rank_map = {old: new for new, old in enumerate(cluster_means.index)}
    stu_stats["cluster"] = stu_stats["cluster"].map(rank_map)

    stu_stats["cluster_label"] = stu_stats["cluster"].map(
        lambda c: CLUSTER_LABELS.get(c, (f"Group {c}", "#a78bfa"))[0]
    )
    stu_stats["cluster_color"] = stu_stats["cluster"].map(
        lambda c: CLUSTER_LABELS.get(c, (f"Group {c}", "#a78bfa"))[1]
    )

    return stu_stats.sort_values("cluster")


def att_pct_fallback(x):
    """Fallback when day-filtered slice is empty."""
    return float((x == "Present").mean() * 100)


# ══════════════════════════════════════════════════════════════════════════════
# 8. DUAL ANOMALY DETECTION (IsolationForest + LOF)
# ══════════════════════════════════════════════════════════════════════════════

def anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dual anomaly detection using IsolationForest and LocalOutlierFactor.
    A student is flagged if EITHER detector flags them.
    Returns a merged DataFrame with scores and combined flag.
    """
    daily_stu = df.groupby(["student_id", "date"]).agg(
        sessions = ("status", "count"),
        present  = ("status", lambda x: (x == "Present").sum()),
        late     = ("late",   "sum"),
    ).reset_index()
    daily_stu["pct"] = daily_stu["present"] / daily_stu["sessions"]

    pivot = daily_stu.pivot_table(
        index="student_id", columns="date", values="pct", aggfunc="mean"
    ).fillna(daily_stu["pct"].mean())

    # IsolationForest
    iso    = IsolationForest(contamination=0.10, random_state=RANDOM_STATE)
    if_lbl = iso.fit_predict(pivot)
    if_scr = iso.decision_function(pivot)

    # LocalOutlierFactor
    lof    = LocalOutlierFactor(n_neighbors=min(10, len(pivot) - 1), contamination=0.10)
    lof_lbl= lof.fit_predict(pivot)
    lof_scr= -lof.negative_outlier_factor_   # higher = more outlier-like

    result = pd.DataFrame({
        "student_id":     pivot.index,
        "if_anomaly":     if_lbl == -1,
        "if_score":       if_scr,
        "lof_anomaly":    lof_lbl == -1,
        "lof_score":      lof_scr,
        "anomaly":        (if_lbl == -1) | (lof_lbl == -1),  # union
        "both_flagged":   (if_lbl == -1) & (lof_lbl == -1),
        "score":          if_scr,   # kept for backward compat with app.py
    })

    return result.merge(
        df[["student_id","student_name","department"]].drop_duplicates(),
        on="student_id",
    ).sort_values("if_score")


# ══════════════════════════════════════════════════════════════════════════════
# 9. ATTENDANCE FORECAST (14-day rolling + linear trend)
# ══════════════════════════════════════════════════════════════════════════════

def forecast_attendance(df: pd.DataFrame, days_ahead: int = 14) -> pd.DataFrame:
    """
    Simple but effective rolling-trend forecast.
    Includes seasonal adjustment for monsoon and exam periods.
    """
    trend    = daily_trend(df)
    last_7   = trend["pct"].tail(7).mean()
    last_30  = trend["pct"].tail(30).mean()
    slope    = trend["trend"].tail(14).mean()
    vol      = trend["volatility"].tail(7).mean()

    forecasts = []
    last_date = pd.to_datetime(trend["date"].max())

    for i in range(1, days_ahead + 2):
        fdate = last_date + timedelta(days=i)
        if fdate.weekday() >= 5:   # skip weekends
            continue

        # Seasonal adjustments
        seasonal = 0.0
        if fdate.month in [10, 11, 12]:
            seasonal -= 1.2   # monsoon dip
        if fdate.month in [11, 12]:
            seasonal += 2.0   # exam season boost (fear of shortage)
        if fdate.weekday() == 0:
            seasonal -= 1.0   # Monday dip
        if fdate.weekday() == 4:
            seasonal -= 0.5   # Friday dip

        predicted = round(float(
            np.clip(last_7 + slope * i * 0.25 + seasonal, 58, 99)
        ), 1)
        ci_width  = round(float(np.clip(vol * 1.5 + i * 0.15, 2, 8)), 1)

        forecasts.append({
            "date":      fdate,
            "predicted": predicted,
            "lower":     round(max(predicted - ci_width, 55), 1),
            "upper":     round(min(predicted + ci_width, 100), 1),
        })
        if len(forecasts) >= days_ahead:
            break

    return pd.DataFrame(forecasts)


# ══════════════════════════════════════════════════════════════════════════════
# 10. SMART ALERT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

ALERT_TEMPLATES = {
    "critical_attendance": (
        "URGENT: {name} ({dept}, Y{year}) attendance has dropped to {pct}% — "
        "immediate intervention required. {absent} absences in {total} sessions."
    ),
    "rapid_decline": (
        "ALERT: {name} missed {recent_absent} out of last 10 sessions. "
        "Pattern suggests accelerating disengagement."
    ),
    "late_chronic": (
        "NOTICE: {name} is chronically late — {late_pct}% of attended sessions. "
        "May indicate transport or motivation issues."
    ),
    "anomaly_flagged": (
        "ANOMALY: {name}'s attendance pattern is statistically unusual — "
        "possible sudden life event or academic crisis."
    ),
    "dropout_risk": (
        "DROP-OUT RISK: {name} (Dropout Risk Score: {score}/100) — "
        "multi-factor analysis suggests elevated dropout probability."
    ),
    "medical_recurrent": (
        "MEDICAL PATTERN: {name} has {count} medical leave episodes this semester. "
        "Recommend health counselling referral."
    ),
    "positive_streak": (
        "RECOGNITION: {name} has maintained perfect attendance for {streak} consecutive days! "
        "Consider for attendance award."
    ),
}


def generate_smart_alerts(
    df_att: pd.DataFrame,
    df_leaves: Optional[pd.DataFrame] = None,
    df_anomalies: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate priority-ranked smart alerts for the notification center.
    Alert priorities: P1 (Critical) → P2 (High) → P3 (Medium) → P4 (Info).
    """
    alerts = []
    stu_rpt = student_report(df_att)

    for _, row in stu_rpt.iterrows():
        sid  = row["student_id"]
        name = row["name"]
        dept = row["dept"]
        year = row["year"]
        pct  = row["pct"]
        late_pct    = row["late_rate"]
        dropout_sc  = row["dropout_risk"]

        # P1: Critical attendance
        if pct < 60:
            alerts.append({
                "priority":    1, "priority_label": "P1 CRITICAL",
                "student_id":  sid, "student_name": name,
                "department":  dept, "year": year,
                "alert_type":  "critical_attendance",
                "message": ALERT_TEMPLATES["critical_attendance"].format(
                    name=name, dept=dept, year=year, pct=pct,
                    absent=int(row["absent"]), total=int(row["total"])
                ),
                "action": "Issue attendance warning letter + HOD meeting",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

        # P2: High risk / rapid decline
        elif pct < 75:
            recent = df_att[df_att["student_id"] == sid].sort_values("date").tail(10)
            recent_absent = int((recent["status"] == "Absent").sum())
            if recent_absent >= 5:
                alerts.append({
                    "priority": 2, "priority_label": "P2 HIGH",
                    "student_id": sid, "student_name": name,
                    "department": dept, "year": year,
                    "alert_type": "rapid_decline",
                    "message": ALERT_TEMPLATES["rapid_decline"].format(
                        name=name, recent_absent=recent_absent
                    ),
                    "action": "Assign faculty mentor — immediate",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })

        # P2: Dropout risk
        if dropout_sc >= 70:
            alerts.append({
                "priority": 2, "priority_label": "P2 HIGH",
                "student_id": sid, "student_name": name,
                "department": dept, "year": year,
                "alert_type": "dropout_risk",
                "message": ALERT_TEMPLATES["dropout_risk"].format(
                    name=name, score=int(dropout_sc)
                ),
                "action": "Counselling session + parent notification",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

        # P3: Chronic lateness
        if late_pct >= 20 and pct >= 75:
            alerts.append({
                "priority": 3, "priority_label": "P3 MEDIUM",
                "student_id": sid, "student_name": name,
                "department": dept, "year": year,
                "alert_type": "late_chronic",
                "message": ALERT_TEMPLATES["late_chronic"].format(
                    name=name, late_pct=round(late_pct, 1)
                ),
                "action": "Informal check-in with advisor",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    # Anomaly alerts
    if df_anomalies is not None and len(df_anomalies) > 0:
        flagged = df_anomalies[df_anomalies["anomaly"] == True]
        for _, row in flagged.iterrows():
            alerts.append({
                "priority": 2, "priority_label": "P2 HIGH",
                "student_id": row["student_id"],
                "student_name": row["student_name"],
                "department": row["department"],
                "year": None,
                "alert_type": "anomaly_flagged",
                "message": ALERT_TEMPLATES["anomaly_flagged"].format(
                    name=row["student_name"]
                ),
                "action": "Pattern review — check recent life events",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    # Medical leave alerts
    if df_leaves is not None and not df_leaves.empty:
        med_counts = df_leaves.groupby("student_id").size().reset_index(name="count")
        for _, row in med_counts[med_counts["count"] >= 3].iterrows():
            sid = row["student_id"]
            stu = stu_rpt[stu_rpt["student_id"] == sid]
            if len(stu) == 0:
                continue
            stu = stu.iloc[0]
            alerts.append({
                "priority": 3, "priority_label": "P3 MEDIUM",
                "student_id": sid,
                "student_name": stu["name"],
                "department": stu["dept"],
                "year": stu["year"],
                "alert_type": "medical_recurrent",
                "message": ALERT_TEMPLATES["medical_recurrent"].format(
                    name=stu["name"], count=int(row["count"])
                ),
                "action": "Refer to student health center",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    # P4: Positive recognition
    streaks = streak_analysis(df_att)
    for _, row in streaks[streaks["streak"] >= 30].iterrows():
        alerts.append({
            "priority": 4, "priority_label": "P4 INFO",
            "student_id": row["student_id"],
            "student_name": row["student_name"],
            "department": row["dept"],
            "year": None,
            "alert_type": "positive_streak",
            "message": ALERT_TEMPLATES["positive_streak"].format(
                name=row["student_name"], streak=int(row["streak"])
            ),
            "action": "Issue appreciation certificate",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    df_alerts = pd.DataFrame(alerts).sort_values("priority") if alerts else pd.DataFrame()
    return df_alerts


# ══════════════════════════════════════════════════════════════════════════════
# 11. SHAP EXPLAINABILITY (optional)
# ══════════════════════════════════════════════════════════════════════════════

def shap_summary(model, X_sample: np.ndarray, feature_names: List[str]) -> Optional[Dict]:
    """
    Compute SHAP values for the given model and sample.
    Returns None if SHAP is not installed or model is unsupported.
    """
    if not HAS_SHAP:
        return None
    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_sample)

        shap_vals   = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # class=1 (absent) for binary

        mean_abs = np.abs(shap_vals).mean(axis=0)
        return {
            "shap_values":    shap_vals,
            "mean_abs_shap":  dict(zip(feature_names, mean_abs.round(4))),
            "feature_names":  feature_names,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 12. FACE RECOGNITION SIMULATION + LIVE ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════

def simulate_face_recognition(student_name: str, confidence_threshold: float = 0.80) -> Dict:
    """
    Simulates the full OpenCV face recognition pipeline response.
    In production this would call the actual cv2 + face_recognition pipeline.
    """
    time.sleep(0.25)
    # Realistic confidence distribution: mostly 0.80-0.98, occasional false negatives
    conf       = float(np.clip(np.random.beta(9, 2), 0.70, 0.99))
    recognized = conf >= confidence_threshold
    liveness   = bool(np.random.random() > 0.05)  # 95% liveness pass rate

    return {
        "recognized":          recognized,
        "student":             student_name if recognized else "Unknown",
        "confidence":          round(conf, 3),
        "liveness_check":      liveness,
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method":              "Haar Cascade + HOG Feature Matching",
        "status":              "Present" if recognized else "Unrecognized",
        "face_id":             f"FID_{np.random.randint(10000, 99999)}",
        "bbox":                [42, 38, 180, 195],  # simulated bounding box (x,y,w,h)
        "processing_ms":       round(np.random.uniform(18, 45), 1),
    }


def mark_attendance(
    student_id: str,
    student_name: str,
    subject: str,
    method: str = "Face Recognition",
    confidence: float = 0.95,
    csv_path: str = "data/live_attendance.csv",
) -> Tuple[bool, str]:
    """Write an attendance record to the live CSV. Prevents duplicates."""
    os.makedirs(DATA_DIR, exist_ok=True)
    now    = datetime.now()
    record = {
        "student_id":   student_id,
        "student_name": student_name,
        "subject":      subject,
        "date":         now.strftime("%Y-%m-%d"),
        "time":         now.strftime("%H:%M:%S"),
        "method":       method,
        "confidence":   round(confidence, 3),
        "status":       "Present",
    }
    df_new = pd.DataFrame([record])

    if os.path.exists(csv_path):
        df_ex = pd.read_csv(csv_path)
        dup   = (
            (df_ex["student_id"] == student_id)
            & (df_ex["subject"]  == subject)
            & (df_ex["date"]     == record["date"])
        )
        if dup.any():
            return False, "Already marked for today"
        df_ex = pd.concat([df_ex, df_new], ignore_index=True)
    else:
        df_ex = df_new

    df_ex.to_csv(csv_path, index=False)
    return True, "Attendance marked successfully"


# ══════════════════════════════════════════════════════════════════════════════
# 13. ENGAGEMENT SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_engagement_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-student engagement score (0-100) using:
    - Attendance rate (45%)
    - Punctuality (20%)
    - Face recognition confidence (15%)
    - Attendance trend direction (10%)
    - Exam-period attendance (10%)
    """
    stu_data = []
    for sid in df["student_id"].unique():
        sdf = df[df["student_id"] == sid]

        att_pct  = (sdf["status"] == "Present").mean() * 100
        late_pct = sdf["late"].mean() * 100
        avg_conf = sdf[sdf["confidence"] > 0]["confidence"].mean() * 100 if (sdf["confidence"] > 0).any() else 75

        # Trend: is the student improving or declining?
        daily = sdf.groupby("date")["status"].apply(lambda x: (x=="Present").mean()).reset_index()
        daily = daily.sort_values("date")
        if len(daily) >= 7:
            first_half = daily["status"].head(len(daily)//2).mean() * 100
            second_half= daily["status"].tail(len(daily)//2).mean() * 100
            trend_score= min(max(50 + (second_half - first_half) * 2, 0), 100)
        else:
            trend_score = 50

        # Exam period attendance
        exam_sdf = sdf[sdf.get("exam_period", pd.Series([""]*len(sdf))).ne("")]
        exam_pct = (exam_sdf["status"] == "Present").mean() * 100 if len(exam_sdf) > 0 else att_pct

        engagement = round(
            0.45 * att_pct
            + 0.20 * max(0, 100 - late_pct * 3)
            + 0.15 * min(avg_conf, 100)
            + 0.10 * trend_score
            + 0.10 * exam_pct,
            1
        )

        stu_data.append({
            "student_id":    sid,
            "student_name":  sdf["student_name"].iloc[0],
            "department":    sdf["department"].iloc[0],
            "year":          sdf["year"].iloc[0],
            "att_pct":       round(att_pct, 1),
            "late_pct":      round(late_pct, 1),
            "avg_conf":      round(avg_conf, 1),
            "trend_score":   round(trend_score, 1),
            "exam_pct":      round(exam_pct, 1),
            "engagement":    engagement,
        })

    return pd.DataFrame(stu_data).sort_values("engagement", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 14. DEPARTMENT BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════

def department_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Head-to-head department comparison across multiple KPIs.
    """
    return df.groupby("department").agg(
        attendance_pct = ("status",      lambda x: round((x=="Present").mean()*100, 1)),
        late_rate_pct  = ("late",        lambda x: round(x.mean()*100, 1)),
        face_recog_pct = ("method",      lambda x: round((x=="Face Recognition").mean()*100, 1)),
        avg_confidence = ("confidence",  lambda x: round(x[x>0].mean()*100, 1) if (x>0).any() else 0),
        students       = ("student_id",  "nunique"),
        subjects       = ("subject",     "nunique"),
        total_sessions = ("status",      "count"),
    ).reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# 15. TEACHER ENGAGEMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def teacher_attendance_impact(df: pd.DataFrame, df_teachers: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate which subjects (proxied to teachers) have higher attendance.
    Teacher engagement score from teachers.csv is correlated with subject attendance.
    """
    sub_att = subject_report(df)[["subject", "pct", "dept"]].copy()
    if df_teachers.empty:
        return sub_att

    # Teachers map subjects to departments — simple dept-level join
    dept_eng = df_teachers.groupby("department")["engagement_score"].mean().reset_index()
    dept_eng.columns = ["dept", "teacher_engagement"]

    merged = sub_att.merge(dept_eng, on="dept", how="left")
    merged["teacher_engagement"] = merged["teacher_engagement"].fillna(0.80)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — standalone training run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 56)
    print("  AttendanceIQ Pro - ML Pipeline v5.0")
    print("  Deol Allwyn Samuel J B - VLSI - CIT - Afynix Digital")
    print("=" * 56)

    df_att    = load_attendance()
    df_grades = load_grades()
    df_leaves = load_medical_leaves()

    print(f"\n  Records loaded : {len(df_att):,}")
    s = attendance_summary(df_att)
    print(f"  Present        : {s['attendance_pct']}%   "
          f"Students: {s['total_students']}   Depts: {s['total_depts']}")

    results, scaler, feat, best, encoders = train_ensemble_v3(df_att)

    if not df_grades.empty:
        gm, gs, gf, gm_metrics = train_grade_predictor(df_att, df_grades)
        if gm is not None:
            print(f"\n  Grade predictor : MAE={gm_metrics['mae']}  R2={gm_metrics['r2']}")

    print(f"\n  [DONE] Pipeline complete - Best model: {best}")
    print("=" * 56)
