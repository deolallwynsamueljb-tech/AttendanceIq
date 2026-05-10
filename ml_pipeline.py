"""
AttendanceIQ Pro · ML Pipeline v6.0
Author: Deol Allwyn Samuel J B · VLSI · CIT · Afynix Digital
"""
import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics         import (accuracy_score, f1_score, roc_auc_score,
                                     confusion_matrix, precision_score, recall_score, roc_curve)
from sklearn.cluster         import KMeans

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

VERSION      = "6.0.0"
RANDOM_STATE = 42
BASE         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE, "data")
MODEL_DIR    = os.path.join(BASE, "models")
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_attendance(path: str = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, "attendance.csv")
    if not os.path.exists(path):
        from generate_data import generate
        generate()
    df = pd.read_csv(path, parse_dates=["date"])
    df["late"]    = df["late"].astype(bool)
    df["hostel"]  = df["hostel"].astype(bool)
    df["present"] = df["status"] == "Present"
    return df


# ── ANALYTICS ─────────────────────────────────────────────────────────────────

def student_report(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["student_id", "name", "department", "year"])
    r = grp.agg(total=("present","count"), present=("present","sum")).reset_index()
    r["pct"]       = (r["present"] / r["total"] * 100).round(1)
    r["absent"]    = r["total"] - r["present"]
    late_c = df[df["late"]].groupby("student_id").size().reset_index(name="late")
    r = r.merge(late_c, on="student_id", how="left").fillna({"late": 0})
    r["late"]      = r["late"].astype(int)
    r["late_rate"] = (r["late"] / r["total"] * 100).round(1)
    r["risk_score"]= ((100 - r["pct"]) * 0.6 + r["late_rate"] * 0.4).round(1)
    r["dropout_risk"] = pd.cut(r["pct"], bins=[0,60,75,85,101],
                                labels=["Critical","High","Medium","Low"])
    r["ahi"]       = (r["pct"] * 0.7 + (100 - r["late_rate"]) * 0.3).round(1)
    return r.sort_values("pct")


def attendance_summary(df: pd.DataFrame) -> dict:
    total   = len(df)
    present = int(df["present"].sum())
    face_df = df[df["method"] == "Face"]
    s_rpt   = student_report(df)
    return {
        "total_records":   total,
        "total_students":  df["student_id"].nunique(),
        "total_depts":     df["department"].nunique(),
        "total_subjects":  df["subject"].nunique(),
        "total_days":      df["date"].nunique(),
        "attendance_pct":  round(present / total * 100, 1),
        "face_recog_pct":  round(len(face_df) / total * 100, 1),
        "avg_confidence":  round(face_df["confidence"].mean(), 1) if len(face_df) else 0,
        "at_risk_count":   int((s_rpt["pct"] < 75).sum()),
    }


def subject_report(df: pd.DataFrame) -> pd.DataFrame:
    r = df.groupby("subject").agg(total=("present","count"), present=("present","sum")).reset_index()
    r["pct"]    = (r["present"] / r["total"] * 100).round(1)
    r["absent"] = r["total"] - r["present"]
    return r.sort_values("pct")


def dept_report(df: pd.DataFrame) -> pd.DataFrame:
    r = df.groupby("department").agg(
        total=("present","count"), present=("present","sum"),
        students=("student_id","nunique")).reset_index()
    r["pct"] = (r["present"] / r["total"] * 100).round(1)
    return r.sort_values("pct", ascending=False)


def daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby("date").agg(total=("present","count"), present=("present","sum")).reset_index()
    t["pct"]       = (t["present"] / t["total"] * 100).round(1)
    t["rolling_7"] = t["pct"].rolling(7, min_periods=1).mean().round(1)
    return t


def hourly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    r = df.groupby("hour").agg(total=("present","count"), present=("present","sum")).reset_index()
    r["pct"] = (r["present"] / r["total"] * 100).round(1)
    return r.sort_values("hour")


def at_risk_students(df: pd.DataFrame) -> pd.DataFrame:
    return student_report(df)


def streak_analysis(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for sid, grp in df.groupby("student_id"):
        name = grp["name"].iloc[0]
        dept = grp["department"].iloc[0]
        daily = grp.groupby("date")["present"].any().reset_index().sort_values("date")
        max_s = cur = 0
        for _, row in daily.iterrows():
            cur   = cur + 1 if row["present"] else 0
            max_s = max(max_s, cur)
        results.append({"student_id": sid, "name": name, "department": dept, "max_streak": max_s})
    return pd.DataFrame(results).sort_values("max_streak", ascending=False)


def weekly_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["week"] = d["date"].dt.isocalendar().week.astype(int)
    d["dow"]  = d["date"].dt.day_name()
    pivot = d.groupby(["week", "dow"])["present"].mean().unstack(fill_value=0) * 100
    return pivot.round(1)


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month_name"] = d["date"].dt.strftime("%b %Y")
    d["month_num"]  = d["date"].dt.month
    r = d.groupby(["month_num", "month_name"]).agg(
        total=("present","count"), present=("present","sum")).reset_index()
    r["pct"] = (r["present"] / r["total"] * 100).round(1)
    return r.sort_values("month_num")


def semester_comparison(df: pd.DataFrame) -> pd.DataFrame:
    r = df.groupby("semester").agg(
        total=("present","count"), present=("present","sum"),
        days=("date","nunique")).reset_index()
    r["pct"] = (r["present"] / r["total"] * 100).round(1)
    return r


def department_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    return dept_report(df)


def compute_engagement_scores(df: pd.DataFrame) -> pd.DataFrame:
    r = student_report(df)
    r["engagement"] = (r["pct"] * 0.6 + (100 - r["late_rate"]) * 0.4).round(1)
    return r


def generate_smart_alerts(df, leaves=None, anom=None) -> pd.DataFrame:
    s = student_report(df)
    alerts = []
    for _, row in s.iterrows():
        if row["pct"] < 60:
            alerts.append({"student_id": row["student_id"], "student_name": row["name"],
                           "department": row["department"], "priority": 1,
                           "priority_label": "CRITICAL", "alert_type": "attendance_critical",
                           "message": f"Attendance {row['pct']}% — immediate action needed",
                           "action": "Issue warning + HOD meeting"})
        elif row["pct"] < 75:
            alerts.append({"student_id": row["student_id"], "student_name": row["name"],
                           "department": row["department"], "priority": 2,
                           "priority_label": "HIGH", "alert_type": "attendance_low",
                           "message": f"Attendance {row['pct']}% — below threshold",
                           "action": "Assign mentor + SMS alert"})
    return pd.DataFrame(alerts) if alerts else pd.DataFrame()


def cluster_students(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    s = student_report(df)
    X = s[["pct", "late_rate", "risk_score"]].fillna(0).values
    if len(s) < n:
        s["cluster_label"] = "All Students"
        return s
    km = KMeans(n_clusters=n, random_state=RANDOM_STATE, n_init=10)
    s["cluster"] = km.fit_predict(X)
    avg = s.groupby("cluster")["pct"].mean().sort_values(ascending=False)
    labels = ["High Performers", "Average Performers", "At-Risk Students"]
    label_map = {cid: labels[i] for i, cid in enumerate(avg.index)}
    s["cluster_label"] = s["cluster"].map(label_map)
    return s


def grade_analysis(df, grades=None):
    return student_report(df)


def teacher_attendance_impact(df, teachers=None):
    return subject_report(df)


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], dict]:
    d = df.copy()
    grp = d.groupby("student_id")

    feat = grp.agg(
        total_classes=("present","count"),
        total_present=("present","sum"),
        total_late=("late","sum"),
        avg_confidence=("confidence","mean"),
        n_subjects=("subject","nunique"),
        n_days=("date","nunique"),
        hostel=("hostel","first"),
        year=("year","first"),
    ).reset_index()

    feat["dept"]         = d.groupby("student_id")["department"].first().values
    feat["overall_pct"]  = (feat["total_present"] / feat["total_classes"] * 100).round(2)
    feat["late_rate"]    = (feat["total_late"] / feat["total_classes"] * 100).round(2)
    feat["absence_rate"] = 100 - feat["overall_pct"]

    # Semester split
    sem = d.groupby(["student_id","semester"])["present"].mean().unstack(fill_value=0) * 100
    for s_id, col_name in [(1,"sem1_pct"), (2,"sem2_pct")]:
        if s_id in sem.columns:
            feat = feat.merge(sem[[s_id]].rename(columns={s_id: col_name}), on="student_id", how="left")
        else:
            feat[col_name] = feat["overall_pct"]
    feat["sem_trend"] = feat["sem2_pct"] - feat["sem1_pct"]

    # Day effects
    dow = d.groupby(["student_id","day_of_week"])["present"].mean().unstack(fill_value=0) * 100
    for day, col in [("Monday","pct_monday"), ("Friday","pct_friday")]:
        if day in dow.columns:
            feat = feat.merge(dow[[day]].rename(columns={day: col}), on="student_id", how="left")
        else:
            feat[col] = feat["overall_pct"]

    # Method ratios
    mth = d.groupby(["student_id","method"]).size().unstack(fill_value=0)
    for m in ["Face","Manual","RFID"]:
        feat[f"n_{m.lower()}"] = mth[m].values if m in mth.columns else 0
    feat["face_ratio"] = (feat["n_face"] / (feat["total_classes"] + 1) * 100).round(2)

    # Encode
    encoders = {}
    for col in ["dept","hostel"]:
        le = LabelEncoder()
        feat[f"{col}_enc"] = le.fit_transform(feat[col].astype(str))
        encoders[col] = le

    FEATURES = [
        "overall_pct","late_rate","absence_rate","avg_confidence",
        "sem1_pct","sem2_pct","sem_trend",
        "pct_monday","pct_friday","face_ratio",
        "n_face","n_manual","total_classes",
        "n_subjects","n_days","year","dept_enc","hostel_enc",
    ]
    FEATURES = [f for f in FEATURES if f in feat.columns]
    feat["target"] = (feat["overall_pct"] < 75).astype(int)
    feat = feat.fillna(0)
    return feat, FEATURES, encoders


# ── ML TRAINING ───────────────────────────────────────────────────────────────

def train_absence_predictor_v2(df: pd.DataFrame, fast_mode: bool = False):
    feat, FEATURES, encoders = engineer_features(df)
    X = feat[FEATURES].values
    y = feat["target"].values

    if len(X) < 6:
        dummy = {"accuracy": 0, "f1": 0, "auc": 0.5, "precision": 0, "recall": 0,
                 "cv_f1_mean": 0, "cv_f1_std": 0, "cm": [[0,0],[0,0]], "fpr": [0,1], "tpr": [0,1], "fi": {}, "model": None}
        return {"Random Forest": dummy}, StandardScaler(), FEATURES, "Random Forest", encoders

    stratify = y if y.sum() >= 2 and (len(y) - y.sum()) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                               random_state=RANDOM_STATE, stratify=stratify)

    # SMOTE — fix F1=0 caused by class imbalance
    if HAS_SMOTE and y_tr.sum() >= 2 and (len(y_tr) - y_tr.sum()) >= 2:
        try:
            k = min(3, int(y_tr.sum()) - 1, int(len(y_tr) - y_tr.sum()) - 1)
            if k >= 1:
                sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        except Exception:
            pass

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    n_splits = min(5, max(2, int(y_tr.sum()), int(len(y_tr) - y_tr.sum())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    models_def = {
        "Random Forest": RandomForestClassifier(
            n_estimators=50 if fast_mode else 120,
            max_depth=7, min_samples_leaf=2, random_state=RANDOM_STATE),
        "Gradient Boost": GradientBoostingClassifier(
            n_estimators=50 if fast_mode else 100,
            max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE),
        "Logistic Reg": LogisticRegression(
            max_iter=300, C=1.0, random_state=RANDOM_STATE),
    }

    results  = {}
    best_f1  = -1
    best_name = "Random Forest"

    for name, mdl in models_def.items():
        try:
            mdl.fit(X_tr_sc, y_tr)
            y_pred = mdl.predict(X_te_sc)
            y_prob = mdl.predict_proba(X_te_sc)[:,1] if hasattr(mdl, "predict_proba") else y_pred.astype(float)

            acc  = round(accuracy_score(y_te, y_pred), 3)
            f1   = round(f1_score(y_te, y_pred, zero_division=0), 3)
            prec = round(precision_score(y_te, y_pred, zero_division=0), 3)
            rec  = round(recall_score(y_te, y_pred, zero_division=0), 3)
            try:
                auc = round(roc_auc_score(y_te, y_prob), 3)
                fpr, tpr, _ = roc_curve(y_te, y_prob)
            except Exception:
                auc, fpr, tpr = 0.5, [0, 1], [0, 1]

            cv_sc = cross_val_score(mdl, X_tr_sc, y_tr, cv=cv, scoring="f1", error_score=0)
            cm    = confusion_matrix(y_te, y_pred).tolist()
            fi    = dict(zip(FEATURES, mdl.feature_importances_.round(4))) if hasattr(mdl, "feature_importances_") else {}

            results[name] = {
                "model": mdl, "accuracy": acc, "f1": f1, "precision": prec,
                "recall": rec, "auc": auc, "cv_f1_mean": round(cv_sc.mean(), 3),
                "cv_f1_std": round(cv_sc.std(), 3), "cm": cm,
                "fpr": list(fpr), "tpr": list(tpr), "fi": fi,
            }
            if f1 > best_f1:
                best_f1, best_name = f1, name
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")

    return results, scaler, FEATURES, best_name, encoders


# ── ANOMALY DETECTION ─────────────────────────────────────────────────────────

def anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    feat, FEATURES, _ = engineer_features(df)
    X = feat[FEATURES].fillna(0).values
    if len(X) < 3:
        feat["anomaly"] = False
        feat["score"]   = 0.0
        return feat[["student_id","overall_pct","anomaly","score"]]
    iso = IsolationForest(contamination=0.15, random_state=RANDOM_STATE)
    feat["_pred"]   = iso.fit_predict(X)
    feat["score"]   = iso.score_samples(X).round(4)
    feat["anomaly"] = feat["_pred"] == -1
    s = student_report(df)[["student_id","name","department","pct","risk_score"]]
    r = feat[["student_id","anomaly","score"]].merge(s, on="student_id", how="left")
    r = r.rename(columns={"pct": "overall_pct"})
    return r


# ── FORECAST ──────────────────────────────────────────────────────────────────

def forecast_attendance(df: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    trend = daily_trend(df)
    if len(trend) < 7:
        return pd.DataFrame()
    recent  = trend["pct"].iloc[-14:].values
    rolling = float(np.mean(recent))
    slope   = float((recent[-1] - recent[0]) / max(len(recent) - 1, 1)) * 0.08
    last    = pd.to_datetime(trend["date"].max())
    future  = pd.date_range(last + timedelta(days=1), periods=days, freq="B")
    rows    = []
    for i, d in enumerate(future):
        pred = min(99, max(50, rolling + slope * i + float(np.random.normal(0, 1.2))))
        rows.append({"date": d.strftime("%Y-%m-%d"),
                     "forecast": round(pred, 1),
                     "lower":    round(max(50, pred - 4.5), 1),
                     "upper":    round(min(99, pred + 4.5), 1)})
    return pd.DataFrame(rows)


# ── FACE RECOGNITION SIMULATION ───────────────────────────────────────────────

def simulate_face_recognition(student_name: str = None) -> dict:
    attempts = [{"attempt": i+1, "confidence": round(random.uniform(68, 99), 1)} for i in range(3)]
    for a in attempts:
        a["matched"] = a["confidence"] > 80
    best = max(attempts, key=lambda x: x["confidence"])
    return {"matched": best["matched"], "confidence": best["confidence"],
            "attempts": attempts, "student": student_name or "Unknown",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def mark_attendance(name: str, subject: str, csv_path: str = None) -> bool:
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "live_attendance.csv")
    now = datetime.now()
    try:
        live = pd.read_csv(csv_path)
        if ((live["Name"] == name) & (live["Date"] == now.strftime("%Y-%m-%d")) &
                (live["Subject"] == subject)).any():
            return False
    except (FileNotFoundError, pd.errors.EmptyDataError):
        live = pd.DataFrame(columns=["Name","Subject","Time","Date","Confidence"])
    row = pd.DataFrame([{"Name": name, "Subject": subject,
                         "Time": now.strftime("%H:%M:%S"),
                         "Date": now.strftime("%Y-%m-%d"),
                         "Confidence": round(random.uniform(82, 98), 1)}])
    pd.concat([live, row], ignore_index=True).to_csv(csv_path, index=False)
    return True
