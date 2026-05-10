"""
AttendanceIQ Pro · Synthetic Data Generator v6.0
Author: Deol Allwyn Samuel J B · VLSI · CIT · Afynix Digital
Generates 5000+ realistic attendance records for 20 students over 6 months.
"""
import numpy as np
import pandas as pd
import os
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEPARTMENTS = ["CSE", "ECE", "MECH", "CIVIL", "IT", "VLSI"]

DEPT_SUBJECTS = {
    "CSE":   ["Data Structures", "DBMS", "Operating Systems", "Computer Networks"],
    "ECE":   ["Signals & Systems", "Digital Electronics", "VLSI Design", "Microprocessors"],
    "MECH":  ["Thermodynamics", "Fluid Mechanics", "Machine Design", "Manufacturing"],
    "CIVIL": ["Structural Analysis", "Surveying", "Concrete Technology", "Env Engineering"],
    "IT":    ["Web Technologies", "Cloud Computing", "Data Mining", "Software Engineering"],
    "VLSI":  ["CMOS VLSI", "Physical Design", "Verification", "Embedded Systems"],
}

STUDENTS = [
    {"name": "Deol Allwyn Samuel J B", "dept": "VLSI",  "year": 3, "hostel": True,  "base": 0.95},
    {"name": "Priya Sharma",            "dept": "CSE",   "year": 2, "hostel": False, "base": 0.92},
    {"name": "Arjun Mehta",             "dept": "ECE",   "year": 3, "hostel": True,  "base": 0.78},
    {"name": "Kavya Reddy",             "dept": "IT",    "year": 2, "hostel": False, "base": 0.88},
    {"name": "Rohit Verma",             "dept": "MECH",  "year": 3, "hostel": True,  "base": 0.62},
    {"name": "Sneha Pillai",            "dept": "CIVIL", "year": 2, "hostel": False, "base": 0.91},
    {"name": "Aditya Kumar",            "dept": "CSE",   "year": 3, "hostel": True,  "base": 0.84},
    {"name": "Meera Nair",              "dept": "VLSI",  "year": 2, "hostel": False, "base": 0.93},
    {"name": "Rahul Singh",             "dept": "ECE",   "year": 3, "hostel": True,  "base": 0.70},
    {"name": "Pooja Iyer",              "dept": "IT",    "year": 2, "hostel": False, "base": 0.87},
    {"name": "Karthik Rajan",           "dept": "MECH",  "year": 3, "hostel": True,  "base": 0.58},
    {"name": "Divya Krishnan",          "dept": "CIVIL", "year": 2, "hostel": False, "base": 0.89},
    {"name": "Vikram Patel",            "dept": "CSE",   "year": 3, "hostel": True,  "base": 0.76},
    {"name": "Ananya Bose",             "dept": "VLSI",  "year": 2, "hostel": False, "base": 0.96},
    {"name": "Suresh Babu",             "dept": "ECE",   "year": 3, "hostel": True,  "base": 0.82},
    {"name": "Lakshmi Devi",            "dept": "IT",    "year": 2, "hostel": False, "base": 0.90},
    {"name": "Nikhil Joshi",            "dept": "MECH",  "year": 3, "hostel": True,  "base": 0.67},
    {"name": "Ranjini Murugan",         "dept": "CIVIL", "year": 2, "hostel": False, "base": 0.85},
    {"name": "Ashwin Chandran",         "dept": "CSE",   "year": 3, "hostel": True,  "base": 0.79},
    {"name": "Harini Venkatesh",        "dept": "VLSI",  "year": 2, "hostel": False, "base": 0.94},
]

FESTIVALS = [
    "2025-01-14", "2025-01-26", "2025-02-26",
    "2025-04-14", "2025-04-18", "2025-05-01",
]

TIME_SLOTS = ["08:30", "09:30", "10:30", "11:30", "14:00", "15:00"]


def get_working_days():
    start = pd.Timestamp("2025-01-06")
    end   = pd.Timestamp("2025-06-28")
    bdays = pd.date_range(start, end, freq="B")
    sats  = pd.date_range(start, end, freq="W-SAT")
    fest  = [pd.Timestamp(d) for d in FESTIVALS]
    all_days = sorted(set(bdays.tolist() + sats.tolist()))
    return [d for d in all_days if d not in fest]


def generate():
    working_days = get_working_days()
    records = []

    for idx, stu in enumerate(STUDENTS):
        sid  = f"CIT2024{idx+1:03d}"
        dept = stu["dept"]
        subjects = DEPT_SUBJECTS[dept]
        base = stu["base"]

        for date in working_days:
            is_sat = date.dayofweek == 5
            n_subj = 2 if is_sat else 4
            day_subjects = random.sample(subjects, min(n_subj, len(subjects)))

            for subj in day_subjects:
                dow_effect   = {0: -0.05, 1: 0.01, 2: 0.02, 3: 0.01, 4: -0.03, 5: -0.07}.get(date.dayofweek, 0)
                month_effect = 0.04 if date.month in [5, 6] else 0
                noise        = float(np.random.normal(0, 0.04))
                prob         = min(0.99, max(0.05, base + dow_effect + month_effect + noise))
                present      = random.random() < prob

                if present:
                    method     = random.choices(["Face", "Manual", "RFID"], weights=[65, 25, 10])[0]
                    confidence = round(random.uniform(78, 99), 1) if method == "Face" else 0.0
                    late       = random.random() < 0.12
                else:
                    method     = "Absent"
                    confidence = 0.0
                    late       = False

                records.append({
                    "student_id":  sid,
                    "name":        stu["name"],
                    "department":  dept,
                    "year":        stu["year"],
                    "hostel":      stu["hostel"],
                    "subject":     subj,
                    "date":        date.strftime("%Y-%m-%d"),
                    "hour":        random.choice(TIME_SLOTS[:2] if is_sat else TIME_SLOTS[:4]),
                    "day_of_week": date.strftime("%A"),
                    "month":       date.month,
                    "semester":    1 if date.month <= 3 else 2,
                    "status":      "Present" if present else "Absent",
                    "method":      method,
                    "confidence":  confidence,
                    "late":        late,
                    "is_saturday": is_sat,
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DATA_DIR, "attendance.csv"), index=False)

    students_df = pd.DataFrame([{
        "student_id": f"CIT2024{i+1:03d}",
        "name":       s["name"],
        "department": s["dept"],
        "year":       s["year"],
        "hostel":     s["hostel"],
    } for i, s in enumerate(STUDENTS)])
    students_df.to_csv(os.path.join(DATA_DIR, "students.csv"), index=False)

    print(f"Generated {len(df):,} attendance records · {len(students_df)} students")
    return df


if __name__ == "__main__":
    generate()
