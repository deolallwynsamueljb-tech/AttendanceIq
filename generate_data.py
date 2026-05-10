"""
AttendanceIQ Pro · Synthetic Dataset Generator v5.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chennai Institute of Technology | Smart Attendance System
50 students · 6 departments · 3 semesters · grade correlation · festival effects

Author : Deol Allwyn Samuel J B · VLSI · CIT · Afynix Digital
Version: 5.0.0 | Generated: 2025

Outputs
-------
  data/attendance.csv       — main record (150k+ rows)
  data/students.csv         — student master with GPA, AHI
  data/grades.csv           — subject-wise marks per student
  data/teachers.csv         — faculty profiles
  data/medical_leaves.csv   — medical leave records
"""

import numpy as np
import pandas as pd
import os
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

random.seed(42)
np.random.seed(42)

DATA_DIR = "data"

# ─── SEMESTER CALENDAR ────────────────────────────────────────────────────────

SEMESTERS = [
    {
        "label": "ODD",
        "start": datetime(2023, 8,  1),
        "end":   datetime(2023, 12, 15),
        "exam_mid_start": datetime(2023, 10,  2),
        "exam_mid_end":   datetime(2023, 10, 14),
        "exam_end_start": datetime(2023, 11, 20),
        "exam_end_end":   datetime(2023, 12, 10),
    },
    {
        "label": "EVEN",
        "start": datetime(2024, 1, 15),
        "end":   datetime(2024, 5, 10),
        "exam_mid_start": datetime(2024, 3,  4),
        "exam_mid_end":   datetime(2024, 3, 16),
        "exam_end_start": datetime(2024, 4, 15),
        "exam_end_end":   datetime(2024, 5,  5),
    },
    {
        "label": "ODD",
        "start": datetime(2024, 8,  1),
        "end":   datetime(2025, 1, 25),
        "exam_mid_start": datetime(2024, 10,  7),
        "exam_mid_end":   datetime(2024, 10, 19),
        "exam_end_start": datetime(2024, 11, 25),
        "exam_end_end":   datetime(2025, 1,  18),
    },
]

SESSIONS   = ["08:00", "09:00", "10:00", "11:00", "14:00", "15:00", "16:00"]
DAYS       = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
METHODS    = ["Face Recognition", "Manual", "RFID", "QR Code"]
METHOD_W   = [0.68, 0.13, 0.11, 0.08]

# ─── TAMIL FESTIVAL / HOLIDAY CALENDAR ───────────────────────────────────────
# (month, day, name, holiday_flag, pre_day_impact, post_day_impact)
# pre/post impact: multiplier on base_pct for the day before/after
FESTIVALS = [
    (1,  14, "Pongal",                   True,  0.70, 0.80),
    (1,  15, "Thiruvalluvar Day",         True,  0.75, 0.80),
    (1,  16, "Uzhavar Thirunal",          True,  0.75, 0.85),
    (1,  26, "Republic Day",              True,  0.80, 0.85),
    (2,  14, "Valentines (half-day)",    False, 1.00, 1.00),
    (3,  17, "Holi",                     False, 0.90, 0.90),
    (4,   1, "April Fool (fun dip)",     False, 0.90, 1.00),
    (4,  14, "Tamil New Year",            True,  0.72, 0.82),
    (5,   1, "Labour Day",                True,  0.80, 0.88),
    (6,   5, "World Environment Day",    False, 0.95, 0.98),
    (8,  15, "Independence Day",          True,  0.82, 0.90),
    (9,   5, "Teachers Day",             False, 0.85, 0.92),
    (10,  2, "Gandhi Jayanti",            True,  0.80, 0.87),
    (10, 12, "Ayudha Puja",              True,  0.68, 0.78),
    (10, 13, "Vijaya Dasami",            True,  0.65, 0.82),
    (10, 31, "Diwali Eve",               False, 0.60, 0.70),
    (11,  1, "Diwali",                    True,  0.55, 0.75),
    (11,  2, "Diwali Holiday",            True,  0.72, 0.85),
    (12, 25, "Christmas",                 True,  0.75, 0.82),
    (12, 31, "New Year Eve",             False, 0.58, 0.65),
]

# ─── WEATHER EFFECTS ──────────────────────────────────────────────────────────
# Chennai monsoon (Oct-Dec) causes lower attendance on heavy rain days
# This is a random probability to simulate heavy rain on any monsoon day
MONSOON_MONTHS = {10, 11, 12}
MONSOON_RAIN_PROB = 0.18       # 18% chance of heavy rain on any monsoon day
RAIN_ATTENDANCE_MULTIPLIER = 0.82   # attendance drops 18% on heavy rain days

# ─── TEACHER PROFILES ─────────────────────────────────────────────────────────
TEACHERS = [
    ("TCH001", "Dr. Ramachandran M",   "VLSI", "Professor",         "ramachandran@cit.ac.in",   0.95),
    ("TCH002", "Dr. Kumari S",         "VLSI", "Associate Professor","kumari@cit.ac.in",          0.90),
    ("TCH003", "Prof. Selvakumar T",   "VLSI", "Assistant Professor","selvakumar@cit.ac.in",      0.78),
    ("TCH004", "Dr. Krishnamurthy P",  "CSE",  "Professor",         "krishnamurthy@cit.ac.in",  0.92),
    ("TCH005", "Dr. Anandhi V",        "CSE",  "Associate Professor","anandhi@cit.ac.in",         0.88),
    ("TCH006", "Prof. Ganesh R",       "CSE",  "Assistant Professor","ganesh@cit.ac.in",          0.72),
    ("TCH007", "Prof. Balaji N",       "CSE",  "Assistant Professor","balaji@cit.ac.in",          0.65),
    ("TCH008", "Dr. Meenakshi L",      "ECE",  "Professor",         "meenakshi@cit.ac.in",       0.91),
    ("TCH009", "Dr. Prakash K",        "ECE",  "Associate Professor","prakash@cit.ac.in",         0.85),
    ("TCH010", "Prof. Kavitha R",      "ECE",  "Assistant Professor","kavitha@cit.ac.in",         0.70),
    ("TCH011", "Dr. Subramaniam T",    "MECH", "Professor",         "subramaniam@cit.ac.in",     0.89),
    ("TCH012", "Dr. Rajkumar S",       "MECH", "Associate Professor","rajkumar@cit.ac.in",        0.82),
    ("TCH013", "Prof. Chandrasekaran","MECH",  "Assistant Professor","chandrasekaran@cit.ac.in",  0.68),
    ("TCH014", "Dr. Mythili P",        "IT",   "Professor",         "mythili@cit.ac.in",         0.93),
    ("TCH015", "Prof. Senthilkumar A", "IT",   "Associate Professor","senthilkumar@cit.ac.in",    0.80),
    ("TCH016", "Prof. Priya N",        "IT",   "Assistant Professor","priya@cit.ac.in",           0.74),
    ("TCH017", "Dr. Arumugam S",       "AIDS", "Professor",         "arumugam@cit.ac.in",        0.96),
    ("TCH018", "Dr. Saranya D",        "AIDS", "Associate Professor","saranya@cit.ac.in",         0.91),
    ("TCH019", "Prof. Vignesh K",      "AIDS", "Assistant Professor","vignesh@cit.ac.in",         0.75),
]

# ─── DEPARTMENTS & SUBJECTS ───────────────────────────────────────────────────

DEPARTMENTS = {
    "VLSI": {
        "subjects": [
            "VLSI Design", "Digital Systems", "Embedded Systems",
            "CMOS Technology", "HDL Programming", "Analog Electronics",
        ],
        "credits":     [4, 3, 4, 3, 4, 3],
        "difficulty":  [0.85, 0.70, 0.80, 0.90, 0.75, 0.65],
        "teacher_ids": ["TCH001", "TCH001", "TCH002", "TCH001", "TCH003", "TCH002"],
    },
    "CSE": {
        "subjects": [
            "Data Structures", "Operating Systems", "Machine Learning",
            "DBMS", "Computer Networks", "Algorithms", "Software Engineering",
        ],
        "credits":     [4, 3, 4, 3, 3, 4, 3],
        "difficulty":  [0.80, 0.72, 0.88, 0.65, 0.70, 0.85, 0.60],
        "teacher_ids": ["TCH004", "TCH005", "TCH004", "TCH006", "TCH005", "TCH004", "TCH007"],
    },
    "ECE": {
        "subjects": [
            "Signal Processing", "Microcontrollers", "Communication Systems",
            "VLSI Basics", "Antenna Theory", "Control Systems", "Digital Communication",
        ],
        "credits":     [4, 3, 4, 3, 3, 4, 3],
        "difficulty":  [0.88, 0.70, 0.82, 0.75, 0.78, 0.85, 0.72],
        "teacher_ids": ["TCH008", "TCH009", "TCH008", "TCH010", "TCH009", "TCH008", "TCH010"],
    },
    "MECH": {
        "subjects": [
            "Thermodynamics", "Fluid Mechanics", "Machine Design",
            "Manufacturing", "CAD/CAM", "Strength of Materials",
        ],
        "credits":     [4, 4, 3, 3, 4, 4],
        "difficulty":  [0.82, 0.80, 0.75, 0.65, 0.70, 0.85],
        "teacher_ids": ["TCH011", "TCH012", "TCH011", "TCH013", "TCH012", "TCH011"],
    },
    "IT": {
        "subjects": [
            "Web Technologies", "Cloud Computing", "Cybersecurity",
            "Big Data Analytics", "IoT Systems", "Mobile Development",
        ],
        "credits":     [3, 4, 3, 4, 3, 3],
        "difficulty":  [0.62, 0.72, 0.80, 0.78, 0.68, 0.65],
        "teacher_ids": ["TCH014", "TCH015", "TCH014", "TCH016", "TCH015", "TCH016"],
    },
    "AIDS": {
        "subjects": [
            "Deep Learning", "Computer Vision", "Natural Language Processing",
            "Robotics & Automation", "Data Analytics", "Statistical Learning",
        ],
        "credits":     [4, 4, 3, 3, 4, 3],
        "difficulty":  [0.92, 0.90, 0.88, 0.85, 0.75, 0.82],
        "teacher_ids": ["TCH017", "TCH017", "TCH018", "TCH019", "TCH018", "TCH017"],
    },
}

# ─── 50 STUDENTS ──────────────────────────────────────────────────────────────
# (id, name, dept, year, email, hostel, category, phone)

RAW_STUDENTS = [
    # VLSI — 8 students
    ("STU001","Deol Allwyn Samuel J B","VLSI",4,"deol@cit.ac.in",    False,"General", "9876543200"),
    ("STU002","Harish Venkat T",        "VLSI",4,"harish@cit.ac.in",  True, "OBC",     "9876543201"),
    ("STU003","Rahul Krishnan V",       "VLSI",4,"rahul@cit.ac.in",   False,"General", "9876543202"),
    ("STU004","Divya Ramesh T",         "VLSI",3,"divya@cit.ac.in",   True, "SC",      "9876543203"),
    ("STU005","Kavitha Mohan S",        "VLSI",3,"kavitha@cit.ac.in", False,"OBC",     "9876543204"),
    ("STU006","Suresh Kumar N",         "VLSI",2,"suresh@cit.ac.in",  True, "OBC",     "9876543205"),
    ("STU007","Annamalai P",            "VLSI",2,"annamalai@cit.ac.in",True,"SC",      "9876543206"),
    ("STU008","Revathi J",              "VLSI",1,"revathi@cit.ac.in", False,"General", "9876543207"),
    # CSE — 10 students
    ("STU009","Aravind Kumar R",        "CSE", 3,"aravind@cit.ac.in", True, "General", "9876543208"),
    ("STU010","Sowmiya Devi P",         "CSE", 3,"sowmiya@cit.ac.in", False,"OBC",     "9876543209"),
    ("STU011","Vishnu Prasad K",        "CSE", 1,"vishnu@cit.ac.in",  True, "ST",      "9876543210"),
    ("STU012","Meera Anand S",          "CSE", 2,"meera@cit.ac.in",   False,"OBC",     "9876543211"),
    ("STU013","Naveen Chandran K",      "CSE", 4,"naveen@cit.ac.in",  True, "General", "9876543212"),
    ("STU014","Balamurugan N",          "CSE", 2,"bala@cit.ac.in",    True, "SC",      "9876543213"),
    ("STU015","Saranya Prakash V",      "CSE", 1,"saranya@cit.ac.in", False,"OBC",     "9876543214"),
    ("STU016","Arjun Selvam T",         "CSE", 3,"arjun@cit.ac.in",   True, "OBC",     "9876543215"),
    ("STU017","Preethi Murali S",       "CSE", 2,"preethi@cit.ac.in", False,"General", "9876543216"),
    ("STU018","Dinesh Babu K",          "CSE", 4,"dinesh@cit.ac.in",  True, "SC",      "9876543217"),
    # ECE — 9 students
    ("STU019","Priya Nair S",           "ECE", 2,"priya@cit.ac.in",   False,"General", "9876543218"),
    ("STU020","Deepika Sundaram",       "ECE", 2,"deepika@cit.ac.in", True, "OBC",     "9876543219"),
    ("STU021","Sanjay Murugan P",       "ECE", 1,"sanjay@cit.ac.in",  True, "SC",      "9876543220"),
    ("STU022","Gopal Krishnan M",       "ECE", 2,"gopal@cit.ac.in",   True, "OBC",     "9876543221"),
    ("STU023","Renuka Devi R",          "ECE", 3,"renuka@cit.ac.in",  False,"General", "9876543222"),
    ("STU024","Senthil Kumar A",        "ECE", 4,"senthil@cit.ac.in", True, "SC",      "9876543223"),
    ("STU025","Nithya Priya V",         "ECE", 3,"nithya@cit.ac.in",  False,"OBC",     "9876543224"),
    ("STU026","Mahendran T",            "ECE", 1,"mahendran@cit.ac.in",True,"OBC",     "9876543225"),
    ("STU027","Jayashree L",            "ECE", 2,"jayashree@cit.ac.in",False,"General","9876543226"),
    # MECH — 8 students
    ("STU028","Karthik Rajan M",        "MECH",4,"karthik@cit.ac.in", True, "OBC",     "9876543227"),
    ("STU029","Anitha Selvam R",        "MECH",3,"anitha@cit.ac.in",  False,"General", "9876543228"),
    ("STU030","Vijay Shankar R",        "MECH",4,"vijay@cit.ac.in",   True, "SC",      "9876543229"),
    ("STU031","Surya Prakash K",        "MECH",2,"surya@cit.ac.in",   True, "OBC",     "9876543230"),
    ("STU032","Mohan Das T",            "MECH",3,"mohan@cit.ac.in",   True, "SC",      "9876543231"),
    ("STU033","Tamilarasan S",          "MECH",1,"tamilarasan@cit.ac.in",True,"OBC",   "9876543232"),
    ("STU034","Vignesh Kumar M",        "MECH",2,"vignesh@cit.ac.in", False,"General", "9876543233"),
    ("STU035","Pavithra R",             "MECH",1,"pavithra@cit.ac.in",False,"OBC",     "9876543234"),
    # IT — 8 students
    ("STU036","Kalai Selvi P",          "IT",  3,"kalai@cit.ac.in",   False,"OBC",     "9876543235"),
    ("STU037","Manoj Kumar R",          "IT",  2,"manoj@cit.ac.in",   True, "General", "9876543236"),
    ("STU038","Sneha Lakshmi T",        "IT",  3,"sneha@cit.ac.in",   False,"General", "9876543237"),
    ("STU039","Bharath Kumar S",        "IT",  4,"bharath@cit.ac.in", True, "SC",      "9876543238"),
    ("STU040","Keerthana V",            "IT",  2,"keerthana@cit.ac.in",False,"OBC",    "9876543239"),
    ("STU041","Abishek Mohan N",        "IT",  1,"abishek@cit.ac.in", True, "OBC",     "9876543240"),
    ("STU042","Ramya Priya K",          "IT",  4,"ramya@cit.ac.in",   False,"General", "9876543241"),
    ("STU043","Sugumar T",              "IT",  3,"sugumar@cit.ac.in", True, "SC",      "9876543242"),
    # AIDS — 7 students
    ("STU044","Nandini Krishnan R",     "AIDS",2,"nandini@cit.ac.in", False,"General", "9876543243"),
    ("STU045","Aakash Sundar V",        "AIDS",3,"aakash@cit.ac.in",  True, "OBC",     "9876543244"),
    ("STU046","Pooja Venkat S",         "AIDS",2,"pooja@cit.ac.in",   False,"General", "9876543245"),
    ("STU047","Siva Kumar M",           "AIDS",4,"siva@cit.ac.in",    True, "SC",      "9876543246"),
    ("STU048","Dharani Devi T",         "AIDS",3,"dharani@cit.ac.in", False,"OBC",     "9876543247"),
    ("STU049","Balaji Raj N",           "AIDS",1,"balaji@cit.ac.in",  True, "OBC",     "9876543248"),
    ("STU050","Ishwarya Lakshmi P",     "AIDS",2,"ishwarya@cit.ac.in",False,"General", "9876543249"),
]

# ─── STUDENT BEHAVIORAL PROFILES ──────────────────────────────────────────────
# base_pct     : probability of being present on a normal weekday
# late_rate    : probability of being late when present
# pattern      : behavioral category label
# gpa_base     : base GPA (0-10 scale), correlated with attendance
# illness_rate : probability of medical leave in any given week
# study_style  : affects exam-period behavior
# parent_alert : True if parents have been notified already this semester
# scholarship  : True if student holds merit scholarship

PROFILES = {
    "STU001":{"base_pct":0.96,"late_rate":0.03,"pattern":"consistent",  "gpa_base":9.1,"illness_rate":0.02,"study_style":"consistent",    "scholarship":True},
    "STU002":{"base_pct":0.91,"late_rate":0.06,"pattern":"good",        "gpa_base":8.5,"illness_rate":0.03,"study_style":"exam_warrior",   "scholarship":False},
    "STU003":{"base_pct":0.98,"late_rate":0.01,"pattern":"excellent",   "gpa_base":9.6,"illness_rate":0.01,"study_style":"consistent",    "scholarship":True},
    "STU004":{"base_pct":0.66,"late_rate":0.18,"pattern":"at_risk",     "gpa_base":6.1,"illness_rate":0.09,"study_style":"panicker",       "scholarship":False},
    "STU005":{"base_pct":0.88,"late_rate":0.06,"pattern":"good",        "gpa_base":8.2,"illness_rate":0.03,"study_style":"consistent",    "scholarship":False},
    "STU006":{"base_pct":0.54,"late_rate":0.23,"pattern":"critical",    "gpa_base":5.3,"illness_rate":0.11,"study_style":"panicker",       "scholarship":False},
    "STU007":{"base_pct":0.44,"late_rate":0.30,"pattern":"critical",    "gpa_base":4.7,"illness_rate":0.14,"study_style":"dropout_risk",   "scholarship":False},
    "STU008":{"base_pct":0.89,"late_rate":0.07,"pattern":"good",        "gpa_base":8.0,"illness_rate":0.04,"study_style":"consistent",    "scholarship":False},
    "STU009":{"base_pct":0.82,"late_rate":0.09,"pattern":"moderate",    "gpa_base":7.7,"illness_rate":0.05,"study_style":"moderate",       "scholarship":False},
    "STU010":{"base_pct":0.93,"late_rate":0.04,"pattern":"consistent",  "gpa_base":8.9,"illness_rate":0.02,"study_style":"consistent",    "scholarship":True},
    "STU011":{"base_pct":0.50,"late_rate":0.26,"pattern":"critical",    "gpa_base":4.9,"illness_rate":0.13,"study_style":"dropout_risk",   "scholarship":False},
    "STU012":{"base_pct":0.73,"late_rate":0.13,"pattern":"at_risk",     "gpa_base":6.6,"illness_rate":0.07,"study_style":"panicker",       "scholarship":False},
    "STU013":{"base_pct":0.79,"late_rate":0.10,"pattern":"moderate",    "gpa_base":7.3,"illness_rate":0.05,"study_style":"moderate",       "scholarship":False},
    "STU014":{"base_pct":0.47,"late_rate":0.27,"pattern":"critical",    "gpa_base":4.4,"illness_rate":0.14,"study_style":"dropout_risk",   "scholarship":False},
    "STU015":{"base_pct":0.86,"late_rate":0.07,"pattern":"good",        "gpa_base":7.9,"illness_rate":0.04,"study_style":"consistent",    "scholarship":False},
    "STU016":{"base_pct":0.80,"late_rate":0.10,"pattern":"moderate",    "gpa_base":7.4,"illness_rate":0.06,"study_style":"moderate",       "scholarship":False},
    "STU017":{"base_pct":0.95,"late_rate":0.03,"pattern":"consistent",  "gpa_base":9.0,"illness_rate":0.02,"study_style":"consistent",    "scholarship":True},
    "STU018":{"base_pct":0.60,"late_rate":0.17,"pattern":"at_risk",     "gpa_base":5.7,"illness_rate":0.09,"study_style":"panicker",       "scholarship":False},
    "STU019":{"base_pct":0.90,"late_rate":0.05,"pattern":"good",        "gpa_base":8.4,"illness_rate":0.03,"study_style":"consistent",    "scholarship":False},
    "STU020":{"base_pct":0.78,"late_rate":0.11,"pattern":"moderate",    "gpa_base":7.2,"illness_rate":0.06,"study_style":"moderate",       "scholarship":False},
    "STU021":{"base_pct":0.46,"late_rate":0.28,"pattern":"critical",    "gpa_base":4.5,"illness_rate":0.14,"study_style":"dropout_risk",   "scholarship":False},
    "STU022":{"base_pct":0.65,"late_rate":0.16,"pattern":"at_risk",     "gpa_base":5.9,"illness_rate":0.08,"study_style":"panicker",       "scholarship":False},
    "STU023":{"base_pct":0.94,"late_rate":0.04,"pattern":"consistent",  "gpa_base":8.8,"illness_rate":0.02,"study_style":"consistent",    "scholarship":True},
    "STU024":{"base_pct":0.56,"late_rate":0.21,"pattern":"critical",    "gpa_base":5.2,"illness_rate":0.11,"study_style":"dropout_risk",   "scholarship":False},
    "STU025":{"base_pct":0.87,"late_rate":0.07,"pattern":"good",        "gpa_base":8.1,"illness_rate":0.04,"study_style":"consistent",    "scholarship":False},
    "STU026":{"base_pct":0.75,"late_rate":0.12,"pattern":"at_risk",     "gpa_base":6.8,"illness_rate":0.06,"study_style":"moderate",       "scholarship":False},
    "STU027":{"base_pct":0.97,"late_rate":0.02,"pattern":"excellent",   "gpa_base":9.4,"illness_rate":0.01,"study_style":"consistent",    "scholarship":True},
    "STU028":{"base_pct":0.62,"late_rate":0.17,"pattern":"at_risk",     "gpa_base":5.8,"illness_rate":0.09,"study_style":"panicker",       "scholarship":False},
    "STU029":{"base_pct":0.85,"late_rate":0.07,"pattern":"good",        "gpa_base":7.9,"illness_rate":0.04,"study_style":"consistent",    "scholarship":False},
    "STU030":{"base_pct":0.58,"late_rate":0.20,"pattern":"critical",    "gpa_base":5.4,"illness_rate":0.10,"study_style":"dropout_risk",   "scholarship":False},
    "STU031":{"base_pct":0.84,"late_rate":0.08,"pattern":"moderate",    "gpa_base":7.6,"illness_rate":0.05,"study_style":"moderate",       "scholarship":False},
    "STU032":{"base_pct":0.52,"late_rate":0.25,"pattern":"critical",    "gpa_base":4.8,"illness_rate":0.13,"study_style":"dropout_risk",   "scholarship":False},
    "STU033":{"base_pct":0.76,"late_rate":0.12,"pattern":"moderate",    "gpa_base":6.9,"illness_rate":0.06,"study_style":"moderate",       "scholarship":False},
    "STU034":{"base_pct":0.92,"late_rate":0.05,"pattern":"good",        "gpa_base":8.6,"illness_rate":0.03,"study_style":"consistent",    "scholarship":False},
    "STU035":{"base_pct":0.88,"late_rate":0.07,"pattern":"good",        "gpa_base":8.3,"illness_rate":0.04,"study_style":"consistent",    "scholarship":False},
    "STU036":{"base_pct":0.95,"late_rate":0.03,"pattern":"consistent",  "gpa_base":9.0,"illness_rate":0.02,"study_style":"consistent",    "scholarship":True},
    "STU037":{"base_pct":0.74,"late_rate":0.13,"pattern":"at_risk",     "gpa_base":6.7,"illness_rate":0.07,"study_style":"moderate",       "scholarship":False},
    "STU038":{"base_pct":0.99,"late_rate":0.01,"pattern":"excellent",   "gpa_base":9.7,"illness_rate":0.01,"study_style":"consistent",    "scholarship":True},
    "STU039":{"base_pct":0.55,"late_rate":0.22,"pattern":"critical",    "gpa_base":5.1,"illness_rate":0.12,"study_style":"dropout_risk",   "scholarship":False},
    "STU040":{"base_pct":0.83,"late_rate":0.09,"pattern":"moderate",    "gpa_base":7.6,"illness_rate":0.05,"study_style":"moderate",       "scholarship":False},
    "STU041":{"base_pct":0.69,"late_rate":0.15,"pattern":"at_risk",     "gpa_base":6.3,"illness_rate":0.07,"study_style":"panicker",       "scholarship":False},
    "STU042":{"base_pct":0.91,"late_rate":0.05,"pattern":"good",        "gpa_base":8.6,"illness_rate":0.03,"study_style":"consistent",    "scholarship":False},
    "STU043":{"base_pct":0.48,"late_rate":0.27,"pattern":"critical",    "gpa_base":4.6,"illness_rate":0.13,"study_style":"dropout_risk",   "scholarship":False},
    "STU044":{"base_pct":0.97,"late_rate":0.02,"pattern":"excellent",   "gpa_base":9.3,"illness_rate":0.01,"study_style":"consistent",    "scholarship":True},
    "STU045":{"base_pct":0.81,"late_rate":0.10,"pattern":"moderate",    "gpa_base":7.5,"illness_rate":0.06,"study_style":"moderate",       "scholarship":False},
    "STU046":{"base_pct":0.94,"late_rate":0.04,"pattern":"consistent",  "gpa_base":8.9,"illness_rate":0.02,"study_style":"consistent",    "scholarship":True},
    "STU047":{"base_pct":0.63,"late_rate":0.16,"pattern":"at_risk",     "gpa_base":5.8,"illness_rate":0.08,"study_style":"panicker",       "scholarship":False},
    "STU048":{"base_pct":0.87,"late_rate":0.07,"pattern":"good",        "gpa_base":8.1,"illness_rate":0.04,"study_style":"consistent",    "scholarship":False},
    "STU049":{"base_pct":0.72,"late_rate":0.13,"pattern":"at_risk",     "gpa_base":6.6,"illness_rate":0.07,"study_style":"moderate",       "scholarship":False},
    "STU050":{"base_pct":0.96,"late_rate":0.03,"pattern":"consistent",  "gpa_base":9.2,"illness_rate":0.02,"study_style":"consistent",    "scholarship":True},
}

# ─── HELPER: festival lookup ───────────────────────────────────────────────────

def _build_festival_map() -> Dict[str, Tuple[bool, float, float]]:
    """Map 'YYYY-MM-DD' -> (is_holiday, pre_impact, post_impact)
    We spread across all three semester years (2023, 2024, 2025).
    """
    fmap: Dict[str, Tuple[bool, float, float]] = {}
    for year in [2023, 2024, 2025]:
        for month, day, name, is_hol, pre_imp, post_imp in FESTIVALS:
            try:
                base = datetime(year, month, day)
                fmap[base.strftime("%Y-%m-%d")]                          = (is_hol, pre_imp, post_imp, name)
                pre  = base - timedelta(days=1)
                post = base + timedelta(days=1)
                if pre.strftime("%Y-%m-%d") not in fmap:
                    fmap[pre.strftime("%Y-%m-%d")]  = (False, pre_imp, 1.0, f"Pre-{name}")
                if post.strftime("%Y-%m-%d") not in fmap:
                    fmap[post.strftime("%Y-%m-%d")] = (False, 1.0, post_imp, f"Post-{name}")
            except ValueError:
                pass
    return fmap


def _is_exam_period(dt: datetime, sem: Dict) -> str:
    """Return exam period label or empty string."""
    d = dt.date()
    if sem["exam_mid_start"].date() <= d <= sem["exam_mid_end"].date():
        return "mid_sem"
    if sem["exam_end_start"].date() <= d <= sem["exam_end_end"].date():
        return "end_sem"
    return ""


def _attendance_factor(
    sid: str,
    dt: datetime,
    day_name: str,
    sem: Dict,
    festival_map: Dict,
    rain_today: bool,
    hostel: bool,
) -> float:
    """
    Compute effective presence probability for a student on a given day.
    Layers: base profile → day-of-week → festival → exam → hostel → weather.
    """
    profile = PROFILES[sid]
    prob    = profile["base_pct"]

    # Monday/Friday dip — universal across all students
    if day_name == "Monday":
        prob *= 0.91
    elif day_name == "Friday":
        prob *= 0.93

    # Festival pre/post effects
    date_str = dt.strftime("%Y-%m-%d")
    if date_str in festival_map:
        is_hol, pre_imp, post_imp, _ = festival_map[date_str]
        if is_hol:
            return 0.0     # holiday — no attendance
        prob *= min(pre_imp, post_imp)

    # Exam week: most students show up more (fear of shortage)
    exam = _is_exam_period(dt, sem)
    if exam == "end_sem":
        prob = min(prob * 1.10, 0.99)
    elif exam == "mid_sem":
        prob = min(prob * 1.05, 0.98)

    # Chennai monsoon: heavy rain days
    if rain_today:
        # Day scholars affected more than hostel students
        multiplier = RAIN_ATTENDANCE_MULTIPLIER if not hostel else (RAIN_ATTENDANCE_MULTIPLIER * 1.05)
        prob *= multiplier

    # Hostel students have slightly higher volatility
    if hostel:
        prob *= np.random.uniform(0.96, 1.02)

    # Critical/dropout-risk students decline steadily mid-semester
    study_style = profile["study_style"]
    if study_style == "dropout_risk":
        week_of_sem = max(1, (dt - sem["start"]).days // 7)
        decline     = min(week_of_sem * 0.005, 0.10)   # up to 10% decline by week 20
        prob       *= (1.0 - decline)

    return float(np.clip(prob, 0.0, 1.0))


# ─── GRADE GENERATION ─────────────────────────────────────────────────────────

def _gen_grade(gpa_base: float, difficulty: float, attendance_pct: float) -> Dict:
    """
    Generate internal marks, external marks, and grade letter for a subject.
    Attendance has 30% weight on exam performance (real-world correlation).
    """
    att_bonus  = (attendance_pct - 0.75) * 0.3   # +/- bonus based on attendance
    adj_gpa    = np.clip(gpa_base + att_bonus * 2, 3.5, 10.0)
    difficulty_penalty = (difficulty - 0.7) * 3   # harder subjects → lower marks

    # Internal marks (out of 25)
    internal_base = (adj_gpa / 10.0) * 25
    internal      = round(np.clip(internal_base + np.random.normal(0, 1.5) - difficulty_penalty * 0.5, 5, 25), 1)

    # External marks (out of 75)
    external_base = (adj_gpa / 10.0) * 75
    external      = round(np.clip(external_base + np.random.normal(0, 4) - difficulty_penalty * 2, 15, 75), 1)

    total = internal + external

    if   total >= 90: grade, gp = "O",  10
    elif total >= 80: grade, gp = "A+",  9
    elif total >= 70: grade, gp = "A",   8
    elif total >= 60: grade, gp = "B+",  7
    elif total >= 55: grade, gp = "B",   6
    elif total >= 50: grade, gp = "C",   5
    else:             grade, gp = "U",   0   # fail

    return {
        "internal_marks": internal,
        "external_marks": external,
        "total_marks":    round(total, 1),
        "grade":          grade,
        "grade_point":    gp,
    }


# ─── MEDICAL LEAVE GENERATION ─────────────────────────────────────────────────

ILLNESS_TYPES = [
    ("Viral Fever",        3, 5),
    ("Common Cold",        1, 3),
    ("Stomach Infection",  2, 4),
    ("Dengue (suspected)", 5, 8),
    ("Eye Infection",      2, 3),
    ("Migraine",           1, 2),
    ("Typhoid",            6, 10),
    ("COVID-19 (mild)",    7, 10),
    ("Back Pain",          2, 4),
    ("Injury",             3, 7),
]


def _generate_medical_leaves(
    all_dates: List[datetime],
    sem: Dict,
) -> List[Dict]:
    """
    Generate medical leave records for all students across a semester.
    Monsoon months have higher illness rates.
    """
    leaves = []
    for sid, name, dept, year, email, hostel, category, phone in RAW_STUDENTS:
        profile      = PROFILES[sid]
        base_illness = profile["illness_rate"]
        n_weeks      = max(1, (sem["end"] - sem["start"]).days // 7)

        for week_idx in range(n_weeks):
            week_start = sem["start"] + timedelta(weeks=week_idx)
            month      = week_start.month
            # Monsoon amplifies illness
            illness_prob = base_illness * (1.5 if month in MONSOON_MONTHS else 1.0)
            if np.random.random() < illness_prob:
                illness, dur_min, dur_max = random.choice(ILLNESS_TYPES)
                duration = random.randint(dur_min, dur_max)
                leave_start = week_start + timedelta(days=random.randint(0, 4))
                if leave_start < sem["start"] or leave_start > sem["end"]:
                    continue
                leaves.append({
                    "student_id":   sid,
                    "student_name": name,
                    "department":   dept,
                    "illness":      illness,
                    "leave_start":  leave_start.strftime("%Y-%m-%d"),
                    "duration_days":duration,
                    "has_certificate": random.random() < 0.70,
                    "semester":     sem["label"],
                })
    return leaves


# ─── MAIN GENERATE FUNCTION ───────────────────────────────────────────────────

def generate():
    os.makedirs(DATA_DIR, exist_ok=True)
    festival_map = _build_festival_map()

    records        = []
    grade_records  = []
    leave_records  = []

    # Track attendance per student per subject for grade calculation
    stu_subj_att: Dict[str, Dict[str, List[int]]] = {}

    print("=" * 60)
    print("  AttendanceIQ Pro - Data Generator v5.0")
    print("  Chennai Institute of Technology")
    print("=" * 60)

    total_sem_records = 0

    for sem_idx, sem in enumerate(SEMESTERS):
        sem_label = sem["label"]
        print(f"\n  Semester {sem_idx+1} ({sem_label}): "
              f"{sem['start'].strftime('%Y-%m-%d')} to {sem['end'].strftime('%Y-%m-%d')}")

        # Collect all working days this semester
        working_days = []
        cur = sem["start"]
        while cur <= sem["end"]:
            if cur.weekday() < 5:  # Mon-Fri only
                working_days.append(cur)
            cur += timedelta(days=1)

        # Generate medical leaves for this semester
        sem_leaves = _generate_medical_leaves(working_days, sem)
        leave_records.extend(sem_leaves)

        # Build per-student leave date sets for fast lookup
        stu_leave_dates: Dict[str, set] = {}
        for lv in sem_leaves:
            sid = lv["student_id"]
            ls  = datetime.strptime(lv["leave_start"], "%Y-%m-%d")
            for d in range(lv["duration_days"]):
                ld = ls + timedelta(days=d)
                stu_leave_dates.setdefault(sid, set()).add(ld.strftime("%Y-%m-%d"))

        sem_count = 0
        for day_dt in working_days:
            day_name = DAYS[day_dt.weekday()]
            date_str = day_dt.strftime("%Y-%m-%d")

            # Monsoon rain simulation
            rain_today = (
                day_dt.month in MONSOON_MONTHS
                and np.random.random() < MONSOON_RAIN_PROB
            )

            # Holiday check
            if date_str in festival_map and festival_map[date_str][0]:
                continue  # institution is closed

            exam_period = _is_exam_period(day_dt, sem)

            for sid, name, dept, year, email, hostel, category, phone in RAW_STUDENTS:
                profile  = PROFILES[sid]
                subjects = DEPARTMENTS[dept]["subjects"]
                n_sess   = random.randint(2, 4)
                sess_times = random.sample(SESSIONS, n_sess)
                on_leave   = date_str in stu_leave_dates.get(sid, set())

                prob = _attendance_factor(
                    sid, day_dt, day_name, sem, festival_map, rain_today, hostel
                )

                # Medical leave overrides attendance
                if on_leave:
                    prob = 0.0

                for sess in sess_times:
                    subj    = random.choice(subjects)
                    present = bool(np.random.random() < prob)
                    method  = (np.random.choice(METHODS, p=METHOD_W) if present else "Absent")
                    conf    = (round(np.random.uniform(0.82, 0.99), 3)
                               if method == "Face Recognition" else 0.0)
                    late    = (bool(np.random.random() < profile["late_rate"]) if present else False)
                    late_m  = random.randint(1, 25) if late else 0

                    # Attention score: proxy for quality of presence
                    attention_score = 0.0
                    if present:
                        base_att = 0.6 + (profile["gpa_base"] / 10.0) * 0.3
                        if late:
                            base_att -= 0.10
                        if exam_period:
                            base_att += 0.05
                        if day_name in ["Monday", "Friday"]:
                            base_att -= 0.05
                        attention_score = round(np.clip(base_att + np.random.normal(0, 0.06), 0.3, 1.0), 3)

                    # Track for grade computation
                    key = f"{sid}|{subj}"
                    stu_subj_att.setdefault(key, {"total": 0, "present": 0})
                    stu_subj_att[key]["total"]   += 1
                    stu_subj_att[key]["present"] += int(present)

                    records.append({
                        "student_id":      sid,
                        "student_name":    name,
                        "department":      dept,
                        "year":            year,
                        "email":           email,
                        "hostel":          hostel,
                        "category":        category,
                        "subject":         subj,
                        "date":            date_str,
                        "day":             day_name,
                        "session_time":    sess,
                        "status":          "Present" if present else "Absent",
                        "method":          method,
                        "confidence":      conf,
                        "late":            late,
                        "late_minutes":    late_m,
                        "pattern":         profile["pattern"],
                        "semester":        sem_label,
                        "exam_period":     exam_period,
                        "rain_day":        rain_today,
                        "on_medical_leave":on_leave,
                        "attention_score": attention_score,
                    })
                    sem_count += 1

        total_sem_records += sem_count
        print(f"    >> {sem_count:,} attendance records generated")

    # ─── GRADES ───────────────────────────────────────────────────────────────
    for sem_idx, sem in enumerate(SEMESTERS):
        for sid, name, dept, year, email, hostel, category, phone in RAW_STUDENTS:
            profile  = PROFILES[sid]
            subjects = DEPARTMENTS[dept]["subjects"]
            credits  = DEPARTMENTS[dept]["credits"]
            diffs    = DEPARTMENTS[dept]["difficulty"]

            total_credits   = 0
            weighted_gp_sum = 0

            for subj, cred, diff in zip(subjects, credits, diffs):
                key       = f"{sid}|{subj}"
                att_data  = stu_subj_att.get(key, {"total": 1, "present": 1})
                att_pct   = att_data["present"] / max(att_data["total"], 1)
                g         = _gen_grade(profile["gpa_base"], diff, att_pct)

                # Eligibility: need ≥75% attendance to write exam
                eligible  = att_pct >= 0.75
                if not eligible:
                    g["grade"]       = "DETAINED"
                    g["grade_point"] = 0

                grade_records.append({
                    "student_id":      sid,
                    "student_name":    name,
                    "department":      dept,
                    "year":            year,
                    "subject":         subj,
                    "credits":         cred,
                    "difficulty":      round(diff, 2),
                    "attendance_pct":  round(att_pct * 100, 1),
                    "eligible":        eligible,
                    "internal_marks":  g["internal_marks"],
                    "external_marks":  g["external_marks"] if eligible else 0,
                    "total_marks":     g["total_marks"] if eligible else 0,
                    "grade":           g["grade"] if eligible else "DETAINED",
                    "grade_point":     g["grade_point"] if eligible else 0,
                    "semester":        sem["label"],
                    "semester_idx":    sem_idx + 1,
                })

                if eligible:
                    total_credits   += cred
                    weighted_gp_sum += cred * g["grade_point"]

    # ─── STUDENTS CSV ──────────────────────────────────────────────────────────
    att_df = pd.DataFrame(records)

    student_rows = []
    for sid, name, dept, year, email, hostel, category, phone in RAW_STUDENTS:
        profile   = PROFILES[sid]
        stu_att   = att_df[att_df["student_id"] == sid]
        total_rec = len(stu_att)
        present_c = (stu_att["status"] == "Present").sum()
        att_pct   = round(present_c / max(total_rec, 1) * 100, 1)

        # Academic Health Index (AHI): composite of attendance, GPA, streak, lates
        late_rate_act = (stu_att["late"].sum() / max(total_rec, 1)) * 100
        ahi = round(
            0.45 * att_pct
            + 0.35 * (profile["gpa_base"] * 10)
            + 0.10 * max(0, (10 - late_rate_act))
            + 0.10 * (100 if profile["scholarship"] else 60),
            1
        )

        student_rows.append({
            "id":           sid,
            "name":         name,
            "dept":         dept,
            "year":         year,
            "email":        email,
            "phone":        phone,
            "hostel":       hostel,
            "category":     category,
            "pattern":      profile["pattern"],
            "gpa_base":     profile["gpa_base"],
            "scholarship":  profile["scholarship"],
            "study_style":  profile["study_style"],
            "attendance_pct": att_pct,
            "ahi":          ahi,
        })

    # ─── WRITE CSVs ────────────────────────────────────────────────────────────
    att_df.to_csv(os.path.join(DATA_DIR, "attendance.csv"), index=False)

    stu_df = pd.DataFrame(student_rows)
    stu_df.to_csv(os.path.join(DATA_DIR, "students.csv"), index=False)

    grades_df = pd.DataFrame(grade_records)
    grades_df.to_csv(os.path.join(DATA_DIR, "grades.csv"), index=False)

    leaves_df = pd.DataFrame(leave_records)
    leaves_df.to_csv(os.path.join(DATA_DIR, "medical_leaves.csv"), index=False)

    tch_df = pd.DataFrame([
        {"teacher_id":t[0],"name":t[1],"department":t[2],"designation":t[3],
         "email":t[4],"engagement_score":t[5]}
        for t in TEACHERS
    ])
    tch_df.to_csv(os.path.join(DATA_DIR, "teachers.csv"), index=False)

    # Summary stats
    print(f"\n  [OK] Total attendance records : {len(att_df):,}")
    print(f"  [OK] Students                 : {att_df['student_id'].nunique()}")
    print(f"  [OK] Departments              : {att_df['department'].nunique()}")
    print(f"  [OK] Date range               : "
          f"{att_df['date'].min()} to {att_df['date'].max()}")
    print(f"  [OK] Unique days              : {att_df['date'].nunique()}")
    present_pct = (att_df["status"] == "Present").mean() * 100
    print(f"  [OK] Overall present          : {present_pct:.1f}%")
    print(f"  [OK] Grade records            : {len(grades_df):,}")
    print(f"  [OK] Medical leave episodes   : {len(leaves_df):,}")
    print(f"  [OK] Teachers                 : {len(tch_df)}")
    print(f"\n  Saved to -> {DATA_DIR}/")
    print("=" * 60)

    return att_df


if __name__ == "__main__":
    generate()
