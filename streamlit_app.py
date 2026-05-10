"""
AttendanceIQ Pro · Streamlit Cloud-Optimized Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This wrapper ensures Streamlit Cloud deployment compatibility.
Redirects to app.py with cloud-aware configuration.
"""

import os
import sys

# ── Cloud Detection ───────────────────────────────────────────────
IS_CLOUD = "STREAMLIT_SERVER_RUNONSAVE" in os.environ or not any(
    os.path.exists(p) for p in ["/Users", "C:/Users", "/home/user"]
)

# ── Set persistent data directory ─────────────────────────────────
if IS_CLOUD:
    DATA_DIR = os.path.expanduser("~/.cache/attendanceiq_data")
    MODEL_DIR = os.path.expanduser("~/.cache/attendanceiq_models")
else:
    DATA_DIR = "data"
    MODEL_DIR = "models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Set environment for app.py ────────────────────────────────────
os.environ["ATTENDANCEIQ_DATA_DIR"] = DATA_DIR
os.environ["ATTENDANCEIQ_MODEL_DIR"] = MODEL_DIR
os.environ["ATTENDANCEIQ_IS_CLOUD"] = str(IS_CLOUD)
os.environ["ATTENDANCEIQ_FAST_MODE"] = str(IS_CLOUD)

# ── Import and run main app ───────────────────────────────────────
if __name__ == "__main__":
    import app
