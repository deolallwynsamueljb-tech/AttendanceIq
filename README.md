# AttendanceIq — AI-Powered Attendance Intelligence Platform

A Streamlit-based ML application that tracks, analyses, and predicts student/employee attendance patterns using machine learning.

## Features

- **Attendance Tracking** — Upload or generate attendance data for analysis
- **ML Predictions** — Predict absenteeism risk using scikit-learn classification models
- **Imbalance Handling** — SMOTE and imbalanced-learn techniques for accurate minority-class prediction
- **Visual Analytics** — Interactive Plotly charts: attendance trends, risk distribution, heatmaps
- **Data Generation** — Built-in synthetic data generator for testing and demos

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| ML | scikit-learn, imbalanced-learn |
| Data | pandas, numpy, scipy |
| Visualisation | Plotly, matplotlib |
| Model Persistence | joblib |

## Project Structure

```
AttendanceIq/
├── app.py              # Main Streamlit entry point
├── streamlit_app.py    # Streamlit UI components
├── ml_pipeline.py      # ML training, evaluation, prediction pipeline
├── generate_data.py    # Synthetic attendance data generator
├── data/               # Data files
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Upload attendance CSV or use the built-in data generator
2. Run ML pipeline to train absenteeism risk model
3. View predictions and analytics dashboard
4. Export risk reports

## License

MIT
