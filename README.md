# 🌬️ Air Quality Prediction & Forecasting System

**Project:** Hybrid AI System for Real-Time and Future Air Quality Monitoring  
**Author:** Preetham Siddapura Umesh
**LinkedIn:** [https://www.linkedin.com/in/preethamsu/]

**Mail-Id:**[preetham.umesh2003@gmail.com]
**Domain:** Machine Learning / Deep Learning / Data Science / Environmental Analytics  
**Tools:** Python, Streamlit, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Plotly  

---

## 📖 Project Overview

Air pollution is a global health challenge. This project implements a **hybrid AI system** to monitor air quality in real-time and forecast future air quality.  

- **Random Forest:** Predicts **current AQI** quickly and accurately.  
- **LSTM:** Forecasts **next 24 hours AQI**, capturing temporal trends and patterns.  

This dual-model approach ensures **fast predictions + deep temporal forecasting**, making it suitable for **real-time dashboards and decision support systems**.

---

## 🎯 Objectives

1. Predict **current AQI** in real-time using historical data.  
2. Forecast **future AQI (next 24 hours)** using LSTM for temporal patterns.  
3. Display **interactive visualizations** for trend, gauge, heatmap, and pollutants.  
4. Provide **next hour AQI prediction** for immediate decision-making.  

---

## 🧩 Features

| Feature | Model / Method |
|---------|----------------|
| Current AQI | Random Forest |
| Health Status | Based on AQI category |
| Gauge Meter | Random Forest |
| Trend Graph | Historical Data |
| Next 24h AQI Forecast | LSTM |
| Next Hour AQI | LSTM |

---

## 🛠️ Tools & Technologies

- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Plotly, Streamlit  
- **Data Processing:** Feature engineering, lag features, rolling statistics  
- **Visualization:** Line charts, heatmaps, gauge meters, bar charts  
- **Web App:** Streamlit for interactive dashboard  

---

## 🔧 Project Workflow

### 1. Data Collection and Pre-processing
- Historical air quality data including PM2.5, PM10, NO2, O3, and weather features.
- Data stored in `/data/ancona_data.csv`.
- The missing values were handled by using the ffill and bfill because it is time series data so there qont be any etreme outliers.

### 2. Feature Engineering
- Compute AQI from pollutants.  
- Create lag features, rolling mean, rolling std for predictive modeling.

### 3. Model Building
- **Random Forest:** Predicts current AQI using lag & rolling features.  
- **LSTM:** Forecasts next 24 hours AQI using sequential data and MinMax scaling.

### 4. Dashboard Integration
- Streamlit dashboard shows:
  - Current AQI + Health Status + Gauge
  - AQI Trend chart
  - Daily & Hourly Heatmap
  - Pollutant Contribution
  - LSTM Forecast Line Graph
  - Next Hour Predicted AQI

---

## 🚀 How to Run

1. Clone the repository:  
```bash
git clone https://github.com/yourusername/preetham-s-u-engineer-roadmap.git
Navigate to project folder:
cd Python-Projects/Air-Quality-Prediction
Install dependencies:
pip install -r requirements.txt
Run Streamlit dashboard:
streamlit run app.py
📈 Results
Random Forest accurately predicts current AQI.
LSTM forecasts next 24 hours AQI capturing temporal trends.
Dashboard visualizations provide actionable insights for environmental monitoring.
Impact: Real-time + predictive monitoring of air quality suitable for smart city applications and public health interventions.
🔮 Architecture
Dataset
   |
   ├── Random Forest → Current AQI → Dashboard display
   |
   └── LSTM → Future AQI (24 hrs) → Forecast Graph + Next Hour AQI
Hybrid AI System: Combines fast prediction + temporal forecasting for full coverage.
📌 Key Takeaways / How to Explain
Why Random Forest? Fast, stable, accurate for current AQI prediction.
Why LSTM? Captures sequential patterns to forecast next 24 hours.
Why both? Hybrid approach ensures real-time + future predictions, industry-ready solution.
🔗 References
Kaggle: Air Quality Datasets
Python & ML Libraries: Pandas, Scikit-learn, TensorFlow/Keras
Streamlit Documentation: https://docs.streamlit.io/
Plotly Docs: https://plotly.com/python/
📌 Next Steps
Deploy as web API for real-time predictions
Integrate weather forecast data to improve AQI forecast
Expand to multiple cities for generalization
