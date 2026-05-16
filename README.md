# System Capacity & Care Load Analytics for Unaccompanied Children

A healthcare capacity analytics Streamlit app for analyzing CBP and HHS care load, intake pressure, discharge patterns, backlog indicators, and forecasting from the HHS Unaccompanied Alien Children Program dataset.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Core modules
- Project overview
- Dataset quality check
- System load analytics
- CBP vs HHS comparison
- Net intake and backlog analysis
- Advanced forecasting
- ARIMA forecasting
- Random Forest forecasting with lag features
- Forecast CSV download
- Insights and recommendations

## Forecasting models added
- Linear baseline trend model
- ARIMA(7,1,1) time-series model
- Random Forest model using lag features:
  - lag_1
  - lag_7
  - lag_14
  - rolling_7
  - rolling_14
  - rolling_std_7
  - net_intake_lag_1
  - day_of_week
  - month
  - day_index
