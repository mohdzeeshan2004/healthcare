import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import warnings

st.set_page_config(
    page_title="UAC Care Load Analytics",
    page_icon="🏥",
    layout="wide"
)

DATA_PATH = Path("data/HHS_Unaccompanied_Alien_Children_Program.csv")

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    numeric_cols = [col for col in df.columns if col != "Date"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("nan", np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    df["Total System Load"] = df["Children in CBP custody"] + df["Children in HHS Care"]
    df["Net Intake Pressure"] = (
        df["Children transferred out of CBP custody"]
        - df["Children discharged from HHS Care"]
    )
    df["Discharge Offset Ratio"] = np.where(
        df["Children transferred out of CBP custody"] > 0,
        df["Children discharged from HHS Care"] / df["Children transferred out of CBP custody"],
        np.nan
    )
    df["Care Load Growth Rate"] = df["Total System Load"].pct_change() * 100
    df["Backlog Flag"] = np.where(df["Net Intake Pressure"] > 0, "Pressure", "Relief")
    df["7-Day Rolling Load"] = df["Total System Load"].rolling(7, min_periods=1).mean()
    df["14-Day Rolling Load"] = df["Total System Load"].rolling(14, min_periods=1).mean()
    df["7-Day Net Intake Avg"] = df["Net Intake Pressure"].rolling(7, min_periods=1).mean()
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Dataset Quality",
        "System Load Dashboard",
        "CBP vs HHS Comparison",
        "Net Intake & Backlog",
        "Forecasting",
        "Insights & Recommendations"
    ]
)

st.sidebar.markdown("---")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("No records available for the selected date range.")
    st.stop()

latest = filtered_df.iloc[-1]

if page == "Project Overview":
    st.title("🏥 System Capacity & Care Load Analytics for Unaccompanied Children")
    st.markdown("""
    This healthcare analytics project monitors the care pipeline for unaccompanied children using CBP and HHS operational data.
    The goal is to understand total system load, intake pressure, discharge efficiency, backlog risk, and capacity stress over time.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Start Date", df["Date"].min().strftime("%d %b %Y"))
    c3.metric("End Date", df["Date"].max().strftime("%d %b %Y"))
    c4.metric("Latest Total Load", f"{int(latest['Total System Load']):,}")

    st.subheader("Project Objectives")
    st.markdown("""
    - Quantify daily and cumulative care load across CBP and HHS.
    - Identify periods of capacity pressure and relief.
    - Analyze balance between transfers into HHS care and discharges from HHS care.
    - Support staffing, shelter planning, and policy-level decision-making.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)

elif page == "Dataset Quality":
    st.title("🧹 Dataset Quality Check")
    c1, c2, c3 = st.columns(3)
    c1.metric("Valid Rows", f"{len(df):,}")
    c2.metric("Duplicate Dates", int(df["Date"].duplicated().sum()))
    c3.metric("Missing Values", int(df.isna().sum().sum()))

    st.subheader("Missing Values by Column")
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    st.dataframe(missing, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)

elif page == "System Load Dashboard":
    st.title("📊 System Load Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total System Load", f"{int(latest['Total System Load']):,}")
    c2.metric("Children in HHS Care", f"{int(latest['Children in HHS Care']):,}")
    c3.metric("Children in CBP Custody", f"{int(latest['Children in CBP custody']):,}")
    c4.metric("Net Intake Pressure", f"{int(latest['Net Intake Pressure']):,}")

    fig = px.line(filtered_df, x="Date", y="Total System Load", title="Total System Load Over Time")
    st.plotly_chart(fig, use_container_width=True)

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=filtered_df["Date"], y=filtered_df["Total System Load"], mode="lines", name="Daily Load"))
    fig_roll.add_trace(go.Scatter(x=filtered_df["Date"], y=filtered_df["7-Day Rolling Load"], mode="lines", name="7-Day Avg"))
    fig_roll.add_trace(go.Scatter(x=filtered_df["Date"], y=filtered_df["14-Day Rolling Load"], mode="lines", name="14-Day Avg"))
    fig_roll.update_layout(title="Daily Load vs Rolling Averages", xaxis_title="Date", yaxis_title="Children Under Care")
    st.plotly_chart(fig_roll, use_container_width=True)

elif page == "CBP vs HHS Comparison":
    st.title("🏛️ CBP vs HHS Load Comparison")
    long_df = filtered_df.melt(
        id_vars="Date",
        value_vars=["Children in CBP custody", "Children in HHS Care"],
        var_name="Care Stage",
        value_name="Children"
    )
    fig = px.area(long_df, x="Date", y="Children", color="Care Stage", title="CBP Custody vs HHS Care Load")
    st.plotly_chart(fig, use_container_width=True)

    monthly = filtered_df.groupby("Month", as_index=False)[["Children in CBP custody", "Children in HHS Care"]].mean()
    fig_month = px.bar(monthly, x="Month", y=["Children in CBP custody", "Children in HHS Care"], barmode="group", title="Monthly Average CBP vs HHS Load")
    st.plotly_chart(fig_month, use_container_width=True)

elif page == "Net Intake & Backlog":
    st.title("⚠️ Net Intake & Backlog Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Net Intake", f"{filtered_df['Net Intake Pressure'].mean():.1f}")
    c2.metric("Max Pressure Day", f"{int(filtered_df['Net Intake Pressure'].max()):,}")
    c3.metric("Discharge Offset Ratio", f"{filtered_df['Discharge Offset Ratio'].mean():.2f}")

    fig = px.bar(filtered_df, x="Date", y="Net Intake Pressure", color="Backlog Flag", title="Net Intake Pressure: Transfers Minus Discharges")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(filtered_df, x="Date", y="7-Day Net Intake Avg", title="7-Day Average Net Intake Pressure")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("Positive net intake means transfers into HHS care are higher than discharges. Sustained positive values can indicate backlog pressure.")

elif page == "Forecasting":
    st.title("🔮 Advanced Forecasting Module")
    st.markdown("""
    This module compares a simple baseline forecast with **ARIMA time-series forecasting** and a
    **machine learning lag-feature model**. The lag model uses previous care-load values, rolling averages,
    calendar features, and recent intake pressure to predict future system load.
    """)

    target_col = st.selectbox(
        "Select forecasting target",
        ["Total System Load", "Children in HHS Care", "Children in CBP custody", "Net Intake Pressure"],
        index=0
    )
    forecast_days = st.slider("Forecast horizon in days", 7, 120, 30)
    model_choice = st.radio(
        "Forecasting model",
        ["Linear Baseline", "ARIMA", "Random Forest with Lag Features", "Compare All"],
        horizontal=True
    )

    model_df = df[["Date", target_col, "Net Intake Pressure"]].dropna().copy()
    model_df = model_df.sort_values("Date").reset_index(drop=True)
    model_df["Day_Index"] = np.arange(len(model_df))

    def build_lag_features(data: pd.DataFrame, target: str) -> pd.DataFrame:
        out = data.copy()
        out["lag_1"] = out[target].shift(1)
        out["lag_7"] = out[target].shift(7)
        out["lag_14"] = out[target].shift(14)
        out["rolling_7"] = out[target].shift(1).rolling(7).mean()
        out["rolling_14"] = out[target].shift(1).rolling(14).mean()
        out["rolling_std_7"] = out[target].shift(1).rolling(7).std()
        out["net_intake_lag_1"] = out["Net Intake Pressure"].shift(1)
        out["day_of_week"] = out["Date"].dt.dayofweek
        out["month"] = out["Date"].dt.month
        out["day_index"] = np.arange(len(out))
        return out

    def linear_forecast(data: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
        X = data[["Day_Index"]]
        y = data[target]
        model = LinearRegression()
        model.fit(X, y)
        future_index = np.arange(len(data), len(data) + horizon)
        future_dates = pd.date_range(data["Date"].max() + pd.Timedelta(days=1), periods=horizon)
        preds = model.predict(future_index.reshape(-1, 1))
        return pd.DataFrame({"Date": future_dates, "Forecast": preds, "Model": "Linear Baseline"})

    def arima_forecast(data: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            y = data.set_index("Date")[target].asfreq("D").interpolate()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(y, order=(7, 1, 1))
                fitted = model.fit()
            pred = fitted.forecast(steps=horizon)
            return pd.DataFrame({"Date": pred.index, "Forecast": pred.values, "Model": "ARIMA(7,1,1)"})
        except Exception as e:
            st.error(f"ARIMA could not run: {e}")
            return pd.DataFrame(columns=["Date", "Forecast", "Model"])

    def lag_feature_forecast(data: pd.DataFrame, target: str, horizon: int):
        feature_cols = [
            "lag_1", "lag_7", "lag_14", "rolling_7", "rolling_14", "rolling_std_7",
            "net_intake_lag_1", "day_of_week", "month", "day_index"
        ]
        lag_df = build_lag_features(data, target).dropna().reset_index(drop=True)
        if len(lag_df) < 60:
            st.error("Not enough data for lag-feature forecasting after feature creation.")
            return pd.DataFrame(columns=["Date", "Forecast", "Model"]), None, lag_df

        split_idx = int(len(lag_df) * 0.8)
        train = lag_df.iloc[:split_idx]
        test = lag_df.iloc[split_idx:]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42
        )
        model.fit(train[feature_cols], train[target])
        test_pred = model.predict(test[feature_cols])
        mae = mean_absolute_error(test[target], test_pred)
        rmse = np.sqrt(mean_squared_error(test[target], test_pred))
        metrics = {"MAE": mae, "RMSE": rmse}

        history = data[["Date", target, "Net Intake Pressure"]].copy().reset_index(drop=True)
        forecasts = []
        last_date = history["Date"].max()
        median_net_intake = history["Net Intake Pressure"].tail(14).median()

        for step in range(1, horizon + 1):
            future_date = last_date + pd.Timedelta(days=step)
            temp = history.copy()
            temp = pd.concat([
                temp,
                pd.DataFrame({"Date": [future_date], target: [np.nan], "Net Intake Pressure": [median_net_intake]})
            ], ignore_index=True)
            temp_features = build_lag_features(temp, target)
            row = temp_features.iloc[[-1]][feature_cols].fillna(method="ffill", axis=0).fillna(0)
            pred = float(model.predict(row)[0])
            forecasts.append({"Date": future_date, "Forecast": pred, "Model": "Random Forest Lag Features"})
            history = pd.concat([
                history,
                pd.DataFrame({"Date": [future_date], target: [pred], "Net Intake Pressure": [median_net_intake]})
            ], ignore_index=True)

        importances = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        return pd.DataFrame(forecasts), metrics, importances

    forecasts = []
    lag_metrics = None
    importances = None

    if model_choice in ["Linear Baseline", "Compare All"]:
        forecasts.append(linear_forecast(model_df, target_col, forecast_days))

    if model_choice in ["ARIMA", "Compare All"]:
        arima_df = arima_forecast(model_df, target_col, forecast_days)
        if not arima_df.empty:
            forecasts.append(arima_df)

    if model_choice in ["Random Forest with Lag Features", "Compare All"]:
        rf_df, lag_metrics, importances = lag_feature_forecast(model_df, target_col, forecast_days)
        if not rf_df.empty:
            forecasts.append(rf_df)

    if forecasts:
        forecast_df = pd.concat(forecasts, ignore_index=True)
        forecast_df["Forecast"] = forecast_df["Forecast"].clip(lower=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=model_df["Date"],
            y=model_df[target_col],
            mode="lines",
            name="Actual"
        ))
        for name, part in forecast_df.groupby("Model"):
            fig.add_trace(go.Scatter(
                x=part["Date"],
                y=part["Forecast"],
                mode="lines",
                name=name
            ))
        fig.update_layout(
            title=f"{target_col} Forecast",
            xaxis_title="Date",
            yaxis_title=target_col
        )
        st.plotly_chart(fig, use_container_width=True)

        if lag_metrics is not None:
            c1, c2 = st.columns(2)
            c1.metric("Random Forest Test MAE", f"{lag_metrics['MAE']:.2f}")
            c2.metric("Random Forest Test RMSE", f"{lag_metrics['RMSE']:.2f}")

        if importances is not None and not importances.empty:
            st.subheader("Lag Feature Importance")
            fig_imp = px.bar(importances, x="Importance", y="Feature", orientation="h", title="Most Important Forecasting Features")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("Forecast Output")
        st.dataframe(forecast_df, use_container_width=True)
        st.download_button(
            "Download forecast CSV",
            forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="uac_forecast_output.csv",
            mime="text/csv"
        )

    st.info("For your research paper, explain ARIMA as a statistical time-series model and the Random Forest lag model as a feature-based ML forecasting approach using past care-load behavior.")

elif page == "Insights & Recommendations":
    st.title("💡 Insights & Recommendations")
    st.subheader("Key Analytical Insights")
    st.markdown("""
    1. Total system load helps measure the combined operational responsibility across CBP and HHS care systems.
    2. HHS care load is the main driver of total system pressure because it is much larger than CBP custody load.
    3. Net intake pressure shows whether the care pipeline is expanding or relieving pressure.
    4. Rolling averages help identify sustained strain periods more clearly than daily raw values.
    5. Discharge offset ratio is useful for understanding how effectively the system is balancing incoming transfers with successful discharges.
    """)

    st.subheader("Policy & Operations Recommendations")
    st.markdown("""
    - Monitor net intake pressure daily to detect early backlog buildup.
    - Use rolling averages for staffing and shelter capacity planning.
    - Prioritize discharge efficiency during sustained positive intake periods.
    - Build alert thresholds for high-load periods and unusual reporting changes.
    - Use forecasting to support proactive medical, welfare, and shelter resource planning.
    """)

    st.subheader("Download Filtered Dataset")
    st.download_button(
        "Download processed CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="processed_uac_capacity_data.csv",
        mime="text/csv"
    )
