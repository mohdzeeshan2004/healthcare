import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from pathlib import Path

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
    st.title("🔮 Basic Forecasting Module")
    forecast_days = st.slider("Forecast horizon in days", 7, 90, 30)

    model_df = df[["Date", "Total System Load"]].dropna().copy()
    model_df["Day_Index"] = np.arange(len(model_df))
    X = model_df[["Day_Index"]]
    y = model_df["Total System Load"]

    model = LinearRegression()
    model.fit(X, y)

    future_index = np.arange(len(model_df), len(model_df) + forecast_days)
    future_dates = pd.date_range(model_df["Date"].max() + pd.Timedelta(days=1), periods=forecast_days)
    future_pred = model.predict(future_index.reshape(-1, 1))

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Total System Load": future_pred})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model_df["Date"], y=model_df["Total System Load"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecasted Total System Load"], mode="lines", name="Forecast"))
    fig.update_layout(title="Simple Linear Forecast of Total System Load", xaxis_title="Date", yaxis_title="Total System Load")
    st.plotly_chart(fig, use_container_width=True)

    st.warning("This is a starter forecast. For the final capstone, improve this using ARIMA, Prophet, or machine learning models with lag features.")
    st.dataframe(forecast_df, use_container_width=True)

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
