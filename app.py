# pip install streamlit prophet plotly pandas

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from datetime import datetime
import requests

# -------------------------------
# Load Live Data from World Bank API
# -------------------------------
@st.cache_data
def get_world_bank_inflation():
    url = "http://api.worldbank.org/v2/country/NG/indicator/FP.CPI.TOTL.ZG?format=json&per_page=1000"
    response = requests.get(url)
    json_data = response.json()[1]
    data = pd.DataFrame([{
        "Date": pd.to_datetime(d['date']),
        "Inflation": d['value']
    } for d in json_data if d['value'] is not None])
    return data.sort_values("Date")

# -------------------------------
# Prophet Forecasting
# -------------------------------
def forecast_inflation(df, periods):
    data = df.rename(columns={"Date": "ds", "Inflation": "y"})
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast

# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Nigeria Inflation Forecast", layout="wide")
st.title("🇳🇬 Nigeria Inflation Forecasting Model")

# Load live data
data = get_world_bank_inflation()

# Show raw data
with st.expander("📊 Show Raw Data"):
    st.dataframe(data.tail(12))

# User input for forecast period
forecast_period = st.slider("Select number of months to forecast", 3, 36, 12)

# Optional: Input current month's inflation rate
if st.checkbox("📥 Input current month's inflation"):
    current_inflation = st.number_input("Enter current inflation rate (%)", min_value=0.0, max_value=100.0, step=0.1)
    today = datetime.today().replace(day=1)
    if today not in data['Date'].values:
        new_row = pd.DataFrame({"Date": [today], "Inflation": [current_inflation]})
        data = pd.concat([data, new_row], ignore_index=True)

# Forecast
forecast = forecast_inflation(data, forecast_period)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Inflation'], name='Historical Inflation'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
fig.update_layout(title="Inflation Forecast", xaxis_title="Date", yaxis_title="Inflation Rate (%)")
st.plotly_chart(fig, use_container_width=True)

# Show forecast table
with st.expander("📅 Show Forecasted Values"):
    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period)
    forecast_display = forecast_display.rename(columns={
        'ds': 'Date',
        'yhat': 'Forecast',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    })
    st.dataframe(forecast_display)

# Export button
st.download_button(
    label="📥 Download Forecast as CSV",
    data=forecast_display.to_csv(index=False),
    file_name="nigeria_inflation_forecast.csv",
    mime="text/csv"
)

