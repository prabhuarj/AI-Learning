
# IMPORTS



import pandas as pd
import google.generativeai as genai
from prophet import Prophet

from dotenv import load_dotenv
import os
import shutil

# Load key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# STEP 1 — UPLOAD & LOAD CSV

file_path = r"C:\AI Project\Cost_docs\gcp_cost.csv" 

df = pd.read_csv(file_path)

df_long = df.copy()

df_long['date'] = pd.to_datetime(df_long['date'], format='mixed', dayfirst=True)


# STEP 3 — DAILY TOTAL FORECAST (GLOBAL COST)

daily_total = df_long.groupby("date")["cost"].sum().reset_index()
daily_total.rename(columns={"date": "ds", "cost": "y"}, inplace=True)

model_total = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.5,      # more flexible trend
    seasonality_prior_scale=10,       # stronger seasonality fitting
)

model_total.add_seasonality(name='monthly', period=30.5, fourier_order=5)

model_total.fit(daily_total)

future = model_total.make_future_dataframe(periods=365, freq="D")
forecast = model_total.predict(future)

forecast_daily = forecast[["ds", "yhat"]].rename(
    columns={"yhat": "predicted_daily_cost"}
)



# STEP 4 — MONTHLY TOTAL FORECAST

forecast_monthly = (
    forecast_daily.set_index("ds")
    .resample("MS")
    .sum()
    .reset_index()
)

forecast_monthly.rename(
    columns={"ds": "month", "predicted_daily_cost": "predicted_monthly_cost"},
    inplace=True
)



# STEP 5 — INDIVIDUAL SERVICE FORECASTS

all_forecasts = []

for service_name, group in df_long.groupby("service"):
    print(f"Training Prophet for service: {service_name}")

    df_service = group.groupby("date")["cost"].sum().reset_index()
    df_service.rename(columns={"date": "ds", "cost": "y"}, inplace=True)

    if len(df_service) < 20:
        print(f"Skipping '{service_name}' — not enough data.\n")
        continue

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.8,
        seasonality_mode="additive",
        interval_width=0.90
    )
    model.fit(df_service)

    future = model.make_future_dataframe(periods=365, freq="D")
    fc = model.predict(future)

    monthly_fc = (
        fc[["ds", "yhat"]]
        .set_index("ds")
        .resample("MS")
        .sum()
        .clip(lower=0)
        .reset_index()
    )

    monthly_fc["service"] = service_name
    monthly_fc.rename(columns={"yhat": "predicted_monthly_cost"}, inplace=True)

    all_forecasts.append(monthly_fc)

forecast_all_services = pd.concat(all_forecasts, ignore_index=True)
forecast_all_services = forecast_all_services.sort_values(["service", "ds"]).reset_index(drop=True)



# STEP 6 — CONTEXT GENERATOR FOR LLM

def build_context():


    # Monthly totals
    total_monthly_forecast = forecast_monthly.rename(
        columns={"month": "forecast_month", "predicted_monthly_cost": "total_predicted_monthly_cost"}
    )
    total_monthly_summary = total_monthly_forecast.to_string(index=False)

    # Service-wise monthly forecasts
    service_summary = (
        forecast_all_services
        .pivot(index="ds", columns="service", values="predicted_monthly_cost")
        .fillna(0)
        .round(2)
    )
    service_summary_str = service_summary.to_string()

    # Final context block
    context = f"""
GCP Cost Forecast Data:

==============================
1️⃣ TOTAL MONTHLY FORECAST
==============================
{total_monthly_summary}

==============================
2️⃣ SERVICE-WISE MONTHLY FORECAST
==============================
{service_summary_str}

"""
    return context



# def gcp_cost_forecast_tool(question: str):
#     """
#    GCP Cost Forecast Tool
#     Input  : question (string)
#     Output : LLM response using forecast data
#     """

#     import google.generativeai as genai

#     context = build_context()

#     prompt = f"""
# You are a helpful assistant with strict rules:

#  RULES:
# - If the question is about **total monthly cost**, ONLY use the table labeled:
#   **'TOTAL MONTHLY FORECAST'**. This table explicitly contains the already computed total monthly costs. NEITHER sum service costs from the 'SERVICE-WISE MONTHLY FORECAST' table NOR derive totals from the 'DAILY COSTS' table.
#   Critically, for total monthly cost questions, **you MUST use the 'TOTAL MONTHLY FORECAST' table as is, even if it shows zero values for certain months. DO NOT attempt to compute totals by summing individual service costs from the 'SERVICE-WISE MONTHLY FORECAST' table for these questions.**
#   When a query asks for total costs over multiple months, identify the relevant 'total_predicted_monthly_cost' values for each month and sum them up directly from the 'TOTAL MONTHLY FORECAST' table.
# - If the question is about **a specific service's monthly cost**, use the 'SERVICE-WISE MONTHLY FORECAST' table.
#   In this table, the column headers represent individual AWS 'service' names, and the 'ds' column (index) represents the forecast month.
#   To extract the cost for a specific service in a given month, locate the intersection of the correct 'ds' (month) and 'service' (column header).
# - If the question is about **daily cost**, use the 'DAILY COSTS' table.

# Here is your data:

# {context}

# Question: {question}

# Answer using the correct table and show the numbers you used. Be as precise as possible.
# """
#     genai.configure(api_key=GOOGLE_API_KEY)
#     model = genai.GenerativeModel("gemini-2.0-flash")

#     response = model.generate_content(prompt)

#     return response.text.strip()

def gcp_cost_forecast_tool(question: str):
    # Tool returns only raw data + question packaged together
    context = build_context()
    return {
        "context": context,
        "question": question
    }

import os

output_folder = r"C:\AI Project\Cost_docs\gcp_forecast_output"


if os.path.exists(output_folder):
    print(f"Clearing existing Chroma DB at {output_folder}...")
    shutil.rmtree(output_folder)

os.makedirs(output_folder, exist_ok=True)


forecast_monthly.to_csv(
    fr"{output_folder}\forecast_monthly_total.csv",
    index=False
)

forecast_all_services.to_csv(
    fr"{output_folder}\forecast_service_wise_monthly.csv",
    index=False
)


