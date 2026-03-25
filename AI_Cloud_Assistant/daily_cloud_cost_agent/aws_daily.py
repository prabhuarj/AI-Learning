

# IMPORTS

import pandas as pd
import google.generativeai as genai
from prophet import Prophet

from dotenv import load_dotenv
import os
import shutil

from datetime import timedelta

# Load key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# STEP 1 — UPLOAD & LOAD CSV

file_path = r"C:\AI Project\Cost_docs\aws_cost.csv" 

df = pd.read_csv(file_path)


# Remove rows where "Service total" appears
df = df[df["Service"] != "Service total"]

df.rename(columns={df.columns[0]: "date"}, inplace=True)

# Drop total column if exists
df = df.drop(columns=["Total costs($)"], errors="ignore")

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

# Clean numeric columns ($, commas → float)
for col in df.columns[1:]:
    df[col] = df[col].replace(r"[\$,]", "", regex=True).astype(float)

df.fillna(0, inplace=True)



# STEP 2 — LONG FORMAT

df_long = df.melt(
    id_vars=["date"],
    var_name="service",
    value_name="cost"
)

df_long["service"] = (
    df_long["service"]
    .str.replace(r"\(\$\)", "", regex=True)
    .str.strip()
)

df_long = df_long.sort_values(["date", "service"]).reset_index(drop=True)

# STEP 3 — DAILY TOTAL FORECAST (GLOBAL COST)

daily_total = df_long.groupby("date")["cost"].sum().reset_index()
daily_total.rename(columns={"date": "ds", "cost": "y"}, inplace=True)

model_total = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.5,      
    seasonality_prior_scale=10,     
)

model_total.add_seasonality(name='monthly', period=30.5, fourier_order=5)

model_total.fit(daily_total)

future = model_total.make_future_dataframe(periods=365, freq="D")
forecast = model_total.predict(future)

forecast_daily = forecast[["ds", "yhat"]].rename(
    columns={"yhat": "predicted_daily_cost"}
)

# SERVICE-WISE DAILY FORECAST

service_daily_forecasts = []

for service_name, group in df_long.groupby("service"):

    print(f"Training DAILY Prophet for service: {service_name}")

    # Aggregate per day
    df_service = group.groupby("date")["cost"].sum().reset_index()
    df_service.rename(columns={"date": "ds", "cost": "y"}, inplace=True)

    # Skip if not enough historical data
    if len(df_service) < 20:
        print(f"Skipping '{service_name}' — not enough data.\n")
        continue

    # Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.8,
        seasonality_mode="additive",
        interval_width=0.90
    )
    model.fit(df_service)

    # Predict next 365 days
    future = model.make_future_dataframe(periods=365, freq='D')
    forecast = model.predict(future)

    # Daily forecast & clip negatives to 0
    fc_daily = (
        forecast[['ds', 'yhat']]
        .set_index('ds')
        .clip(lower=0)  
        .reset_index()
        .assign(service=service_name)
    )

    fc_daily.rename(columns={'yhat': 'predicted_daily_cost'}, inplace=True)

    service_daily_forecasts.append(fc_daily)

# Combine all services
forecast_service_daily = pd.concat(service_daily_forecasts, ignore_index=True)

# Sort
forecast_service_daily = forecast_service_daily.sort_values(
    ["service", "ds"]
).reset_index(drop=True)


# # STEP 6 — CONTEXT GENERATOR FOR LLM
# def build_context():


#     today = pd.Timestamp.today().normalize()
#     one_year_out = today + timedelta(days=60)

#     filtered_df = forecast_service_daily[
#       (forecast_service_daily['ds'] >= today) &
#       (forecast_service_daily['ds'] <= one_year_out)
#     ].copy()

#     daily_service_summary = (
#       filtered_df
#       .pivot(index='ds', columns='service', values='predicted_daily_cost')
#       .fillna(0)
#     )

#     daily_service_summary_str = daily_service_summary.round(2).to_string()

#      # Daily costs — ALL DATA
#     daily_summary = forecast_daily.to_string(index=False)

   
#     context = f"""
# AWS Cost Forecast Data:
# ============================
# #SERVICE-WISE DAILY COST (all available days)
# # ==============================

#  {daily_service_summary_str}

# ==============================
# 3️⃣ DAILY COSTS (all available days)
# ==============================
# {daily_summary}

# """
#     return context

# STEP 6 — CONTEXT GENERATOR FOR LLM (NEXT 60 DAYS)
def build_context():

    today = pd.Timestamp.today().normalize()
    next_60_days = today + timedelta(days=60)

    # -----------------------------
    # SERVICE-WISE DAILY FORECAST (NEXT 60 DAYS)
    # -----------------------------
    service_filtered_df = forecast_service_daily[
        (forecast_service_daily['ds'] >= today) &
        (forecast_service_daily['ds'] <= next_60_days)
    ].copy()

    daily_service_summary = (
        service_filtered_df
        .pivot(index='ds', columns='service', values='predicted_daily_cost')
        .fillna(0)
    )

    daily_service_summary_str = daily_service_summary.round(2).to_string()

    # -----------------------------
    # TOTAL DAILY FORECAST (NEXT 60 DAYS)
    # -----------------------------
    total_filtered_df = forecast_daily[
        (forecast_daily['ds'] >= today) &
        (forecast_daily['ds'] <= next_60_days)
    ].copy()

    daily_total_summary_str = total_filtered_df.round(2).to_string(index=False)

    # -----------------------------
    # FINAL CONTEXT
    # -----------------------------
    context = f"""
AWS Cost Forecast Data:
==============================
1️⃣ SERVICE-WISE DAILY COST (NEXT 60 DAYS)
==============================
{daily_service_summary_str}

==============================
2️⃣ TOTAL DAILY COST (NEXT 60 DAYS)
==============================
{daily_total_summary_str}
"""
    return context




def aws_cost_forecast_tool(question: str):
   context = build_context()
   return {
        "context": context,
        "question": question
    }

# ============================
# STEP 7 — SAVE FORECAST OUTPUTS
# ============================

import os

output_folder = r"C:\AI Project\cost\aws_daily_forecast_output"


if os.path.exists(output_folder):
    print(f"Clearing existing folder at {output_folder}...")
    shutil.rmtree(output_folder)

os.makedirs(output_folder, exist_ok=True)

forecast_service_daily.to_csv(
   fr"{output_folder}\aws_forecast_service_wise_daily.csv",    
    index=False
 )

forecast_daily.to_csv(
    fr"{output_folder}\forecast_daily.csv",
    index=False
)




