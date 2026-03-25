import os
from google.adk.tools import FunctionTool
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from .aws_monthly import aws_cost_forecast_tool
from .azure_monthly import azure_cost_forecast_tool
from .gcp_monthly import gcp_cost_forecast_tool

# forecast_tool_adk = FunctionTool(azure_cost_forecast_tool)

# root_agent = Agent(
#     name="cloud_cost_agent",
#     model="gemini-2.0-flash",
#     tools=[forecast_tool_adk],
#     description="AZURE cost forecasting agent using preloaded predictions.",
#    instruction="""
# You are an AZURE Cost Forecasting Assistant.

# Your purpose is to answer user questions ONLY by using the AZURE Cost Forecast Tool.

# RULES:
# - Always call the azure_cost_forecast_tool to answer user questions.
# - Never guess or create numbers yourself.
# - Never summarize or modify the tool output; return it EXACTLY as the tool generates it.
# - Do not include any JSON, metadata, or extra explanation.
# - Do not answer directly from your own knowledge. ALWAYS use the tool.

# When the user asks a question:
# 1. Pass the question to the azure_cost_forecast_tool.
# 2. Return the tool output exactly as-is.
# """
# )

forecast_tool_aws   = FunctionTool(aws_cost_forecast_tool)
forecast_tool_azure = FunctionTool(azure_cost_forecast_tool)
forecast_tool_gcp   = FunctionTool(gcp_cost_forecast_tool)

root_agent = Agent(
    name="monthly_cloud_cost_agent",
    #model=LiteLlm("groq/llama-3.3-70b-versatile"),
    model="gemini-2.0-flash",
    #model=LiteLlm("groq/openai/gpt-oss-20b"),
    tools=[forecast_tool_aws, forecast_tool_azure, forecast_tool_gcp],
    description="Multi-cloud cost forecasting agent",
#     instruction="""
# You are an AWS cost forecasting assistant.

# RULES:
# - Always call AWS_cost_forecast_tool first to retrieve forecast data.
# - Then analyze the returned context USING THE MODEL (you).
# - Never guess numbers.
# - When asked any question, follow table rules strictly.
# """

instruction="""
You are a unified Cloud Cost Forecasting Assistant.

RULES:
- Always call the correct tool based on the platform in the user question (AWS / Azure / GCP).
- Never answer without using a tool.
- Return tool output exactly as-is.
"""
)




