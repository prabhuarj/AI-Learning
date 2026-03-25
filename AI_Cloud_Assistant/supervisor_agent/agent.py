from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from daily_cloud_cost_agent.agent import root_agent as daily_cost_agent
from monthly_cloud_cost_agent.agent import root_agent as monthly_cost_agent
from dash_userguide_agent.agent import root_agent as userguide_agent

root_agent = Agent(
    name="supervisor_agent",
    model="gemini-2.0-flash",
    #model=LiteLlm("groq/llama-3.3-70b-versatile"),
    description="Supervisor agent that routes queries to specialized agents",
    instruction="""
You are a supervisor agent.

You MUST re-evaluate the user intent from scratch for EVERY message.
Do NOT assume the topic continues from previous messages.
Do NOT assume the same agent should be reused.

Even if previous messages were about cost,
a new question may require a different agent.

Routing rules:
- If the query mentions daily, today, or last 24 hours → call the daily_cost_agent
- If the query mentions monthly, month, or billing cycle → call the monthly_cost_agent
- Otherwise → call the userguide_agent

Hard rules:
- Delegate to exactly ONE sub-agent
- Do NOT answer the user directly
- Ignore previous conversation context
- Re-evaluate routing for EVERY user message
- Never say phrases like "As an AI..."

You must not answer the user directly.
You must always call exactly one tool.
""",
   sub_agents=[daily_cost_agent, monthly_cost_agent, userguide_agent],

)