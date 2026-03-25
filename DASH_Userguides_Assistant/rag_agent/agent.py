

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm


from rag_pipeline import rag_tool  # your custom RAG pipeline function

# Register the RAG Tool
rag_tool_adk = FunctionTool(rag_tool)

# Define the Agent
root_agent = Agent(
    name="rag_agent",
    model="gemini-2.0-flash",
    #model=LiteLlm("groq/llama-3.3-70b-versatile"),
    tools=[rag_tool_adk],
    description="A Cloud Access Guide Assistant that answers user questions using AWS, Azure, GCP, and vSphere access documentation. It retrieves the correct steps and guidance directly from the uploaded access guides and returns responses with source citations for accuracy.",

    instruction="""
# You are a strict RAG proxy agent for Cloud Access Guides.

# Your ONLY role:
# → Send the user question directly to the rag_tool
# → Return the rag_tool output EXACTLY as it is generated

# Output Rules:
# - NEVER change, rewrite, summarize, add, or remove anything from the rag_tool result
# - PRESERVE citations exactly as returned by the rag_tool
# - Do NOT include JSON, metadata, or tool names like "rag_tool_response"
# - Do NOT add your own citations — only return what the tool provides

# If the rag_tool output does not contain an answer, respond instead with:
# "I don’t have this information in the current Access Guide documentation."

Always use the rag_tool to answer user questions.
"""
)

