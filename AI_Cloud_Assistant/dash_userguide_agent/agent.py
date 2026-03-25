# # ==========================================
# #  Google ADK Agent for RAG Study Assistant
# # ==========================================
# from google.genai.agents import Agent
# from google.genai.tools import FunctionTool
# from rag_pipeline import rag_tool
# import json

# # Register RAG Tool
# rag_tool_adk = FunctionTool(
#     func=rag_tool,
#     name="RAGStudyHelper",
#     description="Answer student questions using uploaded study materials."
# )

# # Define Agent
# study_agent = Agent(
#     name = "rag_agent",
#     model="gemini-1.5-flash",
#     tools=[rag_tool_adk],
#     instructions=(
#         "You are a friendly and knowledgeable study assistant. "
#         "Use RAGStudyHelper to provide accurate, context-based answers "
#         "from the uploaded course materials."
#     )
# )

# ==========================================
# Google ADK Agent for RAG Study Assistant
# ==========================================

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm


from .rag_pipeline import rag_tool  

# Register the RAG Tool
rag_tool_adk = FunctionTool(rag_tool)

# Define the Agent
root_agent = Agent(
    name="rag_agent",
    model="gemini-2.0-flash",
    #model=LiteLlm("groq/llama-3.3-70b-versatile"),
    tools=[rag_tool_adk],
    description="Multi-cloud DASH assistant.",
    instruction="""
You are a STRICT RAG-ONLY assistant.

Rules:
- Always use the RAG tool to answer questions.
- Do not answer from your own knowledge.
- Return the tool result exactly as it is.
"""
)
# instruction="""
# You are a STRICT RAG-ONLY assistant.

# Rules:
# 1. You MUST always call the `rag_tool` to answer every user question.
# 2. You are NOT allowed to use your own knowledge for answering.
# 3. You are NOT allowed to reason beyond the retrieved context.
# 4. If the RAG tool returns "I don't know" or empty/irrelevant content, you MUST reply ONLY with: "I don't know".
# 5. NEVER generate information not present in the RAG tool response.
# 6. NEVER guess, assume, infer, or elaborate.
# 7. Your final output MUST always be exactly the tool's returned text — nothing more, nothing less.
# 8. If the user asks anything outside the context, answer: "I don't know".
# """

# )



