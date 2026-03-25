import streamlit as st
import asyncio
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import uuid


# --- 1. Import your existing agents ---
from supervisor_agent.agent import root_agent


# --- 2. Global Event Loop (SAFE for Streamlit) ---
@st.cache_resource
def get_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


# --- 3. Setup Persistent ADK Services ---
@st.cache_resource
def get_adk_instance():
    service = InMemorySessionService()

    runner = Runner(
        agent=root_agent,
        app_name="supervisor_agent_app",
        session_service=service,
    )

    return service, runner


# Initialize shared resources
event_loop = get_event_loop()
service, runner = get_adk_instance()


# --- 4. UI State ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 5. UI Setup ---
st.title("Cloud Cost Supervisor Agent")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- 6. Agent Runner (ASYNC) ---
async def run_agent(user_input: str) -> str:
    user_id = "streamlit_user"
    session_id = str(uuid.uuid4())  # 🔥 NEW SESSION PER QUESTION

    # Create a fresh session for this request
    await service.create_session(
        app_name="supervisor_agent_app",
        user_id=user_id,
        session_id=session_id,
    )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_input)],
    )

    events = runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    )

    full_response = ""

    async for event in events:
        if event.is_final_response():
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        full_response += part.text

    return full_response or "No response returned."




# --- 7. Input Handling ---
if prompt := st.chat_input("Ask about daily or monthly cloud costs…"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is processing..."):
            response = event_loop.run_until_complete(
                run_agent(prompt)
            )
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
