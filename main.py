
import streamlit as st
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Dict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# ------------------- PAGE CONFIGURATION -------------------

st.set_page_config(
    page_title=":rainbow[AI Study Guide - LangGraph]",
    page_icon="🤖",
    layout="centered"
)

load_dotenv()

# ------------------- API KEY CONFIGURATION ----------------

api_key = os.getenv("gem")

# Also support Streamlit Cloud secrets
if not api_key:
    try:
        api_key = st.secrets["gem"]
    except (KeyError, FileNotFoundError):
        api_key = None

if not api_key:
    st.error(
        "Groq API key not found. "
        "Please configure GROQ_API_KEY in your .env file "
        "or Streamlit secrets."
    )
    st.stop()


# ------------------- MODEL CONFIGURATION ------------------

@st.cache_resource
def load_model(api_key):

    return ChatGroq(
        model_name="openai/gpt-oss-20b",
        groq_api_key=api_key,
        timeout=60,
        max_retries=2
    )


model = load_model(api_key)


# ------------------- MODULES -------------------------------

modules = [
    "Python",
    "SQL",
    "Power BI",
    "EDA",
    "Machine Learning",
    "Deep Learning",
    "Generative AI",
    "Agentic AI"
]


# ------------------- SESSION STATE -------------------------

if "selected_module" not in st.session_state:
    st.session_state.selected_module = None

if "conv" not in st.session_state:
    st.session_state.conv = []

if "feedback" not in st.session_state:
    st.session_state.feedback = None


# ------------------- LANGGRAPH STATE -----------------------

class StudyState(TypedDict):
    module: str
    question: str
    history: List[Dict[str, str]]
    is_valid: bool
    answer: str


# ------------------- HELPER FUNCTION -----------------------

def extract_text(response):

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

            elif isinstance(item, str):
                text_parts.append(item)

        content = "\n".join(text_parts)

    return str(content).strip()


# ------------------- NODE 1: MODULE VALIDATION -------------

def validate_module(state: StudyState):

    validation_prompt = f"""
You are a strict subject relevance checker.

Selected module: {state['module']}

User question: {state['question']}

Determine whether the user's question is related
to the selected module.

Rules:
1. Return YES if the question is related.
2. Return NO if the question is unrelated.
3. Consider relevant subtopics and practical applications.
4. Return ONLY YES or NO.
"""

    response = model.invoke(validation_prompt)

    result = extract_text(response).strip().upper()

    is_valid = result.startswith("YES")

    return {
        "is_valid": is_valid
    }


# ------------------- CONDITIONAL ROUTING -------------------

def check_module(state: StudyState):

    if state["is_valid"]:
        return "generate_answer"

    return "invalid_response"


# ------------------- NODE 2: GENERATE ANSWER ----------------

def generate_answer(state: StudyState):

    history_text = ""

    # Retain the latest 12 messages for context
    recent_history = state["history"][-12:]

    for msg in recent_history:

        role = (
            "User"
            if msg["role"] == "user"
            else "AI Mentor"
        )

        history_text += (
            f"{role}: {msg['content']}\n"
        )

    prompt = f"""
You are an expert AI Study Assistant specialized ONLY in
{state['module']}.

Your objective is to help students understand concepts
and develop practical skills.

Rules:

1. Answer only questions related to {state['module']}.
2. Provide clear and structured explanations.
3. Explain concepts step by step.
4. Use practical examples wherever necessary.
5. Provide Python or SQL code examples when relevant.
6. Format responses using Markdown.
7. Use headings, bullet points and code blocks.
8. Maintain context from previous conversations.
9. If the question is unrelated, do not answer it.
10. Avoid unnecessarily lengthy answers.

Previous Conversation:

{history_text}

Current Question:

{state['question']}

Provide a helpful and accurate response.
"""

    response = model.invoke(prompt)

    answer = extract_text(response)

    return {
        "answer": answer
    }


# ------------------- NODE 3: INVALID RESPONSE --------------

def invalid_response(state: StudyState):

    return {
        "answer": (
            "Sorry, I can only help with questions related "
            f"to **{state['module']}**.\n\n"
            "Please ask a question connected to your "
            "selected learning module."
        )
    }


# ------------------- BUILD LANGGRAPH -----------------------

builder = StateGraph(StudyState)

builder.add_node(
    "validate_module",
    validate_module
)

builder.add_node(
    "generate_answer",
    generate_answer
)

builder.add_node(
    "invalid_response",
    invalid_response
)

builder.add_edge(
    START,
    "validate_module"
)

builder.add_conditional_edges(
    "validate_module",
    check_module,
    {
        "generate_answer": "generate_answer",
        "invalid_response": "invalid_response"
    }
)

builder.add_edge(
    "generate_answer",
    END
)

builder.add_edge(
    "invalid_response",
    END
)

study_graph = builder.compile()


# ------------------- MAIN PAGE -----------------------------

if st.session_state.selected_module is None:

    st.title("👋 :rainbow[Welcome to AI Study Guide]")

    st.write(
        ":green[Your personalized AI learning assistant]."
    )

    st.write(
        ":red[Please select a learning module to begin]."
    )

    selected = st.selectbox(
        "📌 Select Module",
        ["-- Select --"] + modules
    )

    if selected != "-- Select --":

        st.session_state.selected_module = selected

        st.rerun()

    st.stop()


# ------------------- MODULE PAGE ---------------------------

module = st.session_state.selected_module

st.title(f"🎯 {module} Assistant")

st.write(
    f"I am your dedicated AI Assistant for **{module}**."
)

st.write("How can I help you today?")


# ------------------- LANGGRAPH VISUALIZATION ---------------

with st.expander("🔄 View LangGraph Workflow"):

    try:

        graph_image = (
            study_graph.get_graph().draw_mermaid_png()
        )

        st.image(
            graph_image,
            caption="AI Study Guide LangGraph Workflow",
            use_container_width=True
        )

    except Exception:

        st.code(
            study_graph.get_graph().draw_mermaid(),
            language="mermaid"
        )


# ------------------- CHAT HISTORY --------------------------

for msg in st.session_state.conv:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# ------------------- USER QUERY ----------------------------

prompt = st.chat_input("Type your question")

if prompt:

    with st.chat_message("user"):

        st.markdown(prompt)

    try:

        with st.chat_message("assistant"):

            with st.spinner(
                "AI Mentor is thinking..."
            ):

                initial_state = {
                    "module": module,
                    "question": prompt,
                    "history": st.session_state.conv.copy(),
                    "is_valid": False,
                    "answer": ""
                }

                # Execute LangGraph workflow

                result = study_graph.invoke(
                    initial_state
                )

                content = result["answer"]

                st.markdown(content)

        # Save conversation only after successful execution

        st.session_state.conv.append({
            "role": "user",
            "content": prompt
        })

        st.session_state.conv.append({
            "role": "assistant",
            "content": content
        })

    except Exception as e:

        st.error(
            f"Error generating response: {str(e)}"
        )

        st.info(
            "Please retry. Check your Groq API key, "
            "model availability and API rate limits."
        )


# ------------------- SIDEBAR --------------------------------

with st.sidebar:

    st.title("⚙️ Study Guide Settings")

    st.write(f"**Current Module:** {module}")

    st.caption(
        "Model: OpenAI GPT-OSS 20B via Groq"
    )

    if st.button("🔄 Change Module"):

        st.session_state.selected_module = None

        st.session_state.conv = []

        st.session_state.feedback = None

        st.rerun()

    if st.button("🗑️ Clear Conversation"):

        st.session_state.conv = []

        st.session_state.feedback = None

        st.rerun()


# ------------------- DOWNLOAD CHAT --------------------------

if st.session_state.conv:

    chat_text = ""

    for msg in st.session_state.conv:

        role = (
            "User"
            if msg["role"] == "user"
            else "AI Mentor"
        )

        chat_text += (
            f"{role}: {msg['content']}\n\n"
        )

    st.download_button(
        "📥 Download Conversation",
        chat_text,
        file_name=f"{module}_Chat.txt",
        mime="text/plain"
    )

    st.feedback(
        "stars",
        key="user_feedback"
    )


# ------------------- FOOTER ---------------------------------

st.divider()

st.caption(
    "🤖 AI Study Guide | Powered by "
    "LangGraph, Groq and OpenAI GPT-OSS"
)