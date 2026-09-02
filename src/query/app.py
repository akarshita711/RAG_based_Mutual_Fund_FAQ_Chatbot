"""Frontend app (Streamlit) for the HDFC MF FAQ RAG assistant.

Wires Phases 5-7 to a browser UI: user question -> retrieval (Phase 5) ->
generation / answer synthesis (Phase 7) -> rendered answer with citation.

UI follows the PRD (FR-7): welcome line, 3 example questions, and the note
"Facts-only. No investment advice." Visual style mirrors the Groww palette:
white background with light-green (#00D09C family) accents.

Run from the project root:
    streamlit run src/query/app.py
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st  # noqa: E402

from src import config  # noqa: E402

# Streamlit Community Cloud exposes secrets via st.secrets, not os.environ.
# Mirror them into the environment so config/gemini can find GOOGLE_API_KEY
# the same way it does locally (where config.py loads it from .env).
try:
    for _k, _v in (dict(st.secrets) if hasattr(st, "secrets") else {}).items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:  # noqa: BLE001 - secrets access should never crash the app
    pass
from src.query.retriever import Retriever  # noqa: E402
from src.query.generator import generate  # noqa: E402

st.set_page_config(page_title="HDFC MF FAQ", layout="centered")

GROWW_CSS = """
<style>
  :root {
    --groww: #00D09C;
    --groww-dark: #00A87D;
    --groww-light: #E6FFF7;
    --groww-mist: #F2FFF9;
    --ink: #1F1F1F;
    --muted: #555555;
    --border: #D6F5EC;
  }
  .stApp { background: #FFFFFF; }
  .block-container { max-width: 740px; padding-top: 2.5rem; }
  .logo {
    color: var(--groww-dark); font-weight: 800; font-size: 20px;
    letter-spacing: -0.4px; margin-bottom: 1.1rem;
  }
  h1.gtitle { color: var(--ink); font-size: 26px; margin: 0 0 .2rem; }
  div[data-testid="stCaptionContainer"] p { color: var(--muted); }
  .stButton > button {
    background: var(--groww); color: #FFFFFF; border: none;
    font-weight: 600; border-radius: 999px; padding: .45rem 1rem;
    width: 100%; transition: background .12s ease;
  }
  .stButton > button:hover { background: var(--groww-dark); color: #FFFFFF; }
  div[data-testid="stChatMessage"] {
    background: var(--groww-mist); border: 1px solid var(--border);
    border-radius: 12px; padding: .6rem 1rem;
  }
  div[data-testid="stChatMessage"][data-testid="stChatMessageUser"] {
    background: #FFFFFF; border-color: #E6E6E6;
  }
  .note {
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #EEEEEE;
    color: var(--muted); font-size: 13px;
  }
</style>
"""
st.markdown(GROWW_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


EXAMPLE_QUESTIONS = [
    "What is the ELSS lock-in period?",
    "What is the expense ratio of HDFC Large Cap Fund?",
    "Is there an exit load on HDFC Small Cap Fund?",
]


def _linkify(text: str) -> str:
    """Turn bare URLs into clickable markdown links."""
    return re.sub(r"(https?://[^\s)\]]+)", r"[\1](\1)", text)


def answer_question(question: str) -> dict:
    return generate(question, get_retriever().retrieve)


def render_message(role: str, content: str | dict) -> None:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
        return
    result = content if isinstance(content, dict) else {"answer": str(content)}
    with st.chat_message("assistant"):
        st.markdown(_linkify(result.get("answer", "")))
        if result.get("educational_link"):
            st.caption(f"Educational link: {_linkify(result['educational_link'])}")


# ---------------------------------------------------------------------------
# Page: header + example shortcuts
# ---------------------------------------------------------------------------
st.markdown('<div class="logo">groww&nbsp;&nbsp;·&nbsp;&nbsp;HDFC Mutual Fund FAQ</div>',
            unsafe_allow_html=True)
st.markdown('<h1 class="gtitle">Ask anything about an HDFC scheme</h1>',
            unsafe_allow_html=True)
st.caption("Expense ratio · Exit load · Minimum SIP · ELSS lock-in · Risk · Benchmark")

st.session_state.setdefault("chat", [])

example_cols = st.columns(3)
for i, question in enumerate(EXAMPLE_QUESTIONS):
    if example_cols[i].button(question, key=f"example_{i}"):
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.spinner("Thinking…"):
            result = answer_question(question)
        st.session_state["chat"].append({"role": "assistant", "content": result})

# ---------------------------------------------------------------------------
# Chat input + history rendering
# ---------------------------------------------------------------------------
prompt = st.chat_input("Type a factual question…")
if prompt and prompt.strip():
    st.session_state["chat"].append({"role": "user", "content": prompt.strip()})
    with st.spinner("Thinking…"):
        result = answer_question(prompt.strip())
    st.session_state["chat"].append({"role": "assistant", "content": result})

for message in st.session_state["chat"]:
    render_message(message["role"], message["content"])

st.markdown('<div class="note">Facts-only · No investment advice. '
            'Every answer includes one official source link.</div>',
            unsafe_allow_html=True)