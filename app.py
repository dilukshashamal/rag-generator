"""Streamlit entrypoint. Services and data processing live under src/."""

from pathlib import Path
from uuid import UUID, uuid4

import streamlit as st
from pydantic import ValidationError

from src.runtime import get_runtime
from src.ui.pages import ask_questions, evaluation_dashboard, knowledge_base, system_status

st.set_page_config(page_title="RAG Generator", page_icon=":material/library_books:", layout="wide")
stylesheet = Path(__file__).parent / "src" / "ui" / "styles.css"
st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.title("RAG Generator")
    page = st.radio("Workspace", ["Knowledge Bases", "Ask Questions", "Evaluation", "System Status"],
                    label_visibility="collapsed")
    st.divider()

try:
    runtime = get_runtime()
except (ValidationError, ValueError):
    st.header(page)
    st.warning("Configuration is incomplete.", icon=":material/settings:")
    st.stop()

principal = st.session_state.setdefault("principal", str(uuid4()))
try:
    collections = runtime.repository.collections()
except Exception as error:
    runtime.telemetry.event("failure", {"error_type": type(error).__name__})
    collections = []
    if page != "System Status":
        st.error("Database unavailable. Check System Status.")

options = [str(item.collection_id) for item in collections]
names = {str(item.collection_id): item.name for item in collections}
pending = st.session_state.pop("pending_collection", None)
if pending in options:
    st.session_state["selected_collection"] = pending
if st.session_state.get("selected_collection") not in options:
    st.session_state.pop("selected_collection", None)
with st.sidebar:
    selected = st.selectbox("Knowledge base", options, format_func=lambda key: names[key],
                            key="selected_collection", placeholder="No collections")
    st.caption(f"{len(collections)} collections")
collection_id = UUID(selected) if selected else None

if page == "Knowledge Bases":
    knowledge_base(runtime, collection_id, principal)
elif page == "Ask Questions":
    ask_questions(runtime, collection_id, principal)
elif page == "Evaluation":
    evaluation_dashboard(runtime, collection_id)
else:
    system_status(runtime)
