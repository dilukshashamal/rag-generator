from typing import Any
from uuid import UUID

import streamlit as st

from src.models.schemas import Answer, ChatMessage
from src.runtime import Runtime
from src.security.questions import ValidationError


def safe_action(runtime: Runtime, action: Any) -> Any:
    try:
        return action()
    except ValidationError as error:
        st.error(str(error))
    except Exception as error:
        runtime.telemetry.event("failure", {"error_type": type(error).__name__})
        st.error("The operation could not be completed. Check System Status and try again.")
    return None


def knowledge_base(runtime: Runtime, collection_id: UUID | None, principal: str) -> None:
    st.header("Knowledge Bases")
    with st.popover("Create collection", icon=":material/add:"):
        with st.form("create_collection"):
            name = st.text_input("Collection name", max_chars=120, placeholder="Employee handbook")
            create = st.form_submit_button("Create", type="primary")
        if create:
            value = safe_action(runtime, lambda: runtime.repository.create_collection(name, runtime.settings.fingerprint(embedding=True)))
            if value:
                st.session_state["pending_collection"] = str(value.collection_id)
                st.rerun()
    if collection_id is None:
        st.info("No collections yet.", icon=":material/folder_open:")
        return
    st.subheader("Documents")
    files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
                             key=f"uploads-{collection_id}", max_upload_size=runtime.settings.max_upload_mb)
    if st.button("Upload", icon=":material/upload:", type="primary", disabled=not files):
        with st.spinner("Validating uploads..."):
            for file in files:
                document_id = safe_action(runtime, lambda f=file: runtime.ingestion.upload(
                    collection_id, f.name, f.type, f.getvalue(), principal))
                if document_id:
                    st.success(f"{file.name} queued for indexing.")
    document_status(runtime, collection_id)
    st.divider()
    with st.popover("Delete collection", icon=":material/delete:"):
        confirm = st.checkbox("Delete this collection and all its documents", key=f"confirm-{collection_id}")
        if st.button("Delete permanently", disabled=not confirm, key=f"delete-{collection_id}"):
            safe_action(runtime, lambda: runtime.ingestion.delete_collection(collection_id))
            st.rerun()


@st.fragment(run_every="5s")
def document_status(runtime: Runtime, collection_id: UUID) -> None:
    documents = safe_action(runtime, lambda: runtime.repository.documents(collection_id))
    if not documents:
        st.caption("No documents in this collection.")
        return
    st.dataframe([{"Document": doc.filename, "Status": doc.status, "Chunks": doc.chunk_count,
                   "Attempts": doc.attempts, "Issue": doc.error_code or ""} for doc in documents],
                 hide_index=True, width="stretch")
    choice = st.selectbox("Document", documents, format_func=lambda doc: doc.filename,
                           key=f"document-choice-{collection_id}")
    retry, remove, _ = st.columns([1, 1, 3])
    with retry:
        if st.button("Retry", icon=":material/refresh:", disabled=choice.status != "failed"):
            safe_action(runtime, lambda: runtime.repository.retry_document(collection_id, choice.document_id))
            st.rerun(scope="fragment")
    with remove:
        with st.popover("Delete", icon=":material/delete:"):
            if st.button("Confirm document deletion", key=f"remove-{choice.document_id}"):
                safe_action(runtime, lambda: runtime.ingestion.delete_document(collection_id, choice.document_id))
                st.rerun()


def render_answer(answer: Answer) -> None:
    # st.text prevents document-supplied links, HTML or image beacons from rendering.
    st.text(answer.answer)
    status, confidence, duration = st.columns(3)
    status.metric("Evidence", "Grounded" if answer.grounded else answer.status.capitalize())
    confidence.metric("Confidence", answer.confidence.capitalize())
    duration.metric("Total time", f"{answer.metrics.latency_ms / 1000:.2f}s")
    if answer.citations:
        for index, citation in enumerate(answer.citations, 1):
            location = f"Page {citation.page}" if citation.page else citation.heading or "Document"
            with st.expander(f"[{index}] {citation.filename} / {location}"):
                st.text(citation.excerpt)
    with st.expander("Request details"):
        reasons = {
            "no_relevant_evidence": "No retrieved passages passed the relevance checks.",
            "no_supported_claims": "The model did not produce an answer supported by valid source quotes.",
            "collection_changed": "The collection changed during this request. Please ask again.",
        }
        if answer.metrics.abstention_reason in reasons:
            st.caption(reasons[answer.metrics.abstention_reason])
        st.dataframe([{"Retrieval": f"{answer.metrics.retrieval_ms:.0f} ms",
                       "Cache": "Hit" if answer.metrics.cache_hit else "Miss",
                       "Retries": answer.metrics.retries,
                       "Fallback": "Used" if answer.metrics.fallback_used else "Not used",
                       "LLM calls": answer.metrics.llm_calls,
                       "Retrieval attempts": answer.metrics.retrieval_attempts,
                       "Selected sources": len(answer.metrics.selected_chunk_ids),
                       "Generation attempts": answer.metrics.generation_attempts}], hide_index=True, width="stretch")


def ask_questions(runtime: Runtime, collection_id: UUID | None, principal: str) -> None:
    st.header("Ask Questions")
    if collection_id is None:
        st.info("Select a knowledge base to start a conversation.", icon=":material/forum:")
        return
    key = f"conversation-{collection_id}"
    messages = st.session_state.setdefault(key, [])
    if st.button("Clear conversation", icon=":material/delete_sweep:", disabled=not messages):
        st.session_state[key] = []
        st.rerun()
    for entry in messages:
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                st.text(entry["content"])
            else:
                render_answer(Answer.model_validate(entry["answer"]))
    question = st.chat_input("Ask about the selected collection", max_chars=1000)
    if question:
        history = [ChatMessage(role=entry["role"], content=entry["content"][:4000]) for entry in messages]
        with st.chat_message("user"):
            st.text(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                answer = runtime.pipeline.ask(collection_id, question, principal, history)
            render_answer(answer)
        messages.extend([{"role": "user", "content": question},
                         {"role": "assistant", "content": answer.answer, "answer": answer.model_dump(mode="json")}])
        maximum = max(2, runtime.settings.chat_history_messages)
        st.session_state[key] = messages[-maximum:]


def evaluation_dashboard(runtime: Runtime, collection_id: UUID | None) -> None:
    st.header("Evaluation")
    if collection_id is None:
        st.info("Select a knowledge base to view evaluation results.")
        return
    st.caption("Collection evaluation")
    if st.button("Run evaluation", type="primary", icon=":material/play_arrow:"):
        run = safe_action(runtime, lambda: runtime.repository.create_evaluation(collection_id))
        if run:
            st.success("Evaluation queued.")
    evaluation_status(runtime, collection_id)


@st.fragment(run_every="10s")
def evaluation_status(runtime: Runtime, collection_id: UUID) -> None:
    runs = safe_action(runtime, lambda: runtime.repository.evaluations(collection_id))
    if not runs:
        st.info("No evaluation runs yet.")
        return
    run = st.selectbox("Evaluation run", runs, format_func=lambda item: f"{item.created_at:%Y-%m-%d %H:%M} / {item.status}")
    if run.error_code:
        st.error(f"Evaluation failed: {run.error_code}")
    if not run.report:
        st.caption(f"Status: {run.status}")
        return
    metrics = run.report["metrics"]
    columns = st.columns(3)
    for column, key, label in zip(columns, ["recall_at_5", "citation_accuracy", "no_answer_precision"],
                                   ["Recall @ 5", "Citation accuracy", "No-answer precision"], strict=True):
        value = metrics.get(key)
        column.metric(label, f"{value:.1%}" if value is not None else "Not measured")
    st.dataframe([{"Metric": key, "Value": f"{value:.4f}" if value is not None else "Not measured"}
                   for key, value in metrics.items()], hide_index=True, width="stretch")
    st.bar_chart({"Latency (ms)": {"P50": metrics.get("p50_latency_ms", 0), "P95": metrics.get("p95_latency_ms", 0)}}, color=st.get_option("theme.primaryColor"))
    with st.expander("Question results"):
        st.dataframe(run.report["rows"], hide_index=True, width="stretch")


def system_status(runtime: Runtime) -> None:
    st.header("System Status")
    if st.button("Refresh", icon=":material/refresh:"):
        st.rerun()
    rows = runtime.health()
    try:
        from worker.tasks import celery_app
        workers = celery_app.control.inspect(timeout=1).ping() or {}
        rows.append({"Service": "Indexing worker", "Status": "Ready" if workers else "Unavailable"})
    except Exception:
        rows.append({"Service": "Indexing worker", "Status": "Unavailable"})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.subheader("Recent errors")
    errors = list(runtime.telemetry.recent_errors)
    if errors:
        st.dataframe([{"Time": entry["time"], "Type": entry["error_type"]} for entry in errors], hide_index=True)
    else:
        st.caption("No errors recorded in this application process.")
