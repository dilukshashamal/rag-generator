import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter

from src.evaluation.metrics import recall_at_k, reciprocal_rank, summarize
from src.security.questions import ValidationError, injection_flag, validate_question

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EvalCase(BaseModel):
    id: str
    question: str = Field(max_length=1000)
    category: str
    expected_sources: list[str]
    reference: str
    answerable: bool


def load_cases(path: Path | None = None) -> list[EvalCase]:
    content = (path or PROJECT_ROOT / "evals/dataset.json").read_text(encoding="utf-8")
    return TypeAdapter(list[EvalCase]).validate_json(content)


def security_checks() -> dict[str, Any]:
    cases = json.loads((PROJECT_ROOT / "evals/security_dataset.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        actual = "accepted"
        if "document" in case:
            actual = "document_flagged" if injection_flag(case["document"]) else "accepted"
        else:
            try:
                validate_question(case["question"])
            except ValidationError:
                actual = "rejected"
        rows.append({"id": case["id"], "passed": actual == case["expected"]})
    return {"prompt_injection_refusal": sum(row["passed"] for row in rows) / len(rows), "cases": rows}


def run_offline() -> dict[str, Any]:
    """Deterministic lexical baseline, not a simulation of live vector/LLM quality."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    sections: list[tuple[str, str]] = []
    for path in sorted((PROJECT_ROOT / "sample_docs/hr_policies").glob("*.md")):
        from src.ingestion.extract import _text_sections
        sections.extend((f"{path.name}|{section.heading}", section.text)
                        for section in _text_sections(path.read_text(encoding="utf-8")))
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([text for _, text in sections])
    rows = []
    for case in load_cases():
        started = time.perf_counter()
        scores = (matrix @ vectorizer.transform([case.question]).T).toarray().ravel()
        ranked = [sections[index][0] for index in scores.argsort()[::-1][:10]]
        rows.append({"id": case.id, "answerable": case.answerable,
                     "recall_at_5": recall_at_k(ranked, set(case.expected_sources), 5),
                     "mrr_at_10": reciprocal_rank(ranked, set(case.expected_sources)),
                     "latency_ms": (time.perf_counter() - started) * 1000})
    return {"mode": "offline_lexical_baseline", "created_at": datetime.now(UTC).isoformat(),
            "metrics": summarize(rows), "security": security_checks(), "rows": rows,
            "limitations": ["No live pgvector, embedding, reranker or LLM measurement.",
                            "Null metrics were not measured. Lexical relevance and exact support are not semantic judges."]}


def run_live(runtime: Any, collection_id: UUID) -> dict[str, Any]:
    cases = load_cases(runtime.settings.eval_dataset_path)[:runtime.settings.eval_max_questions]
    rows, ragas_samples = [], []
    collection = runtime.repository.collection(collection_id)
    if collection.embedding_fingerprint != runtime.settings.fingerprint(embedding=True):
        raise ValidationError("Embedding configuration changed")
    actual_files = {doc.filename for doc in runtime.repository.documents(collection_id) if doc.status == "indexed"}
    required_files = {source.split("|")[0] for case in cases for source in case.expected_sources}
    if not required_files.issubset(actual_files):
        raise ValidationError("Load and index the documents required by the selected evaluation dataset.")
    for index, case in enumerate(cases):
        if index > 0 and getattr(runtime.settings, "eval_request_delay_seconds", 0) > 0:
            time.sleep(runtime.settings.eval_request_delay_seconds)
        vectors, _ = runtime.embeddings.embed(collection_id, [case.question], query=True)
        candidates = runtime.pipeline.retriever.retrieve(collection_id, case.question, vectors[0])
        ranked = runtime.reranker.rerank(case.question, candidates)
        sources = [f"{item.filename}|{item.heading}" for item in ranked]
        # Evaluation has an isolated rate-limit principal and still observes the same pipeline limits.
        answer = runtime.pipeline.ask(collection_id, case.question, f"evaluation:{collection_id}:{case.id}")
        source_map = {item.chunk_id: item for item in runtime.repository.chunks(
            collection_id, [citation.chunk_id for citation in answer.citations])}
        valid = [citation.excerpt in source_map[citation.chunk_id].content
                 and f"{citation.filename}|{citation.heading}" in case.expected_sources
                 for citation in answer.citations if citation.chunk_id in source_map]
        rows.append({"id": case.id, "answerable": case.answerable, "status": answer.status,
                     "recall_at_5": recall_at_k(sources, set(case.expected_sources), 5),
                     "mrr_at_10": reciprocal_rank(sources, set(case.expected_sources)),
                     "citation_accuracy": sum(valid) / len(answer.citations) if answer.citations else None,
                     "answer_relevance_lexical": float(case.reference.casefold() in answer.answer.casefold()) if case.answerable else None,
                     "faithfulness_extractive": sum(citation.chunk_id in source_map and citation.excerpt in source_map[citation.chunk_id].content
                                                    for citation in answer.citations) / len(answer.citations) if answer.citations else None,
                     "latency_ms": answer.metrics.latency_ms, "cache_hit": float(answer.metrics.cache_hit),
                     "llm_calls": answer.metrics.llm_calls})
        ragas_samples.append({"user_input": case.question, "response": answer.answer,
                              "retrieved_contexts": [item.content for item in ranked[:runtime.settings.context_top_k]],
                              "reference": case.reference})
    report = {"mode": "live", "collection_id": str(collection_id), "created_at": datetime.now(UTC).isoformat(),
              "metrics": summarize(rows), "security": security_checks(), "rows": rows,
              "limitations": ["Recall labels use filename and heading relevance.",
                              "Security score measures explicit validator fixtures, not all adversarial attacks."]}
    if runtime.settings.eval_enable_ragas:
        try:
            report["ragas"] = run_ragas(ragas_samples, runtime.settings)
        except Exception as error:
            report["ragas"] = {"status": "rate_limited" if "RateLimit" in type(error).__name__ else "failed"}
            report.setdefault("limitations", []).append(
                f"RAGAS evaluation skipped due to judge quota/rate limit: {type(error).__name__}"
            )
    if runtime.repository.collection(collection_id).revision != collection.revision:
        raise ValidationError("Collection changed during evaluation. Run the evaluation again.")
    return report


def run_ragas(samples: list[dict[str, Any]], settings: Any) -> dict[str, float]:
    """Explicit opt-in: evaluation content is sent to the configured judge endpoint."""
    if not settings.eval_llm_model or not settings.eval_llm_base_url or not settings.eval_llm_api_key.get_secret_value():
        raise ValidationError("Ragas requires an explicitly configured evaluation model, endpoint and key.")
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall
    from ragas.run_config import RunConfig
    judge = LangchainLLMWrapper(ChatOpenAI(model=settings.eval_llm_model,
                                          base_url=settings.eval_llm_base_url,
                                          api_key=settings.eval_llm_api_key.get_secret_value(),
                                          timeout=settings.llm_timeout_seconds, max_retries=3))
    eval_samples = samples[:10] if len(samples) > 10 else samples
    result = evaluate(EvaluationDataset.from_list(eval_samples), metrics=[Faithfulness(), LLMContextRecall()],
                      llm=judge, run_config=RunConfig(max_retries=3, max_workers=1, timeout=120),
                      show_progress=False, raise_exceptions=True)
    return {key: float(value) for key, value in result.to_pandas().select_dtypes("number").mean().items()}


def write_report(report: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Evaluation Report", "", f"Mode: {report['mode']}", f"Generated: {report['created_at']}",
             "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in report["metrics"].items():
        lines.append(f"| {key} | {'Not measured' if value is None else f'{value:.4f}'} |")
    lines.extend([f"| prompt_injection_refusal | {report['security']['prompt_injection_refusal']:.4f} |", ""])
    lines.extend(f"- {limitation}" for limitation in report.get("limitations", []))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
