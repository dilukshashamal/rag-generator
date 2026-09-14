from src.evaluation.metrics import recall_at_k, reciprocal_rank, summarize
from src.evaluation.runner import load_cases, security_checks


def test_dataset_coverage() -> None:
    cases = load_cases()
    assert len(cases) >= 10
    assert sum(not case.answerable for case in cases) >= 2
    assert {"cross_document", "multi_chunk", "ambiguous", "direct"}.issubset({case.category for case in cases})
    from pathlib import Path
    full_cases = load_cases(Path("evals/dataset_full.json"))
    assert len(full_cases) >= 30
    assert sum(not case.answerable for case in full_cases) >= 6


def test_microgrid_dataset_has_ten_source_supported_cases() -> None:
    from pathlib import Path

    from pypdf import PdfReader

    from src.security.questions import validate_question

    cases = load_cases(Path("evals/microgrid_dataset.json"))
    assert len(cases) == len({case.id for case in cases}) == 10
    assert sum(case.answerable for case in cases) == 8
    assert sum(not case.answerable for case in cases) == 2
    pages = PdfReader("tests/fixtures/microgrid_operations_and_response_guide.pdf").pages
    source_text = [" ".join(page.extract_text().split()) for page in pages]
    for case in cases:
        assert validate_question(case.question)
        if case.answerable:
            assert case.expected_sources == ["microgrid_operations_and_response_guide.pdf|None"]
            assert any(" ".join(case.reference.split()) in page for page in source_text)
        else:
            assert not case.expected_sources


def test_metric_denominators() -> None:
    assert recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == 0.5
    assert reciprocal_rank(["x", "b"], {"b"}) == 0.5
    assert recall_at_k([], set(), 5) is None
    metrics = summarize([{"latency_ms": 1, "status": "abstained", "answerable": False},
                         {"latency_ms": 3, "status": "unavailable", "answerable": False},
                         {"latency_ms": 5, "status": "abstained", "answerable": True}])
    assert metrics["no_answer_precision"] == 0.5
    assert metrics["p50_latency_ms"] == 3
    assert metrics["citation_accuracy"] is None


def test_security_fixture_expectations() -> None:
    assert security_checks()["prompt_injection_refusal"] == 1
