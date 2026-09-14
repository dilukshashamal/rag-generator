import math
import statistics
from typing import Any


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else None


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int = 10) -> float | None:
    if not relevant:
        return None
    return next((1 / rank for rank, item in enumerate(retrieved[:k], 1) if item in relevant), 0.0)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    def mean(key: str) -> float | None:
        values = [row[key] for row in rows if row.get(key) is not None]
        return statistics.mean(values) if values else None
    abstentions = [row for row in rows if row.get("status") == "abstained"]
    return {
        "recall_at_5": mean("recall_at_5"), "mrr_at_10": mean("mrr_at_10"),
        "citation_accuracy": mean("citation_accuracy"),
        "answer_relevance_lexical": mean("answer_relevance_lexical"),
        "faithfulness_extractive": mean("faithfulness_extractive"),
        "no_answer_precision": (sum(not row["answerable"] for row in abstentions) / len(abstentions)) if abstentions else None,
        "p50_latency_ms": percentile([row["latency_ms"] for row in rows], 0.5),
        "p95_latency_ms": percentile([row["latency_ms"] for row in rows], 0.95),
        "cache_hit_rate": mean("cache_hit"), "average_llm_calls": mean("llm_calls"),
    }
