# Microgrid PDF regression verification

Verified on 15 September 2026 against the existing microgrid collection and the
configured live embedding, reranking, and LLM providers, with answer caching disabled.

## Diagnosis

All four pages extracted completely into four page-preserving chunks. The original
normal-range request rejected every candidate at the relevance threshold. The
below-35% request selected only page 1, while the response procedure is on pages 2–3;
the generator correctly returned empty claims for the incomplete context.

## Changes

- `src/providers/reranker.py`: score bounded overlapping passages and retain each
  original chunk for generation/citations. Keep the configured relevance threshold.
- `src/config.py` and `.env.example`: configurable passage length, overlap and count.
- `src/rag/grounding.py`: clarify quoting operational procedures and PDF table rows.
- `src/rag/pipeline.py`, `src/models/schemas.py`, `src/observability/events.py`, and
  `src/ui/pages.py`: metadata-only abstention reasons and source/generation counts.
- PDF regression fixture and tests; updated README and engineering decisions.

The local web, worker and scheduler services were rebuilt and restarted. Existing
collections and documents were preserved; no reindexing was required.

## Results

| Live question | Result | Citations |
| --- | --- | --- |
| Normal battery state-of-charge range | Answered: 35% to 90% | Page 1 |
| Operator response below 35% | Answered: notify on-call technician and verify generator readiness; below 20%, verify automatic start and escalate failure | Page 2 |
| Opening battery cabinets during a thermal alarm | Answered: must not open cabinets; safety precautions retained | Pages 3–4 |
| Exact backup-generator purchase price | Correctly abstained | None |

The answer-bearing passages scored 0.869 for the normal range and 0.602 for the
low-charge procedure, above both the current 0.4 threshold and the example 0.55.
The normal-range request took 22.58 seconds with cold model loading; subsequent
requests took 2.19–2.71 seconds in this run. These are observations, not latency guarantees.

- Offline suite: **68 passed**. Includes real PDF extraction and upload/index/citation
  regression tests using model doubles, plus bounded passage scoring tests.
- PostgreSQL/Redis integration suite: **5 passed**, including both PDF questions and
  collection isolation; model calls are deterministic doubles in this suite.
- Ruff: **all checks passed**.
- The four live checks above used real configured providers separately from those tests.

## Limits

Four live questions do not establish accuracy for every PDF or domain. Exact-quote
grounding remains conservative; complex layouts, scanned PDFs without OCR, and
questions requiring synthesis may still abstain. Passage count is bounded, and a
very token-dense source can exceed complete window coverage. First-use model loading
still adds latency.
