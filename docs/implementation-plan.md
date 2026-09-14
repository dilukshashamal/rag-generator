# Implementation Plan

## Scope and Acceptance

Build a generic, collection-isolated document question-answering application using
Streamlit, PostgreSQL/pgvector, Redis and Celery. Runtime uploads must not require
code changes. Answers must expose verifiable source excerpts or safely abstain.
AGENTS.md applies to every phase. No existing application code was present.

## Phased Checklist

- [ ] Phase 1: foundations. Add configuration validation, schemas, container setup,
  database migrations, environment example and provider contracts. Validate settings
  and compose configuration. Document architectural decisions before implementation.
- [ ] Phase 2: ingestion. Validate signatures, MIME, bounds and extracted text;
  preserve source locations; tokenize and chunk; persist collection-scoped metadata;
  queue idempotent indexing with durable recovery and content-hash embedding reuse.
  Test malformed uploads, duplicates, chunk boundaries and job retries.
- [ ] Phase 3: retrieval and generation. Implement scoped vector/keyword retrieval,
  RRF, local reranking, bounded rewriting/generation, evidence validation, caching,
  transient retry classification, circuit breakers and provider fallback. Test
  isolation, citation forgery, abstention, cache invalidation and failure recovery.
- [ ] Phase 4: evaluation. Add synthetic HR source documents, >=30 labeled questions,
  >=6 unanswerable cases, injection cases, deterministic metrics, optional Ragas and
  collection-scoped asynchronous evaluation reports. Test metric denominators.
- [ ] Phase 5: Streamlit. Implement collection management, chat with source excerpts,
  evaluation results and system health. Keep orchestration in src services. Exercise
  empty/configuration-failure states and workflows with Streamlit AppTest.
- [ ] Phase 6: verification and handoff. Run unit and integration suites, lint and
  compilation checks; run real PostgreSQL/Redis checks when infrastructure is
  available; capture UI screenshots; update README, decisions and evaluation report
  with actual results and remaining operational limitations.

## Architecture and Data Flow

Upload -> validation/extraction -> local storage + pending database record -> Celery
dispatcher -> indexing task -> token chunks -> cached embeddings -> atomic chunk
replacement and collection revision increment. The database is the durable job
source; periodic dispatch recovers pending tasks after broker interruptions.

Question -> validation/rate limit -> revision-aware answer cache -> vector + keyword
search (20 each) -> RRF -> rerank (10) -> confidence threshold -> optional one rewrite
and second retrieval -> up to six contexts -> structured generation -> citation and
extractive claim checks -> at most one regeneration -> grounded answer or abstention.

Evaluation runs on a separate queue and never inside ordinary chat requests. Local
JSON logs and optional allowlisted Langfuse metadata contain no questions/documents.

## UI Direction

Visual thesis: a quiet document workspace with neutral surfaces, green action accents
and distinct success/warning/error states. Content plan: collection navigation,
working table or conversation, source inspector, evaluation and health views.
Interaction thesis: automatic indexing status refresh, bounded progress spinners,
and expandable evidence; no ornamental motion or marketing hero.

## Verification Strategy

Use deterministic provider doubles for offline service and pipeline tests, HTTP
mock transports for actual provider adapters, and an opt-in integration suite against
PostgreSQL/Redis for pgvector/FTS isolation, transactional indexing and job persistence.
Never describe fake-provider tests as live LLM quality evaluation. Reports identify
their execution mode and missing measurements. Preserve user-owned repository files.
