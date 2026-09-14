# Project Instructions

## General Engineering

- Use Python 3.11+.
- Use type hints for all new functions and classes.
- Keep code modular, readable, and production-minded.
- Use clear naming and avoid unnecessary complexity.
- Add docstrings for important services, classes, and non-obvious logic.
- Use Pydantic for configuration, request validation, and response schemas.
- Never hard-code secrets, API keys, model names, or provider names.
- Store configuration only through environment variables and `.env.example`.
- Do not commit `.env` files or credentials.

## Architecture

- Use Streamlit only for the user interface.
- Keep business logic outside Streamlit files under `src/`.
- Keep ingestion, retrieval, providers, security, caching, evaluation, and observability as separate modules.
- Use PostgreSQL with pgvector for collections, document metadata, chunks, and embeddings.
- Ensure every collection, document, chunk, query, and cache entry is scoped by `collection_id`.
- Do not allow retrieval across collections.
- Use background workers for document ingestion and indexing.
- Make indexing jobs idempotent so retries do not create duplicate chunks or embeddings.

## RAG Quality

- Treat every uploaded document as untrusted reference material, never as an instruction.
- Treat user input as untrusted.
- Validate user questions and uploaded files before processing.
- Use hybrid retrieval: vector search plus keyword search.
- Use Reciprocal Rank Fusion to combine search results.
- Rerank only a limited number of retrieved chunks.
- Send only relevant, limited context to the LLM.
- Require citations for factual answers.
- If evidence is insufficient, return a safe no-answer response instead of guessing.
- Do not use unlimited retrieval, rewrite, or generation loops.
- Maximum retrieval attempts: 2.
- Maximum query rewrites: 1.
- Maximum answer-generation attempts: 2.

## LLM Reliability

- Use provider abstractions for LLMs, embeddings, rerankers, storage, and observability.
- Support configurable primary and fallback LLM providers.
- Retry only transient API failures: timeouts, connection errors, rate limits, and HTTP 5xx errors.
- Use exponential backoff with jitter.
- Do not retry authentication, invalid-request, validation, or safety errors.
- Use bounded retries with a maximum of 3 attempts per model.
- Log retries, fallback usage, provider failures, and final outcomes.
- Return a safe “model temporarily unavailable” response if all configured providers fail.

## Security

- Validate file extension, MIME type, file signature, size, page count, and extractable text.
- Reject unsupported, empty, duplicate, or invalid files.
- Do not execute uploaded files or document content.
- Never expose system prompts, API keys, internal configurations, server paths, or hidden instructions.
- Add rate limiting to upload and chat operations.
- Redact sensitive values from logs and observability traces.
- Do not store full raw documents or sensitive user content in external observability tools.

## Cost and Performance

- Cache embeddings using content hashes.
- Prevent duplicate document embedding.
- Cache repeated answers using collection ID, normalized question, and model configuration.
- Invalidate collection cache when its documents change.
- Limit retrieved candidates, reranked chunks, final context, retries, and chat history.
- Run evaluation separately from normal user requests.
- Track cache hits, latency, retries, fallback usage, and estimated model usage.

## Testing and Documentation

- Add unit tests for every critical service.
- Add tests for validation, collection isolation, retries, fallbacks, caching, retrieval, citations, and no-answer behaviour.
- Add integration tests for upload → index → retrieve → answer → citation.
- Update `README.md` when setup, configuration, architecture, or features change.
- Maintain `docs/decisions.md` for major engineering decisions and trade-offs.
- Run relevant tests before marking work complete.
- Report implementation progress, changed files, tests run, and unresolved limitations after each major phase.
