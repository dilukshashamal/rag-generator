# Engineering Decisions

## 001: PostgreSQL Is the Durable Source of Truth

Use SQLAlchemy, pgvector and explicit composite collection/document foreign keys.
Every repository operation takes collection_id. Collections carry a revision used
in cache keys. Deletion and indexing lock the same collection row; successful
indexing atomically replaces chunks and increments the revision. Advisory locks
serialize duplicate deliveries for a document. Celery is a delivery mechanism;
pending/failed jobs and their status remain durable in PostgreSQL.

## 002: Local Storage and Bounded Extraction

Store files beneath a configured root using UUIDs, never user filenames. Validate
extension, reported MIME, magic signature and parser output; bound ZIP expansion
and XML parts for DOCX. Reject encrypted/scanned PDFs with actionable messages.
Use tiktoken to bound chunks, retaining page and heading metadata. Short structural
sections may be smaller than the nominal 450-700 token target.

## 003: Providers and Failure Policy

Use small synchronous provider protocols. HTTP adapters use configurable endpoints
and models; registry identifiers identify implementations, not deployment defaults.
Defaults selecting providers/models/endpoints live only in .env.example. LLM retries
are capped at three attempts per model with jitter, then same-provider alternate
model and alternate provider. Safety refusal stops fallback. Redis circuit state is
shared by workers. Embedding model fingerprints are stored with collections and
must match at query time; model changes require a new/reindexed collection.

## 004: Conservative Grounding

The generator returns claims with chunk IDs and exact supporting quotes. Version 1
only accepts claim text that is an exact source excerpt. This intentionally favors
verifiable extractive answers over fluent unsupported paraphrases. Citation checks
are structural safeguards, not a proof that a source is true or relevant. Relevance
is gated by configurable reranker scores; ambiguous questions may abstain. Documents
and chat history stay untrusted data in the prompt, never system instructions.

### PDF passage scoring correction

The four-page microgrid PDF extracted completely but page-wide relevance scoring
rejected the page containing the requested procedure. The normal-range question
also failed the relevance threshold. Score bounded overlapping windows with the
reranker's own tokenizer and aggregate by maximum, retaining the complete original
chunk for citation validation. This fixes existing indexed collections without
re-embedding or lowering the relevance threshold. More windows cost additional
reranker inference; cap both candidates and windows, spreading capped windows across
the source. Scores are relevance heuristics, not calibrated probabilities. Preserve
safe abstention and expose metadata-only reasons in request details. Explicitly allow
quoting operator procedures and complete extracted table rows as reference facts.

## 005: Cache and Privacy Boundaries

Embedding keys include collection, model fingerprint and content hash. Answer keys
include collection revision, normalized question, bounded history and full pipeline
configuration fingerprint. History matters for follow-ups; cache values stay in local
Redis with TTL. Telemetry uses an explicit metadata allowlist and never exception
messages, raw prompts, answers, filenames or document content. Optional external
evaluation sends content only when explicitly enabled and invoked.

## 006: Evaluation and Deployment Scope

Synthetic HR policies are test fixtures, not real employment advice. Offline tests
and deterministic evaluation are distinguished from live provider measurements.
Ragas is an optional dependency and separately invoked. Initial deployment is a
trusted single-workspace assessment app bound to localhost. Collection isolation is
not authentication: internet/multi-tenant deployments need an identity-aware gateway,
authorization, shared principal rate limits, TLS, malware scanning and operational
backups. These are documented deployment limitations, not silently simulated features.

## 007: Native UI Theme

Use Streamlit theme environment variables for the complete color palette rather than
forcing light backgrounds with CSS over potentially dark native widgets. The default
Compose palette uses white/slate surfaces, dark text and a blue action accent. Custom
CSS is limited to presentation and derives colors from the active theme; selected
navigation retains native radio controls and keyboard focus. Semantic status colors
remain native. This keeps theme changes separate from collection and RAG behavior.

## 008: Evaluation Dataset and Rate Limiting for Free-Tier Quotas

Evaluation in live mode can execute multiple LLM operations per test case (retrieval embedding,
answer generation, validation retries, and optional Ragas judge evaluations). To accommodate
strict free-tier quotas (e.g. Gemini 15 RPM, Groq 6,000 TPM), the default evaluation dataset is
curated to a representative 10-question benchmark covering all policy documents and question
classes, alongside the 10-case benchmark in `evals/microgrid_dataset.json`. Live
evaluations include configurable pacing delays (`eval_request_delay_seconds`) to prevent burst
rate-limit failures.
