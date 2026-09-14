from datetime import UTC, datetime, timedelta
from uuid import UUID

from celery import Celery
from sqlalchemy import select

from src.config import get_settings
from src.models.entities import EvaluationRun
from src.runtime import get_runtime

settings = get_settings()
celery_app = Celery("rag", broker=settings.redis_url.get_secret_value())
celery_app.conf.update(
    task_serializer="json", accept_content=["json"], result_serializer="json",
    task_ignore_result=True, task_acks_late=True, task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1, broker_connection_retry_on_startup=True,
    task_soft_time_limit=840, task_time_limit=900,
    task_default_queue="ingestion",
    task_routes={"worker.tasks.evaluate_collection": {"queue": "evaluation"}},
    beat_schedule={"recover-pending-jobs": {"task": "worker.tasks.dispatch_pending", "schedule": 30.0}},
)


@celery_app.task(name="worker.tasks.index_document")
def index_document(collection_id: str, document_id: str) -> None:
    get_runtime().ingestion.index(UUID(collection_id), UUID(document_id))


@celery_app.task(name="worker.tasks.dispatch_pending")
def dispatch_pending() -> None:
    runtime = get_runtime()
    for collection_id, document_id in runtime.repository.pending_documents():
        index_document.delay(str(collection_id), str(document_id))
    with runtime.database.session() as session:
        stale = session.scalars(select(EvaluationRun).where(EvaluationRun.status == "processing",
                                     EvaluationRun.created_at < datetime.now(UTC) - timedelta(hours=2)))
        for run in stale:
            run.status, run.error_code = "failed", "WorkerInterrupted"
        runs = list(session.scalars(select(EvaluationRun).where(EvaluationRun.status == "pending").limit(10)))
    for run in runs:
        evaluate_collection.delay(str(run.collection_id), str(run.run_id))


@celery_app.task(name="worker.tasks.evaluate_collection", soft_time_limit=3500, time_limit=3600)
def evaluate_collection(collection_id: str, run_id: str) -> None:
    from src.evaluation.runner import run_live, write_report
    runtime = get_runtime()
    cid, rid = UUID(collection_id), UUID(run_id)
    with runtime.repository.indexing_lock(rid) as acquired:
        if not acquired:
            return
        with runtime.database.session() as session:
            run = session.scalar(select(EvaluationRun).where(EvaluationRun.collection_id == cid,
                                                             EvaluationRun.run_id == rid).with_for_update())
            if run is None or run.status != "pending":
                return
            run.status = "processing"
        try:
            report = run_live(runtime, cid)
            with runtime.database.session() as session:
                run = session.scalar(select(EvaluationRun).where(EvaluationRun.collection_id == cid, EvaluationRun.run_id == rid))
                if run is not None:
                    run.status, run.report = "completed", report
            write_report(report, runtime.settings.results_root / collection_id / f"{run_id}.md")
        except Exception as error:
            with runtime.database.session() as session:
                run = session.scalar(select(EvaluationRun).where(EvaluationRun.collection_id == cid, EvaluationRun.run_id == rid))
                if run is not None:
                    run.status, run.error_code = "failed", type(error).__name__
            runtime.telemetry.event("evaluation", {"collection_id": collection_id, "run_id": run_id,
                                                     "error_type": type(error).__name__})
