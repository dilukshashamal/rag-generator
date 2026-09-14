from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.models.entities import Base
from src.providers.storage import LocalStorageProvider


def test_retry_and_context_limits(settings: Settings) -> None:
    for field, value in [("llm_max_attempts", 4), ("context_top_k", 7), ("chunk_overlap", 121)]:
        with pytest.raises(ValidationError):
            Settings.model_validate({**settings.model_dump(), field: value})


def test_model_configuration_changes_fingerprint(settings: Settings) -> None:
    changed = settings.model_copy(update={"embedding_model": "a-different-model"})
    assert changed.fingerprint(embedding=True) != settings.fingerprint(embedding=True)
    assert "test:test" not in settings.fingerprint()


def test_schema_has_composite_scope_constraint() -> None:
    constraints = Base.metadata.tables["chunks"].foreign_key_constraints
    assert any(list(constraint.column_keys) == ["collection_id", "document_id"] for constraint in constraints)


def test_storage_rejects_traversal_and_roundtrips(settings: Settings) -> None:
    storage = LocalStorageProvider(settings.storage_root)
    with pytest.raises(ValueError):
        storage.read("../secret")
    key = storage.put(uuid4(), uuid4(), b"content")
    assert storage.read(key) == b"content"
    storage.delete(key)
    assert not storage.path(key).exists()
