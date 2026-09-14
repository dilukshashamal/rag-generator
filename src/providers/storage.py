import os
from pathlib import Path
from uuid import UUID, uuid4


class LocalStorageProvider:
    """UUID-only object keys with atomic writes and traversal prevention."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        result = (self.root / key).resolve()
        if not result.is_relative_to(self.root) or result == self.root:
            raise ValueError("Invalid storage key")
        return result

    def put(self, collection_id: UUID, document_id: UUID, content: bytes) -> str:
        key = f"{collection_id}/{document_id}"
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{uuid4()}.tmp")
        try:
            temporary.write_bytes(content)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return key

    def read(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)
