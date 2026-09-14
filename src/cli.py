"""Operational commands that use the same services as the Streamlit UI."""

import argparse
from pathlib import Path
from uuid import UUID

from src.runtime import Runtime, get_runtime


def seed_demo(runtime: Runtime) -> UUID:
    name = "Demo HR Policies"
    fingerprint = runtime.settings.fingerprint(embedding=True)
    collection = next((item for item in runtime.repository.collections()
                       if item.name == name and item.embedding_fingerprint == fingerprint), None)
    if collection is None:
        collection = runtime.repository.create_collection(name, fingerprint)
    existing = {item.filename for item in runtime.repository.documents(collection.collection_id)}
    root = Path(__file__).resolve().parents[1] / "sample_docs/hr_policies"
    for path in sorted(root.glob("*.md")):
        if path.name not in existing:
            runtime.ingestion.upload(collection.collection_id, path.name, "text/markdown", path.read_bytes(), "demo-seed")
    return collection.collection_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["seed-demo", "health"])
    arguments = parser.parse_args()
    runtime = get_runtime()
    if arguments.command == "seed-demo":
        print(f"Demo collection queued: {seed_demo(runtime)}")
    else:
        for row in runtime.health():
            print(f"{row['Service']}: {row['Status']}")


if __name__ == "__main__":
    main()
