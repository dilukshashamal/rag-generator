import argparse
from pathlib import Path
from uuid import UUID

from src.evaluation.runner import run_live, run_offline, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicit offline or collection-scoped live evaluation")
    parser.add_argument("--collection-id", type=UUID)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("evals/results/latest_report.md"))
    arguments = parser.parse_args()
    if arguments.offline == bool(arguments.collection_id):
        parser.error("Choose either --offline or --collection-id")
    if arguments.offline:
        report = run_offline()
    else:
        from src.runtime import get_runtime
        report = run_live(get_runtime(), arguments.collection_id)
    write_report(report, arguments.output)
    print(f"Evaluation complete: {arguments.output}")


if __name__ == "__main__":
    main()
