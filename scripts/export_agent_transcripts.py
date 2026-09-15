import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, Field


class ExportOptions(BaseModel):
    sources: list[Path] = Field(min_length=1)
    output: Path
    project_root: Path


class Redactor:
    """Remove known environment secrets and common credential forms without logging values."""

    def __init__(self, secret_values: list[str]) -> None:
        self.secrets = sorted({value for value in secret_values if len(value) >= 6}, key=len, reverse=True)
        self.counts: Counter[str] = Counter()

    def clean(self, text: str) -> str:
        for value in self.secrets:
            count = text.count(value)
            if count:
                self.counts["known_secret"] += count
                text = text.replace(value, "[REDACTED_CREDENTIAL]")
        patterns = {
            "api_token": r"\b(?:gsk_|sk-|AIza|ghp_|github_pat_)[A-Za-z0-9_-]{12,}",
            "bearer_token": r"(?i)\bBearer\s+[A-Za-z0-9_.-]{20,}",
            "private_key": r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z]+ )?PRIVATE KEY-----",
            "embedded_media": r"data:(?:image|audio|video)/[\w.+-]+;base64,[A-Za-z0-9+/=]+",
        }
        for kind, pattern in patterns.items():
            marker = "[BINARY_MEDIA_OMITTED]" if kind == "embedded_media" else "[REDACTED_CREDENTIAL]"
            text, count = re.subn(pattern, marker, text)
            self.counts[kind] += count
        text, count = re.subn(r"(\b[a-z][a-z0-9+.-]*://[^\s/:@]+:)[^\s/@]+(@)",
                              r"\1[REDACTED_CREDENTIAL]\2", text, flags=re.I)
        self.counts["url_password"] += count
        text, count = re.subn(
            r"(?m)^(\s*[A-Z0-9_]*(?:API_KEY|PASSWORD|SECRET|ACCESS_TOKEN|AUTH_TOKEN)\s*=)[^\r\n]+",
            r"\1[REDACTED_CREDENTIAL]", text,
        )
        self.counts["credential_assignment"] += count
        return text


def public_record(event: dict[str, Any], line_number: int) -> dict[str, Any] | None:
    """Allowlist public response records; exclude private and synthetic runtime records."""
    if event.get("type") != "response_item":
        return None
    payload = event.get("payload", {})
    kind = payload.get("type")
    record: dict[str, Any] = {"timestamp": event.get("timestamp"), "source_line": line_number}
    if kind == "message":
        role = payload.get("role")
        if role not in {"user", "assistant"} or payload.get("channel") in {"analysis", "summary"}:
            return None
        parts = payload.get("content", [])
        body = "\n".join(part.get("text", "[NON_TEXT_ATTACHMENT_REFERENCE]") for part in parts)
        if role == "user" and body.lstrip().startswith(("<recommended_plugins>", "# AGENTS.md instructions", "<environment_context>")):
            return None
        record.update(kind=role, text=body)
    elif kind in {"function_call", "custom_tool_call"}:
        record.update(kind="tool_call", name=payload.get("name"), call_id=payload.get("call_id"),
                      text=payload.get("arguments", payload.get("input", "")))
    elif kind in {"function_call_output", "custom_tool_call_output"}:
        output = payload.get("output", "")
        record.update(kind="tool_result", call_id=payload.get("call_id"),
                      text=output if isinstance(output, str) else json.dumps(output, ensure_ascii=False))
    else:
        return None
    return record


def fenced(text: str) -> str:
    """Render transcript content literally, including nested Markdown and untrusted links."""
    runs = re.findall(r"~+", text)
    fence = "~" * max(3, max((len(run) + 1 for run in runs), default=3))
    return f"{fence}text\n{text}\n{fence}"


def export_source(source: Path, output: Path, project_root: Path, redactor: Redactor) -> dict[str, Any]:
    """Snapshot one explicitly selected project session and preserve recorded ordering."""
    raw = source.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    metadata = json.loads(lines[0]).get("payload", {})
    if Path(metadata.get("cwd", "")).resolve() != project_root.resolve():
        raise ValueError("Source session is not associated with this project")
    records = []
    omitted: Counter[str] = Counter()
    for number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if number != len(lines):
                raise
            omitted["incomplete_trailing_record"] += 1
            continue
        record = public_record(event, number)
        if record is None:
            omitted[event.get("type", "unknown")] += 1
            continue
        if record["kind"] == "user" and record["text"].startswith(
            "The following is the Codex agent history"
        ):
            raise ValueError("Source is a runtime review session, not a public project conversation")
        record["text"] = redactor.clean(record["text"])
        records.append(record)
    session_id = str(metadata["id"])
    name = f"codex-{session_id}"
    jsonl = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    (output / f"{name}.jsonl").write_text(jsonl, encoding="utf-8", newline="\n")
    markdown = [f"# Codex transcript: {session_id}", "",
                "Recorded user/assistant messages and tool calls/results, in source order. "
                "Credential redactions are marked. See the index for export scope and limitations.", ""]
    for index, record in enumerate(records, 1):
        markdown.extend([f"## {index}. {record['kind']} — {record['timestamp']}", "",
                         f"Source record: {record['source_line']}", ""])
        if record.get("name"):
            markdown.extend([f"Tool: `{record['name']}`", ""])
        if record.get("call_id"):
            markdown.extend([f"Call ID: `{record['call_id']}`", ""])
        markdown.extend([fenced(record["text"]), ""])
    md = "\n".join(markdown)
    (output / f"{name}.md").write_text(md, encoding="utf-8", newline="\n")
    return {"session_id": session_id, "source_filename": source.name,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "first_record": records[0]["timestamp"] if records else None,
            "last_record": records[-1]["timestamp"] if records else None,
            "record_counts": dict(Counter(record["kind"] for record in records)),
            "omitted_runtime_records": dict(omitted),
            "files": {f"{name}.md": hashlib.sha256(md.encode()).hexdigest(),
                      f"{name}.jsonl": hashlib.sha256(jsonl.encode()).hexdigest()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/ai-transcripts"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    options = ExportOptions(sources=args.source, output=args.output, project_root=args.project_root)
    secrets = [value for key, value in dotenv_values(options.project_root / ".env").items()
               if value and re.search(r"KEY|PASSWORD|SECRET|TOKEN", key, re.I)]
    redactor = Redactor(secrets)
    options.output.mkdir(parents=True, exist_ok=True)
    sessions = [export_source(source, options.output, options.project_root, redactor) for source in options.sources]
    manifest = {"exported_at_utc": datetime.now(UTC).isoformat(),
                "format": "filtered public conversation and tool text, version 1",
                "scope": "Explicitly selected local project sessions through their saved-record cutoff; not a claim of all project AI usage.",
                "sessions": sessions, "redactions": dict(redactor.counts)}
    (options.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exported_sessions": len(sessions), "redaction_counts": dict(redactor.counts)}))


if __name__ == "__main__":
    main()
