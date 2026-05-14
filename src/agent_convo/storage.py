from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_transcript_views(conversation_dir: Path) -> None:
    transcript = read_jsonl(conversation_dir / "transcript.jsonl")
    (conversation_dir / "transcript.json").write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n")
    lines = [f"# Conversation {conversation_dir.name}", ""]
    for message in transcript:
        lines.append(f"## Turn {message['turn']} - {message['agent']}")
        lines.append("")
        lines.append(message["content"])
        lines.append("")
    (conversation_dir / "transcript.md").write_text("\n".join(lines).rstrip() + "\n")


def initialize_run(run_dir: Path, manifest: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "manifest.json", manifest)


def conversation_dir(run_dir: Path, index: int) -> Path:
    return run_dir / "conversations" / f"{index:06d}"

