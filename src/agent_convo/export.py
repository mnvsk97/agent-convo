from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from agent_convo.storage import read_jsonl


def collect_messages(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transcript_path in sorted((run_dir / "conversations").glob("*/transcript.jsonl")):
        conversation_id = transcript_path.parent.name
        for message in read_jsonl(transcript_path):
            rows.append({"conversation_id": conversation_id, **message})
    return rows


def export_run(run_dir: Path, *, fmt: str, out: Path) -> None:
    rows = collect_messages(run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        return
    if fmt == "json":
        out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        return
    if fmt == "csv":
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["conversation_id", "turn", "agent", "content", "ts"])
            writer.writeheader()
            writer.writerows(rows)
        return
    if fmt == "md":
        lines = ["# Agent Convo Export", ""]
        current = None
        for row in rows:
            if row["conversation_id"] != current:
                current = row["conversation_id"]
                lines.extend([f"## Conversation {current}", ""])
            lines.extend([f"### Turn {row['turn']} - {row['agent']}", "", row["content"], ""])
        out.write_text("\n".join(lines).rstrip() + "\n")
        return
    raise ValueError(f"Unsupported export format: {fmt}")
