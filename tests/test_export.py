from __future__ import annotations

import json
from pathlib import Path

from agent_convo.export import export_run


def make_run(tmp_path: Path) -> Path:
    convo_dir = tmp_path / "runs" / "run-1" / "conversations" / "000001"
    convo_dir.mkdir(parents=True)
    (convo_dir / "transcript.jsonl").write_text(
        '{"agent":"tester","content":"hello","ts":"now","turn":1}\n'
    )
    return tmp_path / "runs" / "run-1"


def test_export_produces_jsonl_json_csv_and_markdown(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)

    for fmt in ["jsonl", "json", "csv", "md"]:
        out = tmp_path / f"out.{fmt}"
        export_run(run_dir, fmt=fmt, out=out)
        assert out.exists()
        assert "hello" in out.read_text()

    assert json.loads((tmp_path / "out.json").read_text())[0]["conversation_id"] == "000001"
