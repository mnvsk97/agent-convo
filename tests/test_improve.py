from __future__ import annotations

from pathlib import Path

from agent_convo.config import load_config
from agent_convo.improve import improve_agent


def write_config(tmp_path: Path, output_dir: str | None = None) -> tuple[Path, str]:
    improve_dir = output_dir or str(tmp_path / "improvements")
    body = f"""
name: test
tester:
  model: fake:tester
  system_prompt: Tester.
target:
  type: openai_compatible
  model: fake:target
personas:
  - id: buyer
    name: Buyer
    scenarios:
      - id: pricing
        goal: Check pricing.
        opening_message: What is pricing?
        max_turns: 1
improve:
  output_dir: {improve_dir}
"""
    config_path = tmp_path / "conversation.yaml"
    config_path.write_text(body)
    return config_path, body


def test_improve_writes_artifacts_without_modifying_original_config(tmp_path: Path) -> None:
    config_path, original = write_config(tmp_path)
    run_dir = tmp_path / "runs" / "run-1"
    convo_dir = run_dir / "conversations" / "000001"
    convo_dir.mkdir(parents=True)
    (convo_dir / "transcript.jsonl").write_text(
        '{"agent":"tester","content":"what is pricing?","ts":"now","turn":1}\n'
    )
    config = load_config(config_path, validate_paths=False)

    output_dir = improve_agent(config, run_dir=run_dir, agent_name="tester")

    assert (output_dir / "improve-report.md").exists()
    assert (output_dir / "suggested-agent.yaml").exists()
    assert (output_dir / "skills" / "tester" / "probe-vague-claims" / "SKILL.md").exists()
    assert config_path.read_text() == original


def test_improve_output_dir_is_relative_to_config_file(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config_path, _ = write_config(project_dir, "./improvements")
    run_dir = tmp_path / "runs" / "run-1"
    (run_dir / "conversations" / "000001").mkdir(parents=True)
    config = load_config(config_path, validate_paths=False)

    output_dir = improve_agent(config, run_dir=run_dir, agent_name="tester")

    assert output_dir.parent == project_dir / "improvements"
