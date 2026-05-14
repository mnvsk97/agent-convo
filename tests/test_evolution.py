from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_convo.config import load_config
from agent_convo.evolution import (
    EvolutionNotConfiguredError,
    build_evolution_prompt,
    evolve_tester_agent,
)


def write_config(tmp_path: Path, *, evolution: bool = True) -> Path:
    section = (
        """
tester-evolution:
  agent: codex
  output_dir: ./tester-evolution
  name: evolve-tester
  budget: 2.0
  extra_instructions: Keep edits focused on tester behavior.
"""
        if evolution
        else ""
    )
    path = tmp_path / "conversation.yaml"
    path.write_text(
        f"""
name: test
tester:
  model: fake:tester
  system_prompt: Tester prompt.
  skills:
    - skills/tester/probe
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
        max_turns: 2
{section}
"""
    )
    skill_dir = tmp_path / "skills" / "tester" / "probe"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Probe\n\nAsk for specifics.")
    return path


def test_build_evolution_prompt_mentions_latest_run_and_tester_surface(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    run_dir = tmp_path / "runs" / "run-1"

    prompt = build_evolution_prompt(config, run_dir=run_dir)

    assert str(run_dir.resolve()) in prompt
    assert "Tester prompt." in prompt
    assert "skills/tester/probe" in prompt
    assert "buyer/pricing" in prompt
    assert "system prompt or tester skills" in prompt


def test_evolve_tester_agent_invokes_harnessctl_and_writes_artifacts(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="evolved", stderr="")

    output_dir = evolve_tester_agent(config, run_dir=tmp_path / "runs" / "run-1", command_runner=fake_runner)

    command, kwargs = calls[0]
    assert command[:4] == ["harnessctl", "run", "--agent", "codex"]
    assert "--budget" in command
    assert kwargs["cwd"] == str(tmp_path)
    assert (output_dir / "prompt.md").exists()
    assert "evolved" in (output_dir / "harnessctl-result.md").read_text()


def test_evolve_tester_agent_requires_yaml_section(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, evolution=False))

    with pytest.raises(EvolutionNotConfiguredError):
        evolve_tester_agent(config, run_dir=tmp_path / "runs" / "run-1")
