from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_convo.config import AgentConfig, load_config
from agent_convo.langchain_factory import build_agent, compile_system_prompt


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "conversation.yaml"
    path.write_text(body)
    return path


def valid_body(tmp_path: Path) -> str:
    skill_dir = tmp_path / "skills" / "tester" / "probe"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Probe\n\nAsk for specifics.")
    return f"""
name: test
tester:
  model: fake:tester
  system_prompt: Tester prompt.
  skills:
    - skills/tester/probe
target:
  type: openai_compatible
  model: fake:target
  system_prompt: Target prompt.
observer:
  model: fake:observer
grader:
  model: fake:grader
personas:
  - id: buyer
    name: Buyer
    scenarios:
      - id: pricing
        goal: Check pricing.
        opening_message: What does it cost?
        max_turns: 4
run:
  count: 1
  parallelism: 1
  output_dir: {tmp_path / "runs"}
"""


def test_config_loads_valid_yaml(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, valid_body(tmp_path)))

    assert config.name == "test"
    assert config.tester.model == "fake:tester"
    assert config.scenario_count == 1


def test_config_rejects_missing_personas(tmp_path: Path) -> None:
    body = """
name: test
tester:
  model: fake:tester
target:
  type: openai_compatible
  model: fake:target
personas: []
"""

    with pytest.raises(ValueError, match="persona"):
        load_config(write_config(tmp_path, body))


def test_config_rejects_missing_skill_paths(tmp_path: Path) -> None:
    body = valid_body(tmp_path).replace("skills/tester/probe", "skills/tester/missing")

    with pytest.raises(ValueError, match="SKILL.md"):
        load_config(write_config(tmp_path, body))


def test_config_accepts_mcp_servers(tmp_path: Path) -> None:
    body = valid_body(tmp_path).replace(
        "model: fake:tester",
        "model: fake:tester\n  mcp_servers:\n    - name: demo\n      transport: stdio\n      command: python\n      args: [server.py]",
    )

    config = load_config(write_config(tmp_path, body))

    assert config.tester.mcp_servers[0].name == "demo"


def test_config_loads_tester_evolution_section(tmp_path: Path) -> None:
    body = valid_body(tmp_path) + """
tester-evolution:
  agent: codex
  output_dir: ./tester-evolution
  name: evolve-tester
  budget: 1.5
  extra_instructions: Keep edits small.
"""

    config = load_config(write_config(tmp_path, body))

    assert config.tester_evolution is not None
    assert config.tester_evolution.agent == "codex"
    assert config.tester_evolution.budget == 1.5


def test_factory_compiles_prompt_with_skill_persona_and_scenario(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, valid_body(tmp_path)))
    persona = config.personas[0]
    scenario = persona.scenarios[0]

    prompt = compile_system_prompt(config.tester, config.base_dir, persona=persona, scenario=scenario)

    assert "Tester prompt." in prompt
    assert "Agent Skill" in prompt
    assert "Ask for specifics." in prompt
    assert "Persona: Buyer" in prompt
    assert "Scenario goal: Check pricing." in prompt


def test_factory_passes_model_prompt_tools_and_middleware(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    tool = object()
    middleware = object()

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "agent"

    def fake_import(path: str):
        return {"demo:tool": tool, "demo:middleware": middleware}[path]

    monkeypatch.setattr("agent_convo.langchain_factory.create_agent", fake_create_agent)
    monkeypatch.setattr("agent_convo.langchain_factory.import_object", fake_import)
    agent_config = AgentConfig(
        model="openai:gpt-test",
        system_prompt="Prompt.",
        tools=["demo:tool"],
        middleware=["demo:middleware"],
    )

    result = asyncio.run(build_agent("tester", agent_config, tmp_path))

    assert result == "agent"
    assert captured["model"] == "openai:gpt-test"
    assert captured["system_prompt"] == "Prompt."
    assert captured["tools"] == [tool]
    assert captured["middleware"] == [middleware]
