from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_convo.config import load_config
from agent_convo.runner import ConversationRunner, conversation_cases, run_new
from agent_convo.storage import read_jsonl


class ScriptedAgent:
    def __init__(self, name: str, *, fail: bool = False, delay: float = 0):
        self.name = name
        self.fail = fail
        self.delay = delay
        self.calls = 0

    async def ainvoke(self, payload):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return {"messages": [type("Message", (), {"content": f"{self.name}-{self.calls}"})()]}


def write_config(
    tmp_path: Path,
    *,
    count: int = 1,
    parallelism: int = 1,
    max_turns: int = 4,
    scenarios: int = 1,
) -> Path:
    scenario_blocks = []
    for index in range(1, scenarios + 1):
        scenario_blocks.append(
            f"""
      - id: scenario_{index}
        goal: Goal {index}.
        opening_message: Opening {index}
        max_turns: {max_turns}
        logical_completion:
          halt_when:
            - target gives enough signal
        grades:
          pass:
            - transcript has signal
          fail:
            - transcript has no signal
"""
        )
    path = tmp_path / "conversation.yaml"
    path.write_text(
        f"""
name: test
tester:
  model: fake:tester
  system_prompt: Tester.
target:
  type: openai_compatible
  model: fake:target
  system_prompt: Target.
observer:
  model: fake:observer
grader:
  model: fake:grader
personas:
  - id: buyer
    name: Buyer
    scenarios:
{''.join(scenario_blocks)}
run:
  count: {count}
  parallelism: {parallelism}
  output_dir: {tmp_path / "runs"}
  per_turn_timeout_seconds: 1
  max_retries_per_turn: 1
"""
    )
    return path


def test_cases_expand_by_repetition_persona_and_scenario(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, count=2, scenarios=2), validate_paths=False)

    cases = conversation_cases(config)

    assert len(cases) == 4
    assert [case.scenario.id for case in cases] == ["scenario_1", "scenario_2", "scenario_1", "scenario_2"]


def test_runner_uses_opening_then_target_and_grades(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_convo.runner.build_agent",
        lambda *_args, **_kwargs: ScriptedAgent("tester"),
    )
    monkeypatch.setattr(
        "agent_convo.runner.build_target_agent",
        lambda *_args, **_kwargs: ScriptedAgent("target"),
    )
    config = load_config(write_config(tmp_path, max_turns=4), validate_paths=False)

    run_dir = asyncio.run(run_new(config))
    convo_dir = run_dir / "conversations" / "000001"
    transcript = read_jsonl(convo_dir / "transcript.jsonl")
    grade = json.loads((convo_dir / "grade.json").read_text())

    assert [row["agent"] for row in transcript] == ["tester", "target"]
    assert transcript[0]["content"] == "Opening 1"
    assert grade["result"] == "pass"


def test_runner_emits_message_and_grade_events(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_convo.runner.build_agent",
        lambda *_args, **_kwargs: ScriptedAgent("tester"),
    )
    monkeypatch.setattr(
        "agent_convo.runner.build_target_agent",
        lambda *_args, **_kwargs: ScriptedAgent("target"),
    )
    events = []
    config = load_config(write_config(tmp_path, max_turns=4), validate_paths=False)

    asyncio.run(run_new(config, event_handler=events.append))

    assert [(event["type"], event.get("agent")) for event in events] == [
        ("message", "tester"),
        ("message", "target"),
        ("grade", None),
    ]
    assert events[0]["content"] == "Opening 1"
    assert events[1]["content"] == "target-1"
    assert events[2]["result"] == "pass"


def test_parallel_run_creates_isolated_conversation_folders(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_convo.runner.build_agent",
        lambda *_args, **_kwargs: ScriptedAgent("tester", delay=0.01),
    )
    monkeypatch.setattr(
        "agent_convo.runner.build_target_agent",
        lambda *_args, **_kwargs: ScriptedAgent("target", delay=0.01),
    )
    config = load_config(write_config(tmp_path, count=3, parallelism=2, max_turns=2), validate_paths=False)

    run_dir = asyncio.run(run_new(config))

    assert (run_dir / "conversations" / "000001" / "transcript.jsonl").exists()
    assert (run_dir / "conversations" / "000002" / "transcript.jsonl").exists()
    assert (run_dir / "conversations" / "000003" / "transcript.jsonl").exists()


def test_run_output_dir_is_relative_to_config_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_convo.runner.build_agent",
        lambda *_args, **_kwargs: ScriptedAgent("tester"),
    )
    monkeypatch.setattr(
        "agent_convo.runner.build_target_agent",
        lambda *_args, **_kwargs: ScriptedAgent("target"),
    )
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    config_path = write_config(config_dir, count=1, max_turns=1)
    config = load_config(config_path, validate_paths=False)
    monkeypatch.chdir(tmp_path)

    run_dir = asyncio.run(run_new(config))

    assert run_dir.parent == config_dir / "runs"


def test_failed_conversation_does_not_stop_other_conversations(tmp_path: Path, monkeypatch) -> None:
    calls = {"target": 0}

    async def build_target(*_args, **_kwargs):
        calls["target"] += 1
        return ScriptedAgent("target", fail=calls["target"] == 1)

    monkeypatch.setattr("agent_convo.runner.build_agent", lambda *_args, **_kwargs: ScriptedAgent("tester"))
    monkeypatch.setattr("agent_convo.runner.build_target_agent", build_target)
    config = load_config(write_config(tmp_path, count=2, parallelism=1, max_turns=2), validate_paths=False)

    run_dir = asyncio.run(run_new(config))
    states = [
        json.loads(path.read_text())["status"]
        for path in sorted((run_dir / "conversations").glob("*/state.json"))
    ]

    assert states == ["failed", "completed"]


def test_turn_timeout_retries_only_current_turn(tmp_path: Path, monkeypatch) -> None:
    class SlowOnceAgent(ScriptedAgent):
        async def ainvoke(self, payload):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(0.05)
            return {"messages": [type("Message", (), {"content": f"{self.name}-{self.calls}"})()]}

    target = SlowOnceAgent("target")
    monkeypatch.setattr("agent_convo.runner.build_agent", lambda *_args, **_kwargs: ScriptedAgent("tester"))
    monkeypatch.setattr("agent_convo.runner.build_target_agent", lambda *_args, **_kwargs: target)
    config = load_config(write_config(tmp_path, max_turns=2), validate_paths=False)
    config.run.per_turn_timeout_seconds = 0.01

    run_dir = asyncio.run(run_new(config))
    transcript = read_jsonl(run_dir / "conversations" / "000001" / "transcript.jsonl")
    events = read_jsonl(run_dir / "conversations" / "000001" / "events.jsonl")

    assert [row["agent"] for row in transcript] == ["tester", "target"]
    assert [row["event"] for row in events].count("turn_error") == 1
    assert target.calls == 2


def test_resume_skips_graded_conversations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("agent_convo.runner.build_agent", lambda *_args, **_kwargs: ScriptedAgent("tester"))
    monkeypatch.setattr("agent_convo.runner.build_target_agent", lambda *_args, **_kwargs: ScriptedAgent("target"))
    config = load_config(write_config(tmp_path, count=1, max_turns=2), validate_paths=False)
    runner = ConversationRunner(config)
    run_dir = asyncio.run(runner.start_run())
    before = (run_dir / "conversations" / "000001" / "transcript.jsonl").read_text()

    asyncio.run(runner.resume_run(run_dir))
    after = (run_dir / "conversations" / "000001" / "transcript.jsonl").read_text()

    assert after == before
