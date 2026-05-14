from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_convo.config import AppConfig, PersonaConfig, ScenarioConfig
from agent_convo.evaluation import grade, observe
from agent_convo.langchain_factory import build_agent, build_target_agent, final_content
from agent_convo.storage import (
    append_jsonl,
    atomic_write_json,
    conversation_dir,
    initialize_run,
    read_jsonl,
    utc_now,
    write_transcript_views,
)


@dataclass(frozen=True)
class ConversationCase:
    index: int
    repetition: int
    persona: PersonaConfig
    scenario: ScenarioConfig


def conversation_cases(config: AppConfig) -> list[ConversationCase]:
    cases = []
    index = 1
    for repetition in range(1, config.run.count + 1):
        for persona in config.personas:
            for scenario in persona.scenarios:
                cases.append(ConversationCase(index, repetition, persona, scenario))
                index += 1
    return cases


async def invoke_agent(agent: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(agent, "ainvoke"):
        result = await agent.ainvoke({"messages": messages})
    else:
        result = await asyncio.to_thread(agent.invoke, {"messages": messages})
    return final_content(result)


async def resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def messages_for_agent(
    transcript: list[dict[str, Any]],
    *,
    agent_name: str,
    observer_feedback: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "assistant" if item["agent"] == agent_name else "user", "content": item["content"]}
        for item in transcript
    ]
    if observer_feedback and agent_name == "tester":
        messages.append(
            {
                "role": "user",
                "content": f"Observer feedback for your next turn: {observer_feedback}",
            }
        )
    return messages or [{"role": "user", "content": "Start the conversation."}]


class ConversationRunner:
    def __init__(self, config: AppConfig):
        self.config = config

    async def start_run(self) -> Path:
        run_id = utc_now().replace(":", "").replace(".", "") + "-" + uuid.uuid4().hex[:8]
        run_dir = self.config.resolve_path(self.config.run.output_dir) / run_id
        cases = conversation_cases(self.config)
        initialize_run(
            run_dir,
            {
                "run_id": run_id,
                "name": self.config.name,
                "count": self.config.run.count,
                "parallelism": self.config.run.parallelism,
                "total_conversations": len(cases),
                "created_at": utc_now(),
            },
        )
        await self.run_batch(run_dir, cases=cases, resume=False)
        return run_dir

    async def resume_run(self, run_dir: Path) -> None:
        cases = conversation_cases(self.config)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if int(manifest.get("total_conversations", len(cases))) != len(cases):
                raise ValueError("config no longer matches the run manifest conversation count")
        await self.run_batch(run_dir.resolve(), cases=cases, resume=True)

    async def run_batch(self, run_dir: Path, *, cases: list[ConversationCase], resume: bool) -> None:
        semaphore = asyncio.Semaphore(self.config.run.parallelism)

        async def limited(case: ConversationCase) -> None:
            async with semaphore:
                await self.run_conversation(run_dir, case, resume=resume)

        await asyncio.gather(*(limited(case) for case in cases))

    async def run_conversation(self, run_dir: Path, case: ConversationCase, *, resume: bool) -> None:
        convo_dir = conversation_dir(run_dir, case.index)
        transcript_path = convo_dir / "transcript.jsonl"
        state_path = convo_dir / "state.json"
        grade_path = convo_dir / "grade.json"
        transcript = read_jsonl(transcript_path)

        if resume and grade_path.exists():
            return

        convo_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            convo_dir / "metadata.json",
            {
                "conversation_id": convo_dir.name,
                "repetition": case.repetition,
                "persona_id": case.persona.id,
                "persona_name": case.persona.name,
                "scenario_id": case.scenario.id,
                "scenario_goal": case.scenario.goal,
                "max_turns": case.scenario.max_turns,
            },
        )

        tester = await resolve(
            build_agent(
                "tester",
                self.config.tester,
                self.config.base_dir,
                persona=case.persona,
                scenario=case.scenario,
            )
        )
        target = await resolve(build_target_agent(self.config.target, self.config.base_dir, scenario=case.scenario))
        agents = {"tester": tester, "target": target}

        turn_index = len(transcript)
        next_agent = "tester" if turn_index % 2 == 0 else "target"
        observer_feedback: str | None = None
        if not transcript:
            await self.record_message(
                convo_dir,
                state_path,
                transcript,
                {
                    "turn": 1,
                    "agent": "tester",
                    "content": case.scenario.opening_message,
                    "ts": utc_now(),
                },
                status="running",
                next_agent="target",
                max_turns=case.scenario.max_turns,
            )
            turn_index = 1
            next_agent = "target"

        while turn_index < case.scenario.max_turns:
            agent_name = next_agent
            attempt = 0
            while True:
                append_jsonl(
                    convo_dir / "events.jsonl",
                    {
                        "event": "turn_start",
                        "agent": agent_name,
                        "turn": turn_index + 1,
                        "attempt": attempt,
                        "ts": utc_now(),
                    },
                )
                try:
                    content = await asyncio.wait_for(
                        invoke_agent(
                            agents[agent_name],
                            messages_for_agent(
                                transcript,
                                agent_name=agent_name,
                                observer_feedback=observer_feedback,
                            ),
                        ),
                        timeout=self.config.run.per_turn_timeout_seconds,
                    )
                    observer_feedback = None
                    message = {
                        "turn": turn_index + 1,
                        "agent": agent_name,
                        "content": content,
                        "ts": utc_now(),
                    }
                    turn_index += 1
                    next_agent = "tester" if agent_name == "target" else "target"
                    await self.record_message(
                        convo_dir,
                        state_path,
                        transcript,
                        message,
                        status="running",
                        next_agent=next_agent,
                        max_turns=case.scenario.max_turns,
                        attempt=attempt,
                    )
                    append_jsonl(
                        convo_dir / "events.jsonl",
                        {
                            "event": "turn_success",
                            "agent": agent_name,
                            "turn": turn_index,
                            "attempt": attempt,
                            "ts": utc_now(),
                        },
                    )
                    if agent_name == "target" and self.config.observer.check_after_each_target_turn:
                        decision = await observe(
                            self.config.observer,
                            persona=case.persona,
                            scenario=case.scenario,
                            transcript=transcript,
                        )
                        append_jsonl(
                            convo_dir / "events.jsonl",
                            {
                                "event": "observer_decision",
                                "decision": decision.decision,
                                "feedback": decision.feedback,
                                "turn": turn_index,
                                "ts": utc_now(),
                            },
                        )
                        if decision.decision != "continue":
                            await self.finish_conversation(convo_dir, state_path, transcript, case)
                            return
                        observer_feedback = decision.feedback
                    break
                except Exception as exc:  # noqa: BLE001 - provider failures must be durable.
                    append_jsonl(
                        convo_dir / "events.jsonl",
                        {
                            "event": "turn_error",
                            "agent": agent_name,
                            "turn": turn_index + 1,
                            "attempt": attempt,
                            "error": repr(exc),
                            "ts": utc_now(),
                        },
                    )
                    if attempt >= self.config.run.max_retries_per_turn:
                        atomic_write_json(
                            state_path,
                            {
                                "conversation_id": convo_dir.name,
                                "status": "failed",
                                "turn": turn_index,
                                "next_agent": agent_name,
                                "max_turns": case.scenario.max_turns,
                                "attempt": attempt,
                                "updated_at": utc_now(),
                                "error": repr(exc),
                            },
                        )
                        write_transcript_views(convo_dir)
                        return
                    attempt += 1

        await self.finish_conversation(convo_dir, state_path, transcript, case)

    async def record_message(
        self,
        convo_dir: Path,
        state_path: Path,
        transcript: list[dict[str, Any]],
        message: dict[str, Any],
        *,
        status: str,
        next_agent: str,
        max_turns: int,
        attempt: int = 0,
    ) -> None:
        append_jsonl(convo_dir / "transcript.jsonl", message)
        transcript.append(message)
        atomic_write_json(
            state_path,
            {
                "conversation_id": convo_dir.name,
                "status": status,
                "turn": len(transcript),
                "next_agent": next_agent,
                "max_turns": max_turns,
                "attempt": attempt,
                "updated_at": utc_now(),
            },
        )

    async def finish_conversation(
        self,
        convo_dir: Path,
        state_path: Path,
        transcript: list[dict[str, Any]],
        case: ConversationCase,
    ) -> None:
        result = await grade(
            self.config.grader,
            persona=case.persona,
            scenario=case.scenario,
            transcript=transcript,
        )
        atomic_write_json(convo_dir / "grade.json", result.model_dump())
        atomic_write_json(
            state_path,
            {
                "conversation_id": convo_dir.name,
                "status": "completed",
                "turn": len(transcript),
                "next_agent": None,
                "max_turns": case.scenario.max_turns,
                "grade": result.result,
                "updated_at": utc_now(),
            },
        )
        write_transcript_views(convo_dir)


async def run_new(config: AppConfig) -> Path:
    return await ConversationRunner(config).start_run()


async def resume_existing(config: AppConfig, run_dir: Path) -> None:
    await ConversationRunner(config).resume_run(run_dir)
