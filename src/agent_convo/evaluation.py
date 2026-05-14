from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from langchain.agents import create_agent
from pydantic import BaseModel

from agent_convo.config import GraderConfig, ObserverConfig, PersonaConfig, ScenarioConfig
from agent_convo.langchain_factory import final_content, model_from_config


class ObserverDecision(BaseModel):
    decision: Literal["continue", "halt_success", "halt_failure"]
    feedback: str = ""


class GradeResult(BaseModel):
    result: Literal["pass", "fail"]
    rationale: str


def transcript_text(transcript: list[dict[str, Any]]) -> str:
    return "\n".join(f"{row['agent']}: {row['content']}" for row in transcript)


def fake_observer_decision(transcript: list[dict[str, Any]], scenario: ScenarioConfig) -> ObserverDecision:
    if len(transcript) >= scenario.max_turns:
        return ObserverDecision(decision="halt_success", feedback="Max turns reached.")
    target_messages = [row for row in transcript if row["agent"] == "target"]
    if len(target_messages) >= 1 and scenario.logical_completion.halt_when:
        return ObserverDecision(decision="halt_success", feedback="Enough signal collected for local fake run.")
    return ObserverDecision(decision="continue", feedback="Ask one concrete follow-up.")


def fake_grade_result(transcript: list[dict[str, Any]]) -> GradeResult:
    if transcript:
        return GradeResult(result="pass", rationale="Local fake run produced a transcript to grade.")
    return GradeResult(result="fail", rationale="No transcript messages were produced.")


async def structured_invoke(agent: Any, prompt: str) -> Any:
    result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
    if isinstance(result, dict) and result.get("structured_response") is not None:
        return result["structured_response"]
    content = final_content(result)
    return json.loads(content)


async def observe(
    config: ObserverConfig,
    *,
    persona: PersonaConfig,
    scenario: ScenarioConfig,
    transcript: list[dict[str, Any]],
) -> ObserverDecision:
    if config.model.startswith("fake:"):
        return fake_observer_decision(transcript, scenario)
    prompt = "\n\n".join(
        [
            f"Persona: {persona.name}",
            f"Scenario goal: {scenario.goal}",
            "Logical completion rules:",
            json.dumps(scenario.logical_completion.model_dump(), indent=2),
            "Transcript:",
            transcript_text(transcript),
            "Decide whether to continue, halt_success, or halt_failure.",
        ]
    )
    agent = create_agent(
        model=model_from_config(config.model, "observer"),
        tools=[],
        system_prompt=config.system_prompt,
        response_format=ObserverDecision,
    )
    raw = await asyncio.wait_for(structured_invoke(agent, prompt), timeout=60)
    return raw if isinstance(raw, ObserverDecision) else ObserverDecision.model_validate(raw)


async def grade(
    config: GraderConfig,
    *,
    persona: PersonaConfig,
    scenario: ScenarioConfig,
    transcript: list[dict[str, Any]],
) -> GradeResult:
    if config.model.startswith("fake:"):
        return fake_grade_result(transcript)
    prompt = "\n\n".join(
        [
            f"Persona: {persona.name}",
            f"Scenario goal: {scenario.goal}",
            "Grade rules:",
            json.dumps(scenario.grades.model_dump(by_alias=True), indent=2),
            "Transcript:",
            transcript_text(transcript),
            "Return pass or fail with concise evidence.",
        ]
    )
    agent = create_agent(
        model=model_from_config(config.model, "grader"),
        tools=[],
        system_prompt=config.system_prompt,
        response_format=GradeResult,
    )
    raw = await asyncio.wait_for(structured_invoke(agent, prompt), timeout=60)
    return raw if isinstance(raw, GradeResult) else GradeResult.model_validate(raw)
