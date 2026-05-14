from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

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


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


async def invoke_json_model(
    *,
    model: str,
    name: str,
    system_prompt: str,
    user_prompt: str,
    base_url: str | None,
    base_url_env: str | None,
    api_key_env: str | None,
) -> dict[str, Any]:
    llm = model_from_config(
        model,
        name,
        base_url=base_url,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if hasattr(llm, "ainvoke"):
        result = await llm.ainvoke(messages)
    else:
        result = await asyncio.to_thread(llm.invoke, messages)
    return parse_json_object(final_content(result))


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
            (
                'Return JSON only with shape: {"decision":"continue|halt_success|halt_failure",'
                '"feedback":"short feedback for the tester"}'
            ),
        ]
    )
    raw = await asyncio.wait_for(
        invoke_json_model(
            model=config.model,
            name="observer",
            system_prompt=config.system_prompt,
            user_prompt=prompt,
            base_url=config.base_url,
            base_url_env=config.base_url_env,
            api_key_env=config.api_key_env,
        ),
        timeout=60,
    )
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
            'Return JSON only with shape: {"result":"pass|fail","rationale":"concise evidence"}.',
        ]
    )
    raw = await asyncio.wait_for(
        invoke_json_model(
            model=config.model,
            name="grader",
            system_prompt=config.system_prompt,
            user_prompt=prompt,
            base_url=config.base_url,
            base_url_env=config.base_url_env,
            api_key_env=config.api_key_env,
        ),
        timeout=60,
    )
    return raw if isinstance(raw, GradeResult) else GradeResult.model_validate(raw)
