from __future__ import annotations

import os
from dataclasses import dataclass

from agent_convo.config import AppConfig


@dataclass(frozen=True)
class DoctorResult:
    ok: bool
    messages: list[str]


def check_model(name: str, model: str, messages: list[str]) -> bool:
    if model.startswith("fake:"):
        messages.append(f"{name}: fake model configured for local deterministic runs")
        return True
    provider = model.split(":", 1)[0]
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        messages.append(f"{name}: OPENAI_API_KEY is required for model {model}")
        return False
    messages.append(f"{name}: provider readiness looks configured for {model}")
    return True


def check_config(config: AppConfig) -> DoctorResult:
    messages: list[str] = []
    ok = True
    ok = check_model("tester", config.tester.model, messages) and ok
    ok = check_model("observer", config.observer.model, messages) and ok
    ok = check_model("grader", config.grader.model, messages) and ok
    if config.target.model.startswith("fake:"):
        messages.append("target: fake model configured for local deterministic runs")
    elif config.target.api_key_env and not os.getenv(config.target.api_key_env):
        ok = False
        messages.append(f"target: {config.target.api_key_env} is required for model {config.target.model}")
    else:
        messages.append(f"target: OpenAI-compatible endpoint configured for {config.target.model}")
    return DoctorResult(ok=ok, messages=messages)
