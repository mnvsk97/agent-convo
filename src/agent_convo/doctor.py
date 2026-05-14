from __future__ import annotations

import os
from dataclasses import dataclass

from agent_convo.config import AppConfig


@dataclass(frozen=True)
class DoctorResult:
    ok: bool
    messages: list[str]


def check_model(
    name: str,
    model: str,
    messages: list[str],
    *,
    api_key_env: str | None = None,
    base_url_env: str | None = None,
) -> bool:
    if model.startswith("fake:"):
        messages.append(f"{name}: fake model configured for local deterministic runs")
        return True
    ok = True
    if base_url_env and not os.getenv(base_url_env):
        messages.append(f"{name}: {base_url_env} is required for model {model}")
        ok = False
    if api_key_env and not os.getenv(api_key_env):
        messages.append(f"{name}: {api_key_env} is required for model {model}")
        ok = False
    if base_url_env or api_key_env:
        if ok:
            messages.append(f"{name}: OpenAI-compatible env readiness looks configured for {model}")
        return ok
    provider = model.split(":", 1)[0]
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        messages.append(f"{name}: OPENAI_API_KEY is required for model {model}")
        return False
    messages.append(f"{name}: provider readiness looks configured for {model}")
    return True


def check_config(config: AppConfig) -> DoctorResult:
    messages: list[str] = []
    ok = True
    ok = check_model(
        "tester",
        config.tester.model,
        messages,
        api_key_env=config.tester.api_key_env,
        base_url_env=config.tester.base_url_env,
    ) and ok
    ok = check_model(
        "observer",
        config.observer.model,
        messages,
        api_key_env=config.observer.api_key_env,
        base_url_env=config.observer.base_url_env,
    ) and ok
    ok = check_model(
        "grader",
        config.grader.model,
        messages,
        api_key_env=config.grader.api_key_env,
        base_url_env=config.grader.base_url_env,
    ) and ok
    if config.target.model.startswith("fake:"):
        messages.append("target: fake model configured for local deterministic runs")
    elif config.target.base_url_env and not os.getenv(config.target.base_url_env):
        ok = False
        messages.append(f"target: {config.target.base_url_env} is required for model {config.target.model}")
    elif config.target.api_key_env and not os.getenv(config.target.api_key_env):
        ok = False
        messages.append(f"target: {config.target.api_key_env} is required for model {config.target.model}")
    else:
        messages.append(f"target: OpenAI-compatible endpoint configured for {config.target.model}")
    return DoctorResult(ok=ok, messages=messages)
