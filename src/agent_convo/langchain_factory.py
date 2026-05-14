from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI

from agent_convo.config import AgentConfig, MCPServerConfig, PersonaConfig, ScenarioConfig, TargetConfig


def compile_system_prompt(
    agent: AgentConfig,
    base_dir: Path,
    *,
    persona: PersonaConfig | None = None,
    scenario: ScenarioConfig | None = None,
) -> str:
    parts = [agent.system_prompt.strip()]
    for skill_path in agent.skills:
        skill_file = (base_dir / skill_path).resolve() / "SKILL.md"
        parts.append(f"Agent Skill: {skill_path}\n\n{skill_file.read_text().strip()}")
    if persona is not None:
        parts.append(
            "\n".join(
                [
                    f"Persona: {persona.name}",
                    persona.description,
                    persona.custom_instructions,
                ]
            ).strip()
        )
    if scenario is not None:
        parts.append(
            "\n".join(
                [
                    f"Scenario goal: {scenario.goal}",
                    "Stay focused on this scenario and do not reveal the grading rubric.",
                ]
            )
        )
    return "\n\n".join(part for part in parts if part)


def import_object(dotted_path: str) -> Any:
    module_name, _, attr = dotted_path.partition(":")
    if not attr:
        module_name, _, attr = dotted_path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Expected import path like 'module:object', got '{dotted_path}'")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def load_tools(agent: AgentConfig) -> list[Any]:
    return [import_object(path) for path in agent.tools]


def load_middleware(agent: AgentConfig) -> list[Any]:
    return [import_object(path) for path in agent.middleware]


def model_from_config(model: str, agent_name: str) -> Any:
    if model.startswith("fake:"):
        label = model.split(":", 1)[1] or agent_name
        return FakeListChatModel(
            responses=[
                f"{label} response: ask for one concrete detail.",
                f"{label} response: give one concise concrete answer.",
                f"{label} response: summarize the remaining uncertainty.",
            ]
        )
    return model


def final_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and result.get("messages"):
        message = result["messages"][-1]
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        return str(content)
    return str(result)


def target_model_from_config(target: TargetConfig) -> Any:
    if target.model.startswith("fake:"):
        return model_from_config(target.model, "target")
    kwargs: dict[str, Any] = {"model": target.model}
    if target.base_url:
        kwargs["base_url"] = target.base_url
    if target.api_key_env:
        api_key = os.getenv(target.api_key_env)
        if not api_key:
            raise ValueError(f"{target.api_key_env} is required for target model {target.model}")
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def mcp_connections(servers: list[MCPServerConfig]) -> dict[str, dict[str, Any]]:
    connections = {}
    for server in servers:
        data = server.model_dump(exclude={"name"}, by_alias=True)
        connections[server.name] = data
    return connections


async def load_mcp_tools(agent: AgentConfig) -> list[Any]:
    if not agent.mcp_servers:
        return []
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError("Install agent-convo[mcp] to use mcp_servers") from exc
    client = MultiServerMCPClient(mcp_connections(agent.mcp_servers))
    return await client.get_tools()


async def build_agent(
    agent_name: str,
    config: AgentConfig,
    base_dir: Path,
    *,
    persona: PersonaConfig | None = None,
    scenario: ScenarioConfig | None = None,
) -> Any:
    tools = [*load_tools(config), *(await load_mcp_tools(config))]
    return create_agent(
        model=model_from_config(config.model, agent_name),
        tools=tools,
        system_prompt=compile_system_prompt(config, base_dir, persona=persona, scenario=scenario),
        middleware=load_middleware(config),
    )


async def build_target_agent(
    config: TargetConfig,
    base_dir: Path,
    *,
    scenario: ScenarioConfig,
) -> Any:
    tools = [*load_tools(config), *(await load_mcp_tools(config))]
    return create_agent(
        model=target_model_from_config(config),
        tools=tools,
        system_prompt=compile_system_prompt(config, base_dir, scenario=scenario),
        middleware=load_middleware(config),
    )
