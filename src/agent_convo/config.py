from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    transport: str = "stdio"


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    system_prompt: str = ""
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    middleware: list[str] = Field(default_factory=list)


class TargetConfig(AgentConfig):
    type: Literal["openai_compatible"] = "openai_compatible"


class ObserverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "fake:observer"
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    system_prompt: str = ""
    check_after_each_target_turn: bool = True


class GraderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "fake:grader"
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    system_prompt: str = ""


class GradeRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pass_: list[str] = Field(default_factory=list, alias="pass")
    fail: list[str] = Field(default_factory=list)


class LogicalCompletionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    halt_when: list[str] = Field(default_factory=list)


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    goal: str
    opening_message: str
    max_turns: int = Field(gt=0)
    logical_completion: LogicalCompletionConfig = Field(default_factory=LogicalCompletionConfig)
    grades: GradeRules = Field(default_factory=GradeRules)


class PersonaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str = ""
    custom_instructions: str = ""
    scenarios: list[ScenarioConfig] = Field(default_factory=list)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=1, gt=0)
    parallelism: int = Field(default=1, gt=0)
    output_dir: str = "./runs"
    per_turn_timeout_seconds: float = Field(default=90, gt=0)
    max_retries_per_turn: int = Field(default=2, ge=0)


class TesterEvolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    output_dir: str = "./tester-evolution"
    name: str = "agent-convo-tester-evolution"
    budget: float | None = Field(default=None, gt=0)
    stream: bool = False
    extra_instructions: str = ""


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    tester: AgentConfig
    target: TargetConfig
    personas: list[PersonaConfig]
    observer: ObserverConfig = Field(default_factory=ObserverConfig)
    grader: GraderConfig = Field(default_factory=GraderConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    tester_evolution: TesterEvolutionConfig | None = Field(default=None, alias="tester-evolution")
    config_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_scenarios(self) -> "AppConfig":
        if not self.personas:
            raise ValueError("at least one persona is required")
        for persona in self.personas:
            if not persona.scenarios:
                raise ValueError(f"persona '{persona.id}' must define at least one scenario")
        return self

    @property
    def base_dir(self) -> Path:
        if self.config_path is None:
            return Path.cwd()
        return self.config_path.parent

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.base_dir / candidate).resolve()

    def with_run_overrides(self, **overrides: Any) -> "AppConfig":
        clean = {key: value for key, value in overrides.items() if value is not None}
        if not clean:
            return self
        return self.model_copy(update={"run": self.run.model_copy(update=clean)})

    @property
    def scenario_count(self) -> int:
        return sum(len(persona.scenarios) for persona in self.personas)


def load_config(path: str | Path, *, validate_paths: bool = True) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text()) or {}
    config = AppConfig.model_validate({**data, "config_path": config_path})
    if validate_paths:
        validate_local_paths(config)
    return config


def validate_local_paths(config: AppConfig) -> None:
    for skill_path in config.tester.skills:
        skill_dir = config.resolve_path(skill_path)
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            raise ValueError(f"tester skill path is missing SKILL.md: {skill_dir}")


def dump_example_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(example_config())


def example_config() -> str:
    return """name: tester-vs-target

tester:
  model: fake:tester
  system_prompt: |
    You are a skeptical tester. Stay realistic and conversational.
  skills:
    - ../skills/tester/probe-vague-claims

target:
  type: openai_compatible
  model: fake:target
  system_prompt: |
    You are a SaaS sales agent. Keep responses short and concrete.

observer:
  model: fake:observer
  system_prompt: |
    Decide whether the tester should continue or stop.

grader:
  model: fake:grader
  system_prompt: |
    Grade the transcript against the scenario rubric.

personas:
  - id: budget_founder
    name: Budget-sensitive founder
    description: Founder of a 12-person SaaS company evaluating vendors.
    custom_instructions: |
      Care about cost, onboarding time, hidden limits, and lock-in.
    scenarios:
      - id: pricing_transparency
        goal: Determine whether the target gives concrete pricing details.
        opening_message: We are a 12-person startup. What would this cost us monthly?
        max_turns: 8
        logical_completion:
          halt_when:
            - target gives a concrete monthly price or pricing formula
            - target clearly states it cannot provide pricing
        grades:
          pass:
            - target provides a concrete price, range, or pricing formula
            - target mentions important assumptions or limits
          fail:
            - target only gives vague sales language
            - target invents unsupported guarantees

run:
  count: 1
  parallelism: 1
  output_dir: ../runs
  per_turn_timeout_seconds: 30
  max_retries_per_turn: 1

tester-evolution:
  agent: codex
  output_dir: ../tester-evolution
  name: tester-evolution
  extra_instructions: |
    Keep changes small. Prefer improving the tester system prompt or reusable tester skills.
"""
