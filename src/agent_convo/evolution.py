from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from agent_convo.config import AppConfig
from agent_convo.storage import utc_now


class EvolutionNotConfiguredError(ValueError):
    pass


def build_evolution_prompt(config: AppConfig, *, run_dir: Path) -> str:
    latest = run_dir.resolve()
    skill_paths = "\n".join(f"- {path}" for path in config.tester.skills) or "- none"
    scenario_lines = []
    for persona in config.personas:
        for scenario in persona.scenarios:
            scenario_lines.append(f"- {persona.id}/{scenario.id}: {scenario.goal}")
    scenarios = "\n".join(scenario_lines)
    extra = config.tester_evolution.extra_instructions.strip() if config.tester_evolution else ""
    return "\n\n".join(
        part
        for part in [
            "You are evolving the tester agent for agent-convo.",
            "Use the latest run artifacts to decide whether the tester can be made more effective next time.",
            f"Config file: {config.config_path}",
            f"Latest run directory: {latest}",
            "Inspect the latest run's transcripts, grades, metadata, and observer events.",
            "Current tester system prompt:",
            config.tester.system_prompt.strip() or "(empty)",
            "Current tester skills:",
            skill_paths,
            "Configured scenarios:",
            scenarios,
            (
                "Your task: decide whether the tester's system prompt or tester skills should change. "
                "If the latest run shows weak testing behavior, make the smallest useful repo change. "
                "Prefer edits to the tester system prompt in the YAML or reusable tester skills under skills/. "
                "Do not change target, observer, grader, or run settings unless they block tester evolution. "
                "If no useful improvement is justified, do not edit files; explain why."
            ),
            extra,
        ]
        if part
    )


def evolve_tester_agent(
    config: AppConfig,
    *,
    run_dir: Path,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    if config.tester_evolution is None:
        raise EvolutionNotConfiguredError(
            "--evolve-tester-agent requires a tester-evolution section in the YAML config"
        )

    timestamp = utc_now().replace(":", "").replace(".", "")
    output_dir = config.resolve_path(config.tester_evolution.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_evolution_prompt(config, run_dir=run_dir)
    prompt_path = output_dir / "prompt.md"
    result_path = output_dir / "harnessctl-result.md"
    prompt_path.write_text(prompt + "\n")

    command = [
        "harnessctl",
        "run",
        "--agent",
        config.tester_evolution.agent,
        "--name",
        config.tester_evolution.name,
    ]
    if config.tester_evolution.stream:
        command.append("--stream")
    if config.tester_evolution.budget is not None:
        command.extend(["--budget", str(config.tester_evolution.budget)])
    command.append(prompt)

    result = command_runner(
        command,
        cwd=str(config.base_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    result_path.write_text(
        "\n".join(
            [
                f"# harnessctl result",
                "",
                f"exit_code: {result.returncode}",
                "",
                "## stdout",
                "",
                result.stdout or "",
                "",
                "## stderr",
                "",
                result.stderr or "",
            ]
        ).rstrip()
        + "\n"
    )
    if result.returncode != 0:
        raise RuntimeError(f"harnessctl tester evolution failed; see {result_path}")
    return output_dir
