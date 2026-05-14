from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from agent_convo.config import dump_example_config, load_config
from agent_convo.doctor import check_config
from agent_convo.evolution import EvolutionNotConfiguredError, evolve_tester_agent
from agent_convo.export import export_run
from agent_convo.improve import improve_agent
from agent_convo.runner import resume_existing, run_new

app = typer.Typer(no_args_is_help=True)


@app.command()
def init() -> None:
    example_path = Path("examples/tester_vs_target.yaml")
    if not example_path.exists():
        dump_example_config(example_path)
    env_path = Path(".env.example")
    if not env_path.exists():
        env_path.write_text("OPENAI_API_KEY=\nTARGET_API_KEY=\n")
    skill_dir = Path("skills/tester/probe-vague-claims")
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        skill_file.write_text("# Probe Vague Claims\n\nAsk one concrete follow-up when a claim is vague.\n")
    typer.echo("Created examples/tester_vs_target.yaml, .env.example, and starter skills.")


@app.command()
def validate(config_path: Annotated[Path, typer.Argument(exists=True)]) -> None:
    load_config(config_path)
    typer.echo(f"Valid config: {config_path}")


@app.command()
def doctor(config_path: Annotated[Path, typer.Argument(exists=True)]) -> None:
    load_dotenv()
    config = load_config(config_path)
    result = check_config(config)
    for message in result.messages:
        typer.echo(message)
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def run(
    config_path: Annotated[Path, typer.Argument(exists=True)],
    count: Annotated[int | None, typer.Option("--count")] = None,
    parallelism: Annotated[int | None, typer.Option("--parallelism")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    per_turn_timeout_seconds: Annotated[float | None, typer.Option("--per-turn-timeout-seconds")] = None,
    max_retries_per_turn: Annotated[int | None, typer.Option("--max-retries-per-turn")] = None,
    evolve_tester_agent_flag: Annotated[
        bool,
        typer.Option("--evolve-tester-agent", help="Run tester evolution with harnessctl after the run finishes."),
    ] = False,
) -> None:
    load_dotenv()
    config = load_config(config_path).with_run_overrides(
        count=count,
        parallelism=parallelism,
        output_dir=str(output_dir) if output_dir is not None else None,
        per_turn_timeout_seconds=per_turn_timeout_seconds,
        max_retries_per_turn=max_retries_per_turn,
    )
    run_dir = asyncio.run(run_new(config))
    typer.echo(str(run_dir))
    if evolve_tester_agent_flag:
        try:
            evolution_dir = evolve_tester_agent(config, run_dir=run_dir)
        except EvolutionNotConfiguredError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"tester evolution: {evolution_dir}")


@app.command()
def status(run_dir: Annotated[Path, typer.Argument(exists=True)]) -> None:
    states = {}
    for state_path in sorted((run_dir / "conversations").glob("*/state.json")):
        state = json.loads(state_path.read_text())
        states[state["status"]] = states.get(state["status"], 0) + 1
    if not states:
        typer.echo("No conversations found.")
        return
    for key in sorted(states):
        typer.echo(f"{key}: {states[key]}")


@app.command()
def resume(
    run_dir: Annotated[Path, typer.Argument(exists=True)],
    config_path: Annotated[Path, typer.Option("--config", "-c", exists=True)] = Path("examples/tester_vs_target.yaml"),
    parallelism: Annotated[int | None, typer.Option("--parallelism")] = None,
    per_turn_timeout_seconds: Annotated[float | None, typer.Option("--per-turn-timeout-seconds")] = None,
    max_retries_per_turn: Annotated[int | None, typer.Option("--max-retries-per-turn")] = None,
) -> None:
    load_dotenv()
    config = load_config(config_path).with_run_overrides(
        parallelism=parallelism,
        per_turn_timeout_seconds=per_turn_timeout_seconds,
        max_retries_per_turn=max_retries_per_turn,
    )
    asyncio.run(resume_existing(config, run_dir))
    typer.echo(f"Resumed {run_dir}")


@app.command("export")
def export_cmd(
    run_dir: Annotated[Path, typer.Argument(exists=True)],
    fmt: Annotated[str, typer.Option("--format")] = "jsonl",
    out: Annotated[Path, typer.Option("--out")] = Path("conversations.jsonl"),
) -> None:
    export_run(run_dir, fmt=fmt, out=out)
    typer.echo(str(out))


@app.command()
def improve(
    run_dir: Annotated[Path, typer.Option("--run", exists=True)],
    agent: Annotated[str, typer.Option("--agent")],
    config_path: Annotated[Path, typer.Option("--config", "-c", exists=True)] = Path("examples/tester_vs_target.yaml"),
) -> None:
    config = load_config(config_path)
    output_dir = improve_agent(config, run_dir=run_dir, agent_name=agent)
    typer.echo(str(output_dir))

if __name__ == "__main__":
    app()
