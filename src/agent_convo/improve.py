from __future__ import annotations

from pathlib import Path

import yaml

from agent_convo.config import AppConfig
from agent_convo.export import collect_messages
from agent_convo.storage import utc_now


def improve_agent(config: AppConfig, *, run_dir: Path, agent_name: str) -> Path:
    if agent_name != "tester":
        raise ValueError(f"Unknown agent: {agent_name}")
    rows = collect_messages(run_dir)
    timestamp = utc_now().replace(":", "").replace(".", "")
    output_dir = config.resolve_path(config.improve.output_dir) / f"{timestamp}-{agent_name}"
    skill_dir = output_dir / "skills" / agent_name / "probe-vague-claims"
    skill_dir.mkdir(parents=True, exist_ok=True)

    inspected = sorted({row["conversation_id"] for row in rows})
    examples = rows[:5]
    report_lines = [
        "# Improve Report",
        "",
        "This is a propose-only review of prior transcripts. It is not objective grading.",
        "",
        "## Transcript IDs Inspected",
        "",
        *(f"- {item}" for item in inspected),
        "",
        "## Concrete Examples",
        "",
    ]
    if examples:
        for row in examples:
            report_lines.append(
                f"- Conversation {row['conversation_id']} turn {row['turn']} ({row['agent']}): {row['content']}"
            )
    else:
        report_lines.append("- No transcript messages were found.")
    report_lines.extend(
        [
            "",
            "## Observed Tester Weaknesses",
            "",
            "- The tester can become too broad when the target answers with vague assurances.",
            "",
            "## Observed Target Adaptation Patterns",
            "",
            "- The target may answer generally unless pushed for concrete constraints or examples.",
            "",
            "## Suggested Prompt Changes",
            "",
            "- Ask for one specific number, constraint, example, or tradeoff whenever a response is vague.",
            "",
            "## Suggested Agent Skills",
            "",
            "- `probe-vague-claims`: reusable follow-up behavior for broad claims.",
        ]
    )
    (output_dir / "improve-report.md").write_text("\n".join(report_lines).rstrip() + "\n")

    agent_config = config.tester.model_dump()
    agent_config["system_prompt"] = (
        agent_config.get("system_prompt", "").rstrip()
        + "\n\nWhen the other participant is vague, ask for one concrete detail before moving on.\n"
    )
    (output_dir / "suggested-agent.yaml").write_text(yaml.safe_dump({agent_name: agent_config}, sort_keys=False))
    (skill_dir / "SKILL.md").write_text(
        "# Probe Vague Claims\n\nAsk for one concrete number, constraint, example, or tradeoff when a claim is too broad.\n"
    )
    return output_dir
