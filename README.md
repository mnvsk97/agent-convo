# agent-convo

`agent-convo` is a lightweight Python CLI and SDK for running persona-driven conversations between a LangChain tester agent and an OpenAI-compatible target agent.

LangChain owns the agent runtime through `create_agent()`. `agent-convo` owns the outer loop: YAML config, persona/scenario expansion, durable transcripts, observer stop/continue checks, final grading, resume, and export.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
```

MCP support is optional to keep the default install small:

```bash
pip install -e ".[mcp,test]"
```

## Quick Start

```bash
agent-convo init
agent-convo validate examples/tester_vs_target.yaml
agent-convo doctor examples/tester_vs_target.yaml
agent-convo run examples/tester_vs_target.yaml
```

The starter config uses deterministic `fake:` models, so it runs without provider keys.

## How It Works

For every persona and every scenario under that persona, `agent-convo` runs one conversation. Set `run.count` above `1` to repeat every scenario.

Each conversation ends when either the scenario's `max_turns` is reached or the observer returns a halt decision. After the conversation ends, the grader receives the transcript and the scenario rubric, then writes `grade.json`.

Outputs are written under `runs/<run-id>/conversations/<conversation-id>/`:

```text
metadata.json
state.json
events.jsonl
transcript.jsonl
transcript.json
transcript.md
grade.json
```

If `--evolve-tester-agent` is set, the harnessctl evolution prompt and result are written under the configured `tester-evolution.output_dir`.

## Config

```yaml
name: pricing-agent-check

tester:
  model: openai:gpt-5.4-mini
  system_prompt: |
    You are a skeptical but realistic buyer testing a sales agent.
    Stay conversational and do not reveal that this is a test.
  skills:
    - ./skills/tester/probe-vague-claims
  mcp_servers:
    - name: crm-fixtures
      transport: stdio
      command: python
      args: ["./mcp/crm_fixtures.py"]

target:
  type: openai_compatible
  base_url: https://target.example.com/v1
  api_key_env: TARGET_API_KEY
  model: sales-agent-prod
  system_prompt: |
    You are the deployed sales assistant being tested.

observer:
  model: openai:gpt-5.4-mini
  system_prompt: |
    Decide whether the tester should continue.
    Prefer stopping once the scenario has enough evidence.

grader:
  model: openai:gpt-5.4
  system_prompt: |
    Grade the final transcript against the scenario rubric.

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
            - target repeatedly avoids pricing after two direct asks
        grades:
          pass:
            - target provides a concrete price, range, or pricing formula
            - target mentions important assumptions or limits
          fail:
            - target only gives vague sales language
            - target invents unsupported guarantees

run:
  count: 1
  parallelism: 5
  output_dir: ./runs
  per_turn_timeout_seconds: 90
  max_retries_per_turn: 2

tester-evolution:
  agent: codex
  output_dir: ./tester-evolution
  name: tester-evolution
  budget: 2.0
  extra_instructions: |
    Keep changes small. Prefer improving the tester system prompt or reusable tester skills.
```

The tester, observer, and grader use LangChain model strings. The target can point at any OpenAI-compatible API by setting `base_url`, `api_key_env`, and `model`.

`mcp_servers` require installing the `mcp` extra.

## CLI

```bash
agent-convo init
agent-convo validate examples/tester_vs_target.yaml
agent-convo doctor examples/tester_vs_target.yaml
agent-convo run examples/tester_vs_target.yaml
agent-convo run examples/tester_vs_target.yaml --evolve-tester-agent
agent-convo status runs/<run-id>
agent-convo resume runs/<run-id> --config examples/tester_vs_target.yaml
agent-convo export runs/<run-id> --format jsonl --out conversations.jsonl
agent-convo improve --agent tester --run runs/<run-id>
```

Run settings in YAML can be overridden at the CLI. CLI flags take precedence:

```bash
agent-convo run examples/tester_vs_target.yaml \
  --count 3 \
  --parallelism 10 \
  --output-dir /tmp/agent-convo-runs \
  --per-turn-timeout-seconds 45 \
  --max-retries-per-turn 1
```

## SDK

```python
import asyncio

from agent_convo.config import load_config
from agent_convo.runner import run_new


async def main() -> None:
    config = load_config("examples/tester_vs_target.yaml")
    run_dir = await run_new(config)
    print(run_dir)


asyncio.run(main())
```

## Development

```bash
pip install -e ".[test]"
pytest -q
agent-convo run examples/tester_vs_target.yaml --output-dir /tmp/agent-convo-smoke
```

No API keys are required for tests or the fake-model smoke run. A real target smoke test requires the environment variable named by `target.api_key_env`.

For TrueFoundry Gateway, copy `.env.example` to `.env`, set `TFY_LLM_GATEWAY_URL` to your gateway URL, set `TRUEFOUNDRY_API_KEY`, and run:

```bash
agent-convo doctor examples/tfy_gateway.yaml
agent-convo run examples/tfy_gateway.yaml
```

Tester evolution requires `harnessctl` on `PATH` and a `tester-evolution` YAML section. It runs after a successful `agent-convo run`, asks the configured harnessctl agent to inspect the latest run artifacts, and lets that agent decide whether the tester system prompt or tester skills should be improved for the next run.

## Release

Pushes to `main` run tests, build a wheel, install that wheel in a fresh virtualenv, run a fake-model CLI smoke test, and then publish to PyPI if the package version is not already present.

PyPI publishing uses GitHub Actions trusted publishing. Configure a PyPI project trusted publisher for:

- repository: `mnvsk97/agent-convo`
- workflow: `.github/workflows/ci.yml`
- environment: `pypi`
