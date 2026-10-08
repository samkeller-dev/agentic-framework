# agentfw

Define tool-calling LLM agents as folders and run them unattended.

agentfw is a thin convention layer over [PydanticAI](https://pydantic.dev/docs/ai/): a folder
becomes a `pydantic_ai.Agent`. The agent loop, providers, tool schemas, retries, approvals and
test models are all PydanticAI's. agentfw adds the folder layout, a CLI, limits, deny-by-default
approval for unattended runs, and a JSON record of every run.

- A **tool** is a typed Python function in `tools.py`.
- An **agent** is a folder with `agent.yaml` and `instructions.md`.
- A **skill** is a folder with `SKILL.md` ([Agent Skills](https://agentskills.io/specification)).

## Install

Python 3.12 or newer, on Windows, macOS or Linux.

```sh
pip install "git+https://github.com/samkeller-dev/agentic-framework@v0.1.0"
agentfw --version
```

## Project layout

```
my-project/
  .env                     # API keys (gitignored); loaded once, never overrides existing variables
  agents/
    triage/
      agent.yaml           # model and options
      instructions.md      # system instructions (required, non-empty)
      tools.py             # optional: tools and an optional Output class
  skills/                  # optional; any path works
    summarise-report/
      SKILL.md
  runs/                    # written by agentfw: one JSON record per run
```

The **project dir** is `--project-dir` / `project_dir=`, otherwise the nearest ancestor of the
current directory that contains `agents/`. Relative paths in `agent.yaml` resolve against it. It
is on `sys.path` while agents load, so `tools.py` can import sibling modules.

See [examples/project](examples/project) for a complete, runnable project with one agent, one tool,
one skill and one offline test.

## Agents

An agent lives in `agents/<name>/`. The name must match `^[a-z0-9][a-z0-9-]{0,63}$`.

### agent.yaml

`agent.yaml` is a PydanticAI [agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/)
plus two agentfw keys. Every field the spec accepts (`model`, `model_settings`, `capabilities`,
`retries`, `tool_timeout`, `metadata`, ...) passes through unchanged.

```yaml
model: google:gemini-3.7-flash      # PydanticAI "<provider>:<model>"
model_settings:                     # pass-through
  max_tokens: 8192
capabilities:                       # pass-through; any built-in PydanticAI capability
  - Thinking: {effort: medium}

# agentfw keys (defaults shown)
skills: []                          # skill dirs, relative to the project dir
limits:
  requests: 20                      # model requests per run
  timeout_s: 900                    # wall clock, seconds
```

- Instructions come from `instructions.md`. An `instructions` key in the yaml is an error.
- Unknown keys are an error, so typos never pass silently.
- Provider keys are read by PydanticAI from the environment: `google:` needs `GOOGLE_API_KEY`,
  `anthropic:` needs `ANTHROPIC_API_KEY`, `openai:` needs `OPENAI_API_KEY`. Put them in
  `<project>/.env`.
- `model: test` is PydanticAI's `TestModel` and needs no key.

### tools.py

```python
from pydantic import BaseModel
from pydantic_ai import RunContext, Tool

from agentfw import RunInfo


def list_new_files(ctx: RunContext[RunInfo], folder: str) -> list[dict]:
    """List files added since the last run. Returns name, path and modified time."""
    root = ctx.deps.project_dir / folder
    ...


def summarise(text: str) -> str:
    """Summarise text in three sentences."""
    ...


def _delete_file(path: str) -> str:
    """Delete a file permanently."""
    ...


delete_file = Tool(_delete_file, name="delete_file", requires_approval=True)


class Output(BaseModel):
    count: int
    items: list[str]
```

- **Discovery.** Every public function defined in `tools.py` is a tool, plus every
  `pydantic_ai.Tool` instance in the module. Functions imported from elsewhere are not tools
  unless wrapped in `Tool(...)`. Prefix a helper with `_` to hide it.
- **Schema.** The docstring is the description; parameter types come from the type hints.
  Validation, error feedback and retries are PydanticAI's.
- **Output.** If the module defines a class named `Output`, the run returns an instance of it and
  the CLI prints it as JSON. Otherwise the output is text.
- **Run info.** A tool that declares `ctx: RunContext[RunInfo]` receives `project_dir`, `agent`
  and `run_id`. Tools never need globals or environment variables to find the project.
- **Approval.** `Tool(fn, requires_approval=True)`, or raise `pydantic_ai.ApprovalRequired`
  inside the tool. See [Approval](#approval).

### Skills

Each entry under `skills:` is a directory in the Agent Skills format: a `SKILL.md` with `name`
and `description` frontmatter, plus optional `references/`, `scripts/` and `assets/`.

```markdown
---
name: summarise-report
description: How to summarise a report in three sentences.
---

1. State the most important finding first.
...
```

agentfw appends a short section to the instructions listing each skill as `name: description`
and registers one tool, `read_skill(name, path="SKILL.md")`, which returns the text of a file
inside the skill directory. Unknown skills, missing files and paths outside the directory come
back to the model as error strings. Skill scripts are never executed by the framework.

## Running

### CLI

```sh
agentfw run triage --input "Process new files"
agentfw run triage --input-file request.txt
echo "Process new files" | agentfw run triage
agentfw run triage --input "..." --model anthropic:claude-sonnet-5-5
agentfw check                 # load every agent and report config errors
agentfw check triage          # just one
```

`run` prints one progress line per model request and tool call on stderr, and the output on
stdout: text, or JSON for structured output. `check` prints one line per agent, `ok` or the error.
Loading constructs the model, so a missing provider key shows up in `check` too.

Exit codes: `0` completed or check passed, `1` run failed, `2` configuration error or check
failed, `130` cancelled with Ctrl+C.

### Library

```python
from agentfw import load_agent, run_agent

agent_def = load_agent("triage", project_dir=None)  # None: discover from cwd
result = run_agent(
    agent_def,
    input="Process new files",
    model=None,  # "provider:model" or a pydantic_ai Model; overrides agent.yaml
    approve=None,  # callable(tool_name, args) -> bool; see Approval
)
print(result.status, result.output, result.record_path)
```

`run_agent` never raises for model, tool or limit failures. They become `status="failed"` with
`error` set. Ctrl+C becomes `status="cancelled"`. The only exception agentfw raises is
`ConfigError`, from `load_agent`.

### Limits

`limits.requests` caps model requests per run (PydanticAI `UsageLimits`). `limits.timeout_s` is a
wall-clock timeout around the whole run. Exceeding either ends the run as `failed` with an error
naming the limit.

### Approval

Approval-gated tools are resolved in-process, so an unattended run never blocks:

- By default every gated call is denied and the model is told: *"This tool needs human approval,
  which is not available in an unattended run. Do not retry it."* The run continues.
- Pass `approve=` to `run_agent` to decide per call. Returning `True` runs the tool; `False`
  denies it with *"The user did not approve this call."* An interactive prompt is just an
  `approve=` callable that asks.

### Run records

Every run writes `runs/<agent>/<run_id>.json`, including failed and cancelled runs. `run_id` is
`YYYYMMDDTHHMMSSZ-xxxx` (UTC plus four hex characters). The record holds the input, model,
status, error, output, usage and the full message history. `messages` loads back with
`pydantic_ai.ModelMessagesTypeAdapter.validate_python(record["messages"])`. Records never contain
API keys.

## Scheduling

Any scheduler that can run a command works. Keep the key in `<project>/.env` and let project
discovery find the folder.

**Windows Task Scheduler** (daily at 07:00):

```bat
schtasks /Create /SC DAILY /ST 07:00 /TN "agentfw triage" /TR "cmd /c cd /d C:\projects\my-project && .venv\Scripts\agentfw.exe run triage --input \"Process new files\" >> runs\triage.log 2>&1"
```

**cron** (daily at 07:00):

```cron
0 7 * * * cd /home/me/my-project && .venv/bin/agentfw run triage --input "Process new files" >> runs/triage.log 2>&1
```

## Testing your project offline

Use PydanticAI's `TestModel` or `FunctionModel`; no key and no network needed. Loading an agent
constructs its model, so set a dummy key for the provider named in `agent.yaml`.

```python
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from agentfw import load_agent, run_agent

models.ALLOW_MODEL_REQUESTS = False


def test_triage(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "offline")
    agent_def = load_agent("triage", project_dir="path/to/my-project")
    with agent_def.agent.override(model=TestModel()):
        result = run_agent(agent_def, "Process new files")
    assert result.status == "completed", result.error
```

## Extension points

Nothing below needs agentfw code; it is all PydanticAI configuration.

| Want | How |
|---|---|
| Interactive approval | pass an `approve=` callable that prompts |
| External tool servers | `capabilities: [MCP: {url: ...}]` in `agent.yaml` |
| Web search | `capabilities: [WebSearch: {local: duckduckgo}]` |
| Tracing | `capabilities: [Instrumentation]` plus Logfire |
| Sandboxed shell / files | PydanticAI harness `Shell` / `FileSystem` capabilities |

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest                 # offline; also runs examples/project
uv run pytest -m live         # needs GOOGLE_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY
```

`src/agentfw` is three modules (`loader`, `runner`, `cli`) kept under 300 lines. The
specification is in [PRD.md](PRD.md).
