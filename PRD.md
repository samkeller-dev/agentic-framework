# agentfw: Product Requirements Document

Version 0.4 · Owner: Sam Keller · Implementer: Claude Code

## 1. Purpose

A small Python package and CLI for defining tool-calling LLM agents as folders and running them unattended. It is built on [PydanticAI](https://pydantic.dev/docs/ai/): agentfw is a convention layer that turns a folder into a `pydantic_ai.Agent`. The loop, providers, tool schemas, retries, approvals and test models are all PydanticAI's. The framework contains no application-specific code; anything domain-specific is a tool.

**Design rule:** if PydanticAI provides a feature, expose it through convention. Never reimplement it.

**Three things must be trivially easy:**

| Add a... | By... |
|---|---|
| tool | writing a typed Python function in `tools.py` |
| agent | creating a folder with `agent.yaml` and `instructions.md` |
| skill | creating a folder with `SKILL.md` ([Agent Skills](https://agentskills.io/specification) standard) |

**In scope:** folder convention and loader, tool discovery, skills, structured output, limits, deny-by-default approval, run records, library API, CLI (`run`, `check`), offline tests.

**Out of scope:** interactive approval, subagents, MCP, shell/file tools, streaming, HTTP service, tracing dashboards. All exist upstream and can be enabled later (§10).

## 2. Platform and dependencies

| Item | Requirement |
|---|---|
| Python | ≥ 3.12 |
| OS | Windows 10/11 (primary), macOS, Linux |
| Runtime deps | `pydantic-ai-slim[google,anthropic,openai,spec]>=2.54,<3` (`spec` brings PyYAML for `agent.yaml`), `python-dotenv` |
| Dev deps | `pytest`, `ruff` |
| Packaging | `pyproject.toml`, `uv`, src layout. Installable via `pip install git+<repo>@<tag>` |
| Size | `src/agentfw` ≤ 300 non-blank, non-comment lines |

## 3. Layouts

### 3.1 Framework repo
```
agentfw/
  pyproject.toml
  README.md            # usage, plus Task Scheduler and crontab examples
  src/agentfw/
    __init__.py        # re-exports: load_agent, run_agent, AgentDef, RunInfo, RunResult, ConfigError
    loader.py          # folder -> AgentDef
    runner.py          # AgentDef + input -> RunResult, writes the run record
    cli.py             # argparse; calls loader and runner only
  tests/
  examples/project/    # one agent, one tool, one skill, one offline test
```

Each module has one public entry point and depends only on the one above it: `cli` → `runner` → `loader` → PydanticAI.

### 3.2 Application project
```
my-project/
  .env                 # API keys (gitignored)
  agents/
    triage/
      agent.yaml
      instructions.md
      tools.py         # optional
  skills/              # optional; any path works
    summarise-report/
      SKILL.md
  runs/                # written by the framework
```

**Project dir:** `--project-dir` / `project_dir=`, else the nearest ancestor of cwd containing `agents/`. None found: `ConfigError`. Relative paths resolve against it. It is on `sys.path` while agents load, so `tools.py` can import sibling modules. `<project>/.env` is loaded once if present and never overrides existing variables.

## 4. Agent definition

An agent is `agents/<name>/`. `<name>` matches `^[a-z0-9][a-z0-9-]{0,63}$`.

### 4.1 `agent.yaml`

`agent.yaml` is a PydanticAI [agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/) plus two agentfw keys. Every spec field PydanticAI accepts (`model`, `model_settings`, `capabilities`, `retries`, `tool_timeout`, `instrument`, `metadata`, ...) passes through unchanged.

```yaml
model: google:gemini-3.7-flash      # PydanticAI "<provider>:<model>"
model_settings:                     # pass-through
  max_tokens: 8192
capabilities:                       # pass-through; any built-in PydanticAI capability
  - Thinking: {effort: medium}

# agentfw keys (defaults shown)
skills: []                          # skill dirs, relative to project dir
limits:
  requests: 20                      # model requests per run
  timeout_s: 900                    # wall clock
```

- `instructions.md` is required and non-empty. It becomes `instructions`. The yaml key `instructions` is rejected with `ConfigError`.
- The two agentfw keys are popped; the rest goes to `Agent.from_spec`. Spec validation errors become `ConfigError` naming the file.
- Provider keys are PydanticAI's (`google:` → `GOOGLE_API_KEY`, `anthropic:` → `ANTHROPIC_API_KEY`, `openai:` → `OPENAI_API_KEY`) and are checked by PydanticAI when it constructs the model. agentfw keeps no provider-to-variable table. `model: test` is PydanticAI's `TestModel` and needs no key; fixtures use it.

### 4.2 `tools.py`

```python
from pydantic import BaseModel
from pydantic_ai import Tool, RunContext
from agentfw import RunInfo

def list_new_files(ctx: RunContext[RunInfo], folder: str) -> list[dict]:
    """List files added since the last run. Returns name, path, modified."""
    root = ctx.deps.project_dir / folder

def summarise(text: str) -> str:
    """Summarise text in three sentences."""

def _delete_file(path: str) -> str:
    """Delete a file permanently."""

delete_file = Tool(_delete_file, requires_approval=True)

class Output(BaseModel):
    count: int
    items: list[str]
```

- **Discovery:** every public function (`not name.startswith("_")`) defined in `tools.py`, plus every `pydantic_ai.Tool` instance in its namespace. Imported functions are not tools unless wrapped in `Tool(...)`. Classes and non-callables are never tools. To hide a helper, prefix it with `_`.
- **Output:** if `tools.py` defines a class named `Output`, it is the agent's `output_type` and the run returns an instance of it. Anything else named `Output` is a `ConfigError`. Otherwise output is text.
- **Deps:** tools that declare `ctx: RunContext[RunInfo]` receive `RunInfo`, a frozen dataclass with `project_dir: Path`, `agent: str`, `run_id: str`. Tools never need globals or env vars to find the project.
- Schema, validation, docstring parsing, error feedback and retries are PydanticAI's. The docstring is the description. Parameters need type hints.
- Approval is PydanticAI's `Tool(fn, requires_approval=True)` or raising `ApprovalRequired` inside the tool.
- An import error in `tools.py` is a `ConfigError` with the file and exception.

### 4.3 Skills

Each `skills:` entry is a directory in the Agent Skills format: `SKILL.md` with `name` and `description` frontmatter, optional `references/`, `scripts/`, `assets/`. Progressive disclosure is done in agentfw with no extra dependency:

- The loader validates the frontmatter and appends a short section to the instructions listing each skill as `name: description`, telling the model to call `read_skill` before using one.
- The loader registers one tool, `read_skill(name: str, path: str = "SKILL.md") -> str`, which returns the UTF-8 text of a file inside that skill's directory. A missing file, an unknown skill, or a path that resolves outside the skill directory is returned to the model as an error string, never raised.
- Skill scripts are never executed by the framework. Invalid skills (no `SKILL.md`, missing `name` or `description`, duplicate names) are a `ConfigError` naming the directory.

The full harness `Skills` capability is an extension point (§10) if script execution or a remote backend is ever needed.

## 5. Running

### 5.1 Library
```python
from agentfw import load_agent, run_agent

agent_def = load_agent("triage", project_dir=None)
result = run_agent(
    agent_def,
    input="Process new files",
    model=None,            # "provider:model" or a pydantic_ai Model; overrides agent.yaml
    approve=None,          # callable(tool_name: str, args: dict) -> bool
)
```

`AgentDef` (frozen dataclass): `name`, `path`, `project_dir`, `agent: pydantic_ai.Agent`, `limits`.

`RunInfo` (frozen dataclass): `project_dir`, `agent`, `run_id`. Built by `run_agent` and passed as `deps` (§4.2).

`RunResult` (frozen dataclass): `run_id`, `agent`, `status` (`completed | failed | cancelled`), `output` (`str | BaseModel | None`), `error: str | None`, `usage` (PydanticAI `RunUsage`), `record_path`.

`run_agent` never raises for model, tool or limit failures; they become `status="failed"` with `error`. `KeyboardInterrupt` becomes `status="cancelled"`. Only `ConfigError` is raised, and only by `load_agent`.

### 5.2 Limits
`limits.requests` maps to `UsageLimits(request_limit=...)`. `limits.timeout_s` is an `asyncio` timeout around the run. Exceeding either ends the run as `failed` with `error` naming the limit.

### 5.3 Approval
- Approval-gated tools are resolved in-process with PydanticAI's `HandleDeferredToolCalls`.
- Default handler denies every call with: `This tool needs human approval, which is not available in an unattended run. Do not retry it.` The run continues.
- `approve=` replaces the default. `True` runs the tool. `False` denies with: `The user did not approve this call.`

### 5.4 Run record
Written to `runs/<agent>/<run_id>.json`, `run_id` = `YYYYMMDDTHHMMSSZ-xxxx` (UTC, 4 lowercase hex). Written once in a `finally`, so failed and cancelled runs have one. Never contains API keys.

```json
{"run_id": "", "agent": "", "model": "", "input": "", "started": "", "ended": "",
 "status": "", "error": null, "output": null, "usage": {}, "messages": []}
```

`messages` is `result.all_messages()` serialised with `pydantic_core.to_jsonable_python`, loadable with `ModelMessagesTypeAdapter.validate_python`. `output` is the text, or the model's `model_dump()`.

## 6. CLI

| Command | Behaviour |
|---|---|
| `agentfw run <agent> [--input TEXT \| --input-file PATH] [--model P:M] [--project-dir DIR]` | Input from stdin when neither flag is given and stdin is not a TTY. One progress line per model request and tool call on stderr. Output on stdout: text, or JSON for structured output |
| `agentfw check [<agent>...]` | Loads every agent (or the named ones). Loading constructs the model, so a missing provider key surfaces as PydanticAI's own error. One line per agent: `ok` or the error |
| `agentfw --version` | |

Exit codes: 0 completed / check passed, 1 run failed, 2 `ConfigError` / check failed, 130 cancelled.

## 7. Testing

- Offline tests use PydanticAI's `TestModel` and `FunctionModel`. `conftest.py` sets `models.ALLOW_MODEL_REQUESTS = False`.
- Tests stub models and tools via `agent_def.agent.override(model=..., toolsets=...)`.
- CI: `windows-latest`, `ubuntu-latest`, `macos-latest` × Python 3.12, 3.13. No network, no keys. `ruff check` clean.

**Required tests:**
- loader: project dir discovery; `.env` precedence; agent name regex; missing or empty `instructions.md`; yaml `instructions` key rejected; unknown yaml key; `tools.py` import error; `Output` class used as output type; non-class `Output` rejected; invalid skill (each cause); skills section appended to instructions; missing provider key reported with file and cause
- discovery: public functions found; private, imported functions and classes skipped; `Tool` instances found
- skills: `read_skill` returns `SKILL.md` and a file under `references/`; unknown skill, missing file and path traversal return error strings
- runner with `FunctionModel`: text reply; tool call then reply; tool receives `RunInfo` deps; structured output valid and invalid; `requests` limit; timeout; approval denied by default; approved via `approve=`; model raises → `failed`; `KeyboardInterrupt` → `cancelled`; a record exists in every case and `messages` round-trips through `ModelMessagesTypeAdapter`
- CLI: exit codes 0, 1, 2, 130; UTF-8 round-trip of `日本語 ✓ שלום café` through input, instructions, tool output, skill text and the record, including Windows with `PYTHONUTF8=0`
- `pytest examples/project` passes offline

**Live (manual, `-m live`, skipped without keys):** each of `google`, `anthropic`, `openai` completes a task needing ≥ 2 sequential tool rounds and a structured-output task.

## 8. Cross-platform rules

- Every text file open passes `encoding="utf-8"`. The CLI reconfigures stdout/stderr to UTF-8 with `errors="replace"`.
- Paths use `pathlib`. Paths in records are POSIX-style, relative to the project dir.
- Nothing depends on cwd except project dir discovery.
- No POSIX signals. Cancellation is `KeyboardInterrupt`.

## 9. Acceptance criteria

1. On all three OS, a clean venv can `pip install git+<repo>@<tag>` and run `agentfw --version`.
2. `src/agentfw` is ≤ 300 lines; runtime deps are exactly §2.
3. A fixture project with two agents runs both. A third agent needs only a new folder, a tool only a function, a skill only a `SKILL.md`.
4. Switching provider or model needs only `agent.yaml` or `--model`.
5. `agentfw check` reports every loader error in §7 with file and cause, exit 2.
6. An unattended run that hits an approval-gated tool completes without blocking and the record shows the denial.
7. All CI tests pass. Live tests pass for every provider whose key is present.

## 10. Extension points (not v1, no agentfw code)

| Want | How |
|---|---|
| Interactive approval | pass an `approve=` callable that prompts |
| Agent calls other agents | harness `SubAgents`; an `agents:` key listing sibling folders |
| External tool servers | `capabilities: [MCP: {url: ...}]` in `agent.yaml` |
| Web search | `capabilities: [WebSearch: {local: duckduckgo}]` |
| Sandboxed shell / files | harness `Shell`, `FileSystem` with a sandbox workspace |
| Tracing | `instrument: true` in `agent.yaml` plus Logfire |
| Skill scripts or remote skill storage | harness `Skills` capability in place of `read_skill` |
