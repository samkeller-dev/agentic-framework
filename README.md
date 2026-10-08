# agentfw

Define tool-calling LLM agents as folders and run them unattended.

A thin convention layer over [PydanticAI](https://pydantic.dev/docs/ai/): a folder becomes a
`pydantic_ai.Agent`. The loop, providers, tool schemas, retries and approvals are PydanticAI's.
agentfw adds the folder layout, a CLI, run limits, deny-by-default approval and a JSON record
per run.

```sh
pip install "git+https://github.com/samkeller-dev/agentic-framework@v0.1.0"
```

Python 3.12+, Windows, macOS or Linux.

## Layout

```
my-project/
  .env                     # API keys; loaded once, never overrides the environment
  agents/triage/
    agent.yaml             # model and options
    instructions.md        # system instructions
    tools.py               # optional tools and Output class
  skills/summarise/SKILL.md   # optional
  runs/                    # one JSON record per run, written by agentfw
```

The project dir is `--project-dir`, else the nearest ancestor of the current directory that
contains `agents/`. [examples/project](examples/project) is a complete, runnable project.

## agent.yaml

A PydanticAI [agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/) plus two keys.
Any spec field (`model_settings`, `capabilities`, `retries`, ...) passes through.

```yaml
model: google:gemini-3.7-flash      # "<provider>:<model>"; key from GOOGLE_API_KEY etc.
skills: [skills/summarise]          # agentfw: skill dirs, relative to the project dir
limits: {requests: 20, timeout_s: 900}   # agentfw: defaults shown
```

Instructions live in `instructions.md`; an `instructions` key or any unknown key is an error.
`model: test` is PydanticAI's `TestModel` and needs no key.

## tools.py

```python
from pydantic import BaseModel
from pydantic_ai import RunContext, Tool

from agentfw import RunInfo


def list_new_files(ctx: RunContext[RunInfo], folder: str) -> list[str]:
    """List files added since the last run."""      # docstring = description
    root = ctx.deps.project_dir / folder             # deps: project_dir, agent, run_id
    ...


def _delete_file(path: str) -> str: ...              # _prefix hides it


delete_file = Tool(_delete_file, name="delete_file", requires_approval=True)


class Output(BaseModel):                             # optional: structured output
    count: int
    items: list[str]
```

Every public function defined in the file is a tool, plus every `Tool` instance. Imported
functions are not tools unless wrapped in `Tool(...)`.

## Skills

Each `skills:` entry is a folder with a `SKILL.md` carrying `name` and `description`
frontmatter ([Agent Skills](https://agentskills.io/specification)). agentfw lists the skills in
the instructions and adds one tool, `read_skill(name, path="SKILL.md")`, that returns a file
from the skill folder. Scripts are never executed.

## Running

```sh
agentfw run triage --input "Process new files"      # or --input-file, or stdin
agentfw run triage --input "..." --model openai:gpt-5.5
agentfw check                                        # load every agent, report errors
```

Progress goes to stderr, output to stdout (JSON for structured output). Exit codes: 0 ok,
1 run failed, 2 config error, 130 cancelled.

```python
from agentfw import load_agent, run_agent

result = run_agent(load_agent("triage"), "Process new files", model=None, approve=None)
result.status      # completed | failed | cancelled; failures never raise
result.output      # str or Output instance
result.record_path # runs/triage/<run_id>.json: input, output, usage, full message history
```

Approval-gated tools are denied by default so unattended runs never block; the model is told
not to retry. Pass `approve=lambda tool_name, args: ...` to decide per call. `limits.requests`
and `limits.timeout_s` end the run as `failed` when exceeded.

## Scheduling

```bat
schtasks /Create /SC DAILY /ST 07:00 /TN "agentfw triage" /TR "cmd /c cd /d C:\projects\my-project && .venv\Scripts\agentfw.exe run triage --input \"Process new files\" >> runs\triage.log 2>&1"
```

```cron
0 7 * * * cd /home/me/my-project && .venv/bin/agentfw run triage --input "Process new files" >> runs/triage.log 2>&1
```

## Testing offline

```python
from pydantic_ai.models.test import TestModel

def test_triage(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "offline")   # loading constructs the model
    agent_def = load_agent("triage", project_dir="my-project")
    with agent_def.agent.override(model=TestModel()):
        assert run_agent(agent_def, "Process new files").status == "completed"
```

## Development

```sh
uv sync && uv run ruff check . && uv run pytest      # add -m live for real providers
```

Spec: [PRD.md](PRD.md).
