"""Fixture-project builders and FunctionModel stubs shared by the tests."""

from pathlib import Path
from textwrap import dedent
from typing import Any

from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentfw import AgentDef, run_agent

UNICODE = "日本語 ✓ שלום café"

TOOLS_SRC = '''
from pydantic import BaseModel  # noqa: F401 (used by STRUCTURED_SRC)
from pydantic_ai import RunContext, Tool

from agentfw import RunInfo


def echo(text: str) -> str:
    """Return the text unchanged."""
    return text


def whoami(ctx: RunContext[RunInfo]) -> str:
    """Return agent, run id and project dir from the run info."""
    return f"{ctx.deps.agent}|{ctx.deps.run_id}|{ctx.deps.project_dir.as_posix()}"


def _wipe(path: str) -> str:
    """Delete a path permanently."""
    return f"wiped {path}"


wipe = Tool(_wipe, name="wipe", requires_approval=True)
'''

STRUCTURED_SRC = (
    TOOLS_SRC
    + """

class Output(BaseModel):
    count: int
    items: list[str]
"""
)


def make_agent(
    project: Path,
    name: str,
    *,
    yaml: str = "model: test\n",
    instructions: str = "Be brief.",
    tools: str | None = None,
) -> Path:
    d = project / "agents" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "agent.yaml").write_text(yaml, encoding="utf-8")
    (d / "instructions.md").write_text(instructions, encoding="utf-8")
    if tools is not None:
        (d / "tools.py").write_text(dedent(tools), encoding="utf-8")
    return d


def make_skill(
    project: Path,
    rel: str,
    frontmatter: str = "name: summarise\ndescription: How to summarise.",
    body: str = "Summarise in three sentences.",
) -> Path:
    d = project / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")
    return d


def replying(text: str) -> FunctionModel:
    """A model that always answers with `text`."""
    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(text)]))


def calling(tool: str, args: dict[str, Any]) -> FunctionModel:
    """A model that calls `tool` once, then replies with whatever the tool returned."""

    def fn(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool, args)])
        return ModelResponse(parts=[TextPart(str(messages[-1].parts[-1].content))])

    return FunctionModel(fn)


def call_tool(agent_def: AgentDef, tool: str, **args: Any) -> str:
    """Run the agent so that it calls `tool` once; return what the tool returned."""
    result = run_agent(agent_def, "go", model=calling(tool, args))
    assert result.status == "completed", result.error
    return result.output


def first_request(agent_def: AgentDef) -> dict[str, Any]:
    """Run once; return the instructions and tool names the model saw in its first request."""
    seen: dict[str, Any] = {}

    def fn(messages, info):
        seen["instructions"] = messages[0].instructions
        seen["tools"] = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart("ok")])

    result = run_agent(agent_def, "go", model=FunctionModel(fn))
    assert result.status == "completed", result.error
    return seen
