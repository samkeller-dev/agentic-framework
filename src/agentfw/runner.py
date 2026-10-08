"""Run a loaded agent once and write the run record."""

import asyncio
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_ai import RunContext, RunUsage, ToolDenied, UsageLimits, capture_run_messages
from pydantic_ai.capabilities import HandleDeferredToolCalls, Hooks
from pydantic_core import to_jsonable_python

from .loader import AgentDef, RunInfo

DENIED_UNATTENDED = (
    "This tool needs human approval, which is not available in an unattended run. Do not retry it."
)
DENIED_BY_USER = "The user did not approve this call."


@dataclass(frozen=True)
class RunResult:
    run_id: str
    agent: str
    status: str  # completed | failed | cancelled
    output: Any  # str | BaseModel | None
    error: str | None
    usage: RunUsage
    record_path: Path


def run_agent(
    agent_def: AgentDef,
    input: str,
    *,
    model: Any = None,
    approve: Callable[[str, dict[str, Any]], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    """Run the agent on `input`. `model` is "provider:model" or a pydantic_ai Model, overriding
    agent.yaml. Model, tool and limit failures never raise; they become status "failed"."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(2)
    name, limits, usage = agent_def.name, agent_def.limits, RunUsage()
    info = RunInfo(agent_def.project_dir, name, run_id)
    started, status, output, error = _now(), "failed", None, None
    m = model or agent_def.agent.model
    label = m if isinstance(m, str) else f"{m.system}:{m.model_name}"

    def handle(ctx: RunContext[RunInfo], requests: Any) -> Any:
        return requests.build_results(
            approvals={
                c.tool_call_id: True
                if approve and approve(c.tool_name, c.args_as_dict())
                else ToolDenied(DENIED_BY_USER if approve else DENIED_UNATTENDED)
                for c in requests.approvals
            }
        )

    def on_request(ctx: RunContext[RunInfo], request_context: Any) -> Any:
        progress(f"model request {ctx.run_step}")
        return request_context

    def on_tool(ctx: RunContext[RunInfo], *, call: Any, tool_def: Any, args: Any) -> Any:
        progress(f"tool {call.tool_name} {call.args_as_json_str()}")
        return args

    capabilities: list[Any] = [HandleDeferredToolCalls(handler=handle)]
    if progress:
        capabilities.append(Hooks(before_model_request=on_request, before_tool_execute=on_tool))
    run = agent_def.agent.run(
        input,
        deps=info,
        model=model,
        usage=usage,
        usage_limits=UsageLimits(request_limit=limits.requests),
        capabilities=capabilities,
    )
    with capture_run_messages() as messages:
        try:
            result = asyncio.run(asyncio.wait_for(run, limits.timeout_s))
            status, output = "completed", result.output
        except KeyboardInterrupt:
            status, error = "cancelled", "KeyboardInterrupt"
        except TimeoutError:
            error = f"timed out after {limits.timeout_s}s (limits.timeout_s)"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        finally:
            record = {"run_id": run_id, "agent": name, "model": label, "input": input}
            record |= {"started": started, "ended": _now(), "status": status, "error": error}
            record |= {"output": output, "usage": usage, "messages": messages}
            record_path = agent_def.project_dir / "runs" / name / f"{run_id}.json"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(record, ensure_ascii=False, indent=1, default=to_jsonable_python)
            record_path.write_text(text, encoding="utf-8")
    return RunResult(run_id, name, status, output, error, usage, record_path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
