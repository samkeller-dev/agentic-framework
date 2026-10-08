import json
import re

from pydantic_ai import ModelMessagesTypeAdapter, ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentfw import load_agent, run_agent
from agentfw.runner import DENIED_BY_USER, DENIED_UNATTENDED
from helpers import TOOLS_SRC, calling, make_agent, replying


def record_of(result):
    """Load the run record, check it matches the result, and that messages round-trip."""
    assert result.record_path.is_file()
    data = json.loads(result.record_path.read_text(encoding="utf-8"))
    assert (data["run_id"], data["agent"]) == (result.run_id, result.agent)
    assert (data["status"], data["error"]) == (result.status, result.error)
    assert data["started"] <= data["ended"]
    ModelMessagesTypeAdapter.validate_python(data["messages"])
    return data


def test_text_reply(project):
    r = run_agent(load_agent("plain", project), "hello", model=replying("hi there"))
    assert (r.status, r.output, r.error) == ("completed", "hi there", None)
    assert r.usage.requests == 1
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{4}", r.run_id)
    assert r.record_path == project / "runs" / "plain" / f"{r.run_id}.json"
    rec = record_of(r)
    assert (rec["input"], rec["output"], rec["usage"]["requests"]) == ("hello", "hi there", 1)
    assert isinstance(rec["model"], str) and rec["model"]
    assert rec["messages"][0]["parts"][0]["content"] == "hello"


def test_tool_call_then_reply(project):
    r = run_agent(load_agent("tooled", project), "go", model=calling("echo", {"text": "pong"}))
    assert (r.status, r.output) == ("completed", "pong")
    assert (r.usage.requests, r.usage.tool_calls) == (2, 1)
    kinds = [p["part_kind"] for m in record_of(r)["messages"] for p in m["parts"]]
    assert kinds == ["user-prompt", "tool-call", "tool-return", "text"]


def test_tool_receives_run_info(project):
    r = run_agent(load_agent("tooled", project), "go", model=calling("whoami", {}))
    assert r.output == f"tooled|{r.run_id}|{project.resolve().as_posix()}"


def test_structured_output_valid(project):
    def fn(messages, info):
        tool = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool, {"count": 2, "items": ["a", "b"]})])

    r = run_agent(load_agent("structured", project), "go", model=FunctionModel(fn))
    assert r.status == "completed", r.error
    assert (r.output.count, r.output.items) == (2, ["a", "b"])
    assert record_of(r)["output"] == {"count": 2, "items": ["a", "b"]}


def test_structured_output_invalid(project):
    def fn(messages, info):
        tool = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool, {"count": "many", "items": "x"})])

    r = run_agent(load_agent("structured", project), "go", model=FunctionModel(fn))
    assert r.status == "failed" and r.output is None
    assert "retries" in r.error
    assert record_of(r)["output"] is None


def test_requests_limit(project):
    make_agent(project, "capped", yaml="model: test\nlimits: {requests: 2}\n", tools=TOOLS_SRC)
    looping = FunctionModel(lambda m, i: ModelResponse(parts=[ToolCallPart("echo", {"text": "x"})]))
    r = run_agent(load_agent("capped", project), "go", model=looping)
    assert r.status == "failed"
    assert "request_limit of 2" in r.error
    assert r.usage.requests == 2
    assert record_of(r)["usage"]["requests"] == 2


def test_timeout(project):
    tools = '''
        import asyncio

        async def sleep() -> str:
            """Sleep for a while."""
            await asyncio.sleep(5)
            return "late"
    '''
    make_agent(project, "slow", yaml="model: test\nlimits: {timeout_s: 0.2}\n", tools=tools)
    r = run_agent(load_agent("slow", project), "go", model=calling("sleep", {}))
    assert (r.status, r.error) == ("failed", "timed out after 0.2s (limits.timeout_s)")
    record_of(r)


def test_approval_denied_by_default(project):
    r = run_agent(load_agent("tooled", project), "go", model=calling("wipe", {"path": "/x"}))
    assert (r.status, r.output) == ("completed", DENIED_UNATTENDED)
    returns = [
        p for m in record_of(r)["messages"] for p in m["parts"] if p["part_kind"] == "tool-return"
    ]
    assert returns[0]["tool_name"] == "wipe" and returns[0]["content"] == DENIED_UNATTENDED


def test_approval_granted(project):
    seen = []

    def approve(name, args):
        seen.append((name, args))
        return True

    model = calling("wipe", {"path": "/x"})
    r = run_agent(load_agent("tooled", project), "go", model=model, approve=approve)
    assert (r.status, r.output) == ("completed", "wiped /x")
    assert seen == [("wipe", {"path": "/x"})]


def test_approval_refused(project):
    model = calling("wipe", {"path": "/x"})
    r = run_agent(load_agent("tooled", project), "go", model=model, approve=lambda n, a: False)
    assert (r.status, r.output) == ("completed", DENIED_BY_USER)


def test_model_raises(project):
    def fn(messages, info):
        raise RuntimeError("kaput")

    r = run_agent(load_agent("plain", project), "go", model=FunctionModel(fn))
    assert (r.status, r.output, r.error) == ("failed", None, "RuntimeError: kaput")
    assert record_of(r)["messages"][0]["parts"][0]["content"] == "go"


def test_keyboard_interrupt(project):
    # Async so the interrupt is raised on the event loop thread, as a real Ctrl+C is. A sync
    # model function runs on an anyio worker thread, and raising there orphans that thread,
    # which then keeps the interpreter alive after the test session ends.
    async def fn(messages, info):
        raise KeyboardInterrupt

    r = run_agent(load_agent("plain", project), "go", model=FunctionModel(fn))
    assert (r.status, r.error) == ("cancelled", "KeyboardInterrupt")
    record_of(r)


def test_model_override_by_name(project):
    r = run_agent(load_agent("plain", project), "go", model="test")
    assert r.status == "completed"
    assert record_of(r)["model"] == "test"


def test_agent_override_context(project):
    d = load_agent("plain", project)
    with d.agent.override(model=replying("overridden")):
        r = run_agent(d, "go")
    assert r.output == "overridden"


def test_progress_callback(project):
    lines = []
    model = calling("echo", {"text": "x"})
    run_agent(load_agent("tooled", project), "go", model=model, progress=lines.append)
    assert lines == ["model request 1", 'tool echo {"text":"x"}', "model request 2"]


def test_record_has_no_api_keys(project, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    r = run_agent(load_agent("tooled", project), "go", model=calling("echo", {"text": "x"}))
    assert "sk-super-secret-value" not in r.record_path.read_text(encoding="utf-8")
