"""Live provider tests. Run manually with `pytest -m live`; skipped when the key is absent.

Override a model with AGENTFW_LIVE_MODEL_GOOGLE / _ANTHROPIC / _OPENAI.
"""

import os

import pytest
from pydantic_ai import models

from agentfw import load_agent, run_agent
from helpers import TOOLS_SRC, make_agent

pytestmark = pytest.mark.live

PROVIDERS = {
    "google": ("GOOGLE_API_KEY", "google:gemini-3.7-flash"),
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic:claude-sonnet-5-5"),
    "openai": ("OPENAI_API_KEY", "openai:gpt-5.5"),
}

CHAIN_TOOLS = '''
def get_code() -> str:
    """Return today's access code."""
    return "ZX-41"


def lookup(code: str) -> str:
    """Look up the city for an access code."""
    return {"ZX-41": "Lisbon"}.get(code, "unknown code")
'''


@pytest.fixture(params=list(PROVIDERS))
def model(request):
    key, default = PROVIDERS[request.param]
    if not os.environ.get(key):
        pytest.skip(f"{key} not set")
    return os.environ.get(f"AGENTFW_LIVE_MODEL_{request.param.upper()}", default)


@pytest.fixture(autouse=True)
def allow_requests():
    with models.override_allow_model_requests(True):
        yield


def test_two_sequential_tool_rounds(tmp_path, model):
    instructions = (
        "First call get_code. Then call lookup with the code it returned. "
        "Reply with only the city name."
    )
    make_agent(
        tmp_path, "chain", yaml=f"model: {model}\n", instructions=instructions, tools=CHAIN_TOOLS
    )
    r = run_agent(load_agent("chain", tmp_path), "Which city does today's code map to?")
    assert r.status == "completed", r.error
    assert "Lisbon" in r.output
    assert r.usage.tool_calls >= 2 and r.usage.requests >= 3


def test_structured_output(tmp_path, model):
    make_agent(
        tmp_path,
        "struct",
        yaml=f"model: {model}\n",
        instructions="Return exactly what is asked.",
        tools=TOOLS_SRC,
    )
    r = run_agent(load_agent("struct", tmp_path), "count = 3; items = red, green, blue")
    assert r.status == "completed", r.error
    assert r.output.count == 3
    assert [i.lower() for i in r.output.items] == ["red", "green", "blue"]
