"""Offline test: the agent loads and runs against PydanticAI's TestModel. No key, no network."""

from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from agentfw import load_agent, run_agent

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def summariser(monkeypatch):
    # Loading constructs the model, which needs a key to exist; no request is ever made.
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test")
    return load_agent("summariser", PROJECT)


def test_runs_offline(summariser):
    with summariser.agent.override(model=TestModel()):
        result = run_agent(summariser, "Quarterly revenue grew 12% while costs fell 3%.")
    assert result.status == "completed", result.error
    assert isinstance(result.output.word_count, int)
    assert isinstance(result.output.summary, str)
    assert result.record_path.is_file()
