import os
from pathlib import Path

import pytest
from pydantic_ai import models

from helpers import STRUCTURED_SRC, TOOLS_SRC, make_agent

models.ALLOW_MODEL_REQUESTS = False
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Agents: `plain` (no tools), `tooled` (tools.py), `structured` (tools.py with Output)."""
    make_agent(tmp_path, "plain")
    make_agent(tmp_path, "tooled", tools=TOOLS_SRC)
    make_agent(tmp_path, "structured", tools=STRUCTURED_SRC)
    return tmp_path
