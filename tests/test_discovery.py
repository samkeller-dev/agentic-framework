from agentfw import load_agent
from helpers import first_request, make_agent

TOOLS = '''
from pydantic_ai import Tool

from extra_for_discovery import imported  # imported function: not a tool


def public(x: int) -> int:
    """A tool."""
    return x


def _private(x: int) -> int:
    """Hidden helper."""
    return x


class NotATool:
    def method(self, x: int) -> int:
        return x


CONST = 3
wrapped_private = Tool(_private, name="wrapped_private")
wrapped_imported = Tool(imported, name="wrapped_imported")
'''


def test_discovery(project):
    (project / "extra_for_discovery.py").write_text(
        'def imported(x: int) -> int:\n    """Imported."""\n    return x\n', encoding="utf-8"
    )
    make_agent(project, "disc", tools=TOOLS)
    seen = first_request(load_agent("disc", project))
    assert seen["tools"] == ["public", "wrapped_imported", "wrapped_private"]


def test_fixture_tools(project):
    seen = first_request(load_agent("tooled", project))
    assert seen["tools"] == ["echo", "whoami", "wipe"]
