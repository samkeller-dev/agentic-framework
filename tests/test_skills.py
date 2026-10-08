import pytest

from agentfw import load_agent
from helpers import UNICODE, call_tool, make_agent, make_skill


@pytest.fixture
def skilled(project):
    d = make_skill(project, "skills/summarise", body=f"Summarise. {UNICODE}")
    (d / "references").mkdir()
    (d / "references" / "style.md").write_text(f"Style {UNICODE}", encoding="utf-8")
    make_agent(project, "sk", yaml="model: test\nskills: [skills/summarise]\n")
    return load_agent("sk", project)


def test_read_skill_md(skilled):
    out = call_tool(skilled, "read_skill", name="summarise")
    assert out.startswith("---\nname: summarise\n") and out.endswith(f"Summarise. {UNICODE}\n")


def test_read_reference_file(skilled):
    out = call_tool(skilled, "read_skill", name="summarise", path="references/style.md")
    assert out == f"Style {UNICODE}"


def test_unknown_skill(skilled):
    out = call_tool(skilled, "read_skill", name="nope")
    assert out == "Error: unknown skill 'nope'. Available: ['summarise']"


def test_missing_file(skilled):
    out = call_tool(skilled, "read_skill", name="summarise", path="references/none.md")
    assert out.startswith("Error: cannot read 'references/none.md'")


def test_path_traversal(skilled, project):
    for path in ["../../agents/sk/agent.yaml", str(project / "agents" / "sk" / "agent.yaml")]:
        out = call_tool(skilled, "read_skill", name="summarise", path=path)
        assert out == f"Error: {path!r} is outside the skill directory"
