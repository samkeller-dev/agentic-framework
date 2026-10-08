import os

import pytest

from agentfw import ConfigError, load_agent
from agentfw.loader import find_project_dir
from helpers import call_tool, first_request, make_agent, make_skill


def test_project_dir_discovered_from_cwd(project, monkeypatch):
    monkeypatch.chdir(project / "agents" / "plain")
    assert load_agent("plain").project_dir == project.resolve()


def test_no_project_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="no ancestor of .* contains agents/"):
        find_project_dir()


def test_agent_def_fields(project):
    d = load_agent("plain", project)
    assert (d.name, d.path, d.project_dir) == ("plain", project / "agents" / "plain", project)
    assert (d.limits.requests, d.limits.timeout_s) == (20, 900)


def test_dotenv_loaded_but_never_overrides(project, monkeypatch):
    monkeypatch.setenv("AGENTFW_TEST_A", "preset")
    monkeypatch.delenv("AGENTFW_TEST_B", raising=False)
    (project / ".env").write_text("AGENTFW_TEST_A=from_env\nAGENTFW_TEST_B=from_env\n", "utf-8")
    load_agent("plain", project)
    assert os.environ["AGENTFW_TEST_A"] == "preset"
    assert os.environ["AGENTFW_TEST_B"] == "from_env"


@pytest.mark.parametrize("name", ["Plain", "-x", "a_b", "a" * 65, "..", "", "a/b"])
def test_agent_name_regex(project, name):
    with pytest.raises(ConfigError, match="invalid agent name"):
        load_agent(name, project)


def test_missing_agent(project):
    with pytest.raises(ConfigError, match=r"agent\.yaml: not found"):
        load_agent("nope", project)


def test_missing_instructions(project):
    (project / "agents" / "plain" / "instructions.md").unlink()
    with pytest.raises(ConfigError, match=r"instructions\.md: not found"):
        load_agent("plain", project)


def test_empty_instructions(project):
    (project / "agents" / "plain" / "instructions.md").write_text("  \n\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"instructions\.md: empty"):
        load_agent("plain", project)


def test_yaml_instructions_key_rejected(project):
    make_agent(project, "bad", yaml="model: test\ninstructions: hi\n")
    with pytest.raises(ConfigError, match=r"agent\.yaml: put instructions in instructions\.md"):
        load_agent("bad", project)


def test_unknown_yaml_key(project):
    make_agent(project, "bad", yaml="model: test\ntemperature: 1\n")
    with pytest.raises(ConfigError, match=r"agent\.yaml: unknown key\(s\) \['temperature'\]"):
        load_agent("bad", project)


def test_unknown_limits_key(project):
    make_agent(project, "bad", yaml="model: test\nlimits: {tokens: 5}\n")
    with pytest.raises(ConfigError, match=r"unknown key\(s\) \['limits\.tokens'\]"):
        load_agent("bad", project)


def test_invalid_yaml(project):
    make_agent(project, "bad", yaml="model: [\n")
    with pytest.raises(ConfigError, match=r"agent\.yaml: "):
        load_agent("bad", project)


def test_yaml_must_be_mapping(project):
    make_agent(project, "bad", yaml="- just\n- a list\n")
    with pytest.raises(ConfigError, match=r"agent\.yaml: must be a mapping"):
        load_agent("bad", project)


def test_spec_fields_pass_through(project):
    yaml = (
        "model: test\nmodel_settings: {max_tokens: 10}\nretries: 2\n"
        "capabilities:\n  - Thinking: {effort: medium}\nlimits: {requests: 3, timeout_s: 1.5}\n"
    )
    make_agent(project, "rich", yaml=yaml)
    d = load_agent("rich", project)
    assert (d.limits.requests, d.limits.timeout_s) == (3, 1.5)
    assert d.agent.name == "rich"


def test_spec_validation_error_names_file(project):
    make_agent(project, "bad", yaml="model: test\ncapabilities: [NoSuchCapability]\n")
    with pytest.raises(ConfigError, match=r"agent\.yaml: .*NoSuchCapability"):
        load_agent("bad", project)


def test_missing_provider_key_names_file_and_cause(project, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    make_agent(project, "cloud", yaml="model: openai:gpt-5.5\n")
    with pytest.raises(ConfigError, match=r"agent\.yaml: .*OPENAI_API_KEY"):
        load_agent("cloud", project)


def test_tools_import_error(project):
    make_agent(project, "bad", tools="import nonexistent_module_xyz\n")
    with pytest.raises(ConfigError, match=r"tools\.py: ModuleNotFoundError: .*nonexistent_module"):
        load_agent("bad", project)


def test_tools_can_import_sibling_modules(project):
    (project / "shared_for_test.py").write_text("VALUE = 'shared'\n", encoding="utf-8")
    tools = '''
        from shared_for_test import VALUE

        def get() -> str:
            """Get the shared value."""
            return VALUE
    '''
    make_agent(project, "sib", tools=tools)
    assert call_tool(load_agent("sib", project), "get") == "shared"


def test_output_class_is_output_type(project):
    assert load_agent("structured", project).agent.output_type.__name__ == "Output"


def test_text_output_without_output_class(project):
    assert load_agent("plain", project).agent.output_type is str
    assert load_agent("tooled", project).agent.output_type is str


def test_non_class_output_rejected(project):
    make_agent(project, "bad", tools="Output = 'text'\n")
    with pytest.raises(ConfigError, match=r"tools\.py: Output must be a class, got str"):
        load_agent("bad", project)


@pytest.mark.parametrize(
    ("frontmatter", "cause"),
    [
        (None, "no SKILL.md"),
        ("description: only", "needs non-empty name and description"),
        ("name: only", "needs non-empty name and description"),
        ("name: ''\ndescription: x", "needs non-empty name and description"),
        ("name: [a, b]\ndescription: x", "needs non-empty name and description"),
        ("just text, no mapping", "needs non-empty name and description"),
    ],
)
def test_invalid_skill(project, frontmatter, cause):
    if frontmatter is None:
        (project / "skills" / "s").mkdir(parents=True)
    else:
        make_skill(project, "skills/s", frontmatter)
    make_agent(project, "sk", yaml="model: test\nskills: [skills/s]\n")
    with pytest.raises(ConfigError, match=cause) as info:
        load_agent("sk", project)
    assert str(project / "skills" / "s") in str(info.value)


def test_skill_without_frontmatter(project):
    (project / "skills" / "s").mkdir(parents=True)
    (project / "skills" / "s" / "SKILL.md").write_text("# No frontmatter\n", encoding="utf-8")
    make_agent(project, "sk", yaml="model: test\nskills: [skills/s]\n")
    with pytest.raises(ConfigError, match="needs non-empty name and description"):
        load_agent("sk", project)


def test_duplicate_skill_names(project):
    make_skill(project, "skills/a", "name: same\ndescription: one")
    make_skill(project, "skills/b", "name: same\ndescription: two")
    make_agent(project, "sk", yaml="model: test\nskills: [skills/a, skills/b]\n")
    with pytest.raises(ConfigError, match="duplicate skill name 'same'"):
        load_agent("sk", project)


def test_skills_section_appended_to_instructions(project):
    make_skill(project, "skills/summarise")
    make_skill(project, "other/extract", "name: extract\ndescription: Pull out figures.")
    yaml = "model: test\nskills: [skills/summarise, other/extract]\n"
    make_agent(project, "sk", yaml=yaml, instructions="Base.\n\n")
    seen = first_request(load_agent("sk", project))
    assert seen["instructions"].startswith("Base.\n\n## Skills\n\n")
    assert "call read_skill(name)" in seen["instructions"]
    assert seen["instructions"].endswith(
        "- summarise: How to summarise.\n- extract: Pull out figures."
    )
    assert "read_skill" in seen["tools"]


def test_no_skills_means_no_section_and_no_tool(project):
    seen = first_request(load_agent("plain", project))
    assert seen["instructions"] == "Be brief."
    assert seen["tools"] == []
