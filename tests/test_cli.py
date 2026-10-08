import io
import json
import os
import subprocess
import sys

import pytest

from agentfw.cli import main
from helpers import TOOLS_SRC, UNICODE, make_agent, make_skill


def records(project, agent):
    files = (project / "runs" / agent).glob("*.json")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def test_run_completed(project, capsys):
    assert main(["run", "plain", "--input", "hi", "--project-dir", str(project)]) == 0
    out, err = capsys.readouterr()
    assert out == "success (no tool calls)\n"
    assert "model request 1\n" in err and "completed: " in err


def test_run_structured_output_is_json(project, capsys):
    assert main(["run", "structured", "--input", "hi", "--project-dir", str(project)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) == {"count", "items"}


def test_run_failed_exit_1(project, capsys):
    make_agent(project, "capped", yaml="model: test\nlimits: {requests: 1}\n", tools=TOOLS_SRC)
    assert main(["run", "capped", "--input", "hi", "--project-dir", str(project)]) == 1
    out, err = capsys.readouterr()
    assert out == "" and "failed: " in err and "request_limit of 1" in err


def test_run_config_error_exit_2(project, capsys):
    assert main(["run", "missing", "--input", "hi", "--project-dir", str(project)]) == 2
    assert "error: " in capsys.readouterr().err


def test_run_cancelled_exit_130(project, capsys):
    tools = '''
        def stop() -> str:
            """Stop everything."""
            raise KeyboardInterrupt
    '''
    make_agent(project, "ki", tools=tools)
    assert main(["run", "ki", "--input", "hi", "--project-dir", str(project)]) == 130
    assert "cancelled: " in capsys.readouterr().err


def test_run_input_from_stdin(project, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    assert main(["run", "plain", "--project-dir", str(project)]) == 0
    assert [r["input"] for r in records(project, "plain")] == ["from stdin"]


def test_run_no_input_on_tty(project, monkeypatch, capsys):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Tty())
    assert main(["run", "plain", "--project-dir", str(project)]) == 2
    assert "no input" in capsys.readouterr().err


def test_run_input_file(project, tmp_path, capsys):
    f = tmp_path / "in.txt"
    f.write_text(UNICODE, encoding="utf-8")
    assert main(["run", "plain", "--input-file", str(f), "--project-dir", str(project)]) == 0
    assert [r["input"] for r in records(project, "plain")] == [UNICODE]


def test_run_model_override(project, capsys):
    args = ["run", "plain", "--input", "hi", "--model", "test", "--project-dir", str(project)]
    assert main(args) == 0
    assert [r["model"] for r in records(project, "plain")] == ["test"]


def test_check_all_ok(project, capsys):
    assert main(["check", "--project-dir", str(project)]) == 0
    assert capsys.readouterr().out == "plain: ok\nstructured: ok\ntooled: ok\n"


def test_check_reports_each_error(project, capsys):
    make_agent(project, "bad-yaml", yaml="model: test\ninstructions: no\n")
    make_agent(project, "bad-tools", tools="import nonexistent_module_xyz\n")
    assert main(["check", "--project-dir", str(project)]) == 2
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("bad-tools: ") and "tools.py: ModuleNotFoundError" in lines[0]
    assert lines[1].startswith("bad-yaml: ") and "agent.yaml" in lines[1]
    assert lines[2:] == ["plain: ok", "structured: ok", "tooled: ok"]


def test_check_named_agents(project, capsys):
    assert main(["check", "plain", "nope", "--project-dir", str(project)]) == 2
    assert capsys.readouterr().out.startswith("plain: ok\nnope: ")


def test_check_discovers_project_from_cwd(project, monkeypatch, capsys):
    monkeypatch.chdir(project / "agents")
    assert main(["check"]) == 0


def test_check_no_project(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["check"]) == 2
    assert "error: no project dir" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.startswith("agentfw 0.")


@pytest.mark.parametrize("utf8_mode", ["0", "1"])
def test_utf8_round_trip_in_subprocess(project, utf8_mode):
    # TestModel passes "a" for string arguments, so a skill named "a" gets read.
    make_skill(project, "skills/a", f"name: a\ndescription: {UNICODE}", body=f"Skill {UNICODE}")
    tools = f'''
        def shout(text: str) -> str:
            """Shout."""
            return "tool " + text + " {UNICODE}"
    '''
    yaml = "model: test\nskills: [skills/a]\n"
    make_agent(project, "uni", yaml=yaml, instructions=f"Instructions {UNICODE}", tools=tools)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    env["PYTHONUTF8"] = utf8_mode
    cmd = [sys.executable, "-c", "import sys; from agentfw.cli import main; sys.exit(main())"]
    cmd += ["run", "uni", "--project-dir", str(project)]

    p = subprocess.run(
        [*cmd, "--input", f"Input {UNICODE}"],
        capture_output=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    out = p.stdout.decode("utf-8")
    assert f"tool a {UNICODE}" in out
    assert f"Skill {UNICODE}" in out

    p = subprocess.run(cmd, input=f"Piped {UNICODE}".encode(), capture_output=True, env=env)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")

    recs = records(project, "uni")
    assert sorted(r["input"] for r in recs) == [f"Input {UNICODE}", f"Piped {UNICODE}"]
    for rec in recs:
        assert f"Instructions {UNICODE}" in rec["messages"][0]["instructions"]
        assert f"tool a {UNICODE}" in json.dumps(rec["messages"], ensure_ascii=False)
