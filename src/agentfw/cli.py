"""`agentfw run` and `agentfw check`."""

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

from pydantic_core import to_jsonable_python

from .loader import ConfigError, find_project_dir, load_agent
from .runner import run_agent


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        getattr(stream, "reconfigure", lambda **_: None)(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="agentfw", description="Run folder-defined agents.")
    parser.add_argument("--version", action="version", version=f"agentfw {version('agentfw')}")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one agent; input from stdin when no --input flag")
    run.add_argument("agent")
    run.add_argument("--input", help="input text")
    run.add_argument("--input-file", help="read input from this UTF-8 file")
    run.add_argument("--model", help="override the model, e.g. openai:gpt-5")
    check = sub.add_parser("check", help="load agents and report config errors")
    check.add_argument("agents", nargs="*", help="agent names (default: all)")
    for p in (run, check):
        p.add_argument("--project-dir", help="project root (default: nearest parent with agents/)")
    args = parser.parse_args(argv)
    try:
        return _check(args) if args.command == "check" else _run(args)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    agent_def = load_agent(args.agent, args.project_dir)
    if args.input is not None:
        text = args.input
    elif args.input_file:
        text = Path(args.input_file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise ConfigError("no input: pass --input, --input-file, or pipe text on stdin")
    log = lambda line: print(line, file=sys.stderr)  # noqa: E731
    result = run_agent(agent_def, text, model=args.model, progress=log)
    log(f"{result.status}: {result.record_path}")
    if result.status != "completed":
        log(result.error)
        return 130 if result.status == "cancelled" else 1
    out = result.output
    print(out if isinstance(out, str) else json.dumps(to_jsonable_python(out), ensure_ascii=False))
    return 0


def _check(args: argparse.Namespace) -> int:
    project_dir = find_project_dir(Path(args.project_dir) if args.project_dir else None)
    names = args.agents or sorted(p.name for p in (project_dir / "agents").iterdir() if p.is_dir())
    code = 0
    for name in names:
        try:
            load_agent(name, project_dir)
            print(f"{name}: ok")
        except ConfigError as e:
            print(f"{name}: {e}")
            code = 2
    return code
