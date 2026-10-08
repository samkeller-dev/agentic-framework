"""Turn an agent folder into a PydanticAI Agent."""

import importlib.util
import re
import sys
from dataclasses import dataclass
from inspect import isclass, isfunction
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic_ai import Agent, Tool
from pydantic_ai.agent.spec import AgentSpec


class ConfigError(Exception):
    """A problem with the project or agent folder. The message names the file."""


NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SPEC_KEYS = {f.alias or n for n, f in AgentSpec.model_fields.items()} - {"instructions"}
LIMIT_KEYS = {"requests", "timeout_s"}
SKILLS_PROMPT = (
    "\n\n## Skills\n\nBefore using a skill, call read_skill(name) and follow what it returns. "
    "Use read_skill(name, path) to read a file the skill references.\n\n"
)


@dataclass(frozen=True)
class RunInfo:  # passed to tools as ctx.deps
    project_dir: Path
    agent: str
    run_id: str


@dataclass(frozen=True)
class Limits:
    requests: int = 20
    timeout_s: float = 900


@dataclass(frozen=True)
class AgentDef:
    name: str
    path: Path
    project_dir: Path
    agent: Agent[RunInfo, Any]
    limits: Limits


def find_project_dir(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for d in (start, *start.parents):
        if (d / "agents").is_dir():
            return d
    raise ConfigError(f"no project dir: no ancestor of {start} contains agents/")


def load_agent(name: str, project_dir: Path | str | None = None) -> AgentDef:
    project_dir = Path(project_dir).resolve() if project_dir else find_project_dir()
    load_dotenv(project_dir / ".env", override=False)
    if not NAME_RE.match(name):
        raise ConfigError(f"invalid agent name {name!r}: must match {NAME_RE.pattern}")
    path = project_dir / "agents" / name
    yaml_path = path / "agent.yaml"
    spec = _read_yaml(yaml_path)
    instructions = _read_text(path / "instructions.md")
    if "instructions" in spec:
        raise ConfigError(f"{yaml_path}: put instructions in instructions.md, not agent.yaml")
    skills, limits = spec.pop("skills", None) or [], spec.pop("limits", None) or {}
    if unknown := (set(spec) - SPEC_KEYS) | {f"limits.{k}" for k in set(limits) - LIMIT_KEYS}:
        raise ConfigError(f"{yaml_path}: unknown key(s) {sorted(unknown)}")
    tools, output_type = _load_tools(path / "tools.py", project_dir)
    skill_tools, skill_text = _load_skills(project_dir, skills)
    try:
        agent = Agent.from_spec(
            spec,
            deps_type=RunInfo,
            name=name,
            instructions=instructions.rstrip() + skill_text,
            tools=[*tools, *skill_tools],
            output_type=output_type,
        )
    except Exception as e:
        raise ConfigError(f"{yaml_path}: {e}") from e
    return AgentDef(name, path, project_dir, agent, Limits(**limits))


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ConfigError(f"{path}: not found")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ConfigError(f"{path}: empty")
    return text


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(_read_text(path))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: must be a mapping")
    return data


def _load_tools(file: Path, project_dir: Path) -> tuple[list[Any], type]:
    """Public functions defined in tools.py plus Tool instances; `Output` class if present."""
    if not file.is_file():
        return [], str
    modname = f"agentfw_tools.{file.parent.name.replace('-', '_')}"
    module_spec = importlib.util.spec_from_file_location(modname, file)
    module = importlib.util.module_from_spec(module_spec)  # type: ignore[arg-type]
    sys.modules[modname] = module
    sys.path.insert(0, str(project_dir))
    try:
        module_spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:
        raise ConfigError(f"{file}: {type(e).__name__}: {e}") from e
    finally:
        sys.path.remove(str(project_dir))
    ns = vars(module)
    own = [o for k, o in ns.items() if k[0] != "_" and isfunction(o) and o.__module__ == modname]
    tools = own + [o for o in ns.values() if isinstance(o, Tool)]
    output = ns.get("Output", str)
    if not isclass(output):
        raise ConfigError(f"{file}: Output must be a class, got {type(output).__name__}")
    return tools, output


def _load_skills(project_dir: Path, dirs: list[str]) -> tuple[list[Tool], str]:
    """Validate skill folders; return the `read_skill` tool and the instructions section."""
    skills: dict[str, tuple[Path, str]] = {}
    for d in dirs:
        root = (project_dir / d).resolve()
        md = root / "SKILL.md"
        if not md.is_file():
            raise ConfigError(f"{root}: no SKILL.md")
        m = re.match(r"---\r?\n(.*?)\r?\n---", md.read_text(encoding="utf-8"), re.S)
        meta = yaml.safe_load(m.group(1)) if m else None
        name, desc = (
            (meta.get("name"), meta.get("description")) if isinstance(meta, dict) else (0, 0)
        )
        if not (isinstance(name, str) and name and isinstance(desc, str) and desc):
            raise ConfigError(f"{md}: frontmatter needs non-empty name and description")
        if name in skills:
            raise ConfigError(f"{root}: duplicate skill name {name!r}")
        skills[name] = (root, desc)
    if not skills:
        return [], ""

    def read_skill(name: str, path: str = "SKILL.md") -> str:
        """Read a file from a skill's directory: SKILL.md by default, or a path it references."""
        if name not in skills:
            return f"Error: unknown skill {name!r}. Available: {sorted(skills)}"
        root = skills[name][0]
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            return f"Error: {path!r} is outside the skill directory"
        try:
            return target.read_text(encoding="utf-8")
        except OSError as e:
            return f"Error: cannot read {path!r}: {e.strerror or e}"

    listing = "\n".join(f"- {n}: {d}" for n, (_, d) in skills.items())
    return [Tool(read_skill)], SKILLS_PROMPT + listing
