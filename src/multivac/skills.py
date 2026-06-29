"""Skills subsystem.

A *skill* is a reusable, named instruction document stored on disk as a Markdown file
with a small YAML-ish frontmatter block. The harness can:

* discover & list skills,
* view a skill's full body,
* create a new skill,
* patch (update) an existing skill,
* archive a skill.

These operations are exposed to the model as tools (see `as_tools`), so the agent can
read and write its own skills — the read/write requirement. No external deps: a tiny
frontmatter parser keeps this dependency-free.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class Skill(BaseModel):
    name: str
    description: str = ""
    version: int = 1
    tags: list[str] = Field(default_factory=list)
    body: str = ""
    updated_at: str = ""

    def to_markdown(self) -> str:
        tags = ", ".join(self.tags)
        fm = (
            f"name: {self.name}\n"
            f"description: {self.description}\n"
            f"version: {self.version}\n"
            f"tags: {tags}\n"
            f"updated_at: {self.updated_at}\n"
        )
        return f"---\n{fm}---\n{self.body}"

    @classmethod
    def from_markdown(cls, text: str, fallback_name: str) -> "Skill":
        m = _FRONTMATTER_RE.match(text)
        if not m:
            return cls(name=fallback_name, body=text)
        raw_fm, body = m.group(1), m.group(2)
        fields: dict[str, str] = {}
        for line in raw_fm.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                fields[k.strip()] = v.strip()
        tags = [t.strip() for t in fields.get("tags", "").split(",") if t.strip()]
        return cls(
            name=fields.get("name", fallback_name),
            description=fields.get("description", ""),
            version=int(fields.get("version", "1") or "1"),
            tags=tags,
            body=body.strip(),
            updated_at=fields.get("updated_at", ""),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SkillStore:
    """File-backed CRUD store for skills."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.archive_dir = self.root / "_archive"

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.md"

    @staticmethod
    def _validate_name(name: str) -> None:
        if not _NAME_RE.match(name):
            raise ValueError(
                f"invalid skill name {name!r}: use lowercase letters, digits, hyphens"
            )

    def list(self) -> list[Skill]:
        if not self.root.exists():
            return []
        out: list[Skill] = []
        for p in sorted(self.root.glob("*.md")):
            out.append(Skill.from_markdown(p.read_text(), p.stem))
        return out

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def view(self, name: str) -> Skill:
        p = self._path(name)
        if not p.exists():
            raise FileNotFoundError(f"skill not found: {name}")
        return Skill.from_markdown(p.read_text(), name)

    def create(
        self,
        name: str,
        description: str,
        body: str,
        tags: list[str] | None = None,
    ) -> Skill:
        self._validate_name(name)
        if self.exists(name):
            raise FileExistsError(f"skill already exists: {name} (use patch)")
        self.root.mkdir(parents=True, exist_ok=True)
        skill = Skill(
            name=name,
            description=description,
            body=body.strip(),
            tags=tags or [],
            version=1,
            updated_at=_now(),
        )
        self._path(name).write_text(skill.to_markdown())
        return skill

    def patch(
        self,
        name: str,
        description: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
    ) -> Skill:
        skill = self.view(name)
        if description is not None:
            skill.description = description
        if body is not None:
            skill.body = body.strip()
        if tags is not None:
            skill.tags = tags
        skill.version += 1
        skill.updated_at = _now()
        self._path(name).write_text(skill.to_markdown())
        return skill

    def archive(self, name: str) -> None:
        p = self._path(name)
        if not p.exists():
            raise FileNotFoundError(f"skill not found: {name}")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        p.rename(self.archive_dir / p.name)

    # ----------------------------------------------------------- agent tools
    def as_tools(self) -> list:
        """Plain callables to register on a Pydantic AI agent/toolset.

        Each returns a string the model can read; mutations persist to disk.
        """

        def skill_list() -> str:
            """List all available skills with name, version, and description."""
            skills = self.list()
            if not skills:
                return "No skills yet."
            return "\n".join(
                f"- {s.name} (v{s.version}): {s.description}" for s in skills
            )

        def skill_view(name: str) -> str:
            """View the full body of a skill by name."""
            s = self.view(name)
            return s.to_markdown()

        def skill_create(name: str, description: str, body: str) -> str:
            """Create a new reusable skill (lowercase-hyphen name)."""
            s = self.create(name, description, body)
            return f"created skill {s.name} v{s.version}"

        def skill_patch(name: str, body: str, description: str = "") -> str:
            """Update an existing skill's body (and optionally description)."""
            s = self.patch(name, description=description or None, body=body)
            return f"patched skill {s.name} -> v{s.version}"

        return [skill_list, skill_view, skill_create, skill_patch]
