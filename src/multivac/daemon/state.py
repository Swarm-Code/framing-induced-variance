"""Daemon state — providers, profiles, workspaces, and live sessions.

Persisted as JSON under ~/.multivac (registry.json). Sessions hold a live `Multivac`
harness each; only their *metadata* is persisted, the harness is rebuilt on demand.

* Provider  — a named model endpoint (api_key_env / base_url / model). Multi-provider.
* Profile   — a reusable agent config: provider + system prompt + optional bundle.
* Workspace — a working directory + default profile/bundle (project scoping).
* Session   — a live conversation: workspace + profile -> a Multivac instance.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import Settings
from ..harness import Multivac
from .protocol import daemon_home


class Provider(BaseModel):
    name: str
    base_url: str = "https://api.cerebras.ai/v1"
    model: str = "gemma-4-31b"
    api_key_env: str = "CEREBRAS_API_KEY"


class Profile(BaseModel):
    name: str
    provider: str  # provider name
    system_prompt: str | None = None
    bundle: str | None = None  # path to a YAML bundle folder/file (optional)


class Workspace(BaseModel):
    name: str
    path: str
    profile: str | None = None  # default profile for sessions here
    bundle: str | None = None


class SessionMeta(BaseModel):
    id: str
    workspace: str | None = None
    profile: str | None = None
    title: str = ""
    created_at: str = ""


class Registry(BaseModel):
    providers: dict[str, Provider] = Field(default_factory=dict)
    profiles: dict[str, Profile] = Field(default_factory=dict)
    workspaces: dict[str, Workspace] = Field(default_factory=dict)
    sessions: dict[str, SessionMeta] = Field(default_factory=dict)


class DaemonState:
    """Owns the registry + live harness instances."""

    def __init__(self, home: Path | None = None) -> None:
        self.home = Path(home) if home else daemon_home()
        self.registry_path = self.home / "registry.json"
        self.registry = self._load()
        self._live: dict[str, Multivac] = {}
        self._seed_defaults()

    # --------------------------------------------------------------- persistence
    def _load(self) -> Registry:
        if self.registry_path.exists():
            return Registry.model_validate_json(self.registry_path.read_text())
        return Registry()

    def save(self) -> None:
        self.registry_path.write_text(self.registry.model_dump_json(indent=2))

    def _seed_defaults(self) -> None:
        if "cerebras" not in self.registry.providers:
            self.registry.providers["cerebras"] = Provider(
                name="cerebras",
                base_url="https://api.cerebras.ai/v1",
                model="gemma-4-31b",
                api_key_env="CEREBRAS_API_KEY",
            )
        if "openai" not in self.registry.providers:
            self.registry.providers["openai"] = Provider(
                name="openai",
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
                api_key_env="OPENAI_API_KEY",
            )
        if "default" not in self.registry.profiles:
            self.registry.profiles["default"] = Profile(
                name="default", provider="cerebras"
            )
        self.save()

    # ------------------------------------------------------------------ providers
    def add_provider(self, p: Provider) -> Provider:
        self.registry.providers[p.name] = p
        self.save()
        return p

    def remove_provider(self, name: str) -> None:
        self.registry.providers.pop(name, None)
        self.save()

    # ------------------------------------------------------------------- profiles
    def add_profile(self, p: Profile) -> Profile:
        if p.provider not in self.registry.providers:
            raise ValueError(f"unknown provider: {p.provider}")
        self.registry.profiles[p.name] = p
        self.save()
        return p

    def remove_profile(self, name: str) -> None:
        self.registry.profiles.pop(name, None)
        self.save()

    # ----------------------------------------------------------------- workspaces
    def add_workspace(self, w: Workspace) -> Workspace:
        self.registry.workspaces[w.name] = w
        self.save()
        return w

    def remove_workspace(self, name: str) -> None:
        self.registry.workspaces.pop(name, None)
        self.save()

    @staticmethod
    def discover_bundle(path: str | Path) -> str | None:
        """Find a Multi-Vac bundle near `path` (cwd). Returns a path or None.

        Looks for, in order: ./multivac.yaml, ./multivac.yml, ./.multivac/multivac.yaml,
        ./multivac/ (a bundle folder containing *.yaml).
        """
        root = Path(path)
        candidates = [
            root / "multivac.yaml",
            root / "multivac.yml",
            root / ".multivac" / "multivac.yaml",
            root / ".multivac" / "multivac.yml",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        folder = root / "multivac"
        if folder.is_dir() and (list(folder.glob("*.yaml")) or list(folder.glob("*.yml"))):
            return str(folder)
        return None

    def infer_workspace(self, cwd: str) -> Workspace:
        """Get-or-create a workspace for the current directory.

        The workspace is keyed by absolute path; its name is the directory basename.
        A bundle is auto-discovered so a plain `multivac chat` in a project folder loads
        that project's system prompt, tools, skills, hooks and sub-agents.
        """
        abs_cwd = str(Path(cwd).expanduser().resolve())
        for w in self.registry.workspaces.values():
            if str(Path(w.path).resolve()) == abs_cwd:
                return w
        name = Path(abs_cwd).name or "workspace"
        # Avoid name collisions with a different path.
        if name in self.registry.workspaces and self.registry.workspaces[name].path != abs_cwd:
            name = f"{name}-{abs_cwd.strip('/').replace('/', '-')[-12:]}"
        ws = Workspace(
            name=name,
            path=abs_cwd,
            profile="default",
            bundle=self.discover_bundle(abs_cwd),
        )
        self.registry.workspaces[name] = ws
        self.save()
        return ws

    # -------------------------------------------------------------------- sessions
    def _build_harness(self, profile: Profile, workspace: Workspace | None) -> Multivac:
        provider = self.registry.providers.get(profile.provider)
        if provider is None:
            raise ValueError(f"profile {profile.name!r} references unknown provider")

        bundle = profile.bundle or (workspace.bundle if workspace else None)
        if bundle:
            # Bundle wins; provider/profile still override via settings base.
            base = self._settings_for(provider, workspace)
            return Multivac.from_config(bundle, settings=base)

        settings = self._settings_for(provider, workspace)
        return Multivac(settings, system_prompt=profile.system_prompt)

    def _settings_for(self, provider: Provider, workspace: Workspace | None) -> Settings:
        import os

        s = Settings.load()
        return s.model_copy(
            update={
                "api_key": os.environ.get(provider.api_key_env) or s.api_key,
                "base_url": provider.base_url,
                "model": provider.model,
                "skills_dir": (
                    str(Path(workspace.path) / ".multivac" / "skills")
                    if workspace
                    else s.skills_dir
                ),
            }
        )

    def create_session(
        self,
        *,
        workspace: str | None = None,
        profile: str | None = None,
        title: str = "",
    ) -> SessionMeta:
        from datetime import datetime, timezone

        ws = self.registry.workspaces.get(workspace) if workspace else None
        prof_name = profile or (ws.profile if ws else None) or "default"
        prof = self.registry.profiles.get(prof_name)
        if prof is None:
            raise ValueError(f"unknown profile: {prof_name}")

        sid = uuid.uuid4().hex[:12]
        meta = SessionMeta(
            id=sid,
            workspace=workspace,
            profile=prof_name,
            title=title or f"session-{sid}",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.registry.sessions[sid] = meta
        self._live[sid] = self._build_harness(prof, ws)
        self.save()
        return meta

    def harness(self, session_id: str) -> Multivac:
        if session_id not in self._live:
            meta = self.registry.sessions.get(session_id)
            if meta is None:
                raise KeyError(f"unknown session: {session_id}")
            ws = self.registry.workspaces.get(meta.workspace) if meta.workspace else None
            prof = self.registry.profiles.get(meta.profile or "default")
            if prof is None:
                raise ValueError(f"session {session_id} references unknown profile")
            self._live[session_id] = self._build_harness(prof, ws)
        return self._live[session_id]

    def close_session(self, session_id: str) -> None:
        self._live.pop(session_id, None)
        self.registry.sessions.pop(session_id, None)
        self.save()

    def snapshot(self) -> dict[str, Any]:
        return self.registry.model_dump()
