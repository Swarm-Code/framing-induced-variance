"""Typed settings for the Multi-Vac harness.

No hard dependency on `pydantic-settings`; settings load from process env and an
optional `.env` file. When no API key is present the harness runs OFFLINE with a
deterministic stub model so every subsystem stays exercisable without a network.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _load_dotenv(path: str | os.PathLike[str] = ".env") -> dict[str, str]:
    """Minimal .env reader (KEY=VALUE lines). No external dependency."""
    data: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return data
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


class Settings(BaseModel):
    """Runtime configuration for the harness."""

    # Provider (Cerebras OpenAI-compatible endpoint by default).
    api_key: str | None = None
    # Additional keys to rotate through on rate-limit (429) for constant uptime.
    fallback_keys: list[str] = Field(default_factory=list)
    base_url: str = "https://api.cerebras.ai/v1"
    model: str = "gemma-4-31b"
    # Vision-capable model id (defaults to the same Gemma 4 multimodal model).
    vision_model: str = "gemma-4-31b"
    # Ordered list of providers to rotate across on rate-limit/error. Each entry is
    # {name, base_url, api_key, model}. When non-empty this supersedes the single
    # api_key path and enables cross-provider failover (e.g. Fireworks -> Cerebras).
    providers: list[dict] = Field(default_factory=list)

    # Where the harness keeps state on disk.
    skills_dir: str = ".multivac/skills"
    sessions_dir: str = ".multivac/sessions"

    # Compaction thresholds (see compaction.py).
    compact_after_messages: int = 40
    compact_keep_recent: int = 8

    # Sub-agent recursion guard.
    max_subagent_depth: int = 3

    # Circuit breaker for runaway self-recursive tool loops (Hermes lesson).
    tool_calls_limit: int = 20
    request_limit: int = 25

    # Force offline (deterministic stub) model even if a key is present.
    offline: bool = Field(default=False)

    @classmethod
    def load(cls, dotenv: str | os.PathLike[str] | None = ".env") -> "Settings":
        env: dict[str, str] = {}
        if dotenv is not None:
            env.update(_load_dotenv(dotenv))
        env.update(dict(os.environ))

        def get(name: str, default: str | None = None) -> str | None:
            val = env.get(name, default)
            return val if val else default

        # Build the ordered multi-provider list from PROVIDER_PRIORITY (e.g.
        # "fireworks,cerebras"). Each provider contributes its own base_url, key,
        # and model id. Unknown/keyless providers are skipped. This enables
        # cross-provider failover so one provider's rate limit doesn't stall a run.
        provider_specs = {
            "fireworks": (
                get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
                get("FIREWORKS_API_KEY"),
                get("FIREWORKS_MODEL"),
            ),
            "cerebras": (
                get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
                get("CEREBRAS_API_KEY") or get("MULTIVAC_API_KEY") or get("OPENAI_API_KEY"),
                get("MODEL_VISION") or get("MODEL_TEXT", "gemma-4-31b"),
            ),
        }
        priority = (get("PROVIDER_PRIORITY", "") or "").split(",")
        providers: list[dict] = []
        for name in [p.strip().lower() for p in priority if p.strip()]:
            spec = provider_specs.get(name)
            if spec and spec[1]:  # has a key
                base, key, model = spec
                providers.append({"name": name, "base_url": base, "api_key": key,
                                  "model": model or "gemma-4-31b"})

        return cls(
            # Accept generic OpenAI, Multi-Vac, or Cerebras key/url names (in that order).
            api_key=(
                get("MULTIVAC_API_KEY")
                or get("OPENAI_API_KEY")
                or get("CEREBRAS_API_KEY")
            ),
            fallback_keys=[
                k for k in [
                    get("CEREBRAS_API_KEY_FALLBACK"),
                    get("CEREBRAS_API_KEY_FALLBACK2"),
                    get("MULTIVAC_API_KEY_FALLBACK"),
                ] if k
            ],
            base_url=(
                get("MULTIVAC_BASE_URL")
                or get("OPENAI_BASE_URL")
                or get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
                or "https://api.cerebras.ai/v1"
            ),
            model=(
                get("MULTIVAC_MODEL")
                or get("OPENAI_MODEL")
                or get("MODEL_TEXT", "gemma-4-31b")
                or "gemma-4-31b"
            ),
            vision_model=(
                get("MULTIVAC_VISION_MODEL")
                or get("MODEL_VISION")
                or get("MODEL_TEXT", "gemma-4-31b")
                or "gemma-4-31b"
            ),
            skills_dir=get("MULTIVAC_SKILLS_DIR", ".multivac/skills") or ".multivac/skills",
            sessions_dir=get("MULTIVAC_SESSIONS_DIR", ".multivac/sessions")
            or ".multivac/sessions",
            compact_after_messages=int(get("MULTIVAC_COMPACT_AFTER", "40") or "40"),
            compact_keep_recent=int(get("MULTIVAC_COMPACT_KEEP", "8") or "8"),
            max_subagent_depth=int(get("MULTIVAC_MAX_SUBAGENT_DEPTH", "3") or "3"),
            tool_calls_limit=int(get("MULTIVAC_TOOL_CALLS_LIMIT", "20") or "20"),
            request_limit=int(get("MULTIVAC_REQUEST_LIMIT", "25") or "25"),
            offline=(get("MULTIVAC_OFFLINE", "") or "").lower() in {"1", "true", "yes"},
            providers=providers,
        )

    @property
    def is_live(self) -> bool:
        """True when we can call a real model."""
        return bool(self.api_key) and not self.offline
