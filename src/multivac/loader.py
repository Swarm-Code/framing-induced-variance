"""Bundle loader + dotted-path resolver.

Turns a YAML file or a *bundle folder* into a validated `HarnessConfig`, then resolves:

* prompt file references (`*_file:` keys) into text,
* dotted import specs (`module.path:attr`) into real Python callables.

The bundle's own directory is put on `sys.path` during import resolution, so a bundle
can ship its own `hooks.py` / `tools.py` and reference them as ``hooks:my_hook`` without
being an installed package.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from .configschema import HarnessConfig

_MERGE_LIST_KEYS = {"skills", "hooks", "subagents", "mcp_servers"}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Merge overlay into base. Lists in known list-keys concatenate; dicts merge."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        elif k in out and isinstance(out[k], list) and isinstance(v, list):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> tuple[HarnessConfig, Path]:
    """Load a bundle from a YAML file or a folder. Returns (config, bundle_dir).

    A folder may contain multiple ``*.yaml``/``*.yml`` files; they are merged in sorted
    filename order (later files override / extend earlier ones).
    """
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        bundle_dir = p
        files = sorted([*p.glob("*.yaml"), *p.glob("*.yml")])
        if not files:
            raise FileNotFoundError(f"no .yaml/.yml files in bundle folder: {p}")
    else:
        bundle_dir = p.parent
        files = [p]

    merged: dict[str, Any] = {}
    for f in files:
        data = yaml.safe_load(f.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"bundle file {f} must be a YAML mapping, got {type(data)}")
        merged = _deep_merge(merged, data)

    return HarnessConfig.model_validate(merged), bundle_dir


def read_prompt(
    bundle_dir: Path, inline: str | None, file_ref: str | None
) -> str | None:
    """Resolve a prompt from an inline string or a file reference (bundle-relative)."""
    if inline is not None:
        return inline
    if file_ref:
        fp = (bundle_dir / file_ref).resolve()
        if not fp.exists():
            raise FileNotFoundError(f"prompt file not found: {fp}")
        return fp.read_text().strip()
    return None


@contextlib.contextmanager
def _bundle_on_path(bundle_dir: Path) -> Iterator[None]:
    s = str(bundle_dir)
    added = s not in sys.path
    if added:
        sys.path.insert(0, s)
    try:
        yield
    finally:
        if added:
            with contextlib.suppress(ValueError):
                sys.path.remove(s)


def resolve_ref(ref: str, bundle_dir: Path | None = None) -> Callable[..., Any]:
    """Import a dotted spec ``module.path:attr`` (or ``module.path.attr``) to a callable.

    The bundle directory (if given) is temporarily on sys.path so a bundle can ship its
    own modules.
    """
    if ":" in ref:
        module_name, _, attr = ref.partition(":")
    else:
        module_name, _, attr = ref.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"invalid ref {ref!r}: expected 'module.path:callable'")

    ctx = _bundle_on_path(bundle_dir) if bundle_dir else contextlib.nullcontext()
    with ctx:
        module = importlib.import_module(module_name)
    obj = getattr(module, attr, None)
    if obj is None:
        raise AttributeError(f"{module_name!r} has no attribute {attr!r} (ref {ref!r})")
    if not callable(obj):
        raise TypeError(f"ref {ref!r} resolved to a non-callable: {obj!r}")
    return obj
