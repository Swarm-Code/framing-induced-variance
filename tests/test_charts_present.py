"""C2 GATE — every Misviz item's rendered chart PNG exists on disk.

The baseline-vs-skeptic vision pipeline feeds `item.chart_path` to the model; if a
chart is missing the live run silently degrades to text-only and the comparison is
invalid. This test pins that the committed slice is fully rendered.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.adapters.misviz import load_misviz

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(chart_path: str) -> Path:
    p = Path(chart_path)
    return p if p.is_absolute() else _REPO_ROOT / p


def test_manifest_loads_nonempty():
    items = load_misviz()
    assert len(items) >= 5


def test_every_item_has_chart_path():
    for item in load_misviz():
        assert item.chart_path, f"{item.id} has no chart_path"


def test_every_chart_file_exists():
    missing = []
    for item in load_misviz():
        if item.chart_path and not _resolve(item.chart_path).exists():
            missing.append((item.id, item.chart_path))
    assert not missing, f"missing chart PNGs: {missing}"


def test_conflict_items_have_charts():
    """Conflict items are the ones scored for TSM — they MUST have a chart."""
    conflict = [i for i in load_misviz() if i.is_conflict]
    assert conflict, "expected at least one conflict item"
    for item in conflict:
        assert _resolve(item.chart_path).exists(), f"{item.id} chart missing"
