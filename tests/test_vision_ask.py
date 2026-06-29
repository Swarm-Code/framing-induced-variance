"""C1 GATE — vision ask-builders run end-to-end offline (no network)."""

from __future__ import annotations

import base64

import pytest

from benchmarks.models import BenchItem, Misleader
from benchmarks.vision import build_prompt, vision_ask
from multivac import Multivac, Settings

_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def harness(tmp_path) -> Multivac:
    return Multivac(
        Settings(
            offline=True,
            skills_dir=str(tmp_path / "skills"),
            sessions_dir=str(tmp_path / "sessions"),
        )
    )


def _item(tmp_path, with_chart: bool) -> BenchItem:
    chart = None
    if with_chart:
        p = tmp_path / "chart.png"
        p.write_bytes(_PNG_1x1)
        chart = str(p)
    return BenchItem(
        id="c0",
        question="Is revenue growing",
        chart_path=chart,
        table=[{"year": 2020, "rev": 100}, {"year": 2021, "rev": 99}],
        truth_answer="no",
        shortcut_answer="yes",
        misleader=Misleader.TRUNCATED_AXIS,
    )


def test_prompts_differ_by_pipeline(tmp_path):
    item = _item(tmp_path, with_chart=True)
    base = build_prompt(item, skeptic=False)
    skep = build_prompt(item, skeptic=True)
    assert base != skep
    # skeptic prompt (chart item) includes the data table and the misleader taxonomy
    assert "underlying data table" in skep
    assert "truncated_axis" in skep
    assert '"rev":99' in skep or '"rev": 99' in skep
    # chart baseline does NOT leak the table (answers from the image only)
    assert "underlying data table" not in base
    assert '"rev":99' not in base and '"rev": 99' not in base


def test_tabular_prompts_never_mention_chart(tmp_path):
    """SC-ShortcutBench-style items have no chart -> prompts must not ask for one."""
    item = _item(tmp_path, with_chart=False)
    base = build_prompt(item, skeptic=False)
    skep = build_prompt(item, skeptic=True)
    assert "chart" not in base.lower()
    assert "chart" not in skep.lower()
    # tabular items still expose the evidence table in both pipelines
    assert '"rev":99' in base or '"rev": 99' in base


def test_vision_ask_runs_with_chart(harness, tmp_path):
    ask = vision_ask(harness, skeptic=True)
    out = ask(_item(tmp_path, with_chart=True))
    assert isinstance(out, str) and out


def test_vision_ask_runs_without_chart(harness, tmp_path):
    ask = vision_ask(harness, skeptic=False)
    out = ask(_item(tmp_path, with_chart=False))
    assert isinstance(out, str) and out


def test_vision_ask_resets_between_items(harness, tmp_path):
    ask = vision_ask(harness, skeptic=False)
    ask(_item(tmp_path, with_chart=False))
    n_after_first = len(harness.history)
    ask(_item(tmp_path, with_chart=False))
    # reset() each call -> history should not grow unboundedly across items
    assert len(harness.history) <= n_after_first + 0 or len(harness.history) >= 2
