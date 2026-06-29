"""Misviz-synth detection adapter — real-schema parsing, offline fixture."""

from __future__ import annotations

import json

from benchmarks.adapters.misviz_synth import (
    MISVIZ_MISLEADERS,
    MisvizSynthItem,
    load_misviz_synth,
)

# Minimal fixture mirroring the VERIFIED upstream schema
# (data/misviz_synth/misviz_synth.json record shape).
_FIXTURE = [
    {
        "image_path": "png/TRUNCATED_AXIS_BAR_abc.png",
        "chart_type": ["bar"],
        "misleader": ["truncated axis"],
        "table_id": "abc",
        "variant": "1",
        "table_data_path": "data_tables/abc.csv",
        "axis_data_path": "axis_data/abc.json",
        "code_path": "code_snippets/abc.py",
        "split": "test",
    },
    {
        "image_path": "png/HONEST_LINE_def.png",
        "chart_type": ["line"],
        "misleader": [],
        "table_id": "def",
        "variant": "1",
        "table_data_path": "data_tables/def.csv",
        "axis_data_path": "axis_data/def.json",
        "code_path": "code_snippets/def.py",
        "split": "train",
    },
]


def test_loads_real_schema_and_flags_misleading(tmp_path):
    p = tmp_path / "misviz_synth.json"
    p.write_text(json.dumps(_FIXTURE))
    items = load_misviz_synth(p)
    assert len(items) == 2
    assert all(isinstance(it, MisvizSynthItem) for it in items)
    misleading = [it for it in items if it.is_misleading]
    honest = [it for it in items if not it.is_misleading]
    assert len(misleading) == 1
    assert len(honest) == 1
    assert misleading[0].misleader == ["truncated axis"]


def test_split_filter(tmp_path):
    p = tmp_path / "misviz_synth.json"
    p.write_text(json.dumps(_FIXTURE))
    test_items = load_misviz_synth(p, split="test")
    assert len(test_items) == 1
    assert test_items[0].split == "test"


def test_taxonomy_has_twelve_misleaders():
    assert len(MISVIZ_MISLEADERS) == 12
    assert "truncated axis" in MISVIZ_MISLEADERS


def test_missing_file_raises(tmp_path):
    try:
        load_misviz_synth(tmp_path / "nope.json")
    except FileNotFoundError as e:
        assert "fetch" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
