"""Multi-Vac evaluation harness.

Normalizes external shortcut-learning benchmarks into a common `BenchItem`,
runs a baseline vs. a skeptic pipeline over them, and scores shortcut reliance
with a Truth-Shortcut-Margin (TSM) metric — the non-biotech analogue of
SC-ShortcutBench's headline number.

Anchor dataset: **Misviz / Misviz-synth** (misleading-visualization QA, ACL 2026).
Everything here runs fully offline against a small committed conflict slice; the
full dataset fetch is optional and flag-gated.
"""

from __future__ import annotations

from .metric import shortcut_metric, truth_shortcut_margin
from .models import BenchItem, BenchResult, ShortcutMetric

__all__ = [
    "BenchItem",
    "BenchResult",
    "ShortcutMetric",
    "shortcut_metric",
    "truth_shortcut_margin",
]
