"""Adapters normalize an external benchmark's raw shape into `list[BenchItem]`.

Each adapter is a thin, dependency-light function so the runner and metric never
touch a dataset's idiosyncratic format.
"""

from __future__ import annotations

from .misviz import load_misviz
from .misviz_synth import MISVIZ_MISLEADERS, MisvizSynthItem, load_misviz_synth

__all__ = ["load_misviz", "load_misviz_synth", "MisvizSynthItem", "MISVIZ_MISLEADERS"]
