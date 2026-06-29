"""Misviz-synth adapter — REAL schema, misleader-DETECTION task.

Source (verified): "Is this chart lying to me? Automating the detection of
misleading visualizations" (Tonglet et al., arXiv 2508.21675, ACL 2026;
repo UKPLab/acl2026-misviz). Dataset annotations CC-BY-SA-4.0.

This is NOT a QA task — it is multi-label classification: predict which
misleader(s) (from a 12-type taxonomy), if any, affect a visualization. A chart
with an empty `misleader` list is non-misleading.

Verified record schema (data/misviz_synth/misviz_synth.json):
    image_path: str
    chart_type: list[str]
    misleader: list[str]          # empty => non-misleading
    table_id: str
    variant: str
    table_data_path: str
    axis_data_path: str
    code_path: str
    split: str                    # train | train small | dev | val | test

The labels JSON is plain on GitHub (see data/misviz/fetch.py); images + tables
come from TUdatalib. This adapter loads the labels file alone and exposes the
detection items; it does not require the (large, separately hosted) images.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

# The official 12-misleader taxonomy (Misviz). Kept as a frozen set for reference;
# unknown strings are preserved as-is rather than dropped.
MISVIZ_MISLEADERS: frozenset[str] = frozenset(
    {
        "misrepresentation",
        "3D",
        "truncated axis",
        "inappropriate use of pie chart",
        "inconsistent binning size",
        "discretized continuous variable",
        "inconsistent tick intervals",
        "dual axis",
        "inappropriate use of line chart",
        "inappropriate item order",
        "inverted axis",
        "inappropriate axis range",
    }
)


class MisvizSynthItem(BaseModel):
    """One Misviz-synth detection example (real schema)."""

    image_path: str
    chart_type: list[str] = Field(default_factory=list)
    misleader: list[str] = Field(default_factory=list)
    table_id: str | None = None
    variant: str | None = None
    table_data_path: str | None = None
    axis_data_path: str | None = None
    code_path: str | None = None
    split: str | None = None

    @property
    def is_misleading(self) -> bool:
        """True when at least one misleader is present (empty list => honest chart)."""
        return len(self.misleader) > 0


def load_misviz_synth(
    labels_path: str | Path,
    *,
    split: str | None = None,
) -> list[MisvizSynthItem]:
    """Load Misviz-synth detection items from the real labels JSON.

    `labels_path` is the `misviz_synth.json` downloaded via data/misviz/fetch.py.
    Optionally filter to one `split` (train | train small | dev | val | test).
    """
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Misviz-synth labels not found at {path}. Download with: "
            "python -m data.misviz.fetch misviz-synth-labels --confirm"
        )

    records = json.loads(path.read_text())
    items = [MisvizSynthItem(**rec) for rec in records]
    if split is not None:
        items = [it for it in items if it.split == split]
    return items
