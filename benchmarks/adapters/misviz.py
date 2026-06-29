"""QA-conflict adapter — misleading-visualization question answering.

Reproduces the framing of **"Protecting MLLMs against Misleading Visualizations"**
(Tonglet et al., arXiv 2502.20503; repo UKPLab/arxiv2025-misleading-visualizations):
MLLM QA accuracy on misleading charts collapses to the RANDOM baseline (mean
25.6% == random 25.6%), and the table-based-QA / redraw fix recovers +15.4-19.6pp.
That table-vs-chart contrast is exactly Multi-Vac's baseline-vs-skeptic split.

Each item pairs a chart 'misleader' (the shortcut cue) against the underlying
table (the truth), so the Truth-Shortcut Margin can tell which the model followed.

NOTE ON NAMING: the sibling **Misviz** dataset (arXiv 2508.21675, ACL 2026, repo
UKPLab/acl2026-misviz) is a misleader-DETECTION (multi-label classification) set,
NOT QA — see `misviz_synth.py` for that real-schema adapter. This module keeps the
`misviz` filename for continuity but implements the QA-conflict task from 2502.20503.

This adapter reads a committed JSON manifest so the harness, demo, and tests run
fully OFFLINE, independent of upstream dataset access (real-world Misviz images are
gated + not redistributable; CC-BY-SA-4.0). The small slice in `data/misviz/` is
ORIGINAL synthetic content mirroring the misleader taxonomy, not upstream data.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import BenchItem, Misleader

# Repo-root-relative default; benchmarks/adapters/misviz.py -> repo root is parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = _REPO_ROOT / "data" / "misviz" / "manifest.json"


def load_misviz(manifest_path: str | Path | None = None) -> list[BenchItem]:
    """Load Misviz items from a JSON manifest into normalized `BenchItem`s.

    Manifest schema (per item): id, question, options, truth_answer,
    shortcut_answer, misleader, chart, table. Unknown misleader strings fall
    back to `Misleader.NONE` rather than raising, so a partial manifest still loads.
    """
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"Misviz manifest not found at {path}. Commit a slice to data/misviz/ "
            "or run the (optional) fetch.py to download the full dataset."
        )

    raw = json.loads(path.read_text())
    records = raw.get("items", raw) if isinstance(raw, dict) else raw

    items: list[BenchItem] = []
    for rec in records:
        try:
            misleader = Misleader(rec.get("misleader", "none"))
        except ValueError:
            misleader = Misleader.NONE
        items.append(
            BenchItem(
                id=str(rec["id"]),
                dataset="misviz",
                question=rec["question"],
                chart_path=rec.get("chart"),
                table=rec.get("table", []),
                options=rec.get("options", []),
                truth_answer=str(rec["truth_answer"]),
                shortcut_answer=str(rec["shortcut_answer"]),
                misleader=misleader,
            )
        )
    return items
