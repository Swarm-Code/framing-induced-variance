"""SC-ShortcutBench adapter — metadata-shortcut conflict rows (E2).

Reproduces the framing of **SC-ShortcutBench** (Khalilbraham/sc-shortcutbench-public,
CC-BY-4.0): single-cell foundation models predict labels via a METADATA SHORTCUT
(tissue / disease / study proxy) instead of the EXPRESSION signal. On the conflict
subset the published Truth-Shortcut Margin (TSM) is strongly negative (~-24 to -40pp)
across 10 models — they follow the metadata proxy, not the cell's expression.

This is the biotech instance of the same shortcut-reliance phenomenon the misviz
adapter captures for charts, so the SAME baseline-vs-skeptic split and the SAME
TSM metric apply:
  * baseline — answer from the metadata proxy presented up front (numbers-only
    analogue: the easy, dominant-but-wrong cue).
  * skeptic  — verify on the expression markers (the data) and, when the proxy
    contradicts the markers, trust the markers.

The committed slice in `data/scshortcut/` is ORIGINAL synthetic content mirroring
the conflict-row schema (truth=expression label, shortcut=metadata proxy label),
so the harness/demo/tests run fully OFFLINE without the gated upstream dataset.
There is no chart: SC-ShortcutBench is tabular/text, so `chart_path` is None and
the "table" carries the expression evidence + metadata proxy.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import BenchItem, Misleader

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = _REPO_ROOT / "data" / "scshortcut" / "manifest.json"


def load_scshortcut(manifest_path: str | Path | None = None) -> list[BenchItem]:
    """Load SC-ShortcutBench conflict rows into normalized `BenchItem`s.

    Each row's `question` embeds the metadata proxy (the shortcut cue) and the
    expression evidence is carried in `table` (the truth ground), so a baseline
    that anchors on metadata and a skeptic that anchors on expression diverge on
    the conflict rows exactly as the published benchmark reports.
    """
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"SC-ShortcutBench manifest not found at {path}. Commit a slice to "
            "data/scshortcut/ (synthetic) or fetch the upstream HF dataset."
        )

    raw = json.loads(path.read_text())
    records = raw.get("items", raw) if isinstance(raw, dict) else raw

    items: list[BenchItem] = []
    for rec in records:
        try:
            misleader = Misleader(rec.get("misleader", "none"))
        except ValueError:
            misleader = Misleader.NONE
        proxy = rec.get("metadata_proxy", "")
        evidence = rec.get("expression_evidence", "")
        # Question presents the metadata proxy up front (the shortcut cue a naive
        # model latches onto); the expression evidence is the table (truth ground).
        question = rec["question"]
        if proxy:
            question = f"{question} (Cell metadata: {proxy}.)"
        items.append(
            BenchItem(
                id=str(rec["id"]),
                dataset="scshortcut",
                question=question,
                chart_path=None,  # tabular benchmark — no chart
                table=[{"metadata_proxy": proxy, "expression_markers": evidence}],
                options=rec.get("options", []),
                truth_answer=str(rec["truth_answer"]),
                shortcut_answer=str(rec["shortcut_answer"]),
                misleader=misleader,
            )
        )
    return items
