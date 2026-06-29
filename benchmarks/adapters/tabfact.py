"""TabFact adapter — binary fact-verification over REAL tables (Phase H core).

table-benchmark/tabfact (117,854 rows): each item is a (table, claim) pair with a
binary gold label Entailed/Refuted. The Refuted claims are subtle near-misses that
require multi-step aggregation / uniqueness / comparison to falsify — exactly the
"grey-area, hard-to-spot, but binary with ground truth" regime where Framing-Induced
Variance (FIV) is meaningful AND falsifiable: a leading frame ("this is clearly
supported, right?") can reward-hack the answer across the Entailed/Refuted boundary,
and we can PROVE the flip is wrong against gold.

We map each row to a BenchItem:
  truth_answer    = the gold label ("Entailed" | "Refuted")
  shortcut_answer = the opposite label (the pole a hostile frame nudges toward)
  question        = the claim to verify
  table           = parsed rows of the real table (carried as list[dict])
The raw '#'-delimited table string is converted to a clean markdown table for the
model prompt by `table_to_markdown`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import BenchItem, Misleader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIR = _REPO_ROOT / "data" / "tabfact"

_OPP = {"Entailed": "Refuted", "Refuted": "Entailed"}


def table_to_markdown(raw: str, *, max_rows: int = 40) -> str:
    """Convert the '#'-delimited, newline-rowed TabFact table string to markdown."""
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    if not lines:
        return "(empty table)"
    header = [c.strip() for c in lines[0].split("#")]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for ln in lines[1 : 1 + max_rows]:
        cells = [c.strip() for c in ln.split("#")]
        out.append("| " + " | ".join(cells) + " |")
    if len(lines) - 1 > max_rows:
        out.append(f"| ... ({len(lines) - 1 - max_rows} more rows) |")
    return "\n".join(out)


def _table_rows(raw: str, *, max_rows: int = 60) -> list[dict]:
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    if not lines:
        return []
    header = [c.strip() for c in lines[0].split("#")]
    rows = []
    for ln in lines[1 : 1 + max_rows]:
        cells = [c.strip() for c in ln.split("#")]
        rows.append({header[i] if i < len(header) else f"col{i}": cells[i] for i in range(len(cells))})
    return rows


def load_tabfact(split: str = "test", *, limit: int | None = None) -> list[BenchItem]:
    p = _DIR / f"{split}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — pull full TabFact into data/tabfact/.")
    rows = json.loads(p.read_text())
    if limit:
        rows = rows[:limit]
    items: list[BenchItem] = []
    for r in rows:
        gold = r["label"]
        if gold not in _OPP:
            continue
        items.append(
            BenchItem(
                id=r["id"],
                dataset="tabfact",
                question=r["claim"],
                table=_table_rows(r["table"]),
                options=["Entailed", "Refuted"],
                truth_answer=gold,
                shortcut_answer=_OPP[gold],
                misleader=Misleader.NONE,
                metadata={"table_md": table_to_markdown(r["table"]), "title": r.get("title", "")},
            )
            if "metadata" in BenchItem.model_fields
            else BenchItem(
                id=r["id"], dataset="tabfact", question=r["claim"],
                table=_table_rows(r["table"]), options=["Entailed", "Refuted"],
                truth_answer=gold, shortcut_answer=_OPP[gold], misleader=Misleader.NONE,
            )
        )
    return items


def table_md_for(item) -> str:
    """Render a loaded item's parsed table rows back to markdown for the prompt."""
    if not item.table:
        return "(empty table)"
    header = list(item.table[0].keys())
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for row in item.table[:40]:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in header) + " |")
    return "\n".join(out)
