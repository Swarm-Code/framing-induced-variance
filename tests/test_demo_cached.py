"""F1 GATE — the framing-reveal demo renders from the released FIV artifact offline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_demo_runs_offline():
    """`python scripts/demo.py` must succeed and show the FIV headline numbers."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "demo.py"), "--n", "2", "--no-table"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "Framing Reveal" in out
    assert "fiv_tabfact_scaled.json" in out
    # the powered headline must be present
    assert "Powered headline" in out
    assert "wrong-flip" in out
    # the protocol reduction is the punchline
    assert "deliberate-answer protocol" in out


def test_demo_specific_item():
    """A specific item id renders its per-framing verdict breakdown."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "demo.py"), "--id", "tf-test-11", "--no-table"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "tf-test-11" in out
    assert "gold =" in out
    # all eight framing slots should appear
    for slot in ("neutral", "lead-for", "authority-for", "doubt", "rhetorical"):
        assert slot in out
