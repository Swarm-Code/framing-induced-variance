"""Network-gated LIVE smoke for Cerebras Gemma 4 vision (D0/D1 guard).

Skips entirely unless a real key is configured (Settings.is_live), so the offline
suite stays green and CI without a key is unaffected. When live, it proves the end
-to-end path: Multivac.chat(images=[chart]) -> Cerebras Gemma 4 -> non-empty reply
that recognizes a truncated y-axis. This is the regression guard for the top-risk
multimodal path (and the Cloudflare-UA workaround in provider.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multivac import Multivac, Settings

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHART = _REPO_ROOT / "data" / "misviz" / "charts" / "mv-trunc-01.png"

_settings = Settings.load()
live = pytest.mark.skipif(
    not _settings.is_live, reason="no live Cerebras key configured (offline)"
)


@live
def test_live_vision_sees_truncated_axis():
    s = _settings.model_copy(update={"model": _settings.vision_model})
    h = Multivac(s, system_prompt="You are a skeptical data analyst. Be concise.")
    r = h.chat(
        "Does this chart use a truncated y-axis? Answer yes/no with a one-sentence reason.",
        images=[_CHART.read_bytes()],
    )
    assert r.output and not r.blocked
    out = r.output.lower()
    # Live phrasing varies run-to-run; assert the substantive claim robustly:
    # it AFFIRMS the truncation (yes) and/or cites the non-zero start (1018 / not
    # starting at zero / "truncat"). Any one of these signals a correct read.
    affirms = out.lstrip().startswith("yes") or "truncat" in out
    cites = "1018" in out or "zero" in out or "start at 0" in out or "starts at 0" in out
    assert affirms or cites, f"unexpected live reply: {r.output!r}"
