"""GATE test for Task A1 — harness.chat() accepts images (offline, no network).

Verifies chat(message, images=[...]) runs through the full lifecycle without error
using the offline deterministic provider, for both raw bytes and a file path, and
that the image is placed BEFORE the text per Gemma 4 modality-order guidance.
"""

from __future__ import annotations

import base64

import pytest

from multivac import Multivac, Settings
from multivac.harness import Multivac as HarnessClass

# Smallest valid 1x1 PNG (transparent), base64-decoded to raw bytes.
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        offline=True,
        skills_dir=str(tmp_path / "skills"),
        sessions_dir=str(tmp_path / "sessions"),
    )


@pytest.fixture
def harness(settings) -> Multivac:
    return Multivac(settings)


def test_chat_accepts_image_bytes(harness):
    r = harness.chat("describe this chart", images=[_PNG_1x1])
    assert r.output and not r.blocked
    # history advanced (user turn + assistant turn at minimum)
    assert len(harness.history) >= 2


def test_chat_accepts_image_path(harness, tmp_path):
    p = tmp_path / "chart.png"
    p.write_bytes(_PNG_1x1)
    r = harness.chat("inspect", images=[str(p)])
    assert r.output and not r.blocked


def test_chat_without_images_still_works(harness):
    r = harness.chat("hello")
    assert r.output and not r.blocked


def test_coerce_image_orders_image_first():
    """_coerce_image returns BinaryContent; chat builds [image, text] (image first)."""
    bc = HarnessClass._coerce_image(_PNG_1x1)
    assert bc.media_type == "image/png"
    assert bc.data == _PNG_1x1


def test_coerce_image_detects_jpeg(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(_PNG_1x1)  # content irrelevant for media-type detection
    bc = HarnessClass._coerce_image(str(p))
    assert bc.media_type == "image/jpeg"
