"""Tests for the off-hot-path JPEG encoder used by RLT review capture."""

import time

import numpy as np
import pytest

from lerobot.async_inference.utils.rlt_image_capture import (
    RltImageEncoder,
    encode_rgb_uint8_to_jpeg,
)


def _rand_rgb(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed=h * 1000 + w)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_encode_rgb_uint8_to_jpeg_returns_jpeg_bytes():
    image = _rand_rgb()
    payload = encode_rgb_uint8_to_jpeg(image, quality=80)
    # JPEG SOI marker.
    assert payload[:2] == b"\xff\xd8"
    # JPEG EOI marker.
    assert payload[-2:] == b"\xff\xd9"
    assert len(payload) > 100


def test_encode_rgb_rejects_wrong_dtype():
    bad = _rand_rgb().astype(np.float32)
    with pytest.raises(ValueError):
        encode_rgb_uint8_to_jpeg(bad, quality=80)


def test_encode_rgb_rejects_wrong_shape():
    bad = _rand_rgb()[..., 0]  # drop channels -> (H, W)
    with pytest.raises(ValueError):
        encode_rgb_uint8_to_jpeg(bad, quality=80)


def test_encoder_submit_and_pop_returns_jpegs():
    encoder = RltImageEncoder(quality=80, max_pending=4, max_workers=1)
    try:
        ok = encoder.submit(
            123,
            {
                "observation.images.front": _rand_rgb(),
                "observation.images.wrist": _rand_rgb(64, 48),
            },
        )
        assert ok is True
        # 50ms is plenty for 32x32 / 64x48 single-thread cv2 encodes.
        result = encoder.pop(123, timeout_s=2.0)
        assert result is not None
        assert set(result.keys()) == {
            "observation.images.front",
            "observation.images.wrist",
        }
        for jpeg in result.values():
            assert isinstance(jpeg, bytes)
            assert jpeg[:2] == b"\xff\xd8"
    finally:
        encoder.shutdown(wait=True)


def test_encoder_drops_when_max_pending_reached():
    # max_workers=1, max_pending=2: submit 3 small jobs back-to-back. The third
    # should be rejected synchronously while the first two are still encoding.
    encoder = RltImageEncoder(quality=80, max_pending=2, max_workers=1)
    try:
        for cid in (1, 2):
            assert encoder.submit(cid, {"observation.images.front": _rand_rgb(256, 256)}) is True
        # The third submission lands while the queue is full -- should be dropped.
        # We deliberately keep the encoder busy by NOT popping; encoding holds the
        # slot until pop() is called (or the result is GC'd via timeout).
        rejected = encoder.submit(3, {"observation.images.front": _rand_rgb(256, 256)})
        assert rejected is False
        assert encoder.dropped_count == 1
        # Drain queued slots so the test doesn't leak threads.
        encoder.pop(1, timeout_s=2.0)
        encoder.pop(2, timeout_s=2.0)
    finally:
        encoder.shutdown(wait=True)


def test_encoder_pop_unknown_id_returns_none():
    encoder = RltImageEncoder(quality=80)
    try:
        assert encoder.pop(999, timeout_s=0.01) is None
    finally:
        encoder.shutdown(wait=True)


def test_encoder_pop_timeout_returns_none(monkeypatch):
    """If encoding takes longer than the pop timeout, we get None (no stall)."""
    encoder = RltImageEncoder(quality=80, max_pending=2, max_workers=1)

    # Patch the internal encode helper to sleep, simulating a slow encode.
    original = encoder._encode_all

    def slow_encode(images):
        time.sleep(0.5)
        return original(images)

    monkeypatch.setattr(encoder, "_encode_all", slow_encode)
    try:
        encoder.submit(7, {"observation.images.front": _rand_rgb(16, 16)})
        result = encoder.pop(7, timeout_s=0.01)
        assert result is None
    finally:
        encoder.shutdown(wait=True)


def test_encoder_submit_empty_dict_is_noop():
    encoder = RltImageEncoder()
    try:
        assert encoder.submit(1, {}) is False
    finally:
        encoder.shutdown(wait=True)
