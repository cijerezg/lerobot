"""Asynchronous JPEG encoder for RLT review-buffer image capture.

Used by the policy server to attach the inference-time camera frames to each
`RLTReplaySample` without blocking the inference hot path. Encoding runs on a
small thread pool and exposes a per-`context_id` mailbox; consumers wait on
the mailbox with a short timeout so a slow/overflowed encoder simply yields
`None` (and bumps a drop counter) instead of stalling inference.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import cv2  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


def encode_rgb_uint8_to_jpeg(image_rgb: np.ndarray, quality: int) -> bytes:
    """JPEG-encode a single (H, W, 3) uint8 RGB image.

    The result decodes to natural RGB in standard JPEG decoders (e.g. browsers),
    so we convert RGB->BGR before handing to ``cv2.imencode`` (cv2 expects BGR).
    """
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError(f"expected np.ndarray, got {type(image_rgb)!r}")
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"expected (H, W, 3) uint8 image, got shape={image_rgb.shape} dtype={image_rgb.dtype}"
        )
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("OpenCV failed to JPEG-encode review image")
    return bytes(buf)


class RltImageEncoder:
    """Background JPEG encoder with bounded in-flight queue.

    Usage:
        encoder = RltImageEncoder(quality=80, max_pending=8, max_workers=1)
        encoder.submit(context_id, {"observation.images.front": rgb_uint8, ...})
        # ... later, when the cached context is being finalized:
        images_jpeg = encoder.pop(context_id, timeout_s=0.05)  # may be None
    """

    def __init__(
        self,
        *,
        quality: int = 80,
        max_pending: int = 8,
        max_workers: int = 1,
    ) -> None:
        if not (1 <= int(quality) <= 100):
            raise ValueError(f"jpeg quality must be in [1, 100], got {quality}")
        if int(max_pending) <= 0:
            raise ValueError(f"max_pending must be positive, got {max_pending}")
        if int(max_workers) <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}")
        self._quality = int(quality)
        self._max_pending = int(max_pending)
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers),
            thread_name_prefix="rlt_jpeg_enc",
        )
        # context_id -> Future[dict[str, bytes]]
        self._pending: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._dropped = 0

    @property
    def quality(self) -> int:
        return self._quality

    @property
    def dropped_count(self) -> int:
        """Number of submissions rejected because ``max_pending`` was reached."""
        return self._dropped

    def submit(self, context_id: int, images_rgb_uint8: dict[str, np.ndarray]) -> bool:
        """Schedule encoding for one inference's worth of images.

        Returns True if the work was accepted, False if dropped because the
        pending queue was full (the caller should treat this as "no images for
        this transition").
        """
        if not images_rgb_uint8:
            return False

        cid = int(context_id)
        # Snapshot inputs synchronously so the producer can mutate / overwrite
        # its source arrays after this call returns. Cheap relative to encoding.
        try:
            snapshot = {key: np.ascontiguousarray(img) for key, img in images_rgb_uint8.items()}
        except Exception as e:
            logger.warning("RltImageEncoder: failed to snapshot images for ctx=%d: %s", cid, e)
            return False

        with self._lock:
            if len(self._pending) >= self._max_pending:
                self._dropped += 1
                return False
            future = self._executor.submit(self._encode_all, snapshot)
            self._pending[cid] = future
        return True

    def pop(self, context_id: int, *, timeout_s: float = 0.05) -> dict[str, bytes] | None:
        """Wait briefly for the encoded result; return None on miss/timeout/error.

        On timeout the future is dropped from the mailbox; the worker may still
        be running and will simply have its result GC'd. We never block
        inference longer than ``timeout_s``.
        """
        cid = int(context_id)
        with self._lock:
            future = self._pending.pop(cid, None)
        if future is None:
            return None
        try:
            return future.result(timeout=float(timeout_s))
        except TimeoutError:
            return None
        except Exception as e:
            logger.warning("RltImageEncoder: encode failed for ctx=%d: %s", cid, e)
            return None

    def discard(self, context_id: int) -> None:
        """Drop any pending encode for ``context_id`` (e.g. on context-cache eviction)."""
        with self._lock:
            self._pending.pop(int(context_id), None)

    def shutdown(self, *, wait: bool = False) -> None:
        with self._lock:
            self._pending.clear()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _encode_all(self, images_rgb: dict[str, np.ndarray]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for key, image in images_rgb.items():
            try:
                out[key] = encode_rgb_uint8_to_jpeg(image, self._quality)
            except Exception as e:
                # Per-camera failure: skip that camera, keep the rest.
                logger.warning("RltImageEncoder: skipping %s due to encode error: %s", key, e)
        return out


__all__ = ["RltImageEncoder", "encode_rgb_uint8_to_jpeg"]
