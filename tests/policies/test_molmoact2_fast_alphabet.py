"""FAST tokenizer alphabet snapping.

The shipped MolmoAct2 FAST tokenizer cannot encode some quantized-DCT bins and, having
no UNK token, silently deletes them — which shifts every later coefficient into the
wrong (frequency, joint) cell. _tokenize_discrete_action snaps to the nearest encodable
bin instead. See lerobot/pi07_wiki/fast_tokenizer_alphabet_bug.md.
"""

from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.molmoact2.processor_molmoact2 import (
    _encodable_bin_map,
    _tokenize_discrete_action,
)

TOKENIZER = Path(__file__).resolve().parents[3] / "outputs" / "MolmoAct-FAST-tokenizer"
HORIZON, DIM = 30, 7


@pytest.fixture(scope="module")
def processor():
    if not TOKENIZER.exists():
        pytest.skip(f"FAST tokenizer not available at {TOKENIZER}")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(TOKENIZER), trust_remote_code=True)


def _encodable(bpe, b: int) -> bool:
    return bpe.decode(bpe(chr(b))["input_ids"]) == chr(b)


def _chunk_hitting_bin(processor, target_bin: int) -> np.ndarray:
    """A chunk whose DC coefficient for joint 0 quantizes to target_bin."""
    chunk = np.zeros((HORIZON, DIM), dtype=np.float32)
    coefficient = target_bin + processor.min_token
    chunk[:, 0] = coefficient / (processor.scale * np.sqrt(HORIZON))
    return chunk


def test_map_is_identity_on_encodable_bins(processor):
    mapping = _encodable_bin_map(processor)
    bpe = processor.bpe_tokenizer
    for b in range(300):
        if _encodable(bpe, b):
            assert mapping[b] == b, f"bin {b} is encodable but was remapped to {mapping[b]}"


def test_map_sends_unencodable_bins_somewhere_encodable(processor):
    mapping = _encodable_bin_map(processor)
    bpe = processor.bpe_tokenizer
    for b in range(300):
        assert _encodable(bpe, int(mapping[b])), f"bin {b} snapped to unencodable {mapping[b]}"


def test_low_holes_move_by_a_single_step(processor):
    """The seven sub-33 holes are isolated, so the snap error is one coefficient step."""
    mapping = _encodable_bin_map(processor)
    bpe = processor.bpe_tokenizer
    holes = [b for b in range(33) if not _encodable(bpe, b)]
    assert holes, "expected the shipped tokenizer to have low-range holes"
    for b in holes:
        assert abs(int(mapping[b]) - b) == 1, f"bin {b} snapped {abs(int(mapping[b]) - b)} steps"


def test_hole_chunk_now_encodes_to_a_full_length_span(processor):
    """Before the fix this produced 209 coefficients and scrambled the chunk."""
    bpe = processor.bpe_tokenizer
    holes = [b for b in range(33) if not _encodable(bpe, b)]
    for b in holes:
        ids = _tokenize_discrete_action(_chunk_hitting_bin(processor, b), processor)
        assert len(bpe.decode(ids)) == HORIZON * DIM


def test_matches_the_stock_tokenizer_when_no_snap_is_needed(processor):
    """Chunks that never hit a hole must tokenize byte-identically to the stock path."""
    rng = np.random.default_rng(0)
    bpe = processor.bpe_tokenizer
    compared = 0
    for _ in range(200):
        chunk = (rng.standard_normal((HORIZON, DIM)) * 0.15).astype(np.float32)
        stock = processor(chunk[None])[0]
        if len(bpe.decode(stock)) != HORIZON * DIM:
            continue                                  # this chunk hits a hole; not comparable
        assert _tokenize_discrete_action(chunk, processor) == list(stock)
        compared += 1
    assert compared > 50, f"only {compared} hole-free chunks to compare"


def test_snap_error_equals_one_dc_step(processor):
    """A DC snap offsets that joint by exactly 1 / (scale * sqrt(horizon)) = 0.0183 here.

    Pinned analytically rather than as a literal: this is the price of the fix, and it
    replaces a deleted coefficient whose reconstruction error is ~0.33.
    """
    from scipy.fft import idct

    bpe = processor.bpe_tokenizer
    holes = [b for b in range(33) if not _encodable(bpe, b)]
    expected = 1.0 / (processor.scale * np.sqrt(HORIZON))
    for b in holes:
        chunk = _chunk_hitting_bin(processor, b)
        ids = _tokenize_discrete_action(chunk, processor)
        coefficients = np.array(list(map(ord, bpe.decode(ids)))) + processor.min_token
        recon = idct(coefficients.reshape(HORIZON, DIM) / processor.scale, axis=0, norm="ortho")
        assert np.abs(recon - chunk).max() == pytest.approx(expected, rel=0.02)


def test_batch_input_is_rejected(processor):
    with pytest.raises(ValueError, match="batch"):
        _tokenize_discrete_action(np.zeros((2, HORIZON, DIM), dtype=np.float32), processor)
