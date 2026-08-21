"""Mathematical core of the demonstrated-action spectrum probe."""

import csv

import pytest
import torch

from lerobot.probes.action_spectrum import (
    _phase_analysis,
    _space_analysis,
    _write_csv,
    dct_coefficients,
    orthonormal_dct_matrix,
    parse_band_spec,
)


def test_band_spec_is_an_exact_partition():
    bands = parse_band_spec("dc=0;transport=1-2;maneuver=3-5;fine=6-", horizon=10)
    assert [(band.name, band.start, band.stop) for band in bands] == [
        ("dc", 0, 1),
        ("transport", 1, 3),
        ("maneuver", 3, 6),
        ("fine", 6, 10),
    ]


@pytest.mark.parametrize(
    "spec",
    [
        "low=0-2;high=4-",  # gap
        "low=0-3;high=3-",  # overlap
        "all=0-10",  # outside a length-10 horizon
        "low=0;low=1-",  # duplicate name
    ],
)
def test_band_spec_rejects_ambiguous_geometry(spec):
    with pytest.raises(ValueError):
        parse_band_spec(spec, horizon=10)


def test_dct_is_orthonormal_and_preserves_energy():
    matrix = orthonormal_dct_matrix(30)
    assert torch.allclose(matrix @ matrix.T, torch.eye(30, dtype=matrix.dtype), atol=1e-12)

    chunks = torch.randn(11, 30, 7, dtype=torch.float64)
    transformed = dct_coefficients(chunks)
    assert torch.allclose(
        chunks.square().sum(dim=1), transformed.square().sum(dim=1), atol=1e-10
    )


def test_constant_chunk_has_only_dc_energy():
    chunks = torch.ones(3, 30, 2, dtype=torch.float64)
    transformed = dct_coefficients(chunks)
    assert torch.allclose(transformed[:, 1:], torch.zeros_like(transformed[:, 1:]), atol=1e-12)
    assert torch.allclose(transformed[:, 0], torch.full((3, 2), 30**0.5, dtype=torch.float64))


def test_mse_width_beta_reconstructs_time_domain_mse():
    chunks = torch.randn(13, 30, 4, dtype=torch.float64)
    bands = parse_band_spec("dc=0;transport=1-2;maneuver=3-8;fine=9-", horizon=30)
    analysis = _space_analysis(
        chunks,
        bands,
        names=[f"d{i}" for i in range(4)],
        fps=30.0,
        mask=torch.ones(len(chunks), dtype=torch.bool),
    )

    coefficients = dct_coefficients(chunks)
    beta = analysis["beta_candidates"]["mse_width"]["beta"]
    reconstructed = torch.zeros((), dtype=torch.float64)
    for weight, band in zip(beta, bands, strict=True):
        reconstructed += weight * coefficients[:, band.start : band.stop].square().mean()

    assert reconstructed == pytest.approx(float(chunks.square().mean()), rel=1e-12, abs=1e-12)
    assert beta == pytest.approx([1 / 30, 2 / 30, 6 / 30, 21 / 30])


def test_corpus_equalising_beta_equalises_expected_band_contribution():
    chunks = torch.randn(19, 12, 3, dtype=torch.float64)
    bands = parse_band_spec("low=0-2;middle=3-6;high=7-", horizon=12)
    analysis = _space_analysis(
        chunks,
        bands,
        names=["a", "b", "c"],
        fps=24.0,
        mask=torch.ones(len(chunks), dtype=torch.bool),
    )
    shares = analysis["beta_candidates"]["corpus_equalising_unfloored"][
        "expected_loss_share"
    ]
    assert shares == pytest.approx([1 / 3, 1 / 3, 1 / 3], rel=1e-12, abs=1e-12)


def test_phase_summaries_expose_both_normalization_directions():
    chunks = torch.randn(6, 8, 2, dtype=torch.float64)
    rows = [{"phase": phase} for phase in ("reach", "reach", "reach", "grasp", "grasp", "grasp")]
    bands = parse_band_spec("low=0-2;high=3-", horizon=8)
    _, summary = _phase_analysis(chunks, rows, bands, torch.ones(6, dtype=torch.bool))

    for phase in summary.values():
        assert sum(phase["bands"][band.name]["target_energy_share"] for band in bands) == pytest.approx(1)
    for band in bands:
        assert sum(
            phase["bands"][band.name]["share_of_band_corpus_energy"]
            for phase in summary.values()
        ) == pytest.approx(1)


def test_csv_writer_preserves_union_of_corpus_and_phase_columns(tmp_path):
    path = tmp_path / "mixed.csv"
    _write_csv(path, [{"scope": "corpus", "dimension": "arm"}, {"scope": "phase", "phase": "grasp"}])

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"scope": "corpus", "dimension": "arm", "phase": ""},
        {"scope": "phase", "dimension": "", "phase": "grasp"},
    ]
