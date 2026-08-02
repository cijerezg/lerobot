"""Metric core of the action manifold probe (`lerobot.probes.actions`).

The probe's product is now numbers rather than pictures, so the distance definitions
are worth pinning: a chunk inside the reference span has no off-span residual, pose
matching can only ever make the nearest neighbour further away, and feeding GT in as
the prediction has to score exactly the null.
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from lerobot.probes.actions import manifold_distances, project


def make_manifold(n=400, dim=12, span=4, seed=0):
    """A reference set living in a `span`-dimensional subspace of `dim` dimensions."""
    rng = np.random.RandomState(seed)
    basis = np.linalg.qr(rng.randn(dim, span))[0]          # [dim, span], orthonormal
    latents = rng.randn(n, span)
    vectors = latents @ basis.T
    states = rng.randn(n, 3)

    pca = PCA(n_components=span, random_state=0).fit(vectors)
    return {
        "id": "test",
        "pca": pca,
        "ref_pca": pca.transform(vectors),
        "ref_states_z": states,
        "ref_subtasks": ["reach" if s[0] < 0 else "place" for s in states],
        "state_mean": np.zeros(3),
        "state_std": np.ones(3),
        "basis": basis,
    }, vectors, states


def test_residual_is_zero_inside_the_span():
    manifold, vectors, _ = make_manifold()
    _, residual = project(manifold, vectors[:20])
    assert np.allclose(residual, 0.0, atol=1e-8)


def test_residual_sees_motion_outside_the_span():
    manifold, vectors, _ = make_manifold()
    basis = manifold["basis"]
    off = np.linalg.qr(np.random.RandomState(1).randn(basis.shape[0], basis.shape[0]))[0][:, -1]
    off = off - basis @ (basis.T @ off)                    # strictly outside the span
    off /= np.linalg.norm(off)

    _, residual = project(manifold, vectors[:20] + 3.0 * off)
    assert np.allclose(residual, 3.0, atol=1e-6)


def test_pose_matching_never_finds_a_closer_neighbour():
    manifold, vectors, states = make_manifold()
    query = vectors[:50] + 0.2 * np.random.RandomState(2).randn(50, vectors.shape[1])
    labels = ["reach"] * 50

    dist = manifold_distances(manifold, query, states[:50], labels, k=32)
    assert np.all(dist["nn_state"] >= dist["nn_global"] - 1e-9)


def test_gt_against_itself_scores_the_null():
    """Reference points are in the reference set, so both distances collapse to zero."""
    manifold, vectors, states = make_manifold()
    labels = [manifold["ref_subtasks"][i] for i in range(30)]

    dist = manifold_distances(manifold, vectors[:30], states[:30], labels, k=16)
    assert np.allclose(dist["nn_global"], 0.0, atol=1e-8)
    assert np.allclose(dist["nn_state"], 0.0, atol=1e-8)
    assert np.allclose(dist["residual"], 0.0, atol=1e-8)
    assert dist["subtask_agreement"].all()


def test_neighbourhood_is_clamped_to_the_reference_size():
    manifold, vectors, states = make_manifold(n=8)
    dist = manifold_distances(manifold, vectors[:3], states[:3], ["reach"] * 3, k=999)
    assert dist["nn_state"].shape == (3,)


@pytest.mark.parametrize("scale", [0.25, 4.0])
def test_magnitude_tracks_the_chunk_norm(scale):
    """The hold-still guard: magnitude must follow the size of the motion, nothing else."""
    manifold, vectors, states = make_manifold()
    dist = manifold_distances(manifold, scale * vectors[:10], states[:10], ["reach"] * 10, k=8)
    assert np.allclose(dist["magnitude"], scale * np.linalg.norm(vectors[:10], axis=1))
