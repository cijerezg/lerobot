"""Metric core of the action manifold probe (`lerobot.probes.actions`).

The probe's product is now numbers rather than pictures, so the distance definitions
are worth pinning: a chunk inside the reference span has no off-span residual, pose
matching can only ever make the nearest neighbour further away, and feeding GT in as
the prediction has to score exactly the null.

The grounded figure is covered here too, for the part of it that is a claim rather than
a drawing: which frames it puts on the page, and that a trace is motion rather than pose.
"""

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

from lerobot.probes import actions as actions_probe
from lerobot.probes.actions import _ee_trace, _pair_cards, _region_cards, manifold_distances, project


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


# ── The grounded figure ───────────────────────────────────────────────────────

class LinearKinematics:
    """Stand-in for `RebotKinematics`: EE position is a fixed linear map of the joints.

    Keeps the tests free of pinocchio, which arrives with an extra and is absent from a
    base install.
    """

    def ee_path(self, q_deg):
        q = np.asarray(q_deg, dtype=float)
        return q[:, :3] * np.array([1.0, 0.5, 2.0])


def _make_adapter(chunk_size=4):
    return type("Adapter", (), {"chunk_size": chunk_size})()


def test_a_trace_is_motion_not_pose():
    """The claim the card caption makes: a chunk that holds still draws nothing."""
    state = np.array([10.0, -40.0, 5.0, 0.0, 0.0, 0.0, 30.0])
    hold = np.repeat(state[None, :], 6, axis=0)

    trace = _ee_trace(LinearKinematics(), state, hold)
    assert trace.shape == (7, 3)                      # the state is prepended
    assert np.allclose(trace, 0.0)


def test_the_same_motion_from_a_different_pose_draws_the_same_trace():
    kin, chunk_offset = LinearKinematics(), np.linspace(0, 9, 6)[:, None] * np.ones(7)
    here = np.zeros(7)
    there = np.array([25.0, -10.0, 60.0, 3.0, 3.0, 3.0, 0.0])

    assert np.allclose(_ee_trace(kin, here, here + chunk_offset),
                       _ee_trace(kin, there, there + chunk_offset))


def _fake_eval(pair_distances):
    """An eval set whose GT/pred pairs sit at prescribed distances along one axis."""
    n = len(pair_distances)
    coords = np.zeros((n, 3))
    return {
        "gt": {"coords": coords, "nn_state": np.full(n, 0.5)},
        "pred": {"coords": coords + np.array(pair_distances)[:, None] * [1, 0, 0],
                 "nn_state": np.full(n, 0.6)},
        "metadata": [{"episode_idx": 0, "frame_idx": i, "global_idx": 100 + i,
                      "subtask": "grasp the sock"} for i in range(n)],
        "states": np.zeros((n, 7)),
        "gt_actions": np.zeros((n, 4, 7)),
        "pred_actions": np.ones((n, 4, 7)),
        "gt_emb2": np.zeros((n, 2)),
        "pred_emb2": np.ones((n, 2)),
    }


def test_pair_cards_span_agreement_and_disagreement(monkeypatch):
    """Cards 1-2 must come from the close tail and 3-4 from the far tail, in order."""
    monkeypatch.setattr(actions_probe, "get_frame_data", lambda *a, **k: (
        {"observation.images.top": torch.zeros(1, 3, 8, 8, dtype=torch.uint8)},
        None, None, "", "", 0, 0,
    ))
    distances = np.linspace(0.1, 9.0, 20)[::-1]        # deliberately not pre-sorted
    cards = _pair_cards(LinearKinematics(), _make_adapter(), object(), _fake_eval(distances))

    assert [c["tag"] for c in cards] == ["1", "2", "3", "4"]
    assert [c["kind"] for c in cards] == ["close", "close", "far", "far"]
    picked = [float(c["title"].split("distance ")[1].split(" ")[0]) for c in cards]
    assert picked == sorted(picked)
    assert picked[1] < np.median(distances) < picked[2]
    assert [t["role"] for t in cards[0]["traces"]] == ["gt", "pred"]


def test_region_cards_refuse_a_stale_manifold(monkeypatch):
    """A cached manifold from other sampling parameters cannot be traced back to frames."""
    monkeypatch.setattr(actions_probe, "reference_samples", lambda datasets, p: [(0, 0)])
    cfg = type("Cfg", (), {"probe_parameters": type("P", (), {"random_seed": 0})()})()

    manifold = {"ref_pca": np.zeros((9, 3)), "ref_emb2": np.zeros((9, 2))}
    assert _region_cards(LinearKinematics(), _make_adapter(), manifold, [], cfg) == []
