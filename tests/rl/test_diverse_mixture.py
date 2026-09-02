"""Hierarchical sampling and the 50/50 outer split (integration plan phase I).

Gate I: over at least 10,000 draws the observed proportions match the targets within a
stated tolerance, and no source is forced into every small batch.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.datasets.diverse_actor_selection import (
    open_federated_corpus,
    select_actor_anchors,
)
from lerobot.rl.data_sources.diverse_mixture import (
    HierarchicalAnchorSampler,
    MixtureGroup,
    MixtureTelemetry,
    allocate_group_quotas,
)

DATA_ROOT = Path(__file__).resolve().parents[3] / "outputs/diverse_robot_dataset"
DRAWS = 20_000
# Four standard errors of a binomial proportion at this many draws. Wide enough that a
# correct sampler effectively never trips it, tight enough to catch a wrong weight rule
# (row-proportional would put RoboChallenge at 0.67 against a target of 0.35).
TOLERANCE_SIGMAS = 4.0


@pytest.fixture(scope="module")
def selection():
    if not (DATA_ROOT / "corpus" / "episodes.jsonl").is_file():
        pytest.skip("federated corpora not present")
    return select_actor_anchors(open_federated_corpus(DATA_ROOT))


# ── Outer quotas ─────────────────────────────────────────────────────────────


class _FakeBuffer:
    def get_iterator(self, **kwargs):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _groups(rebot_buffers: int = 4, weights=(1.0, 1.0)):
    return [
        MixtureGroup("rebot", [_FakeBuffer() for _ in range(rebot_buffers)], weight=weights[0]),
        MixtureGroup("diverse", [_FakeBuffer()], weight=weights[1]),
    ]


def test_an_even_batch_splits_exactly_in_half() -> None:
    quotas = allocate_group_quotas(32, _groups())
    assert sum(quotas[0]) == 16
    assert sum(quotas[1]) == 16


def test_an_odd_batch_is_deterministic_and_documented() -> None:
    """Ties break by declaration order, so the extra sample goes to the first group."""
    quotas = allocate_group_quotas(33, _groups())
    assert sum(quotas[0]) == 17
    assert sum(quotas[1]) == 16
    assert allocate_group_quotas(33, _groups()) == quotas  # same every time


def test_the_rebot_inner_mixture_is_preserved() -> None:
    group = MixtureGroup("rebot", [_FakeBuffer() for _ in range(4)], inner_weights=[4, 2, 3, 1])
    quotas = allocate_group_quotas(64, [group, MixtureGroup("diverse", [_FakeBuffer()])])
    rebot = quotas[0]
    assert sum(rebot) == 32
    # Quotas follow the declared weights (not a re-sort), and no source is starved.
    weights = [4, 2, 3, 1]
    assert [q for _, q in sorted(zip(weights, rebot))] == sorted(rebot)
    assert min(rebot) >= 1


def test_a_different_outer_split_is_configurable() -> None:
    quotas = allocate_group_quotas(40, _groups(weights=(3.0, 1.0)))
    assert sum(quotas[0]) > sum(quotas[1])


# ── Gate I: realized proportions ─────────────────────────────────────────────


def test_sqrt_episode_weights_are_what_the_plan_says(selection) -> None:
    targets = HierarchicalAnchorSampler(selection, seed=0).target_proportions()
    expected = {
        "robochallenge": math.sqrt(200),
        "fmb": math.sqrt(100),
        "droid": math.sqrt(50),
        "droid_success": math.sqrt(50),
        "ur7e": math.sqrt(4),
    }
    total = sum(expected.values())
    for source, weight in expected.items():
        assert targets[source] == pytest.approx(weight / total, abs=1e-6)


def test_realized_source_proportions_match_the_targets(selection) -> None:
    sampler = HierarchicalAnchorSampler(selection, seed=42)
    rows = sampler(DRAWS)
    observed = collections.Counter(selection.rows[int(index)]["source"] for index in rows)
    for source, target in sampler.target_proportions().items():
        realized = observed[source] / DRAWS
        sigma = math.sqrt(target * (1 - target) / DRAWS)
        assert abs(realized - target) <= TOLERANCE_SIGMAS * sigma, (
            f"{source}: realized {realized:.4f} vs target {target:.4f} "
            f"(tolerance {TOLERANCE_SIGMAS * sigma:.4f})"
        )


def test_the_mixture_is_not_row_proportional(selection) -> None:
    """The failure mode this rule exists to avoid, checked directly."""
    sampler = HierarchicalAnchorSampler(selection, seed=1)
    rows = sampler(DRAWS)
    observed = collections.Counter(selection.rows[int(index)]["source"] for index in rows)
    row_share = 40_706 / 60_728  # RoboChallenge's share of the anchors
    assert observed["robochallenge"] / DRAWS < row_share - 0.2
    # ... nor source-uniform, which would overfit the four UR7e episodes.
    assert observed["ur7e"] / DRAWS < 0.10


def test_episodes_are_drawn_uniformly_inside_a_source(selection) -> None:
    """A long episode must not outvote a short one within its own source."""
    sampler = HierarchicalAnchorSampler(selection, seed=7)
    rows = sampler(DRAWS)
    ur7e = [
        selection.rows[int(index)]["episode_id"]
        for index in rows
        if selection.rows[int(index)]["source"] == "ur7e"
    ]
    counts = collections.Counter(ur7e)
    assert len(counts) == 4
    share = np.array(sorted(counts.values())) / len(ur7e)
    assert share.max() - share.min() < 0.10


def test_no_source_is_forced_into_every_small_batch(selection) -> None:
    sampler = HierarchicalAnchorSampler(selection, seed=3)
    absent = collections.Counter()
    batches = 500
    for _ in range(batches):
        sources = {selection.rows[int(index)]["source"] for index in sampler(8)}
        for source in sampler.groups:
            if source not in sources:
                absent[source] += 1
    for source in sampler.groups:
        assert absent[source] > 0, f"{source} appears in every 8-sample batch"


def test_ranks_draw_distinct_deterministic_streams(selection) -> None:
    def draw(rank: int, seed: int = 11) -> list[int]:
        return HierarchicalAnchorSampler(
            selection, seed=seed, rank=rank, world_size=4
        )(64).tolist()

    assert draw(0) != draw(1)
    assert draw(0) == draw(0)  # reproducible
    assert draw(2, seed=12) != draw(2, seed=11)


def test_weights_can_be_overridden_per_group(selection) -> None:
    sampler = HierarchicalAnchorSampler(
        selection, seed=0, group_weight="uniform", group_weight_overrides={"ur7e": 4.0}
    )
    targets = sampler.target_proportions()
    assert targets["ur7e"] == pytest.approx(4 / 8)
    with pytest.raises(KeyError, match="unknown group"):
        HierarchicalAnchorSampler(selection, group_weight_overrides={"nope": 1.0})


# ── Telemetry ────────────────────────────────────────────────────────────────


def test_telemetry_reports_every_axis_the_plan_asks_for() -> None:
    telemetry = MixtureTelemetry()
    batch = {
        "reward": torch.zeros(4),
        "complementary_info": {
            "source_id": torch.tensor([0, 3, 3, 5]),
            "embodiment_index": torch.tensor([0, 3, 3, 4]),
            "action_layout_id": torch.tensor([0, 3, 3, 6]),
            "episode_position": torch.tensor([1, 2, 2, 9]),
            "camera_is_present": torch.tensor(
                [[True, True, True], [True, False, True], [True, False, True], [True, False, True]]
            ),
            "depth.wrist_0.depth_is_present": torch.tensor([False, False, False, True]),
            "metadata_quality_is_valid": torch.tensor([True, False, False, True]),
        },
    }
    telemetry.observe(batch)
    assert telemetry.samples == 4
    assert telemetry.proportions(telemetry.source) == {0: 0.25, 3: 0.5, 5: 0.25}
    assert telemetry.proportions(telemetry.camera_count) == {2: 0.75, 3: 0.25}
    assert telemetry.proportions(telemetry.depth)[True] == 0.25
    assert telemetry.proportions(telemetry.quality_valid)[True] == 0.5
    assert len(telemetry.episode) == 3
    assert "Realized mixture over 4 samples" in telemetry.describe()
