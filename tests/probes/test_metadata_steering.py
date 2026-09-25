"""Precision and contact channels through the probes: the steering conditions, the
per-frame channel numbers, the adapter's clause markers and the trainer's forwarded keys."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.datasets.contact_vocab import NA_CODE
from lerobot.probes import metadata_steering as ms
from lerobot.rl.molmoact2.rl_molmoact2_trainer import _forwarded_complementary_keys


def test_older_conditions_carry_no_precision_or_contact() -> None:
    """The quality / mistake / speed rows render today's prompts: neither new key is set."""
    for metadata in ms._STEERED.values():
        assert set(metadata) == {"quality", "mistake", "speed"}


def test_channel_conditions_are_the_rollout_clause_plus_one_sentence() -> None:
    rollout = ms._STEERED[ms._ROLLOUT]
    assert [f"p{k}" for k in range(1, 6)] + [f"c{c}" for c in range(15)] == list(ms._CHANNELS)
    for k in range(1, 6):
        assert ms._CHANNELS[f"p{k}"] == {**rollout, "precision": k}
    for code in range(15):
        assert ms._CHANNELS[f"c{code}"] == {**rollout, "contact": code}
    assert ms._CHANNELS[f"c{NA_CODE}"]["contact"] == 14


def _channel_acts(precision_step: float, contact_step: float) -> dict[str, torch.Tensor]:
    """Chunks placed along one direction: precision k at k*step, contact code c at c*step."""
    direction = torch.ones(30, 7)
    acts = {f"p{k}": precision_step * k * direction for k in ms._PRECISION_LEVELS}
    acts |= {f"c{code}": contact_step * (code + 1) * direction for code in ms._CONTACT_CODES}
    return acts


def test_channel_measurements_read_against_the_clause_off_chunk_and_the_floor() -> None:
    clause_off = torch.zeros(30, 7)
    base = torch.full((30, 7), 10.0)
    row = ms._channel_measurements(_channel_acts(0.1, 0.01), clause_off, base, floor_mean=0.05)

    assert row["p3_clause_rmse"] == pytest.approx(0.3)
    assert row["p3_rmse"] == pytest.approx(9.7)  # from ``none``, like every other condition
    assert row["p3_clause_separation"] == pytest.approx(6.0)
    assert row["precision_range_rmse"] == pytest.approx(0.4)
    assert row["precision_range_separation"] == pytest.approx(8.0)
    assert row["precision_kendall_tau"] == pytest.approx(1.0)
    assert [row[f"proj_p{k}"] for k in ms._PRECISION_LEVELS] == pytest.approx([-0.2, -0.1, 0.0, 0.1, 0.2])
    assert row["contact_clause_rmse"] == pytest.approx(0.08)  # mean of 0.01 .. 0.15
    assert row["contact_spread_rmse"] > 0.0
    assert row["contact_spread_separation"] == pytest.approx(row["contact_spread_rmse"] / 0.05)


def test_an_ignored_clause_measures_zero() -> None:
    """A checkpoint that never saw the sentence: every chunk is the clause-off chunk."""
    clause_off = torch.randn(30, 7)
    acts = {name: clause_off.clone() for name in ms._CHANNELS}
    row = ms._channel_measurements(acts, clause_off, torch.zeros(30, 7), floor_mean=0.05)
    for key in ("precision_range", "precision_clause", "contact_clause", "contact_spread"):
        assert row[f"{key}_separation"] == 0.0
    assert row["precision_kendall_tau"] == 0.0


def test_prompt_markers_find_the_new_clauses_in_order() -> None:
    """The adapter's clause markers, run over the processor's own rendered prompt."""
    pytest.importorskip("transformers", reason="molmoact2 processor imports policy deps")
    from lerobot.policies.molmoact2.processor_molmoact2 import _build_robot_text
    from lerobot.probes.adapters.molmoact2 import MolmoAct2Adapter

    text = _build_robot_text(
        task="put the sock in the basket",
        state_string="<state>",
        num_images=0,
        metadata={"quality": 5, "mistake": False, "speed": 4, "precision": 2, "contact": 0},
    )
    chars = list(text)
    starts = {}
    for name, marker in MolmoAct2Adapter._PROMPT_CLAUSES:
        pos = MolmoAct2Adapter._find_subsequence(chars, list(marker))
        if pos is not None:
            starts[name] = pos
    assert [name for name, _ in sorted(starts.items(), key=lambda kv: kv[1])] == [
        "task_first", "state", "metadata_quality", "metadata_mistake", "metadata_speed",
        "metadata_precision", "metadata_contact", "question",
    ]


def test_trainer_forwards_the_new_columns_only_when_present() -> None:
    cfg = SimpleNamespace(policy=SimpleNamespace(future_visual_loss=None))
    with_columns = {
        "metadata_speed": torch.tensor([3.0]),
        "metadata_precision": torch.tensor([2.0]),
        "metadata_contact": torch.tensor([14.0]),
        "episode_id": torch.tensor([0]),
    }
    keys = _forwarded_complementary_keys(with_columns, cfg)
    assert {"metadata_speed", "metadata_precision", "metadata_contact"} <= set(keys)
    assert "episode_id" not in keys
    assert _forwarded_complementary_keys({"metadata_speed": torch.tensor([3.0])}, cfg) == ["metadata_speed"]
