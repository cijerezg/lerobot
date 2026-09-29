"""AWR crosses subtask boundaries without inventing targets for missing frames."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from lerobot.rl.awr import AWRCalibration
from lerobot.rl.awr_calibration import calibration_provenance
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.data_sources.diverse_actor_buffer import DiverseActorBuffer, DiverseSampleSpec, critic_view
from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer
from lerobot.types import TransitionKey


STATE = "observation.state"


def replay(capacity, count, *, episode_ends=(), mistakes=()):
    buffer = ReplayBuffer(capacity=capacity, device="cpu", state_keys=[STATE],
                          optimize_memory=True, use_drq=False, reward_normalization_constant=12.)
    for i in range(count):
        buffer.add({STATE: torch.tensor([[float(i)]])}, torch.tensor([[float(i)]]),
                   0., next_state=None, done=i in episode_ends, truncated=False,
                   complementary_info={
                       "critic_subtask_terminal": torch.tensor([i in (0, 3)]),
                       "critic_mistake_onset": torch.tensor([i in mistakes]),
                       "subtask_index": torch.tensor([i]),
                       "metadata_speed": torch.tensor([i + 10.]),
                       "depth.wrist.depth": torch.tensor([[float(i)]]),
                   })
    buffer.configure_critic_rewards("subtask", 5., bootstrap_subtasks=True)
    return buffer


def test_replay_bootstraps_subtasks_but_not_episode_ends_and_keeps_penalties():
    buffer = replay(6, 6, episode_ends=(2, 5), mistakes=(0,))
    batch = buffer.sample(4, action_chunk_size=2, indices=[0, 1, 3, 4])
    assert batch["done"].tolist() == [0., 1., 0., 1.]
    assert batch["reward"].tolist() == pytest.approx([-6/12, 0., -1/12, 0.])
    baseline = (batch["done"] - 1) / 12
    assert (batch["reward"] < baseline - 1e-6).tolist() == [True, False, False, False]
    assert batch["next_state"][STATE].flatten().tolist() == [2., 1., 5., 4.]
    info = batch["complementary_info"]
    assert info["subtask_index"].flatten().tolist() == [0, 1, 3, 4]
    assert info["metadata_speed"].flatten().tolist() == [10., 11., 13., 14.]
    assert info["next_depth.wrist.depth"].flatten().tolist() == [2., 1., 5., 4.]
    # The physical markers remain unchanged by the reward view.
    assert buffer.dones.tolist() == [False, False, True, False, False, True]


@pytest.mark.parametrize("capacity,count,index,chunk", [(4, 4, 3, 1), (6, 4, 2, 2)])
def test_replay_end_without_successor_never_wraps_or_reads_unfilled_storage(capacity, count, index, chunk):
    batch = replay(capacity, count).sample(1, action_chunk_size=chunk, indices=[index])
    assert batch["done"].item() == 1.
    assert batch["reward"].item() == 0.
    assert batch["next_state"][STATE].item() == index


def row(frame, end=5, episode="ep", onsets=()):
    return {"episode_id": episode, "anchor_frame": frame, "native_rate_hz": 10.,
            "critic_end_timestep_exclusive": end, "mistake_onset_timesteps": onsets}


def test_diverse_cache_gaps_skip_awr_and_real_episode_ends_do_not():
    rows = [row(0, onsets=(5,)), row(10), row(30, end=35), row(0, episode="other")]
    view = critic_view(rows, 1., bootstrap_subtasks=True, episode_frames={"ep": 40, "other": 5})
    assert view.next_row.tolist() == [1, -1, -1, -1]
    assert view.done.tolist() == [False, False, True, True]
    assert view.skip.tolist() == [False, True, False, False]
    assert view.mistake.tolist() == [True, False, False, False]
    # The first sample used to terminate at its subtask boundary at frame 5.
    assert critic_view(rows, 1.).done[0]


def test_diverse_collation_loads_successor_and_keeps_anchor_metadata():
    rows = [row(0), row(10), row(30, end=35)]
    buffer = object.__new__(DiverseActorBuffer)
    buffer.spec = DiverseSampleSpec(load_images=False, load_depth=False)
    buffer.device = "cpu"
    buffer.serve_critic = True
    buffer.reward_normalization_constant = 12.
    buffer.critic_mistake_penalty = 5.
    buffer._critic = critic_view(rows, 1., bootstrap_subtasks=True, episode_frames={"ep": 40})
    fields = ("source_id", "episode_position", "anchor_index", "action_layout_id", "embodiment_index",
              "task_index", "subtask_index", "quality_provenance_id", "retention_reason_id")
    buffer._identity = [NS(**dict.fromkeys(fields, i), quality=4., quality_is_valid=True,
                           mistake=False, speed=i+1, precision=2, contact=3) for i in range(3)]
    samples = [{"row_index": i, "native_width": 1, "action": np.zeros((30, 1), dtype=np.float32)}
               for i in range(3)]
    loaded = []
    def load(index):
        loaded.append(index)
        return samples[index]
    buffer.load_sample = load
    buffer._observation = lambda values: ({STATE: torch.tensor([[float(v["row_index"])] for v in values])}, {})
    batch = buffer.collate([0, 1, 2])
    assert loaded == [0, 1, 2, 1]  # only the valid, nonterminal successor is loaded
    assert batch["next_state"][STATE].flatten().tolist() == [1., 1., 2.]
    assert batch["done"].tolist() == [0., 0., 1.]
    assert batch["reward"].tolist() == pytest.approx([-1/12, -1/12, 0.])
    assert batch["complementary_info"]["critic_skip"].tolist() == [False, True, False]
    assert batch["complementary_info"]["subtask_index"].tolist() == [0, 1, 2]
    assert batch["complementary_info"]["metadata_speed"].tolist() == [1., 2., 3.]


def test_successor_critic_prompt_retains_all_current_metadata():
    trainer = MolmoAct2Trainer()
    comp = {"subtask_index": torch.tensor([2]), "metadata_quality": torch.tensor([4.]),
            "metadata_speed": torch.tensor([3.]),
            "metadata_mistake": torch.tensor([0.]), "metadata_precision": torch.tensor([2.]),
            "metadata_contact": torch.tensor([1.]), "embodiment_index": torch.tensor([4])}
    raw = {"state": {STATE: torch.tensor([[0.]])}, "next_state": {STATE: torch.tensor([[1.]])},
           "reward": torch.tensor([-1/12]), "done": torch.tensor([False]), "complementary_info": comp}
    cfg = NS(policy=NS(task="pick", critic_reward_mode="subtask", advantage_bootstrap_subtasks=True))
    current, successor, _, done = trainer._critic_batches(raw, lambda values: values, cfg)
    assert current[STATE].item() == 0 and successor[STATE].item() == 1
    assert current["task"] == successor["task"] == ["pick"]
    for key, value in comp.items():
        torch.testing.assert_close(current[TransitionKey.COMPLEMENTARY_DATA][key], value)
        torch.testing.assert_close(successor[TransitionKey.COMPLEMENTARY_DATA][key], value)
    assert not done.item()


def test_terminal_rule_change_rejects_old_calibration(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"test checkpoint identity")
    cfg = NS(policy=NS(critic_pretrained_path=str(tmp_path), advantage_bootstrap_subtasks=False), dataset={})
    original = calibration_provenance(cfg)
    calibration = AWRCalibration.fit(torch.tensor([-1., 0., 1.]), ["group"]*3, 1., 3., original)
    path = tmp_path / "awr_calibration.json"
    calibration.save(path)
    cfg.policy.advantage_bootstrap_subtasks = True
    with pytest.raises(ValueError, match="does not match"):
        AWRCalibration.load(path, provenance=calibration_provenance(cfg))


def test_probe_targets_match_bootstrapped_replay_rewards():
    from lerobot.probes.critic import _transition_target
    targets = {"mode": "subtask", "episode_end": np.array([False, False, False, True]),
               "terminals": np.array([True, False, False, True]),
               "mistake_onset": np.array([True, False, False, False])}
    cfg = NS(policy=NS(advantage_bootstrap_subtasks=True))
    assert _transition_target(targets, range(4), 0, 1, 5., 12., cfg) == (-6/12, False)
    assert _transition_target(targets, range(4), 1, 1, 5., 12., cfg) == (-1/12, False)
    assert _transition_target(targets, range(4), 3, 1, 5., 12., cfg) == (0., True)
    cfg.policy.advantage_bootstrap_subtasks = False
    assert _transition_target(targets, range(4), 0, 1, 5., 12., cfg) == (-5/12, True)
