import torch
import pytest

from lerobot.rl.buffer import ReplayBuffer, shift_with_offsets
from lerobot.rl.data_sources.rebot_role_adapter import RoleAlignedBuffer

KEY = "observation.images.top"


def buffer(*, capacity=12, size=10, position=10, stride=1):
    result = ReplayBuffer(capacity=capacity, device="cpu", state_keys=[KEY])
    result.size, result.position, result.image_stride = size, position, stride
    result.states = {KEY: torch.arange(0, capacity, stride)[:, None, None, None].expand(-1, 3, 2, 2)}
    result.next_states = result.states
    result.dones = torch.zeros(capacity, dtype=torch.bool)
    result.truncateds = torch.zeros(capacity, dtype=torch.bool)
    result.complementary_info = {}
    result.complementary_info_keys = []
    return result


def test_future_frames_respect_episodes_tail_and_sparse_image_grid():
    replay = buffer(stride=2)
    replay.dones[5] = True
    replay.configure_future_images(4, [KEY])
    result = replay._gather_future_images(torch.tensor([0, 2, 4, 6, 8]))
    assert result["future_visual_valid"].tolist() == [True, False, False, False, False]
    assert result[f"future.{KEY}"][:, 0, 0, 0].tolist() == [4, 2, 4, 6, 8]


def test_future_frames_respect_circular_write_head_and_truncation():
    replay = buffer(size=12, position=4)
    replay.truncateds[6] = True
    replay.configure_future_images(3, [KEY])
    result = replay._gather_future_images(torch.tensor([9, 0, 2, 4]))
    assert result["future_visual_valid"].tolist() == [True, True, False, False]
    assert result[f"future.{KEY}"][:, 0, 0, 0].tolist() == [0, 3, 2, 4]


def test_terminal_frame_itself_can_be_target_but_cannot_be_crossed():
    replay = buffer()
    replay.dones[4] = True
    replay.configure_future_images(4, [KEY])
    result = replay._gather_future_images(torch.tensor([0, 1]))
    assert result["future_visual_valid"].tolist() == [True, False]


def test_off_by_default_and_stride_errors_are_explicit():
    replay = buffer(stride=3)
    assert replay.future_image_offset == 0
    with pytest.raises(ValueError, match="aligned"):
        replay.configure_future_images(4, [KEY])
    with pytest.raises(ValueError, match="missing"):
        replay.configure_future_images(3, ["observation.images.missing"])


def test_future_keys_follow_role_renames_and_absent_camera_masks():
    replay = buffer()
    wrist = "observation.images.wrist"
    replay.states[wrist] = replay.states[KEY].clone()
    replay.state_keys.append(wrist)
    replay.configure_future_images(3, [KEY, wrist])
    aligned = RoleAlignedBuffer(replay, action_layout_id=0, image_size=(2, 2))
    assert replay.future_image_keys == (
        "observation.images.external_0", "observation.images.wrist_0"
    )
    sample = {
        "state": {key: value[:2] for key, value in replay.states.items()},
        "next_state": {key: value[:2] for key, value in replay.states.items()},
        "action": torch.zeros(2, 1, 8), "reward": torch.zeros(2),
        "complementary_info": replay._gather_future_images(torch.tensor([0, 1])),
    }
    result = aligned.decorate(sample)["complementary_info"]
    assert "future.observation.images.external_1" in result
    assert not result["future.camera_is_present.observation.images.external_1"].any()
    assert result["future.camera_is_present.observation.images.wrist_0"].all()


def test_shared_drq_does_not_create_false_temporal_change():
    image = torch.randint(256, (3, 3, 8, 8), dtype=torch.uint8)
    offsets = torch.tensor([[0, 4], [8, 0], [3, 6]])
    history = image[:, None].expand(-1, 4, -1, -1, -1)
    shifted = shift_with_offsets(image, offsets)
    shifted_history = shift_with_offsets(history, offsets)
    for frame in range(4):
        torch.testing.assert_close(shifted, shifted_history[:, frame])
    torch.testing.assert_close(shifted, shift_with_offsets(image.clone(), offsets))


def test_sampler_uses_identical_crops_for_current_history_and_future():
    replay = ReplayBuffer(
        20, "cpu", [KEY], use_drq=True, optimize_memory=True,
        history_offsets={KEY: [2, 1]},
    )
    image = torch.randint(256, (1, 3, 12, 12), dtype=torch.uint8)
    for row in range(12):
        replay.add(
            state={KEY: image}, action=torch.zeros(1, 2), reward=0., next_state=None,
            done=row == 11, truncated=False,
        )
    replay.configure_future_images(3, [KEY])
    sample = replay.sample(10, action_chunk_size=1)
    current = sample["state"][KEY]
    torch.testing.assert_close(current, sample["complementary_info"][f"future.{KEY}"])
    for slot in range(2):
        torch.testing.assert_close(current, sample["state"][f"history.{KEY}"][:, slot])


@pytest.mark.parametrize("enabled", [True, False, None])
def test_offline_startup_configures_future_frames_in_its_own_scope(monkeypatch, enabled):
    """Run the real startup through replay setup, with cheap model/data loaders."""
    from types import SimpleNamespace
    import lerobot.scripts.rl_offline as offline
    from lerobot.rl.training_runtime import TrainingRuntime

    class StartupReached(Exception):
        pass

    class StartupBuffer(ReplayBuffer):
        def __len__(self):
            # This is the first statement after every buffer has been configured;
            # stop before iterators or training can begin.
            raise StartupReached

    buffers = []
    for fps in (30, 15):
        replay = StartupBuffer(12, "cpu", [KEY], use_drq=False)
        replay.image_stride = 3
        replay.states = {KEY: torch.zeros(4, 3, 2, 2, dtype=torch.uint8)}
        replay.dataset = SimpleNamespace(fps=fps)
        buffers.append(replay)

    policy_cfg = SimpleNamespace(
        device="cpu", storage_device="cpu", offline_steps=1, offline_buffer_capacity=12,
        reward_normalization_constant=1.0, terminal_failure_reward=-10.0,
    )
    if enabled is not None:
        policy_cfg.future_visual_loss = SimpleNamespace(enabled=enabled, horizon_seconds=4.0)
    cfg = SimpleNamespace(
        policy=policy_cfg, env=SimpleNamespace(fps=30), skip_critic=True,
        log_freq=1, save_freq=1, save_checkpoint=False, seed=None, batch_size=2,
    )
    trainer = SimpleNamespace(
        make_policy=lambda cfg: torch.nn.Linear(1, 1),
        freeze_model=lambda *args: None,
        make_processors=lambda *args, **kwargs: (None, None),
        get_optimizer_groups=lambda *args: [],
        sync_subtask_vocabulary=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(offline.Trainer, "for_config", lambda cfg: trainer)
    monkeypatch.setattr(offline, "get_offline_dataset_sources", lambda cfg: [SimpleNamespace(
        name="fixture", root="unused", embodiment=None,
    )])
    monkeypatch.setattr(offline, "load_offline_dataset", lambda *args: buffers[0].dataset)
    monkeypatch.setattr(offline, "pool_lowdim_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(offline, "materialize_dataset_labels", lambda *args, **kwargs: None)
    monkeypatch.setattr(offline, "buffer_state_keys", lambda cfg: [KEY])
    monkeypatch.setattr(offline.ReplayBuffer, "from_lerobot_dataset", lambda *args, **kwargs: buffers[0])
    monkeypatch.setattr(offline, "load_additional_offline_buffers", lambda **kwargs: buffers[1:])
    for name in ("build_named_adamw_optimizers", "build_named_schedulers", "build_pretrained_merges"):
        monkeypatch.setattr(offline, name, lambda *args, **kwargs: {})

    with pytest.raises(StartupReached):
        offline.run_offline_training(cfg, None, None, TrainingRuntime(device="cpu"))
    assert [replay.future_image_offset for replay in buffers] == ([120, 60] if enabled else [0, 0])
