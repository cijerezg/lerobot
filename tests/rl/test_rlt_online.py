from types import SimpleNamespace

import torch

from lerobot.rl.rlt_buffer import RLTReplayBuffer, RLTReplaySample
from lerobot.rl.rlt_pi05 import (
    RLTCriticEnsemble,
    RLTActorHead,
    rlt_actor_loss,
    rlt_critic_loss,
    save_rlt_head_checkpoint,
    soft_update_rlt_target,
)


def _sample(offset: float = 0.0) -> RLTReplaySample:
    return RLTReplaySample(
        rl_token=torch.ones(4) + offset,
        proprio=torch.ones(2) + offset,
        reference_chunk=torch.zeros(3, 2) + offset,
        executed_chunk=torch.ones(3, 2) + offset,
        next_rl_token=torch.ones(4) * 2 + offset,
        next_proprio=torch.ones(2) * 2 + offset,
        next_reference_chunk=torch.zeros(3, 2) + offset,
        reward=1.0,
        done=False,
        is_intervention=True,
    )


def test_rlt_replay_buffer_samples_expected_shapes():
    replay = RLTReplayBuffer(capacity=3)
    replay.add(_sample(0.0))
    replay.add(_sample(1.0))

    batch = replay.sample(2)

    assert batch["rl_token"].shape == (2, 4)
    assert batch["proprio"].shape == (2, 2)
    assert batch["reference_chunk"].shape == (2, 3, 2)
    assert batch["executed_chunk"].shape == (2, 3, 2)
    assert batch["reward"].shape == (2, 1)
    assert batch["done"].shape == (2, 1)
    assert batch["is_intervention"].shape == (2, 1, 1)


def test_rlt_replay_buffer_persists_round_trip(tmp_path):
    replay = RLTReplayBuffer(capacity=3)
    replay.add(_sample(0.0))
    replay.add(_sample(1.0))
    path = tmp_path / "rlt_replay.pt"

    replay.save(path)
    loaded = RLTReplayBuffer.load(path, capacity=5)

    assert len(loaded) == 2
    assert loaded.capacity == 5
    batch = loaded.sample(2)
    assert batch["rl_token"].shape == (2, 4)
    assert set(batch["reward"].flatten().tolist()) == {1.0}


def test_rlt_heads_support_configurable_depth_and_critic_ensemble():
    actor = RLTActorHead(
        token_dim=4,
        proprio_dim=2,
        action_dim=2,
        chunk_size=3,
        hidden_dim=[16, 12, 8],
    )
    critic = RLTCriticEnsemble(
        num_critics=4,
        token_dim=4,
        proprio_dim=2,
        action_dim=2,
        chunk_size=3,
        hidden_dim=[16, 12, 8],
    )
    sample = _sample()
    rl_token = sample.rl_token.unsqueeze(0)
    proprio = sample.proprio.unsqueeze(0)
    reference = sample.reference_chunk.unsqueeze(0)

    actor_actions = actor(rl_token, proprio, reference)
    critic_values = critic.values(rl_token, proprio, actor_actions)

    assert actor.hidden_dims == [16, 12, 8]
    assert actor_actions.shape == (1, 3, 2)
    assert critic_values.shape == (4, 1, 1)


def test_rlt_losses_and_checkpoint_helpers(tmp_path):
    policy = SimpleNamespace(
        config=SimpleNamespace(
            rlt_bc_beta=0.5,
            rlt_reference_dropout_p=0.0,
            rlt_jerk_beta=0.1,
            rlt_bc_action_weights=[1.0, 0.5],
        ),
        rlt_actor=RLTActorHead(token_dim=4, proprio_dim=2, action_dim=2, chunk_size=3, hidden_dim=8),
        rlt_critic=RLTCriticEnsemble(
            num_critics=2,
            token_dim=4,
            proprio_dim=2,
            action_dim=2,
            chunk_size=3,
            hidden_dim=8,
        ),
    )
    policy.rlt_critic_target = RLTCriticEnsemble(
        num_critics=2,
        token_dim=4,
        proprio_dim=2,
        action_dim=2,
        chunk_size=3,
        hidden_dim=8,
    )
    policy.rlt_critic_target.load_state_dict(policy.rlt_critic.state_dict())

    replay = RLTReplayBuffer(capacity=4)
    replay.add(_sample(0.0))
    replay.add(_sample(1.0))
    batch = replay.sample(2)

    critic_loss = rlt_critic_loss(policy, batch, discount=0.9)
    actor_loss = rlt_actor_loss(policy, batch)
    critic_loss_with_stats, critic_stats = rlt_critic_loss(policy, batch, discount=0.9, return_stats=True)
    actor_loss_with_stats, actor_stats = rlt_actor_loss(policy, batch, return_stats=True)
    assert critic_loss.ndim == 0
    assert actor_loss.ndim == 0
    assert critic_loss_with_stats.ndim == 0
    assert actor_loss_with_stats.ndim == 0
    assert "target_q_abs_max" in critic_stats
    assert "action_deviation_abs_max" in actor_stats

    before = [p.detach().clone() for p in policy.rlt_critic_target.parameters()]
    for p in policy.rlt_critic.parameters():
        p.data.add_(0.1)
    soft_update_rlt_target(policy, tau=0.5)
    after = list(policy.rlt_critic_target.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))

    ckpt_path = tmp_path / "rlt_head.pt"
    save_rlt_head_checkpoint(policy, ckpt_path, step=7, config={"x": 1})
    checkpoint = torch.load(ckpt_path, weights_only=True)
    assert checkpoint["step"] == 7
    assert "rlt_actor" in checkpoint
    assert "rlt_critic" in checkpoint
    assert "rlt_critic_target" in checkpoint
