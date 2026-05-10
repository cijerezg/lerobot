from __future__ import annotations

import logging
import threading

import torch

from lerobot.rl.rlt_buffer import RLTReplayBuffer, RLTReplaySample
from tests.utils import require_package


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


@require_package("grpcio", "grpc")
def test_load_rlt_replay_file_emits_status_with_replay_size(tmp_path, monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    replay_path = tmp_path / "rlt_online_replay.pt"
    replay = RLTReplayBuffer(capacity=8)
    replay.add(_sample(0.0))
    replay.add(_sample(1.0))
    replay.save(replay_path)

    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        policy_server_drtc,
        "emit_status",
        lambda source, event, **fields: emitted.append((source, event, fields)),
    )

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server._rlt_replay_lock = threading.Lock()
    server._rlt_replay_capacity = 8
    server._rlt_replay = RLTReplayBuffer(capacity=8)
    server._rlt_completed_episodes = set()
    server._rlt_train_step = 0
    server._rlt_online_collection_enabled = True
    server._rlt_online_training_enabled = True
    server._rlt_training_head = "idle"
    server._rlt_actor_disabled_by_safety = False
    server._rlt_demo_replay_size = 0
    server._rlt_online_replay_size = 0
    server._rlt_accepted_transitions = 0
    server.logger = logging.getLogger("test_policy_server_drtc")

    loaded_size = server._load_rlt_replay_file(str(replay_path), source="online")

    assert loaded_size == 2
    assert len(server._rlt_replay) == 2
    assert emitted[0][0] == "policy_server"
    assert emitted[0][1] == "rlt_replay_loaded"
    assert emitted[0][2]["rlt_replay_size"] == 2
    assert emitted[0][2]["replay_loaded_size"] == 2
    assert emitted[0][2]["replay_source"] == "online"
    assert emitted[0][2]["replay_path"] == str(replay_path)
