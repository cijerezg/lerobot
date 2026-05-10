from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

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
    server._rlt_episode_id_offset = 0
    server.logger = logging.getLogger("test_policy_server_drtc")

    loaded_size = server._load_rlt_replay_file(str(replay_path), source="online")

    assert loaded_size == 2
    assert len(server._rlt_replay) == 2
    assert server._rlt_episode_id_offset == 0
    assert emitted[0][0] == "policy_server"
    assert emitted[0][1] == "rlt_replay_loaded"
    assert emitted[0][2]["rlt_replay_size"] == 2
    assert emitted[0][2]["rlt_episode_id_offset"] == 0
    assert emitted[0][2]["replay_loaded_size"] == 2
    assert emitted[0][2]["replay_source"] == "online"
    assert emitted[0][2]["replay_path"] == str(replay_path)


@require_package("grpcio", "grpc")
def test_online_replay_load_sets_episode_id_offset(tmp_path, monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    replay_path = tmp_path / "rlt_online_replay.pt"
    replay = RLTReplayBuffer(capacity=8)
    old_sample = _sample(0.0)
    old_sample.episode_id = 41
    replay.add(old_sample)
    replay.save(replay_path)

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

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
    server._rlt_episode_id_offset = 0
    server.logger = logging.getLogger("test_policy_server_drtc")

    assert server._load_rlt_replay_file(str(replay_path), source="online") == 1
    assert server._rlt_episode_id_offset == 41


@require_package("grpcio", "grpc")
def test_load_rlt_review_archive_preserves_existing_samples(tmp_path, monkeypatch):
    from lerobot.async_inference import policy_server_drtc

    archive_path = tmp_path / "rlt_review_archive.pt"
    replay = RLTReplayBuffer(capacity=8)
    replay.add(_sample(0.0))
    archived_sample = _sample(1.0)
    archived_sample.episode_id = 12
    replay.add(archived_sample)
    replay.save(archive_path)

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server._rlt_review_archive_path = str(archive_path)
    server._rlt_review_archive = []
    server._rlt_replay_lock = threading.Lock()
    server._rlt_replay = RLTReplayBuffer(capacity=8)
    server._rlt_replay_capacity = 8
    server._rlt_completed_episodes = set()
    server._rlt_train_step = 0
    server._rlt_online_collection_enabled = True
    server._rlt_online_training_enabled = True
    server._rlt_training_head = "idle"
    server._rlt_actor_disabled_by_safety = False
    server._rlt_demo_replay_size = 0
    server._rlt_online_replay_size = 0
    server._rlt_accepted_transitions = 0
    server._rlt_episode_id_offset = 0
    server.logger = logging.getLogger("test_policy_server_drtc")

    server._load_rlt_review_archive()

    assert len(server._rlt_review_archive) == 2
    assert server._rlt_episode_id_offset == 12


class _DiagnosticStub:
    def counter(self, *_args, **_kwargs):
        pass

    def set_context(self, **_kwargs):
        pass


def _server_for_accept_transition(policy_server_drtc):
    server = policy_server_drtc.PolicyServerDrtc.__new__(policy_server_drtc.PolicyServerDrtc)
    server._rlt_context_cache = policy_server_drtc.RLTSourceContextCache(max_size=4)
    server._rlt_replay_lock = threading.Lock()
    server._rlt_replay = RLTReplayBuffer(capacity=4)
    server._rlt_replay_capacity = 4
    server._rlt_review_archive_path = None
    server._rlt_review_archive = []
    server._rlt_train_step = 0
    server._rlt_online_collection_enabled = True
    server._rlt_online_training_enabled = True
    server._rlt_training_head = "idle"
    server._rlt_actor_disabled_by_safety = False
    server._rlt_demo_replay_size = 0
    server._rlt_accepted_transitions = 0
    server._rlt_online_replay_size = 0
    server._rlt_online_buffer_save_freq_transitions = 0
    server._rlt_buffer_dirty = False
    server._rlt_completed_episodes = set()
    server._rlt_episode_id_offset = 0
    server._action_encoding = "raw"
    server._action_normalizer = None
    server._metrics = SimpleNamespace(diagnostic=_DiagnosticStub())
    return server


@require_package("grpcio", "grpc")
def test_accept_rlt_transition_uses_executed_chunk_as_intervention_reference(monkeypatch):
    from lerobot.async_inference import policy_server_drtc
    from lerobot.transport import services_pb2

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

    server = _server_for_accept_transition(policy_server_drtc)

    source_reference = torch.zeros(3, 2)
    source = policy_server_drtc.RLTSourceContext(
        context_id=1,
        source_control_step=10,
        chunk_start_step=10,
        rl_token=torch.ones(4),
        proprio=torch.ones(2),
        reference_chunk=source_reference,
        anchor_state=None,
    )
    next_context = policy_server_drtc.RLTSourceContext(
        context_id=2,
        source_control_step=13,
        chunk_start_step=13,
        rl_token=torch.ones(4) * 2,
        proprio=torch.ones(2) * 2,
        reference_chunk=torch.ones(3, 2) * 3,
        anchor_state=None,
    )
    server._rlt_context_cache.put(source)
    server._rlt_context_cache.put(next_context)

    executed = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
    transition = services_pb2.RLTTransitionChunk(
        episode_id=7,
        source_rlt_context_id=1,
        next_rlt_context_id=2,
        chunk_start_step=10,
        num_actions=3,
        action_dim=2,
        executed_actions_f32=executed.numpy().tobytes(),
        reward=0.0,
        done=False,
        is_intervention=True,
    )

    server._accept_rlt_transition(transition)

    sample = server._rlt_replay.samples()[0]
    assert torch.equal(sample.reference_chunk, executed)
    assert torch.equal(sample.executed_chunk, executed)
    assert torch.equal(sample.next_reference_chunk, next_context.reference_chunk)


@require_package("grpcio", "grpc")
def test_accept_rlt_transition_keeps_source_reference_for_non_intervention(monkeypatch):
    from lerobot.async_inference import policy_server_drtc
    from lerobot.transport import services_pb2

    monkeypatch.setattr(policy_server_drtc, "emit_status", lambda *_args, **_kwargs: None)

    server = _server_for_accept_transition(policy_server_drtc)

    source_reference = torch.ones(3, 2)
    source = policy_server_drtc.RLTSourceContext(
        context_id=1,
        source_control_step=10,
        chunk_start_step=10,
        rl_token=torch.ones(4),
        proprio=torch.ones(2),
        reference_chunk=source_reference,
        anchor_state=None,
    )
    server._rlt_context_cache.put(source)

    executed = torch.zeros(3, 2, dtype=torch.float32)
    transition = services_pb2.RLTTransitionChunk(
        episode_id=7,
        source_rlt_context_id=1,
        next_rlt_context_id=0,
        chunk_start_step=10,
        num_actions=3,
        action_dim=2,
        executed_actions_f32=executed.numpy().tobytes(),
        reward=0.0,
        done=True,
        is_intervention=False,
    )

    server._accept_rlt_transition(transition)

    sample = server._rlt_replay.samples()[0]
    assert torch.equal(sample.reference_chunk, source_reference)
    assert torch.equal(sample.executed_chunk, executed)
