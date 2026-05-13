from queue import Queue
from types import SimpleNamespace

import numpy as np

from lerobot.async_inference.robot_client_drtc import RobotClientDrtc
from lerobot.transport import services_pb2


class _DiagnosticStub:
    def __init__(self):
        self.counters: dict[str, int] = {}

    def counter(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value


def _make_rlt_client() -> tuple[RobotClientDrtc, _DiagnosticStub]:
    client = RobotClientDrtc.__new__(RobotClientDrtc)
    diagnostic = _DiagnosticStub()
    client.config = SimpleNamespace(rlt_online_collection_enabled=True)
    client._metrics = SimpleNamespace(diagnostic=diagnostic)
    client._rlt_transition_queue = Queue()
    client._rlt_current_episode_transition_buffer = []
    client._rlt_pending_chunks = {}
    client._rlt_executed_actions = {}
    client._rlt_emitted_context_ids = set()
    client._rlt_episode_id = 3
    client._rlt_episode_open = True
    client._rlt_phase = "recording"
    client._rlt_completed_episodes_count = 0
    client._rlt_discarded_episodes_count = 0
    client._rlt_current_episode_transitions = 1
    client._rlt_last_episode_label = None
    client._rlt_phase_intervening = False
    return client, diagnostic


def test_rlt_episode_done_backfills_last_buffered_transition_as_terminal():
    client, diagnostic = _make_rlt_client()
    actions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    client._rlt_current_episode_transition_buffer.append(
        services_pb2.RLTTransitionChunk(
            episode_id=3,
            source_rlt_context_id=11,
            next_rlt_context_id=12,
            chunk_start_step=20,
            num_actions=2,
            action_dim=2,
            executed_actions_f32=actions.tobytes(),
            reward=0.0,
            done=False,
        )
    )

    client._rlt_maybe_emit_transitions(reward=1.0, done=True, success=True, failure=False)

    queued = client._rlt_transition_queue.get_nowait()
    assert queued.done is True
    assert queued.success is True
    assert queued.failure is False
    assert queued.reward == 1.0
    assert queued.next_rlt_context_id == 0
    assert diagnostic.counters["rlt_terminal_transition_backfilled"] == 1
