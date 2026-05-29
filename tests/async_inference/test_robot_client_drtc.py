from queue import Queue
from types import SimpleNamespace

import numpy as np

from lerobot.async_inference.helpers import TimedAction
from lerobot.async_inference.robot_client_drtc import ReceivedActionChunk, RobotClientDrtc
from lerobot.transport import services_pb2


class _DiagnosticStub:
    def __init__(self):
        self.counters: dict[str, int] = {}

    def counter(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value


class _TeleopStub:
    def __init__(self):
        self.is_intervening = False

    def _handle_key_char(self, char: str) -> None:
        if char == "5":
            self.is_intervening = not self.is_intervening


def _make_rlt_client() -> tuple[RobotClientDrtc, _DiagnosticStub]:
    client = RobotClientDrtc.__new__(RobotClientDrtc)
    diagnostic = _DiagnosticStub()
    client.config = SimpleNamespace(rlt_online_collection_enabled=True, rlt_chunk_size=3)
    client._metrics = SimpleNamespace(diagnostic=diagnostic)
    client.logger = SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    client._trajectory_viz_client = None
    client._rlt_transition_queue = Queue()
    client._rlt_current_episode_transition_buffer = []
    client._rlt_pending_chunks = {}
    client._rlt_executed_actions = {}
    client._rlt_emitted_context_ids = set()
    client._rlt_prebuffer_pending_chunks = {}
    client._rlt_prebuffer_executed_actions = {}
    client._rlt_prebuffer_step_window = 6
    client._rlt_rollout_id = 1
    client._rlt_rollout_open = True
    client._rlt_rollout_start_ts = 1.0
    client._rlt_rollout_start_step = 0
    client._rlt_episode_id = 3
    client._rlt_episode_open = True
    client._rlt_critical_start_ts = 1.0
    client._rlt_critical_start_step = 0
    client._rlt_critical_end_ts = None
    client._rlt_critical_end_step = 0
    client._rlt_critical_pending_label = False
    client._rlt_phase = "recording"
    client._rlt_completed_episodes_count = 0
    client._rlt_success_episodes_count = 0
    client._rlt_failure_episodes_count = 0
    client._rlt_discarded_episodes_count = 0
    client._rlt_current_episode_transitions = 1
    client._rlt_last_episode_label = None
    client._rlt_phase_intervening = False
    client._tui_intervention_enabled = False
    client._teleop_device = _TeleopStub()
    client.action_step = 0
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
    client._rlt_current_episode_transition_buffer.append(
        services_pb2.RLTTransitionChunk(
            episode_id=3,
            source_rlt_context_id=12,
            next_rlt_context_id=13,
            chunk_start_step=22,
            num_actions=2,
            action_dim=2,
            executed_actions_f32=actions.tobytes(),
            reward=0.0,
            done=False,
        )
    )

    client._rlt_maybe_emit_transitions(reward=1.0, done=True, success=True, failure=False)

    first = client._rlt_transition_queue.get_nowait()
    terminal = client._rlt_transition_queue.get_nowait()
    assert first.done is False
    assert first.success is True
    assert first.failure is False
    assert first.reward == 0.0
    assert terminal.done is True
    assert terminal.success is True
    assert terminal.failure is False
    assert terminal.reward == 1.0
    assert terminal.next_rlt_context_id == 0
    assert diagnostic.counters["rlt_terminal_transition_backfilled"] == 1


def test_rlt_action_modified_detects_robot_side_action_rewrite():
    requested = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    applied = np.asarray([1.0, 2.5, 3.0], dtype=np.float32)

    assert RobotClientDrtc._rlt_action_was_modified(requested, applied)
    assert not RobotClientDrtc._rlt_action_was_modified(requested, requested.copy())


def test_rlt_collectable_chunk_uses_shifted_window_start():
    client, _diagnostic = _make_rlt_client()
    actions = [
        TimedAction(action=np.asarray([float(i)], dtype=np.float32), action_step=100 + i)
        for i in range(5)
    ]
    chunk = ReceivedActionChunk(
        actions=actions,
        src_control_step=10,
        chunk_start_step=100,
        measured_latency=0.0,
        rlt_context_id=21,
        policy_mode="rlt_actor",
        rlt_collectable=True,
        rlt_window_start_index=2,
    )

    client._rlt_note_collectable_chunk(chunk)

    pending = client._rlt_pending_chunks[21]
    assert pending.chunk_start_step == 102
    assert pending.num_actions == 3
    assert pending.action_dim == 1


def test_end_critical_phase_waits_for_label_before_flushing():
    client, _diagnostic = _make_rlt_client()
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

    client._rlt_end_critical_phase()

    assert client._rlt_episode_open is False
    assert client._rlt_critical_pending_label is True
    assert client._rlt_transition_queue.empty()
    assert client._rlt_current_episode_transition_buffer[-1].done is True

    client._rlt_label_current_critical_phase(success=True)

    terminal = client._rlt_transition_queue.get_nowait()
    assert terminal.done is True
    assert terminal.success is True
    assert terminal.failure is False
    assert terminal.reward == 1.0
    assert client._rlt_critical_pending_label is False
    assert client._rlt_completed_episodes_count == 1
    assert client._rlt_success_episodes_count == 1


def test_discard_critical_phase_keeps_rollout_open():
    client, _diagnostic = _make_rlt_client()
    client._rlt_current_episode_transition_buffer.append(services_pb2.RLTTransitionChunk(episode_id=3))
    client._rlt_current_episode_transitions = 1

    client._rlt_discard_current_episode()

    assert client._rlt_rollout_open is True
    assert client._rlt_episode_open is False
    assert client._rlt_critical_pending_label is False
    assert client._rlt_current_episode_transition_buffer == []
    assert client._rlt_discarded_episodes_count == 1


def test_rollout_state_gates_policy_execution_not_critical_phase():
    client, _diagnostic = _make_rlt_client()

    client._rlt_rollout_open = True
    client._rlt_episode_open = False
    assert not client._waiting_for_rlt_episode_start()

    client._rlt_rollout_open = False
    assert client._waiting_for_rlt_episode_start()


def test_toggle_critical_intervention_starts_and_successfully_labels():
    client, _diagnostic = _make_rlt_client()
    client._rlt_episode_open = False
    client._rlt_current_episode_transition_buffer = []
    client._rlt_current_episode_transitions = 0

    assert client._rlt_toggle_critical_intervention() is True
    assert client._rlt_episode_open is True
    assert client._teleop_device.is_intervening is True

    client._rlt_current_episode_transition_buffer.append(services_pb2.RLTTransitionChunk(episode_id=4))
    client._rlt_current_episode_transitions = 1

    assert client._rlt_toggle_critical_intervention() is False
    assert client._rlt_episode_open is False
    assert client._teleop_device.is_intervening is False
    assert client._rlt_completed_episodes_count == 1
    assert client._rlt_success_episodes_count == 1

    terminal = client._rlt_transition_queue.get_nowait()
    assert terminal.done is True
    assert terminal.success is True
    assert terminal.failure is False
    assert terminal.reward == 1.0


def test_start_critical_phase_seeds_pre_intervention_prebuffer():
    client, _diagnostic = _make_rlt_client()
    client._rlt_episode_open = False
    client._rlt_current_episode_transitions = 0

    actions = [
        TimedAction(action=np.asarray([float(i)], dtype=np.float32), action_step=100 + i)
        for i in range(3)
    ]
    chunk = ReceivedActionChunk(
        actions=actions,
        src_control_step=10,
        chunk_start_step=100,
        measured_latency=0.0,
        rlt_context_id=21,
        policy_mode="rlt_actor",
        rlt_collectable=True,
    )
    client._rlt_note_collectable_chunk(chunk)
    for step in range(100, 102):
        client._rlt_record_executed_action(step, np.asarray([float(step)], dtype=np.float32), is_intervention=False)

    assert 21 in client._rlt_prebuffer_pending_chunks
    assert set(client._rlt_prebuffer_executed_actions) == {100, 101}

    client._rlt_start_critical_phase()

    assert 21 in client._rlt_pending_chunks
    assert set(client._rlt_executed_actions) == {100, 101}
    assert client._rlt_prebuffer_pending_chunks == {}
    assert client._rlt_prebuffer_executed_actions == {}

    client._rlt_record_executed_action(102, np.asarray([102.0], dtype=np.float32), is_intervention=True)
    client._rlt_maybe_emit_transitions(done=True, flush_on_done=False)

    assert client._rlt_current_episode_transition_buffer[-1].is_intervention is True
