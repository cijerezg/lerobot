#!/usr/bin/env python
"""
Generic async actor for distributed HILSerl online RL training.

Works for any model registered with the Trainer ABC (MolmoAct2, PI05, …).
RTC is the default runtime: model-specific observation preprocessing is isolated
in Trainer.build_inference_batch(), while rtc_actor_runtime owns ActionQueue,
latency-aware replanning, intervention resets, and smooth execution.

The older simple chunk-deque runtime is still present as a debug fallback.

Three gRPC background threads (from actor.py):
  receive_policy    — pulls updated weights from the learner.
  send_transitions  — forwards completed episode transitions to the learner.
  send_interactions — forwards episode stats for W&B logging.

Usage:
    python -m lerobot.rl.rl_actor_async \
        --config_path lerobot/src/lerobot/rl/config_rl.yaml
"""
from __future__ import annotations

import logging
import os
import time
import traceback
import threading
from collections import deque
from threading import Thread

import torch
import torch.nn as nn

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.processor import TransitionKey
from lerobot.rl.actor import (
    establish_learner_connection,
    learner_service_client,
    push_transitions_to_transport_queue,
    receive_policy,
    send_interactions,
    send_transitions,
)
from lerobot.rl.gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from lerobot.rl.inference_utils import convert_env_obs_to_policy_format
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.rl.rl_trainer import Trainer
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport.utils import bytes_to_state_dict, python_object_to_bytes
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.utils import init_logging

import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
import lerobot.rl.pi05.rl_pi05            # noqa: F401 — registers PI05RLConfig

logger = logging.getLogger(__name__)


# ── Thread-safe shared state ──────────────────────────────────────────────────


class SharedState:
    """Thread-safe state bag shared between the inference and env threads."""

    def __init__(self, chunk_size: int) -> None:
        self.lock = threading.Lock()
        self._chunk_size = chunk_size

        # Observation + teleop
        self.latest_obs: dict | None = None
        self.is_intervening: bool = False

        # Episode lifecycle
        self.episode_active: bool = False
        self.running: bool = True
        self.policy_reset_requested: bool = False

        # Weight update handshake
        self.update_parameters_requested: bool = False
        self.params_loaded_event = threading.Event()
        self.params_loaded_event.set()  # set so first episode doesn't block if no params arrive

        # Chunk double-buffer: env signals need → inference fills → env consumes
        self._action_deque: deque = deque()
        self.chunk_request_event = threading.Event()
        self.chunk_ready_event = threading.Event()

        # Metrics
        self.env_wait_time: float = 0.0
        self.env_steps: int = 0
        self.env_active_time_total: float = 0.0
        self.inference_latencies: list[float] = []

    # -- observation ----------------------------------------------------------

    def update_observation(self, obs: dict, is_intervening: bool) -> None:
        with self.lock:
            self.latest_obs = dict(obs)
            self.is_intervening = is_intervening

    def get_latest_observation(self) -> dict | None:
        with self.lock:
            return dict(self.latest_obs) if self.latest_obs else None

    def set_intervention(self, status: bool) -> None:
        with self.lock:
            self.is_intervening = status

    # -- episode / reset ------------------------------------------------------

    def set_episode_active(self, status: bool) -> None:
        with self.lock:
            self.episode_active = status

    def request_reset(self) -> None:
        with self.lock:
            self.policy_reset_requested = True

    def check_and_clear_reset(self) -> bool:
        with self.lock:
            if self.policy_reset_requested:
                self.policy_reset_requested = False
                return True
            return False

    # -- weight update --------------------------------------------------------

    def request_parameter_update(self) -> None:
        with self.lock:
            self.update_parameters_requested = True
        self.params_loaded_event.clear()

    def check_and_clear_parameter_update(self) -> bool:
        with self.lock:
            if self.update_parameters_requested:
                self.update_parameters_requested = False
                return True
            return False

    # -- action chunk ---------------------------------------------------------

    def put_chunk(self, actions: list[torch.Tensor]) -> None:
        with self.lock:
            self._action_deque = deque(actions)
        self.chunk_ready_event.set()

    def pop_action(self) -> torch.Tensor | None:
        with self.lock:
            return self._action_deque.popleft() if self._action_deque else None

    def chunk_remaining(self) -> int:
        with self.lock:
            return len(self._action_deque)

    # -- metrics --------------------------------------------------------------

    def get_and_reset_metrics(self) -> dict:
        with self.lock:
            m = {
                "env_wait_time": self.env_wait_time,
                "env_steps": self.env_steps,
                "env_active_time": self.env_active_time_total,
                "inference_latencies": list(self.inference_latencies),
            }
            self.env_wait_time = 0.0
            self.env_steps = 0
            self.env_active_time_total = 0.0
            self.inference_latencies = []
        return m


# ── Weight loader ─────────────────────────────────────────────────────────────


def _pull_weights(policy: nn.Module, parameters_queue, device: torch.device) -> None:
    """Load the most recent weight packet from the learner queue, if any."""
    blob = get_last_item_from_queue(parameters_queue, block=False)
    if blob is None:
        return
    logger.info("[INFERENCE] Loading new parameters from Learner.")
    sd = move_state_dict_to_device(bytes_to_state_dict(blob)["policy"], device=device)
    policy.load_state_dict(sd, strict=False)
    logger.info("[INFERENCE] Parameters loaded.")


# ── Inference thread ──────────────────────────────────────────────────────────


def inference_worker(
    policy: nn.Module,
    trainer: Trainer,
    preprocessor,
    postprocessor,
    shared_state: SharedState,
    parameters_queue,
    device: torch.device,
    cfg,
) -> None:
    """
    Background inference thread.

    Waits for env to request a chunk, runs model inference via
    trainer.build_inference_batch + policy.select_action, fills the chunk
    deque, then signals env.  Weight updates are applied between chunks.

    trainer.build_inference_batch() is the model-agnostic isolation point:
      - PI05:      adds subtask tokens, advantage injection, etc.
      - MolmoAct2: just calls the preprocessor.
    """
    try:
        logger.info("[INFERENCE] Thread started.")
        task_str: str = cfg.policy.task
        chunk_size: int = int(getattr(cfg.policy, "chunk_size", 30))
        action_dim: int = int(next(iter(cfg.policy.output_features.values())).shape[0])

        while shared_state.running:
            # Apply pending weight update between chunks.
            if shared_state.check_and_clear_parameter_update():
                _pull_weights(policy, parameters_queue, device)
                shared_state.params_loaded_event.set()

            # Wait for env to signal it needs a new chunk.
            if not shared_state.chunk_request_event.wait(timeout=0.05):
                continue
            shared_state.chunk_request_event.clear()

            if not shared_state.episode_active or not shared_state.running:
                continue

            obs = shared_state.get_latest_observation()
            if obs is None:
                shared_state.chunk_ready_event.set()  # unblock env even on empty obs
                continue

            obs_filtered = {k: v for k, v in obs.items() if k in cfg.policy.input_features}

            # Model-agnostic batch construction: PI05 adds subtask/advantage here,
            # MolmoAct2 just preprocesses images + state.
            batch = trainer.build_inference_batch(
                obs_filtered, task_str, cfg, preprocessor=preprocessor
            )

            t0 = time.perf_counter()
            with torch.no_grad():
                # select_action caches the chunk internally on first call;
                # subsequent calls pop from the cache without re-running inference.
                actions: list[torch.Tensor] = []
                for _ in range(chunk_size):
                    act = policy.select_action(batch)
                    if postprocessor is not None:
                        act = postprocessor(act)
                    actions.append(act[..., :action_dim].cpu())

            dt = time.perf_counter() - t0
            with shared_state.lock:
                shared_state.inference_latencies.append(dt)
            logger.debug(f"[INFERENCE] Chunk in {dt:.3f}s")

            shared_state.put_chunk(actions)

        logger.info("[INFERENCE] Thread shut down.")
    except Exception:
        logger.error(f"[INFERENCE] Fatal:\n{traceback.format_exc()}")


# ── Env interaction thread ────────────────────────────────────────────────────


def env_worker(
    online_env,
    env_processor,
    action_processor,
    shared_state: SharedState,
    teleop_device,
    transitions_queue,
    interactions_queue,
    cfg,
) -> None:
    """
    Main environment thread.

    Pops actions from the chunk deque (requesting new chunks from the inference
    thread as needed), steps the env at the target FPS, collects transitions,
    and sends completed episodes to the learner.
    """
    try:
        logger.info("[ENV] Thread started.")
        action_interval = 1.0 / cfg.env.fps
        action_dim = int(next(iter(cfg.policy.output_features.values())).shape[0])

        sum_reward = 0.0
        n_intervention_steps = 0
        n_total_steps = 0
        list_transitions: list[Transition] = []
        interaction_step = 0

        def _wait_for_teleop_start() -> None:
            logger.info("[ENV] Waiting for '2' on teleop to start episode…")
            while shared_state.running:
                if teleop_device.get_teleop_events().get(TeleopEvents.START_EPISODE, False):
                    return
                time.sleep(0.1)

        def _reset_env():
            obs, info = online_env.reset()
            env_processor.reset()
            action_processor.reset()
            tr = create_transition(observation=obs, info=info)
            tr[TransitionKey.COMPLEMENTARY_DATA] = {
                "subtask": [""] * (len(obs) if isinstance(obs, list) else 1)
            }
            return env_processor(tr)

        def _request_chunk() -> None:
            shared_state.chunk_ready_event.clear()
            shared_state.chunk_request_event.set()
            while shared_state.running:
                if shared_state.chunk_ready_event.wait(timeout=0.1):
                    break

        # ── initial setup ────────────────────────────────────────────────────
        _wait_for_teleop_start()
        transition = _reset_env()
        policy_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
        shared_state.update_observation(policy_obs, False)
        shared_state.request_parameter_update()

        logger.info("[ENV] Waiting for initial weights from Learner…")
        while shared_state.running:
            if shared_state.params_loaded_event.wait(timeout=0.5):
                break
        logger.info("[ENV] Weights loaded. Starting first episode.")

        _request_chunk()
        shared_state.set_episode_active(True)

        # ── main loop ────────────────────────────────────────────────────────
        was_intervening = False

        while shared_state.running:
            # Episode boundary.
            if not shared_state.episode_active:
                logger.info("[ENV] Episode ended. Waiting for teleop '2'…")
                _wait_for_teleop_start()
                if not shared_state.running:
                    break

                transition = _reset_env()
                shared_state.request_reset()
                shared_state.request_parameter_update()
                was_intervening = False
                sum_reward = 0.0
                n_intervention_steps = 0
                n_total_steps = 0

                policy_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
                shared_state.update_observation(policy_obs, False)

                logger.info("[ENV] Waiting for updated weights from Learner…")
                while shared_state.running:
                    if shared_state.params_loaded_event.wait(timeout=0.5):
                        break

                _request_chunk()
                shared_state.set_episode_active(True)

            step_t0 = time.perf_counter()

            # Teleop state change → policy reset.
            is_intervening = shared_state.is_intervening
            if was_intervening != is_intervening:
                logger.info(f"[ENV] Teleop changed (intervening={is_intervening}), requesting reset.")
                shared_state.request_reset()
            was_intervening = is_intervening

            # Get action.
            if is_intervening:
                if hasattr(online_env, "get_raw_joint_positions"):
                    joints = online_env.get_raw_joint_positions()
                    joint_order = [
                        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
                    ]
                    action = torch.tensor(
                        [float(joints.get(k, 0.0)) for k in joint_order],
                        dtype=torch.float32, device=cfg.policy.device,
                    )
                else:
                    action = torch.zeros(action_dim, dtype=torch.float32, device=cfg.policy.device)
            else:
                if shared_state.chunk_remaining() == 0:
                    _request_chunk()
                action = shared_state.pop_action()
                if action is None:
                    action = torch.zeros(action_dim, dtype=torch.float32, device=cfg.policy.device)
                action = action.to(cfg.policy.device)

            # Step env.
            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA].get(
                "teleop_action", action
            )
            reward = float(new_transition[TransitionKey.REWARD])
            done = bool(new_transition.get(TransitionKey.DONE, False))
            truncated = bool(new_transition.get(TransitionKey.TRUNCATED, False))

            sum_reward += reward
            n_total_steps += 1
            if reward > 0:
                logger.info(f"[ENV] Positive reward: {reward:.3f}")

            info = new_transition[TransitionKey.INFO]
            is_intervening = bool(info.get(TeleopEvents.IS_INTERVENTION, False))
            shared_state.set_intervention(is_intervening)

            if is_intervening:
                n_intervention_steps += 1
            else:
                feedback = {
                    k: (v.item() if isinstance(v, torch.Tensor) else float(v))
                    for k, v in new_transition[TransitionKey.OBSERVATION].items()
                    if k.endswith(".pos")
                }
                if feedback:
                    teleop_device.send_feedback(feedback)

            # Build and buffer transition.
            obs_dict = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
            next_obs_dict = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])
            complementary_info = {
                TeleopEvents.IS_INTERVENTION.value: torch.tensor(
                    [float(is_intervening)], dtype=torch.float32
                ),
            }
            t_send = Transition(
                state=obs_dict,
                action=executed_action[..., :action_dim],
                reward=reward,
                next_state=next_obs_dict,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
            list_transitions.append(move_transition_to_device(t_send, "cpu"))

            next_policy_obs = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])
            shared_state.update_observation(next_policy_obs, is_intervening)
            transition = new_transition
            interaction_step += 1

            # Episode end.
            if done or truncated:
                logger.info(
                    f"[ENV] Episode ended — step={interaction_step} "
                    f"reward={sum_reward:.3f} done={done} truncated={truncated}"
                )
                shared_state.set_episode_active(False)

                if list_transitions:
                    push_transitions_to_transport_queue(list_transitions, transitions_queue)
                    list_transitions = []

                iv_rate = n_intervention_steps / max(n_total_steps, 1)
                interactions_queue.put(
                    python_object_to_bytes({
                        "Episodic reward": sum_reward,
                        "Interaction step": interaction_step,
                        "Episode intervention": int(n_intervention_steps > 0),
                        "Intervention rate": iv_rate,
                    })
                )

                sum_reward = 0.0
                n_intervention_steps = 0
                n_total_steps = 0

            # FPS throttle.
            dt = time.perf_counter() - step_t0
            with shared_state.lock:
                shared_state.env_active_time_total += dt
                shared_state.env_steps += 1
            precise_sleep(max(0.0, action_interval - dt))

        logger.info("[ENV] Thread shut down.")
    except Exception:
        logger.error(f"[ENV] Fatal:\n{traceback.format_exc()}")


# ── Orchestration ─────────────────────────────────────────────────────────────


def act_with_policy_async(
    cfg,
    trainer: Trainer,
    shutdown_event,
    parameters_queue,
    transitions_queue,
    interactions_queue,
) -> None:
    actor_runtime = (
        getattr(cfg, "actor_runtime", None)
        or getattr(cfg.policy, "actor_runtime", None)
        or "rtc"
    )
    if actor_runtime == "rtc":
        from lerobot.rl.rtc_actor_runtime import act_with_policy_rtc

        act_with_policy_rtc(
            cfg=cfg,
            trainer=trainer,
            shutdown_event=shutdown_event,
            parameters_queue=parameters_queue,
            transitions_queue=transitions_queue,
            interactions_queue=interactions_queue,
        )
        return

    if actor_runtime not in {"simple", "chunk_deque"}:
        raise ValueError(f"Unknown actor_runtime={actor_runtime!r}; expected 'rtc' or 'simple'.")

    logger.warning("[ACTOR] Using simple chunk-deque runtime; RTC is the default runtime.")
    set_seed(cfg.seed)
    device_name = getattr(cfg.policy, "actor_device", None) or cfg.policy.device
    device = get_safe_torch_device(device_name, log=True)
    cfg.policy.device = str(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logger.info("[ACTOR] Building actor policy and processors…")
    policy = trainer.make_actor_policy(cfg).to(device).eval()
    preprocessor, postprocessor = trainer.make_processors(cfg)

    logger.info("[ACTOR] Setting up environment…")
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(
        online_env, teleop_device, cfg.env, cfg.policy.device
    )

    chunk_size = int(getattr(cfg.policy, "chunk_size", 30))
    shared = SharedState(chunk_size=chunk_size)
    shared.running = not shutdown_event.is_set()

    inf_thread = Thread(
        target=inference_worker,
        args=(policy, trainer, preprocessor, postprocessor, shared, parameters_queue, device, cfg),
        daemon=True,
        name="rl_inference",
    )
    env_thread = Thread(
        target=env_worker,
        args=(
            online_env, env_processor, action_processor, shared,
            teleop_device, transitions_queue, interactions_queue, cfg,
        ),
        daemon=True,
        name="rl_env",
    )

    try:
        env_thread.start()
        time.sleep(0.5)
        inf_thread.start()

        logger.info("[ACTOR] Supervisor loop active.")
        t_start = time.time()
        while not shutdown_event.is_set():
            time.sleep(10)
            m = shared.get_and_reset_metrics()
            n = max(1, m["env_steps"])
            lats = m["inference_latencies"]
            lat_str = (
                f"avg={sum(lats)/len(lats):.3f}s  max={max(lats):.3f}s" if lats else "N/A"
            )
            logger.info(
                f"[ACTOR] runtime={int(time.time()-t_start)}s  "
                f"env_steps={m['env_steps']}  "
                f"avg_env_active={m['env_active_time']/n:.4f}s  "
                f"chunk_latency={lat_str}"
            )
    except Exception:
        logger.error(f"[ACTOR] Orchestrator error:\n{traceback.format_exc()}")
    finally:
        shutdown_event.set()
        shared.running = False
        shared.chunk_request_event.set()  # unblock inference thread

        for t in (inf_thread, env_thread):
            t.join(timeout=5.0)
        try:
            online_env.close()
        except Exception:
            pass
        logger.info("[ACTOR] Shutdown complete.")


# ── CLI entry point ───────────────────────────────────────────────────────────


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    """Entry point for distributed online RL actor."""
    cfg.validate()

    log_dir = os.path.join(cfg.output_dir or "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    init_logging(log_file=os.path.join(log_dir, f"actor_{cfg.job_name}.log"), display_pid=False)
    logger.info("[ACTOR] Starting.")

    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

    trainer: Trainer = Trainer.for_config(cfg)

    # gRPC transport: receive weights from learner, send transitions and interactions.
    alc = cfg.policy.actor_learner_config
    learner_client, grpc_channel = learner_service_client(
        host=alc.learner_host, port=alc.learner_port
    )
    logger.info("[ACTOR] Establishing connection with Learner…")
    if not establish_learner_connection(learner_client, shutdown_event):
        logger.error("[ACTOR] Could not connect to Learner. Exiting.")
        return

    from torch.multiprocessing import Queue
    parameters_queue: Queue = Queue(maxsize=2)
    transitions_queue: Queue = Queue()
    interactions_queue: Queue = Queue()

    for target, name, args in [
        (receive_policy,    "recv_policy",   (cfg, parameters_queue,  shutdown_event, grpc_channel)),
        (send_transitions,  "send_trans",    (cfg, transitions_queue, shutdown_event, grpc_channel)),
        (send_interactions, "send_interact", (cfg, interactions_queue, shutdown_event, grpc_channel)),
    ]:
        Thread(target=target, args=args, daemon=True, name=name).start()

    try:
        act_with_policy_async(
            cfg=cfg,
            trainer=trainer,
            shutdown_event=shutdown_event,
            parameters_queue=parameters_queue,
            transitions_queue=transitions_queue,
            interactions_queue=interactions_queue,
        )
    finally:
        shutdown_event.set()
        for q in (parameters_queue, transitions_queue, interactions_queue):
            q.close()
            q.cancel_join_thread()
        logger.info("[ACTOR] actor_cli finished.")


if __name__ == "__main__":
    actor_cli()
