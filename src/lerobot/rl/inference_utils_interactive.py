"""
Interactive inference utilities: shared state with subtask override, input thread,
and an inference worker that injects user-supplied subtask tokens.

Imports everything possible from inference_utils.py — only the delta is defined here.
"""

import logging
import math
import threading
import time
import traceback

import torch

from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.rl.inference_utils import SharedState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state extension
# ---------------------------------------------------------------------------

class SharedStateInteractive(SharedState):
    """SharedState + a pending subtask override slot."""

    def __init__(self):
        super().__init__()
        self._pending_tokens: torch.Tensor | None = None  # [max_len] long
        self._pending_masks: torch.Tensor | None = None   # [max_len] bool
        self._pending_text: str | None = None

    def set_pending_override(self, tokens: torch.Tensor, masks: torch.Tensor, text: str):
        """Called by the input thread."""
        with self.lock:
            self._pending_tokens = tokens
            self._pending_masks = masks
            self._pending_text = text

    def pop_pending_override(self):
        """Called by the inference worker. Returns (tokens, masks, text) and clears, or None."""
        with self.lock:
            if self._pending_tokens is None:
                return None
            result = (self._pending_tokens, self._pending_masks, self._pending_text)
            self._pending_tokens = None
            self._pending_masks = None
            self._pending_text = None
            return result

    def clear_pending_override(self):
        """Called at episode boundary so a stale queued override doesn't survive a reset."""
        with self.lock:
            self._pending_tokens = None
            self._pending_masks = None
            self._pending_text = None


# ---------------------------------------------------------------------------
# Input thread
# ---------------------------------------------------------------------------

def terminal_input_worker(
    shared_state: SharedStateInteractive,
    policy,
    cfg,
    shutdown_event: threading.Event,
):
    """
    Daemon thread. Blocks on input(), tokenizes the text, and queues the override.
    Log output from other threads will interleave with the prompt — that is expected.
    """
    tokenizer = policy.model._paligemma_tokenizer
    max_len = cfg.policy.tokenizer_max_length

    interval = getattr(cfg.policy, "subtask_regeneration_interval", -1)
    if interval <= 0:
        logger.warning(
            "[INTERACTIVE] subtask_regeneration_interval is <= 0. "
            "Injected subtasks will be IMMEDIATELY overwritten by generate_subtask_tokens "
            "on the very next inference cycle. Set it to a positive value (e.g. 30) in your config."
        )

    print("[INTERACTIVE] Type a subtask and press Enter at any time. Ctrl+C to stop.")

    while not shutdown_event.is_set():
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            continue

        encoding = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = encoding["input_ids"][0]             # [max_len] long — EOS appended by tokenizer
        masks = encoding["attention_mask"][0].bool()  # [max_len] bool

        shared_state.set_pending_override(tokens, masks, text)
        print(f"[INTERACTIVE] Queued: '{text}' — will be used at next action generation")


# ---------------------------------------------------------------------------
# Inference worker with subtask injection
# ---------------------------------------------------------------------------

_HARDCODED_SUBTASK: str | None = "Subtask: reach and grasp cube;"


def get_actions_worker_interactive(
    policy,
    shared_state: SharedStateInteractive,
    action_queue,
    cfg,
):
    """
    Clone of get_actions_worker (inference_utils.py) with one addition:
    before each predict_action_chunk call, pop any pending subtask override
    and inject it directly into the policy's token cache.
    """
    try:
        logger.info("[GET_ACTIONS] Starting interactive inference thread")
        latency_tracker = LatencyTracker()
        last_subtask_text = None
        inference_step = 0

        execution_horizon = policy.config.rtc_config.execution_horizon
        time_per_chunk = 1.0 / cfg.env.fps

        task_str = cfg.policy.task
        advantage_val = torch.tensor(
            [[cfg.policy.inference_advantage]], device=torch.device("cpu"), dtype=torch.float32
        )

        # Derive device from policy weights — no device arg in the original worker signature
        device = next(policy.parameters()).device

        hardcoded_tokens: torch.Tensor | None = None
        hardcoded_masks: torch.Tensor | None = None
        if _HARDCODED_SUBTASK is not None:
            tokenizer = policy.model._paligemma_tokenizer
            max_len = cfg.policy.tokenizer_max_length
            enc = tokenizer(
                _HARDCODED_SUBTASK,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            hardcoded_tokens = enc["input_ids"].to(device)        # [1, max_len]
            hardcoded_masks = enc["attention_mask"].bool().to(device)
            logger.warning(
                f"[INTERACTIVE] HARDCODED SUBTASK ENABLED: '{_HARDCODED_SUBTASK}' "
                "— will be re-injected before every chunk, overriding typed input and "
                "the model's auto-generation. Set _HARDCODED_SUBTASK = None to disable."
            )

        while shared_state.running:
            # 1. Reset check
            if shared_state.check_and_clear_reset():
                policy.reset()
                shared_state.clear_pending_override()  # don't carry stale override across episodes
                last_subtask_text = None
                continue

            if not shared_state.episode_active:
                time.sleep(0.01)
                continue

            # 2. Queue saturation check
            current_delay = math.ceil(latency_tracker.p95() / time_per_chunk)
            if not action_queue.empty() and action_queue.qsize() > execution_horizon + current_delay:
                wait_start = time.perf_counter()
                time.sleep(0.01)
                shared_state.add_inference_wait_time(time.perf_counter() - wait_start)
                continue

            # 3. Fetch latest observation
            latest_obs = shared_state.get_latest_observation()
            if latest_obs is None:
                time.sleep(0.01)
                continue

            # 4. Filter & format
            batch_for_preprocessor = {
                k: v for k, v in latest_obs.items() if k in cfg.policy.input_features
            }

            # 5. Complementary data
            batch_for_preprocessor["robot_type"] = (
                cfg.env.robot.type if hasattr(cfg.env, "robot") else ""
            )
            batch_for_preprocessor["complementary_data"] = {
                "task": [task_str],
                "subtask": [""],
                "advantage": advantage_val,
            }

            current_time = time.perf_counter()

            with torch.no_grad():
                if hasattr(policy, "preprocessor") and policy.preprocessor is not None:
                    processed_batch = policy.preprocessor(batch_for_preprocessor)
                else:
                    processed_batch = batch_for_preprocessor

                action_index_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                # --- Anchor/Delta alignment ---
                action_encoding = getattr(policy.config, "action_encoding", "absolute")
                anchor_now = None
                if action_encoding in ["anchor", "delta"]:
                    from lerobot.utils.constants import OBS_STATE
                    if OBS_STATE in latest_obs:
                        anchor_now = latest_obs[OBS_STATE]
                        if prev_actions is not None and action_queue.anchor_state is not None:
                            anchor_old = action_queue.anchor_state
                            from lerobot.processor import NormalizerProcessorStep
                            from lerobot.rl.actor_pi05_async_utils import align_prev_actions
                            normalizer = next(
                                s for s in policy.preprocessor.steps
                                if isinstance(s, NormalizerProcessorStep)
                            )
                            logger.debug(
                                f"[RTC] Alignment offset norm: {(anchor_old - anchor_now).norm().item():.4f}"
                            )
                            prev_actions = align_prev_actions(
                                prev_actions=prev_actions,
                                anchor_old=anchor_old,
                                anchor_now=anchor_now,
                                action_encoding=action_encoding,
                                chunk_size=policy.config.chunk_size,
                                postprocessor=policy.postprocessor,
                                normalizer=normalizer,
                            )

                # --- Subtask override injection ---
                override = shared_state.pop_pending_override()
                if override is not None:
                    tokens, masks, text = override
                    interval = getattr(policy.config, "subtask_regeneration_interval", -1)
                    if interval <= 0:
                        logger.warning(
                            f"[INTERACTIVE] Injecting subtask '{text}' but "
                            "subtask_regeneration_interval <= 0 — it will be overwritten immediately."
                        )
                    policy._cached_subtask_tokens = tokens.unsqueeze(0).to(device)  # [1, max_len]
                    policy._cached_subtask_masks = masks.unsqueeze(0).to(device)
                    policy._last_subtask_time = time.time()
                    logger.info(f"[INTERACTIVE] Injecting subtask override: '{text}'")
                # ----------------------------------

                # --- Hardcoded subtask injection (always wins, every cycle) ---
                if hardcoded_tokens is not None:
                    policy._cached_subtask_tokens = hardcoded_tokens
                    policy._cached_subtask_masks = hardcoded_masks
                    policy._last_subtask_time = time.time() + 1e9  # never expire
                # --------------------------------------------------------------

                inference_delay = math.ceil(latency_tracker.p95() / time_per_chunk)

                actions_chunk = policy.predict_action_chunk(
                    processed_batch,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=execution_horizon,
                )

                # --- Subtask token decoding (unchanged from inference_utils.py) ---
                inference_step += 1
                try:
                    cached_tokens = getattr(policy, "_cached_subtask_tokens", None)
                    cached_masks = getattr(policy, "_cached_subtask_masks", None)
                    if cached_tokens is not None and cached_masks is not None:
                        tokenizer = policy.model._paligemma_tokenizer
                        valid_tokens = cached_tokens[0][cached_masks[0]]
                        subtask_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                        with shared_state.lock:
                            shared_state.current_subtask_text = subtask_text
                        if subtask_text != last_subtask_text:
                            logger.info(
                                f"[SUBTASK] inference_step={inference_step} | "
                                f"subtask: \"{subtask_text}\""
                            )
                            last_subtask_text = subtask_text
                except Exception as e:
                    logger.debug(f"[SUBTASK] Could not decode subtask tokens: {e}")

                original_actions = actions_chunk.squeeze(0).clone()

                # --- Unnormalization & absolute action reconstruction ---
                unnormalized_actions = (
                    policy.postprocessor(original_actions)
                    if hasattr(policy, "postprocessor") and policy.postprocessor is not None
                    else original_actions.clone()
                )

                if anchor_now is not None and action_encoding in ["anchor", "delta"]:
                    anchor_sq = anchor_now.squeeze(0) if anchor_now.dim() > 1 else anchor_now
                    if action_encoding == "anchor":
                        processed_actions = (
                            unnormalized_actions + anchor_sq.to(unnormalized_actions.device)[None, :]
                        )
                    elif action_encoding == "delta":
                        processed_actions = (
                            torch.cumsum(unnormalized_actions, dim=0)
                            + anchor_sq.to(unnormalized_actions.device)[None, :]
                        )
                else:
                    processed_actions = unnormalized_actions

                # --- Centered moving average (window=5) ---
                if processed_actions.shape[0] >= 5:
                    padded = torch.cat(
                        [processed_actions[0:1]] * 2
                        + [processed_actions]
                        + [processed_actions[-1:]] * 2,
                        dim=0,
                    )
                    smoothed = (
                        padded[:-4] + padded[1:-3] + padded[2:-2] + padded[3:-1] + padded[4:]
                    ) / 5.0
                    processed_actions = smoothed

                if not hasattr(policy, "_chunk_plot_counter"):
                    policy._chunk_plot_counter = 0
                policy._chunk_plot_counter += 1

            # Track latency
            new_latency = time.perf_counter() - current_time
            new_delay = math.ceil(new_latency / time_per_chunk)
            latency_tracker.add(new_latency)

            current_index = action_queue.get_action_index()
            actions_consumed = max(0, current_index - action_index_before)
            effective_delay = min(new_delay, actions_consumed)

            action_queue.merge(
                original_actions=original_actions,
                processed_actions=processed_actions,
                real_delay=effective_delay,
                action_index_before_inference=action_index_before,
                anchor_state=(
                    anchor_now
                    if getattr(cfg.policy, "action_encoding", "absolute") in ["anchor", "delta"]
                    else None
                ),
            )

        logger.info("[GET_ACTIONS] Interactive inference thread shutting down smoothly.")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception: {e}")
        logger.error(traceback.format_exc())
