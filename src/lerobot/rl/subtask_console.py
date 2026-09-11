#!/usr/bin/env python
"""Operator-driven subtask selection for eval rollouts.

The subtask generation head is untrained (``subtask_loss_weight`` is 0), so at eval
the current step is fed live by the operator instead of decoded. ``cfg.policy.
eval_subtasks`` is the rollout's script: the steps in order, free text. The console
walks it with two global keys, ``n`` (next) and ``b`` (back), and the entry under the
cursor is latched into ``RTCSharedState``; the inference worker renders it as the
prompt's "The current step is ..." clause on its next cycle. An episode starts on
(and resets to) the first entry, the cursor clamps at both ends, and a different
order is a config edit.

The listener is a global hook (no terminal focus required) and fires on its own
thread, so neither the 30Hz executor nor the inference thread polls for input.

Entries are free text: one that matches a checkpoint vocabulary string logs its index
on the buffer's canonical ``subtask_index`` column; anything else logs -1, the same
value the generation path uses when a decoded step misses the vocab. Misses are
listed at startup.
"""
import logging

logger = logging.getLogger(__name__)

NEXT_KEY = "n"
BACK_KEY = "b"


class SubtaskConsole:
    """Walks ``shared_state.subtask_script`` with n (next) / b (back)."""

    def __init__(self, script: list[str], vocabulary: list[str], shared_state) -> None:
        self.shared_state = shared_state
        self.listener = None
        indexed = [(text, vocabulary.index(text) if text in vocabulary else -1) for text in script]
        for text, index in indexed:
            if index < 0:
                logger.info("[SUBTASK] %r is not in the checkpoint vocabulary; logs subtask_index -1.", text)
        shared_state.set_subtask_script(indexed)

    def start(self) -> None:
        from pynput import keyboard

        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        steps = "\n".join(f"  {i + 1}. {text}" for i, (text, _) in enumerate(self.shared_state.subtask_script))
        logger.info(
            "[SUBTASK] Operator console active — generation disabled. [%s] next, [%s] back.\n%s",
            NEXT_KEY,
            BACK_KEY,
            steps,
        )

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def _on_press(self, key) -> None:
        char = getattr(key, "char", None)
        if char == NEXT_KEY:
            self._move(1)
        elif char == BACK_KEY:
            self._move(-1)

    def _move(self, delta: int) -> None:
        cursor = self.shared_state.advance_subtask(delta)
        script = self.shared_state.subtask_script
        following = script[cursor + 1][0] if cursor + 1 < len(script) else "end of script"
        logger.info("[SUBTASK] [%d/%d] %s   (next: %s)", cursor + 1, len(script), script[cursor][0], following)


def make_subtask_console(cfg, trainer, preprocessor, shared_state) -> SubtaskConsole | None:
    """Build the console when ``cfg.policy.eval_subtasks`` is set, else None."""
    script = getattr(cfg.policy, "eval_subtasks", None)
    if not script:
        return None
    return SubtaskConsole(script, trainer.subtask_vocabulary(preprocessor), shared_state)
