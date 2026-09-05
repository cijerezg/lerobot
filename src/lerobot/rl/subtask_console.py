#!/usr/bin/env python
"""Operator-driven subtask selection for eval rollouts.

The subtask generation head is untrained (``subtask_loss_weight`` is 0), so at eval
the current step is fed live by the operator instead of decoded. ``cfg.policy.
eval_subtasks`` maps a keyboard key to a subtask string; a global pynput listener
latches the last pressed key into ``RTCSharedState``, and the inference worker
renders it as the prompt's "The current step is ..." clause on its next cycle.

The listener is a global hook (no terminal focus required) and fires on its own
thread, so neither the 30Hz executor nor the inference thread polls for input.

Bindings are free text: the operator may prompt any step, in or out of the
checkpoint's subtask vocabulary. A binding that matches a vocabulary string logs
its index on the buffer's canonical ``subtask_index`` column; anything else logs
-1, the same value the generation path uses when a decoded step misses the vocab.
"""
import logging

logger = logging.getLogger(__name__)


class SubtaskConsole:
    """Latches operator key presses into ``shared_state.current_subtask_*``."""

    def __init__(self, bindings: dict, vocabulary: list[str], shared_state) -> None:
        self.shared_state = shared_state
        self.listener = None
        self.entries: dict[str, tuple[str, int]] = {}

        for key, text in bindings.items():
            index = vocabulary.index(text) if text in vocabulary else -1
            self.entries[str(key)] = (text, index)
            if index < 0:
                logger.info("[SUBTASK] %r is not in the checkpoint vocabulary; logs subtask_index -1.", text)

    @property
    def initial(self) -> tuple[str, int]:
        """First binding — what an episode starts on and resets to."""
        return next(iter(self.entries.values()))

    def start(self) -> None:
        from pynput import keyboard

        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        table = "\n".join(f"  [{key}] {name}" for key, (name, _) in self.entries.items())
        logger.info("[SUBTASK] Operator console active — generation disabled.\n%s", table)

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def _on_press(self, key) -> None:
        entry = self.entries.get(getattr(key, "char", None))
        if entry is not None:
            self.shared_state.update_subtask(*entry)
            logger.info("[SUBTASK] -> %s", entry[0])


def make_subtask_console(cfg, trainer, preprocessor, shared_state) -> SubtaskConsole | None:
    """Build the console when ``cfg.policy.eval_subtasks`` is set, else None."""
    bindings = getattr(cfg.policy, "eval_subtasks", None)
    if not bindings:
        return None
    return SubtaskConsole(bindings, trainer.subtask_vocabulary(preprocessor), shared_state)
