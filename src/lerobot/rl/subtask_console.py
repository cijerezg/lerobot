#!/usr/bin/env python
"""Operator-driven subtask selection for eval rollouts.

The subtask generation head is untrained (``subtask_loss_weight`` is 0), so at eval
the current step is fed live by the operator instead of decoded. ``cfg.policy.
eval_subtasks`` is the rollout's script: the steps in order, free text. The console
walks it with two global keys, ``n`` (next) and ``b`` (back), and the entry under the
cursor is latched into ``RTCSharedState``; the inference worker renders it as the
prompt's "The current step is ..." clause on its next cycle. An episode starts on
(and resets to) the first entry, the cursor clamps at both ends, and a different
order is a config edit. ``r`` latches ``cfg.policy.eval_home_subtask`` over the
current entry at any point without moving the cursor, so the next ``n``/``b``
continues the script from where it was.

Each entry can also carry prompt metadata, ``cfg.policy.eval_subtask_precisions`` (1-5)
and ``eval_subtask_contacts`` (contact_vocab slugs), aligned with the script. The step's
precision and contact are latched with it, so n/b/r switch "The precision is N of 5."
and "The contact is ..." together with the step (r uses ``eval_home_precision`` /
``eval_home_contact``). A list left unset keeps that channel off: no clause at all.

The listener is a global hook (no terminal focus required) and fires on its own
thread, so neither the 30Hz executor nor the inference thread polls for input.

Entries are free text: one that matches a checkpoint vocabulary string logs its index
on the buffer's canonical ``subtask_index`` column; anything else logs -1, the same
value the generation path uses when a decoded step misses the vocab. Misses are
listed at startup.
"""
import logging

from lerobot.datasets.contact_vocab import code_for, phrase_for

logger = logging.getLogger(__name__)

NEXT_KEY = "n"
BACK_KEY = "b"
HOME_KEY = "r"


class SubtaskConsole:
    """Walks ``shared_state.subtask_script`` with n (next) / b (back); r overrides with home."""

    def __init__(
        self,
        script: list[str],
        home: str,
        vocabulary: list[str],
        shared_state,
        metadata: list[dict] | None = None,
        home_metadata: dict | None = None,
    ) -> None:
        """``metadata[i]`` is step i's prompt metadata ({"precision": int, "contact": code}),
        ``home_metadata`` the home step's; None = no per-step metadata."""
        self.shared_state = shared_state
        self.listener = None
        metadata = list(metadata) if metadata is not None else [{} for _ in script]
        indexed = [
            (text, vocabulary.index(text) if text in vocabulary else -1, dict(meta))
            for text, meta in zip(script + [home], metadata + [home_metadata or {}], strict=True)
        ]
        for text, index, _ in indexed:
            if index < 0:
                logger.info("[SUBTASK] %r is not in the checkpoint vocabulary; logs subtask_index -1.", text)
        self.home = indexed[-1]
        shared_state.set_subtask_script(indexed[:-1])

    def start(self) -> None:
        from pynput import keyboard

        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        steps = "\n".join(
            f"  {i + 1}. {_describe(entry)}" for i, entry in enumerate(self.shared_state.subtask_script)
        )
        logger.info(
            "[SUBTASK] Operator console active — generation disabled. [%s] next, [%s] back, [%s] %s.\n%s",
            NEXT_KEY,
            BACK_KEY,
            HOME_KEY,
            _describe(self.home),
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
        elif char == HOME_KEY:
            self.shared_state.update_subtask(*self.home)
            cursor = self.shared_state.subtask_cursor
            logger.info("[SUBTASK] %s   (script at %d/%d, n/b resume from there)", _describe(self.home), cursor + 1, len(self.shared_state.subtask_script))

    def _move(self, delta: int) -> None:
        cursor = self.shared_state.advance_subtask(delta)
        script = self.shared_state.subtask_script
        following = script[cursor + 1][0] if cursor + 1 < len(script) else "end of script"
        logger.info("[SUBTASK] [%d/%d] %s   (next: %s)", cursor + 1, len(script), _describe(script[cursor]), following)


def _describe(entry: tuple) -> str:
    """Step text plus whatever metadata it latches, for the operator log."""
    meta = entry[2] if len(entry) > 2 else {}
    parts = []
    if "precision" in meta:
        parts.append(f"precision {meta['precision']}")
    if "contact" in meta:
        parts.append(f"contact: {phrase_for(meta['contact'])}")
    return f"{entry[0]} [{', '.join(parts)}]" if parts else entry[0]


def script_metadata(policy_cfg) -> tuple[list[dict], dict]:
    """Per-step and home prompt metadata from the policy config. Only channels whose
    list is set contribute a key (precision int, contact code); the config has
    already checked lengths and values."""
    script = getattr(policy_cfg, "eval_subtasks", None) or []
    precisions = getattr(policy_cfg, "eval_subtask_precisions", None)
    contacts = getattr(policy_cfg, "eval_subtask_contacts", None)
    steps: list[dict] = [{} for _ in script]
    home: dict = {}
    if precisions is not None:
        for meta, precision in zip(steps, precisions, strict=True):
            meta["precision"] = int(precision)
        home["precision"] = int(policy_cfg.eval_home_precision)
    if contacts is not None:
        for meta, slug in zip(steps, contacts, strict=True):
            meta["contact"] = code_for(slug)
        home["contact"] = code_for(policy_cfg.eval_home_contact)
    return steps, home


def make_subtask_console(cfg, trainer, preprocessor, shared_state) -> SubtaskConsole | None:
    """Build the console when ``cfg.policy.eval_subtasks`` is set, else None."""
    script = getattr(cfg.policy, "eval_subtasks", None)
    if not script:
        return None
    metadata, home_metadata = script_metadata(cfg.policy)
    return SubtaskConsole(
        script,
        cfg.policy.eval_home_subtask,
        trainer.subtask_vocabulary(preprocessor),
        shared_state,
        metadata=metadata,
        home_metadata=home_metadata,
    )
