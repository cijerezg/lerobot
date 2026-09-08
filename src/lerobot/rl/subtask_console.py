#!/usr/bin/env python
"""Operator-driven subtask selection for eval rollouts.

The subtask generation head is untrained (``subtask_loss_weight`` is 0), so at eval
the current step is fed live by the operator instead of decoded. ``cfg.policy.
eval_subtasks`` binds keys on two axes: ``verbs`` maps a key to a step template and
``objects`` maps a key to an object name. A chord is a verb key followed by an object
key: the verb press is held pending, the object press renders the template with that
object (``{object}``, plus ``{container}`` from the optional ``containers`` map) and
latches the result into ``RTCSharedState``; the inference worker renders it as the
prompt's "The current step is ..." clause on its next cycle. A template without
placeholders ("return to home") needs no object and sends on the verb press alone.
An episode starts on (and resets to) the first verb rendered with the first object.

The listener is a global hook (no terminal focus required) and fires on its own
thread, so neither the 30Hz executor nor the inference thread polls for input.

Templates are free text: the operator may prompt any step, in or out of the
checkpoint's subtask vocabulary. A rendered chord that matches a vocabulary string
logs its index on the buffer's canonical ``subtask_index`` column; anything else
logs -1, the same value the generation path uses when a decoded step misses the
vocab. Every chord is rendered at startup and the misses are listed.
"""
import logging

logger = logging.getLogger(__name__)


class SubtaskConsole:
    """Latches verb + object chords into ``shared_state.current_subtask_*``."""

    def __init__(self, bindings: dict, vocabulary: list[str], shared_state) -> None:
        self.shared_state = shared_state
        self.vocabulary = vocabulary
        self.listener = None
        self.verbs: dict[str, str] = {str(key): text for key, text in bindings["verbs"].items()}
        self.objects: dict[str, str] = {str(key): name for key, name in bindings["objects"].items()}
        self.containers: dict[str, str] = dict(bindings.get("containers", {}))
        self.first_object = next(iter(self.objects.values()))
        self.pending_verb: str | None = None

        rendered = {
            self.compose(verb, name)[0] for verb in self.verbs.values() for name in self.objects.values()
        }
        for text in sorted(rendered - set(vocabulary)):
            logger.info("[SUBTASK] %r is not in the checkpoint vocabulary; logs subtask_index -1.", text)

    def compose(self, verb: str, name: str) -> tuple[str, int]:
        """Render a verb template with an object; index into the vocabulary or -1."""
        fields = {"object": name}
        if name in self.containers:
            fields["container"] = self.containers[name]
        text = verb.format(**fields)
        index = self.vocabulary.index(text) if text in self.vocabulary else -1
        return text, index

    @property
    def initial(self) -> tuple[str, int]:
        """First verb on the first object — what an episode starts on and resets to."""
        return self.compose(next(iter(self.verbs.values())), self.first_object)

    def start(self) -> None:
        from pynput import keyboard

        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        verbs = "\n".join(f"  [{key}] {text}" for key, text in self.verbs.items())
        objects = "\n".join(f"  [{key}] {name}" for key, name in self.objects.items())
        logger.info(
            "[SUBTASK] Operator console active — generation disabled. Press verb, then object.\n"
            "Verbs:\n%s\nObjects:\n%s",
            verbs,
            objects,
        )

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def _send(self, verb: str, name: str) -> None:
        text, index = self.compose(verb, name)
        self.shared_state.update_subtask(text, index)
        self.pending_verb = None
        logger.info("[SUBTASK] -> %s", text)

    def _on_press(self, key) -> None:
        char = getattr(key, "char", None)
        if char in self.verbs:
            self.pending_verb = self.verbs[char]
            if "{" not in self.pending_verb:
                self._send(self.pending_verb, self.first_object)
        elif char in self.objects and self.pending_verb is not None:
            self._send(self.pending_verb, self.objects[char])


def make_subtask_console(cfg, trainer, preprocessor, shared_state) -> SubtaskConsole | None:
    """Build the console when ``cfg.policy.eval_subtasks`` is set, else None."""
    bindings = getattr(cfg.policy, "eval_subtasks", None)
    if not bindings:
        return None
    return SubtaskConsole(bindings, trainer.subtask_vocabulary(preprocessor), shared_state)
