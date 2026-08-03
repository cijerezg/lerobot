# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-description for probes: the ``index.json`` the viewer renders.

A probe that calls :func:`write_index` at the end of its ``run()`` shows up in
``lerobot-view-probes`` with its documentation, its headline numbers, and a
caption per figure — without the viewer knowing the probe exists. A probe that
doesn't still shows up, but only as a pile of PNGs (see ``fallback_index``).

The prose is not duplicated here. ``doc`` comes from the probe module's own
docstring, so the explanation lives next to the code that produces the numbers
and cannot drift from it.

Declaring a metric means declaring three things the reader would otherwise have
to know already: what it is called in plain words, which direction is good, and
what value is bad enough to worry about. That last one is what turns a wall of
floats into a status dot.

Prose conventions, applying to docstrings, captions, ``how`` and ``note``:

* ``$x$`` and ``$$x$$`` render as math (a LaTeX subset: fractions, scripts,
  roots, Greek, the usual relations). Write the algebra, don't describe it.
* Double backticks mark code, ``**bold**`` and ``*italic*`` work, and a blank
  line starts a paragraph.
* Mark two or three panels ``primary=True``. Everything else is fan-out the
  reader opens deliberately, not on arrival.
"""

from __future__ import annotations

import ast
import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1

# Status ladder, worst first. A probe's status is the worst of its metrics'.
STATUS_ORDER = ("bad", "warn", "good", "info")

_KIND_BY_SUFFIX = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".svg": "image",
    ".html": "html", ".htm": "html",
    ".mp4": "video", ".webm": "video",
    ".json": "data", ".csv": "data",
}


def panel_kind(filename: str) -> str:
    return _KIND_BY_SUFFIX.get(os.path.splitext(filename)[1].lower(), "file")


@dataclass
class Metric:
    """One headline number, with the reader's questions answered up front.

    Args:
        key: where to find it in the probe's summary dict. Dotted for nested
            values, e.g. ``"mse_norm.rgb_only"``.
        label: plain-words name shown in the viewer.
        good: ``"high"``, ``"low"``, or ``"none"`` when the value is descriptive
            rather than better-or-worse (a layer index, a frame count).
        fmt: decimal places.
        note: a caveat that travels with the number — "val v2-1 has no flagged
            mistakes, so this split is empty" is worth more than the number.
        warn/bad: thresholds, read in the direction ``good`` points. With
            ``good="low"``, a value above ``warn`` is a warning and above ``bad``
            is a failure; with ``good="high"``, below. Leave unset when you
            genuinely don't know where the line is — an invented threshold is
            worse than no dot.
        value: set directly when the number isn't in the summary dict.
        baseline: the value that means "no effect" (a noise floor, a uniform
            expectation), drawn as a reference on the sparkline.
        primary: show this metric in the compact headline strip above the figure.
            Every metric still appears in the detailed table. If a legacy manifest
            declares no primary metrics, the viewer shows its first four.
    """

    key: str
    label: str
    good: str = "none"
    fmt: int = 3
    note: str = ""
    warn: float | None = None
    bad: float | None = None
    value: float | None = None
    baseline: float | None = None
    refs: list[str] = field(default_factory=list)
    primary: bool = False

    def status(self, value: float | None) -> str:
        if value is None or self.good == "none" or (self.warn is None and self.bad is None):
            return "info"
        worse = (lambda a, b: a > b) if self.good == "low" else (lambda a, b: a < b)
        if self.bad is not None and worse(value, self.bad):
            return "bad"
        if self.warn is not None and worse(value, self.warn):
            return "warn"
        return "good"


@dataclass
class Panel:
    """One figure, plus how to read it.

    Args:
        how: the sentence you would say standing over someone's shoulder — what
            the colours mean, or which shape in the plot is the bad one. Supports
            ``$math$`` and ``$$display math$$``.
        primary: is this one of the few figures worth looking at every time? The
            viewer shows primary figures by default and hides the rest behind a
            toggle. Most probes should mark two or three; per-episode and
            per-layer fan-out is almost never primary.
        refs: ids of other probes that qualify this one — the noise floor that
            makes this number meaningful, the ablation that contradicts it.
            Rendered as links the reader can follow.
    """

    file: str
    caption: str
    how: str = ""
    kind: str = ""
    primary: bool = False
    refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "file": self.file,
            "caption": self.caption,
            "how": self.how,
            "kind": self.kind or panel_kind(self.file),
            "primary": self.primary,
            "refs": list(self.refs),
        }


def module_doc(module: Any) -> str:
    """The probe module's docstring, dedented, blank-line paragraphs preserved."""
    doc = getattr(module, "__doc__", None) or ""
    return textwrap.dedent(doc).strip()


def module_doc_from_source(path: str) -> str:
    """Same, but parsed from a file — no import, so the viewer stays torch-free."""
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return ""
    return textwrap.dedent(ast.get_docstring(tree) or "").strip()


def dig(summary: dict, key: str) -> Any:
    """Look up ``key`` in ``summary``, following dots into nested dicts."""
    node: Any = summary
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def write_index(
    output_dir: str,
    module: Any,
    *,
    claim: str,
    group: str,
    metrics: list[Metric],
    panels: list[Panel],
    summary: dict | None = None,
    title: str | None = None,
    status: str | None = None,
    see_also: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    """Write ``<output_dir>/index.json`` describing what this probe just produced.

    Args:
        module: the probe module itself (pass ``sys.modules[__name__]``); its
            docstring becomes the probe's documentation in the viewer.
        claim: one sentence, phrased as the question the probe answers.
        group: sidebar grouping — see ``GROUPS``.
        summary: the probe's summary dict, from which metric values are read by
            ``Metric.key``.
        status: override the status derived from the metrics' thresholds.

    Panels whose files don't exist are dropped, so a probe can declare its full
    set and stay correct when a conditional figure wasn't drawn.
    """
    summary = summary or {}
    probe_id = os.path.basename(os.path.normpath(output_dir))

    metric_rows = []
    for metric in metrics:
        value = metric.value if metric.value is not None else _numeric(dig(summary, metric.key))
        metric_rows.append(
            {
                "key": metric.key,
                "label": metric.label,
                "value": value,
                "good": metric.good,
                "fmt": metric.fmt,
                "note": metric.note,
                "baseline": metric.baseline,
                "warn": metric.warn,
                "bad": metric.bad,
                "refs": list(metric.refs),
                "primary": metric.primary,
                "status": metric.status(value),
            }
        )

    kept = [p for p in panels if os.path.exists(os.path.join(output_dir, p.file))]
    statuses = [row["status"] for row in metric_rows]
    derived = next((s for s in STATUS_ORDER if s in statuses), "info")

    index = {
        "schema": SCHEMA_VERSION,
        "id": probe_id,
        "title": title or probe_id.replace("_", " ").title(),
        "group": group,
        "claim": claim,
        "doc": module_doc(module),
        "status": status or derived,
        "metrics": metric_rows,
        "panels": [p.as_dict() for p in kept],
        "see_also": list(see_also or []),
        "extra": extra or {},
    }
    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    return index


GROUPS = (
    "Actions",
    "Attention",
    "Sensitivity",
    "Representation",
    "Critic",
    "History",
    "Steering",
    "Depth",
    "Other",
)

# Fallback identity for probes that don't write an index.json yet: output subdir
# -> (title, group, module filename). Only the module filename is not derivable,
# and only where it differs from the subdir. This table shrinks to nothing as
# probes adopt write_index().
PROBE_META = {
    "offline_inference":                  ("Offline Inference", "Actions", "offline_inference.py"),
    "action_trace":                       ("Action Inspector", "Actions", "action_trace_probe.py"),
    "attention":                          ("Attention Maps", "Attention", "attention.py"),
    "attention_budget":                   ("Attention Budget", "Attention", "attention_budget.py"),
    "spatial_memorization_attention":     ("Spatial Memorization (Attention)", "Attention", "spatial_memorization_attention.py"),
    "action_drift_jacobian":              ("Subtask Action Sensitivity", "Sensitivity", "action_drift_jacobian.py"),
    "spatial_memorization_action_jacobian": ("Spatial Memorization (Jacobian)", "Sensitivity", "spatial_memorization_action_jacobian.py"),
    "representations":                    ("Representations", "Representation", "representations.py"),
    "critic":                             ("Critic Values", "Critic", "critic.py"),
    "mem_history_influence":              ("MEM History Influence", "History", "mem_history_influence.py"),
    "mem_temporal_attention":             ("MEM Temporal Attention", "History", "mem_temporal_attention.py"),
    "metadata_steering":                  ("Metadata Steering", "Steering", "metadata_steering.py"),
    "subtask_sweep":                      ("Subtask Sweep", "Steering", "subtask_sweep.py"),
    "depth_modality":                     ("Depth Modality", "Depth", "depth_modality_probe.py"),
}


# The conventions every probe shares and none of them can explain on its own.
# Rendered as the viewer's "Reading this suite" page.
SUITE_DOC = """
## What this suite is

The configured probes run against a checkpoint on held-out episodes. Offline
Inference measures whether the policy is any good. The remaining probes exist
to explain why, by asking whether each thing the model was given (the
subtask clause, the metadata clause, the history, the depth stream, the language)
actually reaches the actions.

Read them in that order: the headline number first, then the probe that explains
the part you doubt.

**Everything the model produces is an action.** The policy generates no text: the
subtask is a clause the prompt carries in, and the long-term summary memory was
dropped (2026-08-01). So no probe here scores a decode, and "does the subtask
matter" is answered the only way left — by varying the input and measuring the
actions (``subtask_sweep``, ``metadata_steering``).

## Conventions that apply to every probe

**The deployment prompt.** Probes build the prompt a rollout actually issues —
task, subtask, metadata, history, depth — rather than a stripped-down one.
Omitting a clause is a measurement choice a probe makes deliberately (that is
exactly what an ablation is), never a default. A number measured under a prompt
the robot never sees does not describe the robot.

**Fixed-seed flow noise.** Any probe comparing conditions (with clause vs
without, RGB vs RGB+depth) re-runs the same frame with the same seeded
generator, so the only difference between the two runs is the condition. Without
this, every delta is confounded by the sampler, and small effects are
indistinguishable from noise. Where the spread across draws *is* the measurement — the Action Inspector fan —
each draw gets its own deterministic seed. The fan remains independent while sample 0
can be matched exactly to Offline Inference and every checkpoint sees the same noise.

**The noise floor.** A conditioning effect only counts if it exceeds what
re-running the same condition under different noise would produce anyway.
Subtask Sweep measures that floor explicitly and reports the ratio; treat an
effect near 1.0 as absent no matter how large it looks in absolute terms.

**Normalised action space.** Action deltas and MSEs are reported in normalised
model space, not degrees, so joints with different ranges are comparable.
Absolute magnitudes are only meaningful against another number from the same
probe.

**Dropout is suppressed during capture.** Training-time prompt and modality
dropout is neutralised while a probe runs. Attention capture drives a flow loss,
which is exactly what switches those draws on, so without suppression a probe
would be measuring a randomly ablated model.

**Stride-snapped frames.** Sampled frames land on the image-stride grid so the
history and the frame cache line up with what training and rollout see.

**Adapters, not internals.** Probes only call the policy through the
``ProbablePolicy`` adapter, which is why the same probe code runs against
different policies. A probe that needs a new capability adds a method there.

## How to read a status dot

Green, amber and red come from thresholds the probe itself declares next to the
metric. Grey means no threshold was declared — the number is descriptive, or
nobody has decided yet where the line is. Grey is not "fine".

## What this suite cannot tell you

Every number here is open-loop: the model predicts a chunk, and the chunk is
compared against what the operator did. Nothing closes the loop, so nothing here
measures recovery, compounding error, or anything that only appears once the
policy's own actions determine the next observation. A checkpoint can look
healthy across the entire suite and still fail on the robot.
"""
