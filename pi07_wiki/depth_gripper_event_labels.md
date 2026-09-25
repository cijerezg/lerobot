# Action-derived gripper-event labels for the depth auxiliary loss

Status: labels materialized and training consumer implemented, 2026-08-12.

## Purpose

The depth path needs a task-relevant auxiliary objective that discourages its
representation from becoming an arbitrary action-memorization code. The chosen
objective asks a **depth-only readout** to anticipate the next commanded gripper
opening and closing events.

The representation given to the readout must contain depth only. Future actions
are used to construct supervision offline; they are never model inputs. RGB,
`observation.state`, language, subtasks, quality labels, and mistake labels must
not enter either the readout or label construction.

The consumer is LayerNorm per injected depth token, mean pooling over the depth
tokens, and a linear projection to two independent logits. Ordinary soft-target
BCE is averaged over the close/open heads and added to the actor objective with
weight 0.2. A configurable positive `hidden_dim` changes only the readout to a
one-hidden-layer GELU MLP for a controlled future ablation; the live config uses
`null`. Samples for which point-map modality dropout substituted the learned
null bank are masked and the remaining samples are renormalized.

This is intentionally not a success classifier. Failed grasps, retries, setup
motions, and the operator's random gripper movements remain events if they cross
the command thresholds. Filtering them using semantic annotations would change
the target from "future gripper command" into a retrospectively judged task
label and would make the procedure annotation-dependent.

## Locked label definition

### Source signal

Read the raw, unnormalized `action` vector from the dataset parquet. Locate the
unique feature named `gripper.pos` in `meta/info.json`; it is expected to be
dimension 6 in all datasets in scope. Fail rather than silently proceeding if
the name is missing, duplicated, or maps to a different dimension.

For the reBot B601:

- $0^\circ$ is shut;
- increasingly negative values are increasingly open;
- approximately $-270^\circ$ is wide open.

Do not use `observation.state`. The command transition precedes the measured
transition by roughly 8--9 frames in this corpus, and the target is the policy's
future command rather than the actuator's delayed response.

### Hysteresis state machine

Use the thresholds already calibrated by the semantic segmentation code:

$$
\theta_{\mathrm{close}}=-60^\circ,
\qquad
\theta_{\mathrm{open}}=-90^\circ.
$$

Process each episode independently and in frame order. Initialize `closed =
False`, then update it exactly as follows:

```python
closed = g > -60.0 if not closed else g >= -90.0
```

Equivalently:

- while open, enter the closed state when $g_t>-60$;
- while closed, remain closed while $g_t\ge-90$;
- leave the closed state when $g_t<-90$.

This produces half-open closed intervals $[s,e)$. Retain an interval only when
$e-s\ge15$ frames (0.5 seconds at 30 Hz). A retained interval contributes:

- a **close event** at $s$, provided $s>0$;
- an **open event** at $e$, provided $e<T$, where $T$ is the episode length.

Thus an episode-initial closed interval may contribute an observed opening but
not an invented closing before the episode began. A terminal closed interval may
contribute its observed closing but not an opening after the episode ends. A
short rejected interval contributes neither event. Do not connect events across
episode boundaries.

The 15-frame rule is defensive. In the current corpus every detected commanded
closed interval already passes it, but it should remain part of the definition.

### Soft future-event targets

For event type $e\in\{\mathrm{open},\mathrm{close}\}$, let

$$
\Delta_e(t)=\min_{\tau\ge t:\,\tau\text{ is an event of type }e}(\tau-t).
$$

The per-frame target is

$$
y_e(t)=
\begin{cases}
2^{-\Delta_e(t)/30}, & 0\le\Delta_e(t)\le150,\\
0, & \text{otherwise}.
\end{cases}
$$

The half-life is 30 actions (1 second) and the inclusive cutoff is 150 actions
(5 seconds). At the event frame the target is 1. If several events fall in the
window, the next event wins; equivalently, take the maximum exponential target
contributed by all future events.

Distances are measured in **dense 30 Hz action frames**, not stored-image rows.
The training cache uses `image_stride: 3`, but neither 30 nor 150 is divided by
three. Labels should be generated for every frame; the sampler can select the
rows corresponding to its observation anchors later.

## Datasets in scope

Generate labels independently for all four roots:

| split | root | episodes | frames |
|---|---|---:|---:|
| train | `outputs/rebot_socks_basket-annotated-v2` | 15 | 68,635 |
| train | `outputs/rebot_shirts_bin-annotated-v2` | 9 | 30,822 |
| train | `outputs/rebot_two_container-annotated-v2` | 14 | 66,283 |
| validation | `outputs/rebot_val-annotated-v2` | 3 | 14,021 |

Episode boundaries come from
`meta/episodes/chunk-*/file-*.parquet`, using `dataset_from_index` and
`dataset_to_index`. Do not infer boundaries from action discontinuities.

No file under `meta/subtasks.parquet`, `meta/episode_metadata.parquet`, or
`meta/mistakes.parquet` may be read by the materializer. This is an auditable
constraint, not merely a claim that those values are ignored after loading.

## Required outputs

Do not rewrite the source data parquet or any existing annotation. Add
deterministic sidecars under each dataset's `meta/` directory.

### `meta/depth_gripper_events.parquet`

One row per retained, observable transition:

| column | type | definition |
|---|---|---|
| `episode_index` | int64 | episode containing the event |
| `event_type` | string | exactly `open` or `close` |
| `frame_index` | int64 | episode-local event frame |
| `index` | int64 | dataset-global event frame |
| `gripper_command` | float32 | raw `action[..., gripper.pos]` at the event frame |
| `closed_interval_start` | int64 | episode-local $s$ |
| `closed_interval_stop` | int64 | episode-local exclusive $e$ |
| `closed_interval_length` | int64 | $e-s$ |

For a terminal close event, `closed_interval_stop` is the episode length. Open
events cannot occur at that exclusive endpoint.

### `meta/depth_gripper_event_labels.parquet`

Exactly one row per dataset frame, in the same global order as the data parquet:

| column | type | definition |
|---|---|---|
| `episode_index` | int64 | copied frame identity |
| `frame_index` | int64 | copied episode-local identity |
| `index` | int64 | copied dataset-global identity; join key |
| `depth_gripper_close_target` | float32 | $y_{\mathrm{close}}(t)$ |
| `depth_gripper_open_target` | float32 | $y_{\mathrm{open}}(t)$ |
| `depth_gripper_close_delta` | int16 | $\Delta_{\mathrm{close}}$, or -1 when outside the cutoff |
| `depth_gripper_open_delta` | int16 | $\Delta_{\mathrm{open}}$, or -1 when outside the cutoff |

The delta columns are audit fields. They remove ambiguity about whether an
exponential value was calculated from the correct event and can be dropped when
packing training batches.

### `meta/depth_gripper_event_labels_info.json`

Record at least:

- rubric/version string and creation date;
- source signal and resolved gripper dimension;
- FPS;
- close/open thresholds;
- persistence, half-life, and cutoff in both frames and seconds;
- row counts and event counts;
- per-head nonzero fraction, mean, and selected quantiles;
- a statement that state and semantic annotations were not read;
- the exact command used to generate the sidecars.

Generation must be deterministic. Refuse to replace existing sidecars unless an
explicit `--overwrite` flag is supplied, and write via a temporary file followed
by an atomic rename.

### Consolidated and RGB-only roots

A consolidated root (`migration/training_mix_2026-09-21/consolidate_sources.py`) preserves
the leaf sources' event and frame-label tables with rebased indices and writes this file
by merging the leaves (`depth_event_labels_info.py` in the same folder): every field that
is identical across all leaf `_labels_info.json` files is carried over, and the row and
event counts are recomputed from the consolidated sidecars. The probes read
`resolved_gripper_dimension`, `thresholds_degrees`, `fps`, and `rubric_version` from it,
so leaves labelled under different rubrics are refused rather than averaged. The
`depth_gripper_event_info.json` next to it is the consolidation's own source summary,
not a substitute.

An RGB-only prepared source (`meta/cache_ready.json` with `depth_key: null`) carries no
label sidecars at all. Both the training collection and the probe frame loader install
zero targets for its frames; the loss is masked by the encoder's depth-valid flag, so
those frames never contribute.

## Materialization procedure

1. Read and validate `meta/info.json`; resolve `gripper.pos` by name and assert
   30 FPS.
2. Read episode bounds and the data columns `action`, `episode_index`,
   `frame_index`, and `index` only.
3. Assert that global indices and episode-local frame indices are contiguous and
   agree with the recorded bounds.
4. Run the hysteresis state machine separately inside each episode.
5. Filter closed intervals shorter than 15 frames and materialize observable
   opening and closing events.
6. Traverse each episode backward to compute the distance to the next event of
   each type in linear time. Set deltas above 150 to -1 and their targets to 0.
7. Write the event table, frame-label table, and provenance JSON.
8. Reopen the written files and run every validation check below.

Do not generate labels by scanning forward 150 frames independently for every
row. It would produce the same answer but makes the implementation needlessly
quadratic and harder to audit.

## Acceptance checks

### Structural invariants

- The frame-label sidecar has exactly the same row count and `(episode_index,
  frame_index, index)` identities as the source data.
- Every target is finite and lies in $[0,1]$.
- Every stored delta is either -1 or an integer in $[0,150]$.
- `delta == -1` if and only if the corresponding target is exactly zero.
- `delta == 0` if and only if that row is a retained event of the corresponding
  type; its target must be exactly 1.
- For positive consecutive deltas away from an intervening event,
  $y(t+1)=2^{1/30}y(t)$ within float32 tolerance.
- No target uses an event from another episode.
- Every retained closed interval is at least 15 frames and has at most one close
  and one open event under the boundary rules above.

### Expected audit values

These values were obtained from a read-only pass using the locked definition and
should be reproduced exactly for event counts. Small last-digit variation is
acceptable only for floating-point target means.

| dataset | retained closed intervals | close events | open events | close target nonzero | open target nonzero | either nonzero |
|---|---:|---:|---:|---:|---:|---:|
| socks basket | 121 | 112 | 114 | 24.2% | 24.9% | 44.5% |
| shirts bin | 74 | 67 | 67 | 31.3% | 31.9% | 54.0% |
| two container | 136 | 130 | 131 | 28.6% | 29.3% | 50.9% |
| validation | 29 | 27 | 29 | 28.2% | 30.8% | 51.5% |

Expected mean targets are respectively:

| dataset | mean close target | mean open target |
|---|---:|---:|
| socks basket | 0.0690 | 0.0704 |
| shirts bin | 0.0915 | 0.0918 |
| two container | 0.0824 | 0.0836 |
| validation | 0.0812 | 0.0876 |

Also produce a compact plot for at least two episodes per training root and all
three validation episodes showing raw gripper command, the two thresholds, event
markers, and both soft targets on a shared time axis. These plots are QA artifacts,
not manual annotations and must not alter any labels.

## Downstream loss contract

The intended consumer predicts two scalar logits from depth tokens only:

$$
[\ell_{\mathrm{close}},\ell_{\mathrm{open}}]
=h_{\mathrm{depth}}(D_t).
$$

Use ordinary soft-target binary cross-entropy:

$$
\mathcal L_{\mathrm{event}}
=\frac{1}{2}\sum_{e\in\{\mathrm{open},\mathrm{close}\}}
\operatorname{BCEWithLogits}(\ell_e,y_e).
$$

Start without focal loss, positive-class reweighting, subtask conditioning, or
success filtering. The targets are already dense enough at the five-second
cutoff. Samples on which depth modality dropout substituted the learned null bank
must be masked out of this auxiliary loss; otherwise the null tokens are trained
to predict episode-level event priors with no observation.

The auxiliary head and its weight are model-training work and are deliberately
outside the label-generation pass. The materializer's job ends when the four
datasets have verified, action-derived sidecars and QA reports.

## v2 relative-travel rule (2026-09-24, opt-in, bits root only)

`--rule relative_travel` (rubric `depth-gripper-event-labels-v2-relative-travel`). The v1 absolute band fails on tasks
whose gripper never opens past -90: bits open only to -30..-90, so v1 read whole bits episodes as one closed interval
(14 intervals over 9 episodes, ~75 real closes). v2, per episode on the raw command $g$:

- while open, track $m=\min g$ since the last opening; close at the first frame with $g \ge m + 25^\circ$;
- while closed, track $M=\max g$ since the closing; open at the first frame with $g \le M - 25^\circ$;
- 15-frame persistence, event observability, soft targets and sidecar schemas are unchanged.

Read-only comparison of closed intervals v1 -> v2: bits 14 -> 76, bottle-train-v2 33 -> 48, socks-v3 121 -> 124,
shirts-v3 74 -> 73, two_container-v3 136 -> 143, rollouts-v2 116 -> 115
(`migration/bits_annotation_2026-09-24/gripper_rule_proto.py`). Only `outputs/rebot_bits-annotated-v2` carries v2
sidecars (user decision: bits only for now). Its info has `thresholds_degrees: null` + `travel_degrees: 25`, so a
consolidation with v1 leaves is refused by `depth_event_labels_info.py`, and `probes/depth_event_probe.py` (which re-derives
the closed state from the absolute thresholds) cannot run on it until the other roots move to v2.
