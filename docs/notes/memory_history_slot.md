# Short-term memory as its own image slot (design 2026-11-04)

> **Status:** built 2026-11-04, trained once (rebot-mem-slot-internal-20261104, internal-only, base ckpt), PARKED 2026-11-07: memory off in the live yamls (`history_keys: []`, migration/config_memory_off_2026-11-07/), run outputs and probe results deleted; code kept. On landing, this replaces the consumption half of [memory_prompts.md §2.4](../design/memory_prompts.md) and this note moves to archive. Build report: `migration/memory_history_slot_2026-11-04/REPORT.md`.

Supersedes the consumption half of [memory_prompts.md §2.4](../design/memory_prompts.md): the
MEM video encoder stays, but its output no longer replaces the current image. It
becomes an extra image per camera. Everything upstream (buffer windows, actor deque,
age stamps, state projector) is unchanged.

## 1. Why

Two earlier runs are the evidence.

| run | what entered the model | verdict |
|---|---|---|
| mem_loss_v1 (2026-09, in-place MEM, digit state) | current image rows rewritten by temporal attention | history content moved the flow chunk r 0.03, noise floor 0.35 ([notes](mem_temporal_attention_analysis.md)) |
| rebot_mix_memory_20260923 (in-place MEM + continuous current state) | same, plus full-precision current state token | val flow 9-20 % above memory-off; "ignores images" |

The in-place encoder changes the current-frame tokens by a relative norm of about 1.0
to 1.2 the moment history is on, independent of history content (09-08 dissection).
The LLM therefore sees a different image as soon as memory exists and has to recover
before it can benefit. Depth, the one modality that is measurably read
(no_depth penalty t = 3.76), entered as its own tokens in the prefix. Memory gets the
same treatment: the current image stays byte-identical, the past arrives beside it.

## 2. Decisions

| item | decision |
|---|---|
| ladder | -6, -4, -2 s (= the diverse cache's stored ages, no rebuild; 180/120/60 frames, stride-3 aligned) |
| cameras | every present camera gets one history slot (ReBot: external_0, wrist_0; diverse: three) |
| current state | digit tokens, unchanged |
| past states | `state_projector` tokens as today, plus a sinusoidal age stamp |
| depth history | none |
| history dropout | 0 |
| current-frame cutout | none |
| start checkpoint | base MolmoAct2 |
| temporal layers | every 4th resblock, unchanged |

With dropout 0 and no cutout the only pressure to read the past is the data itself.
The hypothesis the run tests is that the past, aligned patch-for-patch to the current
view and delivered in its own slot, gets read where the in-place rewrite was not.
Flagged once here, not repeated.

## 3. Prompt layout

For a ReBot sample with cameras `c = 1..C` (C = 2):

```
Image 1<|image|> Image 2<|image|> Image 3<|image|> Image 4<|image|>
<task> The depth of the scene is <depth x192>.
The current state of the robot is <digits>.
The recent states of the robot, oldest to newest, are <extra_0><extra_0><extra_0>.
Images 3 to 4 are the same cameras as images 1 to 2, 2 seconds earlier, with the frames from 4 and 6 seconds earlier folded in.
<metadata clause> ...
```

Images `1..C` are the current frames, images `C+1..2C` the history slots, same camera
order. A history slot's pixels are the camera's newest past frame (-2 s, taken from the
gathered window and normalized like a current frame); the older frames (-4, -6 s) reach it
only through the video encoder (§4). The new clause is one sentence, worded from the
configured ages; it is absent when a sample has no history (byte-identical legacy prompt,
which is what `history_dropout` would produce). (Changed 2026-11-04 at build time: the
first draft put a copy of the current frame in the slot and ran the temporal pass over
[past..., current].)

Token count per ReBot sample:

| block | now | with memory |
|---|---|---|
| current images, C x 196 | 392 | 392 |
| history slots, C x 196 | 0 | 392 |
| depth, 192 + clause | 208 | 208 |
| text | ~110 | ~160 |
| past-state placeholders | 0 | 3 |
| total | ~710 | ~1160 |

Independent of the ladder length. `infer_molmoact2_max_sequence_length` already
counts images from `max_num_images`, which the pack step sets from the packed image
list, so the budget follows automatically.

## 4. Encoder

Let `x_c` be the current frame of camera `c` and `h_{c,1..K}` its past frames,
oldest first, K = 3, ages `t_k` in seconds (6, 4, 2).

Plain slot (images `1..C`): the stock single-frame ViT, exactly today's
`num_frames == 1` branch.

$$ f^{\text{cur}}_c = \mathrm{ViT}(x_c) $$

History slot (images `C+1..2C`): the MEM path on the past frames alone, T = K = 3 rows,
no t = 0 row. The age stamp e(t_k) is added at every temporal layer input (nonzero on
every row; e(0) = 0 stays the boundary condition of the same function), the time-only
softmax runs causally over the rows then the stock spatial block; after the last
temporal layer the older rows are dropped and the newest row, t = t_3 = 2 s, is the one
taken at the feature taps:

$$ f^{\text{hist}}_c = \mathrm{ViT}_{\text{MEM}}\big([h_{c,1}, h_{c,2}, h_{c,3}]\big)\big|_{t = t_3} $$

So the -2 s row has read -6 and -4 (causal, same patch); the -6 row's temporal step is
self-only by the causal mask. The current frame never enters the temporal pass. In the
batch the slot crop (the -2 s frame, from the prompt image list) is the last time row and
the stash's older frames `history_images[:, :, :-1]` are the earlier rows; the stash shape
is unchanged, the model slices it.

The two slots share every weight. Features are concatenated in crop order `[f^{cur}_1 ..
f^{cur}_C, f^{hist}_1 .. f^{hist}_C]` and scattered onto the `<|image|>` patch positions
by the existing mechanism, so `build_input_embeddings` does not change.

Absent cameras (ReBot's external_1): both of the camera's `<im_patch>` spans leave the
attention mask and its slot runs self-only in the temporal step (§8).

Episode start: slots that fall before the first frame repeat the earliest valid frame
and carry `is_pad`; the temporal attention masks them to self-only as today.

ViT work per ReBot sample: 2 frames now, 8 with memory (2 cameras x (1 current + 3
past)); under the live 3-role config a ReBot row runs 12 rows because the absent
camera's placeholder pair runs too. Measured on the base model: 13.6 -> 54.4 ms for the
ViT, 127 -> 185 ms for the action chunk (report §Forward latency).

`history_image_mode: inplace` keeps the 2026-08-03 behaviour for ablation: the current
frame is the last row with age 0, the whole stash rides before it, and its row survives.

## 5. Past-state tokens

Unchanged transport: `state_values` (B, K, D), `state_values_mask` (B, K),
placeholders `<extra_0>` x K in the history clause, projected by `state_projector`
and added onto the placeholder embeddings. New: the age stamp, same sinusoid the ViT
uses, in the text width `d`:

$$ \tilde h_{t-k} = W s_{t-k} + b + e_d(t_k) $$

`e_d` is `_sinusoidal_seconds_embedding(times, d)`, no parameters. The ages ride in the
batch as `history_image_times` already; the scatter reads them from there. The
generation prompt ships no past states, unchanged.

## 6. Code changes

| file | change |
|---|---|
| `configuration_molmoact2.py` | `history_image_mode: "slot" \| "inplace"` (default `slot`; `inplace` keeps the 2026-08-03 behaviour for ablation) |
| `processor_molmoact2.py` | pack step appends the newest history frame per camera when a history window is present and the sample's history is on; `_build_robot_text` gets the slot clause (one sentence, worded from the ages); `max_num_images` counts the slots |
| `modeling_molmoact2.py` | `_encode_image`: when the stash is present and mode is `slot`, run crops `[:C]` plain and crops `[C:]` through the MEM path, concatenate; state scatter adds `e_d(t_k)` |
| `config_rl.yaml`, `config_rl_validate.yaml` | `history_keys` = 3 camera roles + `observation.state`; offsets -6/-4/-2 (already); `history_dropout 0`; `pretrained_path` = base |
| tests | `test_molmoact2_mem_encoder.py`: slot features for crops `[:C]` bit-match the single-frame path; crops `[C:]` differ; absent camera has no slot. `test_molmoact2_prompt.py`: clause present / absent, placeholder count, budget. |

Nothing in the buffer, the actor runtime, the critic (skipped in this run) or the
freeze whitelist changes. The only trainable addition over today's checkpoints is still
`state_projector`.

## 7. What is deliberately not in this design

Probes and training telemetry are a separate design (occluded held-out frames, history
on vs masked, temporal past-mass, which episodes carry occlusions). Current-frame
cutout and history dropout are available as switches and off.

## 8. As built (2026-11-04)

Deviations from §2-§6, with the reason. The past-frames-only slot (§3-§4 as now written)
was the owner's change during the build and is the built behaviour, not a deviation.

| item | design | built | why |
|---|---|---|---|
| absent camera | "no current slot, no history slot" | both slots exist in the token layout; both `<im_patch>` spans leave the attention mask (`mask_camera_patch_span` on `cam_index` and `cam_index + C`), and the encoder runs the camera's history slot self-only (`history_images_mask`) | the current slot was already a masked placeholder, never omitted: the layout stays uniform across a mixed batch. The history slot follows the same mechanism. So a ReBot row under the 3-role config carries 6 images (2 x 196 masked), not 4 |
| budget | "follows automatically" from `max_num_images` | images yes; the clause text (37 tokens) rides in `MOLMOACT2_HISTORY_CLAUSE_TOKEN_BUDGET`, raised 16 -> 56 (13 past-state lead-in + 37) | the image count covers the slots, not the sentence |
| generation prompt | §5 says "ships no past states, unchanged" and is silent on slots | gets the same slot images and the same clause when history is present (no past states, as before) | in slot mode a stash with C crops would be a shape error; the HL decode keeps seeing image history through the shared encoder as in §2.4 |
| RGB dropout | not mentioned | a dropped camera loses its history slot's span too | otherwise the dropout is defeated through the slot |
| future-visual capture | not mentioned | `_lerobot_future_features` is crops `[:C]` in slot mode | the objective compares against C future frames; `future_visual_loss` is off in the run config, and its eval-time `pixel_values.reshape_as(future_images)` is not adapted to 2C crops |
| state age stamp source | `history_image_times` | as designed; `state_values` therefore needs image history in the same window (true for the configured keys) | a state-only `history_keys` would raise on the missing key |
| slot mode + `history_dropout` > 0 + absent cameras | not covered | unsupported in one batch: rows with and without slots carry different `<im_patch>` counts, which `mask_camera_patch_span` rejects | dropout is 0 in this run; the in-place mode keeps the uniform count |
| `mem_temporal_attention` | "unchanged" | the capture hook still records once per temporal layer with `(BC, heads, ages)` shapes, but the query row is now the -2 s row and the recorded past keys are two (-6, -4), while the probe labels the age axis from `memory.history_times_seconds()` (three ages) | probe not redesigned here; the exact change is in the build report |

Everything else is as specified: `history_image_mode` ("slot" default, "inplace" = 2026-08-03
path), the clause position (after the past-state clause), crop split in `_encode_image` with
the plain path bit-identical to the pre-MEM encoder (unit-tested on the stub ViT), e_d(t_k)
on the projected past states, no new parameters, buffer / actor runtime / critic / freeze
whitelist untouched.
