# text_probe (2026-11-15)

Question: can the language model inside MolmoAct2 still answer plain questions, and does our finetuning
change that? 20 text-only questions ordered from very common to rarer, plus 4 questions on one top-camera
frame, asked to the base checkpoint and to a finetuned one.

## Files

- `run_text_probe.py`: the questions (`TEXT_QUESTIONS`, `IMAGE_QUESTIONS`), the plain chat prompt, greedy
  decoding through `lm_head`, and the answer table. The action expert is never called.

- `chat.py`: browser page for trying prompts by hand. Type a prompt, click images, read each checkpoint's
  answer side by side.

## Chat page

    uv run python lerobot/adhocs/text_probe/chat.py --config_path=config_rl_validate.yaml

Open http://localhost:8770. Loads base, tube 1200 and internal 1200 by default (`--checkpoints`, about
34 GB of GPU memory, about a minute). Images: every file of `--image_dir` plus the validation frames in
`--frames=200,600,1200,2000`; with no image picked the prompt goes in as text only. Answers are shown
raw (greedy decoding), action tokens included, in orange. Earlier runs stay on the page below the newest.

Act (added 2026-11-16, not run yet): pick robot frames (`robot_*` from the validation set, `clutter_*` from
`--clutter_root=outputs/data/rebot_2026-10-14-val-v1`, frames `--clutter_frames`), write one instruction
per line, press Act. Each line goes into the deployed action prompt as the subtask (or as the task) and is
decoded twice: action tokens through `lm_head` and the action expert. Both chunks are drawn as tip paths
in one 3-D plot per checkpoint, as in the action inspector (low-pass, reBot kinematics): solid = action
expert, dashed = action tokens, dotted = the recorded chunk, grey = the arm now, one colour per line.
When the action tokens do not decode to a chunk the cell says why; the raw tokens are under "raw action
tokens" either way. Clutter frames have no velocity or torque column: zero velocity, the level of zero
torque, no tip target clause. Layout: checkpoints and images in the left bar, Ask / Act tabs, one card
per run; a server error is printed in the cell.

## Run

    uv run python lerobot/adhocs/text_probe/run_text_probe.py --config_path=config_rl_validate.yaml \
        --checkpoints=base,outputs/runs/rebot-v2-tube-b128-20261114/checkpoints/000700/pretrained_model \
        --out_dir=outputs/analysis/text_probe/tube-ckpt700_2026-11-15

Flags: `--checkpoints` (comma list; `base` = `policy.base_path`, anything else a `pretrained_model` dir),
`--frame=600` (validation-set frame for the image questions), `--max_new_tokens=96`. About 30 s for two
checkpoints, most of it loading.

Prompt: `<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n`, with `<|image|>` in front for
the image questions. No robot clauses and no action output token. Decoding stops at `<|im_end|>` or at
`max_new_tokens`, so long answers are cut.

## Outputs (under out_dir)

- `answers.md`: one row per question, one column per checkpoint.
- `answers.json`: the same rows.
- `frame.png`: the frame shown with the image questions.

## Found

Base `outputs/models/MolmoAct2` and tube run checkpoint 700 (`rebot-v2-tube-b128-20261114`), one run:

- Both answer all 20 text questions correctly, from "Hi, how are you?" to the capital of Burkina Faso,
  17 x 23 = 391, a Python sum function and the derivative of x^3. Both call themselves "Molmo ... built
  by Ai2".
- Checkpoint 700 gives the same facts as the base; the wording differs on about half the questions. No
  action tokens appear in any answer.
- Image questions (frame 600: three clear pegs and three white tubes on the table, gripper at the top):
  both describe a wooden table with "small clear plastic bags"; the base also names the robot arm and
  "three tiny white objects", checkpoint 700 does not mention the robot. "What color is the tube?" gets
  "black" (base) and "yellow" (checkpoint 700); the tubes are white. Both answer the pointing question
  with a `<points coords=...>` tag near the gripper (499, 200 and 502, 188, if read as x, y out of 1000;
  the gripper is at about 450, 190).

### Long suite and internal-only checkpoint 1200 (same day)

`--suite=long --max_new_tokens=512` runs `LONG_TEXT` (9 long single prompts, a 6-turn and a 5-turn
conversation) and `LONG_IMAGE` (one long prompt, one 5-turn conversation on the frame). In a conversation
the model's own answer is fed back as history. Output: `outputs/analysis/text_probe/long_2026-11-15`
(base, tube checkpoint 700, `rebot-internal-only-20261021` checkpoint 1200); the basic suite on the
internal checkpoint is in `outputs/analysis/text_probe/internal-only-ckpt1200_2026-11-15`.

- Base and tube checkpoint 700 hold a conversation: both recall the name and the number 7421 after four
  turns, and both carry 12 -> 24 -> 19 -> 57 -> "not prime, 3 x 19" across five turns.
- Reasoning is weak on both: nobody finds the binary search bug (`lo = mid`), both LRU caches use a list
  and are O(n) while claiming O(1), word counts of their own poems are wrong, the base garbles the train
  meeting time (checkpoint 700 gets 11:53 and 231 km), and object counts on the frame are unstable (3 or
  4 or 5 white objects; there are 3).
- Tube checkpoint 700 leaks one `<action_80>` token twice in one answer and one Chinese fragment in
  another; otherwise clean.
- Internal-only checkpoint 1200: basic text questions 18 of 20 clean (two answers break into action
  tokens mid-sentence), every image question answered with action tokens only. On the long suite it
  answers 5 of 9 long prompts fully in text (pet puzzle and odd-number proof correct, reading questions
  right with a wrong summary, LRU in prose without the requested code, binary search wrong), breaks
  into action tokens partway through 3, gives action tokens only on 1, and answers every
  conversation turn, with or without image, with `<action_k>` tokens ending in `<action_end>`.
- Both finetuned runs train FAST action tokens through `lm_head` (`action_mode: both`), so the cause of
  the difference between them (steps, data, schedule) is not established here.
- Tube run checkpoint 1200 (same step count, learning rate 5e-5 and effective batch 128 as the internal
  run), both suites, `outputs/analysis/text_probe/tube-ckpt1200_2026-11-15` and `tube-ckpt1200-long_2026-11-15`:
  basic suite 24 of 24 in text, image questions included; long suite 3 stray action tokens in 2 of 26
  answers, conversations intact. So 1200 steps alone do not produce the internal checkpoint's collapse.
  The two runs differ in data and in other settings (weight decay 0.1 vs 0, schedule, prompt clauses),
  which this does not separate.

### Switch ladder, masked decode, weight shift (same day)

New in `run_text_probe.py`: every run also writes `action_mass.md` (probability on the action tokens at the
first answer position / the largest along the answer); `--mask_action_tokens=true` decodes with the action
tokens forbidden; `--suite=switch` runs `SWITCH`, 13 prompts from plain chat to the wording of the training
prompt (last rung: the assistant turn starts with `<action_output>`, as in training; state / velocity /
torque clauses left out). Table headers are now `<run>/<step>`.

`run_weight_shift.py` (no model load, about 30 s): per group of weights, `|checkpoint - base| / |base|`, the
shrink part of the move (`-cos(delta, base)`, what weight decay produces) and the cosine between the moves of
the first two checkpoints.

    uv run python lerobot/adhocs/text_probe/run_weight_shift.py --checkpoints=<dir>,<dir>,... \
        --out_dir=outputs/analysis/text_probe/weight_shift_2026-11-15

Outputs: `switch_2026-11-15` (base, tube 100..1200, internal 1200; masked),
`internal-only-ckpt1200-masked_2026-11-15`, `internal-only-ckpt1200-masked-long_2026-11-15`,
`weight_shift_2026-11-15`. One run each.

Probability on the action tokens at the first answer position:

| prompt | base | tube 100 | tube 400 | tube 800 | tube 1200 | internal 1200 |
|---|---|---|---|---|---|---|
| Hi, how are you? | 0.001 | 0.002 | 0.004 | 0.007 | 0.010 | 0.507 |
| What is the capital of France? | 0.001 | 0.001 | 0.004 | 0.006 | 0.009 | 0.104 |
| How would a robot arm put a small tube on a peg? | 0.002 | 0.003 | 0.010 | 0.018 | 0.022 | 0.900 |
| training question, no image | 0.024 | 0.758 | 0.511 | 0.942 | 0.997 | 0.935 |
| image: Describe this image. | 0.002 | 0.002 | 0.024 | 0.018 | 0.013 | 0.961 |
| image: How many small white objects are on the table? | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 | 0.943 |
| image: What should the robot do next? | 0.946 | 0.995 | 0.947 | 0.968 | 0.999 | 0.972 |
| image: training question | 0.996 | 0.997 | 0.977 | 0.996 | 1.000 | 0.992 |
| image: training question + `<action_output>` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

- The base already answers a robot question on an image with action tokens, without the action output token.
- Tube run: chat prompts, with or without image, stay at or below 0.06 at the first position through step
  1200 and stop rising after about step 400. What changes is the training question without an image
  (0.02 -> 1.00). The tube run is not on the way to the internal checkpoint's state.
- Internal 1200: 0.10 to 0.97 on every prompt, 0.79 or more on every image prompt.
- Internal 1200 with the action tokens forbidden: 20 of 20 basic text questions correct, both text
  conversations intact (name and 7421 recalled, 24 -> 19 -> 57 -> not prime). Some long answers are worse
  than the base (pet puzzle goes wrong, a garbled code answer). Image answers are mixed: the scene is
  described as a bare wooden surface and "no objects on the table" (the base names the robot and the
  objects), the gripper point is off (498, 488; base 499, 200), but it counts three white objects, which is
  right (base: 4 or 5).
### Non-robot images (2026-11-16)

`--suite=images` asks `IMAGES_QUESTIONS` (4 prompts) on the validation frame and on every image of
`--image_dir` (default `outputs/data/coco_val2017_sample`: 20 COCO val2017 photos, both sides 378 px or
more). Output: `outputs/analysis/text_probe/images_2026-11-16` (base, tube 1200, internal 1200), one run.

Probability on the action tokens at the first answer position, robot frame / median of the 20 photos
(photos whose answer starts with action tokens):

| prompt | base | tube 1200 | internal 1200 |
|---|---|---|---|
| Describe this image. | 0.002 / 0.000 (0) | 0.013 / 0.013 (0) | 0.961 / 0.485 (17) |
| What objects are in this image? | 0.001 / 0.000 (0) | 0.009 / 0.010 (0) | 0.896 / 0.328 (12) |
| What should the robot do next? | 0.946 / 0.001 (0) | 0.999 / 0.996 (20) | 0.972 / 0.951 (20) |
| training question | 0.996 / 0.308 (5) | 1.000 / 1.000 (20) | 0.992 / 0.981 (20) |

- Base: the switch reads the image. "What should the robot do next?" gives action tokens on the robot
  frame and plain text on all 20 photos ("move away from the bear"). The training question on a photo is
  mostly answered in text ("There is no tube or peg in this image").
- Both finetuned checkpoints: a robot question gives action tokens on every photo. The image no longer
  gates the switch, the wording does.
- Internal 1200: describe prompts drop from 0.96 to a median of 0.49 on photos, still action tokens first
  on 17 of 20.
- Tube 1200 stays in text on describe prompts, but the descriptions are wrong on most photos: bear ->
  dog, teddy bears -> "wooden object, possibly machinery", swan and woman -> warehouse, two jets -> metal
  structure, apples -> "topsoil of a flower pot". About 4 of 20 name the main subject (bus, cat, woman
  with phone, zebra); the base is right on all 20. Read by eye from the first sentence, not scored.

- Weight shift at step 1200: both runs moved the language model blocks by 3 to 7 % (tube 0.032 to 0.058,
  internal 0.027 to 0.071; internal more in blocks 5 to 21, tube more in blocks 28 to 35). Blocks 0, 1 and
  the input embedding did not move in either. The two moves are mostly different directions (cosine 0.11 to
  0.23 per block; no same-data reference for that number). The tube move has a shrink part of 0.07 to 0.14
  (weight decay 0.1), the internal one none. `lm_head` text rows moved 0.7 % in both.

### Layers tab (2026-11-16, not run yet)

Two runs side by side, A and B: checkpoint, image (or none), chat prompt or one of the three action prompts,
text. Each is forwarded once and the output of all 36 language-model blocks is kept at every position
(`layer_capture.py`). Shown, all with raw cosine (no mean subtracted):

- A against B at the same position: map over layers and positions with the kind of token on top, mean per
  layer by kind of token plus the last position, box plots at the chosen layers (default 0, 7, 14, 21, 28, 35).
- Inside each run: mean cosine between two different positions per layer, participation ratio of the
  positions, mean vector length, and a position-by-position matrix at one layer on request.
- Action expert blocks, A against B, when both runs are action prompts (first flow step, same noise).

Positions are matched by kind (k-th image patch with k-th image patch, k-th state token with k-th state
token, ...), so runs of different length still compare like with like; the page lists what was left unmatched. Same checkpoint + two prompts, or same prompt + two checkpoints, are both just choices of A and B.

### Layer study (2026-11-16)

Question: how the per-layer vectors differ between base, tube 1200 and internal 1200, and where the
instruction goes. Report with methodology, tables and commands:
`outputs/analysis/layer_cosine_2026-11-16/REPORT.md`.

| File | What it does |
|---|---|
| `layer_sweep.py` | forwards a fixed list of prompts per checkpoint, saves all block outputs + action expert context, attention, chunks (GPU; store under `outputs/cache/layer_cosine_2026-11-16/`, 121 GB) |
| `layer_analysis.py <stage>` | cross / prompt / series / expert / stream / lens tables from the store (no GPU) |
| `layer_weights.py` | relative weight change per block against base (no model built) |
| `layer_patch.py` | swaps what the action expert reads (keys / values) between two instructions by position or by layer band; hides high-norm positions (GPU) |
| `layer_vision.py` | cosine to base after every vision block, pooling, projector (GPU) |
| `layer_report.py` | figures 1 to 9 (no GPU) |
| `layer_edit.py` | second round, library: which positions have a high norm; hooks that clamp a norm, undo a block update, read attention and per-block updates |
| `layer_norms.py` | `--stage=trace`: norm per block and per vision stage, language-model attention, action-token decoding; `--stage=chunks`: the chunk under edits of the high-norm positions (GPU) |
| `layer_instruction.py` | one block's output taken from another instruction; the expert's keys and values scaled; `--stage=donors`: one checkpoint's expert on another's language model (GPU) |
| `layer_ability.py` | scored photo and text questions per checkpoint, under the edits and with the vision path of another checkpoint (GPU) |
| `layer_report2.py` | figures 10 to 16 (no GPU) |

A high-norm position = L2 norm of the block output over 10 x the layer median. The saved files of the first
round keep the key `massive` for it.

Found (n and tables in the report):

- Block 16 of the base model gives 20 to 40 positions per prompt a high norm (about 10,000 against a
  median of about 60). This is the sharp edge of the layer map; it is a backbone property.
- The high / low / high cycle is real only for text positions after an image. The late recovery is a shared
  direction: shuffled positions are as close as matched ones.
- Image tokens of a finetuned checkpoint differ from base at layer 0 (cos 0.5). The vision tower drifts
  from its block 2 on; after 100 steps the projected image tokens are at cos 0.66 with base.
- Distance from base saturates by step 300 of the tube run.
- The action expert takes the instruction from the positions after it (95 to 99 % of the chunk change),
  layers 18 to 23 (base, tube) or 24 to 29 (internal), and 1 to 4 % from the instruction tokens.
- Hiding the high-norm positions from the expert changes the chunk 9 to 20 times more than hiding as many
  ordinary image positions; taking their content from another frame changes it no more than for ordinary ones.
- Internal collapses in the last 6 layers: two different positions have cosine 0.80 at layer 35 (base 0.30,
  tube 0.49).

Second round:

- High-norm positions have three sources: 2 fixed patches per image from the vision tower (ViT block 9 +
  projector), 3 to 4 text tokens from the MLP of blocks 6 and 16, 17 to 37 image patches from the MLP of
  block 16. Finetuning changes only the last group, and which patches it picks follows the vision path
  (overlap 0.07 to 0.19 across vision paths, 0.39 to 0.71 across language models).
- Image ability falls with training (one-word photo answers 19, 18, 14, 12, 9 of 20 for base and tube 100
  to 1200); text-only facts stay at 31 to 35 of 35. With the base vision path every checkpoint is back at
  82 of 82 and 18 to 19 of 20. Editing the high-norm positions does not restore or remove it.
- Clamping the norm of the high-norm positions costs the finetuned checkpoints text-only facts (31 to 14,
  35 to 15) and base almost none.
- Withdrawn: "the instruction is lost inside the tube language model". Crossing the experts and language
  models of base and tube does not single out a part.
- Internal's text positions decode to action tokens in layers 30 to 35 on chat prompts (0.69 at layer 35;
  base 0.08, tube 0.10), the same layers as its collapse. No prompt-by-prompt link.

### Backbone crossing (2026-11-16, run by the user)

Question: how much of internal 1200's action output depends on which vision path and which language model
feed its action expert. The expert (and internal's prompt format, stats, depth modules) is fixed; the
vision path is base or internal, the language model base, tube 1200 or internal: 6 cells. Data: the 8
episodes of `outputs/data/rebot_2026-10-14-val-v1`, 20 frames per episode, 4 flow seeds (same noise in
every cell).

    uv run python lerobot/adhocs/text_probe/backbone_cross.py --config_path=config_rl_validate.yaml
    uv run python lerobot/adhocs/text_probe/backbone_cross_view.py outputs/analysis/backbone_cross_2026-11-16/chunks.pt

| file | does |
|---|---|
| `backbone_cross.py` | loads the expert checkpoint with its own saved policy config, swaps `model.model.vision_backbone.*` and `model.model.transformer.*` from the checkpoint files, saves chunks (degrees + normalized), recorded chunk, state, images |
| `backbone_cross_view.py` | page on port 8771, no model: table vision path x language model for a chosen quantity (`METRICS`), per-frame chart, error along the chunk, and per frame the tip paths of every cell against the recorded chunk in 3-D + joints |

Reading the numbers:

- Open-loop on recorded frames. Distance to the recorded chunk is not an error on its own (several chunks
  are valid); read mean and best seed together, and the seed variation as the scale.
- The expert was trained with internal's language model, so every other cell also pays for the mismatch.
- The tube language model was trained on another prompt format (velocity, gripper torque, tip target, no
  quality clauses) and without depth tokens; base never saw depth tokens either. `--depth=false` removes
  them from every cell (then the internal cell is off its training input).
- `lm_head` is not swapped (the continuous path does not use it).

Found (160 frames, 4 seeds, tip, cm; table read from chunks.pt 2026-11-17; seed variation of all-internal 2.33):

| vision path | language model | to the recorded chunk | to all-internal, same seed |
|---|---|---|---|
| internal | internal | 3.21 | 0 |
| base | internal | 3.26 | 1.62 |
| internal | base | 6.27 | 5.60 |
| base | base | 5.91 | 5.40 |
| internal | tube | 5.81 | 5.33 |
| base | tube | 5.74 | 5.33 |

The base vision path under internal's language model stays inside the seed variation; any other language model is about
5.5 cm away whichever vision path feeds it. Report: `docs/reports/vlm_parts_2026-11-17.md`.

### Step swap (2026-11-16, not run yet)

Question: how far does the chunk follow the step text, for internal 1200 whole and for two crossings of it
with base: (a) base vision path + base language model with internal's action expert, (b) internal's vision
path + language model with base's action expert. On the same frames as the backbone crossing, each model
predicts chunks under the frame's own step, "return to home", the other steps of the same episode, and no
step clause; everything else of the prompt and the observation is unchanged, same flow noise everywhere.

    uv run python lerobot/adhocs/text_probe/subtask_swap.py --config_path=config_rl_validate.yaml
    uv run python lerobot/adhocs/text_probe/subtask_swap_view.py outputs/analysis/subtask_swap_2026-11-16/chunks.pt

| file | does |
|---|---|
| `subtask_swap.py` | loads the checkpoint with its own saved policy config; `MODELS` names the parts each model takes from another checkpoint (`model.model.action_expert.*` added to the two parts of the crossing); saves chunks per (model, frame, step text, seed) and the arm pose where each step ends in the recording |
| `subtask_swap_view.py` | page on port 8772, no model: table model x kind of step text for a chosen quantity (`METRICS`), per-frame chart, change along the chunk against seed variation, and per frame one 3-D scene per model with a tip path per step text + gripper |

Reading the numbers:

- The change is the tip distance to the own-step chunk of the same seed; the seed variation under the own
  step is its scale (ratio near 0 = the text is ignored).
- "Approach" is toward the arm pose where that step ends in the same episode (last frame of its stretch).
  For an earlier step the scene has changed since, and a move / release step is ill-posed when nothing is
  held. For home the pose is `--home` (the pose episodes 2, 3, 5, 6, 7 start from, rounded); no home pose
  is configured anywhere.
- Model (b): base's action expert was never trained on internal's action normalization or anchor encoding,
  so its chunks are not expected to be sensible in absolute terms; only its change with the text is read.
- Model (a) pays for the mismatch as in the backbone crossing, and base never saw depth tokens.

Found (run by the user 2026-11-16; 160 frames, 4 seeds; own step against "return to home" on 157 frames, tip, cm):

| model | shift of the chunk with the text | distance to internal, own step / home | ends closer to home than under the own step (frames closer) |
|---|---|---|---|
| internal | 6.29 | 0 / 0 | 5.43 (144 of 157) |
| (a) base vision + language model, internal expert | 1.13 | 5.34 / 5.20 | 0.15 (94 of 157) |
| (b) internal vision + language model, base expert | 5.62 | 5.00 / 4.92 | 4.47 (136 of 157) |

Following "return to home" comes with internal's vision + language model: base's action expert, never trained
with it, gets about 80% of internal's move toward home; internal's expert on base's backbone gets none. Limits:
the home pose is the rounded `--home` guess; (a) shows that base's backbone does not deliver the step to this
expert, not that it lacks the information; both crossings are about 5 cm from internal's chunk under either text.

### Vision path across depth (2026-11-18)

Question: internal 1200 gives about the same chunks on the base vision path (backbone crossing). Does its
language model bring the two sets of image tokens together, and at which layer?

    uv run python lerobot/adhocs/text_probe/vision_depth.py --config_path=config_rl_validate.yaml
    PYTHONPATH=lerobot/adhocs/text_probe uv run python lerobot/adhocs/text_probe/vision_depth_report.py outputs/analysis/vision_depth_2026-11-18/depth2.pt

| file | does |
|---|---|
| `vision_depth.py` | internal whole as the reference a; per frame, conditions x that change only the image tokens: base (the base vision path's tokens), random (a + a random direction per token with the length of base - a), shuffled (a + the differences base - a moved to other patches of the present cameras), next (camera images 1 s later), other (camera images of another episode). Per layer, median over the positions of a kind: \|x - a\|, \|a\|, \|a - mean\| (mean of the reference over frames), cosine(x, a), cosine(x - mean, a - mean), for the block outputs, the keys / values the action expert reads, the expert's cross-attention and block outputs. And the chunks of 8 flow seeds. `image_tokens` puts any tokens in place of the vision backbone's output |
| `vision_depth_report.py` | `depth2.md` (tables), `depth2_output.png`, `depth2_layers.png`, `depth2_lengths.png` (no model) |

Found (160 frames, 8 seeds). Chunk against the reference chunk of the same flow seed:

| | tip distance, step 5 (cm) | mean over the chunk (cm) | frames over their own seed variation | cosine of the tip paths | tip distance to the recorded chunk (cm) |
|---|---|---|---|---|---|
| two flow seeds of the reference | 1.80 | 2.08 | | 0.82 | 3.19 (reference) |
| base vision path | 1.30 | 1.60 | 37 of 160 | 0.90 | 3.23 |
| random | 0.34 | 0.45 | 0 | 0.98 | 3.26 |
| shuffled | 0.83 | 1.10 | 7 | 0.94 | 3.33 |
| images 1 s later | 1.49 | 1.86 | 59 | 0.85 | 3.39 |
| images of another episode | 4.38 | 5.88 | 147 | 0.29 | 6.52 |

The reference chunk takes the tip 5.27 cm from the current pose (mean over the chunk). The base shift is the
same in every seed (1.54 cm between the seed means).

Layers, \|x - a\| / \|a - mean\| and cosine(x - mean, a - mean):

| | base | random | shuffled | 1 s later | another episode |
|---|---|---|---|---|---|
| image positions, input | 1.00, 0.43 | 1.00, 0.71 | 1.00, 0.71 | 0.55, 0.84 | 1.16, 0.26 |
| image positions, block 36 | 1.03, 0.46 | 0.59, 0.82 | 0.85, 0.60 | 0.76, 0.71 | 1.26, 0.19 |
| text positions, block 16 | 0.27, 0.96 | 0.12, 0.99 | 0.25, 0.97 | 0.20, 0.98 | 0.49, 0.88 |
| text positions, block 36 | 0.69, 0.75 | 0.34, 0.94 | 0.58, 0.83 | 0.61, 0.80 | 1.06, 0.43 |
| expert block 36 output | 0.47, 0.87 | 0.16, 0.99 | 0.35, 0.93 | 0.49, 0.86 | 1.11, 0.35 |

- The two streams do not come together: at image positions the base tokens stay one spread away from the
  reference at every block; at text positions the difference grows from 0 to 0.69.
- At the text positions and in the action expert the base vision path is the size of the camera images 1 s
  later, and under the distance between two flow seeds at the output.
- A random direction of the same length is absorbed about 3 times better (chunk 0.45 cm).
- Divide by the spread, not the length: the expert's cross-attention output at block 35 has length 12.6 and
  spread 0.32 (a vector that is the same in every frame), so \|x - a\| / \|a\| drops to 0.02 there in every
  condition while \|x - a\| / \|a - mean\| is 0.90, as in blocks 34 and 36.
- Shuffled is not a clean control in the first layers: it changes the text positions at block 1 by 0.66 of
  the spread (base: 0.05) and the first expert blocks by over 10.
- Limits: open-loop, recorded frames; "1 s later" and "another episode" keep the frame's own state and depth,
  so they are inputs the model never saw. `depth.pt` / `depth.md` / `depth.png` are the first run (4 seeds,
  \|x - a\| / \|a\| only).
