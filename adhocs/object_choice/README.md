# Object choice: does the arm go to the object the instruction names?

Built 2026-11-17. Status: run once 2026-11-17.

## Question

The backbone crossing showed internal 1200's action expert works about as well on the released vision encoder as on its own.
Here that is tied to behaviour in sim, together with a second question: when the language model can tell two objects apart,
does the action side use it?

## Setup

Scenes (`scripts/sim_builders/object_choice`, tier `choice` in `robots/rebot_b601_sim/tasks.py`): the basket and two objects.

| pair | objects | in the training data |
|---|---|---|
| pill_roar | pill bottle, orange bottle | both |
| pill_banana | pill bottle, banana (YCB 011) | pill bottle only |
| pill_mug | pill bottle, white mug | pill bottle only |

Six placements per pair, in couples that exchange the two objects on the same spots: one object on each side (p1, p2), both
on the robot's right (p3, p4), both on the robot's left (p5, p6). Each placement runs under both instructions
("Put the pill bottle in the basket" / "Put the banana in the basket"; same step script and contact word, only the noun
differs): 36 layouts, 4 episodes each, 30 s.

Cells (the action expert is always internal 1200's; depth off, as `config_rl_sim.yaml`):

| cell | vision encoder | language model | checkpoint |
|---|---|---|---|
| internal | internal | internal | `outputs/runs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model` |
| base_vision | base | internal | `outputs/analysis/object_choice_2026-11-17/checkpoints/base_vision/pretrained_model` |
| base_vision_lm | base | base | `outputs/analysis/object_choice_2026-11-17/checkpoints/base_vision_lm/pretrained_model` |

The two merged folders are written by `merge_checkpoints.py` (done 2026-11-17, 11 GB each).

Measures:
- sim: per episode, which object the fingers reached first (within 5 cm), touched first, lifted first (`choice` in the episode
  record, `robots/rebot_b601_sim/choice_tasks.py`); the reward is the named object's pick-and-place ladder.
- questions (`ask_sides.py`): each placement's top-camera start image to each cell as plain chat, action tokens forbidden:
  the object's side in the image, its side from the robot's point of view (opposite, the camera faces the robot), and
  "What objects are on the table?".

## Run

From the workspace root.

```bash
R=outputs/analysis/object_choice_2026-11-17
I=outputs/runs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model
C='from lerobot.robots.rebot_b601_sim.tasks import main; main()'
for cell in internal base_vision base_vision_lm; do
  ckpt=$R/checkpoints/$cell/pretrained_model; [[ $cell == internal ]] && ckpt=$I
  LAYOUTS="$(PYTHONPATH=lerobot/src .venv/bin/python -c "$C" choice_layouts)" PARALLEL=16 SIM_VIDEOS=all \
    lerobot/scripts/run_sim_eval.sh $ckpt $R/sim/$cell
done
uv run python lerobot/adhocs/object_choice/ask_sides.py --config_path=config_rl_validate.yaml
uv run python lerobot/adhocs/object_choice/summary.py     # tables -> $R/summary.md
```

Rebuild the scenes: `build_assets.py`, then `build_choice_envs.py` (`--check` validates only) in `scripts/sim_builders/object_choice`.

## Reading the tables

- "went to named" near half with "couples: one of each" high: the arm goes by position, not by the noun.
- "couples: named in both" high: the arm follows the object when the two are exchanged.
- pill_roar is the control: both nouns were trained, so internal should follow the swap there if it follows any instruction.

## Findings

Run by the user 2026-11-17 (4 episodes per layout, 144 per cell). Tables: `outputs/analysis/object_choice_2026-11-17/summary.md`.

| cell | went to named | went to other | neither | couples: named in both | couples: one of each | success | 
|---|---|---|---|---|---|---|
| internal | 91/144 | 40/144 | 13/144 | 21/72 | 40/72 | 30/144 |
| base_vision | 109/144 | 27/144 | 8/144 | 37/72 | 27/72 | 33/144 |
| base_vision_lm | 0/144 | 0/144 | 144/144 | 0/72 | 0/72 | 0/144 |

- Base vision encoder + internal language model performs at least as well as all-internal (mean reward per row 0.25-0.57 against 0.25-0.50).
- Base vision encoder + base language model: the arm never comes within 5 cm of either object. The expert needs internal's language model, not internal's vision encoder.
- Following the noun is weak in all-internal (named in both placements of a couple 21/72, one of each 40/72: mostly position), also on the trained pair pill_roar (3/24 against 18/24). With the base vision encoder it is higher (37/72), on the unseen banana and mug as well (27/48).
- "Both on the robot's left" is near half named / half other in both working cells.
- Side questions: every cell names the image side right (71/72 or 72/72) and gives the same side for the robot's point of view (0-2 of 36 right). None of them, the released model included, turns the view around.
- Small counts; not repeated, videos not reviewed.
