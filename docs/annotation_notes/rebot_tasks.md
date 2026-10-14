# ReBot task sheet

Generated 2026-10-14 by `lerobot.annotation.rebot.task_sheet` from the roots under `dataset.sources` of `config_rl.yaml`. Regenerate after any root changes:

```bash
uv run python -m lerobot.annotation.rebot.task_sheet config_rl.yaml lerobot/docs/annotation_notes/rebot_tasks.md
```

Use: find the task in the index, open its section, paste the YAML block into `policy:` of the inference config (`task`, `eval_subtasks`, `eval_subtask_precisions`, `eval_subtask_contacts`). The table shows the episode with the highest mean quality for that task; the other episodes of the task sit in the collapsed block under it. A step that repeats the previous one is a retry in the demonstration; collapse it when writing the script. Speed and quality are shown for orientation only; the prompt does not take them at inference.

## Legend

| precision | slack at the commit |
|---|---|
| 1 | > 10 cm slack or no external target (return to home, drop a sock in a basket) |
| 2 | 3-10 cm (cup into a basket, block into a bin) |
| 3 | 1-3 cm (straddle a cup, bottle, handle; mouth a few cm wider than the object) |
| 4 | 3-10 mm (hex bit, screw; bit into a compartment) |
| 5 | < 3 mm or a mating fit (peg, plug, bit into a one-bit slot) |

| contact | definition |
|---|---|
| top-pinch | pads close from above on a rigid object's sides or top, approach < 45 |
| side-pinch | pads close on the body from the side, approach >= 45 (wrap on a cylinder, face pinch on a book) |
| rim-pinch | one pad inside a mouth or ring, one outside its wall, from above |
| handle-grasp | pads close on a handle, bar, knob, tab or lever attached to a larger body |
| cloth-pinch | pads close on a fold, edge, corner or bunch of a deformable or thin sheet |
| push | sustained lateral contact; the object slides on the surface or swings on a hinge |
| press | contact along the normal on a control face; the object does not translate |
| strike | impulsive contact by the gripper or a held tool; the gripper keeps moving |
| pry | closed gripper wedged under an edge and lifted |
| set-down | held object lowered until it rests on a surface, then released |
| drop | pads open while the object is unsupported over the target |
| insert | held object guided into a fitted opening or over a peg; the fit does the last centimetres |
| tilt-pour | held container rotated so its contents leave it |
| tool-drag | held object pressed on a surface and moved across it |
| na | no contact made, broken or worked |

| quality | meaning |
|---|---|
| 5 | exemplary: best grip point, one clean motion |
| 4 | default: straight to the goal, small blemish at most |
| 3 | laboured but converges; worse grip point; hover past the norm |
| 2 | poor: long hover, search, one mistake |
| 1 | failed step, no recovery, or human help |

Speed: `speed_hybrid_v1` bucket 1 (slowest) to 5 (fastest), 1 s smoothed joint motion against the ReBot training distribution. Frames are indices inside the root's episode.

## Roots

| source | root | episodes | tasks | segments |
|---|---|---|---|---|
| internal_main | `outputs/rebot_all-annotated-v14` | 90 | 67 | 1221 |
| internal_additions | `outputs/rebot_main_additions-annotated-v3` | 22 | 9 | 111 |
| internal_bits_kit | `outputs/rebot_bits_kitfold_2026-10-06-annotated-v2` | 18 | 16 | 247 |
| internal_oct04 | `outputs/rebot_additions_2026-10-04-annotated-v12` | 26 | 16 | 162 |
| internal_oct07 | `outputs/rebot_teleop_2026-10-07-annotated-v1` | 5 | 5 | 55 |
| internal_oct08 | `outputs/rebot_teleop_2026-10-08-annotated-v1` | 13 | 11 | 74 |
| external_rebot_train | `outputs/external_rebot_train-annotated-v9` | 241 | 55 | 1294 |

## Task index

| id | task | source | group | episodes | steps |
|---|---|---|---|---|---|
| [T001](#t001-grasp-the-black-sock-farther-from-the-robot-and-move-it-to-the-basket) | Grasp the black sock farther from the robot and move it to the basket | internal_main | clothing | 49 | 6 |
| [T002](#t002-move-the-shirt-to-the-bin-then-put-it-in-the-basket) | Move the shirt to the bin, then put it in the basket | internal_main | clothing | 8 | 9 |
| [T003](#t003-put-a-black-sock-and-a-white-sock-in-the-basket-the-shirt-in-the-bin-then-a-black-sock-the-other-white-sock-and-the-other-black-socks-in-the-basket) | Put a black sock and a white sock in the basket, the shirt in the bin, then a black sock, the other white sock and the other black socks in the basket | internal_main | clothing | 62 | 28 |
| [T004](#t004-put-a-black-sock-and-a-white-sock-in-the-basket-the-shirt-in-the-bin-then-the-other-white-sock-and-two-black-socks-in-the-basket) | Put a black sock and a white sock in the basket, the shirt in the bin, then the other white sock and two black socks in the basket | internal_main | clothing | 57 | 22 |
| [T005](#t005-put-a-sock-in-the-basket-the-shirt-in-the-bin-and-the-other-sock-in-the-basket) | Put a sock in the basket, the shirt in the bin and the other sock in the basket | internal_main | clothing | 61 | 13 |
| [T006](#t006-put-the-beige-shirt-and-the-navy-shirt-in-the-bin) | Put the beige shirt and the navy shirt in the bin | internal_main | clothing | 48 | 10 |
| [T007](#t007-put-the-black-shirt-the-beige-shirt-and-the-navy-shirt-in-the-bin) | Put the black shirt, the beige shirt and the navy shirt in the bin | internal_main | clothing | 23 | 16 |
| [T008](#t008-put-the-black-sock-and-the-white-sock-in-the-basket) | Put the black sock and the white sock in the basket | internal_main | clothing | 82, 83, 84 | 7 |
| [T009](#t009-put-the-black-sock-in-the-basket-and-the-white-sock-in-the-bin) | Put the black sock in the basket and the white sock in the bin | internal_main | clothing | 53 | 7 |
| [T010](#t010-put-the-black-sock-in-the-bin-the-white-sock-in-the-basket-and-the-shirt-in-the-bin) | Put the black sock in the bin, the white sock in the basket and the shirt in the bin | internal_main | clothing | 4 | 12 |
| [T011](#t011-put-the-brown-shirt-the-grey-shirt-and-the-white-shirt-in-the-bin-and-the-socks-in-the-basket) | Put the brown shirt, the grey shirt and the white shirt in the bin and the socks in the basket | internal_main | clothing | 69 | 22 |
| [T012](#t012-put-the-brown-shirt-the-white-shirt-and-the-grey-shirt-in-the-bin) | Put the brown shirt, the white shirt and the grey shirt in the bin | internal_main | clothing | 25 | 16 |
| [T013](#t013-put-the-green-shirt-in-the-bin-the-sock-and-the-white-shirt-in-the-basket-and-the-navy-shirt-in-the-bin) | Put the green shirt in the bin, the sock and the white shirt in the basket, and the navy shirt in the bin | internal_main | clothing | 41 | 17 |
| [T014](#t014-put-the-grey-shirt-and-the-brown-shirt-in-the-bin) | Put the grey shirt and the brown shirt in the bin | internal_main | clothing | 18 | 22 |
| [T015](#t015-put-the-grey-shirt-and-the-navy-shirt-in-the-bin) | Put the grey shirt and the navy shirt in the bin | internal_main | clothing | 86 | 16 |
| [T016](#t016-put-the-grey-shirt-in-the-bin-two-socks-in-the-basket-the-white-shirt-in-the-bin-and-the-other-two-socks-in-the-basket) | Put the grey shirt in the bin, two socks in the basket, the white shirt in the bin and the other two socks in the basket | internal_main | clothing | 65 | 25 |
| [T017](#t017-put-the-navy-shirt-and-the-grey-shirt-in-the-bin) | Put the navy shirt and the grey shirt in the bin | internal_main | clothing | 22 | 10 |
| [T018](#t018-put-the-navy-shirt-and-the-white-shirt-in-the-bin-the-sock-nearest-the-bin-in-the-basket-the-brown-shirt-in-the-bin-and-the-other-sock-in-the-basket) | Put the navy shirt and the white shirt in the bin, the sock nearest the bin in the basket, the brown shirt in the bin and the other sock in the basket | internal_main | clothing | 1 | 37 |
| [T019](#t019-put-the-plain-white-socks-then-the-black-socks-then-the-socks-with-blue-toes-in-the-basket) | Put the plain white socks, then the black socks, then the socks with blue toes in the basket | internal_main | clothing | 39 | 25 |
| [T020](#t020-put-the-red-shirt-in-the-basket-and-the-brown-shirt-in-the-bin-then-grasp-the-black-sock-farther-from-the-robot) | Put the red shirt in the basket and the brown shirt in the bin, then grasp the black sock farther from the robot | internal_main | clothing | 50 | 27 |
| [T021](#t021-put-the-red-shirt-in-the-bin-the-striped-sock-in-the-basket-the-navy-shirt-in-the-bin-the-sock-on-the-shirt-in-the-basket-the-navy-shirt-in-the-bin-again-and-the-plain-black-sock-and-the-last-sock-in-the-basket) | Put the red shirt in the bin, the striped sock in the basket, the navy shirt in the bin, the sock on the shirt in the basket, the navy shirt in the bin again, and the plain black sock and the last sock in the basket | internal_main | clothing | 60 | 23 |
| [T022](#t022-put-the-red-shirt-the-grey-shirt-the-white-shirt-and-the-brown-shirt-in-the-bin) | Put the red shirt, the grey shirt, the white shirt and the brown shirt in the bin | internal_main | clothing | 20 | 25 |
| [T023](#t023-put-the-red-shirt-the-navy-shirt-the-black-shirt-and-the-grey-shirt-in-the-bin) | Put the red shirt, the navy shirt, the black shirt and the grey shirt in the bin | internal_main | clothing | 21 | 22 |
| [T024](#t024-put-the-shirt-in-the-bin) | Put the shirt in the bin | internal_main | clothing | 10, 12, 17, 87, 88, 89 | 4-10 |
| [T025](#t025-put-the-shirt-in-the-bin-and-the-black-socks-and-the-white-socks-in-the-basket) | Put the shirt in the bin and the black socks and the white socks in the basket | internal_main | clothing | 67 | 25 |
| [T026](#t026-put-the-shirt-in-the-bin-and-the-socks-in-the-basket) | Put the shirt in the bin and the socks in the basket | internal_main | clothing | 68 | 13 |
| [T027](#t027-put-the-shirt-in-the-bin-and-the-white-sock-and-the-black-sock-in-the-basket) | Put the shirt in the bin and the white sock and the black sock in the basket | internal_main | clothing | 6 | 10 |
| [T028](#t028-put-the-shirt-in-the-bin-and-the-white-sock-in-the-basket-then-grasp-the-black-sock) | Put the shirt in the bin and the white sock in the basket, then grasp the black sock | internal_main | clothing | 11 | 7 |
| [T029](#t029-put-the-shirt-in-the-bin-then-in-the-basket-then-grasp-the-black-sock) | Put the shirt in the bin, then in the basket, then grasp the black sock | internal_main | clothing | 52 | 20 |
| [T030](#t030-put-the-shirt-the-white-sock-and-the-black-sock-in-the-bin-then-grasp-the-white-sock-again) | Put the shirt, the white sock and the black sock in the bin, then grasp the white sock again | internal_main | clothing | 7 | 10 |
| [T031](#t031-put-the-shirts-in-the-bin) | Put the shirts in the bin | internal_main | clothing | 24 | 7 |
| [T032](#t032-put-the-sock-and-the-shirt-in-the-bin) | Put the sock and the shirt in the bin | internal_main | clothing | 16 | 10 |
| [T033](#t033-put-the-sock-and-the-shirts-in-the-basket) | Put the sock and the shirts in the basket | internal_main | clothing | 43 | 26 |
| [T034](#t034-put-the-sock-in-the-basket) | Put the sock in the basket | internal_main | clothing | 3, 13, 44 | 3-4 |
| [T035](#t035-put-the-sock-in-the-basket-and-the-shirt-in-the-bin) | Put the sock in the basket and the shirt in the bin | internal_main | clothing | 45 | 7 |
| [T036](#t036-put-the-sock-in-the-bin) | Put the sock in the bin | internal_main | clothing | 14 | 3 |
| [T037](#t037-put-the-sock-in-the-bin-and-grasp-the-navy-shirt) | Put the sock in the bin and grasp the navy shirt | internal_main | clothing | 47 | 4 |
| [T038](#t038-put-the-sock-on-the-navy-shirt-in-the-basket) | Put the sock on the navy shirt in the basket | internal_main | clothing | 46 | 4 |
| [T039](#t039-put-the-socks-in-the-basket) | Put the socks in the basket | internal_main | clothing | 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38 | 7-33 |
| [T040](#t040-put-the-socks-in-the-basket-and-the-shirt-in-the-bin) | Put the socks in the basket and the shirt in the bin | internal_main | clothing | 63 | 10 |
| [T041](#t041-put-the-socks-with-the-blue-toe-the-white-socks-and-the-black-socks-in-the-basket) | Put the socks with the blue toe, the white socks and the black socks in the basket | internal_main | clothing | 37 | 19 |
| [T042](#t042-put-the-white-shirt-and-the-grey-shirt-in-the-bin-the-black-sock-and-the-white-socks-in-the-basket-and-grasp-the-other-black-sock) | Put the white shirt and the grey shirt in the bin, the black sock and the white socks in the basket, and grasp the other black sock | internal_main | clothing | 56 | 17 |
| [T043](#t043-put-the-white-shirt-in-the-bin-the-black-sock-with-grey-stripes-a-white-sock-and-the-black-sock-on-the-navy-shirt-in-the-basket-the-navy-shirt-in-the-bin-the-other-white-sock-and-the-black-sock-on-the-grey-shirt-in-the-basket-and-the-grey-shirt-in-the-bin) | Put the white shirt in the bin, the black sock with grey stripes, a white sock and the black sock on the navy shirt in the basket, the navy shirt in the bin, the other white sock and the black sock on the grey shirt in the basket, and the grey shirt in the bin | internal_main | clothing | 58 | 28 |
| [T044](#t044-put-the-white-shirt-in-the-bin-the-black-socks-and-the-white-socks-in-the-basket-and-the-brown-shirt-in-the-bin) | Put the white shirt in the bin, the black socks and the white socks in the basket and the brown shirt in the bin | internal_main | clothing | 66 | 28 |
| [T045](#t045-put-the-white-shirt-the-grey-shirt-and-the-brown-shirt-in-the-bin) | Put the white shirt, the grey shirt and the brown shirt in the bin | internal_main | clothing | 19 | 22 |
| [T046](#t046-put-the-white-shirt-the-navy-shirt-and-the-black-shirt-in-the-bin-and-the-socks-in-the-basket) | Put the white shirt, the navy shirt and the black shirt in the bin and the socks in the basket | internal_main | clothing | 64 | 16 |
| [T047](#t047-put-the-white-sock-and-the-black-sock-in-the-basket) | Put the white sock and the black sock in the basket | internal_main | clothing | 55, 85 | 7 |
| [T048](#t048-put-the-white-sock-in-the-basket) | Put the white sock in the basket | internal_main | clothing | 2, 9 | 3-4 |
| [T049](#t049-put-the-white-sock-in-the-basket-and-the-black-sock-in-the-bin) | Put the white sock in the basket and the black sock in the bin | internal_main | clothing | 51 | 7 |
| [T050](#t050-put-the-white-sock-in-the-basket-the-shirt-in-the-bin-and-the-black-sock-in-the-basket) | Put the white sock in the basket, the shirt in the bin and the black sock in the basket | internal_main | clothing | 5 | 13 |
| [T051](#t051-put-the-white-sock-in-the-bin-the-red-shirt-in-the-basket-the-black-sock-in-the-bin-and-the-navy-shirt-in-the-basket) | Put the white sock in the bin, the red shirt in the basket, the black sock in the bin and the navy shirt in the basket | internal_main | clothing | 54 | 28 |
| [T052](#t052-put-the-white-sock-the-plain-black-sock-and-the-green-shirt-in-the-basket) | Put the white sock, the plain black sock and the green shirt in the basket | internal_main | clothing | 0 | 10 |
| [T053](#t053-put-the-white-socks-and-the-brown-shirt-in-the-bin-then-grasp-the-black-sock) | Put the white socks and the brown shirt in the bin, then grasp the black sock | internal_main | clothing | 42 | 11 |
| [T054](#t054-put-two-black-socks-then-the-four-white-socks-then-the-other-two-black-socks-in-the-basket) | Put two black socks, then the four white socks, then the other two black socks in the basket | internal_main | clothing | 40 | 25 |
| [T055](#t055-put-two-white-socks-a-black-sock-a-white-sock-the-other-black-sock-and-the-last-white-sock-in-the-basket-then-the-white-shirt-and-the-grey-shirt-in-the-bin) | Put two white socks, a black sock, a white sock, the other black sock and the last white sock in the basket, then the white shirt and the grey shirt in the bin | internal_main | clothing | 59 | 28 |
| [T056](#t056-put-the-cup-and-the-pill-bottle-in-the-basket-and-the-spray-bottle-in-the-bin) | Put the cup and the pill bottle in the basket and the spray bottle in the bin | internal_main | rigid | 76 | 10 |
| [T057](#t057-put-the-cup-and-the-spray-bottle-in-the-bin-and-the-pill-bottle-in-the-basket) | Put the cup and the spray bottle in the bin and the pill bottle in the basket | internal_main | rigid | 75 | 10 |
| [T058](#t058-put-the-cup-and-the-tape-roll-in-the-basket-and-the-spray-bottle-and-the-pill-bottle-in-the-bin) | Put the cup and the tape roll in the basket and the spray bottle and the pill bottle in the bin | internal_main | rigid | 79 | 14 |
| [T059](#t059-put-the-cup-in-the-basket) | Put the cup in the basket | internal_main | rigid | 15 | 4 |
| [T060](#t060-put-the-cup-in-the-basket-and-the-pill-bottle-and-the-tape-roll-in-the-bin) | Put the cup in the basket and the pill bottle and the tape roll in the bin | internal_main | rigid | 80 | 9 |
| [T061](#t061-put-the-cup-the-pill-bottle-and-the-spray-bottle-in-the-basket) | Put the cup, the pill bottle and the spray bottle in the basket | internal_main | rigid | 71 | 10 |
| [T062](#t062-put-the-cup-the-spray-bottle-and-the-pill-bottle-in-the-basket) | Put the cup, the spray bottle and the pill bottle in the basket | internal_main | rigid | 72 | 10 |
| [T063](#t063-put-the-pill-bottle-the-cup-and-the-spray-bottle-in-the-basket) | Put the pill bottle, the cup and the spray bottle in the basket | internal_main | rigid | 73 | 10 |
| [T064](#t064-put-the-pill-bottle-the-spray-bottle-and-the-cup-in-the-basket) | Put the pill bottle, the spray bottle and the cup in the basket | internal_main | rigid | 70, 78 | 10 |
| [T065](#t065-put-the-spray-bottle-and-the-cup-in-the-basket-and-the-pill-bottle-in-the-bin) | Put the spray bottle and the cup in the basket and the pill bottle in the bin | internal_main | rigid | 74 | 10 |
| [T066](#t066-put-the-spray-bottle-the-cup-and-the-tape-roll-in-the-bin) | Put the spray bottle, the cup and the tape roll in the bin | internal_main | rigid | 81 | 10 |
| [T067](#t067-put-the-tape-roll-in-the-bin-the-cup-in-the-basket-and-the-pill-bottle-and-the-spray-bottle-in-the-bin) | Put the tape roll in the bin, the cup in the basket, and the pill bottle and the spray bottle in the bin | internal_main | rigid | 77 | 13 |
| [T068](#t068-put-socks-in-basket) | Put socks in basket | internal_additions | clothing | 5 | 22 |
| [T069](#t069-pick-up-the-green-eraser-and-place-it-in-the-brown-basket) | Pick up the green eraser and place it in the brown basket. | internal_additions | rigid | 6, 7, 8, 9, 10, 11 | 4 |
| [T070](#t070-pick-up-the-green-eraser-and-place-it-in-the-gray-basket) | Pick up the green eraser and place it in the gray basket. | internal_additions | rigid | 12, 13, 14 | 4 |
| [T071](#t071-put-a-sugar-cube-in-the-cup) | Put a sugar cube in the cup | internal_additions | rigid | 15, 16, 17, 18, 19, 20, 21 | 2-4 |
| [T072](#t072-put-the-pill-bottle-and-the-deck-of-cards-in-the-basket) | Put the pill bottle and the deck of cards in the basket | internal_additions | rigid | 4 | 7 |
| [T073](#t073-put-the-sock-in-the-bin-and-the-cup-in-the-basket) | Put the sock in the bin and the cup in the basket | internal_additions | rigid | 2 | 6 |
| [T074](#t074-put-the-spray-bottle-and-the-pill-bottle-in-the-basket) | Put the spray bottle and the pill bottle in the basket | internal_additions | rigid | 1 | 7 |
| [T075](#t075-put-the-tape-roll-in-the-basket) | Put the tape roll in the basket | internal_additions | rigid | 0 | 3 |
| [T076](#t076-put-the-tape-roll-in-the-bin) | Put the tape roll in the bin | internal_additions | rigid | 3 | 4 |
| [T077](#t077-put-a-bit-in-the-open-lid-of-the-box) | Put a bit in the open lid of the box | internal_bits_kit | bits_clippers | 9 | 5 |
| [T078](#t078-put-five-bits-in-the-empty-compartment-of-the-box-and-reach-for-the-last-bit) | Put five bits in the empty compartment of the box and reach for the last bit | internal_bits_kit | bits_clippers | 2 | 16 |
| [T079](#t079-put-nine-bits-in-the-empty-compartment-of-the-box) | Put nine bits in the empty compartment of the box | internal_bits_kit | bits_clippers | 4 | 28 |
| [T080](#t080-put-seven-bits-from-the-table-in-the-compartment-with-the-gray-bits) | Put seven bits from the table in the compartment with the gray bits | internal_bits_kit | bits_clippers | 7 | 22 |
| [T081](#t081-put-seven-bits-in-the-compartment-with-the-bits) | Put seven bits in the compartment with the bits | internal_bits_kit | bits_clippers | 5 | 23 |
| [T082](#t082-put-the-bit-in-the-compartment-with-the-blue-bits) | Put the bit in the compartment with the blue bits | internal_bits_kit | bits_clippers | 10 | 3 |
| [T083](#t083-put-the-bits-in-the-empty-compartment-of-the-box) | Put the bits in the empty compartment of the box | internal_bits_kit | bits_clippers | 0, 3 | 13-19 |
| [T084](#t084-put-the-bits-in-the-empty-compartment-of-the-box-the-sock-in-the-bin-and-the-last-bit-in-the-box) | Put the bits in the empty compartment of the box, the sock in the bin and the last bit in the box | internal_bits_kit | bits_clippers | 8 | 34 |
| [T085](#t085-put-the-blue-bit-in-the-compartment-with-the-blue-bits-and-the-red-bit-in-the-compartment-with-the-yellow-bits) | Put the blue bit in the compartment with the blue bits and the red bit in the compartment with the yellow bits | internal_bits_kit | bits_clippers | 12 | 9 |
| [T086](#t086-put-the-blue-bit-in-the-compartment-with-the-blue-bits-the-yellow-bit-in-the-compartment-with-the-yellow-bits-and-the-red-bit-in-the-compartment-with-the-gray-bits) | Put the blue bit in the compartment with the blue bits, the yellow bit in the compartment with the yellow bits and the red bit in the compartment with the gray bits | internal_bits_kit | bits_clippers | 11 | 12 |
| [T087](#t087-put-the-four-bits-in-the-empty-compartment-of-the-component-box) | Put the four bits in the empty compartment of the component box | internal_bits_kit | bits_clippers | 1 | 13 |
| [T088](#t088-put-two-bits-in-the-empty-compartment-of-the-box-and-grasp-a-third-bit) | Put two bits in the empty compartment of the box and grasp a third bit | internal_bits_kit | bits_clippers | 6 | 7 |
| [T089](#t089-close-the-headphone-case-fold-the-bit-kit-closed-and-push-the-bit-kit-next-to-the-headphone-case) | Close the headphone case, fold the bit kit closed and push the bit kit next to the headphone case | internal_bits_kit | cases_kit_box_push | 17 | 10 |
| [T090](#t090-fold-the-flap-with-the-bits-and-then-the-flap-with-the-white-card-to-close-the-bit-kit-move-the-headphone-case-next-to-the-kit-and-close-its-lid) | Fold the flap with the bits and then the flap with the white card to close the bit kit, move the headphone case next to the kit and close its lid | internal_bits_kit | cases_kit_box_push | 13, 14 | 10-13 |
| [T091](#t091-fold-the-flap-with-the-bits-over-the-middle-of-the-bit-kit) | Fold the flap with the bits over the middle of the bit kit | internal_bits_kit | cases_kit_box_push | 15 | 2 |
| [T092](#t092-fold-the-flap-with-the-bits-over-the-middle-of-the-kit-fold-the-flap-with-the-white-card-over-it-press-the-kit-flat-and-close-the-headphone-case) | Fold the flap with the bits over the middle of the kit, fold the flap with the white card over it, press the kit flat and close the headphone case | internal_bits_kit | cases_kit_box_push | 16 | 8 |
| [T093](#t093-close-the-headphone-case-lid) | Close the headphone case lid | internal_oct04 | cases_kit_box_push | 0, 2, 3 | 3-5 |
| [T094](#t094-close-the-headphone-case-lid-and-put-the-spray-bottle-in-the-basket) | Close the headphone case lid and put the spray bottle in the basket | internal_oct04 | cases_kit_box_push | 1 | 8 |
| [T095](#t095-close-the-headphone-case-lid-close-the-box-lid-and-lay-the-box-flat) | Close the headphone case lid, close the box lid and lay the box flat | internal_oct04 | cases_kit_box_push | 4 | 8 |
| [T096](#t096-lift-the-box-lid) | Lift the box lid | internal_oct04 | cases_kit_box_push | 22, 23 | 1 |
| [T097](#t097-push-the-books-off-the-box) | Push the books off the box | internal_oct04 | cases_kit_box_push | 19, 21 | 2 |
| [T098](#t098-push-the-two-books-off-the-box-toward-the-left-edge-of-the-table-then-lift-the-flap-with-the-shipping-label-and-push-it-open) | Push the two books off the box toward the left edge of the table, then lift the flap with the shipping label and push it open | internal_oct04 | cases_kit_box_push | 20 | 4 |
| [T099](#t099-put-the-wall-plug-in-the-compartment-with-the-yellow-wall-plugs-push-the-cap-off-the-pill-bottle-pour-the-pills-into-the-box-and-put-the-pill-bottle-on-the-table) | Put the wall plug in the compartment with the yellow wall plugs, push the cap off the pill bottle, pour the pills into the box and put the pill bottle on the table | internal_oct04 | pills_caps | 13 | 13 |
| [T100](#t100-take-the-cap-off-the-pill-bottle-and-put-it-on-the-table-pour-the-pills-into-the-box-and-put-the-pill-bottle-on-the-table) | Take the cap off the pill bottle and put it on the table, pour the pills into the box and put the pill bottle on the table | internal_oct04 | pills_caps | 11 | 10 |
| [T101](#t101-take-the-cap-off-the-pill-bottle-pour-the-pills-into-the-box-and-put-the-pill-bottle-on-the-table) | Take the cap off the pill bottle, pour the pills into the box and put the pill bottle on the table | internal_oct04 | pills_caps | 10, 12, 15, 16, 17 | 7 |
| [T102](#t102-take-the-cap-off-the-pill-bottle-pour-the-pills-into-the-cup-and-put-the-pill-bottle-on-the-table) | Take the cap off the pill bottle, pour the pills into the cup and put the pill bottle on the table | internal_oct04 | pills_caps | 8 | 7 |
| [T103](#t103-take-the-cap-off-the-pill-bottle-pour-the-pills-into-the-cup-put-the-pill-bottle-on-the-table-and-put-the-cup-on-the-towel) | Take the cap off the pill bottle, pour the pills into the cup, put the pill bottle on the table and put the cup on the towel | internal_oct04 | pills_caps | 6, 7 | 13 |
| [T104](#t104-take-the-cap-off-the-pill-bottle-pour-the-pills-into-the-cup-put-the-pill-bottle-on-the-towel-and-put-the-cup-on-the-towel) | Take the cap off the pill bottle, pour the pills into the cup, put the pill bottle on the towel and put the cup on the towel | internal_oct04 | pills_caps | 9 | 13 |
| [T105](#t105-take-the-cap-off-the-pill-bottle-put-the-cap-on-the-table-pour-the-pills-into-the-box-and-put-the-pill-bottle-on-the-table) | Take the cap off the pill bottle, put the cap on the table, pour the pills into the box and put the pill bottle on the table | internal_oct04 | pills_caps | 14 | 9 |
| [T106](#t106-grasp-the-pill-bottle) | Grasp the pill bottle | internal_oct04 | rigid | 24, 25 | 1 |
| [T107](#t107-open-the-box-and-put-the-pill-bottle-in-the-box) | Open the box and put the pill bottle in the box | internal_oct04 | rigid | 5 | 6 |
| [T108](#t108-put-the-dental-floss-on-the-cutting-board) | Put the dental floss on the cutting board | internal_oct04 | rigid | 18 | 4 |
| [T109](#t109-put-the-large-nail-clipper-and-the-small-nail-clipper-in-the-clipper-case) | Put the large nail clipper and the small nail clipper in the clipper case | internal_oct07 | bits_clippers | 0 | 9 |
| [T110](#t110-put-the-large-nail-clipper-and-the-small-nail-clipper-in-the-clipper-case-fold-the-bit-kit-shut-and-push-the-headphone-case-off-the-box-toward-the-table-edge) | Put the large nail clipper and the small nail clipper in the clipper case, fold the bit kit shut and push the headphone case off the box toward the table edge | internal_oct07 | bits_clippers | 3 | 12 |
| [T111](#t111-put-the-large-nail-clipper-and-the-small-nail-clipper-in-the-clipper-case-pour-the-pills-from-the-open-pill-bottle-into-the-lid-of-the-clipper-case-and-set-the-open-pill-bottle-back-on-the-table) | Put the large nail clipper and the small nail clipper in the clipper case, pour the pills from the open pill bottle into the lid of the clipper case and set the open pill bottle back on the table | internal_oct07 | bits_clippers | 4 | 13 |
| [T112](#t112-put-the-small-nail-clipper-and-the-large-nail-clipper-in-the-clipper-case) | Put the small nail clipper and the large nail clipper in the clipper case | internal_oct07 | bits_clippers | 1 | 8 |
| [T113](#t113-put-the-small-nail-clipper-and-the-large-nail-clipper-in-the-clipper-case-close-the-headphone-case-lid-and-put-the-bit-in-the-compartment-with-the-gray-bits) | Put the small nail clipper and the large nail clipper in the clipper case, close the headphone case lid and put the bit in the compartment with the gray bits | internal_oct07 | bits_clippers | 2 | 13 |
| [T114](#t114-grasp-the-shaker-bottle) | Grasp the shaker bottle | internal_oct08 | bottles_lids | 1 | 1 |
| [T115](#t115-stand-the-pill-bottle-upright) | Stand the pill bottle upright | internal_oct08 | bottles_lids | 10 | 7 |
| [T116](#t116-stand-the-pill-bottle-upright-and-stand-the-water-bottle-upright) | Stand the pill bottle upright and stand the water bottle upright | internal_oct08 | bottles_lids | 11 | 5 |
| [T117](#t117-stand-the-shaker-bottle-upright) | Stand the shaker bottle upright | internal_oct08 | bottles_lids | 5 | 3 |
| [T118](#t118-stand-the-shaker-bottle-upright-and-grasp-the-lid) | Stand the shaker bottle upright and grasp the lid | internal_oct08 | bottles_lids | 2 | 3 |
| [T119](#t119-stand-the-shaker-bottle-upright-and-move-the-lid-to-the-shaker-bottle) | Stand the shaker bottle upright and move the lid to the shaker bottle | internal_oct08 | bottles_lids | 3 | 4 |
| [T120](#t120-stand-the-shaker-bottle-upright-and-put-the-lid-on-the-shaker-bottle) | Stand the shaker bottle upright and put the lid on the shaker bottle | internal_oct08 | bottles_lids | 0, 4 | 6-8 |
| [T121](#t121-stand-the-water-bottle-upright-and-close-the-headphone-case-lid) | Stand the water bottle upright and close the headphone case lid | internal_oct08 | bottles_lids | 8 | 5 |
| [T122](#t122-stand-the-water-bottle-upright-and-fold-the-flap-with-the-white-card-over-the-other-half-of-the-bit-kit) | Stand the water bottle upright and fold the flap with the white card over the other half of the bit kit | internal_oct08 | bottles_lids | 6, 7 | 4-13 |
| [T123](#t123-stand-the-water-bottle-upright-and-stand-the-pill-bottle-upright) | Stand the water bottle upright and stand the pill bottle upright | internal_oct08 | bottles_lids | 12 | 5 |
| [T124](#t124-stand-the-water-bottle-upright-close-the-headphone-case-lid-and-put-the-nail-clipper-in-the-clipper-case) | Stand the water bottle upright, close the headphone case lid and put the nail clipper in the clipper case | internal_oct08 | bottles_lids | 9 | 10 |
| [T125](#t125-fold-the-towel-in-half-twice-straighten-it-turn-it-to-lie-left-to-right-move-it-to-the-middle-of-the-mat-and-push-it-into-the-far-left-corner-of-the-mat) | Fold the towel in half twice, straighten it, turn it to lie left to right, move it to the middle of the mat and push it into the far left corner of the mat | external_rebot_train | external_rebot | 179 | 13 |
| [T126](#t126-fold-the-towel-in-half-twice-turn-it-to-lie-left-to-right-fold-it-in-half-again-and-push-it-into-the-far-left-corner-of-the-mat) | Fold the towel in half twice, turn it to lie left to right, fold it in half again and push it into the far left corner of the mat | external_rebot_train | external_rebot | 183 | 9 |
| [T127](#t127-fold-the-towel-in-half-pull-its-near-left-corner-to-the-left-fold-it-in-half-again-turn-it-to-lie-left-to-right-and-put-it-in-the-far-left-corner-of-the-mat) | Fold the towel in half, pull its near left corner to the left, fold it in half again, turn it to lie left to right and put it in the far left corner of the mat | external_rebot_train | external_rebot | 182 | 11 |
| [T128](#t128-fold-the-towel-in-half-straighten-it-pull-it-to-the-far-edge-slide-it-to-the-left-and-put-it-in-the-far-left-corner-of-the-mat) | Fold the towel in half, straighten it, pull it to the far edge, slide it to the left and put it in the far left corner of the mat | external_rebot_train | external_rebot | 180 | 13 |
| [T129](#t129-fold-the-towel-into-a-square-and-move-it-to-the-far-left-corner-of-the-mat) | Fold the towel into a square and move it to the far left corner of the mat | external_rebot_train | external_rebot | 181 | 10 |
| [T130](#t130-grasp-the-mallet-nearest-the-robot-hit-the-puck-back-to-the-opponent-and-release-the-mallet-on-the-table) | Grasp the mallet nearest the robot, hit the puck back to the opponent and release the mallet on the table | external_rebot_train | external_rebot | 211, 212, 213, 214, 215, 216, 217, 218, 219, 220 | 4-5 |
| [T131](#t131-move-the-block-to-the-other-cross) | Move the block to the other cross | external_rebot_train | external_rebot | 3, 4 | 4 |
| [T132](#t132-open-the-lower-drawer-put-the-green-eraser-in-the-lower-drawer-and-close-the-lower-drawer) | Open the lower drawer, put the green eraser in the lower drawer and close the lower drawer | external_rebot_train | external_rebot | 164, 165, 166 | 7 |
| [T133](#t133-open-the-lower-drawer-put-the-yellow-eraser-in-the-lower-drawer-and-close-the-lower-drawer) | Open the lower drawer, put the yellow eraser in the lower drawer and close the lower drawer | external_rebot_train | external_rebot | 159, 160 | 7 |
| [T134](#t134-open-the-upper-drawer-put-the-green-eraser-in-the-upper-drawer-and-close-the-upper-drawer) | Open the upper drawer, put the green eraser in the upper drawer and close the upper drawer | external_rebot_train | external_rebot | 161, 162, 163 | 7 |
| [T135](#t135-open-the-upper-drawer-put-the-yellow-eraser-in-the-upper-drawer-and-close-the-upper-drawer) | Open the upper drawer, put the yellow eraser in the upper drawer and close the upper drawer | external_rebot_train | external_rebot | 157, 158 | 7 |
| [T136](#t136-plug-the-charger-into-the-power-strip) | Plug the charger into the power strip | external_rebot_train | external_rebot | 231, 232, 233, 234, 235, 236, 237, 238, 239, 240 | 3-4 |
| [T137](#t137-pour-the-egg-into-the-pan-and-put-the-bowl-back-on-the-mat) | Pour the egg into the pan and put the bowl back on the mat | external_rebot_train | external_rebot | 23, 24 | 5-6 |
| [T138](#t138-pour-the-ham-the-egg-and-the-noodles-into-the-pan-and-put-each-bowl-back-on-the-mat) | Pour the ham, the egg and the noodles into the pan and put each bowl back on the mat | external_rebot_train | external_rebot | 29, 30 | 16 |
| [T139](#t139-pour-the-noodles-into-the-pan-and-put-the-bowl-back-on-the-mat) | Pour the noodles into the pan and put the bowl back on the mat | external_rebot_train | external_rebot | 25, 26, 31, 32 | 5-6 |
| [T140](#t140-pour-the-oil-into-the-pan-and-put-the-bowl-back-on-the-mat) | Pour the oil into the pan and put the bowl back on the mat | external_rebot_train | external_rebot | 27, 28 | 6 |
| [T141](#t141-push-the-t-block-onto-the-black-t) | Push the T block onto the black T | external_rebot_train | external_rebot | 107, 108, 109 | 1 |
| [T142](#t142-push-the-t-block-onto-the-green-t) | Push the T block onto the green T | external_rebot_train | external_rebot | 100, 101, 102, 103, 104, 105, 106 | 1 |
| [T143](#t143-put-a-fig-bar-in-each-tray-then-an-apple-in-each-tray) | Put a fig bar in each tray, then an apple in each tray | external_rebot_train | external_rebot | 184, 185, 186, 187, 188, 189, 190, 191, 192, 193 | 13-14 |
| [T144](#t144-put-the-t-block-on-the-green-t) | Put the T block on the green T | external_rebot_train | external_rebot | 110, 111, 112, 113, 114, 115, 116, 117, 118, 119 | 3-4 |
| [T145](#t145-put-the-black-disk-on-the-red-plate) | Put the black disk on the red plate | external_rebot_train | external_rebot | 61, 62, 64, 68 | 4 |
| [T146](#t146-put-the-blue-cube-in-the-tray) | Put the blue cube in the tray | external_rebot_train | external_rebot | 145, 146 | 3 |
| [T147](#t147-put-the-blue-piece-the-yellow-piece-and-the-orange-piece-on-the-peg) | Put the blue piece, the yellow piece and the orange piece on the peg | external_rebot_train | external_rebot | 81, 82, 83, 84, 85, 86, 87, 88, 89 | 10-17 |
| [T148](#t148-put-the-bolt-on-the-green-pad) | Put the bolt on the green pad | external_rebot_train | external_rebot | 33, 34, 35, 36, 37, 38, 39, 40, 41, 42 | 4 |
| [T149](#t149-put-the-box-on-the-plate) | Put the box on the plate | external_rebot_train | external_rebot | 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 | 4-6 |
| [T150](#t150-put-the-can-in-the-small-taped-area) | Put the can in the small taped area | external_rebot_train | external_rebot | 5, 6, 7 | 4 |
| [T151](#t151-put-the-charger-in-the-bin) | Put the charger in the bin | external_rebot_train | external_rebot | 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 | 3-4 |
| [T152](#t152-put-the-crayfish-in-the-box) | Put the crayfish in the box | external_rebot_train | external_rebot | 221, 222, 223, 224, 225, 226, 227, 228, 229, 230 | 4 |
| [T153](#t153-put-the-cube-in-the-box) | Put the cube in the box | external_rebot_train | external_rebot | 177, 178 | 4 |
| [T154](#t154-put-the-cube-on-the-box) | Put the cube on the box | external_rebot_train | external_rebot | 0, 1, 2 | 4 |
| [T155](#t155-put-the-disk-in-the-box) | Put the disk in the box | external_rebot_train | external_rebot | 53, 54, 55, 56, 57, 58, 59, 60 | 3 |
| [T156](#t156-put-the-disk-on-the-blue-plate) | Put the disk on the blue plate | external_rebot_train | external_rebot | 65, 66 | 4 |
| [T157](#t157-put-the-microcontroller-farther-from-the-box-in-the-box) | Put the microcontroller farther from the box in the box | external_rebot_train | external_rebot | 73, 80 | 4 |
| [T158](#t158-put-the-microcontroller-farthest-from-the-box-in-the-box) | Put the microcontroller farthest from the box in the box | external_rebot_train | external_rebot | 76 | 4 |
| [T159](#t159-put-the-microcontroller-in-the-box) | Put the microcontroller in the box | external_rebot_train | external_rebot | 75, 77 | 4 |
| [T160](#t160-put-the-microcontroller-nearest-the-box-in-the-box) | Put the microcontroller nearest the box in the box | external_rebot_train | external_rebot | 71, 72, 74, 79 | 4 |
| [T161](#t161-put-the-microcontroller-nearest-the-far-end-of-the-box-in-the-box) | Put the microcontroller nearest the far end of the box in the box | external_rebot_train | external_rebot | 78 | 4 |
| [T162](#t162-put-the-motor-in-the-box) | Put the motor in the box | external_rebot_train | external_rebot | 201, 202, 203, 204, 205, 206, 207, 208, 209, 210 | 4 |
| [T163](#t163-put-the-paper-ball-on-the-table) | Put the paper ball on the table | external_rebot_train | external_rebot | 43, 44, 45, 46, 47, 48, 49, 50, 51, 52 | 3-4 |
| [T164](#t164-put-the-paper-the-packet-farthest-from-the-trash-can-the-cork-and-the-other-packet-in-the-trash-can) | Put the paper, the packet farthest from the trash can, the cork and the other packet in the trash can | external_rebot_train | external_rebot | 194 | 13 |
| [T165](#t165-put-the-paper-the-packet-the-cork-and-the-tissue-in-the-trash-can) | Put the paper, the packet, the cork and the tissue in the trash can | external_rebot_train | external_rebot | 195, 196, 197, 198, 199, 200 | 13 |
| [T166](#t166-put-the-purple-box-on-the-plate) | Put the purple box on the plate | external_rebot_train | external_rebot | 11, 12 | 3-4 |
| [T167](#t167-put-the-red-can-in-the-small-taped-area) | Put the red can in the small taped area | external_rebot_train | external_rebot | 8, 9, 10 | 4 |
| [T168](#t168-put-the-red-die-in-the-bin) | Put the red die in the bin | external_rebot_train | external_rebot | 167, 168, 169, 170, 171, 172, 173, 174, 175, 176 | 4 |
| [T169](#t169-put-the-tea-bag-in-the-cup) | Put the tea bag in the cup | external_rebot_train | external_rebot | 135, 136, 137, 138, 139, 140, 141, 142, 143, 144 | 3 |
| [T170](#t170-put-the-wooden-disk-on-the-blue-plate) | Put the wooden disk on the blue plate | external_rebot_train | external_rebot | 63, 67, 69, 70 | 4 |
| [T171](#t171-shake-out-the-towel-spread-it-flat-fold-it-in-half-three-times-and-press-it-flat) | Shake out the towel, spread it flat, fold it in half three times and press it flat | external_rebot_train | external_rebot | 132 | 13 |
| [T172](#t172-shake-out-the-towel-spread-it-flat-fold-it-in-half-twice-and-slide-it-toward-the-robot) | Shake out the towel, spread it flat, fold it in half twice and slide it toward the robot | external_rebot_train | external_rebot | 134 | 14 |
| [T173](#t173-shake-out-the-towel-spread-it-flat-fold-it-into-a-triangle-and-then-smaller-and-press-it-flat) | Shake out the towel, spread it flat, fold it into a triangle and then smaller, and press it flat | external_rebot_train | external_rebot | 130 | 18 |
| [T174](#t174-shake-out-the-towel-spread-it-flat-fold-it-into-a-triangle-slide-it-toward-the-robot-lay-it-flat-fold-it-in-half-and-press-it-flat) | Shake out the towel, spread it flat, fold it into a triangle, slide it toward the robot, lay it flat, fold it in half and press it flat | external_rebot_train | external_rebot | 133 | 19 |
| [T175](#t175-shake-the-hand) | Shake the hand | external_rebot_train | external_rebot | 147, 148, 149, 151, 152 | 3 |
| [T176](#t176-shake-the-hand-three-times) | Shake the hand three times | external_rebot_train | external_rebot | 155 | 7 |
| [T177](#t177-shake-the-hand-twice) | Shake the hand twice | external_rebot_train | external_rebot | 150, 153, 154, 156 | 5 |
| [T178](#t178-spread-the-towel-flat-fold-it-spread-it-out-again-fold-it-into-a-triangle-and-then-smaller-and-press-it-flat) | Spread the towel flat, fold it, spread it out again, fold it into a triangle and then smaller, and press it flat | external_rebot_train | external_rebot | 131 | 40 |
| [T179](#t179-stack-the-green-tape-roll-on-the-white-tape-roll) | Stack the green tape roll on the white tape roll | external_rebot_train | external_rebot | 120, 121, 122, 123, 124, 125, 126, 127, 128, 129 | 3-4 |

## Tasks

### T001 Grasp the black sock farther from the robot and move it to the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 49. Table: episode 49 (1374 frames, mean quality 1.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock farther from the robot | 1 | cloth-pinch | 3 | 4 | 0-366 |
| 2 | move the black sock to the basket | 1 | na | 2 | 1 | 366-459 |
| 3 | grasp the black sock farther from the robot | 1 | cloth-pinch | 2 | 1 | 459-651 |
| 4 | move the black sock to the basket | 1 | na | 2 | 1 | 651-756 |
| 5 | grasp the black sock farther from the robot | 1 | cloth-pinch | 2 | 1 | 756-1248 |
| 6 | move the black sock to the basket | 1 | na | 2 | 3 | 1248-1374 |

```yaml
task: Grasp the black sock farther from the robot and move it to the basket
eval_subtasks:
- grasp the black sock farther from the robot
- move the black sock to the basket
- grasp the black sock farther from the robot
- move the black sock to the basket
- grasp the black sock farther from the robot
- move the black sock to the basket
eval_subtask_precisions: [1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, cloth-pinch, na, cloth-pinch, na]
```

### T002 Move the shirt to the bin, then put it in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 8. Table: episode 8 (2634 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 4 | 3 | 0-522 |
| 2 | move the shirt to the bin | 1 | na | 4 | 3 | 522-600 |
| 3 | release the shirt in the bin | 1 | drop | 3 | 2 | 600-741 |
| 4 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 741-1032 |
| 5 | move the shirt to the basket | 1 | na | 3 | 3 | 1032-1365 |
| 6 | release the shirt in the basket | 1 | drop | 3 | 3 | 1365-1612 |
| 7 | grasp the shirt | 1 | cloth-pinch | 4 | 3 | 1612-2210 |
| 8 | move the shirt to the basket | 1 | na | 3 | 4 | 2210-2490 |
| 9 | release the shirt in the basket | 1 | drop | 4 | 4 | 2490-2634 |

```yaml
task: Move the shirt to the bin, then put it in the basket
eval_subtasks:
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt
- move the shirt to the basket
- release the shirt in the basket
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop]
```

### T003 Put a black sock and a white sock in the basket, the shirt in the bin, then a black sock, the other white sock and the other black socks in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 62. Table: episode 62 (6183 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock on the shirt | 1 | cloth-pinch | 4 | 5 | 0-307 |
| 2 | move the black sock to the basket | 1 | na | 3 | 5 | 307-567 |
| 3 | release the black sock in the basket | 1 | drop | 5 | 4 | 567-612 |
| 4 | grasp the white sock nearest the basket | 1 | cloth-pinch | 3 | 3 | 612-1024 |
| 5 | move the white sock to the basket | 1 | na | 3 | 3 | 1024-1249 |
| 6 | release the white sock in the basket | 1 | drop | 4 | 4 | 1249-1303 |
| 7 | grasp the shirt | 1 | cloth-pinch | 3 | 5 | 1303-1641 |
| 8 | move the shirt to the bin | 1 | na | 3 | 3 | 1641-1989 |
| 9 | release the shirt in the bin | 1 | drop | 3 | 3 | 1989-2064 |
| 10 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 2064-2333 |
| 11 | move the shirt to the bin | 1 | na | 3 | 4 | 2333-2466 |
| 12 | release the shirt in the bin | 1 | drop | 3 | 4 | 2466-2585 |
| 13 | grasp the black sock nearest the bin | 1 | cloth-pinch | 2 | 2 | 2585-3005 |
| 14 | move the black sock to the basket | 1 | na | 4 | 4 | 3005-3258 |
| 15 | release the black sock in the basket | 1 | drop | 3 | 4 | 3258-3339 |
| 16 | grasp the white sock | 1 | cloth-pinch | 3 | 3 | 3339-3633 |
| 17 | move the white sock to the basket | 1 | na | 4 | 3 | 3633-3840 |
| 18 | release the white sock in the basket | 1 | drop | 3 | 4 | 3840-3897 |
| 19 | grasp the sock nearest the basket | 1 | cloth-pinch | 3 | 4 | 3897-4158 |
| 20 | move the sock to the basket | 1 | na | 4 | 4 | 4158-4330 |
| 21 | release the sock in the basket | 1 | drop | 3 | 4 | 4330-4416 |
| 22 | grasp the sock with grey stripes | 1 | cloth-pinch | 4 | 4 | 4416-4711 |
| 23 | move the sock with grey stripes to the basket | 1 | na | 5 | 4 | 4711-4888 |
| 24 | release the sock with grey stripes in the basket | 1 | drop | 2 | 4 | 4888-4963 |
| 25 | grasp the sock | 1 | cloth-pinch | 2 | 1 | 4963-5673 |
| 26 | move the sock to the basket | 1 | na | 5 | 4 | 5673-5905 |
| 27 | release the sock in the basket | 1 | drop | 3 | 4 | 5905-5970 |
| 28 | return to home | 1 | na | 5 | 4 | 5970-6183 |

```yaml
task: Put a black sock and a white sock in the basket, the shirt in the bin, then a black sock, the other white sock and the other black socks in the basket
eval_subtasks:
- grasp the black sock on the shirt
- move the black sock to the basket
- release the black sock in the basket
- grasp the white sock nearest the basket
- move the white sock to the basket
- release the white sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the black sock nearest the bin
- move the black sock to the basket
- release the black sock in the basket
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock nearest the basket
- move the sock to the basket
- release the sock in the basket
- grasp the sock with grey stripes
- move the sock with grey stripes to the basket
- release the sock with grey stripes in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T004 Put a black sock and a white sock in the basket, the shirt in the bin, then the other white sock and two black socks in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 57. Table: episode 57 (5037 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock on the shirt | 1 | cloth-pinch | 4 | 5 | 0-303 |
| 2 | move the black sock to the basket | 1 | na | 4 | 5 | 303-504 |
| 3 | release the black sock in the basket | 1 | drop | 4 | 4 | 504-549 |
| 4 | grasp the white sock in the middle of the shirt | 1 | cloth-pinch | 3 | 3 | 549-960 |
| 5 | move the white sock to the basket | 1 | na | 4 | 3 | 960-1236 |
| 6 | release the white sock in the basket | 1 | drop | 2 | 4 | 1236-1317 |
| 7 | grasp the shirt | 1 | cloth-pinch | 2 | 4 | 1317-1665 |
| 8 | move the shirt to the bin | 1 | na | 3 | 3 | 1665-1920 |
| 9 | release the shirt in the bin | 1 | drop | 2 | 3 | 1920-2040 |
| 10 | grasp the shirt | 1 | cloth-pinch | 4 | 3 | 2040-2298 |
| 11 | move the shirt to the bin | 1 | na | 4 | 5 | 2298-2457 |
| 12 | release the shirt in the bin | 1 | drop | 3 | 5 | 2457-2550 |
| 13 | grasp the white sock | 1 | cloth-pinch | 2 | 3 | 2550-2793 |
| 14 | move the white sock to the basket | 1 | na | 4 | 3 | 2793-3066 |
| 15 | release the white sock in the basket | 1 | drop | 2 | 4 | 3066-3123 |
| 16 | grasp the black sock with grey stripes | 1 | cloth-pinch | 1 | 1 | 3123-3918 |
| 17 | move the black sock with grey stripes to the basket | 1 | na | 4 | 3 | 3918-4136 |
| 18 | release the black sock with grey stripes in the basket | 1 | drop | 2 | 4 | 4136-4184 |
| 19 | grasp the sock | 1 | cloth-pinch | 4 | 5 | 4184-4500 |
| 20 | move the sock to the basket | 1 | na | 4 | 5 | 4500-4728 |
| 21 | release the sock in the basket | 1 | drop | 2 | 5 | 4728-4779 |
| 22 | return to home | 1 | na | 5 | 5 | 4779-5037 |

```yaml
task: Put a black sock and a white sock in the basket, the shirt in the bin, then the other white sock and two black socks in the basket
eval_subtasks:
- grasp the black sock on the shirt
- move the black sock to the basket
- release the black sock in the basket
- grasp the white sock in the middle of the shirt
- move the white sock to the basket
- release the white sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the black sock with grey stripes
- move the black sock with grey stripes to the basket
- release the black sock with grey stripes in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T005 Put a sock in the basket, the shirt in the bin and the other sock in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 61. Table: episode 61 (2541 frames, mean quality 3.3).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock in the middle of the shirt | 1 | cloth-pinch | 4 | 5 | 0-347 |
| 2 | move the sock to the basket | 1 | na | 4 | 4 | 347-600 |
| 3 | release the sock in the basket | 1 | drop | 2 | 4 | 600-684 |
| 4 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 684-1038 |
| 5 | move the shirt to the bin | 1 | na | 3 | 2 | 1038-1119 |
| 6 | release the shirt in the bin | 1 | set-down | 2 | 2 | 1119-1149 |
| 7 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 1149-1304 |
| 8 | move the shirt to the bin | 1 | na | 4 | 3 | 1304-1440 |
| 9 | release the shirt in the bin | 1 | drop | 4 | 4 | 1440-1605 |
| 10 | grasp the sock | 1 | cloth-pinch | 3 | 3 | 1605-2022 |
| 11 | move the sock to the basket | 1 | na | 3 | 3 | 2022-2214 |
| 12 | release the sock in the basket | 1 | drop | 5 | 4 | 2214-2307 |
| 13 | return to home | 1 | na | 5 | 4 | 2307-2541 |

```yaml
task: Put a sock in the basket, the shirt in the bin and the other sock in the basket
eval_subtasks:
- grasp the sock in the middle of the shirt
- move the sock to the basket
- release the sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, set-down, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T006 Put the beige shirt and the navy shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 48. Table: episode 48 (2229 frames, mean quality 3.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the beige shirt | 1 | cloth-pinch | 4 | 4 | 0-288 |
| 2 | move the beige shirt to the bin | 1 | na | 3 | 3 | 288-570 |
| 3 | release the beige shirt in the bin | 1 | drop | 3 | 4 | 570-711 |
| 4 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 711-1080 |
| 5 | move the shirt to the bin | 1 | na | 3 | 3 | 1080-1236 |
| 6 | release the shirt in the bin | 1 | drop | 3 | 3 | 1236-1350 |
| 7 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 1350-1690 |
| 8 | move the shirt to the bin | 1 | na | 3 | 4 | 1690-1746 |
| 9 | release the shirt in the bin | 1 | drop | 3 | 4 | 1746-1833 |
| 10 | return to home | 1 | na | 4 | 4 | 1833-2229 |

```yaml
task: Put the beige shirt and the navy shirt in the bin
eval_subtasks:
- grasp the beige shirt
- move the beige shirt to the bin
- release the beige shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T007 Put the black shirt, the beige shirt and the navy shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 23. Table: episode 23 (3465 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black shirt | 1 | cloth-pinch | 3 | 4 | 0-339 |
| 2 | move the black shirt to the bin | 1 | na | 3 | 3 | 339-687 |
| 3 | release the black shirt in the bin | 1 | drop | 2 | 3 | 687-738 |
| 4 | grasp the black shirt | 1 | cloth-pinch | 2 | 2 | 738-1179 |
| 5 | move the black shirt to the bin | 1 | na | 4 | 4 | 1179-1326 |
| 6 | release the black shirt in the bin | 1 | drop | 3 | 4 | 1326-1362 |
| 7 | grasp the beige shirt | 1 | cloth-pinch | 4 | 4 | 1362-1700 |
| 8 | move the beige shirt to the bin | 1 | na | 3 | 3 | 1700-1980 |
| 9 | release the beige shirt in the bin | 1 | drop | 2 | 3 | 1980-2031 |
| 10 | grasp the beige shirt | 1 | cloth-pinch | 5 | 3 | 2031-2274 |
| 11 | move the beige shirt to the bin | 1 | na | 4 | 3 | 2274-2433 |
| 12 | release the beige shirt in the bin | 1 | drop | 2 | 4 | 2433-2532 |
| 13 | grasp the shirt | 1 | cloth-pinch | 4 | 5 | 2532-2914 |
| 14 | move the shirt to the bin | 1 | na | 4 | 5 | 2914-3125 |
| 15 | release the shirt in the bin | 1 | drop | 2 | 4 | 3125-3188 |
| 16 | return to home | 1 | na | 5 | 4 | 3188-3465 |

```yaml
task: Put the black shirt, the beige shirt and the navy shirt in the bin
eval_subtasks:
- grasp the black shirt
- move the black shirt to the bin
- release the black shirt in the bin
- grasp the black shirt
- move the black shirt to the bin
- release the black shirt in the bin
- grasp the beige shirt
- move the beige shirt to the bin
- release the beige shirt in the bin
- grasp the beige shirt
- move the beige shirt to the bin
- release the beige shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T008 Put the black sock and the white sock in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 82, 83, 84. Table: episode 83 (2160 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock | 1 | cloth-pinch | 3 | 3 | 0-606 |
| 2 | move the black sock to the basket | 1 | na | 4 | 5 | 606-894 |
| 3 | release the black sock in the basket | 1 | drop | 2 | 4 | 894-954 |
| 4 | grasp the sock | 1 | cloth-pinch | 3 | 3 | 954-1512 |
| 5 | move the sock to the basket | 1 | na | 4 | 5 | 1512-1794 |
| 6 | release the sock in the basket | 1 | drop | 2 | 3 | 1794-1884 |
| 7 | return to home | 1 | na | 4 | 3 | 1884-2160 |

```yaml
task: Put the black sock and the white sock in the basket
eval_subtasks:
- grasp the black sock
- move the black sock to the basket
- release the black sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

<details><summary>Other episodes (2)</summary>


Episode 84 (1998 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock | 1 | cloth-pinch | 3 | 5 | 0-531 |
| 2 | move the black sock to the basket | 1 | na | 3 | 3 | 531-813 |
| 3 | release the black sock in the basket | 1 | drop | 2 | 4 | 813-903 |
| 4 | grasp the sock | 1 | cloth-pinch | 2 | 2 | 903-1425 |
| 5 | move the sock to the basket | 1 | na | 3 | 4 | 1425-1599 |
| 6 | release the sock in the basket | 1 | drop | 2 | 4 | 1599-1671 |
| 7 | return to home | 1 | na | 4 | 4 | 1671-1998 |


Episode 82 (2541 frames, mean quality 3.6)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock | 1 | cloth-pinch | 4 | 3 | 0-588 |
| 2 | move the black sock to the basket | 1 | na | 4 | 4 | 588-918 |
| 3 | release the black sock in the basket | 1 | drop | 3 | 3 | 918-1158 |
| 4 | grasp the sock | 1 | cloth-pinch | 3 | 3 | 1158-1803 |
| 5 | move the sock to the basket | 1 | na | 3 | 4 | 1803-2100 |
| 6 | release the sock in the basket | 1 | drop | 3 | 4 | 2100-2181 |
| 7 | return to home | 1 | na | 4 | 4 | 2181-2541 |

</details>

### T009 Put the black sock in the basket and the white sock in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 53. Table: episode 53 (1926 frames, mean quality 3.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock | 1 | cloth-pinch | 3 | 2 | 0-591 |
| 2 | move the black sock to the basket | 1 | na | 4 | 5 | 591-825 |
| 3 | release the black sock in the basket | 1 | drop | 3 | 5 | 825-882 |
| 4 | grasp the sock | 1 | cloth-pinch | 2 | 3 | 882-1251 |
| 5 | move the sock to the bin | 1 | na | 4 | 3 | 1251-1488 |
| 6 | release the sock in the bin | 1 | drop | 3 | 3 | 1488-1590 |
| 7 | return to home | 1 | na | 4 | 3 | 1590-1926 |

```yaml
task: Put the black sock in the basket and the white sock in the bin
eval_subtasks:
- grasp the black sock
- move the black sock to the basket
- release the black sock in the basket
- grasp the sock
- move the sock to the bin
- release the sock in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T010 Put the black sock in the bin, the white sock in the basket and the shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 4. Table: episode 4 (3411 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock | 1 | cloth-pinch | 3 | 5 | 0-502 |
| 2 | move the black sock to the bin | 1 | na | 4 | 5 | 502-652 |
| 3 | release the black sock in the bin | 1 | drop | 3 | 4 | 652-715 |
| 4 | grasp the sock | 1 | cloth-pinch | 2 | 2 | 715-1310 |
| 5 | move the sock to the basket | 1 | na | 3 | 1 | 1310-1711 |
| 6 | grasp the sock | 1 | cloth-pinch | 2 | 4 | 1711-2007 |
| 7 | move the sock to the basket | 1 | na | 3 | 4 | 2007-2160 |
| 8 | release the sock in the basket | 1 | drop | 2 | 4 | 2160-2219 |
| 9 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 2219-2822 |
| 10 | move the shirt to the bin | 1 | na | 4 | 4 | 2822-3097 |
| 11 | release the shirt in the bin | 1 | drop | 2 | 4 | 3097-3172 |
| 12 | return to home | 1 | na | 5 | 4 | 3172-3411 |

```yaml
task: Put the black sock in the bin, the white sock in the basket and the shirt in the bin
eval_subtasks:
- grasp the black sock
- move the black sock to the bin
- release the black sock in the bin
- grasp the sock
- move the sock to the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T011 Put the brown shirt, the grey shirt and the white shirt in the bin and the socks in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 69. Table: episode 69 (3957 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the brown shirt | 1 | cloth-pinch | 4 | 4 | 0-363 |
| 2 | move the brown shirt to the bin | 1 | na | 4 | 3 | 363-534 |
| 3 | release the brown shirt in the bin | 1 | drop | 2 | 3 | 534-603 |
| 4 | grasp the brown shirt | 1 | cloth-pinch | 3 | 3 | 603-817 |
| 5 | move the brown shirt to the bin | 1 | na | 4 | 4 | 817-948 |
| 6 | release the brown shirt in the bin | 1 | drop | 3 | 4 | 948-1006 |
| 7 | grasp the grey shirt | 1 | cloth-pinch | 3 | 3 | 1006-1299 |
| 8 | move the grey shirt to the bin | 1 | na | 3 | 3 | 1299-1408 |
| 9 | release the grey shirt in the bin | 1 | drop | 4 | 3 | 1408-1515 |
| 10 | grasp the grey shirt | 1 | cloth-pinch | 4 | 3 | 1515-1715 |
| 11 | move the grey shirt to the bin | 1 | na | 3 | 5 | 1715-1790 |
| 12 | release the grey shirt in the bin | 1 | drop | 2 | 5 | 1790-1855 |
| 13 | grasp the shirt | 1 | cloth-pinch | 4 | 4 | 1855-2219 |
| 14 | move the shirt to the bin | 1 | na | 5 | 3 | 2219-2450 |
| 15 | release the shirt in the bin | 1 | drop | 4 | 4 | 2450-2526 |
| 16 | grasp the sock farther from the robot | 1 | cloth-pinch | 3 | 5 | 2526-2940 |
| 17 | move the sock to the basket | 1 | na | 4 | 5 | 2940-3116 |
| 18 | release the sock in the basket | 1 | drop | 3 | 4 | 3116-3210 |
| 19 | grasp the sock | 1 | cloth-pinch | 3 | 4 | 3210-3485 |
| 20 | move the sock to the basket | 1 | na | 4 | 5 | 3485-3665 |
| 21 | release the sock in the basket | 1 | drop | 3 | 4 | 3665-3735 |
| 22 | return to home | 1 | na | 5 | 4 | 3735-3957 |

```yaml
task: Put the brown shirt, the grey shirt and the white shirt in the bin and the socks in the basket
eval_subtasks:
- grasp the brown shirt
- move the brown shirt to the bin
- release the brown shirt in the bin
- grasp the brown shirt
- move the brown shirt to the bin
- release the brown shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the sock farther from the robot
- move the sock to the basket
- release the sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T012 Put the brown shirt, the white shirt and the grey shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 25. Table: episode 25 (3174 frames, mean quality 3.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the brown shirt | 1 | cloth-pinch | 2 | 3 | 0-330 |
| 2 | move the brown shirt to the bin | 1 | na | 4 | 3 | 330-612 |
| 3 | release the brown shirt in the bin | 1 | drop | 2 | 3 | 612-687 |
| 4 | grasp the brown shirt | 1 | cloth-pinch | 3 | 3 | 687-933 |
| 5 | move the brown shirt to the bin | 1 | na | 3 | 3 | 933-1095 |
| 6 | release the brown shirt in the bin | 1 | drop | 2 | 4 | 1095-1221 |
| 7 | grasp the white shirt | 1 | cloth-pinch | 4 | 3 | 1221-1503 |
| 8 | move the white shirt to the bin | 1 | na | 3 | 3 | 1503-1710 |
| 9 | release the white shirt in the bin | 1 | drop | 2 | 3 | 1710-1820 |
| 10 | grasp the white shirt | 1 | cloth-pinch | 4 | 3 | 1820-2013 |
| 11 | move the white shirt to the bin | 1 | na | 4 | 3 | 2013-2133 |
| 12 | release the white shirt in the bin | 1 | drop | 3 | 4 | 2133-2218 |
| 13 | grasp the shirt | 1 | cloth-pinch | 5 | 5 | 2218-2547 |
| 14 | move the shirt to the bin | 1 | na | 4 | 5 | 2547-2755 |
| 15 | release the shirt in the bin | 1 | drop | 2 | 4 | 2755-2860 |
| 16 | return to home | 1 | na | 5 | 3 | 2860-3174 |

```yaml
task: Put the brown shirt, the white shirt and the grey shirt in the bin
eval_subtasks:
- grasp the brown shirt
- move the brown shirt to the bin
- release the brown shirt in the bin
- grasp the brown shirt
- move the brown shirt to the bin
- release the brown shirt in the bin
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T013 Put the green shirt in the bin, the sock and the white shirt in the basket, and the navy shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 41. Table: episode 41 (5949 frames, mean quality 3.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green shirt | 1 | cloth-pinch | 3 | 3 | 0-642 |
| 2 | move the green shirt to the bin | 1 | na | 3 | 4 | 642-796 |
| 3 | release the green shirt in the bin | 1 | drop | 3 | 4 | 796-883 |
| 4 | grasp the sock nearest the bin | 1 | cloth-pinch | 2 | 4 | 883-1347 |
| 5 | move the sock to the basket | 1 | na | 3 | 5 | 1347-1620 |
| 6 | release the sock in the basket | 1 | drop | 2 | 4 | 1620-1698 |
| 7 | grasp the white shirt nearest the basket | 1 | cloth-pinch | 1 | 1 | 1698-3114 |
| 8 | move the white shirt to the basket | 1 | na | 2 | 3 | 3114-3330 |
| 9 | release the white shirt in the basket | 1 | set-down | 2 | 3 | 3330-3582 |
| 10 | grasp the white shirt nearest the basket | 1 | cloth-pinch | 2 | 1 | 3582-4590 |
| 11 | move the white shirt to the basket | 1 | na | 4 | 4 | 4590-4740 |
| 12 | release the white shirt in the basket | 1 | drop | 3 | 3 | 4740-4812 |
| 13 | grasp the white shirt nearest the basket | 1 | cloth-pinch | 3 | 3 | 4812-5096 |
| 14 | move the white shirt to the basket | 1 | na | 3 | 5 | 5096-5256 |
| 15 | release the white shirt in the basket | 1 | drop | 2 | 4 | 5256-5320 |
| 16 | grasp the navy shirt | 1 | cloth-pinch | 2 | 4 | 5320-5803 |
| 17 | move the navy shirt to the bin | 1 | na | 3 | 3 | 5803-5949 |

```yaml
task: Put the green shirt in the bin, the sock and the white shirt in the basket, and the navy shirt in the bin
eval_subtasks:
- grasp the green shirt
- move the green shirt to the bin
- release the green shirt in the bin
- grasp the sock nearest the bin
- move the sock to the basket
- release the sock in the basket
- grasp the white shirt nearest the basket
- move the white shirt to the basket
- release the white shirt in the basket
- grasp the white shirt nearest the basket
- move the white shirt to the basket
- release the white shirt in the basket
- grasp the white shirt nearest the basket
- move the white shirt to the basket
- release the white shirt in the basket
- grasp the navy shirt
- move the navy shirt to the bin
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, set-down, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na]
```

### T014 Put the grey shirt and the brown shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 18. Table: episode 18 (3894 frames, mean quality 3.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the grey shirt | 1 | cloth-pinch | 3 | 4 | 0-302 |
| 2 | move the grey shirt to the bin | 1 | na | 4 | 3 | 302-570 |
| 3 | release the grey shirt in the bin | 1 | drop | 2 | 3 | 570-636 |
| 4 | grasp the grey shirt | 1 | cloth-pinch | 3 | 3 | 636-898 |
| 5 | move the grey shirt to the bin | 1 | na | 4 | 3 | 898-1078 |
| 6 | release the grey shirt in the bin | 1 | drop | 2 | 3 | 1078-1125 |
| 7 | grasp the grey shirt | 1 | cloth-pinch | 4 | 3 | 1125-1353 |
| 8 | move the grey shirt to the bin | 1 | na | 4 | 3 | 1353-1480 |
| 9 | release the grey shirt in the bin | 1 | drop | 2 | 3 | 1480-1565 |
| 10 | grasp the grey shirt | 1 | cloth-pinch | 3 | 2 | 1565-1950 |
| 11 | move the grey shirt to the bin | 1 | na | 4 | 3 | 1950-2015 |
| 12 | release the grey shirt in the bin | 1 | drop | 4 | 3 | 2015-2045 |
| 13 | grasp the grey shirt | 1 | cloth-pinch | 3 | 3 | 2045-2150 |
| 14 | move the grey shirt to the bin | 1 | na | 4 | 4 | 2150-2295 |
| 15 | release the grey shirt in the bin | 1 | drop | 2 | 4 | 2295-2350 |
| 16 | grasp the shirt | 1 | cloth-pinch | 4 | 4 | 2350-2750 |
| 17 | move the shirt to the bin | 1 | na | 4 | 3 | 2750-2985 |
| 18 | release the shirt in the bin | 1 | drop | 2 | 3 | 2985-3045 |
| 19 | grasp the shirt | 1 | cloth-pinch | 1 | 2 | 3045-3456 |
| 20 | move the shirt to the bin | 1 | na | 3 | 4 | 3456-3565 |
| 21 | release the shirt in the bin | 1 | drop | 3 | 4 | 3565-3606 |
| 22 | return to home | 1 | na | 5 | 3 | 3606-3894 |

```yaml
task: Put the grey shirt and the brown shirt in the bin
eval_subtasks:
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T015 Put the grey shirt and the navy shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 86. Table: episode 86 (3324 frames, mean quality 3.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the grey shirt | 1 | cloth-pinch | 3 | 4 | 0-474 |
| 2 | move the grey shirt to the bin | 1 | na | 3 | 3 | 474-705 |
| 3 | release the grey shirt in the bin | 1 | drop | 2 | 3 | 705-819 |
| 4 | grasp the grey shirt | 1 | cloth-pinch | 3 | 3 | 819-1131 |
| 5 | move the grey shirt to the bin | 1 | na | 3 | 3 | 1131-1305 |
| 6 | release the grey shirt in the bin | 1 | drop | 2 | 3 | 1305-1365 |
| 7 | grasp the grey shirt | 1 | cloth-pinch | 3 | 3 | 1365-1689 |
| 8 | move the grey shirt to the bin | 1 | na | 2 | 5 | 1689-1815 |
| 9 | release the grey shirt in the bin | 1 | drop | 1 | 5 | 1815-1866 |
| 10 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 1866-2169 |
| 11 | move the shirt to the bin | 1 | na | 3 | 3 | 2169-2370 |
| 12 | release the shirt in the bin | 1 | drop | 2 | 3 | 2370-2445 |
| 13 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 2445-2853 |
| 14 | move the shirt to the bin | 1 | na | 3 | 4 | 2853-2970 |
| 15 | release the shirt in the bin | 1 | drop | 3 | 4 | 2970-3006 |
| 16 | return to home | 1 | na | 5 | 4 | 3006-3324 |

```yaml
task: Put the grey shirt and the navy shirt in the bin
eval_subtasks:
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T016 Put the grey shirt in the bin, two socks in the basket, the white shirt in the bin and the other two socks in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 65. Table: episode 65 (5982 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the grey shirt | 1 | cloth-pinch | 4 | 5 | 0-225 |
| 2 | move the grey shirt to the bin | 1 | na | 4 | 3 | 225-450 |
| 3 | release the grey shirt in the bin | 1 | drop | 4 | 3 | 450-597 |
| 4 | grasp the grey shirt | 1 | cloth-pinch | 4 | 3 | 597-825 |
| 5 | move the grey shirt to the bin | 1 | na | 5 | 4 | 825-900 |
| 6 | release the grey shirt in the bin | 1 | drop | 3 | 4 | 900-990 |
| 7 | grasp the sock nearest the bin | 1 | cloth-pinch | 3 | 3 | 990-1203 |
| 8 | move the sock to the basket | 1 | na | 4 | 4 | 1203-1418 |
| 9 | release the sock in the basket | 1 | drop | 4 | 4 | 1418-1503 |
| 10 | grasp the sock in the middle of the shirt | 1 | cloth-pinch | 3 | 4 | 1503-1893 |
| 11 | move the sock to the basket | 1 | na | 3 | 4 | 1893-2076 |
| 12 | release the sock in the basket | 1 | drop | 4 | 4 | 2076-2157 |
| 13 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 2157-2598 |
| 14 | move the shirt to the bin | 1 | na | 4 | 3 | 2598-2835 |
| 15 | release the shirt in the bin | 1 | drop | 4 | 3 | 2835-2967 |
| 16 | grasp the shirt | 1 | cloth-pinch | 3 | 1 | 2967-3462 |
| 17 | move the shirt to the bin | 1 | na | 3 | 4 | 3462-3512 |
| 18 | release the shirt in the bin | 1 | drop | 3 | 5 | 3512-3603 |
| 19 | grasp the sock nearest the bin | 1 | cloth-pinch | 2 | 3 | 3603-4026 |
| 20 | move the sock to the basket | 1 | na | 4 | 4 | 4026-4272 |
| 21 | release the sock in the basket | 1 | drop | 2 | 4 | 4272-4365 |
| 22 | grasp the sock | 1 | cloth-pinch | 2 | 1 | 4365-5454 |
| 23 | move the sock to the basket | 1 | na | 4 | 4 | 5454-5640 |
| 24 | release the sock in the basket | 1 | drop | 4 | 4 | 5640-5720 |
| 25 | return to home | 1 | na | 5 | 5 | 5720-5982 |

```yaml
task: Put the grey shirt in the bin, two socks in the basket, the white shirt in the bin and the other two socks in the basket
eval_subtasks:
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the sock nearest the bin
- move the sock to the basket
- release the sock in the basket
- grasp the sock in the middle of the shirt
- move the sock to the basket
- release the sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the sock nearest the bin
- move the sock to the basket
- release the sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T017 Put the navy shirt and the grey shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 22. Table: episode 22 (2097 frames, mean quality 3.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the navy shirt | 1 | cloth-pinch | 4 | 4 | 0-447 |
| 2 | move the navy shirt to the bin | 1 | na | 3 | 3 | 447-672 |
| 3 | release the navy shirt in the bin | 1 | drop | 2 | 4 | 672-717 |
| 4 | grasp the shirt | 1 | cloth-pinch | 4 | 3 | 717-1101 |
| 5 | move the shirt to the bin | 1 | na | 4 | 3 | 1101-1350 |
| 6 | release the shirt in the bin | 1 | drop | 2 | 3 | 1350-1446 |
| 7 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 1446-1646 |
| 8 | move the shirt to the bin | 1 | na | 3 | 4 | 1646-1764 |
| 9 | release the shirt in the bin | 1 | drop | 2 | 4 | 1764-1824 |
| 10 | return to home | 1 | na | 5 | 4 | 1824-2097 |

```yaml
task: Put the navy shirt and the grey shirt in the bin
eval_subtasks:
- grasp the navy shirt
- move the navy shirt to the bin
- release the navy shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T018 Put the navy shirt and the white shirt in the bin, the sock nearest the bin in the basket, the brown shirt in the bin and the other sock in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 1. Table: episode 1 (6768 frames, mean quality 3.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the navy shirt | 1 | cloth-pinch | 3 | 5 | 0-390 |
| 2 | move the navy shirt to the bin | 1 | na | 3 | 4 | 390-618 |
| 3 | release the navy shirt in the bin | 1 | drop | 3 | 4 | 618-708 |
| 4 | grasp the white shirt | 1 | cloth-pinch | 3 | 4 | 708-996 |
| 5 | move the white shirt to the bin | 1 | na | 3 | 5 | 996-1161 |
| 6 | release the white shirt in the bin | 1 | drop | 2 | 4 | 1161-1209 |
| 7 | grasp the sock nearest the bin | 1 | cloth-pinch | 2 | 1 | 1209-2004 |
| 8 | move the sock to the basket | 1 | na | 3 | 3 | 2004-2340 |
| 9 | release the sock in the basket | 1 | drop | 3 | 4 | 2340-2388 |
| 10 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 2388-2955 |
| 11 | move the shirt to the bin | 1 | na | 3 | 3 | 2955-3222 |
| 12 | release the shirt in the bin | 1 | drop | 2 | 3 | 3222-3291 |
| 13 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 3291-3546 |
| 14 | move the shirt to the bin | 1 | na | 3 | 3 | 3546-3636 |
| 15 | release the shirt in the bin | 1 | drop | 3 | 3 | 3636-3693 |
| 16 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 3693-3834 |
| 17 | move the shirt to the bin | 1 | na | 3 | 3 | 3834-4005 |
| 18 | release the shirt in the bin | 1 | drop | 2 | 3 | 4005-4044 |
| 19 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 4044-4287 |
| 20 | move the shirt to the bin | 1 | na | 4 | 3 | 4287-4332 |
| 21 | release the shirt in the bin | 1 | drop | 2 | 3 | 4332-4368 |
| 22 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 4368-4490 |
| 23 | move the shirt to the bin | 1 | na | 3 | 3 | 4490-4566 |
| 24 | release the shirt in the bin | 1 | drop | 2 | 3 | 4566-4602 |
| 25 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 4602-4718 |
| 26 | move the shirt to the bin | 1 | na | 2 | 3 | 4718-4800 |
| 27 | release the shirt in the bin | 1 | drop | 2 | 3 | 4800-4836 |
| 28 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 4836-4950 |
| 29 | move the shirt to the bin | 1 | na | 3 | 3 | 4950-5088 |
| 30 | release the shirt in the bin | 1 | drop | 3 | 3 | 5088-5124 |
| 31 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 5124-5349 |
| 32 | move the shirt to the bin | 1 | na | 3 | 3 | 5349-5502 |
| 33 | release the shirt in the bin | 1 | drop | 3 | 3 | 5502-5535 |
| 34 | grasp the sock | 1 | cloth-pinch | 2 | 2 | 5535-6250 |
| 35 | move the sock to the basket | 1 | na | 3 | 5 | 6250-6438 |
| 36 | release the sock in the basket | 1 | drop | 2 | 4 | 6438-6498 |
| 37 | return to home | 1 | na | 5 | 4 | 6498-6768 |

```yaml
task: Put the navy shirt and the white shirt in the bin, the sock nearest the bin in the basket, the brown shirt in the bin and the other sock in the basket
eval_subtasks:
- grasp the navy shirt
- move the navy shirt to the bin
- release the navy shirt in the bin
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the sock nearest the bin
- move the sock to the basket
- release the sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T019 Put the plain white socks, then the black socks, then the socks with blue toes in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 39. Table: episode 39 (5850 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the plain white sock on top of the pile | 1 | cloth-pinch | 2 | 4 | 0-489 |
| 2 | move the plain white sock to the basket | 1 | na | 3 | 5 | 489-714 |
| 3 | release the plain white sock in the basket | 1 | drop | 3 | 5 | 714-763 |
| 4 | grasp the plain white sock | 1 | cloth-pinch | 2 | 2 | 763-1256 |
| 5 | move the plain white sock to the basket | 1 | na | 3 | 4 | 1256-1497 |
| 6 | release the plain white sock in the basket | 1 | drop | 2 | 4 | 1497-1539 |
| 7 | grasp the black sock farthest from the basket | 1 | cloth-pinch | 2 | 5 | 1539-1917 |
| 8 | move the black sock to the basket | 1 | na | 4 | 5 | 1917-2141 |
| 9 | release the black sock in the basket | 1 | drop | 2 | 5 | 2141-2189 |
| 10 | grasp the black sock nearest the basket | 1 | cloth-pinch | 2 | 5 | 2189-2578 |
| 11 | move the black sock to the basket | 1 | na | 4 | 4 | 2578-2791 |
| 12 | release the black sock in the basket | 1 | drop | 2 | 4 | 2791-2833 |
| 13 | grasp the black sock on top of the pile | 1 | cloth-pinch | 2 | 3 | 2833-3197 |
| 14 | move the black sock to the basket | 1 | na | 4 | 3 | 3197-3409 |
| 15 | release the black sock in the basket | 1 | drop | 1 | 4 | 3409-3463 |
| 16 | grasp the black sock | 1 | cloth-pinch | 2 | 2 | 3463-3995 |
| 17 | move the black sock to the basket | 1 | na | 4 | 4 | 3995-4178 |
| 18 | release the black sock in the basket | 1 | drop | 2 | 4 | 4178-4240 |
| 19 | grasp the sock nearer the robot | 1 | cloth-pinch | 1 | 2 | 4240-4753 |
| 20 | move the sock to the basket | 1 | na | 4 | 3 | 4753-4926 |
| 21 | release the sock in the basket | 1 | drop | 3 | 3 | 4926-4979 |
| 22 | grasp the sock | 1 | cloth-pinch | 2 | 3 | 4979-5365 |
| 23 | move the sock to the basket | 1 | na | 5 | 3 | 5365-5545 |
| 24 | release the sock in the basket | 1 | drop | 3 | 4 | 5545-5584 |
| 25 | return to home | 1 | na | 5 | 4 | 5584-5850 |

```yaml
task: Put the plain white socks, then the black socks, then the socks with blue toes in the basket
eval_subtasks:
- grasp the plain white sock on top of the pile
- move the plain white sock to the basket
- release the plain white sock in the basket
- grasp the plain white sock
- move the plain white sock to the basket
- release the plain white sock in the basket
- grasp the black sock farthest from the basket
- move the black sock to the basket
- release the black sock in the basket
- grasp the black sock nearest the basket
- move the black sock to the basket
- release the black sock in the basket
- grasp the black sock on top of the pile
- move the black sock to the basket
- release the black sock in the basket
- grasp the black sock
- move the black sock to the basket
- release the black sock in the basket
- grasp the sock nearer the robot
- move the sock to the basket
- release the sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T020 Put the red shirt in the basket and the brown shirt in the bin, then grasp the black sock farther from the robot

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 50. Table: episode 50 (5997 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red shirt | 1 | cloth-pinch | 3 | 4 | 0-378 |
| 2 | move the red shirt to the bin | 1 | na | 2 | 3 | 378-522 |
| 3 | move the red shirt to the basket | 1 | na | 3 | 3 | 522-774 |
| 4 | release the red shirt in the basket | 1 | drop | 3 | 3 | 774-933 |
| 5 | grasp the red shirt | 1 | cloth-pinch | 2 | 3 | 933-1236 |
| 6 | move the red shirt to the basket | 1 | na | 2 | 3 | 1236-1400 |
| 7 | release the red shirt in the basket | 1 | drop | 2 | 3 | 1400-1480 |
| 8 | grasp the red shirt | 1 | cloth-pinch | 2 | 2 | 1480-1758 |
| 9 | move the red shirt to the basket | 1 | na | 3 | 2 | 1758-1812 |
| 10 | grasp the red shirt | 1 | cloth-pinch | 2 | 2 | 1812-2062 |
| 11 | move the red shirt to the basket | 1 | na | 4 | 2 | 2062-2125 |
| 12 | grasp the red shirt | 1 | cloth-pinch | 2 | 2 | 2125-2366 |
| 13 | move the red shirt to the basket | 1 | na | 2 | 3 | 2366-2590 |
| 14 | release the red shirt in the basket | 1 | drop | 2 | 3 | 2590-2680 |
| 15 | grasp the red shirt | 1 | cloth-pinch | 2 | 3 | 2680-2996 |
| 16 | move the red shirt to the basket | 1 | na | 3 | 4 | 2996-3205 |
| 17 | release the red shirt in the basket | 1 | drop | 2 | 3 | 3205-3290 |
| 18 | grasp the red shirt | 1 | cloth-pinch | 2 | 3 | 3290-3598 |
| 19 | move the red shirt to the basket | 1 | na | 3 | 4 | 3598-3700 |
| 20 | release the red shirt in the basket | 1 | drop | 2 | 5 | 3700-3765 |
| 21 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 3765-4345 |
| 22 | move the shirt to the bin | 1 | na | 3 | 3 | 4345-4640 |
| 23 | release the shirt in the bin | 1 | drop | 3 | 3 | 4640-4760 |
| 24 | grasp the shirt | 1 | cloth-pinch | 3 | 2 | 4760-5388 |
| 25 | move the shirt to the bin | 1 | na | 4 | 4 | 5388-5500 |
| 26 | release the shirt in the bin | 1 | drop | 3 | 4 | 5500-5586 |
| 27 | grasp the black sock farther from the robot | 1 | cloth-pinch | 3 | 3 | 5586-5997 |

```yaml
task: Put the red shirt in the basket and the brown shirt in the bin, then grasp the black sock farther from the robot
eval_subtasks:
- grasp the red shirt
- move the red shirt to the bin
- move the red shirt to the basket
- release the red shirt in the basket
- grasp the red shirt
- move the red shirt to the basket
- release the red shirt in the basket
- grasp the red shirt
- move the red shirt to the basket
- grasp the red shirt
- move the red shirt to the basket
- grasp the red shirt
- move the red shirt to the basket
- release the red shirt in the basket
- grasp the red shirt
- move the red shirt to the basket
- release the red shirt in the basket
- grasp the red shirt
- move the red shirt to the basket
- release the red shirt in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the black sock farther from the robot
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, na, drop, cloth-pinch, na, drop, cloth-pinch, na, cloth-pinch, na, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch]
```

### T021 Put the red shirt in the bin, the striped sock in the basket, the navy shirt in the bin, the sock on the shirt in the basket, the navy shirt in the bin again, and the plain black sock and the last sock in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 60. Table: episode 60 (5295 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red shirt | 1 | cloth-pinch | 4 | 4 | 0-288 |
| 2 | move the red shirt to the bin | 1 | na | 4 | 3 | 288-510 |
| 3 | release the red shirt in the bin | 1 | drop | 4 | 4 | 510-612 |
| 4 | grasp the striped sock | 1 | cloth-pinch | 3 | 4 | 612-959 |
| 5 | move the striped sock to the bin | 1 | na | 4 | 4 | 959-1146 |
| 6 | move the striped sock to the basket | 1 | na | 4 | 4 | 1146-1410 |
| 7 | release the striped sock in the basket | 1 | drop | 3 | 4 | 1410-1503 |
| 8 | grasp the shirt | 1 | cloth-pinch | 2 | 1 | 1503-2533 |
| 9 | move the shirt to the bin | 1 | na | 3 | 3 | 2533-2730 |
| 10 | release the shirt in the bin | 1 | drop | 3 | 3 | 2730-2834 |
| 11 | grasp the sock on the shirt | 1 | cloth-pinch | 3 | 3 | 2834-3057 |
| 12 | move the sock to the basket | 1 | na | 4 | 4 | 3057-3320 |
| 13 | release the sock in the basket | 1 | drop | 2 | 4 | 3320-3384 |
| 14 | grasp the shirt | 1 | cloth-pinch | 3 | 5 | 3384-3690 |
| 15 | move the shirt to the bin | 1 | na | 4 | 5 | 3690-3833 |
| 16 | release the shirt in the bin | 1 | drop | 3 | 5 | 3833-3875 |
| 17 | grasp the plain black sock | 1 | cloth-pinch | 4 | 3 | 3875-4142 |
| 18 | move the plain black sock to the basket | 1 | na | 5 | 3 | 4142-4400 |
| 19 | release the plain black sock in the basket | 1 | drop | 4 | 4 | 4400-4451 |
| 20 | grasp the sock | 1 | cloth-pinch | 3 | 5 | 4451-4838 |
| 21 | move the sock to the basket | 1 | na | 4 | 5 | 4838-5025 |
| 22 | release the sock in the basket | 1 | drop | 3 | 4 | 5025-5082 |
| 23 | return to home | 1 | na | 5 | 5 | 5082-5295 |

```yaml
task: Put the red shirt in the bin, the striped sock in the basket, the navy shirt in the bin, the sock on the shirt in the basket, the navy shirt in the bin again, and the plain black sock and the last sock in the basket
eval_subtasks:
- grasp the red shirt
- move the red shirt to the bin
- release the red shirt in the bin
- grasp the striped sock
- move the striped sock to the bin
- move the striped sock to the basket
- release the striped sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the sock on the shirt
- move the sock to the basket
- release the sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the plain black sock
- move the plain black sock to the basket
- release the plain black sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T022 Put the red shirt, the grey shirt, the white shirt and the brown shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 20. Table: episode 20 (5004 frames, mean quality 3.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red shirt | 1 | cloth-pinch | 3 | 5 | 0-267 |
| 2 | move the red shirt to the bin | 1 | na | 5 | 4 | 267-447 |
| 3 | release the red shirt in the bin | 1 | drop | 3 | 4 | 447-546 |
| 4 | grasp the grey shirt | 1 | cloth-pinch | 4 | 3 | 546-903 |
| 5 | move the grey shirt to the bin | 1 | na | 3 | 3 | 903-1140 |
| 6 | release the grey shirt in the bin | 1 | drop | 2 | 3 | 1140-1218 |
| 7 | grasp the grey shirt | 1 | cloth-pinch | 3 | 3 | 1218-1494 |
| 8 | move the grey shirt to the bin | 1 | na | 3 | 5 | 1494-1656 |
| 9 | release the grey shirt in the bin | 1 | drop | 2 | 5 | 1656-1707 |
| 10 | grasp the white shirt | 1 | cloth-pinch | 2 | 2 | 1707-2160 |
| 11 | move the white shirt to the bin | 1 | na | 4 | 3 | 2160-2340 |
| 12 | release the white shirt in the bin | 1 | drop | 5 | 4 | 2340-2430 |
| 13 | grasp the shirt | 1 | cloth-pinch | 3 | 4 | 2430-2796 |
| 14 | move the shirt to the bin | 1 | na | 4 | 3 | 2796-3030 |
| 15 | release the shirt in the bin | 1 | drop | 2 | 3 | 3030-3108 |
| 16 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 3108-3339 |
| 17 | move the shirt to the bin | 1 | na | 3 | 3 | 3339-3432 |
| 18 | release the shirt in the bin | 1 | drop | 4 | 3 | 3432-3477 |
| 19 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 3477-3861 |
| 20 | move the shirt to the bin | 1 | na | 3 | 3 | 3861-4029 |
| 21 | release the shirt in the bin | 1 | drop | 2 | 3 | 4029-4080 |
| 22 | grasp the shirt | 1 | cloth-pinch | 2 | 1 | 4080-4553 |
| 23 | move the shirt to the bin | 1 | na | 3 | 4 | 4553-4712 |
| 24 | release the shirt in the bin | 1 | drop | 2 | 4 | 4712-4772 |
| 25 | return to home | 1 | na | 5 | 4 | 4772-5004 |

```yaml
task: Put the red shirt, the grey shirt, the white shirt and the brown shirt in the bin
eval_subtasks:
- grasp the red shirt
- move the red shirt to the bin
- release the red shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T023 Put the red shirt, the navy shirt, the black shirt and the grey shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 21. Table: episode 21 (5214 frames, mean quality 3.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red shirt | 1 | cloth-pinch | 2 | 2 | 0-522 |
| 2 | move the red shirt to the bin | 1 | na | 4 | 2 | 522-790 |
| 3 | release the red shirt in the bin | 1 | drop | 2 | 2 | 790-836 |
| 4 | grasp the red shirt | 1 | cloth-pinch | 4 | 2 | 836-1122 |
| 5 | move the red shirt to the bin | 1 | na | 3 | 3 | 1122-1404 |
| 6 | release the red shirt in the bin | 1 | drop | 3 | 4 | 1404-1452 |
| 7 | grasp the navy shirt | 1 | cloth-pinch | 5 | 4 | 1452-1762 |
| 8 | move the navy shirt to the bin | 1 | na | 3 | 3 | 1762-1990 |
| 9 | release the navy shirt in the bin | 1 | drop | 2 | 3 | 1990-2058 |
| 10 | grasp the navy shirt | 1 | cloth-pinch | 2 | 2 | 2058-2652 |
| 11 | move the navy shirt to the bin | 1 | na | 4 | 3 | 2652-2760 |
| 12 | release the navy shirt in the bin | 1 | drop | 2 | 3 | 2760-2875 |
| 13 | grasp the navy shirt | 1 | cloth-pinch | 3 | 5 | 2875-3068 |
| 14 | move the navy shirt to the bin | 1 | na | 3 | 4 | 3068-3190 |
| 15 | release the navy shirt in the bin | 1 | drop | 2 | 4 | 3190-3222 |
| 16 | grasp the black shirt | 1 | cloth-pinch | 3 | 3 | 3222-3741 |
| 17 | move the black shirt to the bin | 1 | na | 4 | 3 | 3741-3905 |
| 18 | release the black shirt in the bin | 1 | drop | 2 | 4 | 3905-3954 |
| 19 | grasp the shirt | 1 | cloth-pinch | 3 | 1 | 3954-4606 |
| 20 | move the shirt to the bin | 1 | na | 4 | 3 | 4606-4870 |
| 21 | release the shirt in the bin | 1 | drop | 3 | 4 | 4870-4960 |
| 22 | return to home | 1 | na | 5 | 4 | 4960-5214 |

```yaml
task: Put the red shirt, the navy shirt, the black shirt and the grey shirt in the bin
eval_subtasks:
- grasp the red shirt
- move the red shirt to the bin
- release the red shirt in the bin
- grasp the red shirt
- move the red shirt to the bin
- release the red shirt in the bin
- grasp the navy shirt
- move the navy shirt to the bin
- release the navy shirt in the bin
- grasp the navy shirt
- move the navy shirt to the bin
- release the navy shirt in the bin
- grasp the navy shirt
- move the navy shirt to the bin
- release the navy shirt in the bin
- grasp the black shirt
- move the black shirt to the bin
- release the black shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T024 Put the shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 10, 12, 17, 87, 88, 89. Table: episode 87 (1017 frames, mean quality 4.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 4 | 5 | 0-279 |
| 2 | move the shirt to the bin | 1 | na | 4 | 4 | 279-492 |
| 3 | release the shirt in the bin | 1 | drop | 4 | 5 | 492-630 |
| 4 | return to home | 1 | na | 4 | 5 | 630-1017 |

```yaml
task: Put the shirt in the bin
eval_subtasks:
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, na]
```

<details><summary>Other episodes (5)</summary>


Episode 10 (1221 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 3 | 5 | 0-534 |
| 2 | move the shirt to the bin | 1 | na | 2 | 5 | 534-750 |
| 3 | release the shirt in the bin | 1 | drop | 3 | 4 | 750-846 |
| 4 | return to home | 1 | na | 4 | 4 | 846-1221 |


Episode 12 (1590 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 3 | 5 | 0-423 |
| 2 | move the shirt to the bin | 1 | na | 3 | 4 | 423-819 |
| 3 | release the shirt in the bin | 1 | drop | 2 | 3 | 819-1164 |
| 4 | return to home | 1 | na | 4 | 4 | 1164-1590 |


Episode 88 (1161 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 3 | 4 | 0-450 |
| 2 | move the shirt to the bin | 1 | na | 3 | 3 | 450-762 |
| 3 | release the shirt in the bin | 1 | drop | 2 | 4 | 762-828 |
| 4 | return to home | 1 | na | 4 | 4 | 828-1161 |


Episode 89 (1485 frames, mean quality 3.6)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 4 | 4 | 0-387 |
| 2 | move the shirt to the bin | 1 | na | 4 | 3 | 387-555 |
| 3 | release the shirt in the bin | 1 | drop | 3 | 3 | 555-642 |
| 4 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 642-954 |
| 5 | move the shirt to the bin | 1 | na | 3 | 4 | 954-1118 |
| 6 | release the shirt in the bin | 1 | drop | 2 | 4 | 1118-1182 |
| 7 | return to home | 1 | na | 4 | 4 | 1182-1485 |


Episode 17 (1986 frames, mean quality 3.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 0-423 |
| 2 | move the shirt to the bin | 1 | na | 4 | 3 | 423-651 |
| 3 | release the shirt in the bin | 1 | drop | 2 | 3 | 651-771 |
| 4 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 771-1065 |
| 5 | move the shirt to the bin | 1 | na | 3 | 3 | 1065-1221 |
| 6 | release the shirt in the bin | 1 | drop | 2 | 3 | 1221-1272 |
| 7 | grasp the shirt | 1 | cloth-pinch | 5 | 3 | 1272-1488 |
| 8 | move the shirt to the bin | 1 | na | 3 | 4 | 1488-1653 |
| 9 | release the shirt in the bin | 1 | drop | 2 | 4 | 1653-1692 |
| 10 | return to home | 1 | na | 5 | 4 | 1692-1986 |

</details>

### T025 Put the shirt in the bin and the black socks and the white socks in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 67. Table: episode 67 (5106 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 4 | 4 | 0-369 |
| 2 | move the shirt to the bin | 1 | na | 5 | 3 | 369-504 |
| 3 | release the shirt in the bin | 1 | drop | 4 | 3 | 504-582 |
| 4 | grasp the shirt | 1 | cloth-pinch | 4 | 3 | 582-836 |
| 5 | move the shirt to the bin | 1 | na | 4 | 4 | 836-900 |
| 6 | release the shirt in the bin | 1 | drop | 3 | 4 | 900-999 |
| 7 | grasp the black sock lying across the long black sock | 1 | cloth-pinch | 3 | 3 | 999-1440 |
| 8 | move the black sock to the basket | 1 | na | 4 | 3 | 1440-1629 |
| 9 | release the black sock in the basket | 1 | drop | 4 | 4 | 1629-1701 |
| 10 | grasp the white sock farther from the basket | 1 | cloth-pinch | 3 | 3 | 1701-2129 |
| 11 | move the white sock to the basket | 1 | na | 4 | 3 | 2129-2331 |
| 12 | release the white sock in the basket | 1 | drop | 4 | 4 | 2331-2405 |
| 13 | grasp the long black sock | 1 | cloth-pinch | 3 | 3 | 2405-2742 |
| 14 | move the long black sock to the basket | 1 | na | 4 | 3 | 2742-2910 |
| 15 | release the long black sock in the basket | 1 | drop | 3 | 4 | 2910-3018 |
| 16 | grasp the white sock | 1 | cloth-pinch | 2 | 2 | 3018-3512 |
| 17 | move the white sock to the basket | 1 | na | 4 | 4 | 3512-3672 |
| 18 | release the white sock in the basket | 1 | drop | 3 | 4 | 3672-3741 |
| 19 | grasp the plain black sock | 1 | cloth-pinch | 3 | 5 | 3741-4038 |
| 20 | move the plain black sock to the basket | 1 | na | 3 | 5 | 4038-4200 |
| 21 | release the plain black sock in the basket | 1 | drop | 3 | 4 | 4200-4283 |
| 22 | grasp the sock | 1 | cloth-pinch | 2 | 3 | 4283-4673 |
| 23 | move the sock to the basket | 1 | na | 4 | 4 | 4673-4820 |
| 24 | release the sock in the basket | 1 | drop | 3 | 4 | 4820-4898 |
| 25 | return to home | 1 | na | 5 | 5 | 4898-5106 |

```yaml
task: Put the shirt in the bin and the black socks and the white socks in the basket
eval_subtasks:
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the black sock lying across the long black sock
- move the black sock to the basket
- release the black sock in the basket
- grasp the white sock farther from the basket
- move the white sock to the basket
- release the white sock in the basket
- grasp the long black sock
- move the long black sock to the basket
- release the long black sock in the basket
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the plain black sock
- move the plain black sock to the basket
- release the plain black sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T026 Put the shirt in the bin and the socks in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 68. Table: episode 68 (2583 frames, mean quality 4.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 4 | 4 | 0-435 |
| 2 | move the shirt to the bin | 1 | na | 4 | 3 | 435-600 |
| 3 | release the shirt in the bin | 1 | drop | 3 | 3 | 600-672 |
| 4 | grasp the shirt | 1 | cloth-pinch | 4 | 3 | 672-866 |
| 5 | move the shirt to the bin | 1 | na | 3 | 5 | 866-945 |
| 6 | release the shirt in the bin | 1 | drop | 2 | 4 | 945-990 |
| 7 | grasp the sock farther from the robot | 1 | cloth-pinch | 3 | 5 | 990-1430 |
| 8 | move the sock to the basket | 1 | na | 3 | 4 | 1430-1645 |
| 9 | release the sock in the basket | 1 | drop | 2 | 4 | 1645-1708 |
| 10 | grasp the sock | 1 | cloth-pinch | 3 | 5 | 1708-2110 |
| 11 | move the sock to the basket | 1 | na | 3 | 5 | 2110-2340 |
| 12 | release the sock in the basket | 1 | drop | 2 | 4 | 2340-2390 |
| 13 | return to home | 1 | na | 5 | 4 | 2390-2583 |

```yaml
task: Put the shirt in the bin and the socks in the basket
eval_subtasks:
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the sock farther from the robot
- move the sock to the basket
- release the sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T027 Put the shirt in the bin and the white sock and the black sock in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 6. Table: episode 6 (3144 frames, mean quality 4.3).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 3 | 5 | 0-396 |
| 2 | move the shirt to the bin | 1 | na | 3 | 5 | 396-552 |
| 3 | release the shirt in the bin | 1 | drop | 2 | 4 | 552-663 |
| 4 | grasp the white sock | 1 | cloth-pinch | 3 | 5 | 663-1215 |
| 5 | move the white sock to the basket | 1 | na | 2 | 5 | 1215-1410 |
| 6 | release the white sock in the basket | 1 | drop | 2 | 4 | 1410-1469 |
| 7 | grasp the sock | 1 | cloth-pinch | 1 | 1 | 1469-2694 |
| 8 | move the sock to the basket | 1 | na | 3 | 5 | 2694-2850 |
| 9 | release the sock in the basket | 1 | drop | 3 | 4 | 2850-2919 |
| 10 | return to home | 1 | na | 5 | 5 | 2919-3144 |

```yaml
task: Put the shirt in the bin and the white sock and the black sock in the basket
eval_subtasks:
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T028 Put the shirt in the bin and the white sock in the basket, then grasp the black sock

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 11. Table: episode 11 (2895 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 0-447 |
| 2 | move the shirt to the bin | 1 | na | 2 | 3 | 447-750 |
| 3 | release the shirt in the bin | 1 | drop | 2 | 4 | 750-860 |
| 4 | grasp the white sock | 1 | cloth-pinch | 2 | 3 | 860-1428 |
| 5 | move the white sock to the basket | 1 | na | 3 | 3 | 1428-1627 |
| 6 | release the white sock in the basket | 1 | drop | 2 | 4 | 1627-1676 |
| 7 | grasp the sock | 1 | cloth-pinch | 1 | 1 | 1676-2895 |

```yaml
task: Put the shirt in the bin and the white sock in the basket, then grasp the black sock
eval_subtasks:
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch]
```

### T029 Put the shirt in the bin, then in the basket, then grasp the black sock

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 52. Table: episode 52 (4503 frames, mean quality 2.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 0-384 |
| 2 | move the shirt to the bin | 1 | na | 3 | 2 | 384-528 |
| 3 | release the shirt in the bin | 1 | drop | 2 | 2 | 528-558 |
| 4 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 558-768 |
| 5 | move the shirt to the basket | 1 | na | 3 | 3 | 768-1140 |
| 6 | release the shirt in the basket | 1 | drop | 1 | 3 | 1140-1197 |
| 7 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 1197-1622 |
| 8 | move the shirt to the basket | 1 | na | 3 | 3 | 1622-1866 |
| 9 | release the shirt in the basket | 1 | drop | 2 | 3 | 1866-1905 |
| 10 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 1905-2194 |
| 11 | move the shirt to the basket | 1 | na | 3 | 3 | 2194-2378 |
| 12 | release the shirt in the basket | 1 | drop | 2 | 3 | 2378-2416 |
| 13 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 2416-2742 |
| 14 | move the shirt to the basket | 1 | na | 3 | 3 | 2742-2915 |
| 15 | release the shirt in the basket | 1 | drop | 1 | 3 | 2915-2964 |
| 16 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 2964-3212 |
| 17 | move the shirt to the basket | 1 | na | 3 | 5 | 3212-3384 |
| 18 | release the shirt in the basket | 1 | drop | 2 | 5 | 3384-3429 |
| 19 | grasp the black sock | 1 | cloth-pinch | 2 | 1 | 3429-4200 |
| 20 | return to home | 1 | na | 4 | 1 | 4200-4503 |

```yaml
task: Put the shirt in the bin, then in the basket, then grasp the black sock
eval_subtasks:
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt
- move the shirt to the basket
- release the shirt in the basket
- grasp the black sock
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na]
```

### T030 Put the shirt, the white sock and the black sock in the bin, then grasp the white sock again

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 7. Table: episode 7 (5643 frames, mean quality 2.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt | 1 | cloth-pinch | 2 | 1 | 0-1407 |
| 2 | move the shirt to the bin | 1 | na | 3 | 3 | 1407-1530 |
| 3 | release the shirt in the bin | 1 | drop | 2 | 4 | 1530-1710 |
| 4 | grasp the white sock | 1 | cloth-pinch | 2 | 1 | 1710-2845 |
| 5 | move the white sock to the bin | 1 | na | 4 | 4 | 2845-3060 |
| 6 | release the white sock in the bin | 1 | drop | 3 | 2 | 3060-3168 |
| 7 | grasp the black sock | 1 | cloth-pinch | 2 | 2 | 3168-3870 |
| 8 | move the black sock to the bin | 1 | na | 4 | 3 | 3870-4095 |
| 9 | release the black sock in the bin | 1 | drop | 3 | 4 | 4095-4220 |
| 10 | grasp the sock | 1 | cloth-pinch | 2 | 1 | 4220-5643 |

```yaml
task: Put the shirt, the white sock and the black sock in the bin, then grasp the white sock again
eval_subtasks:
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the white sock
- move the white sock to the bin
- release the white sock in the bin
- grasp the black sock
- move the black sock to the bin
- release the black sock in the bin
- grasp the sock
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch]
```

### T031 Put the shirts in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 24. Table: episode 24 (1749 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shirt farther from the basket | 1 | cloth-pinch | 3 | 4 | 0-336 |
| 2 | move the shirt to the bin | 1 | na | 4 | 3 | 336-590 |
| 3 | release the shirt in the bin | 1 | drop | 4 | 4 | 590-680 |
| 4 | grasp the shirt | 1 | cloth-pinch | 4 | 4 | 680-1063 |
| 5 | move the shirt to the bin | 1 | na | 4 | 3 | 1063-1305 |
| 6 | release the shirt in the bin | 1 | drop | 3 | 4 | 1305-1460 |
| 7 | return to home | 1 | na | 5 | 4 | 1460-1749 |

```yaml
task: Put the shirts in the bin
eval_subtasks:
- grasp the shirt farther from the basket
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T032 Put the sock and the shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 16. Table: episode 16 (2775 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock | 1 | cloth-pinch | 2 | 5 | 0-518 |
| 2 | move the sock to the bin | 1 | na | 3 | 5 | 518-918 |
| 3 | release the sock in the bin | 1 | drop | 3 | 4 | 918-987 |
| 4 | grasp the shirt | 1 | cloth-pinch | 2 | 4 | 987-1393 |
| 5 | move the shirt to the bin | 1 | na | 4 | 3 | 1393-1497 |
| 6 | release the shirt in the bin | 1 | drop | 3 | 2 | 1497-1778 |
| 7 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 1778-2309 |
| 8 | move the shirt to the bin | 1 | na | 3 | 4 | 2309-2355 |
| 9 | release the shirt in the bin | 1 | set-down | 3 | 4 | 2355-2472 |
| 10 | return to home | 1 | na | 4 | 4 | 2472-2775 |

```yaml
task: Put the sock and the shirt in the bin
eval_subtasks:
- grasp the sock
- move the sock to the bin
- release the sock in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, set-down, na]
```

### T033 Put the sock and the shirts in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 43. Table: episode 43 (8007 frames, mean quality 3.3).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock | 1 | cloth-pinch | 1 | 1 | 0-1680 |
| 2 | move the sock to the basket | 1 | na | 3 | 4 | 1680-1944 |
| 3 | release the sock in the basket | 1 | drop | 2 | 4 | 1944-2016 |
| 4 | grasp the shirt nearest the basket | 1 | cloth-pinch | 2 | 2 | 2016-2576 |
| 5 | move the shirt to the basket | 1 | na | 3 | 3 | 2576-2829 |
| 6 | release the shirt in the basket | 1 | set-down | 3 | 3 | 2829-2904 |
| 7 | grasp the shirt hanging over the basket rim | 1 | cloth-pinch | 2 | 2 | 2904-3578 |
| 8 | move the shirt to the basket | 1 | na | 3 | 3 | 3578-3836 |
| 9 | release the shirt in the basket | 1 | set-down | 3 | 3 | 3836-3890 |
| 10 | grasp the shirt hanging over the basket rim | 1 | cloth-pinch | 2 | 2 | 3890-4620 |
| 11 | move the shirt to the basket | 1 | na | 3 | 4 | 4620-4740 |
| 12 | release the shirt in the basket | 1 | set-down | 4 | 3 | 4740-4935 |
| 13 | grasp the shirt hanging over the basket rim | 1 | cloth-pinch | 2 | 3 | 4935-5290 |
| 14 | move the shirt to the basket | 1 | na | 3 | 4 | 5290-5400 |
| 15 | release the shirt in the basket | 1 | set-down | 3 | 5 | 5400-5630 |
| 16 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 5630-6069 |
| 17 | move the shirt to the bin | 1 | na | 4 | 4 | 6069-6132 |
| 18 | move the shirt to the basket | 1 | na | 4 | 4 | 6132-6396 |
| 19 | release the shirt in the basket | 1 | drop | 4 | 3 | 6396-6560 |
| 20 | grasp the shirt | 1 | cloth-pinch | 2 | 3 | 6560-6885 |
| 21 | move the shirt to the basket | 1 | na | 3 | 4 | 6885-6990 |
| 22 | release the shirt in the basket | 1 | set-down | 3 | 3 | 6990-7155 |
| 23 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 7155-7512 |
| 24 | move the shirt to the basket | 1 | na | 3 | 4 | 7512-7590 |
| 25 | release the shirt in the basket | 1 | set-down | 4 | 4 | 7590-7746 |
| 26 | return to home | 1 | na | 5 | 4 | 7746-8007 |

```yaml
task: Put the sock and the shirts in the basket
eval_subtasks:
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- grasp the shirt nearest the basket
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt hanging over the basket rim
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt hanging over the basket rim
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt hanging over the basket rim
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt
- move the shirt to the bin
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt
- move the shirt to the basket
- release the shirt in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, set-down, cloth-pinch, na, set-down, cloth-pinch, na, set-down, cloth-pinch, na, set-down, cloth-pinch, na, na, drop, cloth-pinch, na, set-down, cloth-pinch, na, set-down, na]
```

### T034 Put the sock in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 3, 13, 44. Table: episode 3 (948 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock | 1 | cloth-pinch | 3 | 4 | 0-501 |
| 2 | move the sock to the basket | 1 | na | 4 | 4 | 501-840 |
| 3 | release the sock in the basket | 1 | drop | 3 | 4 | 840-948 |

```yaml
task: Put the sock in the basket
eval_subtasks:
- grasp the sock
- move the sock to the basket
- release the sock in the basket
eval_subtask_precisions: [1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop]
```

<details><summary>Other episodes (2)</summary>


Episode 13 (1308 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock | 1 | cloth-pinch | 4 | 3 | 0-627 |
| 2 | move the sock to the basket | 1 | na | 3 | 4 | 627-912 |
| 3 | release the sock in the basket | 1 | drop | 1 | 4 | 912-966 |
| 4 | return to home | 1 | na | 4 | 3 | 966-1308 |


Episode 44 (1254 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock | 1 | cloth-pinch | 2 | 1 | 0-867 |
| 2 | move the sock to the basket | 1 | na | 3 | 5 | 867-1110 |
| 3 | release the sock in the basket | 1 | drop | 3 | 4 | 1110-1149 |
| 4 | return to home | 1 | na | 5 | 4 | 1149-1254 |

</details>

### T035 Put the sock in the basket and the shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 45. Table: episode 45 (1770 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock | 1 | cloth-pinch | 2 | 3 | 0-450 |
| 2 | move the sock to the basket | 1 | na | 3 | 3 | 450-664 |
| 3 | release the sock in the basket | 1 | drop | 2 | 4 | 664-730 |
| 4 | grasp the shirt | 1 | cloth-pinch | 3 | 5 | 730-1098 |
| 5 | move the shirt to the bin | 1 | na | 3 | 3 | 1098-1320 |
| 6 | release the shirt in the bin | 1 | drop | 3 | 4 | 1320-1420 |
| 7 | return to home | 1 | na | 4 | 4 | 1420-1770 |

```yaml
task: Put the sock in the basket and the shirt in the bin
eval_subtasks:
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T036 Put the sock in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 14. Table: episode 14 (1119 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock | 1 | cloth-pinch | 3 | 3 | 0-675 |
| 2 | move the sock to the bin | 1 | na | 3 | 3 | 675-1042 |
| 3 | release the sock in the bin | 1 | drop | 3 | 3 | 1042-1119 |

```yaml
task: Put the sock in the bin
eval_subtasks:
- grasp the sock
- move the sock to the bin
- release the sock in the bin
eval_subtask_precisions: [1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop]
```

### T037 Put the sock in the bin and grasp the navy shirt

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 47. Table: episode 47 (954 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock | 1 | cloth-pinch | 3 | 3 | 0-420 |
| 2 | move the sock to the bin | 1 | na | 2 | 5 | 420-549 |
| 3 | release the sock in the bin | 1 | drop | 3 | 4 | 549-660 |
| 4 | grasp the navy shirt | 1 | cloth-pinch | 3 | 4 | 660-954 |

```yaml
task: Put the sock in the bin and grasp the navy shirt
eval_subtasks:
- grasp the sock
- move the sock to the bin
- release the sock in the bin
- grasp the navy shirt
eval_subtask_precisions: [1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch]
```

### T038 Put the sock on the navy shirt in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 46. Table: episode 46 (1101 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock on the navy shirt | 1 | cloth-pinch | 3 | 4 | 0-357 |
| 2 | move the sock to the basket | 1 | na | 2 | 3 | 357-720 |
| 3 | release the sock in the basket | 1 | drop | 2 | 4 | 720-783 |
| 4 | return to home | 1 | na | 4 | 4 | 783-1101 |

```yaml
task: Put the sock on the navy shirt in the basket
eval_subtasks:
- grasp the sock on the navy shirt
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, na]
```

### T039 Put the socks in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38. Table: episode 28 (4656 frames, mean quality 4.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock on top of the pile | 1 | cloth-pinch | 3 | 5 | 0-366 |
| 2 | move the white sock to the basket | 1 | na | 4 | 4 | 366-608 |
| 3 | release the white sock in the basket | 1 | drop | 4 | 4 | 608-660 |
| 4 | grasp the black sock lying on the white sock | 1 | cloth-pinch | 2 | 4 | 660-1122 |
| 5 | move the black sock to the basket | 1 | na | 4 | 5 | 1122-1370 |
| 6 | release the black sock in the basket | 1 | drop | 4 | 5 | 1370-1422 |
| 7 | grasp the white sock | 1 | cloth-pinch | 3 | 3 | 1422-1950 |
| 8 | move the white sock to the basket | 1 | na | 4 | 4 | 1950-2146 |
| 9 | release the white sock in the basket | 1 | drop | 3 | 4 | 2146-2224 |
| 10 | grasp the sock with grey stripes on top of the pile | 1 | cloth-pinch | 3 | 5 | 2224-2544 |
| 11 | move the sock with grey stripes to the basket | 1 | na | 3 | 5 | 2544-2820 |
| 12 | release the sock with grey stripes in the basket | 1 | drop | 2 | 5 | 2820-2876 |
| 13 | grasp the sock with grey stripes | 1 | cloth-pinch | 3 | 5 | 2876-3340 |
| 14 | move the sock with grey stripes to the basket | 1 | na | 4 | 4 | 3340-3556 |
| 15 | release the sock with grey stripes in the basket | 1 | drop | 2 | 4 | 3556-3646 |
| 16 | grasp the sock | 1 | cloth-pinch | 2 | 4 | 3646-4044 |
| 17 | move the sock to the basket | 1 | na | 3 | 5 | 4044-4282 |
| 18 | release the sock in the basket | 1 | drop | 3 | 5 | 4282-4333 |
| 19 | return to home | 1 | na | 5 | 5 | 4333-4656 |

```yaml
task: Put the socks in the basket
eval_subtasks:
- grasp the white sock on top of the pile
- move the white sock to the basket
- release the white sock in the basket
- grasp the black sock lying on the white sock
- move the black sock to the basket
- release the black sock in the basket
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock with grey stripes on top of the pile
- move the sock with grey stripes to the basket
- release the sock with grey stripes in the basket
- grasp the sock with grey stripes
- move the sock with grey stripes to the basket
- release the sock with grey stripes in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

<details><summary>Other episodes (11)</summary>


Episode 32 (3090 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock nearest the front edge of the table | 1 | cloth-pinch | 4 | 3 | 0-552 |
| 2 | move the black sock to the basket | 1 | na | 4 | 4 | 552-759 |
| 3 | release the black sock in the basket | 1 | drop | 3 | 4 | 759-858 |
| 4 | grasp the white sock nearest the front edge of the table | 1 | cloth-pinch | 2 | 4 | 858-1245 |
| 5 | move the white sock to the basket | 1 | na | 4 | 5 | 1245-1446 |
| 6 | release the white sock in the basket | 1 | drop | 4 | 5 | 1446-1536 |
| 7 | grasp the black sock | 1 | cloth-pinch | 3 | 3 | 1536-1940 |
| 8 | move the black sock to the basket | 1 | na | 4 | 4 | 1940-2112 |
| 9 | release the black sock in the basket | 1 | drop | 3 | 5 | 2112-2241 |
| 10 | grasp the sock | 1 | cloth-pinch | 3 | 5 | 2241-2572 |
| 11 | move the sock to the basket | 1 | na | 5 | 4 | 2572-2730 |
| 12 | release the sock in the basket | 1 | drop | 5 | 5 | 2730-2851 |
| 13 | return to home | 1 | na | 5 | 5 | 2851-3090 |


Episode 26 (2244 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock farther from the basket | 1 | cloth-pinch | 3 | 5 | 0-411 |
| 2 | move the sock to the basket | 1 | na | 3 | 5 | 411-678 |
| 3 | release the sock in the basket | 1 | drop | 3 | 5 | 678-708 |
| 4 | grasp the sock | 1 | cloth-pinch | 3 | 5 | 708-1278 |
| 5 | move the sock to the basket | 1 | na | 3 | 3 | 1278-1587 |
| 6 | release the sock in the basket | 1 | set-down | 1 | 3 | 1587-1788 |
| 7 | return to home | 1 | na | 4 | 4 | 1788-2244 |


Episode 38 (4494 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the plain white sock on top of the pile | 1 | cloth-pinch | 3 | 5 | 0-342 |
| 2 | move the plain white sock to the basket | 1 | na | 5 | 5 | 342-549 |
| 3 | release the plain white sock in the basket | 1 | drop | 4 | 5 | 549-582 |
| 4 | grasp the plain white sock | 1 | cloth-pinch | 2 | 5 | 582-1041 |
| 5 | move the plain white sock to the basket | 1 | na | 3 | 5 | 1041-1260 |
| 6 | release the plain white sock in the basket | 1 | drop | 2 | 5 | 1260-1290 |
| 7 | grasp the black sock nearest the basket | 1 | cloth-pinch | 2 | 2 | 1290-1806 |
| 8 | move the black sock to the basket | 1 | na | 4 | 3 | 1806-2013 |
| 9 | release the black sock in the basket | 1 | drop | 2 | 3 | 2013-2100 |
| 10 | grasp the black sock | 1 | cloth-pinch | 3 | 3 | 2100-2535 |
| 11 | move the black sock to the basket | 1 | na | 4 | 5 | 2535-2752 |
| 12 | release the black sock in the basket | 1 | drop | 1 | 5 | 2752-2790 |
| 13 | grasp the sock nearest the basket | 1 | cloth-pinch | 2 | 3 | 2790-3250 |
| 14 | move the sock to the basket | 1 | na | 3 | 5 | 3250-3460 |
| 15 | release the sock in the basket | 1 | drop | 2 | 5 | 3460-3528 |
| 16 | grasp the sock | 1 | cloth-pinch | 2 | 2 | 3528-3993 |
| 17 | move the sock to the basket | 1 | na | 4 | 5 | 3993-4204 |
| 18 | release the sock in the basket | 1 | drop | 2 | 5 | 4204-4240 |
| 19 | return to home | 1 | na | 5 | 5 | 4240-4494 |


Episode 35 (2082 frames, mean quality 4.1)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock nearest the basket | 1 | cloth-pinch | 3 | 3 | 0-801 |
| 2 | move the sock to the basket | 1 | na | 2 | 4 | 801-1002 |
| 3 | release the sock in the basket | 1 | drop | 2 | 4 | 1002-1059 |
| 4 | grasp the sock | 1 | cloth-pinch | 2 | 3 | 1059-1478 |
| 5 | move the sock to the basket | 1 | na | 3 | 5 | 1478-1710 |
| 6 | release the sock in the basket | 1 | drop | 2 | 5 | 1710-1764 |
| 7 | return to home | 1 | na | 4 | 5 | 1764-2082 |


Episode 33 (4317 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock nearest the basket | 1 | cloth-pinch | 4 | 4 | 0-381 |
| 2 | move the white sock to the basket | 1 | na | 3 | 4 | 381-591 |
| 3 | release the white sock in the basket | 1 | drop | 2 | 4 | 591-708 |
| 4 | grasp the striped sock farthest from the robot | 1 | cloth-pinch | 3 | 3 | 708-1094 |
| 5 | move the striped sock to the basket | 1 | na | 3 | 4 | 1094-1326 |
| 6 | release the striped sock in the basket | 1 | drop | 2 | 4 | 1326-1419 |
| 7 | grasp the white sock | 1 | cloth-pinch | 2 | 4 | 1419-1812 |
| 8 | move the white sock to the basket | 1 | na | 3 | 4 | 1812-2080 |
| 9 | release the white sock in the basket | 1 | drop | 2 | 4 | 2080-2160 |
| 10 | grasp the plain sock farthest from the basket | 1 | cloth-pinch | 3 | 3 | 2160-2494 |
| 11 | move the plain sock to the basket | 1 | na | 4 | 5 | 2494-2730 |
| 12 | release the plain sock in the basket | 1 | drop | 2 | 4 | 2730-2808 |
| 13 | grasp the plain sock | 1 | cloth-pinch | 3 | 4 | 2808-3100 |
| 14 | move the plain sock to the basket | 1 | na | 4 | 4 | 3100-3308 |
| 15 | release the plain sock in the basket | 1 | drop | 2 | 4 | 3308-3389 |
| 16 | grasp the sock | 1 | cloth-pinch | 3 | 5 | 3389-3684 |
| 17 | move the sock to the basket | 1 | na | 4 | 4 | 3684-3912 |
| 18 | release the sock in the basket | 1 | drop | 2 | 4 | 3912-4017 |
| 19 | return to home | 1 | na | 5 | 4 | 4017-4317 |


Episode 27 (3600 frames, mean quality 3.9)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock farther from the basket | 1 | cloth-pinch | 3 | 2 | 0-732 |
| 2 | move the black sock to the basket | 1 | na | 3 | 5 | 732-993 |
| 3 | release the black sock in the basket | 1 | drop | 2 | 4 | 993-1074 |
| 4 | grasp the white sock farther from the basket | 1 | cloth-pinch | 2 | 4 | 1074-1470 |
| 5 | move the white sock to the basket | 1 | na | 3 | 4 | 1470-1702 |
| 6 | release the white sock in the basket | 1 | drop | 2 | 4 | 1702-1797 |
| 7 | grasp the white sock | 1 | cloth-pinch | 2 | 3 | 1797-2253 |
| 8 | move the white sock to the basket | 1 | na | 3 | 4 | 2253-2454 |
| 9 | release the white sock in the basket | 1 | drop | 2 | 4 | 2454-2547 |
| 10 | grasp the sock | 1 | cloth-pinch | 3 | 5 | 2547-2895 |
| 11 | move the sock to the basket | 1 | na | 3 | 4 | 2895-3120 |
| 12 | release the sock in the basket | 1 | drop | 2 | 4 | 3120-3212 |
| 13 | return to home | 1 | na | 4 | 4 | 3212-3600 |


Episode 29 (7125 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock lying on the white sock | 1 | cloth-pinch | 4 | 4 | 0-465 |
| 2 | move the black sock to the basket | 1 | na | 4 | 4 | 465-735 |
| 3 | release the black sock in the basket | 1 | drop | 3 | 4 | 735-805 |
| 4 | grasp the white sock nearest the basket | 1 | cloth-pinch | 1 | 1 | 805-1618 |
| 5 | move the white sock to the basket | 1 | na | 4 | 3 | 1618-1856 |
| 6 | release the white sock in the basket | 1 | drop | 2 | 4 | 1856-1928 |
| 7 | grasp the white sock farthest from the robot | 1 | cloth-pinch | 4 | 5 | 1928-2408 |
| 8 | move the white sock to the basket | 1 | na | 3 | 5 | 2408-2660 |
| 9 | release the white sock in the basket | 1 | drop | 1 | 4 | 2660-2757 |
| 10 | grasp the white sock nearest the basket | 1 | cloth-pinch | 2 | 1 | 2757-3508 |
| 11 | move the white sock to the basket | 1 | na | 3 | 3 | 3508-3764 |
| 12 | release the white sock in the basket | 1 | drop | 2 | 4 | 3764-3902 |
| 13 | grasp the black sock nearest the basket | 1 | cloth-pinch | 3 | 5 | 3902-4372 |
| 14 | move the black sock to the basket | 1 | na | 3 | 5 | 4372-4576 |
| 15 | release the black sock in the basket | 1 | drop | 3 | 4 | 4576-4680 |
| 16 | grasp the black sock nearest the basket | 1 | cloth-pinch | 4 | 4 | 4680-5057 |
| 17 | move the black sock to the basket | 1 | na | 4 | 5 | 5057-5264 |
| 18 | release the black sock in the basket | 1 | drop | 2 | 4 | 5264-5345 |
| 19 | grasp the black sock | 1 | cloth-pinch | 3 | 2 | 5345-5840 |
| 20 | move the black sock to the basket | 1 | na | 5 | 5 | 5840-6044 |
| 21 | release the black sock in the basket | 1 | drop | 2 | 5 | 6044-6100 |
| 22 | grasp the sock | 1 | cloth-pinch | 3 | 3 | 6100-6463 |
| 23 | move the sock to the basket | 1 | na | 4 | 3 | 6463-6740 |
| 24 | release the sock in the basket | 1 | drop | 2 | 4 | 6740-6815 |
| 25 | return to home | 1 | na | 5 | 4 | 6815-7125 |


Episode 30 (7668 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock nearest the basket | 1 | cloth-pinch | 3 | 5 | 0-333 |
| 2 | move the white sock to the basket | 1 | na | 4 | 5 | 333-540 |
| 3 | release the white sock in the basket | 1 | drop | 3 | 4 | 540-576 |
| 4 | grasp the plain black sock nearest the front edge of the table | 1 | cloth-pinch | 2 | 4 | 576-951 |
| 5 | move the plain black sock to the basket | 1 | na | 3 | 5 | 951-1178 |
| 6 | release the plain black sock in the basket | 1 | drop | 2 | 4 | 1178-1272 |
| 7 | grasp the white sock lying on the plain black sock | 1 | cloth-pinch | 2 | 4 | 1272-1590 |
| 8 | move the white sock to the basket | 1 | na | 3 | 4 | 1590-1800 |
| 9 | release the white sock in the basket | 1 | drop | 2 | 4 | 1800-1863 |
| 10 | grasp the striped sock nearest the front edge of the table | 1 | cloth-pinch | 2 | 4 | 1863-2238 |
| 11 | move the striped sock to the basket | 1 | na | 4 | 5 | 2238-2451 |
| 12 | release the striped sock in the basket | 1 | drop | 2 | 4 | 2451-2511 |
| 13 | grasp the plain black sock nearest the front edge of the table | 1 | cloth-pinch | 2 | 4 | 2511-2862 |
| 14 | move the plain black sock to the basket | 1 | na | 3 | 5 | 2862-3082 |
| 15 | release the plain black sock in the basket | 1 | drop | 3 | 4 | 3082-3157 |
| 16 | grasp the plain black sock nearest the basket | 1 | cloth-pinch | 3 | 4 | 3157-3515 |
| 17 | move the plain black sock to the basket | 1 | na | 4 | 4 | 3515-3760 |
| 18 | release the plain black sock in the basket | 1 | drop | 3 | 4 | 3760-3820 |
| 19 | grasp the white sock nearest the basket | 1 | cloth-pinch | 3 | 4 | 3820-4184 |
| 20 | move the white sock to the basket | 1 | na | 4 | 5 | 4184-4402 |
| 21 | release the white sock in the basket | 1 | drop | 3 | 4 | 4402-4477 |
| 22 | grasp the white sock nearest the front edge of the table | 1 | cloth-pinch | 3 | 4 | 4477-4882 |
| 23 | move the white sock to the basket | 1 | na | 4 | 2 | 4882-5010 |
| 24 | grasp the plain black sock | 1 | cloth-pinch | 2 | 2 | 5010-5226 |
| 25 | move the plain black sock to the basket | 1 | na | 4 | 3 | 5226-5404 |
| 26 | release the plain black sock in the basket | 1 | drop | 2 | 1 | 5404-5655 |
| 27 | grasp the white sock | 1 | cloth-pinch | 2 | 1 | 5655-6156 |
| 28 | move the white sock to the basket | 1 | na | 4 | 3 | 6156-6324 |
| 29 | release the white sock in the basket | 1 | drop | 2 | 3 | 6324-6531 |
| 30 | grasp the striped sock | 1 | cloth-pinch | 4 | 3 | 6531-7095 |
| 31 | move the striped sock to the basket | 1 | na | 4 | 5 | 7095-7336 |
| 32 | release the striped sock in the basket | 1 | drop | 3 | 4 | 7336-7390 |
| 33 | return to home | 1 | na | 5 | 4 | 7390-7668 |


Episode 36 (4368 frames, mean quality 3.6)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock farther from the basket | 1 | cloth-pinch | 2 | 2 | 0-945 |
| 2 | move the white sock to the basket | 1 | na | 4 | 5 | 945-1160 |
| 3 | release the white sock in the basket | 1 | drop | 2 | 4 | 1160-1210 |
| 4 | grasp the white sock | 1 | cloth-pinch | 2 | 4 | 1210-1734 |
| 5 | move the white sock to the basket | 1 | na | 3 | 3 | 1734-2000 |
| 6 | release the white sock in the basket | 1 | drop | 2 | 4 | 2000-2040 |
| 7 | grasp the sock nearest the basket | 1 | cloth-pinch | 2 | 2 | 2040-2724 |
| 8 | move the sock to the basket | 1 | na | 3 | 3 | 2724-2964 |
| 9 | release the sock in the basket | 1 | drop | 2 | 4 | 2964-3033 |
| 10 | grasp the sock | 1 | cloth-pinch | 2 | 2 | 3033-3816 |
| 11 | move the sock to the basket | 1 | na | 3 | 5 | 3816-4066 |
| 12 | release the sock in the basket | 1 | drop | 3 | 4 | 4066-4125 |
| 13 | return to home | 1 | na | 5 | 5 | 4125-4368 |


Episode 34 (1992 frames, mean quality 3.4)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock nearest the basket | 1 | cloth-pinch | 4 | 3 | 0-420 |
| 2 | move the sock to the basket | 1 | na | 3 | 5 | 420-658 |
| 3 | release the sock in the basket | 1 | drop | 2 | 4 | 658-729 |
| 4 | grasp the sock | 1 | cloth-pinch | 2 | 3 | 729-1296 |
| 5 | move the sock to the basket | 1 | na | 4 | 3 | 1296-1588 |
| 6 | release the sock in the basket | 1 | drop | 1 | 3 | 1588-1648 |
| 7 | return to home | 1 | na | 5 | 3 | 1648-1992 |


Episode 31 (4104 frames, mean quality 3.4)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock nearest the front edge of the table | 1 | cloth-pinch | 4 | 3 | 0-504 |
| 2 | move the sock to the basket | 1 | na | 4 | 3 | 504-738 |
| 3 | release the sock in the basket | 1 | drop | 3 | 4 | 738-861 |
| 4 | grasp the sock nearest the basket | 1 | cloth-pinch | 3 | 3 | 861-1335 |
| 5 | move the sock to the basket | 1 | na | 4 | 4 | 1335-1582 |
| 6 | release the sock in the basket | 1 | drop | 3 | 4 | 1582-1680 |
| 7 | grasp the sock nearest the front edge of the table | 1 | cloth-pinch | 2 | 1 | 1680-2532 |
| 8 | move the sock to the basket | 1 | na | 4 | 5 | 2532-2827 |
| 9 | release the sock in the basket | 1 | drop | 2 | 4 | 2827-2907 |
| 10 | grasp the sock | 1 | cloth-pinch | 3 | 2 | 2907-3743 |
| 11 | move the sock to the basket | 1 | na | 5 | 4 | 3743-3994 |
| 12 | release the sock in the basket | 1 | drop | 3 | 4 | 3994-4104 |

</details>

### T040 Put the socks in the basket and the shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 63. Table: episode 63 (2148 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock farther from the robot | 1 | cloth-pinch | 5 | 3 | 0-384 |
| 2 | move the sock to the basket | 1 | na | 3 | 5 | 384-552 |
| 3 | release the sock in the basket | 1 | drop | 2 | 4 | 552-630 |
| 4 | grasp the sock | 1 | cloth-pinch | 3 | 3 | 630-957 |
| 5 | move the sock to the basket | 1 | na | 3 | 4 | 957-1170 |
| 6 | release the sock in the basket | 1 | drop | 3 | 4 | 1170-1251 |
| 7 | grasp the shirt | 1 | cloth-pinch | 4 | 4 | 1251-1548 |
| 8 | move the shirt to the bin | 1 | na | 4 | 3 | 1548-1752 |
| 9 | release the shirt in the bin | 1 | drop | 4 | 4 | 1752-1896 |
| 10 | return to home | 1 | na | 5 | 4 | 1896-2148 |

```yaml
task: Put the socks in the basket and the shirt in the bin
eval_subtasks:
- grasp the sock farther from the robot
- move the sock to the basket
- release the sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T041 Put the socks with the blue toe, the white socks and the black socks in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 37. Table: episode 37 (6015 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock with the blue toe on top of the pile | 1 | cloth-pinch | 3 | 4 | 0-560 |
| 2 | move the sock with the blue toe to the basket | 1 | na | 4 | 5 | 560-855 |
| 3 | release the sock with the blue toe in the basket | 1 | drop | 1 | 5 | 855-903 |
| 4 | grasp the sock with the blue toe | 1 | cloth-pinch | 2 | 2 | 903-1686 |
| 5 | move the sock with the blue toe to the basket | 1 | na | 3 | 3 | 1686-1985 |
| 6 | release the sock with the blue toe in the basket | 1 | drop | 2 | 5 | 1985-2030 |
| 7 | grasp the white sock on top of the pile | 1 | cloth-pinch | 2 | 3 | 2030-2536 |
| 8 | move the white sock to the basket | 1 | na | 4 | 3 | 2536-2784 |
| 9 | release the white sock in the basket | 1 | drop | 3 | 4 | 2784-2856 |
| 10 | grasp the white sock | 1 | cloth-pinch | 1 | 2 | 2856-3473 |
| 11 | move the white sock to the basket | 1 | na | 4 | 5 | 3473-3712 |
| 12 | release the white sock in the basket | 1 | drop | 2 | 5 | 3712-3764 |
| 13 | grasp the black sock farther from the robot | 1 | cloth-pinch | 2 | 3 | 3764-4176 |
| 14 | move the black sock to the basket | 1 | na | 4 | 3 | 4176-4406 |
| 15 | release the black sock in the basket | 1 | drop | 2 | 5 | 4406-4442 |
| 16 | grasp the sock | 1 | cloth-pinch | 2 | 1 | 4442-5516 |
| 17 | move the sock to the basket | 1 | na | 4 | 3 | 5516-5735 |
| 18 | release the sock in the basket | 1 | drop | 3 | 4 | 5735-5773 |
| 19 | return to home | 1 | na | 5 | 5 | 5773-6015 |

```yaml
task: Put the socks with the blue toe, the white socks and the black socks in the basket
eval_subtasks:
- grasp the sock with the blue toe on top of the pile
- move the sock with the blue toe to the basket
- release the sock with the blue toe in the basket
- grasp the sock with the blue toe
- move the sock with the blue toe to the basket
- release the sock with the blue toe in the basket
- grasp the white sock on top of the pile
- move the white sock to the basket
- release the white sock in the basket
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the black sock farther from the robot
- move the black sock to the basket
- release the black sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T042 Put the white shirt and the grey shirt in the bin, the black sock and the white socks in the basket, and grasp the other black sock

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 56. Table: episode 56 (4329 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white shirt | 1 | cloth-pinch | 3 | 5 | 0-306 |
| 2 | move the white shirt to the bin | 1 | na | 4 | 4 | 306-580 |
| 3 | release the white shirt in the bin | 1 | drop | 2 | 4 | 580-643 |
| 4 | grasp the shirt | 1 | cloth-pinch | 3 | 4 | 643-1016 |
| 5 | move the shirt to the bin | 1 | na | 3 | 3 | 1016-1270 |
| 6 | release the shirt in the bin | 1 | drop | 2 | 4 | 1270-1375 |
| 7 | grasp the black sock next to the white socks | 1 | cloth-pinch | 2 | 5 | 1375-1704 |
| 8 | move the black sock to the basket | 1 | na | 3 | 5 | 1704-1960 |
| 9 | release the black sock in the basket | 1 | drop | 2 | 4 | 1960-2085 |
| 10 | grasp the white sock nearest the basket | 1 | cloth-pinch | 3 | 4 | 2085-2434 |
| 11 | move the white sock to the basket | 1 | na | 3 | 4 | 2434-2672 |
| 12 | release the white sock in the basket | 1 | drop | 2 | 4 | 2672-2750 |
| 13 | grasp the white sock | 1 | cloth-pinch | 3 | 4 | 2750-3097 |
| 14 | move the white sock to the basket | 1 | na | 4 | 5 | 3097-3326 |
| 15 | release the white sock in the basket | 1 | drop | 2 | 5 | 3326-3382 |
| 16 | grasp the sock | 1 | cloth-pinch | 2 | 1 | 3382-4062 |
| 17 | return to home | 1 | na | 5 | 1 | 4062-4329 |

```yaml
task: Put the white shirt and the grey shirt in the bin, the black sock and the white socks in the basket, and grasp the other black sock
eval_subtasks:
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the black sock next to the white socks
- move the black sock to the basket
- release the black sock in the basket
- grasp the white sock nearest the basket
- move the white sock to the basket
- release the white sock in the basket
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na]
```

### T043 Put the white shirt in the bin, the black sock with grey stripes, a white sock and the black sock on the navy shirt in the basket, the navy shirt in the bin, the other white sock and the black sock on the grey shirt in the basket, and the grey shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 58. Table: episode 58 (6093 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white shirt | 1 | cloth-pinch | 3 | 4 | 0-260 |
| 2 | move the white shirt to the bin | 1 | na | 4 | 3 | 260-530 |
| 3 | release the white shirt in the bin | 1 | drop | 4 | 4 | 530-630 |
| 4 | grasp the black sock with grey stripes | 1 | cloth-pinch | 4 | 4 | 630-1020 |
| 5 | move the black sock with grey stripes to the basket | 1 | na | 3 | 4 | 1020-1284 |
| 6 | release the black sock with grey stripes in the basket | 1 | drop | 2 | 4 | 1284-1368 |
| 7 | grasp the white sock on the navy shirt | 1 | cloth-pinch | 2 | 3 | 1368-1620 |
| 8 | move the white sock to the basket | 1 | na | 3 | 3 | 1620-1815 |
| 9 | release the white sock in the basket | 1 | drop | 2 | 4 | 1815-1866 |
| 10 | grasp the sock on the navy shirt | 1 | cloth-pinch | 3 | 4 | 1866-2170 |
| 11 | move the sock to the basket | 1 | na | 2 | 3 | 2170-2508 |
| 12 | release the sock in the basket | 1 | drop | 2 | 4 | 2508-2645 |
| 13 | grasp the navy shirt | 1 | cloth-pinch | 2 | 4 | 2645-2916 |
| 14 | move the navy shirt to the bin | 1 | na | 3 | 3 | 2916-3180 |
| 15 | release the navy shirt in the bin | 1 | drop | 2 | 3 | 3180-3255 |
| 16 | grasp the navy shirt | 1 | cloth-pinch | 3 | 3 | 3255-3560 |
| 17 | move the navy shirt to the bin | 1 | na | 3 | 3 | 3560-3720 |
| 18 | release the navy shirt in the bin | 1 | drop | 3 | 4 | 3720-3775 |
| 19 | grasp the white sock | 1 | cloth-pinch | 4 | 3 | 3775-4126 |
| 20 | move the white sock to the basket | 1 | na | 4 | 4 | 4126-4350 |
| 21 | release the white sock in the basket | 1 | drop | 2 | 4 | 4350-4430 |
| 22 | grasp the sock in the middle of the shirt | 1 | cloth-pinch | 2 | 4 | 4430-4810 |
| 23 | move the sock to the basket | 1 | na | 4 | 4 | 4810-5040 |
| 24 | release the sock in the basket | 1 | drop | 2 | 4 | 5040-5110 |
| 25 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 5110-5510 |
| 26 | move the shirt to the bin | 1 | na | 4 | 3 | 5510-5750 |
| 27 | release the shirt in the bin | 1 | drop | 3 | 4 | 5750-5815 |
| 28 | return to home | 1 | na | 5 | 4 | 5815-6093 |

```yaml
task: Put the white shirt in the bin, the black sock with grey stripes, a white sock and the black sock on the navy shirt in the basket, the navy shirt in the bin, the other white sock and the black sock on the grey shirt in the basket, and the grey shirt in the bin
eval_subtasks:
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the black sock with grey stripes
- move the black sock with grey stripes to the basket
- release the black sock with grey stripes in the basket
- grasp the white sock on the navy shirt
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock on the navy shirt
- move the sock to the basket
- release the sock in the basket
- grasp the navy shirt
- move the navy shirt to the bin
- release the navy shirt in the bin
- grasp the navy shirt
- move the navy shirt to the bin
- release the navy shirt in the bin
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock in the middle of the shirt
- move the sock to the basket
- release the sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T044 Put the white shirt in the bin, the black socks and the white socks in the basket and the brown shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 66. Table: episode 66 (6450 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white shirt | 1 | cloth-pinch | 4 | 4 | 0-288 |
| 2 | move the white shirt to the bin | 1 | na | 3 | 3 | 288-717 |
| 3 | release the white shirt in the bin | 1 | drop | 2 | 3 | 717-792 |
| 4 | grasp the white shirt | 1 | cloth-pinch | 3 | 3 | 792-1002 |
| 5 | move the white shirt to the bin | 1 | na | 2 | 4 | 1002-1161 |
| 6 | release the white shirt in the bin | 1 | drop | 3 | 4 | 1161-1245 |
| 7 | grasp the striped black sock next to the bin | 1 | cloth-pinch | 3 | 3 | 1245-1488 |
| 8 | move the striped black sock to the basket | 1 | na | 4 | 3 | 1488-1698 |
| 9 | release the striped black sock in the basket | 1 | drop | 3 | 4 | 1698-1749 |
| 10 | grasp the black sock nearest the basket | 1 | cloth-pinch | 3 | 2 | 1749-2202 |
| 11 | move the black sock to the basket | 1 | na | 4 | 5 | 2202-2352 |
| 12 | release the black sock in the basket | 1 | drop | 3 | 4 | 2352-2391 |
| 13 | grasp the white sock nearest the basket | 1 | cloth-pinch | 3 | 2 | 2391-2853 |
| 14 | move the white sock to the basket | 1 | na | 4 | 5 | 2853-3036 |
| 15 | release the white sock in the basket | 1 | drop | 3 | 4 | 3036-3081 |
| 16 | grasp the black sock next to the white sock | 1 | cloth-pinch | 3 | 3 | 3081-3429 |
| 17 | move the black sock to the basket | 1 | na | 5 | 5 | 3429-3621 |
| 18 | release the black sock in the basket | 1 | drop | 4 | 4 | 3621-3663 |
| 19 | grasp the white sock | 1 | cloth-pinch | 3 | 3 | 3663-4050 |
| 20 | move the white sock to the basket | 1 | na | 4 | 5 | 4050-4287 |
| 21 | release the white sock in the basket | 1 | drop | 3 | 4 | 4287-4314 |
| 22 | grasp the sock | 1 | cloth-pinch | 4 | 3 | 4314-4767 |
| 23 | move the sock to the basket | 1 | na | 5 | 5 | 4767-4995 |
| 24 | release the sock in the basket | 1 | drop | 2 | 4 | 4995-5022 |
| 25 | grasp the shirt | 1 | cloth-pinch | 3 | 1 | 5022-5952 |
| 26 | move the shirt to the bin | 1 | na | 5 | 5 | 5952-6180 |
| 27 | release the shirt in the bin | 1 | drop | 4 | 4 | 6180-6270 |
| 28 | return to home | 1 | na | 5 | 5 | 6270-6450 |

```yaml
task: Put the white shirt in the bin, the black socks and the white socks in the basket and the brown shirt in the bin
eval_subtasks:
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the striped black sock next to the bin
- move the striped black sock to the basket
- release the striped black sock in the basket
- grasp the black sock nearest the basket
- move the black sock to the basket
- release the black sock in the basket
- grasp the white sock nearest the basket
- move the white sock to the basket
- release the white sock in the basket
- grasp the black sock next to the white sock
- move the black sock to the basket
- release the black sock in the basket
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T045 Put the white shirt, the grey shirt and the brown shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 19. Table: episode 19 (4227 frames, mean quality 3.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white shirt | 1 | cloth-pinch | 3 | 2 | 0-366 |
| 2 | move the white shirt to the bin | 1 | na | 5 | 3 | 366-586 |
| 3 | release the white shirt in the bin | 1 | drop | 3 | 3 | 586-672 |
| 4 | grasp the white shirt | 1 | cloth-pinch | 5 | 3 | 672-900 |
| 5 | move the white shirt to the bin | 1 | na | 4 | 4 | 900-975 |
| 6 | release the white shirt in the bin | 1 | drop | 3 | 5 | 975-1090 |
| 7 | grasp the grey shirt | 1 | cloth-pinch | 4 | 5 | 1090-1400 |
| 8 | move the grey shirt to the bin | 1 | na | 4 | 3 | 1400-1595 |
| 9 | release the grey shirt in the bin | 1 | drop | 4 | 3 | 1595-1700 |
| 10 | grasp the grey shirt | 1 | cloth-pinch | 2 | 2 | 1700-2172 |
| 11 | move the grey shirt to the bin | 1 | na | 3 | 3 | 2172-2222 |
| 12 | release the grey shirt in the bin | 1 | drop | 3 | 3 | 2222-2392 |
| 13 | grasp the grey shirt | 1 | cloth-pinch | 4 | 3 | 2392-2652 |
| 14 | move the grey shirt to the bin | 1 | na | 4 | 4 | 2652-2745 |
| 15 | release the grey shirt in the bin | 1 | drop | 4 | 5 | 2745-2852 |
| 16 | grasp the shirt | 1 | cloth-pinch | 4 | 5 | 2852-3135 |
| 17 | move the shirt to the bin | 1 | na | 4 | 3 | 3135-3325 |
| 18 | release the shirt in the bin | 1 | drop | 3 | 3 | 3325-3430 |
| 19 | grasp the shirt | 1 | cloth-pinch | 3 | 2 | 3430-3764 |
| 20 | move the shirt to the bin | 1 | na | 4 | 4 | 3764-3834 |
| 21 | release the shirt in the bin | 1 | drop | 4 | 4 | 3834-3964 |
| 22 | return to home | 1 | na | 5 | 4 | 3964-4227 |

```yaml
task: Put the white shirt, the grey shirt and the brown shirt in the bin
eval_subtasks:
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the grey shirt
- move the grey shirt to the bin
- release the grey shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T046 Put the white shirt, the navy shirt and the black shirt in the bin and the socks in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 64. Table: episode 64 (3465 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white shirt | 1 | cloth-pinch | 4 | 4 | 0-294 |
| 2 | move the white shirt to the bin | 1 | na | 4 | 4 | 294-594 |
| 3 | release the white shirt in the bin | 1 | drop | 2 | 4 | 594-699 |
| 4 | grasp the navy shirt | 1 | cloth-pinch | 3 | 3 | 699-1127 |
| 5 | move the navy shirt to the bin | 1 | na | 3 | 4 | 1127-1260 |
| 6 | release the navy shirt in the bin | 1 | drop | 3 | 4 | 1260-1386 |
| 7 | grasp the shirt | 1 | cloth-pinch | 4 | 4 | 1386-1698 |
| 8 | move the shirt to the bin | 1 | na | 3 | 3 | 1698-2005 |
| 9 | release the shirt in the bin | 1 | drop | 4 | 4 | 2005-2163 |
| 10 | grasp the sock nearer the robot | 1 | cloth-pinch | 3 | 4 | 2163-2497 |
| 11 | move the sock to the basket | 1 | na | 4 | 3 | 2497-2712 |
| 12 | release the sock in the basket | 1 | drop | 3 | 4 | 2712-2790 |
| 13 | grasp the sock | 1 | cloth-pinch | 3 | 4 | 2790-3084 |
| 14 | move the sock to the basket | 1 | na | 4 | 5 | 3084-3222 |
| 15 | release the sock in the basket | 1 | drop | 4 | 4 | 3222-3291 |
| 16 | return to home | 1 | na | 5 | 5 | 3291-3465 |

```yaml
task: Put the white shirt, the navy shirt and the black shirt in the bin and the socks in the basket
eval_subtasks:
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the navy shirt
- move the navy shirt to the bin
- release the navy shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the sock nearer the robot
- move the sock to the basket
- release the sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T047 Put the white sock and the black sock in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 55, 85. Table: episode 55 (1779 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock | 1 | cloth-pinch | 3 | 5 | 0-501 |
| 2 | move the white sock to the basket | 1 | na | 3 | 5 | 501-756 |
| 3 | release the white sock in the basket | 1 | drop | 2 | 4 | 756-852 |
| 4 | grasp the sock | 1 | cloth-pinch | 2 | 4 | 852-1245 |
| 5 | move the sock to the basket | 1 | na | 3 | 4 | 1245-1404 |
| 6 | release the sock in the basket | 1 | drop | 3 | 3 | 1404-1515 |
| 7 | return to home | 1 | na | 5 | 3 | 1515-1779 |

```yaml
task: Put the white sock and the black sock in the basket
eval_subtasks:
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

<details><summary>Other episodes (1)</summary>


Episode 85 (1944 frames, mean quality 3.6)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock | 1 | cloth-pinch | 3 | 3 | 0-540 |
| 2 | move the white sock to the basket | 1 | na | 3 | 4 | 540-840 |
| 3 | release the white sock in the basket | 1 | drop | 4 | 4 | 840-906 |
| 4 | grasp the sock | 1 | cloth-pinch | 2 | 4 | 906-1320 |
| 5 | move the sock to the basket | 1 | na | 3 | 3 | 1320-1540 |
| 6 | release the sock in the basket | 1 | drop | 2 | 3 | 1540-1608 |
| 7 | return to home | 1 | na | 4 | 4 | 1608-1944 |

</details>

### T048 Put the white sock in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 2, 9. Table: episode 9 (750 frames, mean quality 4.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock | 1 | cloth-pinch | 3 | 5 | 0-462 |
| 2 | move the white sock to the basket | 1 | na | 3 | 5 | 462-699 |
| 3 | release the white sock in the basket | 1 | drop | 2 | 4 | 699-750 |

```yaml
task: Put the white sock in the basket
eval_subtasks:
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
eval_subtask_precisions: [1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop]
```

<details><summary>Other episodes (1)</summary>


Episode 2 (1020 frames, mean quality 2.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock | 1 | cloth-pinch | 3 | 5 | 0-441 |
| 2 | move the white sock to the basket | 1 | na | 3 | 1 | 441-668 |
| 3 | release the white sock in the basket | 1 | drop | 3 | 1 | 668-720 |
| 4 | return to home | 1 | na | 4 | 1 | 720-1020 |

</details>

### T049 Put the white sock in the basket and the black sock in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 51. Table: episode 51 (1989 frames, mean quality 3.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock | 1 | cloth-pinch | 3 | 4 | 0-459 |
| 2 | move the white sock to the basket | 1 | na | 4 | 3 | 459-738 |
| 3 | release the white sock in the basket | 1 | drop | 3 | 4 | 738-834 |
| 4 | grasp the sock | 1 | cloth-pinch | 3 | 5 | 834-1240 |
| 5 | move the sock to the bin | 1 | na | 3 | 1 | 1240-1398 |
| 6 | release the sock in the bin | 1 | drop | 3 | 1 | 1398-1608 |
| 7 | return to home | 1 | na | 4 | 4 | 1608-1989 |

```yaml
task: Put the white sock in the basket and the black sock in the bin
eval_subtasks:
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock
- move the sock to the bin
- release the sock in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T050 Put the white sock in the basket, the shirt in the bin and the black sock in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 5. Table: episode 5 (3111 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock | 1 | cloth-pinch | 3 | 4 | 0-456 |
| 2 | move the white sock to the basket | 1 | na | 2 | 4 | 456-663 |
| 3 | release the white sock in the basket | 1 | drop | 3 | 4 | 663-702 |
| 4 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 702-1100 |
| 5 | move the shirt to the bin | 1 | na | 3 | 3 | 1100-1275 |
| 6 | release the shirt in the bin | 1 | drop | 3 | 3 | 1275-1510 |
| 7 | grasp the shirt | 1 | cloth-pinch | 3 | 3 | 1510-1773 |
| 8 | move the shirt to the bin | 1 | na | 3 | 4 | 1773-1924 |
| 9 | release the shirt in the bin | 1 | drop | 3 | 4 | 1924-2062 |
| 10 | grasp the sock | 1 | cloth-pinch | 3 | 3 | 2062-2611 |
| 11 | move the sock to the basket | 1 | na | 3 | 4 | 2611-2840 |
| 12 | release the sock in the basket | 1 | drop | 2 | 4 | 2840-2884 |
| 13 | return to home | 1 | na | 5 | 4 | 2884-3111 |

```yaml
task: Put the white sock in the basket, the shirt in the bin and the black sock in the basket
eval_subtasks:
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T051 Put the white sock in the bin, the red shirt in the basket, the black sock in the bin and the navy shirt in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 54. Table: episode 54 (5952 frames, mean quality 3.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock | 1 | cloth-pinch | 3 | 4 | 0-438 |
| 2 | move the white sock to the bin | 1 | na | 2 | 2 | 438-660 |
| 3 | release the white sock in the bin | 1 | drop | 2 | 2 | 660-735 |
| 4 | grasp the white sock | 1 | cloth-pinch | 2 | 3 | 735-936 |
| 5 | move the white sock to the bin | 1 | na | 2 | 3 | 936-1029 |
| 6 | release the white sock in the bin | 1 | drop | 3 | 3 | 1029-1062 |
| 7 | grasp the white sock | 1 | cloth-pinch | 2 | 3 | 1062-1212 |
| 8 | move the white sock to the bin | 1 | na | 2 | 5 | 1212-1311 |
| 9 | release the white sock in the bin | 1 | drop | 2 | 4 | 1311-1353 |
| 10 | grasp the red shirt | 1 | cloth-pinch | 2 | 2 | 1353-1878 |
| 11 | move the red shirt to the basket | 1 | na | 4 | 3 | 1878-2229 |
| 12 | release the red shirt in the basket | 1 | drop | 2 | 3 | 2229-2310 |
| 13 | grasp the red shirt | 1 | cloth-pinch | 3 | 4 | 2310-2550 |
| 14 | move the red shirt to the basket | 1 | na | 3 | 3 | 2550-2709 |
| 15 | release the red shirt in the basket | 1 | drop | 3 | 3 | 2709-2766 |
| 16 | grasp the red shirt | 1 | cloth-pinch | 2 | 3 | 2766-3012 |
| 17 | move the red shirt to the basket | 1 | na | 3 | 3 | 3012-3228 |
| 18 | release the red shirt in the basket | 1 | drop | 2 | 3 | 3228-3318 |
| 19 | grasp the red shirt | 1 | cloth-pinch | 3 | 3 | 3318-3570 |
| 20 | move the red shirt to the basket | 1 | na | 3 | 4 | 3570-3792 |
| 21 | release the red shirt in the basket | 1 | drop | 2 | 4 | 3792-3840 |
| 22 | grasp the sock | 1 | cloth-pinch | 3 | 1 | 3840-4578 |
| 23 | move the sock to the bin | 1 | na | 3 | 5 | 4578-4866 |
| 24 | release the sock in the bin | 1 | drop | 4 | 4 | 4866-4917 |
| 25 | grasp the shirt | 1 | cloth-pinch | 3 | 4 | 4917-5406 |
| 26 | move the shirt to the basket | 1 | na | 3 | 3 | 5406-5595 |
| 27 | release the shirt in the basket | 1 | drop | 2 | 3 | 5595-5655 |
| 28 | grasp the shirt | 1 | cloth-pinch | 2 | 2 | 5655-5952 |

```yaml
task: Put the white sock in the bin, the red shirt in the basket, the black sock in the bin and the navy shirt in the basket
eval_subtasks:
- grasp the white sock
- move the white sock to the bin
- release the white sock in the bin
- grasp the white sock
- move the white sock to the bin
- release the white sock in the bin
- grasp the white sock
- move the white sock to the bin
- release the white sock in the bin
- grasp the red shirt
- move the red shirt to the basket
- release the red shirt in the basket
- grasp the red shirt
- move the red shirt to the basket
- release the red shirt in the basket
- grasp the red shirt
- move the red shirt to the basket
- release the red shirt in the basket
- grasp the red shirt
- move the red shirt to the basket
- release the red shirt in the basket
- grasp the sock
- move the sock to the bin
- release the sock in the bin
- grasp the shirt
- move the shirt to the basket
- release the shirt in the basket
- grasp the shirt
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch]
```

### T052 Put the white sock, the plain black sock and the green shirt in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 0. Table: episode 0 (4065 frames, mean quality 3.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock | 1 | cloth-pinch | 3 | 5 | 0-390 |
| 2 | move the white sock to the basket | 1 | na | 2 | 5 | 390-645 |
| 3 | release the white sock in the basket | 1 | drop | 2 | 4 | 645-720 |
| 4 | grasp the plain black sock | 1 | cloth-pinch | 2 | 1 | 720-1202 |
| 5 | move the plain black sock to the basket | 1 | na | 2 | 5 | 1202-1440 |
| 6 | release the plain black sock in the basket | 1 | drop | 2 | 4 | 1440-1503 |
| 7 | grasp the green shirt | 1 | cloth-pinch | 2 | 1 | 1503-3084 |
| 8 | move the green shirt to the basket | 1 | na | 2 | 2 | 3084-3345 |
| 9 | release the green shirt in the basket | 1 | drop | 2 | 2 | 3345-3415 |
| 10 | grasp the green shirt | 1 | cloth-pinch | 2 | 2 | 3415-4065 |

```yaml
task: Put the white sock, the plain black sock and the green shirt in the basket
eval_subtasks:
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the plain black sock
- move the plain black sock to the basket
- release the plain black sock in the basket
- grasp the green shirt
- move the green shirt to the basket
- release the green shirt in the basket
- grasp the green shirt
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch]
```

### T053 Put the white socks and the brown shirt in the bin, then grasp the black sock

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 42. Table: episode 42 (4074 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock farther from the robot | 1 | cloth-pinch | 3 | 5 | 0-483 |
| 2 | move the white sock to the bin | 1 | na | 3 | 4 | 483-708 |
| 3 | release the white sock in the bin | 1 | drop | 1 | 4 | 708-798 |
| 4 | grasp the white sock | 1 | cloth-pinch | 2 | 2 | 798-1312 |
| 5 | move the white sock to the bin | 1 | na | 3 | 3 | 1312-1488 |
| 6 | release the white sock in the bin | 1 | drop | 2 | 4 | 1488-1566 |
| 7 | grasp the brown shirt | 1 | cloth-pinch | 2 | 1 | 1566-2355 |
| 8 | move the brown shirt to the bin | 1 | na | 3 | 4 | 2355-2550 |
| 9 | release the brown shirt in the bin | 1 | drop | 2 | 4 | 2550-2640 |
| 10 | grasp the sock | 1 | cloth-pinch | 2 | 1 | 2640-3663 |
| 11 | return to home | 1 | na | 3 | 1 | 3663-4074 |

```yaml
task: Put the white socks and the brown shirt in the bin, then grasp the black sock
eval_subtasks:
- grasp the white sock farther from the robot
- move the white sock to the bin
- release the white sock in the bin
- grasp the white sock
- move the white sock to the bin
- release the white sock in the bin
- grasp the brown shirt
- move the brown shirt to the bin
- release the brown shirt in the bin
- grasp the sock
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na]
```

### T054 Put two black socks, then the four white socks, then the other two black socks in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 40. Table: episode 40 (6375 frames, mean quality 4.3).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black sock with grey stripes nearest the basket | 1 | cloth-pinch | 3 | 3 | 0-495 |
| 2 | move the black sock with grey stripes to the basket | 1 | na | 4 | 5 | 495-702 |
| 3 | release the black sock with grey stripes in the basket | 1 | drop | 1 | 5 | 702-747 |
| 4 | grasp the black sock with grey stripes | 1 | cloth-pinch | 2 | 5 | 747-1209 |
| 5 | move the black sock with grey stripes to the basket | 1 | na | 3 | 5 | 1209-1464 |
| 6 | release the black sock with grey stripes in the basket | 1 | drop | 1 | 5 | 1464-1506 |
| 7 | grasp the plain white sock | 1 | cloth-pinch | 1 | 2 | 1506-2104 |
| 8 | move the plain white sock to the basket | 1 | na | 3 | 5 | 2104-2330 |
| 9 | release the plain white sock in the basket | 1 | drop | 1 | 5 | 2330-2378 |
| 10 | grasp the white sock with the black logo | 1 | cloth-pinch | 2 | 5 | 2378-2814 |
| 11 | move the white sock with the black logo to the basket | 1 | na | 3 | 5 | 2814-3044 |
| 12 | release the white sock with the black logo in the basket | 1 | drop | 1 | 5 | 3044-3084 |
| 13 | grasp the white sock under the black sock | 1 | cloth-pinch | 2 | 3 | 3084-3460 |
| 14 | move the white sock to the basket | 1 | na | 4 | 3 | 3460-3644 |
| 15 | release the white sock in the basket | 1 | drop | 3 | 5 | 3644-3696 |
| 16 | grasp the white sock | 1 | cloth-pinch | 1 | 2 | 3696-4478 |
| 17 | move the white sock to the basket | 1 | na | 5 | 4 | 4478-4674 |
| 18 | release the white sock in the basket | 1 | drop | 3 | 5 | 4674-4710 |
| 19 | grasp the sock farther from the robot | 1 | cloth-pinch | 2 | 5 | 4710-5109 |
| 20 | move the sock to the basket | 1 | na | 4 | 4 | 5109-5294 |
| 21 | release the sock in the basket | 1 | drop | 2 | 4 | 5294-5340 |
| 22 | grasp the sock | 1 | cloth-pinch | 2 | 2 | 5340-5892 |
| 23 | move the sock to the basket | 1 | na | 4 | 5 | 5892-6076 |
| 24 | release the sock in the basket | 1 | drop | 2 | 5 | 6076-6124 |
| 25 | return to home | 1 | na | 5 | 5 | 6124-6375 |

```yaml
task: Put two black socks, then the four white socks, then the other two black socks in the basket
eval_subtasks:
- grasp the black sock with grey stripes nearest the basket
- move the black sock with grey stripes to the basket
- release the black sock with grey stripes in the basket
- grasp the black sock with grey stripes
- move the black sock with grey stripes to the basket
- release the black sock with grey stripes in the basket
- grasp the plain white sock
- move the plain white sock to the basket
- release the plain white sock in the basket
- grasp the white sock with the black logo
- move the white sock with the black logo to the basket
- release the white sock with the black logo in the basket
- grasp the white sock under the black sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the sock farther from the robot
- move the sock to the basket
- release the sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T055 Put two white socks, a black sock, a white sock, the other black sock and the last white sock in the basket, then the white shirt and the grey shirt in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `clothing`, episodes 59. Table: episode 59 (7098 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock nearest the front left corner of the table | 1 | cloth-pinch | 3 | 3 | 0-618 |
| 2 | move the white sock to the basket | 1 | na | 3 | 4 | 618-879 |
| 3 | release the white sock in the basket | 1 | drop | 2 | 4 | 879-963 |
| 4 | grasp the white sock nearest the basket | 1 | cloth-pinch | 3 | 4 | 963-1380 |
| 5 | move the white sock to the basket | 1 | na | 3 | 5 | 1380-1599 |
| 6 | release the white sock in the basket | 1 | drop | 2 | 4 | 1599-1689 |
| 7 | grasp the plain black sock | 1 | cloth-pinch | 3 | 4 | 1689-2185 |
| 8 | move the plain black sock to the basket | 1 | na | 3 | 4 | 2185-2445 |
| 9 | release the plain black sock in the basket | 1 | drop | 2 | 5 | 2445-2521 |
| 10 | grasp the white sock nearest the bin | 1 | cloth-pinch | 3 | 5 | 2521-3000 |
| 11 | move the white sock to the basket | 1 | na | 4 | 4 | 3000-3243 |
| 12 | release the white sock in the basket | 1 | drop | 3 | 4 | 3243-3312 |
| 13 | grasp the black sock | 1 | cloth-pinch | 3 | 3 | 3312-3730 |
| 14 | move the black sock to the basket | 1 | na | 4 | 4 | 3730-3945 |
| 15 | release the black sock in the basket | 1 | drop | 3 | 4 | 3945-4011 |
| 16 | grasp the sock | 1 | cloth-pinch | 3 | 1 | 4011-4835 |
| 17 | move the sock to the basket | 1 | na | 4 | 4 | 4835-5070 |
| 18 | release the sock in the basket | 1 | drop | 4 | 4 | 5070-5139 |
| 19 | grasp the white shirt | 1 | cloth-pinch | 3 | 4 | 5139-5424 |
| 20 | move the white shirt to the bin | 1 | na | 4 | 3 | 5424-5612 |
| 21 | release the white shirt in the bin | 1 | drop | 4 | 4 | 5612-5724 |
| 22 | grasp the shirt | 1 | cloth-pinch | 4 | 4 | 5724-6050 |
| 23 | move the shirt to the bin | 1 | na | 5 | 3 | 6050-6285 |
| 24 | release the shirt in the bin | 1 | drop | 4 | 3 | 6285-6384 |
| 25 | grasp the shirt | 1 | cloth-pinch | 4 | 2 | 6384-6735 |
| 26 | move the shirt to the bin | 1 | na | 4 | 5 | 6735-6841 |
| 27 | release the shirt in the bin | 1 | drop | 4 | 5 | 6841-6893 |
| 28 | return to home | 1 | na | 5 | 5 | 6893-7098 |

```yaml
task: Put two white socks, a black sock, a white sock, the other black sock and the last white sock in the basket, then the white shirt and the grey shirt in the bin
eval_subtasks:
- grasp the white sock nearest the front left corner of the table
- move the white sock to the basket
- release the white sock in the basket
- grasp the white sock nearest the basket
- move the white sock to the basket
- release the white sock in the basket
- grasp the plain black sock
- move the plain black sock to the basket
- release the plain black sock in the basket
- grasp the white sock nearest the bin
- move the white sock to the basket
- release the white sock in the basket
- grasp the black sock
- move the black sock to the basket
- release the black sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- grasp the white shirt
- move the white shirt to the bin
- release the white shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- grasp the shirt
- move the shirt to the bin
- release the shirt in the bin
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T056 Put the cup and the pill bottle in the basket and the spray bottle in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 76. Table: episode 76 (3144 frames, mean quality 4.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cup | 3 | rim-pinch | 3 | 4 | 0-516 |
| 2 | move the cup to the basket | 1 | na | 4 | 3 | 516-840 |
| 3 | release the cup in the basket | 1 | drop | 2 | 4 | 840-930 |
| 4 | grasp the pill bottle | 3 | side-pinch | 3 | 5 | 930-1488 |
| 5 | move the pill bottle to the basket | 1 | na | 4 | 5 | 1488-1752 |
| 6 | release the pill bottle in the basket | 1 | drop | 2 | 4 | 1752-1806 |
| 7 | grasp the spray bottle | 3 | side-pinch | 3 | 4 | 1806-2268 |
| 8 | move the spray bottle to the bin | 1 | na | 3 | 4 | 2268-2655 |
| 9 | release the spray bottle in the bin | 1 | drop | 3 | 4 | 2655-2742 |
| 10 | return to home | 1 | na | 3 | 4 | 2742-3144 |

```yaml
task: Put the cup and the pill bottle in the basket and the spray bottle in the bin
eval_subtasks:
- grasp the cup
- move the cup to the basket
- release the cup in the basket
- grasp the pill bottle
- move the pill bottle to the basket
- release the pill bottle in the basket
- grasp the spray bottle
- move the spray bottle to the bin
- release the spray bottle in the bin
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 1, 1]
eval_subtask_contacts: [rim-pinch, na, drop, side-pinch, na, drop, side-pinch, na, drop, na]
```

### T057 Put the cup and the spray bottle in the bin and the pill bottle in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 75. Table: episode 75 (3180 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cup | 3 | rim-pinch | 3 | 4 | 0-471 |
| 2 | move the cup to the bin | 1 | na | 3 | 5 | 471-780 |
| 3 | release the cup in the bin | 1 | drop | 2 | 4 | 780-828 |
| 4 | grasp the spray bottle | 3 | top-pinch | 3 | 1 | 828-1866 |
| 5 | move the spray bottle to the bin | 1 | na | 3 | 4 | 1866-2130 |
| 6 | release the spray bottle in the bin | 1 | drop | 2 | 4 | 2130-2205 |
| 7 | grasp the pill bottle | 3 | side-pinch | 3 | 4 | 2205-2534 |
| 8 | move the pill bottle to the basket | 1 | na | 4 | 5 | 2534-2760 |
| 9 | release the pill bottle in the basket | 1 | drop | 3 | 4 | 2760-2865 |
| 10 | return to home | 1 | na | 4 | 4 | 2865-3180 |

```yaml
task: Put the cup and the spray bottle in the bin and the pill bottle in the basket
eval_subtasks:
- grasp the cup
- move the cup to the bin
- release the cup in the bin
- grasp the spray bottle
- move the spray bottle to the bin
- release the spray bottle in the bin
- grasp the pill bottle
- move the pill bottle to the basket
- release the pill bottle in the basket
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 1, 1]
eval_subtask_contacts: [rim-pinch, na, drop, top-pinch, na, drop, side-pinch, na, drop, na]
```

### T058 Put the cup and the tape roll in the basket and the spray bottle and the pill bottle in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 79. Table: episode 79 (4239 frames, mean quality 2.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cup | 3 | rim-pinch | 3 | 4 | 0-510 |
| 2 | move the cup to the basket | 1 | na | 3 | 4 | 510-735 |
| 3 | release the cup in the basket | 1 | drop | 2 | 3 | 735-846 |
| 4 | grasp the tape roll | 3 | rim-pinch | 2 | 3 | 846-1320 |
| 5 | move the tape roll to the basket | 1 | na | 2 | 4 | 1320-1510 |
| 6 | release the tape roll in the basket | 1 | drop | 2 | 4 | 1510-1626 |
| 7 | grasp the spray bottle | 3 | side-pinch | 2 | 3 | 1626-2286 |
| 8 | move the spray bottle to the bin | 1 | na | 4 | 4 | 2286-2545 |
| 9 | release the spray bottle in the bin | 1 | drop | 2 | 4 | 2545-2608 |
| 10 | grasp the pill bottle | 3 | side-pinch | 3 | 3 | 2608-3168 |
| 11 | move the pill bottle to the bin | 1 | na | 4 | 1 | 3168-3475 |
| 12 | release the pill bottle in the bin | 1 | drop | 3 | 1 | 3475-3531 |
| 13 | grasp the pill bottle | 3 | side-pinch | 3 | 1 | 3531-3858 |
| 14 | return to home | 1 | na | 4 | 1 | 3858-4239 |

```yaml
task: Put the cup and the tape roll in the basket and the spray bottle and the pill bottle in the bin
eval_subtasks:
- grasp the cup
- move the cup to the basket
- release the cup in the basket
- grasp the tape roll
- move the tape roll to the basket
- release the tape roll in the basket
- grasp the spray bottle
- move the spray bottle to the bin
- release the spray bottle in the bin
- grasp the pill bottle
- move the pill bottle to the bin
- release the pill bottle in the bin
- grasp the pill bottle
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1]
eval_subtask_contacts: [rim-pinch, na, drop, rim-pinch, na, drop, side-pinch, na, drop, side-pinch, na, drop, side-pinch, na]
```

### T059 Put the cup in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 15. Table: episode 15 (1632 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cup | 3 | rim-pinch | 2 | 2 | 0-969 |
| 2 | move the cup to the basket | 1 | na | 3 | 4 | 969-1254 |
| 3 | release the cup in the basket | 1 | drop | 3 | 5 | 1254-1332 |
| 4 | return to home | 1 | na | 5 | 5 | 1332-1632 |

```yaml
task: Put the cup in the basket
eval_subtasks:
- grasp the cup
- move the cup to the basket
- release the cup in the basket
- return to home
eval_subtask_precisions: [3, 1, 1, 1]
eval_subtask_contacts: [rim-pinch, na, drop, na]
```

### T060 Put the cup in the basket and the pill bottle and the tape roll in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 80. Table: episode 80 (2517 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cup | 3 | rim-pinch | 3 | 3 | 0-474 |
| 2 | move the cup to the basket | 1 | na | 3 | 4 | 474-717 |
| 3 | release the cup in the basket | 1 | drop | 2 | 4 | 717-810 |
| 4 | grasp the pill bottle | 3 | side-pinch | 2 | 5 | 810-1206 |
| 5 | move the pill bottle to the bin | 1 | na | 3 | 4 | 1206-1440 |
| 6 | release the pill bottle in the bin | 1 | drop | 3 | 4 | 1440-1551 |
| 7 | grasp the tape roll | 3 | rim-pinch | 3 | 4 | 1551-2071 |
| 8 | move the tape roll to the bin | 1 | na | 3 | 4 | 2071-2412 |
| 9 | release the tape roll in the bin | 1 | drop | 3 | 4 | 2412-2517 |

```yaml
task: Put the cup in the basket and the pill bottle and the tape roll in the bin
eval_subtasks:
- grasp the cup
- move the cup to the basket
- release the cup in the basket
- grasp the pill bottle
- move the pill bottle to the bin
- release the pill bottle in the bin
- grasp the tape roll
- move the tape roll to the bin
- release the tape roll in the bin
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 1]
eval_subtask_contacts: [rim-pinch, na, drop, side-pinch, na, drop, rim-pinch, na, drop]
```

### T061 Put the cup, the pill bottle and the spray bottle in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 71. Table: episode 71 (2871 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cup | 3 | rim-pinch | 3 | 5 | 0-465 |
| 2 | move the cup to the basket | 1 | na | 4 | 3 | 465-723 |
| 3 | release the cup in the basket | 1 | drop | 2 | 4 | 723-834 |
| 4 | grasp the pill bottle | 3 | top-pinch | 3 | 3 | 834-1446 |
| 5 | move the pill bottle to the basket | 1 | na | 5 | 4 | 1446-1710 |
| 6 | release the pill bottle in the basket | 1 | drop | 3 | 4 | 1710-1788 |
| 7 | grasp the spray bottle | 3 | side-pinch | 3 | 4 | 1788-2226 |
| 8 | move the spray bottle to the basket | 1 | na | 4 | 5 | 2226-2415 |
| 9 | release the spray bottle in the basket | 2 | set-down | 1 | 4 | 2415-2622 |
| 10 | return to home | 1 | na | 4 | 4 | 2622-2871 |

```yaml
task: Put the cup, the pill bottle and the spray bottle in the basket
eval_subtasks:
- grasp the cup
- move the cup to the basket
- release the cup in the basket
- grasp the pill bottle
- move the pill bottle to the basket
- release the pill bottle in the basket
- grasp the spray bottle
- move the spray bottle to the basket
- release the spray bottle in the basket
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 2, 1]
eval_subtask_contacts: [rim-pinch, na, drop, top-pinch, na, drop, side-pinch, na, set-down, na]
```

### T062 Put the cup, the spray bottle and the pill bottle in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 72. Table: episode 72 (2880 frames, mean quality 3.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cup | 3 | rim-pinch | 2 | 4 | 0-442 |
| 2 | move the cup to the basket | 1 | na | 3 | 3 | 442-755 |
| 3 | release the cup in the basket | 1 | drop | 2 | 4 | 755-855 |
| 4 | grasp the spray bottle | 3 | side-pinch | 2 | 3 | 855-1264 |
| 5 | move the spray bottle to the basket | 1 | na | 4 | 3 | 1264-1432 |
| 6 | release the spray bottle in the basket | 1 | drop | 2 | 4 | 1432-1534 |
| 7 | grasp the pill bottle | 3 | top-pinch | 3 | 3 | 1534-2116 |
| 8 | move the pill bottle to the basket | 1 | na | 4 | 3 | 2116-2420 |
| 9 | release the pill bottle in the basket | 1 | drop | 2 | 4 | 2420-2486 |
| 10 | return to home | 1 | na | 3 | 4 | 2486-2880 |

```yaml
task: Put the cup, the spray bottle and the pill bottle in the basket
eval_subtasks:
- grasp the cup
- move the cup to the basket
- release the cup in the basket
- grasp the spray bottle
- move the spray bottle to the basket
- release the spray bottle in the basket
- grasp the pill bottle
- move the pill bottle to the basket
- release the pill bottle in the basket
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 1, 1]
eval_subtask_contacts: [rim-pinch, na, drop, side-pinch, na, drop, top-pinch, na, drop, na]
```

### T063 Put the pill bottle, the cup and the spray bottle in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 73. Table: episode 73 (2385 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the pill bottle | 3 | side-pinch | 4 | 5 | 0-363 |
| 2 | move the pill bottle to the basket | 1 | na | 3 | 4 | 363-564 |
| 3 | release the pill bottle in the basket | 1 | drop | 2 | 4 | 564-648 |
| 4 | grasp the cup | 3 | rim-pinch | 2 | 3 | 648-1149 |
| 5 | move the cup to the basket | 1 | na | 3 | 5 | 1149-1360 |
| 6 | release the cup in the basket | 1 | drop | 2 | 3 | 1360-1420 |
| 7 | grasp the spray bottle | 3 | side-pinch | 3 | 3 | 1420-1767 |
| 8 | move the spray bottle to the basket | 1 | na | 3 | 5 | 1767-1920 |
| 9 | release the spray bottle in the basket | 2 | set-down | 3 | 4 | 1920-2034 |
| 10 | return to home | 1 | na | 4 | 4 | 2034-2385 |

```yaml
task: Put the pill bottle, the cup and the spray bottle in the basket
eval_subtasks:
- grasp the pill bottle
- move the pill bottle to the basket
- release the pill bottle in the basket
- grasp the cup
- move the cup to the basket
- release the cup in the basket
- grasp the spray bottle
- move the spray bottle to the basket
- release the spray bottle in the basket
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 2, 1]
eval_subtask_contacts: [side-pinch, na, drop, rim-pinch, na, drop, side-pinch, na, set-down, na]
```

### T064 Put the pill bottle, the spray bottle and the cup in the basket

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 70, 78. Table: episode 70 (2613 frames, mean quality 4.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the pill bottle | 3 | side-pinch | 3 | 5 | 0-375 |
| 2 | move the pill bottle to the basket | 1 | na | 2 | 4 | 375-558 |
| 3 | release the pill bottle in the basket | 1 | drop | 2 | 4 | 558-645 |
| 4 | grasp the spray bottle | 3 | side-pinch | 3 | 3 | 645-1349 |
| 5 | move the spray bottle to the basket | 1 | na | 4 | 5 | 1349-1588 |
| 6 | release the spray bottle in the basket | 1 | drop | 3 | 4 | 1588-1665 |
| 7 | grasp the cup | 3 | rim-pinch | 3 | 4 | 1665-2070 |
| 8 | move the cup to the basket | 1 | na | 4 | 5 | 2070-2296 |
| 9 | release the cup in the basket | 1 | drop | 2 | 5 | 2296-2366 |
| 10 | return to home | 1 | na | 4 | 5 | 2366-2613 |

```yaml
task: Put the pill bottle, the spray bottle and the cup in the basket
eval_subtasks:
- grasp the pill bottle
- move the pill bottle to the basket
- release the pill bottle in the basket
- grasp the spray bottle
- move the spray bottle to the basket
- release the spray bottle in the basket
- grasp the cup
- move the cup to the basket
- release the cup in the basket
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 1, 1]
eval_subtask_contacts: [side-pinch, na, drop, side-pinch, na, drop, rim-pinch, na, drop, na]
```

<details><summary>Other episodes (1)</summary>


Episode 78 (2895 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the pill bottle | 3 | side-pinch | 3 | 3 | 0-498 |
| 2 | move the pill bottle to the basket | 1 | na | 2 | 3 | 498-798 |
| 3 | release the pill bottle in the basket | 1 | drop | 2 | 4 | 798-867 |
| 4 | grasp the spray bottle | 3 | side-pinch | 2 | 4 | 867-1193 |
| 5 | move the spray bottle to the basket | 1 | na | 3 | 4 | 1193-1398 |
| 6 | release the spray bottle in the basket | 1 | drop | 2 | 4 | 1398-1467 |
| 7 | grasp the cup | 3 | rim-pinch | 3 | 3 | 1467-2100 |
| 8 | move the cup to the basket | 1 | na | 4 | 4 | 2100-2385 |
| 9 | release the cup in the basket | 1 | drop | 2 | 4 | 2385-2457 |
| 10 | return to home | 1 | na | 3 | 4 | 2457-2895 |

</details>

### T065 Put the spray bottle and the cup in the basket and the pill bottle in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 74. Table: episode 74 (2622 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the spray bottle | 3 | side-pinch | 2 | 4 | 0-486 |
| 2 | move the spray bottle to the basket | 1 | na | 2 | 4 | 486-606 |
| 3 | release the spray bottle in the basket | 1 | drop | 3 | 4 | 606-756 |
| 4 | grasp the cup | 3 | rim-pinch | 3 | 4 | 756-1250 |
| 5 | move the cup to the basket | 1 | na | 4 | 4 | 1250-1494 |
| 6 | release the cup in the basket | 1 | drop | 2 | 3 | 1494-1590 |
| 7 | grasp the pill bottle | 3 | side-pinch | 3 | 3 | 1590-2012 |
| 8 | move the pill bottle to the bin | 1 | na | 4 | 5 | 2012-2100 |
| 9 | release the pill bottle in the bin | 1 | drop | 4 | 4 | 2100-2210 |
| 10 | return to home | 1 | na | 4 | 4 | 2210-2622 |

```yaml
task: Put the spray bottle and the cup in the basket and the pill bottle in the bin
eval_subtasks:
- grasp the spray bottle
- move the spray bottle to the basket
- release the spray bottle in the basket
- grasp the cup
- move the cup to the basket
- release the cup in the basket
- grasp the pill bottle
- move the pill bottle to the bin
- release the pill bottle in the bin
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 1, 1]
eval_subtask_contacts: [side-pinch, na, drop, rim-pinch, na, drop, side-pinch, na, drop, na]
```

### T066 Put the spray bottle, the cup and the tape roll in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 81. Table: episode 81 (2685 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the spray bottle | 3 | side-pinch | 3 | 4 | 0-540 |
| 2 | move the spray bottle to the bin | 1 | na | 4 | 4 | 540-600 |
| 3 | release the spray bottle in the bin | 1 | drop | 3 | 5 | 600-672 |
| 4 | grasp the cup | 3 | side-pinch | 3 | 2 | 672-1270 |
| 5 | move the cup to the bin | 1 | na | 4 | 3 | 1270-1452 |
| 6 | release the cup in the bin | 1 | drop | 3 | 4 | 1452-1521 |
| 7 | grasp the tape roll | 3 | rim-pinch | 3 | 4 | 1521-1980 |
| 8 | move the tape roll to the bin | 1 | na | 4 | 4 | 1980-2268 |
| 9 | release the tape roll in the bin | 1 | drop | 3 | 5 | 2268-2328 |
| 10 | return to home | 1 | na | 4 | 5 | 2328-2685 |

```yaml
task: Put the spray bottle, the cup and the tape roll in the bin
eval_subtasks:
- grasp the spray bottle
- move the spray bottle to the bin
- release the spray bottle in the bin
- grasp the cup
- move the cup to the bin
- release the cup in the bin
- grasp the tape roll
- move the tape roll to the bin
- release the tape roll in the bin
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 1, 1]
eval_subtask_contacts: [side-pinch, na, drop, side-pinch, na, drop, rim-pinch, na, drop, na]
```

### T067 Put the tape roll in the bin, the cup in the basket, and the pill bottle and the spray bottle in the bin

Source `internal_main` (`outputs/rebot_all-annotated-v14`), group `rigid`, episodes 77. Table: episode 77 (3192 frames, mean quality 4.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tape roll | 3 | rim-pinch | 3 | 3 | 0-472 |
| 2 | move the tape roll to the bin | 1 | na | 2 | 4 | 472-620 |
| 3 | release the tape roll in the bin | 1 | drop | 3 | 5 | 620-738 |
| 4 | grasp the cup | 3 | rim-pinch | 2 | 5 | 738-1092 |
| 5 | move the cup to the basket | 1 | na | 4 | 4 | 1092-1310 |
| 6 | release the cup in the basket | 1 | drop | 3 | 4 | 1310-1430 |
| 7 | grasp the pill bottle | 3 | side-pinch | 2 | 4 | 1430-1814 |
| 8 | move the pill bottle to the bin | 1 | na | 3 | 4 | 1814-2040 |
| 9 | release the pill bottle in the bin | 1 | drop | 3 | 5 | 2040-2150 |
| 10 | grasp the spray bottle | 3 | side-pinch | 3 | 5 | 2150-2546 |
| 11 | move the spray bottle to the bin | 1 | na | 3 | 5 | 2546-2780 |
| 12 | release the spray bottle in the bin | 1 | drop | 2 | 5 | 2780-2868 |
| 13 | return to home | 1 | na | 4 | 5 | 2868-3192 |

```yaml
task: Put the tape roll in the bin, the cup in the basket, and the pill bottle and the spray bottle in the bin
eval_subtasks:
- grasp the tape roll
- move the tape roll to the bin
- release the tape roll in the bin
- grasp the cup
- move the cup to the basket
- release the cup in the basket
- grasp the pill bottle
- move the pill bottle to the bin
- release the pill bottle in the bin
- grasp the spray bottle
- move the spray bottle to the bin
- release the spray bottle in the bin
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1]
eval_subtask_contacts: [rim-pinch, na, drop, rim-pinch, na, drop, side-pinch, na, drop, side-pinch, na, drop, na]
```

### T068 Put socks in basket

Source `internal_additions` (`outputs/rebot_main_additions-annotated-v3`), group `clothing`, episodes 5. Table: episode 5 (5337 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the white sock lying on a black sock | 1 | cloth-pinch | 4 | 4 | 0-448 |
| 2 | move the white sock to the basket | 1 | na | 3 | 3 | 448-705 |
| 3 | release the white sock in the basket | 1 | drop | 2 | 3 | 705-819 |
| 4 | grasp the white sock | 1 | cloth-pinch | 3 | 5 | 819-1209 |
| 5 | move the white sock to the basket | 1 | na | 4 | 5 | 1209-1425 |
| 6 | release the white sock in the basket | 1 | drop | 2 | 4 | 1425-1471 |
| 7 | grasp the black sock lying apart from the others | 1 | cloth-pinch | 4 | 5 | 1471-1891 |
| 8 | move the black sock to the basket | 1 | na | 4 | 5 | 1891-2104 |
| 9 | release the black sock in the basket | 1 | drop | 2 | 4 | 2104-2168 |
| 10 | grasp the striped sock farther from the basket | 1 | cloth-pinch | 3 | 2 | 2168-2650 |
| 11 | move the striped sock to the basket | 1 | na | 4 | 5 | 2650-2858 |
| 12 | release the striped sock in the basket | 1 | drop | 2 | 5 | 2858-2940 |
| 13 | grasp the striped sock | 1 | cloth-pinch | 3 | 2 | 2940-3352 |
| 14 | move the striped sock to the basket | 1 | na | 4 | 2 | 3352-3510 |
| 15 | release the striped sock in the basket | 1 | drop | 3 | 4 | 3510-3560 |
| 16 | grasp the striped sock | 1 | cloth-pinch | 2 | 2 | 3560-3921 |
| 17 | move the striped sock to the basket | 1 | na | 4 | 4 | 3921-4067 |
| 18 | release the striped sock in the basket | 1 | drop | 3 | 4 | 4067-4167 |
| 19 | grasp the sock | 1 | cloth-pinch | 3 | 2 | 4167-4668 |
| 20 | move the sock to the basket | 1 | na | 5 | 5 | 4668-4903 |
| 21 | release the sock in the basket | 1 | drop | 3 | 4 | 4903-4984 |
| 22 | return to home | 1 | na | 5 | 4 | 4984-5337 |

```yaml
task: Put socks in basket
eval_subtasks:
- grasp the white sock lying on a black sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the white sock
- move the white sock to the basket
- release the white sock in the basket
- grasp the black sock lying apart from the others
- move the black sock to the basket
- release the black sock in the basket
- grasp the striped sock farther from the basket
- move the striped sock to the basket
- release the striped sock in the basket
- grasp the striped sock
- move the striped sock to the basket
- release the striped sock in the basket
- grasp the striped sock
- move the striped sock to the basket
- release the striped sock in the basket
- grasp the sock
- move the sock to the basket
- release the sock in the basket
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T069 Pick up the green eraser and place it in the brown basket.

Source `internal_additions` (`outputs/rebot_main_additions-annotated-v3`), group `rigid`, episodes 6, 7, 8, 9, 10, 11. Table: episode 9 (909 frames, mean quality 4.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green eraser | 3 | top-pinch | 3 | 5 | 0-397 |
| 2 | move the green eraser to the brown basket | 1 | na | 5 | 5 | 397-442 |
| 3 | release the green eraser in the brown basket | 2 | drop | 2 | 5 | 442-531 |
| 4 | return to home | 1 | na | 4 | 4 | 531-909 |

```yaml
task: Pick up the green eraser and place it in the brown basket.
eval_subtasks:
- grasp the green eraser
- move the green eraser to the brown basket
- release the green eraser in the brown basket
- return to home
eval_subtask_precisions: [3, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (5)</summary>


Episode 6 (885 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green eraser | 3 | top-pinch | 3 | 4 | 0-376 |
| 2 | move the green eraser to the brown basket | 1 | na | 5 | 4 | 376-441 |
| 3 | release the green eraser in the brown basket | 2 | drop | 3 | 3 | 441-722 |
| 4 | return to home | 1 | na | 5 | 4 | 722-885 |


Episode 10 (996 frames, mean quality 3.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green eraser | 3 | top-pinch | 2 | 1 | 0-539 |
| 2 | move the green eraser to the brown basket | 1 | na | 5 | 3 | 539-600 |
| 3 | release the green eraser in the brown basket | 2 | drop | 2 | 5 | 600-682 |
| 4 | return to home | 1 | na | 4 | 4 | 682-996 |


Episode 8 (819 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green eraser | 3 | top-pinch | 4 | 3 | 0-374 |
| 2 | move the green eraser to the brown basket | 1 | na | 5 | 3 | 374-441 |
| 3 | release the green eraser in the brown basket | 2 | drop | 2 | 3 | 441-632 |
| 4 | return to home | 1 | na | 5 | 3 | 632-819 |


Episode 11 (1047 frames, mean quality 2.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green eraser | 3 | top-pinch | 3 | 1 | 0-531 |
| 2 | move the green eraser to the brown basket | 1 | na | 5 | 3 | 531-593 |
| 3 | release the green eraser in the brown basket | 2 | drop | 4 | 5 | 593-647 |
| 4 | return to home | 1 | na | 1 | 1 | 647-1047 |


Episode 7 (1026 frames, mean quality 2.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green eraser | 3 | top-pinch | 3 | 1 | 0-379 |
| 2 | move the green eraser to the brown basket | 1 | na | 5 | 3 | 379-510 |
| 3 | release the green eraser in the brown basket | 2 | set-down | 1 | 1 | 510-831 |
| 4 | return to home | 1 | na | 5 | 4 | 831-1026 |

</details>

### T070 Pick up the green eraser and place it in the gray basket.

Source `internal_additions` (`outputs/rebot_main_additions-annotated-v3`), group `rigid`, episodes 12, 13, 14. Table: episode 12 (444 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green eraser | 3 | top-pinch | 4 | 5 | 0-234 |
| 2 | move the green eraser to the gray basket | 1 | na | 5 | 5 | 234-266 |
| 3 | release the green eraser in the gray basket | 2 | drop | 4 | 5 | 266-305 |
| 4 | return to home | 1 | na | 5 | 5 | 305-444 |

```yaml
task: Pick up the green eraser and place it in the gray basket.
eval_subtasks:
- grasp the green eraser
- move the green eraser to the gray basket
- release the green eraser in the gray basket
- return to home
eval_subtask_precisions: [3, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (2)</summary>


Episode 14 (576 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green eraser | 3 | top-pinch | 4 | 5 | 0-261 |
| 2 | move the green eraser to the gray basket | 1 | na | 5 | 5 | 261-304 |
| 3 | release the green eraser in the gray basket | 2 | drop | 4 | 4 | 304-361 |
| 4 | return to home | 1 | na | 5 | 4 | 361-576 |


Episode 13 (564 frames, mean quality 3.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green eraser | 3 | top-pinch | 5 | 3 | 0-239 |
| 2 | move the green eraser to the gray basket | 1 | na | 5 | 3 | 239-280 |
| 3 | release the green eraser in the gray basket | 2 | drop | 4 | 3 | 280-329 |
| 4 | return to home | 1 | na | 5 | 4 | 329-564 |

</details>

### T071 Put a sugar cube in the cup

Source `internal_additions` (`outputs/rebot_main_additions-annotated-v3`), group `rigid`, episodes 15, 16, 17, 18, 19, 20, 21. Table: episode 15 (333 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp a sugar cube | 4 | top-pinch | 4 | 5 | 0-171 |
| 2 | move the sugar cube to the cup | 1 | na | 5 | 5 | 171-208 |
| 3 | release the sugar cube in the cup | 2 | drop | 3 | 5 | 208-259 |
| 4 | return to home | 1 | na | 5 | 5 | 259-333 |

```yaml
task: Put a sugar cube in the cup
eval_subtasks:
- grasp a sugar cube
- move the sugar cube to the cup
- release the sugar cube in the cup
- return to home
eval_subtask_precisions: [4, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (6)</summary>


Episode 16 (444 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp a sugar cube | 4 | top-pinch | 3 | 5 | 0-241 |
| 2 | move the sugar cube to the cup | 1 | na | 5 | 5 | 241-303 |
| 3 | release the sugar cube in the cup | 2 | drop | 3 | 5 | 303-360 |
| 4 | return to home | 1 | na | 5 | 5 | 360-444 |


Episode 20 (363 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp a sugar cube | 4 | top-pinch | 3 | 5 | 0-187 |
| 2 | move the sugar cube to the cup | 1 | na | 5 | 5 | 187-231 |
| 3 | release the sugar cube in the cup | 2 | drop | 3 | 5 | 231-301 |
| 4 | return to home | 1 | na | 5 | 5 | 301-363 |


Episode 19 (168 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | release the sugar cube in the cup | 2 | drop | 1 | 4 | 0-81 |
| 2 | return to home | 1 | na | 5 | 5 | 81-168 |


Episode 17 (564 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp a sugar cube | 4 | top-pinch | 3 | 4 | 0-312 |
| 2 | move the sugar cube to the cup | 1 | na | 5 | 4 | 312-393 |
| 3 | release the sugar cube in the cup | 2 | drop | 2 | 4 | 393-486 |
| 4 | return to home | 1 | na | 5 | 5 | 486-564 |


Episode 21 (420 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp a sugar cube | 4 | top-pinch | 3 | 4 | 0-212 |
| 2 | move the sugar cube to the cup | 1 | na | 5 | 4 | 212-262 |
| 3 | release the sugar cube in the cup | 2 | drop | 2 | 4 | 262-337 |
| 4 | return to home | 1 | na | 5 | 5 | 337-420 |


Episode 18 (540 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp a sugar cube | 4 | top-pinch | 4 | 4 | 0-286 |
| 2 | move the sugar cube to the cup | 1 | na | 5 | 4 | 286-342 |
| 3 | release the sugar cube in the cup | 2 | drop | 4 | 4 | 342-452 |
| 4 | return to home | 1 | na | 5 | 4 | 452-540 |

</details>

### T072 Put the pill bottle and the deck of cards in the basket

Source `internal_additions` (`outputs/rebot_main_additions-annotated-v3`), group `rigid`, episodes 4. Table: episode 4 (3264 frames, mean quality 3.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the pill bottle | 3 | side-pinch | 3 | 4 | 0-410 |
| 2 | move the pill bottle to the basket | 1 | na | 4 | 4 | 410-706 |
| 3 | release the pill bottle in the basket | 1 | drop | 2 | 4 | 706-763 |
| 4 | grasp the deck of cards | 3 | top-pinch | 1 | 1 | 763-2465 |
| 5 | move the deck of cards to the basket | 1 | na | 2 | 3 | 2465-2766 |
| 6 | release the deck of cards in the basket | 1 | drop | 1 | 4 | 2766-2821 |
| 7 | return to home | 1 | na | 4 | 4 | 2821-3264 |

```yaml
task: Put the pill bottle and the deck of cards in the basket
eval_subtasks:
- grasp the pill bottle
- move the pill bottle to the basket
- release the pill bottle in the basket
- grasp the deck of cards
- move the deck of cards to the basket
- release the deck of cards in the basket
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 1]
eval_subtask_contacts: [side-pinch, na, drop, top-pinch, na, drop, na]
```

### T073 Put the sock in the bin and the cup in the basket

Source `internal_additions` (`outputs/rebot_main_additions-annotated-v3`), group `rigid`, episodes 2. Table: episode 2 (1596 frames, mean quality 3.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the sock | 1 | cloth-pinch | 4 | 3 | 0-428 |
| 2 | move the sock to the bin | 1 | na | 3 | 2 | 428-695 |
| 3 | release the sock in the bin | 1 | drop | 2 | 4 | 695-749 |
| 4 | grasp the cup | 3 | rim-pinch | 3 | 2 | 749-1395 |
| 5 | move the cup to the basket | 1 | na | 3 | 4 | 1395-1543 |
| 6 | release the cup in the basket | 1 | drop | 2 | 4 | 1543-1596 |

```yaml
task: Put the sock in the bin and the cup in the basket
eval_subtasks:
- grasp the sock
- move the sock to the bin
- release the sock in the bin
- grasp the cup
- move the cup to the basket
- release the cup in the basket
eval_subtask_precisions: [1, 1, 1, 3, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, rim-pinch, na, drop]
```

### T074 Put the spray bottle and the pill bottle in the basket

Source `internal_additions` (`outputs/rebot_main_additions-annotated-v3`), group `rigid`, episodes 1. Table: episode 1 (1707 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the spray bottle | 3 | side-pinch | 4 | 3 | 0-427 |
| 2 | move the spray bottle to the basket | 1 | na | 3 | 4 | 427-634 |
| 3 | release the spray bottle in the basket | 1 | drop | 2 | 4 | 634-683 |
| 4 | grasp the pill bottle | 3 | top-pinch | 3 | 3 | 683-1082 |
| 5 | move the pill bottle to the basket | 1 | na | 4 | 3 | 1082-1435 |
| 6 | release the pill bottle in the basket | 1 | drop | 3 | 4 | 1435-1490 |
| 7 | return to home | 1 | na | 5 | 4 | 1490-1707 |

```yaml
task: Put the spray bottle and the pill bottle in the basket
eval_subtasks:
- grasp the spray bottle
- move the spray bottle to the basket
- release the spray bottle in the basket
- grasp the pill bottle
- move the pill bottle to the basket
- release the pill bottle in the basket
- return to home
eval_subtask_precisions: [3, 1, 1, 3, 1, 1, 1]
eval_subtask_contacts: [side-pinch, na, drop, top-pinch, na, drop, na]
```

### T075 Put the tape roll in the basket

Source `internal_additions` (`outputs/rebot_main_additions-annotated-v3`), group `rigid`, episodes 0. Table: episode 0 (1227 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tape roll | 3 | rim-pinch | 2 | 1 | 0-928 |
| 2 | move the tape roll to the basket | 1 | na | 3 | 3 | 928-1170 |
| 3 | release the tape roll in the basket | 1 | drop | 3 | 5 | 1170-1227 |

```yaml
task: Put the tape roll in the basket
eval_subtasks:
- grasp the tape roll
- move the tape roll to the basket
- release the tape roll in the basket
eval_subtask_precisions: [3, 1, 1]
eval_subtask_contacts: [rim-pinch, na, drop]
```

### T076 Put the tape roll in the bin

Source `internal_additions` (`outputs/rebot_main_additions-annotated-v3`), group `rigid`, episodes 3. Table: episode 3 (2187 frames, mean quality 2.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tape roll | 3 | rim-pinch | 2 | 2 | 0-1398 |
| 2 | move the tape roll to the bin | 1 | na | 3 | 2 | 1398-1790 |
| 3 | release the tape roll in the bin | 1 | drop | 4 | 2 | 1790-1844 |
| 4 | return to home | 1 | na | 4 | 4 | 1844-2187 |

```yaml
task: Put the tape roll in the bin
eval_subtasks:
- grasp the tape roll
- move the tape roll to the bin
- release the tape roll in the bin
- return to home
eval_subtask_precisions: [3, 1, 1, 1]
eval_subtask_contacts: [rim-pinch, na, drop, na]
```

### T077 Put a bit in the open lid of the box

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 9. Table: episode 9 (5520 frames, mean quality 2.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bit farthest from the robot base | 5 | top-pinch | 2 | 1 | 0-1203 |
| 2 | grasp the bit nearest the box | 5 | top-pinch | 1 | 1 | 1203-3375 |
| 3 | grasp the bit right of the bit nearest the robot base | 5 | top-pinch | 1 | 2 | 3375-4830 |
| 4 | move the bit to the open lid of the box | 1 | na | 1 | 4 | 4830-5220 |
| 5 | release the bit in the open lid of the box | 2 | drop | 1 | 4 | 5220-5520 |

```yaml
task: Put a bit in the open lid of the box
eval_subtasks:
- grasp the bit farthest from the robot base
- grasp the bit nearest the box
- grasp the bit right of the bit nearest the robot base
- move the bit to the open lid of the box
- release the bit in the open lid of the box
eval_subtask_precisions: [5, 5, 5, 1, 2]
eval_subtask_contacts: [top-pinch, top-pinch, top-pinch, na, drop]
```

### T078 Put five bits in the empty compartment of the box and reach for the last bit

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 2. Table: episode 2 (5502 frames, mean quality 4.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bit left of the box nearest the robot base | 5 | top-pinch | 2 | 5 | 0-582 |
| 2 | move the bit to the empty compartment | 3 | na | 2 | 5 | 582-720 |
| 3 | release the bit in the empty compartment | 4 | drop | 1 | 5 | 720-825 |
| 4 | grasp the bit next to the box | 5 | top-pinch | 1 | 1 | 825-1572 |
| 5 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 1572-1666 |
| 6 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 1666-1711 |
| 7 | grasp the bit nearest the box | 5 | top-pinch | 2 | 5 | 1711-2016 |
| 8 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 2016-2184 |
| 9 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 2184-2316 |
| 10 | grasp the bit farthest from the box | 5 | top-pinch | 1 | 2 | 2316-3756 |
| 11 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 3756-4089 |
| 12 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 4089-4264 |
| 13 | grasp the bit nearest the box | 5 | top-pinch | 1 | 5 | 4264-4736 |
| 14 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 4736-4929 |
| 15 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 4929-5048 |
| 16 | grasp the bit | 5 | top-pinch | 2 | 3 | 5048-5502 |

```yaml
task: Put five bits in the empty compartment of the box and reach for the last bit
eval_subtasks:
- grasp the bit left of the box nearest the robot base
- move the bit to the empty compartment
- release the bit in the empty compartment
- grasp the bit next to the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit farthest from the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit
eval_subtask_precisions: [5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]
eval_subtask_contacts: [top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch]
```

### T079 Put nine bits in the empty compartment of the box

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 4. Table: episode 4 (8322 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bit farthest from the robot base on the side away from the box | 5 | top-pinch | 2 | 4 | 0-572 |
| 2 | move the bit to the empty compartment | 3 | na | 2 | 4 | 572-802 |
| 3 | release the bit in the empty compartment | 4 | drop | 1 | 4 | 802-902 |
| 4 | grasp the bit farthest from the robot base | 5 | top-pinch | 1 | 4 | 902-1199 |
| 5 | move the bit to the compartment with the gray bits | 3 | na | 1 | 4 | 1199-1394 |
| 6 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 1394-1484 |
| 7 | grasp the bit farthest from the robot base | 5 | top-pinch | 2 | 3 | 1484-1929 |
| 8 | move the bit to the compartment with the gray bits | 3 | na | 1 | 4 | 1929-2109 |
| 9 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 2109-2269 |
| 10 | grasp the bit nearest the box | 5 | top-pinch | 2 | 3 | 2269-2564 |
| 11 | move the bit to the compartment with the gray bits | 3 | na | 1 | 4 | 2564-2784 |
| 12 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 2784-2899 |
| 13 | grasp the bit nearest the box | 5 | top-pinch | 2 | 5 | 2899-3369 |
| 14 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 3369-3564 |
| 15 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 3564-3674 |
| 16 | grasp the bit nearest the box | 5 | top-pinch | 2 | 5 | 3674-4214 |
| 17 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 4214-4459 |
| 18 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 4459-4549 |
| 19 | grasp the bit nearest the box | 5 | top-pinch | 2 | 5 | 4549-4955 |
| 20 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 4955-5205 |
| 21 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 5205-5325 |
| 22 | grasp one of the two touching bits | 5 | top-pinch | 1 | 3 | 5325-5920 |
| 23 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 5920-6145 |
| 24 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 3 | 6145-6235 |
| 25 | grasp the bit farther from the robot base | 5 | top-pinch | 1 | 2 | 6235-7393 |
| 26 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 7393-7663 |
| 27 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 7663-7791 |
| 28 | grasp the bit | 5 | top-pinch | 2 | 2 | 7791-8322 |

```yaml
task: Put nine bits in the empty compartment of the box
eval_subtasks:
- grasp the bit farthest from the robot base on the side away from the box
- move the bit to the empty compartment
- release the bit in the empty compartment
- grasp the bit farthest from the robot base
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit farthest from the robot base
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp one of the two touching bits
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit farther from the robot base
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit
eval_subtask_precisions: [5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]
eval_subtask_contacts: [top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch]
```

### T080 Put seven bits from the table in the compartment with the gray bits

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 7. Table: episode 7 (9519 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the nearer of the two bits left of the tape roll | 5 | top-pinch | 1 | 3 | 0-774 |
| 2 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 774-1068 |
| 3 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 1068-1188 |
| 4 | grasp the bit between the tape roll and the box, nearer the tape roll | 5 | top-pinch | 2 | 5 | 1188-1533 |
| 5 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 1533-1737 |
| 6 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 1737-1797 |
| 7 | grasp the lowest of the three bits on the right | 5 | top-pinch | 1 | 2 | 1797-2499 |
| 8 | move the bit to the compartment with the gray bits | 3 | na | 1 | 4 | 2499-2901 |
| 9 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 2901-2985 |
| 10 | grasp the leftmost bit | 5 | top-pinch | 2 | 5 | 2985-3420 |
| 11 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 3420-3666 |
| 12 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 3666-3810 |
| 13 | grasp the bit next to the box | 5 | top-pinch | 2 | 2 | 3810-4566 |
| 14 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 4566-4740 |
| 15 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 4740-4902 |
| 16 | grasp the lower of the two bits on the right | 5 | top-pinch | 1 | 1 | 4902-6825 |
| 17 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 6825-7125 |
| 18 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 7125-7275 |
| 19 | grasp the bit on the right | 5 | top-pinch | 1 | 2 | 7275-8430 |
| 20 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 8430-8790 |
| 21 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 8790-8880 |
| 22 | return to home | 1 | na | 2 | 4 | 8880-9519 |

```yaml
task: Put seven bits from the table in the compartment with the gray bits
eval_subtasks:
- grasp the nearer of the two bits left of the tape roll
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit between the tape roll and the box, nearer the tape roll
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the lowest of the three bits on the right
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the leftmost bit
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit next to the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the lower of the two bits on the right
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit on the right
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- return to home
eval_subtask_precisions: [5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 1]
eval_subtask_contacts: [top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, na]
```

### T081 Put seven bits in the compartment with the bits

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 5. Table: episode 5 (9228 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the slanted bit of the two touching bits | 5 | top-pinch | 2 | 5 | 0-618 |
| 2 | move the bit to the empty compartment | 2 | na | 2 | 4 | 618-825 |
| 3 | release the bit in the empty compartment | 3 | drop | 1 | 4 | 825-903 |
| 4 | grasp the bottom-left bit of the group | 5 | top-pinch | 1 | 2 | 903-2610 |
| 5 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 2610-2852 |
| 6 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 2852-2947 |
| 7 | grasp the bit nearest the box | 5 | top-pinch | 1 | 2 | 2947-4074 |
| 8 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 4074-4294 |
| 9 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 4294-4349 |
| 10 | grasp the bit nearest the box | 5 | top-pinch | 2 | 3 | 4349-4884 |
| 11 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 4884-5059 |
| 12 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 5059-5144 |
| 13 | grasp the bit nearest the box | 5 | top-pinch | 1 | 2 | 5144-5977 |
| 14 | move the bit to the compartment with the gray bits | 3 | na | 3 | 5 | 5977-6147 |
| 15 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 6147-6227 |
| 16 | grasp the leftmost bit | 5 | top-pinch | 2 | 5 | 6227-6842 |
| 17 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 6842-7057 |
| 18 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 7057-7212 |
| 19 | grasp the right bit of the two bits on the left | 5 | top-pinch | 1 | 2 | 7212-7848 |
| 20 | grasp the leftmost bit | 5 | top-pinch | 1 | 2 | 7848-8191 |
| 21 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 8191-8431 |
| 22 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 8431-8631 |
| 23 | grasp the bit farthest from the box | 5 | top-pinch | 2 | 4 | 8631-9228 |

```yaml
task: Put seven bits in the compartment with the bits
eval_subtasks:
- grasp the slanted bit of the two touching bits
- move the bit to the empty compartment
- release the bit in the empty compartment
- grasp the bottom-left bit of the group
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the leftmost bit
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the right bit of the two bits on the left
- grasp the leftmost bit
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit farthest from the box
eval_subtask_precisions: [5, 2, 3, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 5, 3, 4, 5]
eval_subtask_contacts: [top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, top-pinch, na, drop, top-pinch]
```

### T082 Put the bit in the compartment with the blue bits

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 10. Table: episode 10 (4230 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the leftmost bit | 5 | top-pinch | 1 | 1 | 0-3876 |
| 2 | move the bit to the compartment with the blue bits | 3 | na | 2 | 5 | 3876-4104 |
| 3 | release the bit in the compartment with the blue bits | 4 | drop | 1 | 5 | 4104-4230 |

```yaml
task: Put the bit in the compartment with the blue bits
eval_subtasks:
- grasp the leftmost bit
- move the bit to the compartment with the blue bits
- release the bit in the compartment with the blue bits
eval_subtask_precisions: [5, 3, 4]
eval_subtask_contacts: [top-pinch, na, drop]
```

### T083 Put the bits in the empty compartment of the box

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 0, 3. Table: episode 3 (7452 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bit nearest the robot base | 5 | top-pinch | 1 | 3 | 0-927 |
| 2 | move the bit to the empty compartment | 3 | na | 2 | 4 | 927-1287 |
| 3 | release the bit in the empty compartment | 4 | drop | 1 | 1 | 1287-1413 |
| 4 | grasp the bit on the left nearest the box | 5 | top-pinch | 1 | 2 | 1413-2167 |
| 5 | move the bit to the empty compartment | 3 | na | 2 | 4 | 2167-2487 |
| 6 | release the bit in the empty compartment | 4 | drop | 1 | 4 | 2487-2592 |
| 7 | grasp the bit on the left farther from the robot base | 5 | top-pinch | 2 | 3 | 2592-3140 |
| 8 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 3140-3480 |
| 9 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 3480-3589 |
| 10 | grasp the bit on the left | 5 | top-pinch | 2 | 5 | 3589-4179 |
| 11 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 4179-4524 |
| 12 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 4524-4689 |
| 13 | grasp the bit farthest from the robot base | 5 | top-pinch | 2 | 2 | 4689-5958 |
| 14 | move the bit to the compartment with the gray bits | 3 | na | 3 | 4 | 5958-6147 |
| 15 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 6147-6360 |
| 16 | grasp the bit | 5 | top-pinch | 2 | 3 | 6360-6750 |
| 17 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 6750-6942 |
| 18 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 6942-7020 |
| 19 | return to home | 1 | na | 3 | 5 | 7020-7452 |

```yaml
task: Put the bits in the empty compartment of the box
eval_subtasks:
- grasp the bit nearest the robot base
- move the bit to the empty compartment
- release the bit in the empty compartment
- grasp the bit on the left nearest the box
- move the bit to the empty compartment
- release the bit in the empty compartment
- grasp the bit on the left farther from the robot base
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit on the left
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit farthest from the robot base
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- return to home
eval_subtask_precisions: [5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 1]
eval_subtask_contacts: [top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, na]
```

<details><summary>Other episodes (1)</summary>


Episode 0 (4242 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the leftmost bit | 4 | top-pinch | 1 | 5 | 0-624 |
| 2 | move the bit to the empty compartment | 3 | na | 2 | 4 | 624-810 |
| 3 | release the bit in the empty compartment | 4 | drop | 1 | 4 | 810-897 |
| 4 | grasp the bit next to the box | 5 | top-pinch | 2 | 3 | 897-1362 |
| 5 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 1362-1524 |
| 6 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 1 | 1524-1656 |
| 7 | grasp the bit on the right | 5 | top-pinch | 1 | 2 | 1656-2568 |
| 8 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 2568-2766 |
| 9 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 2766-2889 |
| 10 | grasp the bit | 4 | top-pinch | 1 | 2 | 2889-3450 |
| 11 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 3450-3591 |
| 12 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 3591-3732 |
| 13 | return to home | 1 | na | 2 | 5 | 3732-4242 |

</details>

### T084 Put the bits in the empty compartment of the box, the sock in the bin and the last bit in the box

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 8. Table: episode 8 (10578 frames, mean quality 4.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bit between the sock and the box nearest the sock | 5 | top-pinch | 1 | 5 | 0-570 |
| 2 | move the bit to the empty compartment | 3 | na | 1 | 5 | 570-840 |
| 3 | release the bit in the empty compartment | 4 | drop | 1 | 4 | 840-960 |
| 4 | grasp the bit nearest the box | 5 | top-pinch | 2 | 5 | 960-1305 |
| 5 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 1305-1434 |
| 6 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 1434-1494 |
| 7 | grasp the bit left of the sock | 5 | top-pinch | 1 | 2 | 1494-2659 |
| 8 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 2659-2979 |
| 9 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 2979-3105 |
| 10 | grasp the leftmost bit | 5 | top-pinch | 1 | 4 | 3105-3705 |
| 11 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 3705-4035 |
| 12 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 4035-4200 |
| 13 | grasp the bit nearest the box | 5 | top-pinch | 1 | 5 | 4200-4635 |
| 14 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 4635-4755 |
| 15 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 4755-4820 |
| 16 | grasp the bit left of the box nearer the robot base | 5 | top-pinch | 2 | 5 | 4820-5170 |
| 17 | move the bit to the compartment with the gray bits | 3 | na | 1 | 4 | 5170-5328 |
| 18 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 5328-5380 |
| 19 | grasp the bit left of the box | 5 | top-pinch | 2 | 4 | 5380-5810 |
| 20 | move the bit to the compartment with the gray bits | 3 | na | 1 | 5 | 5810-5955 |
| 21 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 5955-6030 |
| 22 | grasp the bit nearest the table edge | 5 | top-pinch | 1 | 3 | 6030-6658 |
| 23 | move the bit to the compartment with the gray bits | 3 | na | 1 | 4 | 6658-6963 |
| 24 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 6963-7043 |
| 25 | grasp the bit next to the sock | 5 | top-pinch | 2 | 3 | 7043-7683 |
| 26 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 7683-7998 |
| 27 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 7998-8098 |
| 28 | grasp the sock | 1 | cloth-pinch | 1 | 2 | 8098-8748 |
| 29 | move the sock to the bin | 1 | na | 3 | 5 | 8748-8958 |
| 30 | release the sock in the bin | 1 | drop | 2 | 4 | 8958-9013 |
| 31 | grasp the bit | 5 | top-pinch | 2 | 4 | 9013-9651 |
| 32 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 9651-9786 |
| 33 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 9786-9911 |
| 34 | return to home | 1 | na | 2 | 4 | 9911-10578 |

```yaml
task: Put the bits in the empty compartment of the box, the sock in the bin and the last bit in the box
eval_subtasks:
- grasp the bit between the sock and the box nearest the sock
- move the bit to the empty compartment
- release the bit in the empty compartment
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit left of the sock
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the leftmost bit
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit left of the box nearer the robot base
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit left of the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit nearest the table edge
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit next to the sock
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the sock
- move the sock to the bin
- release the sock in the bin
- grasp the bit
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- return to home
eval_subtask_precisions: [5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 1, 1, 1, 5, 3, 4, 1]
eval_subtask_contacts: [top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, cloth-pinch, na, drop, top-pinch, na, drop, na]
```

### T085 Put the blue bit in the compartment with the blue bits and the red bit in the compartment with the yellow bits

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 12. Table: episode 12 (4371 frames, mean quality 2.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue bit | 5 | side-pinch | 1 | 2 | 0-897 |
| 2 | move the bit to the compartment with the blue bits | 3 | na | 2 | 4 | 897-1041 |
| 3 | release the bit in the compartment with the blue bits | 4 | drop | 1 | 4 | 1041-1164 |
| 4 | grasp the bit | 5 | top-pinch | 1 | 1 | 1164-2655 |
| 5 | move the bit to the compartment with the red bits | 4 | na | 2 | 1 | 2655-2796 |
| 6 | grasp the bit | 5 | top-pinch | 1 | 1 | 2796-3885 |
| 7 | move the bit to the compartment with the yellow bits | 3 | na | 1 | 4 | 3885-3990 |
| 8 | release the bit in the compartment with the yellow bits | 4 | drop | 1 | 4 | 3990-4113 |
| 9 | return to home | 1 | na | 3 | 4 | 4113-4371 |

```yaml
task: Put the blue bit in the compartment with the blue bits and the red bit in the compartment with the yellow bits
eval_subtasks:
- grasp the blue bit
- move the bit to the compartment with the blue bits
- release the bit in the compartment with the blue bits
- grasp the bit
- move the bit to the compartment with the red bits
- grasp the bit
- move the bit to the compartment with the yellow bits
- release the bit in the compartment with the yellow bits
- return to home
eval_subtask_precisions: [5, 3, 4, 5, 4, 5, 3, 4, 1]
eval_subtask_contacts: [side-pinch, na, drop, top-pinch, na, top-pinch, na, drop, na]
```

### T086 Put the blue bit in the compartment with the blue bits, the yellow bit in the compartment with the yellow bits and the red bit in the compartment with the gray bits

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 11. Table: episode 11 (4653 frames, mean quality 3.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue bit | 5 | top-pinch | 1 | 4 | 0-705 |
| 2 | move the bit to the compartment with the blue bits | 3 | na | 2 | 5 | 705-930 |
| 3 | release the bit in the compartment with the blue bits | 4 | drop | 1 | 3 | 930-1056 |
| 4 | grasp the yellow bit | 5 | top-pinch | 1 | 2 | 1056-1848 |
| 5 | move the bit to the compartment with the yellow bits | 3 | na | 2 | 5 | 1848-1980 |
| 6 | release the bit in the compartment with the yellow bits | 4 | drop | 1 | 4 | 1980-2157 |
| 7 | grasp the bit | 5 | top-pinch | 1 | 1 | 2157-3072 |
| 8 | move the bit to the compartment with the gray bits | 4 | na | 3 | 1 | 3072-3189 |
| 9 | grasp the bit | 5 | top-pinch | 1 | 1 | 3189-4044 |
| 10 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 4044-4212 |
| 11 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 4 | 4212-4356 |
| 12 | return to home | 1 | na | 3 | 4 | 4356-4653 |

```yaml
task: Put the blue bit in the compartment with the blue bits, the yellow bit in the compartment with the yellow bits and the red bit in the compartment with the gray bits
eval_subtasks:
- grasp the blue bit
- move the bit to the compartment with the blue bits
- release the bit in the compartment with the blue bits
- grasp the yellow bit
- move the bit to the compartment with the yellow bits
- release the bit in the compartment with the yellow bits
- grasp the bit
- move the bit to the compartment with the gray bits
- grasp the bit
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- return to home
eval_subtask_precisions: [5, 3, 4, 5, 3, 4, 5, 4, 5, 3, 4, 1]
eval_subtask_contacts: [top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, top-pinch, na, drop, na]
```

### T087 Put the four bits in the empty compartment of the component box

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 1. Table: episode 1 (4437 frames, mean quality 4.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bit farthest from the box | 5 | top-pinch | 2 | 4 | 0-702 |
| 2 | move the bit to the empty compartment | 3 | na | 2 | 5 | 702-1014 |
| 3 | release the bit in the empty compartment | 4 | drop | 1 | 5 | 1014-1125 |
| 4 | grasp the bit left of the box | 5 | top-pinch | 1 | 5 | 1125-1557 |
| 5 | move the bit to the compartment with the gray bits | 3 | na | 1 | 5 | 1557-1701 |
| 6 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 1701-1767 |
| 7 | grasp the bit farthest from the robot base | 5 | top-pinch | 1 | 2 | 1767-2934 |
| 8 | move the bit to the compartment with the gray bits | 3 | na | 3 | 5 | 2934-3099 |
| 9 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 3 | 3099-3285 |
| 10 | grasp the bit | 5 | top-pinch | 2 | 4 | 3285-3714 |
| 11 | move the bit to the compartment with the gray bits | 3 | na | 2 | 5 | 3714-3891 |
| 12 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 3891-4002 |
| 13 | return to home | 1 | na | 3 | 4 | 4002-4437 |

```yaml
task: Put the four bits in the empty compartment of the component box
eval_subtasks:
- grasp the bit farthest from the box
- move the bit to the empty compartment
- release the bit in the empty compartment
- grasp the bit left of the box
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit farthest from the robot base
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- return to home
eval_subtask_precisions: [5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 1]
eval_subtask_contacts: [top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, top-pinch, na, drop, na]
```

### T088 Put two bits in the empty compartment of the box and grasp a third bit

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `bits_clippers`, episodes 6. Table: episode 6 (2685 frames, mean quality 4.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bit nearest the tape roll on its left | 5 | top-pinch | 1 | 3 | 0-825 |
| 2 | move the bit to the empty compartment | 3 | na | 2 | 5 | 825-900 |
| 3 | release the bit in the empty compartment | 4 | drop | 2 | 4 | 900-1080 |
| 4 | grasp the bit nearest the tape roll on its right | 5 | top-pinch | 2 | 5 | 1080-1620 |
| 5 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 1620-1850 |
| 6 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 1850-2010 |
| 7 | grasp the bit right of the tape roll | 5 | top-pinch | 1 | 3 | 2010-2685 |

```yaml
task: Put two bits in the empty compartment of the box and grasp a third bit
eval_subtasks:
- grasp the bit nearest the tape roll on its left
- move the bit to the empty compartment
- release the bit in the empty compartment
- grasp the bit nearest the tape roll on its right
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- grasp the bit right of the tape roll
eval_subtask_precisions: [5, 3, 4, 5, 3, 4, 5]
eval_subtask_contacts: [top-pinch, na, drop, top-pinch, na, drop, top-pinch]
```

### T089 Close the headphone case, fold the bit kit closed and push the bit kit next to the headphone case

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `cases_kit_box_push`, episodes 17. Table: episode 17 (4080 frames, mean quality 3.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the lid | 3 | handle-grasp | 3 | 5 | 0-474 |
| 2 | close the lid | 2 | handle-grasp | 2 | 2 | 474-840 |
| 3 | grasp the lid | 3 | handle-grasp | 2 | 2 | 840-1416 |
| 4 | close the lid | 2 | handle-grasp | 1 | 4 | 1416-1797 |
| 5 | grasp the outer edge of the flap with the bits | 3 | cloth-pinch | 2 | 3 | 1797-2430 |
| 6 | fold the flap with the bits over the middle of the kit | 2 | cloth-pinch | 4 | 5 | 2430-2600 |
| 7 | grasp the outer edge of the flap with the white card | 3 | cloth-pinch | 3 | 3 | 2600-2961 |
| 8 | fold the flap with the white card over the folded flap | 2 | cloth-pinch | 3 | 3 | 2961-3171 |
| 9 | push the bit kit next to the headphone case | 2 | push | 2 | 3 | 3171-3882 |
| 10 | return to home | 1 | na | 4 | 5 | 3882-4080 |

```yaml
task: Close the headphone case, fold the bit kit closed and push the bit kit next to the headphone case
eval_subtasks:
- grasp the lid
- close the lid
- grasp the lid
- close the lid
- grasp the outer edge of the flap with the bits
- fold the flap with the bits over the middle of the kit
- grasp the outer edge of the flap with the white card
- fold the flap with the white card over the folded flap
- push the bit kit next to the headphone case
- return to home
eval_subtask_precisions: [3, 2, 3, 2, 3, 2, 3, 2, 2, 1]
eval_subtask_contacts: [handle-grasp, handle-grasp, handle-grasp, handle-grasp, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, push, na]
```

### T090 Fold the flap with the bits and then the flap with the white card to close the bit kit, move the headphone case next to the kit and close its lid

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `cases_kit_box_push`, episodes 13, 14. Table: episode 13 (3432 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the outer edge of the flap with the bits | 3 | cloth-pinch | 2 | 2 | 0-772 |
| 2 | fold the flap with the bits over the middle of the kit | 2 | set-down | 3 | 2 | 772-832 |
| 3 | grasp the outer edge of the flap with the bits | 3 | cloth-pinch | 1 | 2 | 832-1164 |
| 4 | fold the flap with the bits over the middle of the kit | 2 | set-down | 3 | 5 | 1164-1326 |
| 5 | grasp the outer edge of the flap with the white card | 3 | cloth-pinch | 3 | 5 | 1326-1650 |
| 6 | fold the flap with the white card over the folded flap | 2 | set-down | 3 | 3 | 1650-1845 |
| 7 | grasp the lid | 3 | handle-grasp | 2 | 3 | 1845-2580 |
| 8 | move the headphone case next to the bit kit | 1 | na | 3 | 4 | 2580-2958 |
| 9 | close the lid | 2 | drop | 2 | 5 | 2958-3036 |
| 10 | return to home | 1 | na | 2 | 5 | 3036-3432 |

```yaml
task: Fold the flap with the bits and then the flap with the white card to close the bit kit, move the headphone case next to the kit and close its lid
eval_subtasks:
- grasp the outer edge of the flap with the bits
- fold the flap with the bits over the middle of the kit
- grasp the outer edge of the flap with the bits
- fold the flap with the bits over the middle of the kit
- grasp the outer edge of the flap with the white card
- fold the flap with the white card over the folded flap
- grasp the lid
- move the headphone case next to the bit kit
- close the lid
- return to home
eval_subtask_precisions: [3, 2, 3, 2, 3, 2, 3, 1, 2, 1]
eval_subtask_contacts: [cloth-pinch, set-down, cloth-pinch, set-down, cloth-pinch, set-down, handle-grasp, na, drop, na]
```

<details><summary>Other episodes (1)</summary>


Episode 14 (3786 frames, mean quality 3.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the outer edge of the flap with the bits | 3 | cloth-pinch | 2 | 2 | 0-778 |
| 2 | fold the flap with the bits over the middle of the kit | 2 | set-down | 2 | 2 | 778-847 |
| 3 | grasp the outer edge of the flap with the bits | 3 | cloth-pinch | 1 | 2 | 847-1164 |
| 4 | fold the flap with the bits over the middle of the kit | 2 | set-down | 2 | 2 | 1164-1257 |
| 5 | grasp the outer edge of the flap with the bits | 3 | cloth-pinch | 2 | 2 | 1257-1640 |
| 6 | fold the flap with the bits over the middle of the kit | 2 | drop | 3 | 5 | 1640-1785 |
| 7 | grasp the outer edge of the flap with the white card | 3 | cloth-pinch | 3 | 5 | 1785-2142 |
| 8 | fold the flap with the white card over the folded flap | 2 | set-down | 3 | 3 | 2142-2330 |
| 9 | push the flap with the white card down onto the folded flap | 2 | push | 2 | 3 | 2330-2500 |
| 10 | grasp the lid | 3 | handle-grasp | 3 | 5 | 2500-3060 |
| 11 | move the headphone case next to the bit kit | 1 | na | 3 | 4 | 3060-3366 |
| 12 | close the lid | 2 | drop | 2 | 4 | 3366-3414 |
| 13 | return to home | 1 | na | 2 | 4 | 3414-3786 |

</details>

### T091 Fold the flap with the bits over the middle of the bit kit

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `cases_kit_box_push`, episodes 15. Table: episode 15 (1914 frames, mean quality 2.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the outer edge of the flap with the bits | 3 | cloth-pinch | 1 | 2 | 0-1596 |
| 2 | fold the flap with the bits over the middle of the kit | 2 | cloth-pinch | 3 | 3 | 1596-1914 |

```yaml
task: Fold the flap with the bits over the middle of the bit kit
eval_subtasks:
- grasp the outer edge of the flap with the bits
- fold the flap with the bits over the middle of the kit
eval_subtask_precisions: [3, 2]
eval_subtask_contacts: [cloth-pinch, cloth-pinch]
```

### T092 Fold the flap with the bits over the middle of the kit, fold the flap with the white card over it, press the kit flat and close the headphone case

Source `internal_bits_kit` (`outputs/rebot_bits_kitfold_2026-10-06-annotated-v2`), group `cases_kit_box_push`, episodes 16. Table: episode 16 (2496 frames, mean quality 4.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the outer edge of the flap with the bits | 3 | cloth-pinch | 3 | 5 | 0-522 |
| 2 | fold the flap with the bits over the middle of the kit | 2 | drop | 4 | 3 | 522-705 |
| 3 | grasp the outer edge of the flap with the white card | 3 | cloth-pinch | 2 | 3 | 705-1122 |
| 4 | fold the flap with the white card over the folded flap | 2 | set-down | 4 | 4 | 1122-1266 |
| 5 | press the kit flat | 1 | press | 2 | 3 | 1266-1503 |
| 6 | grasp the lid | 3 | rim-pinch | 2 | 5 | 1503-1878 |
| 7 | close the lid | 2 | drop | 2 | 5 | 1878-2100 |
| 8 | return to home | 1 | na | 3 | 5 | 2100-2496 |

```yaml
task: Fold the flap with the bits over the middle of the kit, fold the flap with the white card over it, press the kit flat and close the headphone case
eval_subtasks:
- grasp the outer edge of the flap with the bits
- fold the flap with the bits over the middle of the kit
- grasp the outer edge of the flap with the white card
- fold the flap with the white card over the folded flap
- press the kit flat
- grasp the lid
- close the lid
- return to home
eval_subtask_precisions: [3, 2, 3, 2, 1, 3, 2, 1]
eval_subtask_contacts: [cloth-pinch, drop, cloth-pinch, set-down, press, rim-pinch, drop, na]
```

### T093 Close the headphone case lid

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `cases_kit_box_push`, episodes 0, 2, 3. Table: episode 0 (1317 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the headphone case lid | 3 | rim-pinch | 2 | 4 | 0-510 |
| 2 | close the headphone case lid | 2 | rim-pinch | 2 | 4 | 510-813 |
| 3 | return to home | 1 | na | 2 | 4 | 813-1317 |

```yaml
task: Close the headphone case lid
eval_subtasks:
- grasp the headphone case lid
- close the headphone case lid
- return to home
eval_subtask_precisions: [3, 2, 1]
eval_subtask_contacts: [rim-pinch, rim-pinch, na]
```

<details><summary>Other episodes (2)</summary>


Episode 2 (1293 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the headphone case lid | 3 | rim-pinch | 2 | 4 | 0-564 |
| 2 | close the headphone case lid | 2 | rim-pinch | 2 | 4 | 564-873 |
| 3 | return to home | 1 | na | 2 | 4 | 873-1293 |


Episode 3 (1788 frames, mean quality 3.6)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the headphone case lid | 3 | rim-pinch | 2 | 5 | 0-456 |
| 2 | close the headphone case lid | 2 | rim-pinch | 2 | 2 | 456-741 |
| 3 | grasp the headphone case lid | 3 | rim-pinch | 2 | 5 | 741-1088 |
| 4 | close the headphone case lid | 2 | rim-pinch | 2 | 3 | 1088-1404 |
| 5 | return to home | 1 | na | 3 | 3 | 1404-1788 |

</details>

### T094 Close the headphone case lid and put the spray bottle in the basket

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `cases_kit_box_push`, episodes 1. Table: episode 1 (3036 frames, mean quality 2.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the headphone case lid | 3 | rim-pinch | 2 | 2 | 0-633 |
| 2 | close the headphone case lid | 2 | rim-pinch | 3 | 2 | 633-741 |
| 3 | grasp the headphone case lid | 3 | rim-pinch | 1 | 2 | 741-1284 |
| 4 | close the headphone case lid | 2 | rim-pinch | 1 | 3 | 1284-1650 |
| 5 | grasp the spray bottle | 3 | side-pinch | 2 | 3 | 1650-2256 |
| 6 | move the spray bottle to the basket | 1 | na | 2 | 3 | 2256-2550 |
| 7 | release the spray bottle in the basket | 1 | set-down | 2 | 4 | 2550-2766 |
| 8 | return to home | 1 | na | 3 | 4 | 2766-3036 |

```yaml
task: Close the headphone case lid and put the spray bottle in the basket
eval_subtasks:
- grasp the headphone case lid
- close the headphone case lid
- grasp the headphone case lid
- close the headphone case lid
- grasp the spray bottle
- move the spray bottle to the basket
- release the spray bottle in the basket
- return to home
eval_subtask_precisions: [3, 2, 3, 2, 3, 1, 1, 1]
eval_subtask_contacts: [rim-pinch, rim-pinch, rim-pinch, rim-pinch, side-pinch, na, set-down, na]
```

### T095 Close the headphone case lid, close the box lid and lay the box flat

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `cases_kit_box_push`, episodes 4. Table: episode 4 (3177 frames, mean quality 3.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the headphone case lid | 3 | rim-pinch | 2 | 2 | 0-621 |
| 2 | close the headphone case lid | 2 | rim-pinch | 2 | 2 | 621-846 |
| 3 | grasp the headphone case lid | 3 | rim-pinch | 1 | 2 | 846-1306 |
| 4 | close the headphone case lid | 2 | rim-pinch | 2 | 5 | 1306-1539 |
| 5 | grasp the box lid | 3 | cloth-pinch | 2 | 3 | 1539-2091 |
| 6 | close the box lid | 2 | cloth-pinch | 2 | 3 | 2091-2412 |
| 7 | lay the box flat | 2 | push | 3 | 5 | 2412-2703 |
| 8 | return to home | 1 | na | 2 | 4 | 2703-3177 |

```yaml
task: Close the headphone case lid, close the box lid and lay the box flat
eval_subtasks:
- grasp the headphone case lid
- close the headphone case lid
- grasp the headphone case lid
- close the headphone case lid
- grasp the box lid
- close the box lid
- lay the box flat
- return to home
eval_subtask_precisions: [3, 2, 3, 2, 3, 2, 2, 1]
eval_subtask_contacts: [rim-pinch, rim-pinch, rim-pinch, rim-pinch, cloth-pinch, cloth-pinch, push, na]
```

### T096 Lift the box lid

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `cases_kit_box_push`, episodes 22, 23. Table: episode 23 (666 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | lift the box lid | 4 | pry | 1 | 3 | 0-666 |

```yaml
task: Lift the box lid
eval_subtasks:
- lift the box lid
eval_subtask_precisions: [4]
eval_subtask_contacts: [pry]
```

<details><summary>Other episodes (1)</summary>


Episode 22 (489 frames, mean quality 1.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | lift the box lid | 4 | pry | 1 | 1 | 0-489 |

</details>

### T097 Push the books off the box

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `cases_kit_box_push`, episodes 19, 21. Table: episode 19 (645 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the books off the box toward the front edge of the table | 1 | push | 5 | 2 | 0-570 |
| 2 | return to home | 1 | na | 5 | 4 | 570-645 |

```yaml
task: Push the books off the box
eval_subtasks:
- push the books off the box toward the front edge of the table
- return to home
eval_subtask_precisions: [1, 1]
eval_subtask_contacts: [push, na]
```

<details><summary>Other episodes (1)</summary>


Episode 21 (1521 frames, mean quality 2.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the top book off the box toward the left edge of the table | 1 | push | 4 | 3 | 0-348 |
| 2 | push the book off the box toward the front edge of the table | 1 | push | 2 | 2 | 348-1521 |

</details>

### T098 Push the two books off the box toward the left edge of the table, then lift the flap with the shipping label and push it open

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `cases_kit_box_push`, episodes 20. Table: episode 20 (642 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the top book off the box toward the left edge of the table | 1 | push | 4 | 5 | 0-141 |
| 2 | push the book off the box toward the left edge of the table | 1 | push | 4 | 4 | 141-312 |
| 3 | lift the flap with the shipping label | 3 | cloth-pinch | 3 | 4 | 312-429 |
| 4 | push the flap with the shipping label open | 1 | push | 4 | 2 | 429-642 |

```yaml
task: Push the two books off the box toward the left edge of the table, then lift the flap with the shipping label and push it open
eval_subtasks:
- push the top book off the box toward the left edge of the table
- push the book off the box toward the left edge of the table
- lift the flap with the shipping label
- push the flap with the shipping label open
eval_subtask_precisions: [1, 1, 3, 1]
eval_subtask_contacts: [push, push, cloth-pinch, push]
```

### T099 Put the wall plug in the compartment with the yellow wall plugs, push the cap off the pill bottle, pour the pills into the box and put the pill bottle on the table

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `pills_caps`, episodes 13. Table: episode 13 (3738 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the wall plug | 5 | top-pinch | 2 | 4 | 0-567 |
| 2 | move the wall plug to the compartment with the yellow wall plugs | 2 | na | 3 | 5 | 567-900 |
| 3 | release the wall plug in the compartment with the yellow wall plugs | 3 | drop | 1 | 2 | 900-1041 |
| 4 | grasp the wall plug | 5 | top-pinch | 1 | 2 | 1041-1258 |
| 5 | move the wall plug to the compartment with the yellow wall plugs | 2 | na | 1 | 4 | 1258-1356 |
| 6 | release the wall plug in the compartment with the yellow wall plugs | 3 | drop | 1 | 4 | 1356-1518 |
| 7 | push the cap off the pill bottle | 3 | push | 2 | 4 | 1518-2103 |
| 8 | grasp the pill bottle | 3 | side-pinch | 2 | 5 | 2103-2346 |
| 9 | move the pill bottle to the box | 1 | na | 1 | 5 | 2346-2502 |
| 10 | pour the pills into the box | 2 | tilt-pour | 2 | 4 | 2502-3021 |
| 11 | move the pill bottle to the table | 1 | na | 1 | 4 | 3021-3135 |
| 12 | release the pill bottle on the table | 1 | set-down | 2 | 4 | 3135-3171 |
| 13 | return to home | 1 | na | 2 | 4 | 3171-3738 |

```yaml
task: Put the wall plug in the compartment with the yellow wall plugs, push the cap off the pill bottle, pour the pills into the box and put the pill bottle on the table
eval_subtasks:
- grasp the wall plug
- move the wall plug to the compartment with the yellow wall plugs
- release the wall plug in the compartment with the yellow wall plugs
- grasp the wall plug
- move the wall plug to the compartment with the yellow wall plugs
- release the wall plug in the compartment with the yellow wall plugs
- push the cap off the pill bottle
- grasp the pill bottle
- move the pill bottle to the box
- pour the pills into the box
- move the pill bottle to the table
- release the pill bottle on the table
- return to home
eval_subtask_precisions: [5, 2, 3, 5, 2, 3, 3, 3, 1, 2, 1, 1, 1]
eval_subtask_contacts: [top-pinch, na, drop, top-pinch, na, drop, push, side-pinch, na, tilt-pour, na, set-down, na]
```

### T100 Take the cap off the pill bottle and put it on the table, pour the pills into the box and put the pill bottle on the table

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `pills_caps`, episodes 11. Table: episode 11 (2712 frames, mean quality 4.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cap | 3 | side-pinch | 3 | 5 | 0-483 |
| 2 | take the cap off the pill bottle | 3 | side-pinch | 1 | 5 | 483-522 |
| 3 | move the cap to the table | 1 | na | 1 | 5 | 522-573 |
| 4 | release the cap on the table | 1 | set-down | 1 | 5 | 573-666 |
| 5 | grasp the pill bottle | 3 | side-pinch | 1 | 5 | 666-963 |
| 6 | move the pill bottle to the box | 1 | na | 2 | 5 | 963-1488 |
| 7 | pour the pills into the box | 2 | tilt-pour | 2 | 4 | 1488-1992 |
| 8 | move the pill bottle to the table | 1 | na | 1 | 5 | 1992-2124 |
| 9 | release the pill bottle on the table | 1 | set-down | 1 | 5 | 2124-2214 |
| 10 | return to home | 1 | na | 2 | 5 | 2214-2712 |

```yaml
task: Take the cap off the pill bottle and put it on the table, pour the pills into the box and put the pill bottle on the table
eval_subtasks:
- grasp the cap
- take the cap off the pill bottle
- move the cap to the table
- release the cap on the table
- grasp the pill bottle
- move the pill bottle to the box
- pour the pills into the box
- move the pill bottle to the table
- release the pill bottle on the table
- return to home
eval_subtask_precisions: [3, 3, 1, 1, 3, 1, 2, 1, 1, 1]
eval_subtask_contacts: [side-pinch, side-pinch, na, set-down, side-pinch, na, tilt-pour, na, set-down, na]
```

### T101 Take the cap off the pill bottle, pour the pills into the box and put the pill bottle on the table

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `pills_caps`, episodes 10, 12, 15, 16, 17. Table: episode 10 (2175 frames, mean quality 4.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the cap off the pill bottle | 3 | push | 2 | 4 | 0-567 |
| 2 | grasp the pill bottle | 3 | side-pinch | 2 | 3 | 567-885 |
| 3 | move the pill bottle to the box | 1 | na | 2 | 4 | 885-1290 |
| 4 | pour the pills into the box | 2 | tilt-pour | 3 | 5 | 1290-1635 |
| 5 | move the pill bottle to the table | 1 | na | 2 | 5 | 1635-1716 |
| 6 | release the pill bottle on the table | 1 | set-down | 1 | 5 | 1716-1776 |
| 7 | return to home | 1 | na | 3 | 3 | 1776-2175 |

```yaml
task: Take the cap off the pill bottle, pour the pills into the box and put the pill bottle on the table
eval_subtasks:
- push the cap off the pill bottle
- grasp the pill bottle
- move the pill bottle to the box
- pour the pills into the box
- move the pill bottle to the table
- release the pill bottle on the table
- return to home
eval_subtask_precisions: [3, 3, 1, 2, 1, 1, 1]
eval_subtask_contacts: [push, side-pinch, na, tilt-pour, na, set-down, na]
```

<details><summary>Other episodes (4)</summary>


Episode 12 (2226 frames, mean quality 3.9)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | take the cap off the pill bottle | 3 | side-pinch | 3 | 2 | 0-501 |
| 2 | grasp the pill bottle | 3 | side-pinch | 1 | 2 | 501-783 |
| 3 | move the pill bottle to the box | 1 | na | 3 | 5 | 783-921 |
| 4 | pour the pills into the box | 2 | tilt-pour | 2 | 4 | 921-1320 |
| 5 | move the pill bottle to the table | 1 | na | 3 | 4 | 1320-1440 |
| 6 | release the pill bottle on the table | 1 | set-down | 1 | 5 | 1440-1584 |
| 7 | return to home | 1 | na | 2 | 5 | 1584-2226 |


Episode 17 (1950 frames, mean quality 3.9)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | take the cap off the pill bottle | 3 | side-pinch | 3 | 3 | 0-534 |
| 2 | grasp the pill bottle | 3 | side-pinch | 1 | 3 | 534-834 |
| 3 | move the pill bottle to the box | 1 | na | 4 | 5 | 834-990 |
| 4 | pour the pills into the box | 2 | tilt-pour | 2 | 4 | 990-1380 |
| 5 | move the pill bottle to the table | 1 | na | 3 | 4 | 1380-1440 |
| 6 | release the pill bottle on the table | 1 | set-down | 2 | 4 | 1440-1638 |
| 7 | return to home | 1 | na | 3 | 4 | 1638-1950 |


Episode 15 (2703 frames, mean quality 3.6)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the cap off the pill bottle | 3 | push | 2 | 4 | 0-552 |
| 2 | grasp the pill bottle | 3 | side-pinch | 1 | 3 | 552-853 |
| 3 | move the pill bottle to the box | 1 | na | 2 | 4 | 853-1290 |
| 4 | pour the pills into the box | 2 | tilt-pour | 1 | 2 | 1290-2145 |
| 5 | move the pill bottle to the table | 1 | na | 2 | 2 | 2145-2382 |
| 6 | release the pill bottle on the table | 1 | set-down | 1 | 5 | 2382-2433 |
| 7 | return to home | 1 | na | 3 | 5 | 2433-2703 |


Episode 16 (2574 frames, mean quality 3.6)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | take the cap off the pill bottle | 3 | side-pinch | 3 | 3 | 0-573 |
| 2 | grasp the pill bottle | 3 | side-pinch | 1 | 2 | 573-1104 |
| 3 | move the pill bottle to the box | 1 | na | 2 | 5 | 1104-1360 |
| 4 | pour the pills into the box | 2 | tilt-pour | 2 | 2 | 1360-1960 |
| 5 | move the pill bottle to the table | 1 | na | 3 | 5 | 1960-2070 |
| 6 | release the pill bottle on the table | 1 | set-down | 1 | 4 | 2070-2178 |
| 7 | return to home | 1 | na | 2 | 4 | 2178-2574 |

</details>

### T102 Take the cap off the pill bottle, pour the pills into the cup and put the pill bottle on the table

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `pills_caps`, episodes 8. Table: episode 8 (2748 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cap | 3 | side-pinch | 1 | 1 | 0-1104 |
| 2 | take the cap off the pill bottle | 3 | side-pinch | 2 | 3 | 1104-1143 |
| 3 | grasp the pill bottle | 3 | side-pinch | 2 | 5 | 1143-1446 |
| 4 | move the pill bottle to the cup | 2 | na | 2 | 5 | 1446-1760 |
| 5 | pour the pills into the cup | 3 | tilt-pour | 1 | 2 | 1760-2408 |
| 6 | release the pill bottle on the table | 1 | drop | 2 | 1 | 2408-2490 |
| 7 | return to home | 1 | na | 4 | 4 | 2490-2748 |

```yaml
task: Take the cap off the pill bottle, pour the pills into the cup and put the pill bottle on the table
eval_subtasks:
- grasp the cap
- take the cap off the pill bottle
- grasp the pill bottle
- move the pill bottle to the cup
- pour the pills into the cup
- release the pill bottle on the table
- return to home
eval_subtask_precisions: [3, 3, 3, 2, 3, 1, 1]
eval_subtask_contacts: [side-pinch, side-pinch, side-pinch, na, tilt-pour, drop, na]
```

### T103 Take the cap off the pill bottle, pour the pills into the cup, put the pill bottle on the table and put the cup on the towel

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `pills_caps`, episodes 6, 7. Table: episode 6 (3750 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cap | 3 | side-pinch | 1 | 2 | 0-735 |
| 2 | take the cap off the pill bottle | 3 | side-pinch | 1 | 5 | 735-810 |
| 3 | move the cap to the table | 1 | na | 1 | 4 | 810-890 |
| 4 | release the cap on the table | 1 | set-down | 1 | 4 | 890-938 |
| 5 | grasp the pill bottle | 3 | side-pinch | 2 | 4 | 938-1260 |
| 6 | move the pill bottle to the cup | 2 | na | 2 | 5 | 1260-1500 |
| 7 | pour the pills into the cup | 3 | tilt-pour | 2 | 4 | 1500-1920 |
| 8 | move the pill bottle to the table | 1 | na | 1 | 4 | 1920-2040 |
| 9 | release the pill bottle on the table | 1 | set-down | 1 | 4 | 2040-2178 |
| 10 | grasp the cup | 3 | rim-pinch | 1 | 2 | 2178-2742 |
| 11 | move the cup to the towel | 1 | na | 2 | 4 | 2742-2985 |
| 12 | release the cup on the towel | 1 | set-down | 1 | 4 | 2985-3315 |
| 13 | return to home | 1 | na | 3 | 4 | 3315-3750 |

```yaml
task: Take the cap off the pill bottle, pour the pills into the cup, put the pill bottle on the table and put the cup on the towel
eval_subtasks:
- grasp the cap
- take the cap off the pill bottle
- move the cap to the table
- release the cap on the table
- grasp the pill bottle
- move the pill bottle to the cup
- pour the pills into the cup
- move the pill bottle to the table
- release the pill bottle on the table
- grasp the cup
- move the cup to the towel
- release the cup on the towel
- return to home
eval_subtask_precisions: [3, 3, 1, 1, 3, 2, 3, 1, 1, 3, 1, 1, 1]
eval_subtask_contacts: [side-pinch, side-pinch, na, set-down, side-pinch, na, tilt-pour, na, set-down, rim-pinch, na, set-down, na]
```

<details><summary>Other episodes (1)</summary>


Episode 7 (3600 frames, mean quality 3.4)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cap | 3 | side-pinch | 2 | 4 | 0-572 |
| 2 | take the cap off the pill bottle | 3 | side-pinch | 1 | 4 | 572-610 |
| 3 | move the cap to the table | 1 | na | 1 | 4 | 610-703 |
| 4 | release the cap on the table | 1 | drop | 1 | 3 | 703-742 |
| 5 | grasp the pill bottle | 3 | side-pinch | 2 | 3 | 742-1142 |
| 6 | move the pill bottle to the cup | 2 | na | 2 | 4 | 1142-1400 |
| 7 | pour the pills into the cup | 3 | tilt-pour | 2 | 1 | 1400-1985 |
| 8 | move the pill bottle to the table | 1 | na | 1 | 4 | 1985-2172 |
| 9 | release the pill bottle on the table | 1 | set-down | 2 | 4 | 2172-2428 |
| 10 | grasp the cup | 3 | rim-pinch | 2 | 4 | 2428-2788 |
| 11 | move the cup to the towel | 1 | na | 2 | 4 | 2788-3124 |
| 12 | release the cup on the towel | 1 | set-down | 1 | 4 | 3124-3296 |
| 13 | return to home | 1 | na | 4 | 1 | 3296-3600 |

</details>

### T104 Take the cap off the pill bottle, pour the pills into the cup, put the pill bottle on the towel and put the cup on the towel

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `pills_caps`, episodes 9. Table: episode 9 (3285 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cap | 3 | side-pinch | 3 | 5 | 0-534 |
| 2 | take the cap off the pill bottle | 3 | side-pinch | 1 | 5 | 534-600 |
| 3 | move the cap to the table | 1 | na | 2 | 4 | 600-645 |
| 4 | release the cap on the table | 1 | drop | 2 | 3 | 645-741 |
| 5 | grasp the pill bottle | 3 | side-pinch | 2 | 3 | 741-1047 |
| 6 | move the pill bottle to the cup | 2 | na | 2 | 3 | 1047-1440 |
| 7 | pour the pills into the cup | 3 | tilt-pour | 2 | 1 | 1440-1932 |
| 8 | move the pill bottle to the towel | 1 | na | 3 | 4 | 1932-2070 |
| 9 | release the pill bottle on the towel | 1 | set-down | 2 | 5 | 2070-2262 |
| 10 | grasp the cup | 3 | rim-pinch | 2 | 5 | 2262-2617 |
| 11 | move the cup to the towel | 1 | na | 2 | 4 | 2617-2820 |
| 12 | release the cup on the towel | 2 | drop | 1 | 4 | 2820-2919 |
| 13 | return to home | 1 | na | 2 | 3 | 2919-3285 |

```yaml
task: Take the cap off the pill bottle, pour the pills into the cup, put the pill bottle on the towel and put the cup on the towel
eval_subtasks:
- grasp the cap
- take the cap off the pill bottle
- move the cap to the table
- release the cap on the table
- grasp the pill bottle
- move the pill bottle to the cup
- pour the pills into the cup
- move the pill bottle to the towel
- release the pill bottle on the towel
- grasp the cup
- move the cup to the towel
- release the cup on the towel
- return to home
eval_subtask_precisions: [3, 3, 1, 1, 3, 2, 3, 1, 1, 3, 1, 2, 1]
eval_subtask_contacts: [side-pinch, side-pinch, na, drop, side-pinch, na, tilt-pour, na, set-down, rim-pinch, na, drop, na]
```

### T105 Take the cap off the pill bottle, put the cap on the table, pour the pills into the box and put the pill bottle on the table

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `pills_caps`, episodes 14. Table: episode 14 (1998 frames, mean quality 4.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cap | 3 | side-pinch | 3 | 5 | 0-557 |
| 2 | take the cap off the pill bottle | 3 | side-pinch | 2 | 5 | 557-620 |
| 3 | move the cap to the table | 1 | na | 2 | 3 | 620-749 |
| 4 | release the cap on the table | 1 | drop | 1 | 3 | 749-785 |
| 5 | grasp the pill bottle | 3 | side-pinch | 2 | 3 | 785-1001 |
| 6 | move the pill bottle to the box | 1 | na | 3 | 5 | 1001-1212 |
| 7 | pour the pills into the box | 2 | tilt-pour | 2 | 5 | 1212-1485 |
| 8 | release the pill bottle on the table | 1 | set-down | 1 | 4 | 1485-1634 |
| 9 | return to home | 1 | na | 2 | 4 | 1634-1998 |

```yaml
task: Take the cap off the pill bottle, put the cap on the table, pour the pills into the box and put the pill bottle on the table
eval_subtasks:
- grasp the cap
- take the cap off the pill bottle
- move the cap to the table
- release the cap on the table
- grasp the pill bottle
- move the pill bottle to the box
- pour the pills into the box
- release the pill bottle on the table
- return to home
eval_subtask_precisions: [3, 3, 1, 1, 3, 1, 2, 1, 1]
eval_subtask_contacts: [side-pinch, side-pinch, na, drop, side-pinch, na, tilt-pour, set-down, na]
```

### T106 Grasp the pill bottle

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `rigid`, episodes 24, 25. Table: episode 24 (975 frames, mean quality 2.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the pill bottle | 3 | side-pinch | 2 | 2 | 0-975 |

```yaml
task: Grasp the pill bottle
eval_subtasks:
- grasp the pill bottle
eval_subtask_precisions: [3]
eval_subtask_contacts: [side-pinch]
```

<details><summary>Other episodes (1)</summary>


Episode 25 (1299 frames, mean quality 2.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the pill bottle | 3 | side-pinch | 1 | 2 | 0-1299 |

</details>

### T107 Open the box and put the pill bottle in the box

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `rigid`, episodes 5. Table: episode 5 (2181 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | lift the box lid | 4 | pry | 2 | 3 | 0-768 |
| 2 | push the box lid open | 1 | push | 3 | 3 | 768-1011 |
| 3 | grasp the pill bottle | 3 | side-pinch | 3 | 3 | 1011-1599 |
| 4 | move the pill bottle to the box | 1 | na | 3 | 5 | 1599-1800 |
| 5 | release the pill bottle in the box | 2 | drop | 2 | 4 | 1800-1896 |
| 6 | return to home | 1 | na | 3 | 4 | 1896-2181 |

```yaml
task: Open the box and put the pill bottle in the box
eval_subtasks:
- lift the box lid
- push the box lid open
- grasp the pill bottle
- move the pill bottle to the box
- release the pill bottle in the box
- return to home
eval_subtask_precisions: [4, 1, 3, 1, 2, 1]
eval_subtask_contacts: [pry, push, side-pinch, na, drop, na]
```

### T108 Put the dental floss on the cutting board

Source `internal_oct04` (`outputs/rebot_additions_2026-10-04-annotated-v12`), group `rigid`, episodes 18. Table: episode 18 (591 frames, mean quality 3.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the dental floss | 3 | top-pinch | 3 | 4 | 0-168 |
| 2 | move the dental floss to the cutting board | 1 | na | 5 | 4 | 168-309 |
| 3 | release the dental floss on the cutting board | 1 | drop | 4 | 3 | 309-405 |
| 4 | return to home | 1 | na | 5 | 3 | 405-591 |

```yaml
task: Put the dental floss on the cutting board
eval_subtasks:
- grasp the dental floss
- move the dental floss to the cutting board
- release the dental floss on the cutting board
- return to home
eval_subtask_precisions: [3, 1, 1, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

### T109 Put the large nail clipper and the small nail clipper in the clipper case

Source `internal_oct07` (`outputs/rebot_teleop_2026-10-07-annotated-v1`), group `bits_clippers`, episodes 0. Table: episode 0 (2742 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the large nail clipper | 4 | top-pinch | 2 | 5 | 0-555 |
| 2 | move the large nail clipper to the clipper case | 4 | na | 3 | 5 | 555-765 |
| 3 | release the large nail clipper in the large slot | 5 | insert | 1 | 3 | 765-1062 |
| 4 | push the large nail clipper into the large slot | 3 | push | 1 | 4 | 1062-1221 |
| 5 | grasp the nail clipper | 4 | top-pinch | 1 | 2 | 1221-1848 |
| 6 | move the nail clipper to the clipper case | 4 | na | 2 | 5 | 1848-2022 |
| 7 | release the nail clipper in the small slot | 5 | insert | 1 | 4 | 2022-2184 |
| 8 | push the nail clipper into the small slot | 3 | push | 1 | 4 | 2184-2313 |
| 9 | return to home | 1 | na | 3 | 4 | 2313-2742 |

```yaml
task: Put the large nail clipper and the small nail clipper in the clipper case
eval_subtasks:
- grasp the large nail clipper
- move the large nail clipper to the clipper case
- release the large nail clipper in the large slot
- push the large nail clipper into the large slot
- grasp the nail clipper
- move the nail clipper to the clipper case
- release the nail clipper in the small slot
- push the nail clipper into the small slot
- return to home
eval_subtask_precisions: [4, 4, 5, 3, 4, 4, 5, 3, 1]
eval_subtask_contacts: [top-pinch, na, insert, push, top-pinch, na, insert, push, na]
```

### T110 Put the large nail clipper and the small nail clipper in the clipper case, fold the bit kit shut and push the headphone case off the box toward the table edge

Source `internal_oct07` (`outputs/rebot_teleop_2026-10-07-annotated-v1`), group `bits_clippers`, episodes 3. Table: episode 3 (4692 frames, mean quality 3.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the large nail clipper | 4 | top-pinch | 2 | 4 | 0-603 |
| 2 | move the large nail clipper to the clipper case | 4 | na | 3 | 2 | 603-975 |
| 3 | release the large nail clipper in the large slot | 5 | set-down | 1 | 2 | 975-1113 |
| 4 | grasp the large nail clipper | 4 | top-pinch | 1 | 2 | 1113-1158 |
| 5 | push the large nail clipper into the large slot | 3 | push | 1 | 2 | 1158-1650 |
| 6 | grasp the nail clipper | 4 | top-pinch | 1 | 2 | 1650-2694 |
| 7 | move the nail clipper to the clipper case | 4 | na | 2 | 5 | 2694-2872 |
| 8 | release the nail clipper in the small slot | 5 | insert | 2 | 5 | 2872-3135 |
| 9 | grasp the outer edge of the flap with the white card | 3 | cloth-pinch | 2 | 5 | 3135-3442 |
| 10 | fold the flap with the white card over the empty flap | 2 | cloth-pinch | 2 | 3 | 3442-3723 |
| 11 | push the headphone case off the box toward the table edge | 1 | push | 2 | 4 | 3723-4245 |
| 12 | return to home | 1 | na | 3 | 3 | 4245-4692 |

```yaml
task: Put the large nail clipper and the small nail clipper in the clipper case, fold the bit kit shut and push the headphone case off the box toward the table edge
eval_subtasks:
- grasp the large nail clipper
- move the large nail clipper to the clipper case
- release the large nail clipper in the large slot
- grasp the large nail clipper
- push the large nail clipper into the large slot
- grasp the nail clipper
- move the nail clipper to the clipper case
- release the nail clipper in the small slot
- grasp the outer edge of the flap with the white card
- fold the flap with the white card over the empty flap
- push the headphone case off the box toward the table edge
- return to home
eval_subtask_precisions: [4, 4, 5, 4, 3, 4, 4, 5, 3, 2, 1, 1]
eval_subtask_contacts: [top-pinch, na, set-down, top-pinch, push, top-pinch, na, insert, cloth-pinch, cloth-pinch, push, na]
```

### T111 Put the large nail clipper and the small nail clipper in the clipper case, pour the pills from the open pill bottle into the lid of the clipper case and set the open pill bottle back on the table

Source `internal_oct07` (`outputs/rebot_teleop_2026-10-07-annotated-v1`), group `bits_clippers`, episodes 4. Table: episode 4 (5277 frames, mean quality 3.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the large nail clipper | 4 | top-pinch | 1 | 4 | 0-777 |
| 2 | move the large nail clipper to the clipper case | 4 | na | 2 | 4 | 777-1134 |
| 3 | release the large nail clipper in the large slot | 5 | insert | 2 | 1 | 1134-1740 |
| 4 | grasp the large nail clipper | 4 | top-pinch | 1 | 2 | 1740-2308 |
| 5 | release the large nail clipper in the large slot | 5 | insert | 1 | 3 | 2308-2636 |
| 6 | grasp the nail clipper | 4 | top-pinch | 2 | 5 | 2636-3130 |
| 7 | move the nail clipper to the clipper case | 4 | na | 3 | 4 | 3130-3387 |
| 8 | release the nail clipper in the small slot | 5 | insert | 2 | 2 | 3387-3747 |
| 9 | grasp the open pill bottle | 3 | side-pinch | 2 | 2 | 3747-4218 |
| 10 | pour the pills into the lid of the clipper case | 2 | tilt-pour | 2 | 4 | 4218-4722 |
| 11 | move the open pill bottle to the table | 1 | na | 3 | 4 | 4722-4798 |
| 12 | release the open pill bottle on the table | 1 | set-down | 2 | 5 | 4798-4900 |
| 13 | return to home | 1 | na | 2 | 5 | 4900-5277 |

```yaml
task: Put the large nail clipper and the small nail clipper in the clipper case, pour the pills from the open pill bottle into the lid of the clipper case and set the open pill bottle back on the table
eval_subtasks:
- grasp the large nail clipper
- move the large nail clipper to the clipper case
- release the large nail clipper in the large slot
- grasp the large nail clipper
- release the large nail clipper in the large slot
- grasp the nail clipper
- move the nail clipper to the clipper case
- release the nail clipper in the small slot
- grasp the open pill bottle
- pour the pills into the lid of the clipper case
- move the open pill bottle to the table
- release the open pill bottle on the table
- return to home
eval_subtask_precisions: [4, 4, 5, 4, 5, 4, 4, 5, 3, 2, 1, 1, 1]
eval_subtask_contacts: [top-pinch, na, insert, top-pinch, insert, top-pinch, na, insert, side-pinch, tilt-pour, na, set-down, na]
```

### T112 Put the small nail clipper and the large nail clipper in the clipper case

Source `internal_oct07` (`outputs/rebot_teleop_2026-10-07-annotated-v1`), group `bits_clippers`, episodes 1. Table: episode 1 (3063 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the small nail clipper | 4 | top-pinch | 2 | 4 | 0-528 |
| 2 | move the small nail clipper to the clipper case | 4 | na | 2 | 5 | 528-756 |
| 3 | release the small nail clipper in the small slot | 5 | insert | 1 | 3 | 756-978 |
| 4 | push the small nail clipper into the small slot | 3 | push | 1 | 2 | 978-1410 |
| 5 | grasp the nail clipper | 4 | top-pinch | 2 | 4 | 1410-2028 |
| 6 | move the nail clipper to the clipper case | 4 | na | 2 | 4 | 2028-2340 |
| 7 | release the nail clipper in the large slot | 5 | insert | 2 | 3 | 2340-2718 |
| 8 | return to home | 1 | na | 4 | 4 | 2718-3063 |

```yaml
task: Put the small nail clipper and the large nail clipper in the clipper case
eval_subtasks:
- grasp the small nail clipper
- move the small nail clipper to the clipper case
- release the small nail clipper in the small slot
- push the small nail clipper into the small slot
- grasp the nail clipper
- move the nail clipper to the clipper case
- release the nail clipper in the large slot
- return to home
eval_subtask_precisions: [4, 4, 5, 3, 4, 4, 5, 1]
eval_subtask_contacts: [top-pinch, na, insert, push, top-pinch, na, insert, na]
```

### T113 Put the small nail clipper and the large nail clipper in the clipper case, close the headphone case lid and put the bit in the compartment with the gray bits

Source `internal_oct07` (`outputs/rebot_teleop_2026-10-07-annotated-v1`), group `bits_clippers`, episodes 2. Table: episode 2 (4350 frames, mean quality 4.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the small nail clipper | 4 | top-pinch | 2 | 5 | 0-615 |
| 2 | move the small nail clipper to the clipper case | 4 | na | 2 | 5 | 615-846 |
| 3 | release the small nail clipper in the small slot | 5 | insert | 1 | 4 | 846-1026 |
| 4 | grasp the nail clipper | 4 | top-pinch | 2 | 3 | 1026-1515 |
| 5 | move the nail clipper to the clipper case | 4 | na | 3 | 4 | 1515-1722 |
| 6 | release the nail clipper in the large slot | 5 | insert | 1 | 4 | 1722-1959 |
| 7 | straighten the nail clipper in the large slot | 4 | top-pinch | 1 | 2 | 1959-2286 |
| 8 | grasp the headphone case lid | 3 | rim-pinch | 2 | 4 | 2286-2742 |
| 9 | close the headphone case lid | 2 | rim-pinch | 2 | 5 | 2742-3036 |
| 10 | grasp the bit | 5 | top-pinch | 2 | 5 | 3036-3555 |
| 11 | move the bit to the compartment with the gray bits | 3 | na | 2 | 4 | 3555-3936 |
| 12 | release the bit in the compartment with the gray bits | 4 | drop | 1 | 5 | 3936-4080 |
| 13 | return to home | 1 | na | 4 | 5 | 4080-4350 |

```yaml
task: Put the small nail clipper and the large nail clipper in the clipper case, close the headphone case lid and put the bit in the compartment with the gray bits
eval_subtasks:
- grasp the small nail clipper
- move the small nail clipper to the clipper case
- release the small nail clipper in the small slot
- grasp the nail clipper
- move the nail clipper to the clipper case
- release the nail clipper in the large slot
- straighten the nail clipper in the large slot
- grasp the headphone case lid
- close the headphone case lid
- grasp the bit
- move the bit to the compartment with the gray bits
- release the bit in the compartment with the gray bits
- return to home
eval_subtask_precisions: [4, 4, 5, 4, 4, 5, 4, 3, 2, 5, 3, 4, 1]
eval_subtask_contacts: [top-pinch, na, insert, top-pinch, na, insert, top-pinch, rim-pinch, rim-pinch, top-pinch, na, drop, na]
```

### T114 Grasp the shaker bottle

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 1. Table: episode 1 (1572 frames, mean quality 1.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shaker bottle | 3 | side-pinch | 2 | 1 | 0-1572 |

```yaml
task: Grasp the shaker bottle
eval_subtasks:
- grasp the shaker bottle
eval_subtask_precisions: [3]
eval_subtask_contacts: [side-pinch]
```

### T115 Stand the pill bottle upright

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 10. Table: episode 10 (3873 frames, mean quality 2.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the pill bottle | 3 | top-pinch | 2 | 3 | 0-726 |
| 2 | stand the pill bottle upright | 2 | set-down | 3 | 2 | 726-1035 |
| 3 | grasp the pill bottle | 3 | top-pinch | 2 | 2 | 1035-2040 |
| 4 | stand the pill bottle upright | 2 | set-down | 3 | 2 | 2040-2430 |
| 5 | grasp the pill bottle | 3 | top-pinch | 3 | 2 | 2430-3036 |
| 6 | stand the pill bottle upright | 2 | set-down | 3 | 4 | 3036-3414 |
| 7 | return to home | 1 | na | 2 | 4 | 3414-3873 |

```yaml
task: Stand the pill bottle upright
eval_subtasks:
- grasp the pill bottle
- stand the pill bottle upright
- grasp the pill bottle
- stand the pill bottle upright
- grasp the pill bottle
- stand the pill bottle upright
- return to home
eval_subtask_precisions: [3, 2, 3, 2, 3, 2, 1]
eval_subtask_contacts: [top-pinch, set-down, top-pinch, set-down, top-pinch, set-down, na]
```

### T116 Stand the pill bottle upright and stand the water bottle upright

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 11. Table: episode 11 (3063 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the pill bottle | 3 | top-pinch | 3 | 5 | 0-657 |
| 2 | stand the pill bottle upright | 2 | set-down | 3 | 4 | 657-1084 |
| 3 | grasp the water bottle | 3 | top-pinch | 2 | 1 | 1084-2138 |
| 4 | stand the water bottle upright | 2 | set-down | 2 | 4 | 2138-2721 |
| 5 | return to home | 1 | na | 3 | 5 | 2721-3063 |

```yaml
task: Stand the pill bottle upright and stand the water bottle upright
eval_subtasks:
- grasp the pill bottle
- stand the pill bottle upright
- grasp the water bottle
- stand the water bottle upright
- return to home
eval_subtask_precisions: [3, 2, 3, 2, 1]
eval_subtask_contacts: [top-pinch, set-down, top-pinch, set-down, na]
```

### T117 Stand the shaker bottle upright

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 5. Table: episode 5 (2040 frames, mean quality 1.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shaker bottle | 3 | rim-pinch | 2 | 2 | 0-1072 |
| 2 | stand the shaker bottle upright | 2 | set-down | 3 | 2 | 1072-1395 |
| 3 | grasp the shaker bottle | 3 | rim-pinch | 2 | 1 | 1395-2040 |

```yaml
task: Stand the shaker bottle upright
eval_subtasks:
- grasp the shaker bottle
- stand the shaker bottle upright
- grasp the shaker bottle
eval_subtask_precisions: [3, 2, 3]
eval_subtask_contacts: [rim-pinch, set-down, rim-pinch]
```

### T118 Stand the shaker bottle upright and grasp the lid

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 2. Table: episode 2 (1515 frames, mean quality 2.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shaker bottle | 3 | rim-pinch | 2 | 3 | 0-570 |
| 2 | stand the shaker bottle upright | 2 | set-down | 3 | 3 | 570-1050 |
| 3 | grasp the lid | 3 | top-pinch | 1 | 2 | 1050-1515 |

```yaml
task: Stand the shaker bottle upright and grasp the lid
eval_subtasks:
- grasp the shaker bottle
- stand the shaker bottle upright
- grasp the lid
eval_subtask_precisions: [3, 2, 3]
eval_subtask_contacts: [rim-pinch, set-down, top-pinch]
```

### T119 Stand the shaker bottle upright and move the lid to the shaker bottle

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 3. Table: episode 3 (2424 frames, mean quality 2.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shaker bottle | 3 | rim-pinch | 2 | 2 | 0-969 |
| 2 | stand the shaker bottle upright | 2 | set-down | 3 | 4 | 969-1140 |
| 3 | grasp the lid | 3 | handle-grasp | 1 | 2 | 1140-2100 |
| 4 | move the lid to the shaker bottle | 1 | na | 2 | 3 | 2100-2424 |

```yaml
task: Stand the shaker bottle upright and move the lid to the shaker bottle
eval_subtasks:
- grasp the shaker bottle
- stand the shaker bottle upright
- grasp the lid
- move the lid to the shaker bottle
eval_subtask_precisions: [3, 2, 3, 1]
eval_subtask_contacts: [rim-pinch, set-down, handle-grasp, na]
```

### T120 Stand the shaker bottle upright and put the lid on the shaker bottle

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 0, 4. Table: episode 0 (1881 frames, mean quality 4.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shaker bottle | 3 | rim-pinch | 2 | 3 | 0-624 |
| 2 | stand the shaker bottle upright | 2 | set-down | 3 | 5 | 624-796 |
| 3 | grasp the lid | 3 | top-pinch | 2 | 5 | 796-1164 |
| 4 | move the lid to the shaker bottle | 3 | na | 4 | 4 | 1164-1344 |
| 5 | release the lid on the shaker bottle | 4 | insert | 1 | 4 | 1344-1542 |
| 6 | return to home | 1 | na | 3 | 4 | 1542-1881 |

```yaml
task: Stand the shaker bottle upright and put the lid on the shaker bottle
eval_subtasks:
- grasp the shaker bottle
- stand the shaker bottle upright
- grasp the lid
- move the lid to the shaker bottle
- release the lid on the shaker bottle
- return to home
eval_subtask_precisions: [3, 2, 3, 3, 4, 1]
eval_subtask_contacts: [rim-pinch, set-down, top-pinch, na, insert, na]
```

<details><summary>Other episodes (1)</summary>


Episode 4 (3282 frames, mean quality 3.4)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the shaker bottle | 3 | rim-pinch | 2 | 2 | 0-759 |
| 2 | stand the shaker bottle upright | 2 | set-down | 3 | 1 | 759-1014 |
| 3 | grasp the shaker bottle | 3 | rim-pinch | 2 | 1 | 1014-1800 |
| 4 | stand the shaker bottle upright | 2 | set-down | 3 | 5 | 1800-2034 |
| 5 | grasp the lid | 3 | top-pinch | 2 | 5 | 2034-2552 |
| 6 | move the lid to the shaker bottle | 3 | na | 2 | 5 | 2552-2760 |
| 7 | release the lid on the shaker bottle | 4 | insert | 1 | 3 | 2760-2886 |
| 8 | return to home | 1 | na | 2 | 5 | 2886-3282 |

</details>

### T121 Stand the water bottle upright and close the headphone case lid

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 8. Table: episode 8 (3036 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the water bottle | 3 | top-pinch | 1 | 1 | 0-1440 |
| 2 | stand the water bottle upright | 2 | set-down | 2 | 4 | 1440-1956 |
| 3 | grasp the headphone case lid | 3 | rim-pinch | 2 | 5 | 1956-2463 |
| 4 | close the headphone case lid | 2 | rim-pinch | 2 | 5 | 2463-2685 |
| 5 | return to home | 1 | na | 3 | 5 | 2685-3036 |

```yaml
task: Stand the water bottle upright and close the headphone case lid
eval_subtasks:
- grasp the water bottle
- stand the water bottle upright
- grasp the headphone case lid
- close the headphone case lid
- return to home
eval_subtask_precisions: [3, 2, 3, 2, 1]
eval_subtask_contacts: [top-pinch, set-down, rim-pinch, rim-pinch, na]
```

### T122 Stand the water bottle upright and fold the flap with the white card over the other half of the bit kit

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 6, 7. Table: episode 6 (1980 frames, mean quality 3.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the water bottle | 3 | top-pinch | 2 | 4 | 0-606 |
| 2 | stand the water bottle upright | 2 | set-down | 2 | 3 | 606-1080 |
| 3 | grasp the outer edge of the flap with the white card | 3 | cloth-pinch | 2 | 3 | 1080-1752 |
| 4 | fold the flap with the white card over the other half of the bit kit | 2 | cloth-pinch | 2 | 3 | 1752-1980 |

```yaml
task: Stand the water bottle upright and fold the flap with the white card over the other half of the bit kit
eval_subtasks:
- grasp the water bottle
- stand the water bottle upright
- grasp the outer edge of the flap with the white card
- fold the flap with the white card over the other half of the bit kit
eval_subtask_precisions: [3, 2, 3, 2]
eval_subtask_contacts: [top-pinch, set-down, cloth-pinch, cloth-pinch]
```

<details><summary>Other episodes (1)</summary>


Episode 7 (6528 frames, mean quality 2.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the water bottle | 3 | top-pinch | 2 | 3 | 0-735 |
| 2 | stand the water bottle upright | 2 | set-down | 2 | 2 | 735-948 |
| 3 | grasp the water bottle | 3 | top-pinch | 2 | 5 | 948-1268 |
| 4 | stand the water bottle upright | 2 | set-down | 2 | 2 | 1268-1890 |
| 5 | grasp the water bottle | 3 | top-pinch | 3 | 3 | 1890-2264 |
| 6 | stand the water bottle upright | 2 | set-down | 3 | 2 | 2264-2810 |
| 7 | grasp the water bottle | 3 | top-pinch | 2 | 1 | 2810-3596 |
| 8 | stand the water bottle upright | 2 | set-down | 4 | 2 | 3596-3920 |
| 9 | grasp the water bottle | 3 | top-pinch | 3 | 2 | 3920-4530 |
| 10 | stand the water bottle upright | 2 | set-down | 3 | 4 | 4530-4973 |
| 11 | grasp the outer edge of the flap with the white card | 3 | cloth-pinch | 4 | 3 | 4973-5604 |
| 12 | fold the flap with the white card over the other half of the bit kit | 2 | set-down | 2 | 3 | 5604-5900 |
| 13 | return to home | 1 | na | 1 | 3 | 5900-6528 |

</details>

### T123 Stand the water bottle upright and stand the pill bottle upright

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 12. Table: episode 12 (2796 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the water bottle | 3 | top-pinch | 1 | 3 | 0-800 |
| 2 | stand the water bottle upright | 2 | set-down | 3 | 3 | 800-1282 |
| 3 | grasp the pill bottle | 3 | top-pinch | 3 | 3 | 1282-2004 |
| 4 | stand the pill bottle upright | 2 | set-down | 3 | 3 | 2004-2472 |
| 5 | return to home | 1 | na | 3 | 3 | 2472-2796 |

```yaml
task: Stand the water bottle upright and stand the pill bottle upright
eval_subtasks:
- grasp the water bottle
- stand the water bottle upright
- grasp the pill bottle
- stand the pill bottle upright
- return to home
eval_subtask_precisions: [3, 2, 3, 2, 1]
eval_subtask_contacts: [top-pinch, set-down, top-pinch, set-down, na]
```

### T124 Stand the water bottle upright, close the headphone case lid and put the nail clipper in the clipper case

Source `internal_oct08` (`outputs/rebot_teleop_2026-10-08-annotated-v1`), group `bottles_lids`, episodes 9. Table: episode 9 (5019 frames, mean quality 2.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the water bottle | 3 | top-pinch | 2 | 2 | 0-1446 |
| 2 | stand the water bottle upright | 2 | set-down | 3 | 5 | 1446-1896 |
| 3 | grasp the headphone case lid | 3 | rim-pinch | 3 | 2 | 1896-2679 |
| 4 | close the headphone case lid | 2 | rim-pinch | 3 | 2 | 2679-2844 |
| 5 | grasp the headphone case lid | 3 | rim-pinch | 2 | 2 | 2844-3108 |
| 6 | close the headphone case lid | 2 | rim-pinch | 2 | 3 | 3108-3432 |
| 7 | grasp the nail clipper | 4 | top-pinch | 1 | 2 | 3432-4237 |
| 8 | move the nail clipper to the clipper case | 4 | na | 2 | 5 | 4237-4456 |
| 9 | release the nail clipper in the large slot | 5 | drop | 2 | 1 | 4456-4749 |
| 10 | return to home | 1 | na | 4 | 1 | 4749-5019 |

```yaml
task: Stand the water bottle upright, close the headphone case lid and put the nail clipper in the clipper case
eval_subtasks:
- grasp the water bottle
- stand the water bottle upright
- grasp the headphone case lid
- close the headphone case lid
- grasp the headphone case lid
- close the headphone case lid
- grasp the nail clipper
- move the nail clipper to the clipper case
- release the nail clipper in the large slot
- return to home
eval_subtask_precisions: [3, 2, 3, 2, 3, 2, 4, 4, 5, 1]
eval_subtask_contacts: [top-pinch, set-down, rim-pinch, rim-pinch, rim-pinch, rim-pinch, top-pinch, na, drop, na]
```

### T125 Fold the towel in half twice, straighten it, turn it to lie left to right, move it to the middle of the mat and push it into the far left corner of the mat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 179. Table: episode 179 (1737 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the far right corner of the towel | 1 | cloth-pinch | 4 | 1 | 0-186 |
| 2 | fold the towel in half from right to left | 1 | cloth-pinch | 5 | 5 | 186-264 |
| 3 | grasp the near left corner of the towel | 1 | cloth-pinch | 3 | 2 | 264-690 |
| 4 | grasp the far right corner of the towel | 1 | cloth-pinch | 5 | 5 | 690-774 |
| 5 | fold the towel in half from right to left | 1 | cloth-pinch | 5 | 3 | 774-924 |
| 6 | grasp the near end of the towel | 1 | cloth-pinch | 5 | 5 | 924-1017 |
| 7 | straighten the towel | 1 | cloth-pinch | 5 | 4 | 1017-1080 |
| 8 | turn the towel to lie left to right | 1 | push | 5 | 3 | 1080-1344 |
| 9 | grasp the right end of the towel | 1 | cloth-pinch | 3 | 5 | 1344-1452 |
| 10 | move the towel to the middle of the mat | 1 | na | 5 | 3 | 1452-1506 |
| 11 | release the towel on the mat | 1 | set-down | 5 | 4 | 1506-1536 |
| 12 | push the towel into the far left corner of the mat | 1 | push | 5 | 4 | 1536-1656 |
| 13 | return to home | 1 | na | 5 | 5 | 1656-1737 |

```yaml
task: Fold the towel in half twice, straighten it, turn it to lie left to right, move it to the middle of the mat and push it into the far left corner of the mat
eval_subtasks:
- grasp the far right corner of the towel
- fold the towel in half from right to left
- grasp the near left corner of the towel
- grasp the far right corner of the towel
- fold the towel in half from right to left
- grasp the near end of the towel
- straighten the towel
- turn the towel to lie left to right
- grasp the right end of the towel
- move the towel to the middle of the mat
- release the towel on the mat
- push the towel into the far left corner of the mat
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, push, cloth-pinch, na, set-down, push, na]
```

### T126 Fold the towel in half twice, turn it to lie left to right, fold it in half again and push it into the far left corner of the mat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 183. Table: episode 183 (1251 frames, mean quality 4.1).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the far right corner of the towel | 1 | cloth-pinch | 5 | 5 | 0-219 |
| 2 | fold the towel in half from right to left | 1 | cloth-pinch | 4 | 5 | 219-399 |
| 3 | grasp the folded edge of the towel | 1 | cloth-pinch | 5 | 3 | 399-585 |
| 4 | fold the towel in half from right to left | 1 | cloth-pinch | 5 | 3 | 585-690 |
| 5 | turn the towel to lie left to right | 1 | push | 5 | 4 | 690-840 |
| 6 | grasp the right end of the towel | 1 | cloth-pinch | 4 | 5 | 840-939 |
| 7 | fold the towel in half from right to left | 1 | cloth-pinch | 5 | 3 | 939-1029 |
| 8 | push the towel into the far left corner of the mat | 1 | push | 5 | 4 | 1029-1143 |
| 9 | return to home | 1 | na | 5 | 5 | 1143-1251 |

```yaml
task: Fold the towel in half twice, turn it to lie left to right, fold it in half again and push it into the far left corner of the mat
eval_subtasks:
- grasp the far right corner of the towel
- fold the towel in half from right to left
- grasp the folded edge of the towel
- fold the towel in half from right to left
- turn the towel to lie left to right
- grasp the right end of the towel
- fold the towel in half from right to left
- push the towel into the far left corner of the mat
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, push, cloth-pinch, cloth-pinch, push, na]
```

### T127 Fold the towel in half, pull its near left corner to the left, fold it in half again, turn it to lie left to right and put it in the far left corner of the mat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 182. Table: episode 182 (1275 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the far right corner of the towel | 2 | cloth-pinch | 5 | 1 | 0-225 |
| 2 | fold the towel in half from right to left | 1 | cloth-pinch | 4 | 4 | 225-384 |
| 3 | grasp the near left corner of the towel | 2 | cloth-pinch | 3 | 5 | 384-489 |
| 4 | pull the near left corner of the towel to the left | 1 | cloth-pinch | 5 | 4 | 489-549 |
| 5 | grasp the far right corner of the towel | 2 | cloth-pinch | 4 | 5 | 549-633 |
| 6 | fold the towel in half from right to left | 1 | cloth-pinch | 4 | 3 | 633-798 |
| 7 | turn the towel to lie left to right | 1 | push | 5 | 3 | 798-942 |
| 8 | grasp the right end of the towel | 1 | cloth-pinch | 3 | 5 | 942-987 |
| 9 | move the towel to the far left corner of the mat | 1 | na | 5 | 3 | 987-1152 |
| 10 | release the towel in the far left corner of the mat | 1 | set-down | 5 | 4 | 1152-1185 |
| 11 | return to home | 1 | na | 5 | 5 | 1185-1275 |

```yaml
task: Fold the towel in half, pull its near left corner to the left, fold it in half again, turn it to lie left to right and put it in the far left corner of the mat
eval_subtasks:
- grasp the far right corner of the towel
- fold the towel in half from right to left
- grasp the near left corner of the towel
- pull the near left corner of the towel to the left
- grasp the far right corner of the towel
- fold the towel in half from right to left
- turn the towel to lie left to right
- grasp the right end of the towel
- move the towel to the far left corner of the mat
- release the towel in the far left corner of the mat
- return to home
eval_subtask_precisions: [2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, push, cloth-pinch, na, set-down, na]
```

### T128 Fold the towel in half, straighten it, pull it to the far edge, slide it to the left and put it in the far left corner of the mat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 180. Table: episode 180 (975 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the far right corner of the towel | 1 | cloth-pinch | 4 | 5 | 0-189 |
| 2 | fold the towel in half from right to left | 1 | cloth-pinch | 5 | 5 | 189-279 |
| 3 | grasp the near left corner of the towel | 1 | cloth-pinch | 5 | 4 | 279-387 |
| 4 | straighten the towel | 1 | cloth-pinch | 5 | 4 | 387-420 |
| 5 | grasp the far right corner of the towel | 1 | cloth-pinch | 5 | 5 | 420-498 |
| 6 | pull the towel toward the far edge of the mat | 1 | cloth-pinch | 5 | 3 | 498-582 |
| 7 | slide the towel to the left side of the mat | 1 | push | 5 | 3 | 582-669 |
| 8 | grasp the right end of the towel | 1 | cloth-pinch | 5 | 5 | 669-735 |
| 9 | move the towel to the right side of the mat | 1 | na | 5 | 2 | 735-783 |
| 10 | release the towel on the right side of the mat | 1 | drop | 5 | 2 | 783-822 |
| 11 | move the towel to the far left corner of the mat | 1 | na | 5 | 3 | 822-867 |
| 12 | release the towel in the far left corner of the mat | 1 | drop | 5 | 3 | 867-885 |
| 13 | return to home | 1 | na | 5 | 5 | 885-975 |

```yaml
task: Fold the towel in half, straighten it, pull it to the far edge, slide it to the left and put it in the far left corner of the mat
eval_subtasks:
- grasp the far right corner of the towel
- fold the towel in half from right to left
- grasp the near left corner of the towel
- straighten the towel
- grasp the far right corner of the towel
- pull the towel toward the far edge of the mat
- slide the towel to the left side of the mat
- grasp the right end of the towel
- move the towel to the right side of the mat
- release the towel on the right side of the mat
- move the towel to the far left corner of the mat
- release the towel in the far left corner of the mat
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, push, cloth-pinch, na, drop, na, drop, na]
```

### T129 Fold the towel into a square and move it to the far left corner of the mat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 181. Table: episode 181 (927 frames, mean quality 3.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the far right corner of the towel | 1 | cloth-pinch | 4 | 1 | 0-168 |
| 2 | fold the towel in half from right to left | 1 | cloth-pinch | 5 | 5 | 168-262 |
| 3 | grasp the end of the top layer of the towel | 1 | cloth-pinch | 4 | 4 | 262-333 |
| 4 | line up the ends of the towel | 2 | cloth-pinch | 5 | 5 | 333-375 |
| 5 | grasp the folded edge of the towel | 1 | cloth-pinch | 5 | 5 | 375-447 |
| 6 | fold the towel in half again | 1 | cloth-pinch | 5 | 3 | 447-612 |
| 7 | grasp the right end of the towel | 1 | cloth-pinch | 4 | 3 | 612-688 |
| 8 | fold the towel into a square | 1 | cloth-pinch | 5 | 4 | 688-757 |
| 9 | move the towel to the far left corner of the mat | 1 | na | 5 | 3 | 757-852 |
| 10 | return to home | 1 | na | 5 | 5 | 852-927 |

```yaml
task: Fold the towel into a square and move it to the far left corner of the mat
eval_subtasks:
- grasp the far right corner of the towel
- fold the towel in half from right to left
- grasp the end of the top layer of the towel
- line up the ends of the towel
- grasp the folded edge of the towel
- fold the towel in half again
- grasp the right end of the towel
- fold the towel into a square
- move the towel to the far left corner of the mat
- return to home
eval_subtask_precisions: [1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, na, na]
```

### T130 Grasp the mallet nearest the robot, hit the puck back to the opponent and release the mallet on the table

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 211, 212, 213, 214, 215, 216, 217, 218, 219, 220. Table: episode 212 (513 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the mallet nearest the robot | 3 | handle-grasp | 4 | 5 | 0-160 |
| 2 | hit the puck back to the opponent | 2 | strike | 5 | 5 | 160-306 |
| 3 | release the mallet on the table | 1 | set-down | 3 | 5 | 306-394 |
| 4 | return to home | 1 | na | 5 | 5 | 394-513 |

```yaml
task: Grasp the mallet nearest the robot, hit the puck back to the opponent and release the mallet on the table
eval_subtasks:
- grasp the mallet nearest the robot
- hit the puck back to the opponent
- release the mallet on the table
- return to home
eval_subtask_precisions: [3, 2, 1, 1]
eval_subtask_contacts: [handle-grasp, strike, set-down, na]
```

<details><summary>Other episodes (9)</summary>


Episode 219 (585 frames, mean quality 4.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the mallet nearest the robot | 3 | handle-grasp | 4 | 5 | 0-204 |
| 2 | hit the puck back to the opponent | 2 | strike | 3 | 4 | 204-391 |
| 3 | release the mallet on the table | 1 | set-down | 4 | 5 | 391-460 |
| 4 | return to home | 1 | na | 5 | 5 | 460-585 |


Episode 211 (1740 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the mallet nearest the robot | 3 | handle-grasp | 3 | 5 | 0-182 |
| 2 | hit the puck back to the opponent | 2 | strike | 5 | 3 | 182-1515 |
| 3 | release the mallet on the table | 1 | set-down | 4 | 5 | 1515-1595 |
| 4 | return to home | 1 | na | 5 | 5 | 1595-1740 |


Episode 218 (1020 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the mallet nearest the robot | 3 | handle-grasp | 5 | 5 | 0-143 |
| 2 | hit the puck back to the opponent | 2 | strike | 4 | 3 | 143-852 |
| 3 | release the mallet on the table | 1 | set-down | 4 | 5 | 852-912 |
| 4 | return to home | 1 | na | 5 | 5 | 912-1020 |


Episode 220 (1989 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the mallet nearest the robot | 3 | handle-grasp | 4 | 5 | 0-180 |
| 2 | hit the puck back to the opponent | 2 | strike | 3 | 3 | 180-1788 |
| 3 | release the mallet on the table | 1 | set-down | 5 | 5 | 1788-1851 |
| 4 | return to home | 1 | na | 5 | 5 | 1851-1989 |


Episode 213 (2172 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the mallet nearest the robot | 3 | handle-grasp | 3 | 5 | 0-196 |
| 2 | hit the puck back to the opponent | 2 | strike | 4 | 2 | 196-1968 |
| 3 | release the mallet on the table | 1 | set-down | 5 | 5 | 1968-2038 |
| 4 | return to home | 1 | na | 5 | 5 | 2038-2172 |


Episode 217 (837 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the mallet nearest the robot | 3 | handle-grasp | 4 | 5 | 0-171 |
| 2 | hit the puck back to the opponent | 2 | strike | 3 | 2 | 171-624 |
| 3 | move the mallet to the bottom-right circle | 1 | na | 5 | 4 | 624-687 |
| 4 | release the mallet on the table | 1 | set-down | 5 | 5 | 687-720 |
| 5 | return to home | 1 | na | 4 | 5 | 720-837 |


Episode 214 (786 frames, mean quality 3.4)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the mallet nearest the robot | 3 | handle-grasp | 5 | 5 | 0-138 |
| 2 | hit the puck back to the opponent | 2 | strike | 4 | 5 | 138-378 |
| 3 | move the mallet to the circle on the robot's left | 1 | na | 4 | 5 | 378-432 |
| 4 | release the mallet on the table | 1 | set-down | 2 | 1 | 432-612 |
| 5 | return to home | 1 | na | 3 | 1 | 612-786 |


Episode 216 (1692 frames, mean quality 3.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the mallet nearest the robot | 3 | handle-grasp | 4 | 5 | 0-197 |
| 2 | hit the puck back to the opponent | 2 | strike | 4 | 2 | 197-1434 |
| 3 | release the mallet on the table | 1 | set-down | 4 | 5 | 1434-1527 |
| 4 | return to home | 1 | na | 4 | 1 | 1527-1692 |


Episode 215 (1215 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the mallet nearest the robot | 3 | handle-grasp | 2 | 1 | 0-225 |
| 2 | hit the puck back to the opponent | 2 | strike | 4 | 5 | 225-915 |
| 3 | release the mallet on the table | 1 | set-down | 2 | 5 | 915-1007 |
| 4 | return to home | 1 | na | 4 | 1 | 1007-1215 |

</details>

### T131 Move the block to the other cross

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 3, 4. Table: episode 3 (444 frames, mean quality 4.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the block | 3 | top-pinch | 5 | 5 | 0-237 |
| 2 | move the block to the other cross | 1 | na | 5 | 5 | 237-285 |
| 3 | release the block on the other cross | 2 | drop | 4 | 4 | 285-312 |
| 4 | return to home | 1 | na | 5 | 4 | 312-444 |

```yaml
task: Move the block to the other cross
eval_subtasks:
- grasp the block
- move the block to the other cross
- release the block on the other cross
- return to home
eval_subtask_precisions: [3, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (1)</summary>


Episode 4 (378 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the block | 3 | top-pinch | 5 | 5 | 0-177 |
| 2 | move the block to the other cross | 1 | na | 5 | 5 | 177-210 |
| 3 | release the block on the other cross | 2 | set-down | 4 | 5 | 210-243 |
| 4 | return to home | 1 | na | 5 | 3 | 243-378 |

</details>

### T132 Open the lower drawer, put the green eraser in the lower drawer and close the lower drawer

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 164, 165, 166. Table: episode 166 (882 frames, mean quality 4.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the handle of the lower drawer | 3 | handle-grasp | 5 | 3 | 0-207 |
| 2 | open the lower drawer | 1 | handle-grasp | 2 | 5 | 207-297 |
| 3 | grasp the green eraser | 3 | top-pinch | 5 | 5 | 297-433 |
| 4 | move the green eraser to the lower drawer | 1 | na | 5 | 5 | 433-487 |
| 5 | release the green eraser in the lower drawer | 2 | drop | 2 | 4 | 487-530 |
| 6 | close the lower drawer | 1 | push | 3 | 5 | 530-691 |
| 7 | return to home | 1 | na | 5 | 4 | 691-882 |

```yaml
task: Open the lower drawer, put the green eraser in the lower drawer and close the lower drawer
eval_subtasks:
- grasp the handle of the lower drawer
- open the lower drawer
- grasp the green eraser
- move the green eraser to the lower drawer
- release the green eraser in the lower drawer
- close the lower drawer
- return to home
eval_subtask_precisions: [3, 1, 3, 1, 2, 1, 1]
eval_subtask_contacts: [handle-grasp, handle-grasp, top-pinch, na, drop, push, na]
```

<details><summary>Other episodes (2)</summary>


Episode 165 (888 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the handle of the lower drawer | 3 | handle-grasp | 5 | 3 | 0-222 |
| 2 | open the lower drawer | 1 | handle-grasp | 4 | 5 | 222-282 |
| 3 | grasp the green eraser | 3 | top-pinch | 5 | 4 | 282-491 |
| 4 | move the green eraser to the lower drawer | 1 | na | 5 | 5 | 491-540 |
| 5 | release the green eraser in the lower drawer | 2 | drop | 4 | 4 | 540-574 |
| 6 | close the lower drawer | 1 | push | 4 | 5 | 574-712 |
| 7 | return to home | 1 | na | 5 | 4 | 712-888 |


Episode 164 (1164 frames, mean quality 3.6)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the handle of the lower drawer | 3 | handle-grasp | 3 | 2 | 0-396 |
| 2 | open the lower drawer | 1 | handle-grasp | 2 | 5 | 396-474 |
| 3 | grasp the green eraser | 3 | top-pinch | 5 | 3 | 474-654 |
| 4 | move the green eraser to the lower drawer | 1 | na | 5 | 4 | 654-705 |
| 5 | release the green eraser in the lower drawer | 2 | drop | 4 | 4 | 705-762 |
| 6 | close the lower drawer | 1 | push | 2 | 3 | 762-978 |
| 7 | return to home | 1 | na | 5 | 4 | 978-1164 |

</details>

### T133 Open the lower drawer, put the yellow eraser in the lower drawer and close the lower drawer

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 159, 160. Table: episode 160 (1269 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the handle of the lower drawer | 3 | handle-grasp | 4 | 1 | 0-335 |
| 2 | open the lower drawer | 1 | handle-grasp | 4 | 5 | 335-398 |
| 3 | grasp the yellow eraser | 3 | top-pinch | 5 | 3 | 398-744 |
| 4 | move the yellow eraser to the lower drawer | 1 | na | 5 | 4 | 744-816 |
| 5 | release the yellow eraser in the lower drawer | 2 | drop | 4 | 4 | 816-876 |
| 6 | close the lower drawer | 1 | push | 3 | 5 | 876-1038 |
| 7 | return to home | 1 | na | 5 | 4 | 1038-1269 |

```yaml
task: Open the lower drawer, put the yellow eraser in the lower drawer and close the lower drawer
eval_subtasks:
- grasp the handle of the lower drawer
- open the lower drawer
- grasp the yellow eraser
- move the yellow eraser to the lower drawer
- release the yellow eraser in the lower drawer
- close the lower drawer
- return to home
eval_subtask_precisions: [3, 1, 3, 1, 2, 1, 1]
eval_subtask_contacts: [handle-grasp, handle-grasp, top-pinch, na, drop, push, na]
```

<details><summary>Other episodes (1)</summary>


Episode 159 (1419 frames, mean quality 2.9)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the handle of the lower drawer | 3 | handle-grasp | 4 | 1 | 0-348 |
| 2 | open the lower drawer | 1 | handle-grasp | 3 | 5 | 348-429 |
| 3 | grasp the yellow eraser | 3 | top-pinch | 4 | 2 | 429-867 |
| 4 | move the yellow eraser to the lower drawer | 1 | na | 5 | 4 | 867-954 |
| 5 | release the yellow eraser in the lower drawer | 2 | drop | 2 | 4 | 954-1005 |
| 6 | close the lower drawer | 1 | push | 3 | 3 | 1005-1191 |
| 7 | return to home | 1 | na | 5 | 1 | 1191-1419 |

</details>

### T134 Open the upper drawer, put the green eraser in the upper drawer and close the upper drawer

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 161, 162, 163. Table: episode 163 (729 frames, mean quality 4.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the handle of the upper drawer | 3 | handle-grasp | 5 | 3 | 0-189 |
| 2 | open the upper drawer | 1 | handle-grasp | 5 | 5 | 189-240 |
| 3 | grasp the green eraser | 3 | top-pinch | 5 | 5 | 240-375 |
| 4 | move the green eraser to the upper drawer | 1 | na | 5 | 5 | 375-438 |
| 5 | release the green eraser in the upper drawer | 2 | drop | 4 | 4 | 438-466 |
| 6 | close the upper drawer | 1 | push | 5 | 5 | 466-572 |
| 7 | return to home | 1 | na | 5 | 4 | 572-729 |

```yaml
task: Open the upper drawer, put the green eraser in the upper drawer and close the upper drawer
eval_subtasks:
- grasp the handle of the upper drawer
- open the upper drawer
- grasp the green eraser
- move the green eraser to the upper drawer
- release the green eraser in the upper drawer
- close the upper drawer
- return to home
eval_subtask_precisions: [3, 1, 3, 1, 2, 1, 1]
eval_subtask_contacts: [handle-grasp, handle-grasp, top-pinch, na, drop, push, na]
```

<details><summary>Other episodes (2)</summary>


Episode 161 (1455 frames, mean quality 3.9)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the handle of the upper drawer | 3 | handle-grasp | 5 | 3 | 0-345 |
| 2 | open the upper drawer | 1 | handle-grasp | 3 | 5 | 345-408 |
| 3 | grasp the green eraser | 3 | top-pinch | 5 | 3 | 408-807 |
| 4 | move the green eraser to the upper drawer | 1 | na | 5 | 3 | 807-975 |
| 5 | release the green eraser in the upper drawer | 2 | drop | 2 | 4 | 975-1035 |
| 6 | close the upper drawer | 1 | push | 4 | 5 | 1035-1257 |
| 7 | return to home | 1 | na | 5 | 4 | 1257-1455 |


Episode 162 (1029 frames, mean quality 3.4)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the handle of the upper drawer | 3 | handle-grasp | 5 | 1 | 0-300 |
| 2 | open the upper drawer | 1 | handle-grasp | 4 | 5 | 300-357 |
| 3 | grasp the green eraser | 3 | top-pinch | 5 | 3 | 357-618 |
| 4 | move the green eraser to the upper drawer | 1 | na | 5 | 5 | 618-700 |
| 5 | release the green eraser in the upper drawer | 2 | drop | 4 | 3 | 700-744 |
| 6 | close the upper drawer | 1 | push | 4 | 3 | 744-848 |
| 7 | return to home | 1 | na | 4 | 4 | 848-1029 |

</details>

### T135 Open the upper drawer, put the yellow eraser in the upper drawer and close the upper drawer

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 157, 158. Table: episode 157 (1557 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the handle of the upper drawer | 3 | handle-grasp | 1 | 1 | 0-606 |
| 2 | open the upper drawer | 1 | handle-grasp | 3 | 5 | 606-690 |
| 3 | grasp the yellow eraser | 3 | top-pinch | 4 | 3 | 690-933 |
| 4 | move the yellow eraser to the upper drawer | 1 | na | 5 | 3 | 933-1056 |
| 5 | release the yellow eraser in the upper drawer | 2 | drop | 1 | 4 | 1056-1107 |
| 6 | close the upper drawer | 1 | push | 3 | 5 | 1107-1305 |
| 7 | return to home | 1 | na | 4 | 4 | 1305-1557 |

```yaml
task: Open the upper drawer, put the yellow eraser in the upper drawer and close the upper drawer
eval_subtasks:
- grasp the handle of the upper drawer
- open the upper drawer
- grasp the yellow eraser
- move the yellow eraser to the upper drawer
- release the yellow eraser in the upper drawer
- close the upper drawer
- return to home
eval_subtask_precisions: [3, 1, 3, 1, 2, 1, 1]
eval_subtask_contacts: [handle-grasp, handle-grasp, top-pinch, na, drop, push, na]
```

<details><summary>Other episodes (1)</summary>


Episode 158 (1686 frames, mean quality 2.9)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the handle of the upper drawer | 3 | handle-grasp | 2 | 1 | 0-444 |
| 2 | open the upper drawer | 1 | handle-grasp | 3 | 5 | 444-522 |
| 3 | grasp the yellow eraser | 3 | top-pinch | 2 | 1 | 522-1146 |
| 4 | move the yellow eraser to the upper drawer | 1 | na | 5 | 3 | 1146-1268 |
| 5 | release the yellow eraser in the upper drawer | 2 | drop | 3 | 3 | 1268-1322 |
| 6 | close the upper drawer | 1 | push | 4 | 3 | 1322-1466 |
| 7 | return to home | 1 | na | 4 | 4 | 1466-1686 |

</details>

### T136 Plug the charger into the power strip

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 231, 232, 233, 234, 235, 236, 237, 238, 239, 240. Table: episode 237 (582 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | side-pinch | 4 | 5 | 0-242 |
| 2 | plug the charger into the power strip | 5 | insert | 2 | 5 | 242-450 |
| 3 | return to home | 1 | na | 5 | 5 | 450-582 |

```yaml
task: Plug the charger into the power strip
eval_subtasks:
- grasp the charger
- plug the charger into the power strip
- return to home
eval_subtask_precisions: [3, 5, 1]
eval_subtask_contacts: [side-pinch, insert, na]
```

<details><summary>Other episodes (9)</summary>


Episode 236 (384 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | side-pinch | 5 | 5 | 0-126 |
| 2 | plug the charger into the power strip | 5 | insert | 3 | 3 | 126-300 |
| 3 | return to home | 1 | na | 5 | 5 | 300-384 |


Episode 231 (744 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | side-pinch | 4 | 1 | 0-280 |
| 2 | move the charger to the power strip | 4 | na | 5 | 5 | 280-348 |
| 3 | plug the charger into the power strip | 5 | insert | 2 | 4 | 348-594 |
| 4 | return to home | 1 | na | 5 | 4 | 594-744 |


Episode 235 (300 frames, mean quality 3.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | side-pinch | 5 | 1 | 0-114 |
| 2 | plug the charger into the power strip | 5 | insert | 4 | 5 | 114-222 |
| 3 | return to home | 1 | na | 5 | 4 | 222-300 |


Episode 232 (735 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | side-pinch | 3 | 1 | 0-282 |
| 2 | move the charger to the power strip | 4 | na | 5 | 5 | 282-342 |
| 3 | plug the charger into the power strip | 5 | insert | 2 | 2 | 342-570 |
| 4 | return to home | 1 | na | 5 | 4 | 570-735 |


Episode 238 (618 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | side-pinch | 4 | 1 | 0-236 |
| 2 | plug the charger into the power strip | 5 | insert | 2 | 4 | 236-474 |
| 3 | return to home | 1 | na | 5 | 4 | 474-618 |


Episode 239 (639 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | side-pinch | 4 | 1 | 0-258 |
| 2 | plug the charger into the power strip | 5 | insert | 2 | 4 | 258-480 |
| 3 | return to home | 1 | na | 5 | 4 | 480-639 |


Episode 240 (480 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | side-pinch | 4 | 1 | 0-207 |
| 2 | plug the charger into the power strip | 5 | insert | 3 | 4 | 207-360 |
| 3 | return to home | 1 | na | 5 | 4 | 360-480 |


Episode 233 (606 frames, mean quality 2.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | side-pinch | 4 | 1 | 0-257 |
| 2 | plug the charger into the power strip | 5 | insert | 5 | 3 | 257-459 |
| 3 | return to home | 1 | na | 5 | 4 | 459-606 |


Episode 234 (612 frames, mean quality 2.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | side-pinch | 4 | 1 | 0-260 |
| 2 | plug the charger into the power strip | 5 | insert | 2 | 3 | 260-498 |
| 3 | return to home | 1 | na | 5 | 4 | 498-612 |

</details>

### T137 Pour the egg into the pan and put the bowl back on the mat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 23, 24. Table: episode 24 (672 frames, mean quality 4.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bowl with the egg | 3 | rim-pinch | 4 | 5 | 0-244 |
| 2 | move the bowl with the egg to the pan | 1 | na | 5 | 5 | 244-312 |
| 3 | pour the egg into the pan | 2 | tilt-pour | 5 | 5 | 312-498 |
| 4 | move the bowl to the mat | 1 | na | 5 | 5 | 498-546 |
| 5 | release the bowl on the mat | 2 | set-down | 3 | 4 | 546-634 |
| 6 | return to home | 1 | na | 5 | 5 | 634-672 |

```yaml
task: Pour the egg into the pan and put the bowl back on the mat
eval_subtasks:
- grasp the bowl with the egg
- move the bowl with the egg to the pan
- pour the egg into the pan
- move the bowl to the mat
- release the bowl on the mat
- return to home
eval_subtask_precisions: [3, 1, 2, 1, 2, 1]
eval_subtask_contacts: [rim-pinch, na, tilt-pour, na, set-down, na]
```

<details><summary>Other episodes (1)</summary>


Episode 23 (690 frames, mean quality 4.6)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bowl with the egg | 3 | rim-pinch | 5 | 5 | 0-207 |
| 2 | pour the egg into the pan | 2 | tilt-pour | 5 | 5 | 207-432 |
| 3 | move the bowl to the mat | 1 | na | 5 | 5 | 432-492 |
| 4 | release the bowl on the mat | 2 | set-down | 4 | 4 | 492-558 |
| 5 | return to home | 1 | na | 4 | 4 | 558-690 |

</details>

### T138 Pour the ham, the egg and the noodles into the pan and put each bowl back on the mat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 29, 30. Table: episode 29 (1593 frames, mean quality 4.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bowl with the ham | 3 | rim-pinch | 5 | 5 | 0-228 |
| 2 | move the bowl with the ham to the pan | 1 | na | 5 | 5 | 228-282 |
| 3 | pour the ham into the pan | 2 | tilt-pour | 5 | 4 | 282-345 |
| 4 | move the bowl to the mat | 1 | na | 5 | 5 | 345-390 |
| 5 | release the bowl on the mat | 2 | set-down | 5 | 4 | 390-435 |
| 6 | grasp the bowl with the egg | 3 | rim-pinch | 5 | 2 | 435-729 |
| 7 | move the bowl with the egg to the pan | 1 | na | 5 | 5 | 729-798 |
| 8 | pour the egg into the pan | 2 | tilt-pour | 5 | 4 | 798-867 |
| 9 | move the bowl to the mat | 1 | na | 5 | 5 | 867-924 |
| 10 | release the bowl on the mat | 2 | set-down | 5 | 4 | 924-963 |
| 11 | grasp the bowl with the noodles | 3 | rim-pinch | 5 | 2 | 963-1251 |
| 12 | move the bowl with the noodles to the pan | 1 | na | 5 | 5 | 1251-1320 |
| 13 | pour the noodles into the pan | 2 | tilt-pour | 5 | 4 | 1320-1431 |
| 14 | move the bowl to the mat | 1 | na | 5 | 5 | 1431-1473 |
| 15 | release the bowl on the mat | 2 | set-down | 5 | 4 | 1473-1518 |
| 16 | return to home | 1 | na | 5 | 5 | 1518-1593 |

```yaml
task: Pour the ham, the egg and the noodles into the pan and put each bowl back on the mat
eval_subtasks:
- grasp the bowl with the ham
- move the bowl with the ham to the pan
- pour the ham into the pan
- move the bowl to the mat
- release the bowl on the mat
- grasp the bowl with the egg
- move the bowl with the egg to the pan
- pour the egg into the pan
- move the bowl to the mat
- release the bowl on the mat
- grasp the bowl with the noodles
- move the bowl with the noodles to the pan
- pour the noodles into the pan
- move the bowl to the mat
- release the bowl on the mat
- return to home
eval_subtask_precisions: [3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1]
eval_subtask_contacts: [rim-pinch, na, tilt-pour, na, set-down, rim-pinch, na, tilt-pour, na, set-down, rim-pinch, na, tilt-pour, na, set-down, na]
```

<details><summary>Other episodes (1)</summary>


Episode 30 (1704 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bowl with the ham | 3 | rim-pinch | 5 | 5 | 0-192 |
| 2 | move the bowl with the ham to the pan | 1 | na | 5 | 5 | 192-258 |
| 3 | pour the ham into the pan | 2 | tilt-pour | 5 | 4 | 258-330 |
| 4 | move the bowl to the mat | 1 | na | 5 | 5 | 330-390 |
| 5 | release the bowl on the mat | 2 | set-down | 4 | 4 | 390-456 |
| 6 | grasp the bowl with the egg | 3 | rim-pinch | 5 | 2 | 456-750 |
| 7 | move the bowl with the egg to the pan | 1 | na | 5 | 5 | 750-819 |
| 8 | pour the egg into the pan | 2 | tilt-pour | 5 | 4 | 819-891 |
| 9 | move the bowl to the mat | 1 | na | 5 | 5 | 891-963 |
| 10 | release the bowl on the mat | 2 | set-down | 4 | 4 | 963-1047 |
| 11 | grasp the bowl with the noodles | 3 | rim-pinch | 5 | 2 | 1047-1346 |
| 12 | move the bowl with the noodles to the pan | 1 | na | 5 | 5 | 1346-1419 |
| 13 | pour the noodles into the pan | 2 | tilt-pour | 5 | 4 | 1419-1509 |
| 14 | move the bowl to the mat | 1 | na | 5 | 5 | 1509-1572 |
| 15 | release the bowl on the mat | 2 | set-down | 4 | 4 | 1572-1653 |
| 16 | return to home | 1 | na | 5 | 5 | 1653-1704 |

</details>

### T139 Pour the noodles into the pan and put the bowl back on the mat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 25, 26, 31, 32. Table: episode 25 (570 frames, mean quality 4.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bowl with the noodles | 3 | rim-pinch | 5 | 5 | 0-189 |
| 2 | move the bowl with the noodles to the pan | 1 | na | 5 | 5 | 189-240 |
| 3 | pour the noodles into the pan | 2 | tilt-pour | 5 | 5 | 240-340 |
| 4 | release the bowl on the mat | 2 | set-down | 5 | 4 | 340-455 |
| 5 | return to home | 1 | na | 4 | 4 | 455-570 |

```yaml
task: Pour the noodles into the pan and put the bowl back on the mat
eval_subtasks:
- grasp the bowl with the noodles
- move the bowl with the noodles to the pan
- pour the noodles into the pan
- release the bowl on the mat
- return to home
eval_subtask_precisions: [3, 1, 2, 2, 1]
eval_subtask_contacts: [rim-pinch, na, tilt-pour, set-down, na]
```

<details><summary>Other episodes (3)</summary>


Episode 31 (483 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bowl | 3 | rim-pinch | 4 | 2 | 0-230 |
| 2 | move the bowl to the pan | 1 | na | 5 | 5 | 230-276 |
| 3 | pour the noodles into the pan | 2 | tilt-pour | 5 | 4 | 276-336 |
| 4 | move the bowl to the mat | 1 | na | 5 | 5 | 336-400 |
| 5 | release the bowl on the mat | 1 | set-down | 5 | 5 | 400-436 |
| 6 | return to home | 1 | na | 5 | 5 | 436-483 |


Episode 26 (633 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bowl with the noodles | 3 | rim-pinch | 4 | 5 | 0-189 |
| 2 | move the bowl with the noodles to the pan | 1 | na | 5 | 4 | 189-270 |
| 3 | pour the noodles into the pan | 2 | tilt-pour | 5 | 4 | 270-381 |
| 4 | move the bowl to the mat | 2 | na | 5 | 4 | 381-459 |
| 5 | release the bowl on the mat | 3 | set-down | 4 | 4 | 459-507 |
| 6 | return to home | 1 | na | 5 | 4 | 507-633 |


Episode 32 (522 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bowl | 3 | rim-pinch | 5 | 1 | 0-210 |
| 2 | move the bowl to the pan | 1 | na | 5 | 5 | 210-267 |
| 3 | pour the noodles into the pan | 2 | tilt-pour | 5 | 4 | 267-333 |
| 4 | move the bowl to the mat | 1 | na | 5 | 4 | 333-393 |
| 5 | release the bowl on the mat | 1 | set-down | 5 | 5 | 393-465 |
| 6 | return to home | 1 | na | 5 | 5 | 465-522 |

</details>

### T140 Pour the oil into the pan and put the bowl back on the mat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 27, 28. Table: episode 27 (579 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bowl with the oil | 3 | rim-pinch | 4 | 5 | 0-257 |
| 2 | move the bowl with the oil to the pan | 1 | na | 5 | 5 | 257-330 |
| 3 | pour the oil into the pan | 2 | tilt-pour | 5 | 5 | 330-462 |
| 4 | move the bowl to the mat | 1 | na | 5 | 5 | 462-500 |
| 5 | release the bowl on the mat | 2 | set-down | 5 | 5 | 500-549 |
| 6 | return to home | 1 | na | 5 | 5 | 549-579 |

```yaml
task: Pour the oil into the pan and put the bowl back on the mat
eval_subtasks:
- grasp the bowl with the oil
- move the bowl with the oil to the pan
- pour the oil into the pan
- move the bowl to the mat
- release the bowl on the mat
- return to home
eval_subtask_precisions: [3, 1, 2, 1, 2, 1]
eval_subtask_contacts: [rim-pinch, na, tilt-pour, na, set-down, na]
```

<details><summary>Other episodes (1)</summary>


Episode 28 (537 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bowl with the oil | 3 | rim-pinch | 4 | 5 | 0-207 |
| 2 | move the bowl with the oil to the pan | 1 | na | 5 | 5 | 207-282 |
| 3 | pour the oil into the pan | 2 | tilt-pour | 5 | 5 | 282-390 |
| 4 | move the bowl to the mat | 1 | na | 5 | 5 | 390-438 |
| 5 | release the bowl on the mat | 2 | set-down | 5 | 5 | 438-504 |
| 6 | return to home | 1 | na | 5 | 5 | 504-537 |

</details>

### T141 Push the T block onto the black T

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 107, 108, 109. Table: episode 107 (801 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the T block onto the black T | 4 | push | 4 | 3 | 0-801 |

```yaml
task: Push the T block onto the black T
eval_subtasks:
- push the T block onto the black T
eval_subtask_precisions: [4]
eval_subtask_contacts: [push]
```

<details><summary>Other episodes (2)</summary>


Episode 108 (849 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the T block onto the black T | 4 | push | 4 | 3 | 0-849 |


Episode 109 (900 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the T block onto the black T | 4 | push | 4 | 3 | 0-900 |

</details>

### T142 Push the T block onto the green T

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 100, 101, 102, 103, 104, 105, 106. Table: episode 100 (597 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the T block onto the green T | 4 | push | 4 | 3 | 0-597 |

```yaml
task: Push the T block onto the green T
eval_subtasks:
- push the T block onto the green T
eval_subtask_precisions: [4]
eval_subtask_contacts: [push]
```

<details><summary>Other episodes (6)</summary>


Episode 102 (699 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the T block onto the green T | 4 | push | 4 | 3 | 0-699 |


Episode 104 (516 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the T block onto the green T | 4 | push | 5 | 3 | 0-516 |


Episode 105 (471 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the T block onto the green T | 4 | push | 4 | 3 | 0-471 |


Episode 106 (741 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the T block onto the green T | 4 | push | 4 | 3 | 0-741 |


Episode 101 (693 frames, mean quality 2.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the T block onto the green T | 4 | push | 4 | 2 | 0-693 |


Episode 103 (789 frames, mean quality 2.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | push the T block onto the green T | 4 | push | 4 | 2 | 0-789 |

</details>

### T143 Put a fig bar in each tray, then an apple in each tray

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 184, 185, 186, 187, 188, 189, 190, 191, 192, 193. Table: episode 192 (1269 frames, mean quality 4.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the fig bar on the robot's left | 3 | top-pinch | 5 | 3 | 0-204 |
| 2 | move the fig bar to the tray on the robot's left | 1 | na | 5 | 5 | 204-258 |
| 3 | release the fig bar in the tray on the robot's left | 2 | set-down | 4 | 5 | 258-300 |
| 4 | grasp the fig bar | 3 | top-pinch | 3 | 3 | 300-435 |
| 5 | move the fig bar to the tray on the robot's left | 1 | na | 5 | 4 | 435-492 |
| 6 | move the fig bar to the tray on the robot's right | 1 | na | 5 | 4 | 492-576 |
| 7 | release the fig bar in the tray on the robot's right | 2 | set-down | 3 | 5 | 576-621 |
| 8 | grasp the apple on the robot's left | 3 | top-pinch | 4 | 5 | 621-726 |
| 9 | move the apple to the tray on the robot's left | 1 | na | 5 | 5 | 726-786 |
| 10 | release the apple in the tray on the robot's left | 2 | set-down | 4 | 4 | 786-822 |
| 11 | grasp the apple | 3 | top-pinch | 3 | 5 | 822-975 |
| 12 | move the apple to the tray on the robot's right | 1 | na | 4 | 5 | 975-1038 |
| 13 | release the apple in the tray on the robot's right | 2 | set-down | 3 | 4 | 1038-1122 |
| 14 | return to home | 1 | na | 5 | 4 | 1122-1269 |

```yaml
task: Put a fig bar in each tray, then an apple in each tray
eval_subtasks:
- grasp the fig bar on the robot's left
- move the fig bar to the tray on the robot's left
- release the fig bar in the tray on the robot's left
- grasp the fig bar
- move the fig bar to the tray on the robot's left
- move the fig bar to the tray on the robot's right
- release the fig bar in the tray on the robot's right
- grasp the apple on the robot's left
- move the apple to the tray on the robot's left
- release the apple in the tray on the robot's left
- grasp the apple
- move the apple to the tray on the robot's right
- release the apple in the tray on the robot's right
- return to home
eval_subtask_precisions: [3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, set-down, top-pinch, na, na, set-down, top-pinch, na, set-down, top-pinch, na, set-down, na]
```

<details><summary>Other episodes (9)</summary>


Episode 184 (1488 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the fig bar on the robot's left | 3 | top-pinch | 4 | 3 | 0-384 |
| 2 | move the fig bar to the tray on the robot's left | 1 | na | 5 | 5 | 384-450 |
| 3 | release the fig bar in the tray on the robot's left | 2 | set-down | 5 | 5 | 450-510 |
| 4 | grasp the fig bar | 3 | top-pinch | 3 | 3 | 510-666 |
| 5 | move the fig bar to the tray on the robot's right | 1 | na | 5 | 5 | 666-756 |
| 6 | release the fig bar in the tray on the robot's right | 2 | set-down | 3 | 4 | 756-822 |
| 7 | grasp the apple on the robot's left | 3 | top-pinch | 4 | 5 | 822-996 |
| 8 | move the apple to the tray on the robot's left | 1 | na | 5 | 5 | 996-1050 |
| 9 | release the apple in the tray on the robot's left | 2 | set-down | 3 | 4 | 1050-1116 |
| 10 | grasp the apple | 3 | top-pinch | 4 | 5 | 1116-1272 |
| 11 | move the apple to the tray on the robot's right | 1 | na | 5 | 5 | 1272-1308 |
| 12 | release the apple in the tray on the robot's right | 2 | set-down | 4 | 4 | 1308-1368 |
| 13 | return to home | 1 | na | 5 | 3 | 1368-1488 |


Episode 190 (1581 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the fig bar on the robot's left | 3 | top-pinch | 4 | 3 | 0-252 |
| 2 | move the fig bar to the tray on the robot's left | 1 | na | 3 | 4 | 252-387 |
| 3 | release the fig bar in the tray on the robot's left | 2 | set-down | 3 | 5 | 387-429 |
| 4 | grasp the fig bar | 3 | top-pinch | 3 | 3 | 429-603 |
| 5 | move the fig bar to the tray on the robot's right | 1 | na | 5 | 4 | 603-714 |
| 6 | release the fig bar in the tray on the robot's right | 2 | set-down | 5 | 5 | 714-768 |
| 7 | grasp the apple on the robot's left | 3 | top-pinch | 5 | 5 | 768-906 |
| 8 | move the apple to the tray on the robot's left | 1 | na | 5 | 5 | 906-990 |
| 9 | release the apple in the tray on the robot's left | 2 | drop | 3 | 4 | 990-1047 |
| 10 | grasp the apple | 3 | top-pinch | 4 | 5 | 1047-1248 |
| 11 | move the apple to the tray on the robot's right | 1 | na | 5 | 5 | 1248-1326 |
| 12 | release the apple in the tray on the robot's right | 2 | drop | 3 | 4 | 1326-1374 |
| 13 | return to home | 1 | na | 5 | 4 | 1374-1581 |


Episode 193 (1164 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the fig bar on the robot's left | 3 | cloth-pinch | 3 | 1 | 0-234 |
| 2 | move the fig bar to the tray on the robot's left | 1 | na | 5 | 5 | 234-294 |
| 3 | release the fig bar in the tray on the robot's left | 2 | drop | 3 | 5 | 294-327 |
| 4 | grasp the fig bar | 3 | cloth-pinch | 3 | 3 | 327-444 |
| 5 | move the fig bar to the tray on the robot's right | 1 | na | 5 | 5 | 444-519 |
| 6 | release the fig bar in the tray on the robot's right | 2 | set-down | 5 | 5 | 519-555 |
| 7 | grasp the apple on the robot's left | 3 | top-pinch | 5 | 5 | 555-654 |
| 8 | move the apple to the tray on the robot's left | 1 | na | 5 | 5 | 654-708 |
| 9 | release the apple in the tray on the robot's left | 2 | set-down | 5 | 4 | 708-744 |
| 10 | grasp the apple | 3 | top-pinch | 4 | 5 | 744-882 |
| 11 | move the apple to the tray on the robot's right | 1 | na | 5 | 5 | 882-936 |
| 12 | release the apple in the tray on the robot's right | 2 | set-down | 5 | 4 | 936-1002 |
| 13 | return to home | 1 | na | 5 | 4 | 1002-1164 |


Episode 191 (1467 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the fig bar on the robot's left | 3 | top-pinch | 4 | 1 | 0-294 |
| 2 | move the fig bar to the tray on the robot's left | 1 | na | 4 | 5 | 294-366 |
| 3 | release the fig bar in the tray on the robot's left | 2 | drop | 2 | 5 | 366-420 |
| 4 | grasp the fig bar | 3 | top-pinch | 3 | 3 | 420-579 |
| 5 | move the fig bar to the tray on the robot's right | 1 | na | 5 | 3 | 579-672 |
| 6 | release the fig bar in the tray on the robot's right | 2 | drop | 4 | 5 | 672-741 |
| 7 | grasp the apple on the robot's left | 3 | top-pinch | 3 | 5 | 741-852 |
| 8 | move the apple to the tray on the robot's left | 1 | na | 5 | 5 | 852-936 |
| 9 | release the apple in the tray on the robot's left | 2 | drop | 4 | 4 | 936-978 |
| 10 | grasp the apple | 3 | top-pinch | 4 | 5 | 978-1182 |
| 11 | move the apple to the tray on the robot's right | 1 | na | 5 | 5 | 1182-1242 |
| 12 | release the apple in the tray on the robot's right | 2 | drop | 4 | 4 | 1242-1296 |
| 13 | return to home | 1 | na | 5 | 4 | 1296-1467 |


Episode 187 (1377 frames, mean quality 4.1)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the fig bar on the robot's left | 3 | top-pinch | 4 | 2 | 0-294 |
| 2 | move the fig bar to the tray on the robot's left | 1 | na | 5 | 3 | 294-360 |
| 3 | release the fig bar in the tray on the robot's left | 2 | drop | 3 | 4 | 360-423 |
| 4 | grasp the fig bar | 3 | top-pinch | 4 | 3 | 423-552 |
| 5 | move the fig bar to the tray on the robot's right | 1 | na | 5 | 3 | 552-627 |
| 6 | release the fig bar in the tray on the robot's right | 2 | drop | 4 | 5 | 627-702 |
| 7 | grasp the apple on the robot's left | 3 | top-pinch | 3 | 5 | 702-834 |
| 8 | move the apple to the tray on the robot's left | 1 | na | 5 | 5 | 834-900 |
| 9 | release the apple in the tray on the robot's left | 2 | set-down | 3 | 4 | 900-978 |
| 10 | grasp the apple | 3 | top-pinch | 4 | 5 | 978-1152 |
| 11 | move the apple to the tray on the robot's right | 1 | na | 5 | 5 | 1152-1197 |
| 12 | release the apple in the tray on the robot's right | 2 | set-down | 4 | 4 | 1197-1275 |
| 13 | return to home | 1 | na | 5 | 5 | 1275-1377 |


Episode 185 (1317 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the fig bar on the robot's left | 3 | top-pinch | 3 | 3 | 0-297 |
| 2 | move the fig bar to the tray on the robot's left | 1 | na | 5 | 5 | 297-362 |
| 3 | release the fig bar in the tray on the robot's left | 2 | drop | 4 | 4 | 362-417 |
| 4 | grasp the fig bar | 3 | top-pinch | 5 | 3 | 417-555 |
| 5 | move the fig bar to the tray on the robot's right | 1 | na | 5 | 4 | 555-650 |
| 6 | release the fig bar in the tray on the robot's right | 2 | drop | 4 | 4 | 650-682 |
| 7 | grasp the apple on the robot's left | 3 | top-pinch | 4 | 3 | 682-813 |
| 8 | move the apple to the tray on the robot's left | 1 | na | 5 | 5 | 813-867 |
| 9 | release the apple in the tray on the robot's left | 2 | set-down | 4 | 4 | 867-924 |
| 10 | grasp the apple | 3 | top-pinch | 5 | 5 | 924-1089 |
| 11 | move the apple to the tray on the robot's right | 1 | na | 5 | 5 | 1089-1137 |
| 12 | release the apple in the tray on the robot's right | 2 | drop | 4 | 4 | 1137-1197 |
| 13 | return to home | 1 | na | 5 | 3 | 1197-1317 |


Episode 189 (1500 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the fig bar on the robot's left | 3 | top-pinch | 3 | 1 | 0-264 |
| 2 | move the fig bar to the tray on the robot's left | 1 | na | 4 | 3 | 264-357 |
| 3 | release the fig bar in the tray on the robot's left | 2 | set-down | 4 | 5 | 357-405 |
| 4 | grasp the fig bar | 3 | top-pinch | 3 | 3 | 405-567 |
| 5 | move the fig bar to the tray on the robot's right | 1 | na | 5 | 3 | 567-690 |
| 6 | release the fig bar in the tray on the robot's right | 2 | drop | 5 | 4 | 690-726 |
| 7 | grasp the apple on the robot's left | 3 | top-pinch | 4 | 5 | 726-900 |
| 8 | move the apple to the tray on the robot's left | 1 | na | 5 | 5 | 900-960 |
| 9 | release the apple in the tray on the robot's left | 2 | drop | 2 | 4 | 960-1008 |
| 10 | grasp the apple | 3 | top-pinch | 3 | 5 | 1008-1191 |
| 11 | move the apple to the tray on the robot's right | 1 | na | 5 | 5 | 1191-1263 |
| 12 | release the apple in the tray on the robot's right | 2 | drop | 3 | 5 | 1263-1317 |
| 13 | return to home | 1 | na | 5 | 4 | 1317-1500 |


Episode 188 (1344 frames, mean quality 3.9)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the fig bar on the robot's left | 3 | top-pinch | 5 | 2 | 0-297 |
| 2 | move the fig bar to the tray on the robot's left | 1 | na | 5 | 3 | 297-369 |
| 3 | release the fig bar in the tray on the robot's left | 2 | drop | 4 | 4 | 369-420 |
| 4 | grasp the fig bar | 3 | top-pinch | 4 | 3 | 420-540 |
| 5 | move the fig bar to the tray on the robot's right | 1 | na | 5 | 3 | 540-648 |
| 6 | release the fig bar in the tray on the robot's right | 2 | drop | 3 | 5 | 648-711 |
| 7 | grasp the apple on the robot's left | 3 | top-pinch | 3 | 5 | 711-816 |
| 8 | move the apple to the tray on the robot's left | 1 | na | 5 | 4 | 816-906 |
| 9 | release the apple in the tray on the robot's left | 2 | drop | 4 | 4 | 906-945 |
| 10 | grasp the apple | 3 | top-pinch | 4 | 5 | 945-1089 |
| 11 | move the apple to the tray on the robot's right | 1 | na | 4 | 5 | 1089-1152 |
| 12 | release the apple in the tray on the robot's right | 2 | drop | 2 | 4 | 1152-1203 |
| 13 | return to home | 1 | na | 5 | 4 | 1203-1344 |


Episode 186 (1239 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the fig bar on the robot's left | 3 | top-pinch | 5 | 1 | 0-234 |
| 2 | move the fig bar to the tray on the robot's left | 1 | na | 5 | 5 | 234-278 |
| 3 | release the fig bar in the tray on the robot's left | 2 | drop | 5 | 4 | 278-342 |
| 4 | grasp the fig bar | 3 | top-pinch | 5 | 3 | 342-458 |
| 5 | move the fig bar to the tray on the robot's right | 1 | na | 5 | 4 | 458-532 |
| 6 | release the fig bar in the tray on the robot's right | 2 | drop | 4 | 5 | 532-578 |
| 7 | grasp the apple on the robot's left | 3 | top-pinch | 4 | 3 | 578-744 |
| 8 | move the apple to the tray on the robot's left | 1 | na | 5 | 3 | 744-778 |
| 9 | release the apple in the tray on the robot's left | 2 | drop | 5 | 3 | 778-802 |
| 10 | grasp the apple | 3 | top-pinch | 4 | 3 | 802-990 |
| 11 | move the apple to the tray on the robot's right | 1 | na | 5 | 5 | 990-1036 |
| 12 | release the apple in the tray on the robot's right | 2 | drop | 5 | 4 | 1036-1096 |
| 13 | return to home | 1 | na | 5 | 3 | 1096-1239 |

</details>

### T144 Put the T block on the green T

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 110, 111, 112, 113, 114, 115, 116, 117, 118, 119. Table: episode 112 (276 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the T block | 3 | top-pinch | 5 | 5 | 0-141 |
| 2 | move the T block to the green T | 3 | na | 5 | 5 | 141-171 |
| 3 | release the T block on the green T | 4 | set-down | 5 | 5 | 171-225 |
| 4 | return to home | 1 | na | 5 | 5 | 225-276 |

```yaml
task: Put the T block on the green T
eval_subtasks:
- grasp the T block
- move the T block to the green T
- release the T block on the green T
- return to home
eval_subtask_precisions: [3, 3, 4, 1]
eval_subtask_contacts: [top-pinch, na, set-down, na]
```

<details><summary>Other episodes (9)</summary>


Episode 110 (279 frames, mean quality 4.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the T block | 3 | top-pinch | 5 | 5 | 0-138 |
| 2 | move the T block to the green T | 3 | na | 5 | 5 | 138-178 |
| 3 | release the T block on the green T | 4 | set-down | 3 | 4 | 178-279 |


Episode 116 (297 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the T block | 3 | top-pinch | 5 | 5 | 0-174 |
| 2 | move the T block to the green T | 3 | na | 5 | 5 | 174-224 |
| 3 | release the T block on the green T | 4 | set-down | 4 | 4 | 224-258 |
| 4 | return to home | 1 | na | 5 | 4 | 258-297 |


Episode 118 (273 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the T block | 3 | top-pinch | 5 | 4 | 0-165 |
| 2 | move the T block to the green T | 3 | na | 5 | 5 | 165-195 |
| 3 | release the T block on the green T | 4 | set-down | 5 | 4 | 195-249 |
| 4 | return to home | 1 | na | 5 | 4 | 249-273 |


Episode 113 (243 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the T block | 3 | top-pinch | 5 | 3 | 0-123 |
| 2 | move the T block to the green T | 3 | na | 5 | 3 | 123-168 |
| 3 | release the T block on the green T | 4 | set-down | 5 | 5 | 168-213 |
| 4 | return to home | 1 | na | 5 | 5 | 213-243 |


Episode 114 (237 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the T block | 3 | top-pinch | 5 | 5 | 0-138 |
| 2 | move the T block to the green T | 3 | na | 5 | 5 | 138-171 |
| 3 | release the T block on the green T | 4 | set-down | 5 | 3 | 171-207 |
| 4 | return to home | 1 | na | 5 | 3 | 207-237 |


Episode 119 (252 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the T block | 3 | top-pinch | 5 | 3 | 0-156 |
| 2 | move the T block to the green T | 3 | na | 5 | 5 | 156-192 |
| 3 | release the T block on the green T | 4 | set-down | 5 | 4 | 192-222 |
| 4 | return to home | 1 | na | 5 | 4 | 222-252 |


Episode 117 (294 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the T block | 3 | top-pinch | 5 | 3 | 0-168 |
| 2 | move the T block to the green T | 3 | na | 5 | 4 | 168-207 |
| 3 | release the T block on the green T | 4 | set-down | 5 | 4 | 207-249 |
| 4 | return to home | 1 | na | 5 | 4 | 249-294 |


Episode 111 (429 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the T block | 3 | top-pinch | 4 | 2 | 0-315 |
| 2 | move the T block to the green T | 3 | na | 5 | 5 | 315-350 |
| 3 | release the T block on the green T | 4 | set-down | 4 | 4 | 350-429 |


Episode 115 (312 frames, mean quality 3.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the T block | 3 | top-pinch | 5 | 3 | 0-200 |
| 2 | move the T block to the green T | 3 | na | 5 | 4 | 200-252 |
| 3 | release the T block on the green T | 4 | set-down | 5 | 3 | 252-312 |

</details>

### T145 Put the black disk on the red plate

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 61, 62, 64, 68. Table: episode 61 (771 frames, mean quality 4.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black disk in the middle of the grey plate | 4 | top-pinch | 2 | 3 | 0-417 |
| 2 | move the black disk to the red plate | 1 | na | 5 | 5 | 417-522 |
| 3 | release the black disk on the red plate | 2 | set-down | 3 | 5 | 522-597 |
| 4 | return to home | 1 | na | 5 | 5 | 597-771 |

```yaml
task: Put the black disk on the red plate
eval_subtasks:
- grasp the black disk in the middle of the grey plate
- move the black disk to the red plate
- release the black disk on the red plate
- return to home
eval_subtask_precisions: [4, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, set-down, na]
```

<details><summary>Other episodes (3)</summary>


Episode 64 (867 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black disk | 3 | top-pinch | 2 | 3 | 0-483 |
| 2 | move the black disk to the red plate | 1 | na | 5 | 4 | 483-609 |
| 3 | release the black disk on the red plate | 2 | drop | 4 | 5 | 609-700 |
| 4 | return to home | 1 | na | 5 | 5 | 700-867 |


Episode 62 (819 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black disk next to the wooden disks | 3 | top-pinch | 2 | 1 | 0-471 |
| 2 | move the black disk to the red plate | 1 | na | 5 | 5 | 471-585 |
| 3 | release the black disk on the red plate | 2 | set-down | 3 | 5 | 585-675 |
| 4 | return to home | 1 | na | 5 | 5 | 675-819 |


Episode 68 (1038 frames, mean quality 2.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the black disk | 3 | top-pinch | 2 | 1 | 0-714 |
| 2 | move the black disk to the red plate | 1 | na | 5 | 4 | 714-846 |
| 3 | release the black disk on the red plate | 2 | drop | 5 | 3 | 846-921 |
| 4 | return to home | 1 | na | 5 | 3 | 921-1038 |

</details>

### T146 Put the blue cube in the tray

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 145, 146. Table: episode 145 (150 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue cube | 3 | top-pinch | 5 | 5 | 0-69 |
| 2 | release the blue cube in the tray | 2 | drop | 5 | 5 | 69-108 |
| 3 | return to home | 1 | na | 5 | 5 | 108-150 |

```yaml
task: Put the blue cube in the tray
eval_subtasks:
- grasp the blue cube
- release the blue cube in the tray
- return to home
eval_subtask_precisions: [3, 2, 1]
eval_subtask_contacts: [top-pinch, drop, na]
```

<details><summary>Other episodes (1)</summary>


Episode 146 (171 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue cube | 3 | top-pinch | 5 | 5 | 0-87 |
| 2 | release the blue cube in the tray | 2 | drop | 5 | 5 | 87-130 |
| 3 | return to home | 1 | na | 5 | 5 | 130-171 |

</details>

### T147 Put the blue piece, the yellow piece and the orange piece on the peg

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 81, 82, 83, 84, 85, 86, 87, 88, 89. Table: episode 83 (864 frames, mean quality 4.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue piece | 3 | top-pinch | 4 | 5 | 0-186 |
| 2 | move the blue piece to the peg | 3 | na | 5 | 5 | 186-216 |
| 3 | release the blue piece on the peg | 4 | insert | 2 | 5 | 216-327 |
| 4 | grasp the yellow piece | 3 | top-pinch | 5 | 5 | 327-441 |
| 5 | move the yellow piece to the peg | 3 | na | 5 | 4 | 441-471 |
| 6 | release the yellow piece on the peg | 4 | insert | 2 | 3 | 471-582 |
| 7 | grasp the orange piece | 3 | top-pinch | 5 | 5 | 582-669 |
| 8 | move the orange piece to the peg | 3 | na | 5 | 5 | 669-720 |
| 9 | release the orange piece on the peg | 4 | insert | 3 | 3 | 720-825 |
| 10 | return to home | 1 | na | 5 | 5 | 825-864 |

```yaml
task: Put the blue piece, the yellow piece and the orange piece on the peg
eval_subtasks:
- grasp the blue piece
- move the blue piece to the peg
- release the blue piece on the peg
- grasp the yellow piece
- move the yellow piece to the peg
- release the yellow piece on the peg
- grasp the orange piece
- move the orange piece to the peg
- release the orange piece on the peg
- return to home
eval_subtask_precisions: [3, 3, 4, 3, 3, 4, 3, 3, 4, 1]
eval_subtask_contacts: [top-pinch, na, insert, top-pinch, na, insert, top-pinch, na, insert, na]
```

<details><summary>Other episodes (8)</summary>


Episode 85 (657 frames, mean quality 4.4)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue piece | 3 | top-pinch | 3 | 4 | 0-154 |
| 2 | move the blue piece to the peg | 3 | na | 5 | 5 | 154-186 |
| 3 | release the blue piece on the peg | 4 | insert | 3 | 5 | 186-252 |
| 4 | grasp the yellow piece | 3 | top-pinch | 5 | 5 | 252-329 |
| 5 | move the yellow piece to the peg | 3 | na | 5 | 4 | 329-357 |
| 6 | release the yellow piece on the peg | 4 | insert | 2 | 4 | 357-430 |
| 7 | grasp the orange piece | 3 | top-pinch | 4 | 3 | 430-538 |
| 8 | move the orange piece to the peg | 3 | na | 5 | 4 | 538-576 |
| 9 | release the orange piece on the peg | 4 | insert | 4 | 5 | 576-618 |
| 10 | return to home | 1 | na | 5 | 5 | 618-657 |


Episode 86 (846 frames, mean quality 4.4)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue piece | 3 | top-pinch | 3 | 3 | 0-198 |
| 2 | move the blue piece to the peg | 3 | na | 5 | 5 | 198-237 |
| 3 | release the blue piece on the peg | 4 | insert | 1 | 3 | 237-342 |
| 4 | grasp the yellow piece | 3 | top-pinch | 5 | 5 | 342-429 |
| 5 | move the yellow piece to the peg | 3 | na | 5 | 4 | 429-483 |
| 6 | release the yellow piece on the peg | 4 | insert | 2 | 5 | 483-561 |
| 7 | grasp the orange piece | 3 | top-pinch | 4 | 5 | 561-690 |
| 8 | move the orange piece to the peg | 3 | na | 5 | 5 | 690-741 |
| 9 | release the orange piece on the peg | 4 | insert | 2 | 4 | 741-813 |
| 10 | return to home | 1 | na | 5 | 5 | 813-846 |


Episode 89 (795 frames, mean quality 4.4)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue piece | 3 | top-pinch | 5 | 5 | 0-116 |
| 2 | move the blue piece to the peg | 3 | na | 5 | 5 | 116-146 |
| 3 | release the blue piece on the peg | 4 | insert | 3 | 5 | 146-222 |
| 4 | grasp the yellow piece | 3 | top-pinch | 4 | 3 | 222-375 |
| 5 | move the yellow piece to the peg | 3 | na | 5 | 5 | 375-412 |
| 6 | release the yellow piece on the peg | 4 | insert | 2 | 4 | 412-505 |
| 7 | grasp the orange piece | 3 | top-pinch | 1 | 3 | 505-663 |
| 8 | move the orange piece to the peg | 3 | na | 5 | 5 | 663-700 |
| 9 | release the orange piece on the peg | 4 | insert | 3 | 4 | 700-755 |
| 10 | return to home | 1 | na | 5 | 5 | 755-795 |


Episode 88 (777 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue piece | 3 | top-pinch | 5 | 5 | 0-82 |
| 2 | move the blue piece to the peg | 3 | na | 5 | 5 | 82-126 |
| 3 | release the blue piece on the peg | 4 | insert | 1 | 3 | 126-276 |
| 4 | grasp the yellow piece | 3 | top-pinch | 4 | 2 | 276-452 |
| 5 | move the yellow piece to the peg | 3 | na | 5 | 5 | 452-484 |
| 6 | release the yellow piece on the peg | 4 | insert | 3 | 5 | 484-548 |
| 7 | grasp the orange piece | 3 | top-pinch | 5 | 5 | 548-654 |
| 8 | move the orange piece to the peg | 3 | na | 5 | 4 | 654-687 |
| 9 | release the orange piece on the peg | 4 | insert | 3 | 4 | 687-741 |
| 10 | return to home | 1 | na | 5 | 5 | 741-777 |


Episode 87 (1179 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue piece | 3 | top-pinch | 4 | 5 | 0-150 |
| 2 | move the blue piece to the peg | 3 | na | 5 | 5 | 150-195 |
| 3 | release the blue piece on the peg | 4 | insert | 2 | 5 | 195-288 |
| 4 | grasp the yellow piece | 3 | top-pinch | 3 | 5 | 288-411 |
| 5 | move the yellow piece to the peg | 3 | na | 5 | 4 | 411-459 |
| 6 | release the yellow piece on the peg | 4 | insert | 3 | 2 | 459-510 |
| 7 | grasp the yellow piece | 3 | side-pinch | 4 | 2 | 510-669 |
| 8 | move the yellow piece to the peg | 3 | na | 5 | 4 | 669-741 |
| 9 | release the yellow piece on the peg | 4 | insert | 3 | 3 | 741-882 |
| 10 | grasp the orange piece | 3 | top-pinch | 3 | 3 | 882-1051 |
| 11 | release the orange piece on the peg | 4 | insert | 4 | 5 | 1051-1155 |
| 12 | return to home | 1 | na | 5 | 5 | 1155-1179 |


Episode 84 (1134 frames, mean quality 3.9)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue piece | 3 | top-pinch | 3 | 4 | 0-156 |
| 2 | move the blue piece to the peg | 3 | na | 5 | 5 | 156-189 |
| 3 | release the blue piece on the peg | 4 | insert | 4 | 5 | 189-243 |
| 4 | grasp the yellow piece | 3 | top-pinch | 5 | 1 | 243-648 |
| 5 | move the yellow piece to the peg | 3 | na | 5 | 4 | 648-699 |
| 6 | release the yellow piece on the peg | 4 | insert | 3 | 4 | 699-801 |
| 7 | grasp the orange piece | 3 | top-pinch | 2 | 3 | 801-972 |
| 8 | move the orange piece to the peg | 3 | na | 5 | 4 | 972-1002 |
| 9 | release the orange piece on the peg | 4 | insert | 3 | 4 | 1002-1095 |
| 10 | return to home | 1 | na | 5 | 5 | 1095-1134 |


Episode 81 (1482 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue piece | 3 | top-pinch | 4 | 1 | 0-315 |
| 2 | move the blue piece to the peg | 3 | na | 4 | 3 | 315-522 |
| 3 | release the blue piece on the peg | 4 | insert | 2 | 5 | 522-675 |
| 4 | grasp the yellow piece | 3 | top-pinch | 3 | 5 | 675-806 |
| 5 | move the yellow piece to the peg | 3 | na | 5 | 5 | 806-843 |
| 6 | release the yellow piece on the peg | 4 | insert | 1 | 3 | 843-1017 |
| 7 | grasp the orange piece | 3 | top-pinch | 1 | 3 | 1017-1278 |
| 8 | move the orange piece to the peg | 3 | na | 5 | 4 | 1278-1329 |
| 9 | release the orange piece on the peg | 4 | insert | 2 | 3 | 1329-1398 |
| 10 | return to home | 1 | na | 5 | 5 | 1398-1482 |


Episode 82 (2193 frames, mean quality 3.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the blue piece | 3 | top-pinch | 5 | 5 | 0-234 |
| 2 | move the blue piece to the peg | 3 | na | 5 | 5 | 234-300 |
| 3 | release the blue piece on the peg | 4 | insert | 2 | 3 | 300-450 |
| 4 | grasp the yellow piece | 3 | top-pinch | 5 | 5 | 450-544 |
| 5 | move the yellow piece to the peg | 3 | na | 5 | 4 | 544-585 |
| 6 | release the yellow piece on the peg | 4 | insert | 2 | 2 | 585-843 |
| 7 | release the yellow piece on the table | 1 | drop | 3 | 4 | 843-908 |
| 8 | grasp the yellow piece | 3 | top-pinch | 3 | 2 | 908-1202 |
| 9 | move the yellow piece to the peg | 3 | na | 5 | 5 | 1202-1250 |
| 10 | release the yellow piece on the peg | 4 | insert | 2 | 4 | 1250-1358 |
| 11 | grasp the orange piece | 3 | top-pinch | 2 | 1 | 1358-1552 |
| 12 | move the orange piece to the peg | 3 | na | 5 | 1 | 1552-1600 |
| 13 | release the orange piece on the peg | 4 | insert | 4 | 1 | 1600-1640 |
| 14 | grasp the orange piece | 3 | side-pinch | 5 | 1 | 1640-1933 |
| 15 | move the orange piece to the peg | 3 | na | 5 | 4 | 1933-2012 |
| 16 | release the orange piece on the peg | 4 | insert | 3 | 3 | 2012-2109 |
| 17 | return to home | 1 | na | 5 | 5 | 2109-2193 |

</details>

### T148 Put the bolt on the green pad

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 33, 34, 35, 36, 37, 38, 39, 40, 41, 42. Table: episode 33 (336 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bolt | 4 | top-pinch | 5 | 5 | 0-127 |
| 2 | move the bolt to the green pad | 1 | na | 5 | 5 | 127-175 |
| 3 | release the bolt on the green pad | 2 | set-down | 4 | 5 | 175-216 |
| 4 | return to home | 1 | na | 5 | 5 | 216-336 |

```yaml
task: Put the bolt on the green pad
eval_subtasks:
- grasp the bolt
- move the bolt to the green pad
- release the bolt on the green pad
- return to home
eval_subtask_precisions: [4, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, set-down, na]
```

<details><summary>Other episodes (9)</summary>


Episode 34 (327 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bolt | 4 | top-pinch | 5 | 5 | 0-125 |
| 2 | move the bolt to the green pad | 1 | na | 5 | 5 | 125-174 |
| 3 | release the bolt on the green pad | 2 | set-down | 5 | 5 | 174-200 |
| 4 | return to home | 1 | na | 5 | 5 | 200-327 |


Episode 35 (294 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bolt | 4 | top-pinch | 5 | 5 | 0-123 |
| 2 | move the bolt to the green pad | 1 | na | 5 | 5 | 123-164 |
| 3 | release the bolt on the green pad | 2 | set-down | 5 | 5 | 164-198 |
| 4 | return to home | 1 | na | 5 | 5 | 198-294 |


Episode 36 (342 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bolt | 4 | top-pinch | 5 | 5 | 0-120 |
| 2 | move the bolt to the green pad | 1 | na | 5 | 5 | 120-168 |
| 3 | release the bolt on the green pad | 2 | set-down | 4 | 5 | 168-214 |
| 4 | return to home | 1 | na | 5 | 5 | 214-342 |


Episode 37 (321 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bolt | 4 | top-pinch | 5 | 5 | 0-132 |
| 2 | move the bolt to the green pad | 1 | na | 5 | 5 | 132-177 |
| 3 | release the bolt on the green pad | 2 | set-down | 5 | 5 | 177-209 |
| 4 | return to home | 1 | na | 5 | 5 | 209-321 |


Episode 39 (297 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bolt | 4 | top-pinch | 5 | 5 | 0-133 |
| 2 | move the bolt to the green pad | 1 | na | 5 | 5 | 133-186 |
| 3 | release the bolt on the green pad | 2 | set-down | 5 | 5 | 186-205 |
| 4 | return to home | 1 | na | 5 | 5 | 205-297 |


Episode 40 (297 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bolt | 4 | top-pinch | 5 | 5 | 0-114 |
| 2 | move the bolt to the green pad | 1 | na | 5 | 5 | 114-168 |
| 3 | release the bolt on the green pad | 2 | set-down | 3 | 5 | 168-201 |
| 4 | return to home | 1 | na | 5 | 5 | 201-297 |


Episode 42 (270 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bolt | 4 | top-pinch | 5 | 5 | 0-117 |
| 2 | move the bolt to the green pad | 1 | na | 5 | 5 | 117-159 |
| 3 | release the bolt on the green pad | 2 | set-down | 5 | 5 | 159-189 |
| 4 | return to home | 1 | na | 5 | 5 | 189-270 |


Episode 41 (294 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bolt | 4 | top-pinch | 4 | 5 | 0-135 |
| 2 | move the bolt to the green pad | 1 | na | 5 | 5 | 135-165 |
| 3 | release the bolt on the green pad | 2 | set-down | 5 | 4 | 165-195 |
| 4 | return to home | 1 | na | 5 | 4 | 195-294 |


Episode 38 (327 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the bolt | 4 | top-pinch | 5 | 4 | 0-126 |
| 2 | move the bolt to the green pad | 1 | na | 5 | 5 | 126-183 |
| 3 | release the bolt on the green pad | 2 | set-down | 4 | 5 | 183-213 |
| 4 | return to home | 1 | na | 5 | 3 | 213-327 |

</details>

### T149 Put the box on the plate

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 13, 14, 15, 16, 17, 18, 19, 20, 21, 22. Table: episode 16 (315 frames, mean quality 4.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the box | 3 | top-pinch | 5 | 4 | 0-192 |
| 2 | move the box to the plate | 1 | na | 4 | 5 | 192-243 |
| 3 | release the box on the plate | 2 | set-down | 3 | 5 | 243-282 |
| 4 | return to home | 1 | na | 5 | 5 | 282-315 |

```yaml
task: Put the box on the plate
eval_subtasks:
- grasp the box
- move the box to the plate
- release the box on the plate
- return to home
eval_subtask_precisions: [3, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, set-down, na]
```

<details><summary>Other episodes (9)</summary>


Episode 18 (360 frames, mean quality 4.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the box | 3 | side-pinch | 5 | 5 | 0-156 |
| 2 | move the box to the plate | 1 | na | 5 | 5 | 156-207 |
| 3 | release the box on the plate | 2 | set-down | 5 | 4 | 207-258 |
| 4 | return to home | 1 | na | 5 | 5 | 258-360 |


Episode 15 (345 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the box | 3 | top-pinch | 5 | 5 | 0-100 |
| 2 | move the box to the plate | 1 | na | 5 | 5 | 100-152 |
| 3 | release the box on the plate | 2 | set-down | 3 | 4 | 152-192 |
| 4 | return to home | 1 | na | 5 | 4 | 192-345 |


Episode 13 (399 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the box | 3 | top-pinch | 4 | 4 | 0-207 |
| 2 | move the box to the plate | 1 | na | 5 | 4 | 207-273 |
| 3 | release the box on the plate | 2 | drop | 5 | 4 | 273-297 |
| 4 | return to home | 1 | na | 5 | 4 | 297-399 |


Episode 22 (612 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the box | 3 | top-pinch | 4 | 1 | 0-273 |
| 2 | move the box to the plate | 1 | na | 5 | 5 | 273-348 |
| 3 | release the box on the plate | 2 | set-down | 4 | 5 | 348-444 |
| 4 | return to home | 1 | na | 5 | 5 | 444-612 |


Episode 14 (447 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the box | 3 | top-pinch | 4 | 1 | 0-282 |
| 2 | move the box to the plate | 1 | na | 5 | 5 | 282-336 |
| 3 | release the box on the plate | 2 | set-down | 4 | 4 | 336-378 |
| 4 | return to home | 1 | na | 5 | 5 | 378-447 |


Episode 21 (825 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the box | 3 | top-pinch | 3 | 2 | 0-636 |
| 2 | move the box to the plate | 1 | na | 4 | 5 | 636-696 |
| 3 | release the box on the plate | 2 | drop | 4 | 4 | 696-717 |
| 4 | return to home | 1 | na | 5 | 4 | 717-825 |


Episode 17 (447 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the box | 3 | top-pinch | 4 | 1 | 0-240 |
| 2 | move the box to the plate | 1 | na | 5 | 5 | 240-296 |
| 3 | release the box on the plate | 2 | set-down | 4 | 4 | 296-337 |
| 4 | return to home | 1 | na | 5 | 4 | 337-447 |


Episode 20 (831 frames, mean quality 3.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the box | 3 | top-pinch | 4 | 2 | 0-456 |
| 2 | move the box to the plate | 1 | na | 4 | 4 | 456-524 |
| 3 | release the box on the plate | 2 | set-down | 4 | 4 | 524-612 |
| 4 | return to home | 1 | na | 5 | 3 | 612-831 |


Episode 19 (744 frames, mean quality 3.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the box | 3 | top-pinch | 3 | 2 | 0-258 |
| 2 | move the box to the plate | 2 | na | 5 | 2 | 258-312 |
| 3 | grasp the box | 3 | top-pinch | 4 | 2 | 312-468 |
| 4 | move the box to the plate | 1 | na | 5 | 5 | 468-540 |
| 5 | release the box on the plate | 2 | drop | 3 | 4 | 540-588 |
| 6 | return to home | 1 | na | 5 | 4 | 588-744 |

</details>

### T150 Put the can in the small taped area

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 5, 6, 7. Table: episode 5 (669 frames, mean quality 4.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the can in the large taped area | 3 | side-pinch | 5 | 5 | 0-210 |
| 2 | move the can to the small taped area | 1 | na | 5 | 5 | 210-315 |
| 3 | release the can in the small taped area | 2 | set-down | 3 | 3 | 315-504 |
| 4 | return to home | 1 | na | 5 | 4 | 504-669 |

```yaml
task: Put the can in the small taped area
eval_subtasks:
- grasp the can in the large taped area
- move the can to the small taped area
- release the can in the small taped area
- return to home
eval_subtask_precisions: [3, 1, 2, 1]
eval_subtask_contacts: [side-pinch, na, set-down, na]
```

<details><summary>Other episodes (2)</summary>


Episode 7 (762 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the can in the large taped area | 3 | top-pinch | 3 | 5 | 0-303 |
| 2 | move the can to the small taped area | 1 | na | 4 | 4 | 303-477 |
| 3 | release the can in the small taped area | 2 | set-down | 2 | 4 | 477-582 |
| 4 | return to home | 1 | na | 5 | 4 | 582-762 |


Episode 6 (627 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the can in the large taped area | 3 | side-pinch | 4 | 1 | 0-290 |
| 2 | move the can to the small taped area | 1 | na | 5 | 5 | 290-405 |
| 3 | release the can in the small taped area | 2 | set-down | 3 | 4 | 405-474 |
| 4 | return to home | 1 | na | 5 | 4 | 474-627 |

</details>

### T151 Put the charger in the bin

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 90, 91, 92, 93, 94, 95, 96, 97, 98, 99. Table: episode 91 (279 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | top-pinch | 5 | 5 | 0-155 |
| 2 | move the charger to the bin | 1 | na | 5 | 5 | 155-201 |
| 3 | release the charger in the bin | 2 | drop | 4 | 5 | 201-279 |

```yaml
task: Put the charger in the bin
eval_subtasks:
- grasp the charger
- move the charger to the bin
- release the charger in the bin
eval_subtask_precisions: [3, 1, 2]
eval_subtask_contacts: [top-pinch, na, drop]
```

<details><summary>Other episodes (9)</summary>


Episode 95 (309 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | top-pinch | 5 | 5 | 0-189 |
| 2 | move the charger to the bin | 1 | na | 5 | 5 | 189-231 |
| 3 | release the charger in the bin | 2 | drop | 4 | 5 | 231-309 |


Episode 96 (288 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | top-pinch | 5 | 5 | 0-177 |
| 2 | move the charger to the bin | 1 | na | 5 | 5 | 177-222 |
| 3 | release the charger in the bin | 2 | drop | 3 | 5 | 222-288 |


Episode 97 (249 frames, mean quality 4.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | top-pinch | 5 | 4 | 0-144 |
| 2 | move the charger to the bin | 1 | na | 5 | 5 | 144-177 |
| 3 | release the charger in the bin | 2 | drop | 5 | 5 | 177-204 |
| 4 | return to home | 1 | na | 5 | 5 | 204-249 |


Episode 92 (273 frames, mean quality 4.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | top-pinch | 5 | 5 | 0-168 |
| 2 | move the charger to the bin | 1 | na | 5 | 5 | 168-216 |
| 3 | release the charger in the bin | 2 | drop | 4 | 4 | 216-273 |


Episode 99 (252 frames, mean quality 4.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | top-pinch | 5 | 5 | 0-138 |
| 2 | move the charger to the bin | 1 | na | 5 | 5 | 138-177 |
| 3 | release the charger in the bin | 2 | drop | 5 | 4 | 177-252 |


Episode 90 (357 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | top-pinch | 5 | 4 | 0-201 |
| 2 | move the charger to the bin | 1 | na | 5 | 5 | 201-255 |
| 3 | release the charger in the bin | 2 | drop | 5 | 4 | 255-357 |


Episode 93 (438 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | top-pinch | 5 | 2 | 0-336 |
| 2 | move the charger to the bin | 1 | na | 5 | 5 | 336-384 |
| 3 | release the charger in the bin | 2 | drop | 4 | 4 | 384-438 |


Episode 94 (393 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | top-pinch | 4 | 2 | 0-261 |
| 2 | move the charger to the bin | 1 | na | 5 | 5 | 261-327 |
| 3 | release the charger in the bin | 2 | drop | 5 | 4 | 327-393 |


Episode 98 (297 frames, mean quality 3.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the charger | 3 | top-pinch | 5 | 2 | 0-204 |
| 2 | release the charger in the bin | 2 | drop | 5 | 4 | 204-270 |
| 3 | return to home | 1 | na | 5 | 4 | 270-297 |

</details>

### T152 Put the crayfish in the box

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 221, 222, 223, 224, 225, 226, 227, 228, 229, 230. Table: episode 223 (837 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the crayfish | 2 | top-pinch | 3 | 5 | 0-387 |
| 2 | move the crayfish to the box | 1 | na | 5 | 5 | 387-525 |
| 3 | release the crayfish in the box | 2 | drop | 3 | 5 | 525-621 |
| 4 | return to home | 1 | na | 5 | 5 | 621-837 |

```yaml
task: Put the crayfish in the box
eval_subtasks:
- grasp the crayfish
- move the crayfish to the box
- release the crayfish in the box
- return to home
eval_subtask_precisions: [2, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (9)</summary>


Episode 227 (720 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the crayfish | 2 | top-pinch | 4 | 5 | 0-309 |
| 2 | move the crayfish to the box | 1 | na | 5 | 5 | 309-414 |
| 3 | release the crayfish in the box | 2 | set-down | 4 | 5 | 414-495 |
| 4 | return to home | 1 | na | 5 | 5 | 495-720 |


Episode 229 (741 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the crayfish | 2 | top-pinch | 4 | 5 | 0-324 |
| 2 | move the crayfish to the box | 1 | na | 5 | 5 | 324-456 |
| 3 | release the crayfish in the box | 2 | set-down | 4 | 5 | 456-540 |
| 4 | return to home | 1 | na | 5 | 5 | 540-741 |


Episode 230 (723 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the crayfish | 2 | top-pinch | 4 | 5 | 0-321 |
| 2 | move the crayfish to the box | 1 | na | 5 | 5 | 321-477 |
| 3 | release the crayfish in the box | 2 | set-down | 4 | 5 | 477-555 |
| 4 | return to home | 1 | na | 5 | 5 | 555-723 |


Episode 222 (771 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the crayfish | 2 | top-pinch | 4 | 5 | 0-312 |
| 2 | move the crayfish to the box | 1 | na | 5 | 5 | 312-420 |
| 3 | release the crayfish in the box | 2 | set-down | 4 | 5 | 420-504 |
| 4 | return to home | 1 | na | 4 | 3 | 504-771 |


Episode 224 (756 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the crayfish | 2 | top-pinch | 4 | 5 | 0-318 |
| 2 | move the crayfish to the box | 1 | na | 5 | 5 | 318-432 |
| 3 | release the crayfish in the box | 2 | drop | 3 | 5 | 432-528 |
| 4 | return to home | 1 | na | 5 | 3 | 528-756 |


Episode 225 (747 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the crayfish | 2 | top-pinch | 4 | 5 | 0-294 |
| 2 | move the crayfish to the box | 1 | na | 5 | 5 | 294-424 |
| 3 | release the crayfish in the box | 2 | set-down | 4 | 5 | 424-504 |
| 4 | return to home | 1 | na | 5 | 3 | 504-747 |


Episode 226 (738 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the crayfish | 2 | top-pinch | 5 | 5 | 0-303 |
| 2 | move the crayfish to the box | 1 | na | 5 | 5 | 303-426 |
| 3 | release the crayfish in the box | 2 | set-down | 4 | 5 | 426-492 |
| 4 | return to home | 1 | na | 5 | 3 | 492-738 |


Episode 221 (867 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the crayfish | 2 | top-pinch | 4 | 1 | 0-363 |
| 2 | move the crayfish to the box | 1 | na | 5 | 5 | 363-486 |
| 3 | release the crayfish in the box | 2 | drop | 2 | 5 | 486-597 |
| 4 | return to home | 1 | na | 5 | 5 | 597-867 |


Episode 228 (798 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the crayfish | 2 | top-pinch | 3 | 1 | 0-345 |
| 2 | move the crayfish to the box | 1 | na | 5 | 5 | 345-452 |
| 3 | release the crayfish in the box | 2 | set-down | 4 | 5 | 452-534 |
| 4 | return to home | 1 | na | 4 | 3 | 534-798 |

</details>

### T153 Put the cube in the box

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 177, 178. Table: episode 178 (414 frames, mean quality 4.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cube | 3 | top-pinch | 4 | 5 | 0-209 |
| 2 | move the cube to the box | 1 | na | 5 | 4 | 209-279 |
| 3 | release the cube in the box | 2 | drop | 5 | 4 | 279-306 |
| 4 | return to home | 1 | na | 5 | 5 | 306-414 |

```yaml
task: Put the cube in the box
eval_subtasks:
- grasp the cube
- move the cube to the box
- release the cube in the box
- return to home
eval_subtask_precisions: [3, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (1)</summary>


Episode 177 (351 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cube | 3 | top-pinch | 3 | 1 | 0-193 |
| 2 | move the cube to the box | 1 | na | 5 | 5 | 193-241 |
| 3 | release the cube in the box | 2 | drop | 5 | 4 | 241-264 |
| 4 | return to home | 1 | na | 5 | 4 | 264-351 |

</details>

### T154 Put the cube on the box

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 0, 1, 2. Table: episode 1 (420 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cube | 3 | top-pinch | 4 | 5 | 0-180 |
| 2 | move the cube to the box | 2 | na | 5 | 5 | 180-230 |
| 3 | release the cube on the box | 3 | set-down | 3 | 5 | 230-291 |
| 4 | return to home | 1 | na | 5 | 5 | 291-420 |

```yaml
task: Put the cube on the box
eval_subtasks:
- grasp the cube
- move the cube to the box
- release the cube on the box
- return to home
eval_subtask_precisions: [3, 2, 3, 1]
eval_subtask_contacts: [top-pinch, na, set-down, na]
```

<details><summary>Other episodes (2)</summary>


Episode 2 (492 frames, mean quality 4.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cube | 3 | top-pinch | 4 | 5 | 0-192 |
| 2 | move the cube to the box | 2 | na | 5 | 5 | 192-252 |
| 3 | release the cube on the box | 3 | set-down | 2 | 4 | 252-360 |
| 4 | return to home | 1 | na | 5 | 5 | 360-492 |


Episode 0 (447 frames, mean quality 3.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the cube | 3 | top-pinch | 2 | 1 | 0-204 |
| 2 | move the cube to the box | 2 | na | 5 | 4 | 204-249 |
| 3 | release the cube on the box | 3 | set-down | 2 | 3 | 249-414 |
| 4 | return to home | 1 | na | 5 | 5 | 414-447 |

</details>

### T155 Put the disk in the box

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 53, 54, 55, 56, 57, 58, 59, 60. Table: episode 56 (282 frames, mean quality 4.3).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the disk | 3 | top-pinch | 2 | 5 | 0-180 |
| 2 | move the disk to the box | 1 | na | 5 | 4 | 180-237 |
| 3 | release the disk in the box | 2 | drop | 2 | 4 | 237-282 |

```yaml
task: Put the disk in the box
eval_subtasks:
- grasp the disk
- move the disk to the box
- release the disk in the box
eval_subtask_precisions: [3, 1, 2]
eval_subtask_contacts: [top-pinch, na, drop]
```

<details><summary>Other episodes (7)</summary>


Episode 57 (312 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the disk | 3 | top-pinch | 3 | 5 | 0-198 |
| 2 | move the disk to the box | 1 | na | 5 | 4 | 198-255 |
| 3 | release the disk in the box | 2 | drop | 4 | 4 | 255-312 |


Episode 58 (279 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the disk | 3 | top-pinch | 3 | 5 | 0-180 |
| 2 | move the disk to the box | 1 | na | 5 | 4 | 180-249 |
| 3 | release the disk in the box | 2 | drop | 5 | 4 | 249-279 |


Episode 60 (312 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the disk | 3 | top-pinch | 4 | 5 | 0-207 |
| 2 | move the disk to the box | 1 | na | 5 | 4 | 207-258 |
| 3 | release the disk in the box | 2 | drop | 3 | 4 | 258-312 |


Episode 59 (381 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the disk | 3 | top-pinch | 2 | 3 | 0-255 |
| 2 | move the disk to the box | 1 | na | 5 | 4 | 255-318 |
| 3 | release the disk in the box | 2 | drop | 3 | 4 | 318-381 |


Episode 53 (267 frames, mean quality 3.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the disk | 3 | top-pinch | 4 | 1 | 0-168 |
| 2 | move the disk to the box | 1 | na | 5 | 5 | 168-207 |
| 3 | release the disk in the box | 2 | drop | 3 | 4 | 207-267 |


Episode 54 (360 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the disk | 3 | top-pinch | 3 | 1 | 0-246 |
| 2 | move the disk to the box | 1 | na | 5 | 4 | 246-291 |
| 3 | release the disk in the box | 2 | drop | 4 | 4 | 291-360 |


Episode 55 (351 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the disk | 3 | top-pinch | 3 | 1 | 0-250 |
| 2 | move the disk to the box | 1 | na | 5 | 4 | 250-298 |
| 3 | release the disk in the box | 2 | drop | 3 | 4 | 298-351 |

</details>

### T156 Put the disk on the blue plate

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 65, 66. Table: episode 65 (879 frames, mean quality 4.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the disk nearer the robot | 3 | top-pinch | 2 | 3 | 0-498 |
| 2 | move the disk to the blue plate | 2 | na | 5 | 5 | 498-603 |
| 3 | release the disk on the blue plate | 3 | drop | 3 | 5 | 603-711 |
| 4 | return to home | 1 | na | 5 | 5 | 711-879 |

```yaml
task: Put the disk on the blue plate
eval_subtasks:
- grasp the disk nearer the robot
- move the disk to the blue plate
- release the disk on the blue plate
- return to home
eval_subtask_precisions: [3, 2, 3, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (1)</summary>


Episode 66 (990 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the disk | 3 | top-pinch | 2 | 3 | 0-546 |
| 2 | move the disk to the blue plate | 2 | na | 5 | 4 | 546-666 |
| 3 | release the disk on the blue plate | 3 | set-down | 4 | 5 | 666-786 |
| 4 | return to home | 1 | na | 5 | 5 | 786-990 |

</details>

### T157 Put the microcontroller farther from the box in the box

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 73, 80. Table: episode 73 (654 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the microcontroller farther from the box | 3 | top-pinch | 2 | 3 | 0-396 |
| 2 | move the microcontroller to the box | 1 | na | 5 | 4 | 396-504 |
| 3 | release the microcontroller in the box | 1 | drop | 4 | 4 | 504-555 |
| 4 | return to home | 1 | na | 5 | 5 | 555-654 |

```yaml
task: Put the microcontroller farther from the box in the box
eval_subtasks:
- grasp the microcontroller farther from the box
- move the microcontroller to the box
- release the microcontroller in the box
- return to home
eval_subtask_precisions: [3, 1, 1, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (1)</summary>


Episode 80 (447 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the microcontroller farther from the box | 3 | top-pinch | 3 | 5 | 0-285 |
| 2 | move the microcontroller to the box | 1 | na | 5 | 3 | 285-354 |
| 3 | release the microcontroller in the box | 1 | drop | 5 | 3 | 354-380 |
| 4 | return to home | 1 | na | 5 | 3 | 380-447 |

</details>

### T158 Put the microcontroller farthest from the box in the box

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 76. Table: episode 76 (651 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the microcontroller farthest from the box | 3 | top-pinch | 3 | 3 | 0-387 |
| 2 | move the microcontroller to the box | 1 | na | 5 | 4 | 387-480 |
| 3 | release the microcontroller in the box | 1 | drop | 4 | 4 | 480-551 |
| 4 | return to home | 1 | na | 5 | 5 | 551-651 |

```yaml
task: Put the microcontroller farthest from the box in the box
eval_subtasks:
- grasp the microcontroller farthest from the box
- move the microcontroller to the box
- release the microcontroller in the box
- return to home
eval_subtask_precisions: [3, 1, 1, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

### T159 Put the microcontroller in the box

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 75, 77. Table: episode 77 (411 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the microcontroller | 3 | top-pinch | 4 | 5 | 0-264 |
| 2 | move the microcontroller to the box | 1 | na | 5 | 5 | 264-318 |
| 3 | release the microcontroller in the box | 1 | drop | 4 | 3 | 318-348 |
| 4 | return to home | 1 | na | 5 | 3 | 348-411 |

```yaml
task: Put the microcontroller in the box
eval_subtasks:
- grasp the microcontroller
- move the microcontroller to the box
- release the microcontroller in the box
- return to home
eval_subtask_precisions: [3, 1, 1, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (1)</summary>


Episode 75 (735 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the microcontroller | 3 | top-pinch | 2 | 1 | 0-492 |
| 2 | move the microcontroller to the box | 1 | na | 5 | 5 | 492-585 |
| 3 | release the microcontroller in the box | 1 | drop | 5 | 4 | 585-615 |
| 4 | return to home | 1 | na | 5 | 5 | 615-735 |

</details>

### T160 Put the microcontroller nearest the box in the box

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 71, 72, 74, 79. Table: episode 79 (462 frames, mean quality 4.8).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the microcontroller nearest the box | 3 | top-pinch | 4 | 5 | 0-278 |
| 2 | move the microcontroller to the box | 1 | na | 5 | 5 | 278-366 |
| 3 | release the microcontroller in the box | 1 | drop | 4 | 4 | 366-400 |
| 4 | return to home | 1 | na | 5 | 5 | 400-462 |

```yaml
task: Put the microcontroller nearest the box in the box
eval_subtasks:
- grasp the microcontroller nearest the box
- move the microcontroller to the box
- release the microcontroller in the box
- return to home
eval_subtask_precisions: [3, 1, 1, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (3)</summary>


Episode 74 (687 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the microcontroller nearest the box | 3 | top-pinch | 4 | 3 | 0-378 |
| 2 | move the microcontroller to the box | 1 | na | 5 | 3 | 378-558 |
| 3 | release the microcontroller in the box | 1 | drop | 2 | 4 | 558-594 |
| 4 | return to home | 1 | na | 5 | 5 | 594-687 |


Episode 71 (582 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the microcontroller nearest the box | 3 | top-pinch | 2 | 1 | 0-356 |
| 2 | move the microcontroller to the box | 1 | na | 5 | 5 | 356-446 |
| 3 | release the microcontroller in the box | 1 | drop | 3 | 4 | 446-481 |
| 4 | return to home | 1 | na | 5 | 4 | 481-582 |


Episode 72 (498 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the microcontroller nearest the box | 3 | top-pinch | 4 | 1 | 0-320 |
| 2 | move the microcontroller to the box | 1 | na | 4 | 5 | 320-384 |
| 3 | release the microcontroller in the box | 1 | drop | 4 | 3 | 384-433 |
| 4 | return to home | 1 | na | 5 | 3 | 433-498 |

</details>

### T161 Put the microcontroller nearest the far end of the box in the box

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 78. Table: episode 78 (615 frames, mean quality 3.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the microcontroller nearest the far end of the box | 3 | top-pinch | 3 | 1 | 0-417 |
| 2 | move the microcontroller to the box | 1 | na | 5 | 5 | 417-459 |
| 3 | release the microcontroller in the box | 1 | drop | 4 | 3 | 459-516 |
| 4 | return to home | 1 | na | 5 | 3 | 516-615 |

```yaml
task: Put the microcontroller nearest the far end of the box in the box
eval_subtasks:
- grasp the microcontroller nearest the far end of the box
- move the microcontroller to the box
- release the microcontroller in the box
- return to home
eval_subtask_precisions: [3, 1, 1, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

### T162 Put the motor in the box

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 201, 202, 203, 204, 205, 206, 207, 208, 209, 210. Table: episode 201 (612 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the motor | 3 | side-pinch | 3 | 5 | 0-297 |
| 2 | move the motor to the box | 2 | na | 5 | 5 | 297-372 |
| 3 | release the motor in the box | 3 | insert | 1 | 5 | 372-461 |
| 4 | return to home | 1 | na | 5 | 5 | 461-612 |

```yaml
task: Put the motor in the box
eval_subtasks:
- grasp the motor
- move the motor to the box
- release the motor in the box
- return to home
eval_subtask_precisions: [3, 2, 3, 1]
eval_subtask_contacts: [side-pinch, na, insert, na]
```

<details><summary>Other episodes (9)</summary>


Episode 203 (570 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the motor | 3 | side-pinch | 3 | 5 | 0-277 |
| 2 | move the motor to the box | 2 | na | 5 | 5 | 277-339 |
| 3 | release the motor in the box | 3 | set-down | 3 | 5 | 339-432 |
| 4 | return to home | 1 | na | 5 | 5 | 432-570 |


Episode 204 (642 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the motor | 3 | side-pinch | 4 | 5 | 0-337 |
| 2 | move the motor to the box | 2 | na | 4 | 5 | 337-414 |
| 3 | release the motor in the box | 3 | insert | 3 | 5 | 414-507 |
| 4 | return to home | 1 | na | 5 | 5 | 507-642 |


Episode 205 (615 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the motor | 3 | top-pinch | 3 | 5 | 0-318 |
| 2 | move the motor to the box | 2 | na | 5 | 5 | 318-402 |
| 3 | release the motor in the box | 3 | insert | 3 | 5 | 402-480 |
| 4 | return to home | 1 | na | 5 | 5 | 480-615 |


Episode 206 (636 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the motor | 3 | side-pinch | 5 | 5 | 0-300 |
| 2 | move the motor to the box | 2 | na | 5 | 5 | 300-390 |
| 3 | release the motor in the box | 3 | insert | 2 | 5 | 390-480 |
| 4 | return to home | 1 | na | 5 | 5 | 480-636 |


Episode 207 (567 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the motor | 3 | top-pinch | 4 | 5 | 0-336 |
| 2 | move the motor to the box | 2 | na | 5 | 5 | 336-402 |
| 3 | release the motor in the box | 3 | insert | 4 | 5 | 402-474 |
| 4 | return to home | 1 | na | 5 | 5 | 474-567 |


Episode 208 (483 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the motor | 3 | side-pinch | 4 | 5 | 0-225 |
| 2 | move the motor to the box | 2 | na | 5 | 5 | 225-300 |
| 3 | release the motor in the box | 3 | insert | 3 | 5 | 300-354 |
| 4 | return to home | 1 | na | 5 | 5 | 354-483 |


Episode 202 (552 frames, mean quality 4.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the motor | 3 | side-pinch | 4 | 5 | 0-255 |
| 2 | move the motor to the box | 2 | na | 5 | 5 | 255-324 |
| 3 | release the motor in the box | 3 | insert | 2 | 4 | 324-450 |
| 4 | return to home | 1 | na | 5 | 5 | 450-552 |


Episode 209 (585 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the motor | 3 | top-pinch | 3 | 5 | 0-273 |
| 2 | move the motor to the box | 2 | na | 5 | 5 | 273-348 |
| 3 | release the motor in the box | 3 | insert | 3 | 5 | 348-465 |
| 4 | return to home | 1 | na | 5 | 3 | 465-585 |


Episode 210 (606 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the motor | 3 | top-pinch | 4 | 3 | 0-276 |
| 2 | move the motor to the box | 2 | na | 5 | 5 | 276-357 |
| 3 | release the motor in the box | 3 | insert | 3 | 5 | 357-452 |
| 4 | return to home | 1 | na | 5 | 3 | 452-606 |

</details>

### T163 Put the paper ball on the table

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 43, 44, 45, 46, 47, 48, 49, 50, 51, 52. Table: episode 51 (414 frames, mean quality 4.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper ball | 2 | top-pinch | 5 | 3 | 0-249 |
| 2 | move the paper ball to the table | 1 | na | 5 | 5 | 249-285 |
| 3 | release the paper ball on the table | 1 | set-down | 5 | 5 | 285-297 |
| 4 | return to home | 1 | na | 3 | 5 | 297-414 |

```yaml
task: Put the paper ball on the table
eval_subtasks:
- grasp the paper ball
- move the paper ball to the table
- release the paper ball on the table
- return to home
eval_subtask_precisions: [2, 1, 1, 1]
eval_subtask_contacts: [top-pinch, na, set-down, na]
```

<details><summary>Other episodes (9)</summary>


Episode 48 (456 frames, mean quality 4.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper ball | 2 | side-pinch | 5 | 5 | 0-177 |
| 2 | move the paper ball to the table | 1 | na | 5 | 4 | 177-210 |
| 3 | release the paper ball on the table | 1 | drop | 5 | 3 | 210-351 |
| 4 | return to home | 1 | na | 5 | 4 | 351-456 |


Episode 45 (330 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper ball | 2 | top-pinch | 5 | 5 | 0-177 |
| 2 | release the paper ball on the table | 1 | drop | 5 | 3 | 177-217 |
| 3 | return to home | 1 | na | 2 | 3 | 217-330 |


Episode 46 (348 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper ball | 2 | top-pinch | 5 | 5 | 0-183 |
| 2 | release the paper ball on the table | 1 | drop | 5 | 3 | 183-231 |
| 3 | return to home | 1 | na | 2 | 3 | 231-348 |


Episode 49 (498 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper ball | 2 | cloth-pinch | 5 | 2 | 0-357 |
| 2 | release the paper ball on the table | 1 | set-down | 5 | 4 | 357-393 |
| 3 | return to home | 1 | na | 5 | 5 | 393-498 |


Episode 44 (375 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper ball | 2 | top-pinch | 5 | 5 | 0-216 |
| 2 | move the paper ball to the table | 1 | na | 5 | 3 | 216-256 |
| 3 | release the paper ball on the table | 1 | drop | 5 | 3 | 256-268 |
| 4 | return to home | 1 | na | 5 | 3 | 268-375 |


Episode 52 (285 frames, mean quality 3.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper ball | 2 | cloth-pinch | 5 | 4 | 0-150 |
| 2 | release the paper ball on the table | 1 | drop | 5 | 3 | 150-180 |
| 3 | return to home | 1 | na | 4 | 3 | 180-285 |


Episode 43 (408 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper ball | 2 | top-pinch | 5 | 3 | 0-225 |
| 2 | release the paper ball on the table | 1 | drop | 5 | 3 | 225-264 |
| 3 | return to home | 1 | na | 5 | 3 | 264-408 |


Episode 47 (504 frames, mean quality 2.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper ball | 2 | top-pinch | 4 | 2 | 0-348 |
| 2 | release the paper ball on the table | 1 | drop | 5 | 3 | 348-382 |
| 3 | return to home | 1 | na | 4 | 3 | 382-504 |


Episode 50 (459 frames, mean quality 2.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper ball | 2 | cloth-pinch | 5 | 2 | 0-291 |
| 2 | release the paper ball on the table | 1 | drop | 5 | 3 | 291-345 |
| 3 | return to home | 1 | na | 2 | 3 | 345-459 |

</details>

### T164 Put the paper, the packet farthest from the trash can, the cork and the other packet in the trash can

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 194. Table: episode 194 (2001 frames, mean quality 3.5).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper | 2 | cloth-pinch | 5 | 1 | 0-456 |
| 2 | move the paper to the trash can | 1 | na | 5 | 5 | 456-588 |
| 3 | release the paper in the trash can | 1 | drop | 5 | 4 | 588-648 |
| 4 | grasp the packet farthest from the trash can | 2 | cloth-pinch | 5 | 2 | 648-1014 |
| 5 | move the packet to the trash can | 1 | na | 5 | 5 | 1014-1095 |
| 6 | release the packet in the trash can | 1 | drop | 5 | 3 | 1095-1125 |
| 7 | grasp the cork | 3 | top-pinch | 5 | 3 | 1125-1323 |
| 8 | move the cork to the trash can | 1 | na | 5 | 5 | 1323-1395 |
| 9 | release the cork in the trash can | 1 | drop | 5 | 3 | 1395-1419 |
| 10 | grasp the packet | 2 | cloth-pinch | 4 | 2 | 1419-1773 |
| 11 | move the packet to the trash can | 1 | na | 5 | 5 | 1773-1851 |
| 12 | release the packet in the trash can | 1 | drop | 5 | 4 | 1851-1884 |
| 13 | return to home | 1 | na | 5 | 4 | 1884-2001 |

```yaml
task: Put the paper, the packet farthest from the trash can, the cork and the other packet in the trash can
eval_subtasks:
- grasp the paper
- move the paper to the trash can
- release the paper in the trash can
- grasp the packet farthest from the trash can
- move the packet to the trash can
- release the packet in the trash can
- grasp the cork
- move the cork to the trash can
- release the cork in the trash can
- grasp the packet
- move the packet to the trash can
- release the packet in the trash can
- return to home
eval_subtask_precisions: [2, 1, 1, 2, 1, 1, 3, 1, 1, 2, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, top-pinch, na, drop, cloth-pinch, na, drop, na]
```

### T165 Put the paper, the packet, the cork and the tissue in the trash can

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 195, 196, 197, 198, 199, 200. Table: episode 197 (1422 frames, mean quality 4.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper | 2 | cloth-pinch | 4 | 1 | 0-219 |
| 2 | move the paper to the trash can | 1 | na | 5 | 5 | 219-330 |
| 3 | release the paper in the trash can | 1 | drop | 5 | 4 | 330-384 |
| 4 | grasp the packet | 2 | cloth-pinch | 5 | 3 | 384-606 |
| 5 | move the packet to the trash can | 1 | na | 5 | 5 | 606-700 |
| 6 | release the packet in the trash can | 1 | drop | 5 | 4 | 700-750 |
| 7 | grasp the cork | 3 | top-pinch | 5 | 5 | 750-921 |
| 8 | move the cork to the trash can | 1 | na | 5 | 5 | 921-996 |
| 9 | release the cork in the trash can | 1 | drop | 5 | 4 | 996-1035 |
| 10 | grasp the tissue | 2 | cloth-pinch | 5 | 5 | 1035-1194 |
| 11 | move the tissue to the trash can | 1 | na | 5 | 5 | 1194-1254 |
| 12 | release the tissue in the trash can | 1 | drop | 5 | 4 | 1254-1317 |
| 13 | return to home | 1 | na | 5 | 4 | 1317-1422 |

```yaml
task: Put the paper, the packet, the cork and the tissue in the trash can
eval_subtasks:
- grasp the paper
- move the paper to the trash can
- release the paper in the trash can
- grasp the packet
- move the packet to the trash can
- release the packet in the trash can
- grasp the cork
- move the cork to the trash can
- release the cork in the trash can
- grasp the tissue
- move the tissue to the trash can
- release the tissue in the trash can
- return to home
eval_subtask_precisions: [2, 1, 1, 2, 1, 1, 3, 1, 1, 2, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, na, drop, cloth-pinch, na, drop, top-pinch, na, drop, cloth-pinch, na, drop, na]
```

<details><summary>Other episodes (5)</summary>


Episode 195 (1605 frames, mean quality 4.1)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper | 2 | cloth-pinch | 5 | 1 | 0-373 |
| 2 | move the paper to the trash can | 1 | na | 5 | 5 | 373-465 |
| 3 | release the paper in the trash can | 1 | drop | 5 | 4 | 465-523 |
| 4 | grasp the packet | 2 | cloth-pinch | 5 | 4 | 523-743 |
| 5 | move the packet to the trash can | 1 | na | 5 | 5 | 743-825 |
| 6 | release the packet in the trash can | 1 | drop | 5 | 3 | 825-869 |
| 7 | grasp the cork | 3 | cloth-pinch | 5 | 5 | 869-1035 |
| 8 | move the cork to the trash can | 1 | na | 5 | 5 | 1035-1103 |
| 9 | release the cork in the trash can | 1 | drop | 5 | 4 | 1103-1131 |
| 10 | grasp the tissue | 2 | cloth-pinch | 5 | 4 | 1131-1363 |
| 11 | move the tissue to the trash can | 1 | na | 5 | 5 | 1363-1443 |
| 12 | release the tissue in the trash can | 1 | drop | 5 | 4 | 1443-1473 |
| 13 | return to home | 1 | na | 5 | 4 | 1473-1605 |


Episode 199 (1854 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper | 2 | cloth-pinch | 5 | 1 | 0-270 |
| 2 | move the paper to the trash can | 1 | na | 5 | 5 | 270-348 |
| 3 | release the paper in the trash can | 1 | drop | 5 | 4 | 348-387 |
| 4 | grasp the packet | 2 | cloth-pinch | 4 | 1 | 387-1014 |
| 5 | move the packet to the trash can | 1 | na | 5 | 5 | 1014-1102 |
| 6 | release the packet in the trash can | 1 | drop | 5 | 4 | 1102-1135 |
| 7 | grasp the cork | 3 | top-pinch | 5 | 5 | 1135-1288 |
| 8 | move the cork to the trash can | 1 | na | 5 | 5 | 1288-1360 |
| 9 | release the cork in the trash can | 1 | drop | 5 | 4 | 1360-1393 |
| 10 | grasp the tissue | 1 | cloth-pinch | 5 | 3 | 1393-1609 |
| 11 | move the tissue to the trash can | 1 | na | 5 | 4 | 1609-1699 |
| 12 | release the tissue in the trash can | 1 | drop | 5 | 4 | 1699-1747 |
| 13 | return to home | 1 | na | 5 | 4 | 1747-1854 |


Episode 196 (1836 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper | 2 | cloth-pinch | 4 | 1 | 0-234 |
| 2 | move the paper to the trash can | 1 | na | 5 | 5 | 234-318 |
| 3 | release the paper in the trash can | 1 | drop | 5 | 3 | 318-351 |
| 4 | grasp the packet | 2 | cloth-pinch | 5 | 3 | 351-603 |
| 5 | move the packet to the trash can | 1 | na | 5 | 4 | 603-726 |
| 6 | release the packet in the trash can | 1 | drop | 5 | 3 | 726-840 |
| 7 | grasp the cork | 3 | top-pinch | 5 | 5 | 840-1038 |
| 8 | move the cork to the trash can | 1 | na | 5 | 5 | 1038-1128 |
| 9 | release the cork in the trash can | 1 | drop | 5 | 4 | 1128-1155 |
| 10 | grasp the tissue | 2 | cloth-pinch | 3 | 2 | 1155-1572 |
| 11 | move the tissue to the trash can | 1 | na | 5 | 5 | 1572-1647 |
| 12 | release the tissue in the trash can | 1 | drop | 5 | 4 | 1647-1716 |
| 13 | return to home | 1 | na | 5 | 4 | 1716-1836 |


Episode 198 (2001 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper | 2 | cloth-pinch | 5 | 1 | 0-414 |
| 2 | move the paper to the trash can | 1 | na | 5 | 5 | 414-498 |
| 3 | release the paper in the trash can | 1 | drop | 5 | 3 | 498-570 |
| 4 | grasp the packet | 2 | cloth-pinch | 5 | 2 | 570-1011 |
| 5 | move the packet to the trash can | 1 | na | 5 | 4 | 1011-1116 |
| 6 | release the packet in the trash can | 1 | drop | 5 | 4 | 1116-1149 |
| 7 | grasp the cork | 3 | top-pinch | 5 | 5 | 1149-1404 |
| 8 | move the cork to the trash can | 1 | na | 5 | 4 | 1404-1488 |
| 9 | release the cork in the trash can | 1 | drop | 5 | 4 | 1488-1512 |
| 10 | grasp the tissue | 2 | cloth-pinch | 4 | 3 | 1512-1800 |
| 11 | move the tissue to the trash can | 1 | na | 5 | 5 | 1800-1884 |
| 12 | release the tissue in the trash can | 1 | drop | 5 | 4 | 1884-1911 |
| 13 | return to home | 1 | na | 5 | 4 | 1911-2001 |


Episode 200 (1965 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the paper | 1 | cloth-pinch | 4 | 1 | 0-303 |
| 2 | move the paper to the trash can | 1 | na | 5 | 5 | 303-429 |
| 3 | release the paper in the trash can | 1 | drop | 5 | 4 | 429-459 |
| 4 | grasp the packet | 2 | cloth-pinch | 4 | 1 | 459-1185 |
| 5 | move the packet to the trash can | 1 | na | 5 | 5 | 1185-1272 |
| 6 | release the packet in the trash can | 1 | drop | 5 | 4 | 1272-1300 |
| 7 | grasp the cork | 3 | handle-grasp | 5 | 5 | 1300-1450 |
| 8 | move the cork to the trash can | 1 | na | 5 | 3 | 1450-1518 |
| 9 | release the cork in the trash can | 1 | drop | 5 | 3 | 1518-1542 |
| 10 | grasp the tissue | 1 | cloth-pinch | 5 | 3 | 1542-1767 |
| 11 | move the tissue to the trash can | 1 | na | 5 | 4 | 1767-1840 |
| 12 | release the tissue in the trash can | 1 | drop | 5 | 4 | 1840-1869 |
| 13 | return to home | 1 | na | 5 | 4 | 1869-1965 |

</details>

### T166 Put the purple box on the plate

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 11, 12. Table: episode 12 (201 frames, mean quality 3.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the purple box | 3 | top-pinch | 5 | 3 | 0-117 |
| 2 | move the purple box to the plate | 1 | na | 5 | 4 | 117-165 |
| 3 | release the purple box on the plate | 2 | set-down | 5 | 4 | 165-201 |

```yaml
task: Put the purple box on the plate
eval_subtasks:
- grasp the purple box
- move the purple box to the plate
- release the purple box on the plate
eval_subtask_precisions: [3, 1, 2]
eval_subtask_contacts: [top-pinch, na, set-down]
```

<details><summary>Other episodes (1)</summary>


Episode 11 (498 frames, mean quality 3.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the purple box | 3 | top-pinch | 5 | 3 | 0-312 |
| 2 | move the purple box to the plate | 1 | na | 5 | 3 | 312-396 |
| 3 | release the purple box on the plate | 2 | drop | 4 | 4 | 396-456 |
| 4 | return to home | 1 | na | 5 | 4 | 456-498 |

</details>

### T167 Put the red can in the small taped area

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 8, 9, 10. Table: episode 8 (576 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red can in the large taped area | 3 | top-pinch | 5 | 5 | 0-186 |
| 2 | move the red can to the small taped area | 1 | na | 5 | 5 | 186-348 |
| 3 | release the red can in the small taped area | 2 | set-down | 3 | 5 | 348-438 |
| 4 | return to home | 1 | na | 5 | 5 | 438-576 |

```yaml
task: Put the red can in the small taped area
eval_subtasks:
- grasp the red can in the large taped area
- move the red can to the small taped area
- release the red can in the small taped area
- return to home
eval_subtask_precisions: [3, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, set-down, na]
```

<details><summary>Other episodes (2)</summary>


Episode 10 (624 frames, mean quality 4.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red can in the large taped area | 3 | side-pinch | 4 | 5 | 0-234 |
| 2 | move the red can to the small taped area | 1 | na | 5 | 5 | 234-402 |
| 3 | release the red can in the small taped area | 2 | set-down | 3 | 4 | 402-516 |
| 4 | return to home | 1 | na | 5 | 5 | 516-624 |


Episode 9 (585 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red can in the large taped area | 3 | top-pinch | 3 | 5 | 0-216 |
| 2 | move the red can to the small taped area | 1 | na | 4 | 4 | 216-396 |
| 3 | release the red can in the small taped area | 2 | set-down | 2 | 4 | 396-441 |
| 4 | return to home | 1 | na | 5 | 4 | 441-585 |

</details>

### T168 Put the red die in the bin

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 167, 168, 169, 170, 171, 172, 173, 174, 175, 176. Table: episode 176 (717 frames, mean quality 3.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red die | 3 | top-pinch | 3 | 1 | 0-382 |
| 2 | move the red die to the bin | 1 | na | 5 | 5 | 382-465 |
| 3 | release the red die in the bin | 2 | drop | 3 | 4 | 465-502 |
| 4 | return to home | 1 | na | 4 | 3 | 502-717 |

```yaml
task: Put the red die in the bin
eval_subtasks:
- grasp the red die
- move the red die to the bin
- release the red die in the bin
- return to home
eval_subtask_precisions: [3, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (9)</summary>


Episode 167 (765 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red die | 3 | top-pinch | 2 | 1 | 0-342 |
| 2 | move the red die to the bin | 1 | na | 5 | 4 | 342-440 |
| 3 | release the red die in the bin | 2 | drop | 2 | 4 | 440-484 |
| 4 | return to home | 1 | na | 3 | 3 | 484-765 |


Episode 168 (942 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red die | 3 | top-pinch | 2 | 1 | 0-376 |
| 2 | move the red die to the bin | 1 | na | 5 | 4 | 376-474 |
| 3 | release the red die in the bin | 2 | drop | 1 | 4 | 474-513 |
| 4 | return to home | 1 | na | 3 | 3 | 513-942 |


Episode 169 (843 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red die | 3 | top-pinch | 2 | 1 | 0-354 |
| 2 | move the red die to the bin | 1 | na | 5 | 4 | 354-444 |
| 3 | release the red die in the bin | 2 | drop | 2 | 4 | 444-483 |
| 4 | return to home | 1 | na | 3 | 3 | 483-843 |


Episode 170 (888 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red die | 3 | top-pinch | 2 | 1 | 0-417 |
| 2 | move the red die to the bin | 1 | na | 5 | 4 | 417-510 |
| 3 | release the red die in the bin | 2 | drop | 2 | 4 | 510-555 |
| 4 | return to home | 1 | na | 4 | 3 | 555-888 |


Episode 171 (765 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red die | 3 | top-pinch | 2 | 1 | 0-353 |
| 2 | move the red die to the bin | 1 | na | 5 | 4 | 353-430 |
| 3 | release the red die in the bin | 2 | drop | 2 | 4 | 430-480 |
| 4 | return to home | 1 | na | 4 | 3 | 480-765 |


Episode 172 (1056 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red die | 3 | top-pinch | 2 | 1 | 0-567 |
| 2 | move the red die to the bin | 1 | na | 5 | 4 | 567-651 |
| 3 | release the red die in the bin | 2 | drop | 2 | 4 | 651-687 |
| 4 | return to home | 1 | na | 3 | 3 | 687-1056 |


Episode 173 (858 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red die | 3 | top-pinch | 2 | 1 | 0-357 |
| 2 | move the red die to the bin | 1 | na | 5 | 4 | 357-444 |
| 3 | release the red die in the bin | 2 | drop | 2 | 4 | 444-492 |
| 4 | return to home | 1 | na | 3 | 3 | 492-858 |


Episode 174 (939 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red die | 3 | top-pinch | 1 | 1 | 0-447 |
| 2 | move the red die to the bin | 1 | na | 5 | 4 | 447-519 |
| 3 | release the red die in the bin | 2 | drop | 2 | 4 | 519-573 |
| 4 | return to home | 1 | na | 4 | 3 | 573-939 |


Episode 175 (825 frames, mean quality 3.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the red die | 3 | top-pinch | 2 | 1 | 0-447 |
| 2 | move the red die to the bin | 1 | na | 5 | 4 | 447-516 |
| 3 | release the red die in the bin | 2 | drop | 3 | 4 | 516-558 |
| 4 | return to home | 1 | na | 4 | 3 | 558-825 |

</details>

### T169 Put the tea bag in the cup

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 135, 136, 137, 138, 139, 140, 141, 142, 143, 144. Table: episode 135 (180 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tea bag | 3 | cloth-pinch | 5 | 5 | 0-88 |
| 2 | release the tea bag in the cup | 2 | set-down | 5 | 5 | 88-142 |
| 3 | return to home | 1 | na | 5 | 5 | 142-180 |

```yaml
task: Put the tea bag in the cup
eval_subtasks:
- grasp the tea bag
- release the tea bag in the cup
- return to home
eval_subtask_precisions: [3, 2, 1]
eval_subtask_contacts: [cloth-pinch, set-down, na]
```

<details><summary>Other episodes (9)</summary>


Episode 136 (159 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tea bag | 3 | cloth-pinch | 5 | 5 | 0-75 |
| 2 | release the tea bag in the cup | 2 | set-down | 5 | 5 | 75-111 |
| 3 | return to home | 1 | na | 5 | 5 | 111-159 |


Episode 138 (150 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tea bag | 3 | cloth-pinch | 5 | 5 | 0-82 |
| 2 | release the tea bag in the cup | 2 | set-down | 5 | 5 | 82-118 |
| 3 | return to home | 1 | na | 5 | 5 | 118-150 |


Episode 139 (168 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tea bag | 3 | cloth-pinch | 5 | 5 | 0-78 |
| 2 | release the tea bag in the cup | 2 | set-down | 5 | 5 | 78-126 |
| 3 | return to home | 1 | na | 5 | 5 | 126-168 |


Episode 140 (159 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tea bag | 3 | cloth-pinch | 5 | 5 | 0-81 |
| 2 | release the tea bag in the cup | 2 | set-down | 5 | 5 | 81-117 |
| 3 | return to home | 1 | na | 5 | 5 | 117-159 |


Episode 141 (162 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tea bag | 3 | cloth-pinch | 5 | 5 | 0-64 |
| 2 | release the tea bag in the cup | 2 | set-down | 5 | 5 | 64-100 |
| 3 | return to home | 1 | na | 5 | 5 | 100-162 |


Episode 142 (165 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tea bag | 3 | cloth-pinch | 5 | 5 | 0-92 |
| 2 | release the tea bag in the cup | 2 | set-down | 5 | 5 | 92-127 |
| 3 | return to home | 1 | na | 5 | 5 | 127-165 |


Episode 143 (174 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tea bag | 3 | cloth-pinch | 5 | 5 | 0-91 |
| 2 | release the tea bag in the cup | 2 | set-down | 5 | 5 | 91-125 |
| 3 | return to home | 1 | na | 5 | 5 | 125-174 |


Episode 144 (246 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tea bag | 3 | cloth-pinch | 4 | 5 | 0-135 |
| 2 | release the tea bag in the cup | 2 | set-down | 5 | 5 | 135-176 |
| 3 | return to home | 1 | na | 5 | 5 | 176-246 |


Episode 137 (183 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the tea bag | 3 | cloth-pinch | 5 | 5 | 0-87 |
| 2 | release the tea bag in the cup | 2 | set-down | 5 | 3 | 87-150 |
| 3 | return to home | 1 | na | 5 | 5 | 150-183 |

</details>

### T170 Put the wooden disk on the blue plate

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 63, 67, 69, 70. Table: episode 63 (834 frames, mean quality 4.2).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the wooden disk nearest the black disk | 3 | top-pinch | 2 | 3 | 0-483 |
| 2 | move the wooden disk to the blue plate | 1 | na | 5 | 4 | 483-597 |
| 3 | release the wooden disk on the blue plate | 2 | drop | 4 | 5 | 597-681 |
| 4 | return to home | 1 | na | 5 | 5 | 681-834 |

```yaml
task: Put the wooden disk on the blue plate
eval_subtasks:
- grasp the wooden disk nearest the black disk
- move the wooden disk to the blue plate
- release the wooden disk on the blue plate
- return to home
eval_subtask_precisions: [3, 1, 2, 1]
eval_subtask_contacts: [top-pinch, na, drop, na]
```

<details><summary>Other episodes (3)</summary>


Episode 67 (771 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the wooden disk apart from the others | 3 | top-pinch | 2 | 3 | 0-414 |
| 2 | move the wooden disk to the blue plate | 1 | na | 5 | 4 | 414-570 |
| 3 | release the wooden disk on the blue plate | 2 | drop | 3 | 5 | 570-645 |
| 4 | return to home | 1 | na | 5 | 5 | 645-771 |


Episode 70 (774 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the wooden disk | 3 | top-pinch | 2 | 2 | 0-450 |
| 2 | move the wooden disk to the blue plate | 2 | na | 5 | 5 | 450-590 |
| 3 | release the wooden disk on the blue plate | 3 | set-down | 3 | 5 | 590-621 |
| 4 | return to home | 1 | na | 5 | 5 | 621-774 |


Episode 69 (846 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the wooden disk in the middle of the grey plate | 3 | top-pinch | 2 | 3 | 0-456 |
| 2 | move the wooden disk to the blue plate | 1 | na | 5 | 4 | 456-615 |
| 3 | release the wooden disk on the blue plate | 2 | drop | 3 | 4 | 615-672 |
| 4 | return to home | 1 | na | 5 | 4 | 672-846 |

</details>

### T171 Shake out the towel, spread it flat, fold it in half three times and press it flat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 132. Table: episode 132 (2079 frames, mean quality 3.4).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 4 | 1 | 0-198 |
| 2 | shake out the towel | 1 | cloth-pinch | 5 | 4 | 198-330 |
| 3 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 3 | 3 | 330-579 |
| 4 | spread the towel toward the robot | 1 | cloth-pinch | 5 | 3 | 579-717 |
| 5 | spread the towel away from the wall | 1 | push | 5 | 4 | 717-798 |
| 6 | grasp the corner of the towel nearest the robot | 1 | cloth-pinch | 4 | 1 | 798-1140 |
| 7 | fold the towel in half away from the robot | 1 | cloth-pinch | 4 | 4 | 1140-1293 |
| 8 | grasp the edge of the towel nearest the robot | 1 | cloth-pinch | 4 | 5 | 1293-1401 |
| 9 | fold the towel in half away from the robot | 1 | cloth-pinch | 4 | 4 | 1401-1545 |
| 10 | grasp the edge of the towel farthest from the wall | 1 | cloth-pinch | 4 | 2 | 1545-1884 |
| 11 | fold the towel in half toward the wall | 1 | cloth-pinch | 5 | 5 | 1884-1968 |
| 12 | press the towel flat | 1 | press | 5 | 4 | 1968-2019 |
| 13 | return to home | 1 | na | 5 | 4 | 2019-2079 |

```yaml
task: Shake out the towel, spread it flat, fold it in half three times and press it flat
eval_subtasks:
- grasp the corner of the towel with the blue tag
- shake out the towel
- grasp the corner of the towel with the blue tag
- spread the towel toward the robot
- spread the towel away from the wall
- grasp the corner of the towel nearest the robot
- fold the towel in half away from the robot
- grasp the edge of the towel nearest the robot
- fold the towel in half away from the robot
- grasp the edge of the towel farthest from the wall
- fold the towel in half toward the wall
- press the towel flat
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, push, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, press, na]
```

### T172 Shake out the towel, spread it flat, fold it in half twice and slide it toward the robot

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 134. Table: episode 134 (1764 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 5 | 5 | 0-135 |
| 2 | shake out the towel | 1 | cloth-pinch | 5 | 4 | 135-270 |
| 3 | grasp the edge of the towel nearest the robot | 1 | cloth-pinch | 4 | 4 | 270-372 |
| 4 | spread the towel away from the robot | 1 | cloth-pinch | 5 | 4 | 372-435 |
| 5 | grasp the corner of the towel nearest the robot | 1 | cloth-pinch | 5 | 2 | 435-632 |
| 6 | spread the towel toward the wall | 1 | cloth-pinch | 5 | 4 | 632-680 |
| 7 | grasp the corner of the towel nearest the robot | 1 | cloth-pinch | 4 | 4 | 680-783 |
| 8 | fold the towel in half away from the robot | 1 | cloth-pinch | 5 | 5 | 783-914 |
| 9 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 4 | 3 | 914-1193 |
| 10 | pull the corner with the blue tag toward the wall | 1 | cloth-pinch | 4 | 4 | 1193-1321 |
| 11 | grasp the edge of the towel farthest from the wall | 1 | cloth-pinch | 3 | 3 | 1321-1563 |
| 12 | fold the towel in half toward the wall | 1 | cloth-pinch | 4 | 4 | 1563-1660 |
| 13 | slide the towel toward the robot | 1 | push | 5 | 4 | 1660-1724 |
| 14 | return to home | 1 | na | 5 | 4 | 1724-1764 |

```yaml
task: Shake out the towel, spread it flat, fold it in half twice and slide it toward the robot
eval_subtasks:
- grasp the corner of the towel with the blue tag
- shake out the towel
- grasp the edge of the towel nearest the robot
- spread the towel away from the robot
- grasp the corner of the towel nearest the robot
- spread the towel toward the wall
- grasp the corner of the towel nearest the robot
- fold the towel in half away from the robot
- grasp the corner of the towel with the blue tag
- pull the corner with the blue tag toward the wall
- grasp the edge of the towel farthest from the wall
- fold the towel in half toward the wall
- slide the towel toward the robot
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, push, na]
```

### T173 Shake out the towel, spread it flat, fold it into a triangle and then smaller, and press it flat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 130. Table: episode 130 (3039 frames, mean quality 4.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 4 | 2 | 0-735 |
| 2 | shake out the towel | 1 | cloth-pinch | 5 | 4 | 735-999 |
| 3 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 5 | 5 | 999-1094 |
| 4 | spread the towel toward the robot | 1 | cloth-pinch | 5 | 4 | 1094-1166 |
| 5 | grasp the folded edge of the towel | 1 | cloth-pinch | 2 | 3 | 1166-1421 |
| 6 | spread the towel away from the robot | 1 | cloth-pinch | 5 | 5 | 1421-1478 |
| 7 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 4 | 5 | 1478-1560 |
| 8 | spread the towel toward the wall | 1 | cloth-pinch | 5 | 4 | 1560-1648 |
| 9 | grasp the corner of the towel nearest the robot | 1 | cloth-pinch | 5 | 5 | 1648-1811 |
| 10 | fold the towel into a triangle | 1 | cloth-pinch | 4 | 5 | 1811-1973 |
| 11 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 5 | 3 | 1973-2314 |
| 12 | pull the corner with the blue tag away from the robot | 1 | cloth-pinch | 4 | 4 | 2314-2388 |
| 13 | grasp the corner of the towel nearest the robot | 1 | cloth-pinch | 5 | 3 | 2388-2559 |
| 14 | fold the towel away from the wall | 1 | cloth-pinch | 5 | 4 | 2559-2626 |
| 15 | grasp the edge of the towel nearest the robot | 1 | cloth-pinch | 5 | 3 | 2626-2785 |
| 16 | fold the towel in half away from the robot | 1 | cloth-pinch | 5 | 5 | 2785-2881 |
| 17 | press the towel flat | 1 | press | 4 | 4 | 2881-2998 |
| 18 | return to home | 1 | na | 5 | 4 | 2998-3039 |

```yaml
task: Shake out the towel, spread it flat, fold it into a triangle and then smaller, and press it flat
eval_subtasks:
- grasp the corner of the towel with the blue tag
- shake out the towel
- grasp the corner of the towel with the blue tag
- spread the towel toward the robot
- grasp the folded edge of the towel
- spread the towel away from the robot
- grasp the corner of the towel with the blue tag
- spread the towel toward the wall
- grasp the corner of the towel nearest the robot
- fold the towel into a triangle
- grasp the corner of the towel with the blue tag
- pull the corner with the blue tag away from the robot
- grasp the corner of the towel nearest the robot
- fold the towel away from the wall
- grasp the edge of the towel nearest the robot
- fold the towel in half away from the robot
- press the towel flat
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, press, na]
```

### T174 Shake out the towel, spread it flat, fold it into a triangle, slide it toward the robot, lay it flat, fold it in half and press it flat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 133. Table: episode 133 (2193 frames, mean quality 3.9).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the towel | 1 | cloth-pinch | 5 | 4 | 0-204 |
| 2 | shake out the towel | 1 | cloth-pinch | 5 | 4 | 204-318 |
| 3 | grasp the folded edge of the towel | 1 | cloth-pinch | 3 | 3 | 318-498 |
| 4 | spread the towel away from the robot | 1 | cloth-pinch | 5 | 4 | 498-576 |
| 5 | grasp the edge of the towel nearest the robot | 1 | cloth-pinch | 3 | 3 | 576-768 |
| 6 | spread the towel toward the wall | 1 | cloth-pinch | 5 | 4 | 768-812 |
| 7 | grasp the corner of the towel nearest the robot | 1 | cloth-pinch | 3 | 4 | 812-918 |
| 8 | fold the towel into a triangle | 1 | cloth-pinch | 5 | 4 | 918-1068 |
| 9 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 5 | 5 | 1068-1167 |
| 10 | fold the corner with the blue tag away from the wall | 1 | cloth-pinch | 4 | 3 | 1167-1299 |
| 11 | slide the towel toward the robot | 1 | push | 5 | 3 | 1299-1407 |
| 12 | grasp the edge of the towel farthest from the robot | 1 | cloth-pinch | 3 | 4 | 1407-1530 |
| 13 | lay the towel flat | 1 | cloth-pinch | 5 | 5 | 1530-1599 |
| 14 | grasp the edge of the towel nearest the robot | 1 | cloth-pinch | 5 | 5 | 1599-1686 |
| 15 | spread the towel toward the robot | 1 | cloth-pinch | 5 | 4 | 1686-1767 |
| 16 | grasp the edge of the towel nearest the robot | 1 | cloth-pinch | 5 | 2 | 1767-1977 |
| 17 | fold the towel in half away from the robot | 1 | cloth-pinch | 5 | 5 | 1977-2061 |
| 18 | press the towel flat | 1 | press | 5 | 4 | 2061-2148 |
| 19 | return to home | 1 | na | 5 | 4 | 2148-2193 |

```yaml
task: Shake out the towel, spread it flat, fold it into a triangle, slide it toward the robot, lay it flat, fold it in half and press it flat
eval_subtasks:
- grasp the towel
- shake out the towel
- grasp the folded edge of the towel
- spread the towel away from the robot
- grasp the edge of the towel nearest the robot
- spread the towel toward the wall
- grasp the corner of the towel nearest the robot
- fold the towel into a triangle
- grasp the corner of the towel with the blue tag
- fold the corner with the blue tag away from the wall
- slide the towel toward the robot
- grasp the edge of the towel farthest from the robot
- lay the towel flat
- grasp the edge of the towel nearest the robot
- spread the towel toward the robot
- grasp the edge of the towel nearest the robot
- fold the towel in half away from the robot
- press the towel flat
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, push, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, press, na]
```

### T175 Shake the hand

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 147, 148, 149, 151, 152. Table: episode 149 (492 frames, mean quality 4.7).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the hand | 2 | side-pinch | 5 | 5 | 0-150 |
| 2 | shake the hand | 1 | side-pinch | 3 | 5 | 150-363 |
| 3 | return to home | 1 | na | 5 | 4 | 363-492 |

```yaml
task: Shake the hand
eval_subtasks:
- grasp the hand
- shake the hand
- return to home
eval_subtask_precisions: [2, 1, 1]
eval_subtask_contacts: [side-pinch, side-pinch, na]
```

<details><summary>Other episodes (4)</summary>


Episode 148 (516 frames, mean quality 3.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the hand | 2 | side-pinch | 2 | 1 | 0-240 |
| 2 | shake the hand | 1 | side-pinch | 5 | 5 | 240-393 |
| 3 | return to home | 1 | na | 3 | 5 | 393-516 |


Episode 147 (612 frames, mean quality 3.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the hand | 2 | side-pinch | 4 | 1 | 0-339 |
| 2 | shake the hand | 1 | side-pinch | 4 | 5 | 339-501 |
| 3 | return to home | 1 | na | 4 | 4 | 501-612 |


Episode 151 (606 frames, mean quality 2.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the hand | 2 | side-pinch | 5 | 2 | 0-195 |
| 2 | shake the hand | 1 | side-pinch | 4 | 3 | 195-462 |
| 3 | return to home | 1 | na | 5 | 3 | 462-606 |


Episode 152 (648 frames, mean quality 1.7)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the hand | 2 | side-pinch | 5 | 1 | 0-222 |
| 2 | shake the hand | 1 | side-pinch | 5 | 3 | 222-477 |
| 3 | return to home | 1 | na | 3 | 1 | 477-648 |

</details>

### T176 Shake the hand three times

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 155. Table: episode 155 (729 frames, mean quality 4.3).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the hand | 2 | top-pinch | 5 | 5 | 0-108 |
| 2 | shake the hand | 1 | top-pinch | 5 | 5 | 108-138 |
| 3 | grasp the hand | 2 | top-pinch | 5 | 5 | 138-210 |
| 4 | shake the hand | 1 | top-pinch | 5 | 5 | 210-282 |
| 5 | return to home | 1 | na | 5 | 4 | 282-366 |
| 6 | grasp the hand | 2 | side-pinch | 3 | 1 | 366-660 |
| 7 | shake the hand | 1 | side-pinch | 4 | 5 | 660-729 |

```yaml
task: Shake the hand three times
eval_subtasks:
- grasp the hand
- shake the hand
- grasp the hand
- shake the hand
- return to home
- grasp the hand
- shake the hand
eval_subtask_precisions: [2, 1, 2, 1, 1, 2, 1]
eval_subtask_contacts: [top-pinch, top-pinch, top-pinch, top-pinch, na, side-pinch, side-pinch]
```

### T177 Shake the hand twice

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 150, 153, 154, 156. Table: episode 154 (471 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the hand | 2 | top-pinch | 5 | 5 | 0-150 |
| 2 | shake the hand | 1 | top-pinch | 5 | 5 | 150-222 |
| 3 | grasp the hand | 2 | top-pinch | 5 | 5 | 222-291 |
| 4 | shake the hand | 1 | top-pinch | 5 | 5 | 291-381 |
| 5 | return to home | 1 | na | 5 | 5 | 381-471 |

```yaml
task: Shake the hand twice
eval_subtasks:
- grasp the hand
- shake the hand
- grasp the hand
- shake the hand
- return to home
eval_subtask_precisions: [2, 1, 2, 1, 1]
eval_subtask_contacts: [top-pinch, top-pinch, top-pinch, top-pinch, na]
```

<details><summary>Other episodes (3)</summary>


Episode 153 (516 frames, mean quality 4.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the hand | 2 | side-pinch | 5 | 5 | 0-138 |
| 2 | shake the hand | 1 | side-pinch | 5 | 5 | 138-198 |
| 3 | grasp the hand | 2 | top-pinch | 5 | 5 | 198-306 |
| 4 | shake the hand | 1 | top-pinch | 5 | 5 | 306-378 |
| 5 | return to home | 1 | na | 5 | 4 | 378-516 |


Episode 156 (516 frames, mean quality 4.4)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the hand | 2 | top-pinch | 5 | 4 | 0-132 |
| 2 | shake the hand | 1 | top-pinch | 5 | 5 | 132-195 |
| 3 | grasp the hand | 2 | side-pinch | 5 | 4 | 195-330 |
| 4 | shake the hand | 1 | side-pinch | 5 | 5 | 330-393 |
| 5 | return to home | 1 | na | 5 | 4 | 393-516 |


Episode 150 (540 frames, mean quality 3.8)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the hand | 2 | side-pinch | 5 | 1 | 0-156 |
| 2 | shake the hand | 1 | side-pinch | 5 | 5 | 156-264 |
| 3 | grasp the hand | 2 | side-pinch | 5 | 4 | 264-348 |
| 4 | shake the hand | 1 | side-pinch | 5 | 4 | 348-450 |
| 5 | return to home | 1 | na | 5 | 5 | 450-540 |

</details>

### T178 Spread the towel flat, fold it, spread it out again, fold it into a triangle and then smaller, and press it flat

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 131. Table: episode 131 (5502 frames, mean quality 3.6).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 3 | 3 | 0-252 |
| 2 | lift the towel and drop it | 1 | drop | 5 | 4 | 252-303 |
| 3 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 3 | 5 | 303-456 |
| 4 | spread the towel away from the robot | 1 | cloth-pinch | 5 | 4 | 456-510 |
| 5 | grasp the folded edge of the towel | 1 | cloth-pinch | 4 | 3 | 510-678 |
| 6 | spread the towel toward the robot | 1 | cloth-pinch | 5 | 4 | 678-735 |
| 7 | grasp the edge of the towel nearest the robot | 1 | cloth-pinch | 3 | 5 | 735-808 |
| 8 | spread the towel toward the robot | 1 | cloth-pinch | 4 | 4 | 808-847 |
| 9 | grasp the corner of the towel nearest the wall | 1 | cloth-pinch | 5 | 3 | 847-1067 |
| 10 | fold the corner of the towel nearest the wall toward the middle | 1 | cloth-pinch | 5 | 3 | 1067-1248 |
| 11 | grasp the corner of the towel nearest the robot | 1 | cloth-pinch | 4 | 5 | 1248-1365 |
| 12 | fold the towel away from the wall | 1 | cloth-pinch | 5 | 5 | 1365-1440 |
| 13 | grasp the edge of the towel farthest from the robot | 1 | cloth-pinch | 4 | 3 | 1440-1638 |
| 14 | spread the towel away from the robot | 1 | cloth-pinch | 4 | 4 | 1638-1695 |
| 15 | grasp the edge of the towel farthest from the robot | 1 | cloth-pinch | 3 | 4 | 1695-1765 |
| 16 | spread the towel toward the wall | 1 | cloth-pinch | 3 | 4 | 1765-1815 |
| 17 | grasp the edge of the towel farthest from the wall | 1 | cloth-pinch | 5 | 3 | 1815-1988 |
| 18 | fold the towel toward the wall | 1 | cloth-pinch | 5 | 4 | 1988-2052 |
| 19 | grasp the edge of the towel farthest from the robot | 1 | cloth-pinch | 5 | 2 | 2052-2229 |
| 20 | spread the towel toward the wall | 1 | cloth-pinch | 5 | 4 | 2229-2292 |
| 21 | grasp the folded edge of the towel | 1 | cloth-pinch | 5 | 3 | 2292-2553 |
| 22 | spread the towel away from the wall | 1 | cloth-pinch | 5 | 4 | 2553-2601 |
| 23 | grasp the edge of the towel nearest the robot | 1 | cloth-pinch | 5 | 2 | 2601-2876 |
| 24 | spread the towel toward the wall | 1 | cloth-pinch | 5 | 4 | 2876-2912 |
| 25 | grasp the corner of the towel nearest the robot | 1 | cloth-pinch | 5 | 2 | 2912-3244 |
| 26 | spread the towel away from the robot | 1 | cloth-pinch | 5 | 4 | 3244-3333 |
| 27 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 4 | 1 | 3333-4160 |
| 28 | fold the corner with the blue tag away from the wall | 1 | cloth-pinch | 5 | 4 | 4160-4211 |
| 29 | grasp the corner of the towel with the blue tag | 1 | cloth-pinch | 3 | 4 | 4211-4268 |
| 30 | fold the towel into a triangle | 1 | cloth-pinch | 4 | 3 | 4268-4472 |
| 31 | grasp the corner of the towel nearest the robot | 1 | cloth-pinch | 4 | 2 | 4472-4640 |
| 32 | fold the towel away from the wall | 1 | cloth-pinch | 4 | 5 | 4640-4731 |
| 33 | grasp the corner of the towel farthest from the wall | 1 | cloth-pinch | 3 | 3 | 4731-4852 |
| 34 | fold the towel away from the robot | 1 | cloth-pinch | 5 | 5 | 4852-4956 |
| 35 | grasp the edge of the towel farthest from the robot | 1 | cloth-pinch | 3 | 4 | 4956-5085 |
| 36 | fold the towel toward the robot | 1 | cloth-pinch | 5 | 4 | 5085-5136 |
| 37 | grasp the folded edge of the towel | 1 | cloth-pinch | 3 | 3 | 5136-5292 |
| 38 | fold the towel toward the wall | 1 | cloth-pinch | 4 | 4 | 5292-5385 |
| 39 | press the towel flat | 1 | press | 5 | 4 | 5385-5448 |
| 40 | return to home | 1 | na | 5 | 4 | 5448-5502 |

```yaml
task: Spread the towel flat, fold it, spread it out again, fold it into a triangle and then smaller, and press it flat
eval_subtasks:
- grasp the corner of the towel with the blue tag
- lift the towel and drop it
- grasp the corner of the towel with the blue tag
- spread the towel away from the robot
- grasp the folded edge of the towel
- spread the towel toward the robot
- grasp the edge of the towel nearest the robot
- spread the towel toward the robot
- grasp the corner of the towel nearest the wall
- fold the corner of the towel nearest the wall toward the middle
- grasp the corner of the towel nearest the robot
- fold the towel away from the wall
- grasp the edge of the towel farthest from the robot
- spread the towel away from the robot
- grasp the edge of the towel farthest from the robot
- spread the towel toward the wall
- grasp the edge of the towel farthest from the wall
- fold the towel toward the wall
- grasp the edge of the towel farthest from the robot
- spread the towel toward the wall
- grasp the folded edge of the towel
- spread the towel away from the wall
- grasp the edge of the towel nearest the robot
- spread the towel toward the wall
- grasp the corner of the towel nearest the robot
- spread the towel away from the robot
- grasp the corner of the towel with the blue tag
- fold the corner with the blue tag away from the wall
- grasp the corner of the towel with the blue tag
- fold the towel into a triangle
- grasp the corner of the towel nearest the robot
- fold the towel away from the wall
- grasp the corner of the towel farthest from the wall
- fold the towel away from the robot
- grasp the edge of the towel farthest from the robot
- fold the towel toward the robot
- grasp the folded edge of the towel
- fold the towel toward the wall
- press the towel flat
- return to home
eval_subtask_precisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
eval_subtask_contacts: [cloth-pinch, drop, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, cloth-pinch, press, na]
```

### T179 Stack the green tape roll on the white tape roll

Source `external_rebot_train` (`outputs/external_rebot_train-annotated-v9`), group `external_rebot`, episodes 120, 121, 122, 123, 124, 125, 126, 127, 128, 129. Table: episode 120 (186 frames, mean quality 5.0).

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green tape roll | 3 | rim-pinch | 5 | 5 | 0-111 |
| 2 | move the green tape roll to the white tape roll | 2 | na | 5 | 5 | 111-141 |
| 3 | release the green tape roll on the white tape roll | 3 | set-down | 4 | 5 | 141-186 |

```yaml
task: Stack the green tape roll on the white tape roll
eval_subtasks:
- grasp the green tape roll
- move the green tape roll to the white tape roll
- release the green tape roll on the white tape roll
eval_subtask_precisions: [3, 2, 3]
eval_subtask_contacts: [rim-pinch, na, set-down]
```

<details><summary>Other episodes (9)</summary>


Episode 121 (216 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green tape roll | 3 | rim-pinch | 5 | 5 | 0-124 |
| 2 | move the green tape roll to the white tape roll | 2 | na | 5 | 5 | 124-155 |
| 3 | release the green tape roll on the white tape roll | 3 | set-down | 4 | 5 | 155-193 |
| 4 | return to home | 1 | na | 5 | 5 | 193-216 |


Episode 122 (204 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green tape roll | 3 | rim-pinch | 5 | 5 | 0-124 |
| 2 | move the green tape roll to the white tape roll | 2 | na | 5 | 5 | 124-155 |
| 3 | release the green tape roll on the white tape roll | 3 | set-down | 5 | 5 | 155-186 |
| 4 | return to home | 1 | na | 5 | 5 | 186-204 |


Episode 123 (201 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green tape roll | 3 | rim-pinch | 5 | 5 | 0-120 |
| 2 | move the green tape roll to the white tape roll | 2 | na | 5 | 5 | 120-150 |
| 3 | release the green tape roll on the white tape roll | 3 | set-down | 4 | 5 | 150-183 |
| 4 | return to home | 1 | na | 5 | 5 | 183-201 |


Episode 124 (210 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green tape roll | 3 | rim-pinch | 5 | 5 | 0-117 |
| 2 | move the green tape roll to the white tape roll | 2 | na | 5 | 5 | 117-159 |
| 3 | release the green tape roll on the white tape roll | 3 | set-down | 4 | 5 | 159-189 |
| 4 | return to home | 1 | na | 5 | 5 | 189-210 |


Episode 126 (195 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green tape roll | 3 | rim-pinch | 5 | 5 | 0-102 |
| 2 | move the green tape roll to the white tape roll | 2 | na | 5 | 5 | 102-135 |
| 3 | release the green tape roll on the white tape roll | 3 | set-down | 5 | 5 | 135-195 |


Episode 129 (243 frames, mean quality 5.0)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green tape roll | 3 | rim-pinch | 5 | 5 | 0-129 |
| 2 | move the green tape roll to the white tape roll | 2 | na | 5 | 5 | 129-183 |
| 3 | release the green tape roll on the white tape roll | 3 | set-down | 5 | 5 | 183-219 |
| 4 | return to home | 1 | na | 4 | 5 | 219-243 |


Episode 128 (228 frames, mean quality 4.5)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green tape roll | 3 | rim-pinch | 4 | 3 | 0-138 |
| 2 | move the green tape roll to the white tape roll | 2 | na | 5 | 5 | 138-171 |
| 3 | release the green tape roll on the white tape roll | 3 | set-down | 5 | 5 | 171-205 |
| 4 | return to home | 1 | na | 5 | 5 | 205-228 |


Episode 127 (303 frames, mean quality 4.3)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green tape roll | 3 | rim-pinch | 5 | 5 | 0-159 |
| 2 | move the green tape roll to the white tape roll | 2 | na | 5 | 4 | 159-208 |
| 3 | release the green tape roll on the white tape roll | 3 | set-down | 3 | 4 | 208-303 |


Episode 125 (297 frames, mean quality 4.2)

| step | subtask | precision | contact | speed | quality | frames |
|---|---|---|---|---|---|---|
| 1 | grasp the green tape roll | 3 | rim-pinch | 5 | 3 | 0-186 |
| 2 | move the green tape roll to the white tape roll | 2 | na | 5 | 5 | 186-219 |
| 3 | release the green tape roll on the white tape roll | 3 | set-down | 4 | 4 | 219-276 |
| 4 | return to home | 1 | na | 5 | 5 | 276-297 |

</details>

