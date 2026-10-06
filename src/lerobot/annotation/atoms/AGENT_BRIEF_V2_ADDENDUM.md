# v2 addendum to AGENT_BRIEF.md (read AGENT_BRIEF.md and REVIEW_GUIDE.md first)

This round reviews the v2 corpus. EVERY command you run must be prefixed with these three
environment variables, exactly:

    DIVERSE_DATASET_ROOT=outputs/diverse_robot_dataset_v2 ATOMS_REVIEW_ROOT=outputs/_annotation/subtask_atoms_review_v2 ATOMS_WORK_ROOT=migration/subtask_atoms_v2_2026-09-17 uv run python migration/subtask_atoms_2026-09-08/<tool>.py ...

Paths differ from the brief accordingly: sheets are under
`outputs/_annotation/subtask_atoms_review_v2/sheets/<source>/`, agent reviews go to
`outputs/_annotation/subtask_atoms_review_v2/agent_reviews/<episode_id>.json`, proposals are
`migration/subtask_atoms_v2_2026-09-17/proposals.jsonl`.

Source `molmoact` (MolmoAct Dataset, Franka): native rate is 20 Hz for `molmoact__tabletop__*`
and 15 Hz for `molmoact__household__*`. Cameras: `primary` (external, shown on the sheets),
`secondary` (second external), `wrist`. The state is an end-effector pose, so the gripper trace
`g` is a 0-1 closedness ratio like DROID. The task string is a full annotated instruction
sentence; use its object names. Each episode has ONE parent covering the whole episode.

Tabletop tasks and how to cut them (the guide's contact-atom table applies):
- close the box / close the laptop / close the top drawer: `close the box lid` style contact
  atoms: `grasp the lid` only if the fingers actually close on it, otherwise one `push the lid`
  / `close the box` atom from the approach to the lift-off, then `return to home`.
- knock down the bottle / sanitizer / dish soap: one `push the bottle` atom to the lift-off,
  then `return to home`.
- stand the bottle / stand the sanitizer / pick up the mug and stand it: `grasp the X`,
  `rotate the X` or `lift the X` while carried, `release the X on the table`, `return to home`.
- unload the bowl / plate (from a stand to the table), hang the mug (on a stand/tree), place
  the plate on the stand: pick-and-place: `grasp the X`, `move the X to the C`,
  `release the X on the C`, `return to home`.
- fold the shorts / unfold: `grasp the shorts`, `fold the shorts` (or `unfold the shorts`) per
  grasp cycle, as the guide says for towels.
If a cycle has no transport (the object is manipulated in place), do not invent move/release.
Every atom needs at least 0.5 s; the whole parent must be tiled exactly.

## 2026-09-18 round: `molmoact__household__*` and `droid_success__CLVR__* / __RAIL__*`

Same three environment variables on every command. Sheets, dense strips and closeups work
exactly as in the brief; the sheets for these episodes are already rendered
(`sheets/molmoact/`, `sheets/droid_success/`); the trace on the household overview is the
end-effector speed and the 0-1 gripper closedness.

### Household (Franka, 15 Hz, one parent per episode, `primary` external + `wrist` on the sheets)

Every parent is a whole ~5-30 s episode with one annotated instruction; use its nouns. The
scene is a real home (kitchen counter, dishwasher rack, sink, drawers, cabinet/oven/microwave
doors, toilet, bookshelf, laundry basket). Cut by the guide's tables:

- pick-and-place (mugs out of the dishwasher rack, fruit out of the bowl, fork/spoon/knife
  into the sink or dishwasher, book onto the shelf, apple/scissors/mug into the drawer, toy
  into the basket, garment into the laundry basket, can/bottle/tissue/chip bag into the trash
  bin, headphones onto the backrest, plate into the dishwasher, lid onto the container/pot):
  `grasp the X`, `move the X to the C`, `release the X in/on the C`, `return to home` only when
  the final park lasts about 1 s or more. When the instruction only says "lift it out" /
  "take the X out of the bowl" and the object is set down somewhere, the set-down is a
  `release the X on the table`; when it ends held in the air, end with `lift the X` (move with
  container null) instead of inventing a release. A "drop it inside" into the trash bin is
  still `release the X in the trash bin` (a deliberate letting-go over the target).
- pour: `grasp the bowl` (or the jug / the pitcher), `move the bowl to the box`, `pour the bowl
  into the box` while tilted, then `move the bowl to the table` + `release the bowl on the
  table` if it is put back, else `return to home`.
- wipe / scrub: `grasp the cloth`, `wipe the plate with the cloth` per pass (a circular wipe
  that never lifts off is ONE atom), `release the cloth on the sink` if put down; toilet brush:
  `grasp the toilet brush`, `scrub the toilet bowl with the toilet brush`, `release the toilet
  brush in the holder`.
- doors, drawers, lids, switches, knobs, levers, pumps (contact atoms, no move/release):
  `close the cabinet door` / `close the oven door` / `close the microwave door` / `close the
  drawer` / `close the laptop lid` as ONE atom from the approach to the lift-off when the
  gripper pushes; when the fingers actually shut on the handle first, `grasp the door handle`
  then `close the cabinet door` (or `open the oven door`); `press the toaster lever`, `press
  the flush handle`, `press the pump`, `turn on the light switch` / `turn off the light
  switch`, `turn the faucet handle`, `turn the burner knob` (or `the stove knob`, whichever the
  task says); `push the sanitizer bottle`. The proposal shows these as grasp + move because
  the fingers shut on the handle: rename, do not invent a release. Finish with `return to
  home` only if the park is about 1 s or longer.
- Object names come from the task string (`the gray mug`, `the yellow pear`, `the chip bag`,
  `the protein bar`, `the white stuffed toy`, `the cereal package`); containers likewise (`the
  tray`, `the trash bin` or `the bin` as the task says, `the dishwasher rack`, `the counter`,
  `the table`, `the drawer`, `the shelf`, `the sink`, `the holder`, `the basket`, `the
  container`, `the box`, `the laundry basket`). Both episodes of a task are in the same batch:
  keep their names identical.
- The three reviewed mistake events (ep005319 failed_close on the mango, ep005931 drop of the
  tissue, ep007303 slip of the mug) are in the headers and map automatically; their siblings
  need their own grade (rule 7).

### DROID CLVR / RAIL (Franka, 15 Hz, several parents per episode, `left_external` + `wrist`)

The parents are this round's reviewed subtask spans (recovery-first selection). Typical
sequence: `reach toward the X` (approach only, no close inside) -> `attempt to grasp the X`
(quality 1-2, carries the reviewed failed_close / slip / drop / knock events) -> `grasp the X
and put it in the C` (the recovery, quality 3-4). Cut them so that:
- an approach-only parent is ONE `grasp the X` atom (grasp absorbs the approach; the close
  happens in the next parent) - never a `move`;
- an attempt parent is ONE `grasp the X` atom holding the events (quality null); if the
  frames show the object was carried and lost inside it, `grasp the X` + `move the X to the C`
  with the loss inside the move (rule 3 of the brief);
- a recovery parent starts with `grasp the X` (its close is inside the parent unless the
  parent starts with the fingers already shut - then start with `move the X to the C`), then
  `move` / `release`, `return to home` only for a final park of about 1 s or more;
- multi-step parents (8691 bottles into the tray, 9699 cloth hang / pull off / hang again,
  29857 plushies, 29854 plushies + towel, 26671 cloth + rabbit) get one cycle per object
  handled, named from the task string; the towel/cloth work follows the guide's wipe / fold /
  hang rows.
Names: `the pen`, `the yellow cup`, `the bottle`, `the tray`, `the clamshell`, `the marker`,
`the green cup`, `the yellow mug`, `the cloth`, `the black stand`, `the lid`, `the clear
container`, `the screwdriver`, `the purple mat`, `the glass barrier`, `the pot`, `the green
plush`, `the silver pot`, `the blue bottle`, `the keyboard`, `the towel`, `the counter`, `the
stove`, `the pink plush`, `the turtle plush`, `the leopard plush`, `the orange plush`, `the
rabbit`, `the drawer`, `the doll`, `the pot lid`, `the orange arch block`, `the pale block`,
`the green plushie`, `the orange plate`.
