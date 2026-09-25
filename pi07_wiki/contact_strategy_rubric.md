# Annotation rubric: contact strategy per step (plan 2026-09-24; nothing annotated yet)

Companion to `precision_rubric.md` (fourth channel) and `annotation_rubric.md` (quality +
mistakes). Fifth metadata channel. Vocabulary evidence: `migration/contact_strategy_2026-09-24/`
(`vocab.md`, candidate sheets, per-element example sheets, text-rule dry run).

## Name

| where | name | why |
|---|---|---|
| documents, conversation | **contact strategy** | it covers pushes, presses and releases, not only grasps |
| code, storage, batch keys | **`contact`** | one word like `quality`, `speed`, `precision`: `meta/contact.parquet`, `contact_atoms.jsonl`, `metadata_contact`, `CONTACT_VOCAB` |
| prompt clause | **`The contact is <phrase>.`** | a fixed frame with one slot, like the other four; the prefix ` The contact is` is unique among clause prefixes, so probes and steering find and swap it the way they do the numbers |

Rendered after speed and precision: `... The speed is 3 of 5. The precision is 4 of 5. The
contact is a rim pinch. Given these, ...`. Steps without a contact render **`The contact is
not applicable.`** (a move, a return to home): the model learns "no contact" as a value,
not as a missing sentence. The clause is omitted only where no label exists at all (a root
annotated before this channel, the -1 sentinel), the way a -1 quality is omitted.

Considered and not chosen: verb sentences per element ("The robot pinches the cup from
above."). They read more naturally but have no fixed prefix (they collide with "The robot
is" and "The robot made"), so every consumer that locates clauses by prefix and every
steering sweep would need a per-element table. The slot phrases below are plain English
already.

## What the label means

> **contact = how the gripper, or the object it holds, engages the thing the step acts
> on.** One value per step. `na` ("not applicable") when the step neither makes nor
> breaks a contact and does no contact work (move / carry, return to home, a held object
> lifted or turned in the hand).

Rules:

- A step that **makes** a contact is labelled by that contact (grasp, push, press).
- A step that **breaks** a contact is labelled by how it lets go (set-down, drop).
- A step that works **through a held object** is labelled by what the object does
  (insert, pour, tool drag).
- When one step makes and breaks contact (fold the towel over), the **making** wins: the
  label is informative before the close, and the wrist camera shows the grip after it.
- Read from the object and the gripper at the commit. Never from how well it went
  (quality), how quickly (speed) or how tight the target is (precision).

Unlike precision, contact is a **choice** the operator makes before the observation
reveals it, so at inference it is written by the user, like the subtask (see "Inference").

## Vocabulary (14 elements + not applicable)

`code` is the integer stored per frame in the buffer; `phrase` fills the clause slot.
Approach angles are the ReBot FK angle between the gripper axis and straight down
(0 = top-down, 90 = horizontal); on other robots read the same thing visually.

| code | element | phrase | definition | wrist read at the commit |
|---|---|---|---|---|
| 0 | top-pinch | a top pinch | pads close from above on a rigid object's sides or top, approach < 45 | object between the pads seen from above, table behind it |
| 1 | side-pinch | a side pinch | pads close on the body from the side, approach >= 45 (wrap on a cylinder, face pinch on a book) | the object's side wall fills the frame, table at the edge or absent |
| 2 | rim-pinch | a rim pinch | one pad inside a mouth or ring, one outside its wall, from above | mouth or ring under the pads, interior visible |
| 3 | handle-grasp | a handle grasp | pads close on a handle, bar, knob, tab or lever attached to a larger body | the handle between the pads, the body beyond it |
| 4 | cloth-pinch | a cloth pinch | pads close on a fold, edge, corner or bunch of a deformable or thin sheet | fabric fills the frame, a fold or edge captured |
| 5 | push | a push | sustained lateral contact; the object slides on the surface or swings on a hinge | gripper tip (usually closed) against the object's side while it moves |
| 6 | press | a press | contact along the normal on a control face; the object does not translate | gripper tip landing on the face, then leaving |
| 7 | strike | a strike | impulsive contact by the gripper or a held tool; the gripper keeps moving | fast pass, object gone in the next frame |
| 8 | pry | a pry | closed gripper wedged under an edge and lifted | pads sliding under a flap, the flap rising |
| 9 | set-down | a set-down | held object lowered until it rests on a surface, then released | object touching the surface before the pads open |
| 10 | drop | a drop | pads open while the object is unsupported over the target | air or the container interior under the object when the pads open |
| 11 | insert | an insertion | held object guided into a fitted opening or over a peg; the fit does the last centimetres | the opening around the object, the object entering it |
| 12 | tilt-pour | a pour | held container rotated so its contents leave it | container tilting, contents moving |
| 13 | tool-drag | a tool drag | held object pressed on a surface and moved across it | rag / brush / spoon under the pads, surface sliding past |
| 14 | na | not applicable | no contact made, broken or worked | move, return to home, FMB lift and in-hand rotate, tongs |
| -1 | (unlabelled) | (clause omitted) | no row covers the frame: a root annotated before this channel | never written by the annotator |

Full definitions, corpus examples per element and the evidence behind the choices:
`migration/contact_strategy_2026-09-24/vocab.md` and `sheets/examples/<element>.jpg`.

### Not contact

- **Approach angle beyond top / side.** Computable by FK from proprio on every ReBot
  root; never annotated. A cloth pinch of a hanging shirt is a note, not an element.
- **Pad orientation across / along an elongated object** (bits: wrist roll spreads
  evenly 0-80 deg, no lobes). Not in the vocabulary; the roll-mod-180 canonicalisation
  is the zero-annotation route if bits need it.
- **Sub-techniques inside an element**: wad vs corner cloth pinch, body vs neck on a
  bottle, fingertip vs pad pinch, open-on-arrival vs lower-then-open drops, straight vs
  twist insertion, tip vs body push. Measured 2026-09-24 and left out on the user's
  decision: mostly object-determined or continuous, and the 14 elements are enough.
- **Grasp point, release height, carry height, effort, tempo**: other channels or
  measurements.
- **How the demonstration went**: a failed close is a mistake row and a quality value;
  its contact label is still the mode the operator attempted.

## Units and storage

| half | unit | storage |
|---|---|---|
| ReBot | one value per semantic segment, constant across it | `meta/contact.parquet`, rows 1:1 with `episode_metadata.parquet` (mirrors `speed_hybrid_v1.parquet`) |
| diverse | one value per reviewed atom | `contact_atoms.jsonl` keyed by `(episode_id, atom_index)` (mirrors `speed_atoms_hybrid_v1.jsonl`) |

Columns: `episode_index, segment_index, from_index, to_index, subtask, contact, code,
prior, image_read, note` (`prior` = the text-rule element, `image_read` = whether a vision
read changed or confirmed it). `meta/contact_info.json` / sidecar info: vocabulary version,
the phrase table, the rules version, annotator, the definition and the "not contact" list
verbatim. Written to **new roots**, never in place.

Per-frame materialisation: `ReplayBuffer.materialize_metadata` gains `contact_rows` and
fills a bf16 column `metadata_contact` with the code (14 for `na`, -1 only where no row
covers the frame), broadcast over `from_index:to_index` like speed. Small integers are
exact in bf16. The columns are materialised from parquet at load, after the cache is
read, so **no buffer cache rebuild**. The diverse collate reads `contact_atoms.jsonl`
through a `CONTACT_ATOMS_VIEW` constant beside `SPEED_ATOMS_VIEW` and looks the atom up by
native timestep, as speed does.

## Text rules: the prior for every class

The verb and noun fix the element for most classes. The rules live in
`migration/contact_strategy_2026-09-24/class_map.py` today and move next to the annotation
scripts as `contact_prior(subtask, verb=None, object=None, container=None)`; the diverse
atoms already carry parsed `verb`, `object`, `container`, `preposition`, `instrument`
fields, so the diverse half uses those instead of regexes. The rules are an annotation
aid only; they play no part at inference.

| verb | rule | image needed when |
|---|---|---|
| grasp | cloth noun -> cloth-pinch; handle / knob / can-with-handle noun -> handle-grasp; cup / mug / bowl / tape roll / lid / plate / disk -> rim-pinch; bottle / can / motor / book -> side-pinch; else top-pinch | rim nouns (body wrap or handle instead?), side nouns (top-of-cap instead? DROID and MolmoAct bottles are top-down), placeholder nouns ("the object", "the cooking ingredient", "the item") |
| release | in slot / shredder / vase / socket / board / rack / peg / stack / fixture -> insert; on table / pad / mat / plate / outline / tray / shelf / beside -> set-down; in basket / bin / box / trash / cup / pan / drawer -> drop | container releases where the object might be lowered to rest; every rack / peg release (hung = insert) |
| push, close, knock, hit | push (close = push the panel); knock / hit = strike; "push the X handle" -> handle-grasp | push via a handle: pads closed on it? (foosball yes, portafilter no) |
| press, turn on / off | press; "press X into Y" -> insert; knob / tap / faucet -> handle-grasp | knobs (pinch-and-twist vs push the lever) |
| rotate | knob / faucet -> handle-grasp; container on the table -> rim-pinch; body -> side-pinch; "the object" (FMB) -> na | the tabletop rotates (sanitizer bottle, mug, plate) |
| pour, water, tilt, dump | tilt-pour | never |
| wipe, scrub, stir | tool-drag | never |
| insert, plug, stack on peg / base | insert; stack on a roll -> set-down | never |
| fold, unfold, straighten, spread, flatten, pull (cloth, rope, sheet) | cloth-pinch; pull a lid / handle -> handle-grasp | an open-gripper drag is a push |
| open | handle-grasp (fridge, oven, drawer, container lid); tongs -> na | never (which part is pinched is a note) |
| lift | box lid (own) -> pry; the object (FMB) -> na; towel -> cloth-pinch; a lid -> top-pinch | lids (pinch vs pry) |
| hold, shake | handle-grasp; shake the hand -> side-pinch | never |
| move, return | na | never |

Dry run over today's corpus (segments for ReBot, atoms for diverse; every one of the
1,211 unique (source, subtask) classes mapped, 341 flagged for an image):

| element | own | external ReBot | diverse + FMB |
|---|---|---|---|
| top-pinch | 73 | 141 | 854 |
| side-pinch | 25 | 38 | 62 |
| rim-pinch | 20 | 33 | 185 |
| handle-grasp | 0 | 0 | 91 |
| cloth-pinch | 349 | 136 | 278 |
| push | 2 | 19 | 62 |
| press | 0 | 0 | 117 |
| strike | 0 | 31 | 0 |
| pry | 1 | 0 | 0 |
| set-down | 0 | 134 | 325 |
| drop | 399 | 138 | 411 |
| insert | 51 | 26 | 368 |
| tilt-pour | 0 | 6 | 40 |
| tool-drag | 0 | 0 | 65 |
| na | 560 | 375 | 1586 |

Own data has 8 of the 14 elements; press, strike, set-down, tilt-pour, tool-drag and
handle-grasp come only from the other halves. Coverage, not calibration: report the
per-half histogram after the pass; there is no target distribution.

## Procedure (one pass with precision)

Contact is a class property with the same exception list as precision, so it is
annotated class-first, in the **same pass** as precision: the same commit frames, one JSON
read that returns both labels.

1. **Group** by subtask text (ReBot ~60 strings, diverse 445 atom strings). The text rule
   gives the prior; it is never the verdict for a flagged class.
2. **One vision read per flagged class**, wrist + top at the commit frame:

   | verb | commit frame | why |
   |---|---|---|
   | grasp | `to-15` | pads on the object, starting to lift |
   | release | the frame the gripper opens (read off the gripper channel of the state) | the only frame that tells a drop from a set-down from an insertion |
   | push, press, strike, pry, pour, drag | `from+15` and `to-15` | contact onset and what the object did |
   | everything else | `to-15` | the end pose |

   Precision reads the same frames (its release read moves from `from+15` to the opening
   frame; `precision_rubric.md` updated), so one pair of images per class serves both.

   Evidence-first JSON, so the label conditions on what was articulated:
   `gripper_closes_on_object` (yes / no), `part_touched` (top / side / rim / handle /
   fold / none), `object_state_at_open` (resting / unsupported / fitted / n.a.),
   `held_object_action` (none / insert / pour / drag / strike), then `contact`, `note`,
   followed by precision's fields (`object_vs_pads`, `target_opening_vs_object`,
   `orientation_pin`, `precision`).
3. **Unflagged classes** take the prior with `image_read = false`. Classes the read marks
   as mixed go **per segment** (same read on each segment). Expected on today's data:
   none in the own half (all 399 own container releases are drops: the end link opens at
   arrival height over the basket, 37 cm, and never rests an object; the bin and bit
   releases lower first but still open unsupported), a handful in MolmoAct (mug rim vs
   handle vs body, tabletop rotates).
4. **Verify.** A random sample of segments per class gets the same read; a class whose
   sample disagrees with the class value goes whole to step 3.
5. **Consistency.** Within one dataset the same (verb, object, target) triple maps to one
   element unless a note explains the scene difference. Group the output by subtask text
   and list groups with more than one element.

| scheme | images |
|---|---|
| every segment, two cameras | ~14,000 |
| flagged classes + sampled checks, shared with precision | ~1,500 to 2,500 in total for both channels |

## Inference

Contact is a **wish** at rollout: the operator chose it before the observation revealed
it, so the policy cannot read it off the image and the user commands it.

- **The user writes the contact strategy the way subtasks are written today**: one of the
  14 phrases, or `not applicable`, entered alongside the subtask (eval console / inference
  config) and held until changed. Nothing is inferred automatically; the text rules above
  are for annotation only.
- `not applicable` is written for steps without a contact (move, return), so the rendered
  prompt always carries the clause, as in training.
- Quality, mistake and speed stay constants (5, none, 5). `metadata_dropout` stays 0.0.

Acceptance test, before training on it: the lexical-prior gate. On the current
checkpoint, sweep the clause over the 14 phrases on grasp anchors (own bottle, cup, sock;
`n_seeds >= 3`) and measure how far the chunk moves per phrase against the no-clause
chunk. After training: the steering ramp on bottle anchors, side-pinch -> top-pinch,
read out as the FK approach angle of the predicted chunk's end pose. Never scored against
a single demonstration.

## Code touch points (no behaviour change until a root carries the table)

- `datasets/contact_vocab.py` (new): `CONTACT_VOCAB` (slug, code, phrase, definition,
  including code 14 `not applicable`), vocabulary version; `contact_prior(...)` beside the
  annotation scripts.
- `policies/molmoact2/processor_molmoact2.py`: `metadata_clause += f" The contact is {phrase}."`
  when `metadata["contact"] >= 0` (code 14 renders `not applicable`); the batch-metadata
  builder reads `metadata_contact` with the -1 sentinel like speed.
- `rl/buffer.py` `materialize_metadata(..., contact_rows=None)`; `rl/offline_dataset_utils.py`
  `load_metadata_rows` reads `meta/contact.parquet` when present (`REBOT_CONTACT_TABLE`),
  absent on older roots means the clause is omitted, as speed was on pre-speed buffers.
- `datasets/diverse_corpus.py` `CONTACT_ATOMS_VIEW` + `contact_atoms()`; the actor
  collate copies the code per anchor like `speed_atom`.
- `probes/adapters/molmoact2.py` `_PROMPT_CLAUSES += ("metadata_contact", " The contact is")`;
  `probes/metadata_steering.py`: a categorical sweep next to the numeric ramp;
  `scripts/diverse_smoke.py`: a `_CONTACT_CLAUSE` regex.
- Inference: a `contact` field next to the subtask in the inference config and on the
  eval console (the 14 phrases + `not applicable`), rendered into the clause each step.

## Order of work

1. Confirm the name and the clause (this page).
2. Code the vocabulary module, clause, storage and probe hooks; run the smoke check with
   an empty table (clause absent, prompts byte-identical to today).
3. Annotation pass for precision + contact together, per the procedure above; new roots
   (`-annotated-v(N+1)`), per-half histograms for both channels.
4. Lexical-prior gate on the current checkpoint.
5. Train; steering ramp on bottle and cup anchors; then the hardware check with the
   override on the bottle task.

**Nothing is annotated until the user says go.**
