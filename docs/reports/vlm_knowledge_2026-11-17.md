# Knowledge benchmarks: are the mistakes the same? (2026-11-17)

Four models on MMLU-Redux (5,700 text questions), CV-Bench (2,638), EmbSpatial (3,640), a VQA v2 sample (2,000)
and our sim scenes (144 yes/no): Molmo2-ER (`er`, the original VLM), MolmoAct2 (`base`), tube 700 and internal 1200.
On the image sets every language model was also run on every vision encoder (16 cells, named language model /
vision encoder). One 378 px crop per image, as in our runs.

Everything below is computed from the saved answers by `lerobot/adhocs/vlm_knowledge/mistakes.py`. All tables and
example questions with images: [mistakes.md](../../../outputs/analysis/vlm_knowledge_2026-11-17/mistakes.md).
Accuracy with intervals: [summary.md](../../../outputs/analysis/vlm_knowledge_2026-11-17/summary.md).

A multiple-choice question counts as right when the right option letter is the most likely letter at the first
position (`letters`). VQA uses the answer decoded with action tokens forbidden (`masked`).

## Correction to the first numbers

The MMLU-Redux numbers I gave first used the decoded answer. base and tube 700 start a sentence ("Let", "The")
instead of a letter on about 550 questions, mostly maths, and those were counted wrong. With the `letters` reading:

| | er | base | tube 700 | internal |
|---|---|---|---|---|
| answers that are not a letter | 78 | 552 | 542 | 61 |
| decoded answer (reported before) | 70.5 | 63.7 | 56.9 | 55.3 |
| `letters` | 71.0 | 67.7 | 60.3 | 55.8 |

So er to base is -3.3 (not -6.8), base to tube 700 is -7.4, base to internal is -11.9. The image benchmarks do not
change (the models answer with a letter there).

## Short answers

1. The mistakes are largely the same questions. The error sets are close to nested: most of what er gets wrong,
   base gets wrong; most of what base gets wrong, ours get wrong; ours add more on top.
2. tube 700 and internal lose mostly the same questions against base, and on MMLU-Redux they pick the same wrong
   option 79% of the time (chance 33%).
3. The two fine-tuned vision encoders lose the same image questions, under either language model.
4. Most image questions lost against base come back with base's vision encoder. For internal a second part comes
   back only with base's language model, and internal prefers option A.
5. With our vision encoders the answers look like the model sees less: more "0", "nothing", "no", and present
   objects reported absent.

## 1. Text (MMLU-Redux)

Wrong-question overlap. "Unrelated" is the number both would get wrong if the mistakes were unrelated (product of
the two error rates). Questions differ in difficulty, so any two models overlap more than that; the er / base row
is the yardstick.

| A | B | A wrong | B wrong | both wrong | unrelated | A's wrong also wrong in B | same wrong option (chance 33%) |
|---|---|---|---|---|---|---|---|
| er | base | 1653 | 1842 | 1397 | 534 | 85% | 85% |
| base | tube 700 | 1842 | 2265 | 1526 | 732 | 83% | 80% |
| base | internal | 1842 | 2518 | 1488 | 814 | 81% | 72% |
| tube 700 | internal | 2265 | 2518 | 1896 | 1001 | 84% | 79% |

Of the 3,858 questions base gets right, tube 700 loses 739 and internal 1,030; 557 are lost by both (197 if
unrelated). In the other direction tube 700 gets 316 right that base gets wrong, internal 354. So the net loss
(-423, -676) is the difference of two larger numbers; a good part of the moved questions are ones the models are
unsure about.

By subject: tube 700 is below base in 54 of 57 subjects, internal in all 57. It is a broad loss, not a few subjects.

| | base | tube 700 | internal |
|---|---|---|---|
| largest drops: high_school_statistics | 68 | 53 | 44 |
| logical_fallacies | 79 | 66 | 57 |
| electrical_engineering | 66 | 56 | 45 |
| professional_law | 46 | 37 | 27 |
| professional_medicine | 68 | 40 | 53 |
| smallest: marketing | 90 | 87 | 86 |
| world_religions | 74 | 72 | 73 |
| computer_security | 74 | 72 | 70 |

Subjects that need working through a long question (statistics, logic, law, medicine cases) drop most; recall
subjects drop least. This is a reading of the table, not a test.

Questions that MMLU-Redux marks as faulty do not explain anything: on the 5,330 clean questions the order is the
same (73.0 / 69.6 / 61.7 / 57.1).

Option letters: internal answers A or B on 58% of the questions (right answers: 48%), and is right 64 / 60 / 50 / 50%
of the time when the right option is A / B / C / D. base is flat (69 / 68 / 68 / 66). tube 700 leans to C.

## 2. Images: which part loses the question

Accuracy of the 16 cells (rows language model, columns vision encoder):

| | CV-Bench | | | | EmbSpatial | | | | VQA v2 | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| language model | er | base | tube | int | er | base | tube | int | er | base | tube | int |
| er | 85.4 | 85.6 | 75.1 | 72.5 | 77.7 | 78.0 | 57.5 | 54.0 | 84.1 | 83.5 | 63.0 | 58.0 |
| base | 86.5 | 86.9 | 77.0 | 77.0 | 78.7 | 78.4 | 57.5 | 55.4 | 82.5 | 83.0 | 61.0 | 57.4 |
| tube 700 | 83.9 | 85.3 | 72.4 | 72.9 | 74.5 | 74.1 | 51.5 | 51.5 | 81.1 | 81.1 | 56.0 | 54.2 |
| internal | 82.9 | 84.2 | 73.7 | 72.0 | 62.6 | 63.5 | 44.3 | 42.6 | 78.6 | 78.8 | 56.9 | 51.1 |

The two fine-tuned vision encoders lose the same questions (base language model; right with base vision, wrong
with ours):

| | right with base vision | lost with tube 700 vision | lost with internal vision | lost with both | unrelated |
|---|---|---|---|---|---|
| CV-Bench | 2292 | 300 | 308 | 191 | 40 |
| EmbSpatial | 2853 | 960 | 1039 | 705 | 350 |
| VQA v2 | 1654 | 505 | 588 | 405 | 180 |

The same holds under the er language model (239 / 727 / 382 lost with both).

Where the questions that our full models lose against base come back when one part is swapped for base's:

| | model | lost | only with base vision | only with base language model | with either | with neither |
|---|---|---|---|---|---|---|
| CV-Bench | tube 700 | 428 | 192 | 25 | 156 | 55 |
| | internal | 468 | 123 | 50 | 231 | 64 |
| EmbSpatial | tube 700 | 1195 | 550 | 71 | 353 | 221 |
| | internal | 1496 | 414 | 237 | 449 | 396 |
| VQA v2 | tube 700 | 615 | 400 | 19 | 148 | 48 |
| | internal | 707 | 418 | 30 | 171 | 88 |

For tube 700, 76-89% of the lost questions come back with base vision, 6% or less come back only with base's
language model. For internal on EmbSpatial 237 come back only with the base language model and 396 need both;
that is the language-model part of internal's loss.

## 3. Images: what kind of question

CV-Bench:

| cell | Count (788) | Depth (600) | Distance (600) | Relation (650) |
|---|---|---|---|---|
| base / base | 70 | 96 | 92 | 94 |
| base / tube 700 | 54 | 85 | 85 | 90 |
| base / internal | 51 | 87 | 87 | 90 |
| tube 700 / base | 69 | 94 | 89 | 94 |
| internal / base | 71 | 91 | 87 | 92 |
| tube 700 / tube 700 | 49 | 78 | 81 | 87 |
| internal / internal | 58 | 76 | 80 | 78 |

EmbSpatial (chance 25):

| cell | above | under | close | far | left | right |
|---|---|---|---|---|---|---|
| base / base | 88 | 90 | 64 | 68 | 82 | 78 |
| base / tube 700 | 82 | 76 | 41 | 48 | 53 | 46 |
| base / internal | 79 | 72 | 45 | 50 | 47 | 41 |
| tube 700 / base | 89 | 92 | 58 | 67 | 74 | 65 |
| internal / base | 82 | 78 | 38 | 64 | 68 | 51 |
| tube 700 / tube 700 | 78 | 76 | 42 | 38 | 45 | 32 |
| internal / internal | 67 | 53 | 31 | 39 | 48 | 19 |

VQA v2:

| cell | number (247) | other (967) | yes/no (786) |
|---|---|---|---|
| base / base | 76 | 76 | 94 |
| base / tube 700 | 51 | 47 | 81 |
| base / internal | 44 | 43 | 79 |
| tube 700 / base | 72 | 73 | 93 |
| internal / base | 66 | 71 | 92 |
| tube 700 / tube 700 | 48 | 40 | 78 |
| internal / internal | 39 | 35 | 75 |

- Our vision encoders cost most on counting, left / right, close / far, and open questions ("other", "number").
  Above / under and CV-Bench Relation hold up best.
- Left / right with our vision under the base language model is 41-53, from 78-82.
- internal's language model with base vision is weak on close (38) and right (51). internal / internal on "right"
  is 19, below chance.
- VQA answers with our vision: "0" 62-88 times (base 27, humans 24); "no" on the 786 yes/no questions 419 and 451
  times under the base language model and 448 for tube 700 / tube 700 (base 394, humans 356; internal / internal
  396); more "white" and "nothing". tube 700 and internal give the same wrong answer on 46% of the questions both get wrong (base and
  ours: 21-28%).

## 4. internal prefers option A

EmbSpatial, share of answers that are A (right answers: 26%) and accuracy by the position of the right option:

| cell | answers A | right is A | right is B | right is C | right is D |
|---|---|---|---|---|---|
| base / base | 28% | 82 | 79 | 74 | 77 |
| base / internal | 32% | 65 | 56 | 47 | 53 |
| tube 700 / tube 700 | 30% | 58 | 49 | 54 | 44 |
| internal / base | 37% | 79 | 66 | 54 | 54 |
| internal / internal | 49% | 69 | 38 | 27 | 34 |

internal answers A on half the EmbSpatial questions and is near chance when the right option is not A. The
preference is there with base vision too (37%), so it sits in the language model and grows when the image is
harder to read. tube 700 does not have it. Part of internal's low EmbSpatial score is this preference and not
missing spatial knowledge; the benchmark cannot separate the two as run.

## 5. Sim scenes ("Is there a ... in this image?", 72 present, 72 absent)

| cell | present right | absent right |
|---|---|---|
| er / er | 94 | 96 |
| base / base | 100 | 83 |
| base / tube 700 | 96 | 83 |
| base / internal | 96 | 79 |
| tube 700 / base | 100 | 86 |
| internal / base | 100 | 83 |
| tube 700 / tube 700 | 74 | 89 |
| internal / internal | 79 | 81 |
| er / internal | 47 | 88 |

- base's mistakes are all "yes" to an absent object; tube 700's are mostly "no" to a present one. base and
  tube 700 share no wrong question.
- A present object is missed only when our vision meets a language model other than base (er: 47; ours: 74-79).
  base's language model reads our vision encoders almost as well as its own on these images (96).
- 144 questions, 18 images: small.

## 6. What kind of questions are lost, and are they related to robotics

All tables: [lost_questions.md](../../../outputs/analysis/vlm_knowledge_2026-11-17/lost_questions.md)
(`lerobot/adhocs/vlm_knowledge/lost_questions.py`).

### Text

Agents labelled 1,114 MMLU-Redux questions without knowing which were lost: the 557 lost by both runs and 557 of the
2,646 kept by both. Rubric: `outputs/analysis/vlm_knowledge_2026-11-17/question_types/RUBRIC.md`. Loss rate = lost
by both / (lost by both + kept by both).

| | lost by both | kept (sample of 557) | loss rate |
|---|---|---|---|
| all | 557 | 557 | 17% |
| one step of thought | 403 | 512 | 14% |
| several steps chained | 154 | 45 | 42% |
| recall of a fact | 289 | 406 | 13% |
| apply a rule to a case, one step | 81 | 84 | 17% |
| apply a rule to a case, several steps | 78 | 22 | 43% |
| calculate | 68 | 27 | 35% |
| logic, several steps | 34 | 11 | 39% |

- What the lost questions share is that they take several steps, in any subject: a law case, a patient, a
  calculation, a logic problem all lose about 40%. One-step questions lose 13-17% whatever the skill.
- Questions base gets right but Molmo2-ER gets wrong are lost 50% (tube 700) and 60% (internal), against 17% and
  24% for the rest. These are questions base was already unsure about.
- Loss grows with question length up to 120 words (15% to 33% for tube 700), then drops for the longest (20%).

Relation to a robot arm's world (2 = mechanics, geometry, control, circuits, programs for machines; 1 = other
physical world; 0 = none):

| | lost by both | kept (sample) | loss rate |
|---|---|---|---|
| 0 | 362 | 387 | 16% |
| 1 | 146 | 143 | 18% |
| 2 | 49 | 27 | 28% |
| 2, without calculation | 31 | 21 | 24% |
| 0, without calculation | 322 | 372 | 15% |

Robotics-related knowledge is not protected. It is lost somewhat more, mostly because those questions are
calculations; without calculation the difference (24% vs 15%) rests on 52 questions. Among the lost: Newton's
laws (4), projectile motion (2), force vectors, rotation, angular momentum, DC motors.

### Images (questions lost when base's language model reads our vision encoders)

- Counting more than one object. CV-Bench Count by the right number:

| right number | questions | base / base | base / tube 700 | base / internal |
|---|---|---|---|---|
| 1 | 466 | 96 | 84 | 82 |
| 2 | 167 | 41 | 12 | 12 |
| 3-4 | 100 | 29 | 10 | 5 |
| 5 or more | 54 | 11 | 2 | 0 |

  The VQA number questions show the same (right number 2: 86 to 50 / 40; 5 or more: 56 to 28 / 9). The answers
  move toward fewer objects.
- Naming things. VQA "what", "what is", "what kind of" lose 50-58%; yes/no forms ("is the", "does the", "is this
  a") lose 11-24%.
- Position relative to the viewer. EmbSpatial close / far / left / right lose 39-52%, above / under 12-24%.
- Not tied to the kind of scene or object. ai2thor (simulated kitchens, small objects) loses 39%, scannet 34-41%,
  mp3d 28-30%. Small handheld objects and furniture lose the same (left / right: 48 / 44 vs 49 / 42 accuracy).
  Nothing in the public sets is better kept because it looks like robot data.

### Our own scene (sim, 18 images)

Questions right, objects that are in the picture:

| object | questions | in training data | base / base | base / tube 700 | base / internal | tube 700 / tube 700 | internal / internal |
|---|---|---|---|---|---|---|---|
| basket | 18 | yes | 18 | 18 | 18 | 12 | 7 |
| pill bottle | 18 | yes | 18 | 17 | 18 | 15 | 18 |
| orange bottle | 6 | yes | 6 | 4 | 4 | 1 | 6 |
| robot arm | 18 | | 18 | 18 | 18 | 13 | 15 |
| banana | 6 | no | 6 | 6 | 5 | 6 | 5 |
| mug | 6 | no | 6 | 6 | 6 | 6 | 6 |

The misses of our full models are on the things they were trained to act on (basket, robot arm, orange bottle), not
on the objects they never saw. base's language model reading the same vision encoders finds them (18 of 18), and
our language models with base vision find them too. So the information is in the vision output; our own language
model does not say "yes" from it. Small set, one question wording.

### Reading

Fine-tuning did not trade general knowledge for robotics knowledge. What is lost is general: chains of several
steps in text, and counting, naming and viewer-relative position in images. Robot-related text knowledge and the
trained objects in our own scene are not better kept.

## What this says about the statements

- Statement 1 (the knowledge is still there): mostly. tube 700 keeps 81% of the MMLU-Redux questions base gets
  right, internal 73%, and their wrong answers are mostly the wrong answers base and er already give. The loss is
  real (-7.4, -11.9), broad across subjects, and shared between the two runs.
- Statement 2 (image answering is lost in the vision encoder): holds for tube 700 (76-89% of lost questions come
  back with base vision). For internal it needs a second clause: its language model also lost part (EmbSpatial,
  option A preference).

## Not checked

- Type labels come from one pass of agents (no second read); the 2 / 1 / 0 robotics scale is my rubric.
- Whether multi-step questions are lost because of the steps or because they are harder in general (they are
  also the ones Molmo2-ER fails more).
- Whether the option A preference of internal is tied to the probability it puts on action tokens.
- Whether shuffling the options changes internal's EmbSpatial score (would separate preference from knowledge).
- Cosine of each vision encoder's output to Molmo2-ER's.
