# Frame-level speed label: alternatives measured

> **Status:** decided, 2026-10-26. The adopted rule (round + minimum run of 1 s,
> `hybrid_motion_speed.frame_runs`) is the as-built one in
> [`design/memory_prompts.md`](../design/memory_prompts.md) (speed method 4). This note keeps the
> numbers behind the choice.

Rule, for reference: frame f carries the trace's f -> f+1 score (no finite score = 3); label =
clip(round(score), 1, 5); then, shortest first, every run under `MIN_RUN_SECONDS = 1.0` (one action
chunk) takes the label of the neighbouring run closer to its mean score.

## ReBot, 14 roots, 913,074 frames

"far" = frames whose label is more than one bucket from their own score.

| label | changes/s | median run | frames in runs < 1 s | far |
|---|---|---|---|---|
| per segment (old) | 0.070 | 8.2 s | 0.0 % | 23.7 % |
| rounded trace (bare) | 0.766 | 0.73 s | 21.2 % | 0.0 % |
| smooth 1 s + round | 0.625 | 0.97 s | 17.9 % | 0.1 % |
| smooth 2 s + round | 0.467 | 1.37 s | 11.4 % | 2.0 % |
| smooth 3 s + round | 0.366 | 1.80 s | 6.0 % | 5.2 % |
| hysteresis 0.2 | 0.500 | 1.03 s | 10.9 % | 0.0 % |
| hysteresis 0.3 | 0.432 | 1.10 s | 8.8 % | 0.0 % |
| **round + min run 1 s (adopted)** | **0.311** | **2.27 s** | **0** | **5.4 %** |
| hysteresis 0.2 + min run 1 s | 0.256 | 2.57 s | 0 | 3.3 % |
| round + min run 2 s | 0.169 | 4.23 s | 0 | 10.6 % |

Smoothing or hysteresis alone never removes sub-chunk runs; the minimum run does, with one parameter
that has a meaning (the chunk). Hysteresis on top buys 0.05 changes/s and 2 points of "far" for a
second parameter: not taken.

## Diverse training store

| store | label | changes/s | median run | far |
|---|---|---|---|---|
| corpus | per atom (old) | 0.109 | 4.50 s | 22.4 % |
| corpus | rounded trace | 1.102 | 0.57 s | 0.0 % |
| corpus | frame runs (adopted) | 0.353 | 1.97 s | 7.5 % |
| fmb | per atom (old) | 0.143 | 4.25 s | 24.0 % |
| fmb | rounded trace | 0.996 | 0.70 s | 0.0 % |
| fmb | frame runs (adopted) | 0.383 | 2.00 s | 6.7 % |

## Effect on the trained label

50.8 % of ReBot frames and 51.2 % of diverse anchors changed speed label. Medians over a segment
pulled everything toward 3; per frame the ends fill, and speed 5 (what inference asks for) gains support.

| share of label 1..5 | per segment / atom | frame runs |
|---|---|---|
| ReBot frames | 14.2 / 33.1 / 25.8 / 15.3 / 11.5 % | 24.0 / 24.6 / 17.3 / 15.1 / 19.0 % |
| diverse anchors | 2.0 / 19.6 / 44.1 / 28.5 / 5.8 % | 6.3 / 26.8 / 25.3 / 25.3 / 16.3 % |

## Invariants

- No cache rebuild on a speed change: the ReBot memmap cache stores no metadata
  (`materialize_metadata` runs at startup) and the diverse cache neither keys on nor stores speed.
- Frames without a valid trace are each stretch's last frame only; they count as 3 and are absorbed
  by their run.
- A new corrected diverse store needs its own `speed_runs_hybrid_v1.jsonl`
  (`python -m lerobot.annotation.speed.hybrid_diverse runs <store> [<trace map>]`).
- The training store (`..._screen125_corrected_v4_tasks2`) has no `speed_hybrid_v1/` trace folder;
  its runs file was built through a trace map resolved over older stores and fix dirs. The runs file
  is self-contained; the traces are provenance only.
