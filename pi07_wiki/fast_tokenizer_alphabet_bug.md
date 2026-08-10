# The FAST tokenizer silently deletes DCT coefficients

**Status: FIXED 2026-08-08** (§9). Affected **4.2% of training chunks** under the live
`action_encoding: anchor` configuration; 29% had `action_encoding` been `absolute`.
§1–§8 document the bug as found.

The FAST action tokenizer cannot encode certain quantized-DCT values. It has no UNK
token, so those characters are **dropped**, not clamped. A dropped coefficient shifts
every later value in the flat array, so the decoder assigns them to the wrong
`(frequency, joint)` cell and the rest of the chunk is scrambled. Every observed
deletion hits $k=0$ — the DC coefficient — and because DC occupies the first 7 slots,
a deletion misassigns essentially the whole chunk.

Two scripts:

| Script | Purpose |
|---|---|
| [`fast_alphabet_verify.py`](../../fast_alphabet_verify.py) | 24 pass/fail checks: exact failing set, deletion vs substitution, vocabulary forensics, and a **causal reproduction** that re-runs `fit`'s trainer call and reproduces both failure modes on demand |
| [`fast_alphabet_diagnostic.py`](../../fast_alphabet_diagnostic.py) | Prevalence and consequence on the real corpus, with plots |

---

## 1. Headline numbers

Reproduces the deployed encode path: anchor encode $a_{t:t+30} - s_t$
(`AnchorEncodeStep`), per-timestep $(30,7)$ q01/q99 from
`action_stats_anchor_rebot-annot-v2.pt`, clamp to $[-1,1]$
(`MolmoAct2ClampNormalizedProcessorStep`), then the FAST processor. Chunks never cross
an episode boundary. 9701 chunks over all three roots in `dataset.sources`.

| Quantity | Value |
|---|---|
| Chunks containing an unencodable coefficient | **406 / 9701 = 4.19%** |
| …of which the deletion hits $k=0$ (DC) | **406 / 406 (all of them)** |
| Median reconstruction RMSE, intact chunks | **0.0133** |
| Median reconstruction RMSE, affected chunks | **0.326** |
| Prevalence by source | shirts_bin **4.8%**, two_container **4.1%**, socks_basket **3.9%** |
| Chunks hitting the $[-1,1]$ clamp | 0.5% |

> **Correction.** Two earlier figures in this document were wrong because they were
> measured on *absolute* actions. The live config is `action_encoding: anchor`
> ([`config_rl.yaml:248`](../../config_rl.yaml#L248)), so the tokenizer sees deltas from
> the current proprioceptive state, normalized with per-timestep anchor quantiles — not
> absolute joint positions with pooled per-dim quantiles. Anchor deltas cluster near
> zero, far from the hole bands, which is why the true rate is **4.19%**, not the 29.3%
> reported earlier.

### The absolute-encoding counterfactual

Running the same scan with `--encoding absolute` (same chunks, same tokenizer):

| | anchor (live) | absolute |
|---|---|---|
| Chunks corrupted | **4.19%** | **29.2%** |
| Median RMSE, affected | 0.326 | 0.757 |

Anchor encoding is accidentally protective here. That is worth knowing before anyone
flips `action_encoding` back to `absolute` — it would multiply this bug by seven.

---

## 2. The alphabet holes

![alphabet map](../../outputs/fast_alphabet/01_alphabet_map.png)

Exactly, verified context-independently across six embeddings (alone, mid-string,
prefix, suffix, runs, alternating) over all 2048 bins:

$$c \in \{-52,\, -46,\, -43,\, -41,\, -36,\, -33,\, -28\}\;\cup\;\{c \ge +67\}
\qquad\text{cannot be encoded}$$

(bins $b = c + 55 \in \{3, 9, 12, 14, 19, 22, 27\} \cup \{b \ge 122\}$.)

The seven low holes are **isolated single values** — no two adjacent, minimum gap 2.
That is what makes the fix in §7 cheap.

The failure is deletion, not substitution: $k$ bad characters shorten the string by
exactly $k$, the surviving content is the good characters alone, and a lone bad
character encodes to `[]`.

### Why they exist — two separate causes

The tokenizer has `unk_token: null`, no normalizer, no added tokens, and a **ByteLevel**
pre-tokenizer. ByteLevel rewrites input into GPT-2 byte-level characters *before* the
BPE model sees it: bytes in $[33,126] \cup [161,172] \cup [174,255]$ map to themselves,
all others (including every control byte) map to `U+0100 + n`. A byte-level character
absent from the vocabulary produces **no token at all**.

Checking the vocabulary against that mapping predicts the failing set with **zero
symmetric difference**.

**Cause 1 — the ceiling at $c \ge 67$ is by construction.**
`UniversalActionProcessor.fit`
([`processing_action_tokenizer.py:99`](../../outputs/MolmoAct-FAST-tokenizer/processing_action_tokenizer.py#L99))
builds `initial_alphabet = [chr(i) for i in range(max_token - min_token + 1)]` — an
alphabet spanning exactly the coefficient range *observed while fitting*. The shipped
tokenizer encodes bin 121 but not 122, so $\text{max\_token} = +66$. Nothing was dropped
by the trainer; coefficients above $+66$ were simply never in the alphabet.

**Cause 2 — the seven interior holes are a raw-vs-byte-level mismatch in `fit`.**
`initial_alphabet` is given **raw codepoints** `chr(0)…chr(121)`, but the model's units
are byte-level characters. For bins 33–121 the two coincide (printable ASCII maps to
itself), so those entries work. For bins 0–32 they do not: `chr(3)` reaches the model as
`U+0103` while `initial_alphabet` inserted `U+0003`. All 122 raw codepoints are vocabulary
keys, and the 33 control ones are **dead entries the encoder can never emit**. The low
bins survive only where the byte-level form happened to occur in the fitting data; seven
did not.

**Causal reproduction (H6 in the verify script).** Re-running the exact `BpeTrainer` call
from `fit` on synthetic data that deliberately omits one control-range bin and one
printable-range bin:

- omitted **control-range** bin 5 → **unencodable** (initial_alphabet ineffective)
- omitted **printable-range** bin 50 → **still encodable** (initial_alphabet worked)
- bin 122, just past the fitted range → **unencodable** (cause 1)

That is the mechanism proven rather than inferred.

So the low range is intact *by luck of the data*, not by design — 26 of bins 0–32 are
present, 7 are missing. A different corpus leaves a different set of holes.

---

## 3. Why the DC coefficient

$C_{0,t} = \sqrt{1/T}$ for every $t$, so DC is the chunk mean scaled:

$$c_{0,d} \;=\; s\sqrt{T}\,\bar a_{d} \;=\; 54.77\,\bar a_{d},
\qquad \bar a_d = \tfrac{1}{T}\textstyle\sum_t a_{t,d}$$

DC spans exactly $[-55, +55]$: it reaches the seven negative holes but never the
$c \ge 67$ region. AC coefficients satisfy $|c_{k\ge1}| \le s\sqrt{2/T}\cdot T = 77.5$ so
they *can* exceed 67, but it takes a near-square-wave at that exact frequency — under
anchor encoding it never happens in this corpus (0 of 406).

The seven holes map back to forbidden bands of chunk-mean normalized action, each about
$\pm 0.009$ wide, at $\bar a_d \approx -0.95, -0.84, -0.79, -0.75, -0.66, -0.60, -0.51$.

![hole bands](../../outputs/fast_alphabet/02_hole_bands.png)

Under anchor encoding the chunk means concentrate near zero, so most of the density sits
clear of the bands — this is the whole reason the rate is 4.2% rather than 29%. The
residual hits come from joints with a wide low-side tail in their delta distribution:

| joint | affected chunks |
|---|---|
| elbow_flex | 177 |
| shoulder_lift | 170 |
| shoulder_pan | 35 |
| wrist_roll | 20 |
| wrist_yaw | 10 |
| gripper | 10 |
| wrist_flex | 6 |

---

## 4. The mechanism

![slot shift](../../outputs/fast_alphabet/04_slot_shift.png)

The flat array is **frequency-major**, $j = 7k + d$, so DC occupies slots 0–6. Deleting
slot $j$ shifts every later slot down by one, and the decoder — which reads slot $j$ as
$(k, d) = (j \,\text{div}\, 7,\; j \bmod 7)$ — assigns all of them to the wrong cell.
Every joint after the deletion inherits its neighbour's offset. Because DC lives in the
first 7 slots, a DC deletion misassigns **209 of 210 slots**.

The tokenizer's own `decode` never gets that far: $209 \bmod 7 \ne 0$, so
`decoded_dct_coeff.reshape(-1, action_dim)` raises into the bare `except` in
`UniversalActionProcessor.decode` and returns **zeros**.

---

## 5. What it does to the trajectory

![corruption example](../../outputs/fast_alphabet/03_corruption_example.png)

Blue is the FAST round-trip with no deletion — it tracks ground truth to quantization
precision. Red is what the token sequence actually encodes. In the worked example
elbow_flex goes from $-0.8$ to $+0.5$ and wrist_yaw flips sign. The per-chunk damage is
total even though the population rate is low.

![error distribution](../../outputs/fast_alphabet/05_error_distribution.png)

The two error populations are disjoint — this is not a tail that averaging absorbs, it
is a small subset of chunks whose targets are entirely wrong.

![prevalence](../../outputs/fast_alphabet/06_prevalence.png)

---

## 6. What it breaks

**Training — yes, now, at 4.2%.** Under `action_mode: both` the FAST head is KI-style:
it exists to shape the shared backbone. So roughly one chunk in 24 teaches the backbone
a wrong prompt→action association. Small enough that it will not dominate the loss;
large enough that it is pure label noise with no upside, and it is trivially fixable.

**Inference — no.** `inference_action_mode: continuous`, so the FAST decode path is
unused. Flipping to discrete would trip the length guard at
[`modeling_molmoact2.py:2444`](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L2444).
This is the encode-side twin of the decode-side zero-return bug already fixed there.

**If `action_encoding` ever goes back to `absolute` — 29%.** See §1.

---

## 7. Fix options

**(B) Snap to the nearest encodable value — APPLIED, see §9.** The seven low holes are
isolated, so the nearest encodable value is always exactly one coefficient step away.
That offsets the affected joint's mean position by

$$\Delta \bar a_d \;=\; \frac{1}{s\sqrt{T}} \;=\; \frac{1}{10\sqrt{30}} \;=\; 0.0183$$

uniformly across the chunk — **comparable to the 0.0133 median quantization error, not
below it**. Spread over 7 joints that is $0.0183/\sqrt{7} = 0.0069$ of chunk RMSE, added
in quadrature to what is already there. It buys the removal of a 0.326 error on the same
chunks. Preserves the coefficient count, the slot alignment, the vocabulary, and the
pretrained FAST head.

**(C) Assert representability at encode time — do this regardless.** Turns silent
corruption into a loud failure. This matters more than it first appears: per §2 the low
bins are present only because the fitting data happened to contain them, and per §1 the
rate swings 7x with an encoding change. A normalization, horizon, or embodiment change
can open new holes with no warning.

**(D) Patch the vocabulary — the only zero-error fix.** 33 vocabulary entries are dead
control codepoints the encoder can never emit (§2, cause 2). Binding the seven missing
byte-level characters to seven of those dead ids restores exact representation without
disturbing any *live* token id, so the pretrained head's embeddings stay valid. Costs an
edit to `tokenizer.json` (which lives in `outputs/`, so it is untracked and a re-download
clobbers it) and seven effectively untrained embeddings. Since (B)'s error turned out to
be comparable to quantization noise rather than far below it, this is a real alternative —
but §9 shows (B) already flattens the whole error distribution, so it is not needed.

**(A) Refit the tokenizer on rebot data — only if refitting anyway.** Correct in
principle and fixes the BPE merges to the real distribution too, but it changes what each
bin id *means*, scrambling whatever the pretrained MolmoAct2 FAST head learned about the
`<action_NNNN>` embeddings. **If you do refit, fix `fit` first** — as written it
reproduces cause 2 on any corpus, because it passes raw codepoints to `initial_alphabet`
while the model consumes byte-level characters. Pass `bytes_to_unicode()` forms instead,
and widen the alphabet past the observed range so cause 1 does not recur either.

---

## 8. Related figure retired: the "31.2% clamp"

`basis_roundtrip.py` was **not** at fault — `--encoding anchor` and
`--stats outputs/stats/action_stats_anchor_rebot-annot-v2.pt` are both its defaults, and
it hard-errors unless the quantiles are per-timestep $(H, D)$. Re-running its own
normalization path:

| encoding | stats | clamped values | clamped chunks |
|---|---|---|---|
| **anchor** | **anchor `.pt` (default)** | **0.02%** | **0.55%** |
| anchor | `meta/stats.json` fallback | 21.9% | 89.2% |
| absolute | `meta/stats.json` fallback | 13.5% | 66.1% |

So the clamp is a **non-issue on the deployed path** and 31.2% came from a stats
mismatch, not from the pipeline. Retire that number.

---

## 9. Fix applied

Option (B), in the seam we own rather than the vendored tokenizer:
`_tokenize_discrete_action` in
[`processor_molmoact2.py`](../src/lerobot/policies/molmoact2/processor_molmoact2.py) —
the single call site — now performs the FAST encode explicitly (DCT, quantize, **snap via
`_encodable_bin_map`**, BPE) and **asserts the span decodes to exactly $T \times D$
coefficients**. The alphabet probe is cached on the processor. Tests:
`lerobot/tests/policies/test_molmoact2_fast_alphabet.py` (7 checks, including
byte-identical output against the stock tokenizer on hole-free chunks).

Verified over the same 9701 chunks through the patched path:

| | before | after |
|---|---|---|
| Chunks with a short span | 406 (4.19%) | **0** |
| Median chunk RMSE | 0.01327 | 0.01334 |
| p99 chunk RMSE | 0.32 | 0.01630 |
| **Max chunk RMSE** | **0.326** | **0.01938** |

The median moves by 0.5% and the tail disappears entirely — the worst chunk in the whole
corpus is now barely above the median. The diagnostic script still reports the bug because
it characterizes the *tokenizer*, which is unchanged; the fix lives in the caller.

Still open:

- Re-measure the val/train FAST CE ratio; the v6 memorization conclusion was drawn with
  4.2% corrupted targets in the mix. Small, but the number may move.
- The slot-tiling invariant $\sum_i L[y_i] = 210$ now holds, which unblocked the
  FAST-logit auxiliary loss. That loss was subsequently built — see
  [05 §2.2](05_training.md) for what shipped. Its
  [design note is archived and its numbers retracted](archive/fast_soft_decode_auxiliary.md);
  the tiling invariant itself is independently re-checked by
  [`fast_soft_decode_probe.py`](../../fast_soft_decode_probe.py) (T2).
- **No cache rebuild needed** (checked): `buffer_cache-rebot-annot-v2/*/` holds
  `actions.bin` — raw actions, not packed labels. The processor runs per batch, so the fix
  applies to existing caches on the next run.
