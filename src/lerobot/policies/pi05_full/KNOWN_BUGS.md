# Known Bugs / Pending Fixes

---

## BOS token missing from subtask segment in `sample_actions` at inference

**File:** `modeling_pi05.py` → `sample_actions`
**Status:** Unverified impact — needs a model trained on real subtask labels to test
**Priority:** Fix before evaluating subtask-conditioned action quality

### What the bug is

During training, the subtask tokens fed into `embed_prefix` (inside `forward`) include a leading BOS token:

```
subtask_tokens = [<bos>, Subtask, :, grasp, red, truck, ;, \n, <eos>, <pad>, ...]
```

This is because the tokenizer prepends BOS when tokenizing the subtask string (confirmed by printing ground-truth labels in the CE loss — they all start with `<bos>`).

At inference, the flow is:
1. `generate_subtask_tokens` places a bare BOS token in the **subtask segment** of `embed_prefix` to seed generation (this was the BOS attention fix applied on 2026-03-30).
2. The autoregressive loop generates `[tok1, tok2, ...]` — the first real token onward.
3. `generate_subtask_tokens` returns `generated_subtask_tokens = [tok1, tok2, ...]` — **BOS is not included**.
4. `predict_action_chunk` passes these directly to `sample_actions`.
5. `sample_actions` calls `embed_prefix(subtask_tokens=[tok1, tok2, ...])`.

So the subtask segment seen by the action diffusion model is:
- **Training:** `[<bos>, tok1, tok2, ...]`
- **Inference:** `[tok1, tok2, ...]`

BOS is present during training but absent at inference.

### Why it matters

The action diffusion model (expert Gemma suffix) attends to the full prefix including the subtask segment. During training it always sees BOS at position 0 of the subtask segment. At inference it never does. This is a train/inference distribution mismatch for the action model, distinct from the earlier BOS attention mismatch that was already fixed.

### The fix

In `generate_subtask_tokens`, prepend BOS to the returned tensor before returning:

```python
# After the decoding loop, before returning:
bos_prefix = torch.full((bsize, 1), self._paligemma_tokenizer.bos_token_id,
                         dtype=torch.long, device=device)
bos_mask_prefix = torch.ones((bsize, 1), dtype=torch.bool, device=device)

generated_subtask_tokens = torch.cat([bos_prefix, generated_subtask_tokens], dim=1)
generated_subtask_masks = torch.cat([bos_mask_prefix, generated_subtask_masks], dim=1)
```

This ensures `sample_actions` receives `[<bos>, tok1, tok2, ...]`, matching the training distribution.

**Caveat:** Before applying this fix, verify that `max_decoding_steps` is sized to accommodate the extra BOS token, or the sequence will be 1 token longer than expected. Also check whether any downstream consumer of `generated_subtask_tokens` (e.g. the `[SUBTASK]` log in `inference_utils.py`) needs to strip the leading BOS before decoding.

### Related fixes already applied (2026-03-30)

1. **`hydrate_subtasks` bug** (`pi05_train_utils.py`): was doing `rows.iloc[0]["subtask"]` on a DataFrame whose subtask names are the index, not a column. Silent `KeyError` caused all subtask labels to be empty strings `""` during actor training. Fixed to `rows.iloc[0].name`.

2. **BOS attention mismatch in `generate_subtask_tokens`** (`modeling_pi05.py`): BOS was appended to the language token sequence (`tokens_in = cat([tokens, bos_token])`), placing it in the language attention segment. During training, BOS seeds subtask generation from the subtask segment. Fixed by passing `bos_token` as `subtask_tokens` to `embed_prefix` instead of appending to `tokens`.
