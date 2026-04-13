# Interactive Subtask Chat Interface for Inference

## Overview

A minimal extension of the standalone inference scripts that lets the operator type a subtask string during live inference. The model normally auto-generates its own subtask tokens via `generate_subtask_tokens`; this interface lets the human inject a typed string instead, consumed exactly once at the next action chunk generation. After that, the model's normal time-based cache resumes.

No special UI. The inference thread already prints `[SUBTASK]` whenever the active subtask changes (this is in `inference_utils.py` today). The user types into the same terminal.

---

## Verified Architecture (from reading the code)

### Cache fields — live directly on `policy` (PI05RLPolicy)
Initialized in `PI05FullPolicy.reset()` (modeling_pi05.py:1790), called by `PI05RLPolicy.__init__` at line 810:

```python
policy._cached_subtask_tokens   # Tensor | None, shape [1, max_len], long
policy._cached_subtask_masks    # Tensor | None, shape [1, max_len], bool
policy._last_subtask_time       # float | None
```

The time-based regeneration check in `predict_action_chunk` (rl_pi05.py:1095–1100) is:
```python
should_regenerate = (
    self._cached_subtask_tokens is None
    or self._last_subtask_time is None
    or interval <= 0
    or (current_time - self._last_subtask_time) >= interval
)
```
So setting both `_cached_subtask_tokens` and `_last_subtask_time` before calling `predict_action_chunk` is sufficient to skip `generate_subtask_tokens` for that cycle.

### Tokenizer — `policy.model._paligemma_tokenizer`
Initialized in `PI05Pytorch.__init__` (modeling_pi05.py:770):
```python
self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
    config.text_tokenizer_name, trust_remote_code=True,
    add_eos_token=True,   # ← EOS appended automatically
    add_bos_token=False,  # ← no BOS
)
```
Output token format from `generate_subtask_tokens`: `[first_content_token, ..., EOS, 0, 0, ...]` (right-padded zeros). This matches exactly what you get by calling this tokenizer with `padding="max_length"`.

### Existing subtask display — already in `inference_utils.py:267–285`
```python
tokenizer = policy.model._paligemma_tokenizer
valid_tokens = cached_tokens[0][cached_masks[0]]
subtask_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
if subtask_text != last_subtask_text:
    logger.info(f"[SUBTASK] inference_step={inference_step} | subtask: \"{subtask_text}\"")
    last_subtask_text = subtask_text
```
The interactive script reuses this exact pattern. Nothing new needed for display.

---

## Consumption Model

The override is a **one-shot signal**:

1. User types text → tokenized → stored as pending in shared state
2. On the **next inference cycle**, the worker pops the override (clears it atomically), injects the tokens into `policy._cached_subtask_tokens` + resets `policy._last_subtask_time`, then calls `predict_action_chunk` normally
3. `predict_action_chunk` sees a fresh cache, skips `generate_subtask_tokens`, calls `sample_actions` with the injected tokens
4. From here the model's normal time-based cache takes over — injected tokens persist until `subtask_regeneration_interval` expires

```
User types "pick up the cup"
        │
        ▼
shared_state.set_pending_override(tokens, masks, text)
        │
        ▼  (next inference cycle)
override = shared_state.pop_pending_override()   ← atomically consumed, now None
policy._cached_subtask_tokens = override.tokens.unsqueeze(0).to(device)
policy._cached_subtask_masks  = override.masks.unsqueeze(0).to(device)
policy._last_subtask_time     = time.time()
actions = policy.predict_action_chunk(batch)
    └─ should_regenerate = False  (cache is fresh)
       └─ sample_actions(..., injected_tokens)
# existing [SUBTASK] logger prints the injected text
        │
        ▼  (subsequent cycles, no new override pending)
override = shared_state.pop_pending_override()   → None
actions = policy.predict_action_chunk(batch)
    └─ uses cached tokens until interval expires, then auto-generates
```

---

## Shared State

Add three fields to `SharedState` (or subclass it):

```python
class SharedStateInteractive(SharedState):
    def __init__(self):
        super().__init__()
        self._pending_tokens: torch.Tensor | None = None   # [max_len] long
        self._pending_masks:  torch.Tensor | None = None   # [max_len] bool
        self._pending_text:   str | None = None

    def set_pending_override(self, tokens: torch.Tensor, masks: torch.Tensor, text: str):
        """Called by input thread."""
        with self.lock:
            self._pending_tokens = tokens
            self._pending_masks  = masks
            self._pending_text   = text

    def pop_pending_override(self):
        """Called by inference worker. Returns (tokens, masks, text) and clears, or None."""
        with self.lock:
            if self._pending_tokens is None:
                return None
            result = (self._pending_tokens, self._pending_masks, self._pending_text)
            self._pending_tokens = None
            self._pending_masks  = None
            self._pending_text   = None
            return result

    def clear_pending_override(self):
        """Called at episode boundary."""
        with self.lock:
            self._pending_tokens = None
            self._pending_masks  = None
            self._pending_text   = None
```

---

## Input Thread

```python
def terminal_input_worker(
    shared_state: SharedStateInteractive,
    policy,
    cfg,
    shutdown_event: threading.Event,
):
    """Daemon thread. Blocks on input(), tokenizes with policy.model._paligemma_tokenizer."""
    tokenizer = policy.model._paligemma_tokenizer
    max_len = cfg.policy.tokenizer_max_length
    print("[INTERACTIVE] Type a subtask and press Enter. Ctrl+C to stop.")

    while not shutdown_event.is_set():
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            continue

        encoding = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = encoding["input_ids"][0]            # [max_len] long — EOS added by tokenizer
        masks  = encoding["attention_mask"][0].bool() # [max_len] bool
        shared_state.set_pending_override(tokens, masks, text)
        print(f"[INTERACTIVE] Queued: '{text}' — will be used at next action generation")
```

Tokenization exactly matches `generate_subtask_tokens` output format: no BOS, EOS added by tokenizer, right-padded zeros.

---

## Modified Inference Worker

`get_actions_worker_interactive` is identical to the existing inference worker in `inference_pi05_async_utils.py` except for this block inserted **before** `predict_action_chunk`:

```python
# --- subtask override injection ---
override = shared_state.pop_pending_override()
if override is not None:
    tokens, masks, text = override
    policy._cached_subtask_tokens = tokens.unsqueeze(0).to(device)  # [1, max_len]
    policy._cached_subtask_masks  = masks.unsqueeze(0).to(device)
    policy._last_subtask_time     = time.time()
    # existing [SUBTASK] logger will print it after predict_action_chunk returns
# ----------------------------------

actions = policy.predict_action_chunk(batch, ...)
# [SUBTASK] logger already handles display (inference_utils.py:267-285 pattern)
```

That's it. The `[SUBTASK]` logger block is already inside `get_actions_worker` and is carried over verbatim into the clone — no new display code.

The inference worker already has a reset block (inference_utils.py:175–178):
```python
if shared_state.check_and_clear_reset():
    policy.reset()
    last_subtask_text = None
    continue
```
Add `shared_state.clear_pending_override()` here so a stale queued override doesn't survive an episode boundary:
```python
if shared_state.check_and_clear_reset():
    policy.reset()
    shared_state.clear_pending_override()   # ← new
    last_subtask_text = None
    continue
```

---

## Main Entry Point: `interactive_inference_pi05.py`

Clone of `inference_pi05_async.py` with:
1. `SharedState` → `SharedStateInteractive`
2. `terminal_input_worker` daemon thread added
3. `get_actions_worker` → `get_actions_worker_interactive`
4. `env_interaction_worker` unchanged

```python
threads = [
    Thread(target=terminal_input_worker,
           args=(shared_state, policy, cfg, shutdown_event),
           daemon=True, name="InputThread"),
    Thread(target=get_actions_worker_interactive,
           args=(policy, shared_state, action_queue, device, cfg),
           daemon=True, name="InferenceThread"),
    Thread(target=env_interaction_worker,
           args=(env, env_processor, action_processor, action_queue,
                 shared_state, teleop_device, cfg),   # teleop_device is a required arg
           name="EnvThread"),
]
```

---

## Files Created / Modified

| Action | File | Change |
|--------|------|--------|
| **Created** | `inference_utils_interactive.py` | `SharedStateInteractive`, `terminal_input_worker`, `get_actions_worker_interactive` |
| **Created** | `inference_pi05_async_interactive.py` | Entry point: 3 threads (input, inference, env). `env_interaction_worker` imported unchanged from `inference_utils.py` |
| **No change** | `rl_pi05.py` | Cache injection works via existing `_cached_subtask_tokens` / `_last_subtask_time` on `policy` |
| **No change** | `modeling_pi05.py` | `reset()` already initializes the cache fields |
| **No change** | `inference_utils.py` | `SharedState`, `env_interaction_worker`, subtask logger pattern all reused verbatim |

---

## Config Note

**`subtask_regeneration_interval` must be > 0.** When set to `0` the check `interval <= 0` forces `should_regenerate = True` on every cycle, which means `generate_subtask_tokens` always runs and immediately overwrites any injected tokens — injection is a silent no-op.

For interactive use set it to a value larger than the expected gap between user inputs, e.g. `"subtask_regeneration_interval": 30` in `config-hiserl.json`. The injected tokens then persist naturally until the interval expires, at which point the model auto-generates again (unless the user has typed a new override before that).

The script warns you at startup **and** at injection time if this is misconfigured.

---

## End-to-End Usage

### How to run
```bash
python lerobot/src/lerobot/rl/inference_pi05_async_interactive.py --config-path=config-hiserl.json
```

### What you see at startup
```
[INTERACTIVE] Type a subtask and press Enter at any time. Ctrl+C to stop.
>
```
The `>` prompt is printed once. After that, log lines from the inference and env threads will scroll past it. The prompt is still active — just type and press Enter.

### How to inject a subtask
1. Type the subtask text into the terminal (e.g. `pick up the cup`)
2. Press **Enter**
3. You immediately see:
   ```
   [INTERACTIVE] Queued: 'pick up the cup' — will be used at next action generation
   ```
   This is printed by the input thread confirming it was received.

4. On the **next inference cycle** (typically within one chunk period, ~200–500 ms), the inference thread pops the override, injects it, and logs:
   ```
   [INTERACTIVE] Injecting subtask override: 'pick up the cup'
   ```
5. Because the active subtask changed, the existing subtask logger fires immediately after:
   ```
   [SUBTASK] inference_step=42 | subtask: "pick up the cup"
   ```

So you get **two confirmations**: one from the input thread (queued) and one from the inference thread (injected + decoded from cache).

### What happens next
- The injected tokens stay in the policy cache for `subtask_regeneration_interval` seconds
- After that, the model auto-generates its next subtask as normal
- If you type again before the interval expires, the new text replaces the pending override atomically (old one is overwritten, never used)
- At episode boundary (`policy.reset()`), any pending override is cleared — a subtask typed during the reset won't bleed into the next episode

### Timing nuance
There is at most **one inference cycle** of lag between typing Enter and the subtask taking effect. The robot is already executing the current action chunk; the injected subtask conditions the *next* chunk. This is unavoidable in async RTC inference and is the expected behavior.
