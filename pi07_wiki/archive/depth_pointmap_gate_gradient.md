# Why `pointmap_gate` was stuck at 0.0000 — mechanism + resolution

Scratch note (2026-06-20). First GPU train steps of the point-map MoT ran clean, but the
actor gate `pointmap_gate` stayed exactly `0.0000` while `loss_flow` fell 0.97 -> 0.044.
**Root cause (confirmed): the actor's point-map depth modules were frozen**
(`requires_grad=False`), so the gate α never received a gradient. Fixed by adding an
always-trainable branch for them in `_apply_actor_freeze`. The sections below explain
what the gate is, why it is the single linchpin at init, how we localized the freeze, and
what we ruled out. Companion: `depth_pointmap_design.md` §B.4-B.5.

---

## 1. What `pointmap_gate` measures

The action expert reads the depth stream at every layer ℓ as an additive, gated term
(design §B.4). The addition is one line in the patched `cross_attn_forward`
([modeling_molmoact2.py:692-694](lerobot/src/lerobot/policies/molmoact2/modeling_molmoact2.py#L692-L694)):

```python
out = self._attention(q, k, v, attn_mask=attn_mask)        # frozen context read
if depth_kv is not None:
    out = out + depth_gate.to(out.dtype) * gated_depth_read(q, depth_kv, depth_sink)
```

So per layer

$$\text{out}_\ell = \mathrm{SDPA}(q, K_{ctx}, V_{ctx}) + g_\ell \cdot r_\ell,
\qquad g_\ell = \tanh(\alpha_\ell),\quad r_\ell = \texttt{gated\_depth\_read}(\dots).$$

`pointmap_gate` is the mean over layers of $g_\ell = \tanh(\alpha_\ell)$, where $\alpha_\ell$
is the per-layer scalar `DepthStream.gate`, initialized to 0
([modeling_stream.py:232](lerobot/src/lerobot/policies/depth_pointmap/modeling_stream.py#L232)).
At init every $\alpha_\ell = 0$, so the depth term vanishes exactly and the policy is
bit-identical to the depth-free model (the property the probe already confirmed). So
`pointmap_gate = 0.0000` at **step 0** is correct and required. The bug was that it never
moved afterward.

---

## 2. At init, the gate is the only depth parameter that can move

This is the structural fact that made a frozen gate fatal (the "soft deadlock", §B.5).

The depth term is $g_\ell \cdot r_\ell = \tanh(\alpha_\ell)\, r_\ell$. At init ($\alpha_\ell = 0$):

- **The gate $\alpha_\ell$:** $\dfrac{\partial L}{\partial \alpha_\ell}
  = \big\langle \partial L/\partial\,\text{out}_\ell,\; r_\ell \big\rangle \cdot (1 - \tanh^2(\alpha_\ell))
  = \big\langle \partial L/\partial\,\text{out}_\ell,\; r_\ell \big\rangle$, since $\tanh'(0)=1$
  and $r_\ell \neq 0$. The gate has a **nonzero gradient at init** — it is the one knob that
  can start moving.
- **Everything else** producing $r_\ell$ (stream blocks, read projections, sink):
  $\dfrac{\partial L}{\partial \theta} = \tanh(\alpha_\ell)\,(\dots) = 0$. Zero gradient
  **until the gate lifts.**

So the entire depth stream learns nothing until the gate moves, and the gate moves only if
it gets its gradient. Freeze the gate and the whole stream is dead forever — exactly the
symptom. The linchpin and the frozen parameter were the same scalar.

---

## 3. Root cause: the actor depth modules were frozen

When `cfg.policy.trainable_params` is set (the authoritative per-name freeze; the 4B/9B
run), `_apply_actor_freeze` walks every actor parameter and sets `requires_grad` by name
pattern, with a catch-all **else that freezes anything unrecognized**
([rl_molmoact2_trainer.py:322-325](lerobot/src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L322-L325)):

```python
else:
    # Unknown actor param — freeze. Safer than accidentally training ...
    param.requires_grad = False
```

The actor's point-map params are named `pointmap_encoder.*` and `depth_stream.*`. They
match none of the patterns (`.action_expert.`, `.transformer.`, `.image_vit.`, ...), so they
fell into that else and were frozen — including `depth_stream.gate`.

The **critic** side never had this bug: `_apply_critic_freeze` has an explicit
always-trainable branch for its fresh depth modules
([rl_molmoact2_trainer.py:340-347](lerobot/src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L340-L347),
`name.startswith("depth_")`). The actor freeze was simply never given the same branch — an
actor/critic asymmetry. (Note the coarse `__init__` freeze
`_freeze_non_action_expert_parameters` does include `depth_stream`, but it runs *before* the
depth modules are constructed and is also overridden by the per-name freeze, so it didn't
save us.)

### The fix

Add the mirror branch at the top of `_apply_actor_freeze`
([rl_molmoact2_trainer.py:306](lerobot/src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L306)):

```python
if "pointmap_encoder" in name or "depth_stream" in name:
    param.requires_grad = True   # fresh depth modules, trained from scratch
elif ".action_expert." in name:
    ...
```

---

## 4. How we localized it (the diagnostic that settled it)

A per-iteration line in the trainer prints `requires_grad` + grad status for three depth
params read identically inside `run_layer` — the gate, the sink, and a stream-block weight
([rl_molmoact2_trainer.py:921-935](lerobot/src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L921-L935)).
The decisive output was:

```
[gate] mean=+0.000e+00 absmax=0.000e+00 | gate[no-req-grad] sink[no-req-grad] block0[no-req-grad]
```

`no-req-grad` on all three is unambiguous: the parameters were frozen, not merely receiving
a zero or severed gradient. The earlier symptom `grad_absmax=nan` was just `gate.grad is
None` — a frozen parameter never gets a `.grad`, which reads as None.

The intermediate cases the diagnostic was built to distinguish:
- all three `grad=None` (but `requires_grad` true) -> depth read disconnected from the loss;
- gate `None`, sink/block `grad_absmax=0.00e+00` (zero tensors) -> gate-specific graph break;
- any `no-req-grad` -> a freeze bug. <- this is what we got.

---

## 5. What we ruled out

- **Gradient checkpointing severing a closure-captured gate.** Initial hypothesis: the gate
  was precomputed outside `run_layer` (`depth_gate = gate_value()`) and only captured by
  closure into the checkpointed region, so its gradient was dropped. We changed the read to
  pull the parameter in-region (`torch.tanh(self.depth_stream.gate[layer_idx])`), matching
  how `sink_logit` is read. This did **not** fix it (the param was frozen, not severed), but
  it is kept because it is strictly safer and symmetric with the sink — no fragile
  closure-captured non-leaf. It was never the cause: the action-expert blocks are read the
  same way inside the same checkpoint and train fine.
- **Mean cancelling across signed per-layer gates.** The console prints the mean of
  $\tanh(\alpha_\ell)$, which could hide signed movement; but `pointmap_gate_absmax` was also
  exactly 0, so no gate moved.

---

## 6. One-line summary

`pointmap_gate` is the mean per-layer depth gate $\tanh(\alpha_\ell)$; at init it must be 0
and the gate is the *only* depth parameter with a nonzero gradient (soft deadlock), so the
whole stream depends on it. It was stuck because the actor's `pointmap_encoder` /
`depth_stream` params were frozen by the unrecognized-param else-branch in
`_apply_actor_freeze`, while the critic already had an always-trainable depth branch. Fix:
add the same branch for the actor.
