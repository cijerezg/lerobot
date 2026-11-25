# weight_change (2026-11-24)

Question: which layers' weights moved most between base MolmoAct2 and a checkpoint? Relative change
|w - w0| / |w0| per block and per part of the block (language model: self_attn, mlp, norms; action expert:
self_attn, cross_attn, mlp, modulation), base cast to the checkpoint's dtype so frozen weights read exactly 0.

## Run

    uv run python lerobot/adhocs/weight_change/weight_change.py outputs/runs/<run>/checkpoints/<step> [...]

About 1 minute per checkpoint on the CPU. Prints the table, writes `outputs/analysis/weight_change/<run>_<step>.json`.

## Found (rebot-v2-tube-rep-beta0-20261123, steps 200 and 1400)

- Language model (trained from layer 2): block change at 1400 rises smoothly from 0.035 at layer 2 to 0.060 at
  layers 21-25 and falls to 0.034 at layer 35; attention and MLP move the same amount. Nothing stands out at
  layer 16, where the activations' norm part jumps to ~1000 (docs/notes/representation_penalty.md).
- Action expert: 0.014 at layer 0, 0.045 at layers 21-24, 0.027 at layer 35. The MLP carries the late layers
  (0.05); cross attention is 0.034-0.040 at layers 16-24 and halves at layer 25. `context_k_proj` /
  `context_v_proj` 0.07.
- Step 200 has the same profile at about 0.4 of the size.

## Reuse

Any checkpoint saved as a full `model.safetensors` with keys `model.<base key>`.
