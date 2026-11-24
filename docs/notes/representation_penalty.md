# Representation penalty: stay close to base MolmoAct2 while fine-tuning (design 2026-11-23)

> **Status:** built, off 2026-11-23; smoke-tested for 5 steps; no run yet. First run is the plain one (`representation_beta: 0.0`, distances logged only).

Fine-tuning on a new task with the usual loss plus a term that keeps every layer's residual
stream close to what base MolmoAct2 computes on the same frame.

## 1. Objective

For one token at one layer, $r$ is the trained model's residual vector after the block and
$r_0$ is base MolmoAct2's on the same input (same frame, same noise, same flow time).

$$e(r, r_0) = \frac{\lVert r - r_0\rVert^2}{\lVert r_0\rVert^2} = (\rho - 1)^2 + 2\rho\,(1 - \cos(r, r_0)), \qquad \rho = \frac{\lVert r\rVert}{\lVert r_0\rVert}$$

$$L = L_{\text{flow}} + L_{\text{discrete}} + \beta \cdot \tfrac{1}{2}\Big(\operatorname*{mean}_{l,\,\text{tokens}} e^{\text{LM}}_l + \operatorname*{mean}_{l,\,\text{action tokens}} e^{\text{expert}}_l\Big)$$

- Layers: the 36 language-model blocks and the 36 action-expert blocks. Tokens: all valid positions.
- Frames: the training frames (tube). No other data.
- One $\beta$. With `knowledge_insulation: true` the language model is trained by the discrete loss only and the expert by the flow loss only, so the one $\beta$ prices two separate trade-offs.
- No running statistics: $\lVert r_0\rVert$ is the scale, and it does not move with the drift.

## 2. Code

| What | Where |
| --- | --- |
| Switch | `policy.representation_beta`: `null` = off (no base copy), `0.0` = distances logged, `> 0` = added to the loss |
| Base copy | `MolmoAct2Policy._load_hf_model`: deep copy of the language-model blocks and the action expert, taken before any checkpoint is overlaid; held in a plain list, so it is not saved and not trained |
| Distance | `_relative_distance` and the joint loop `_compute_flow_matching_loss_joint_per_layer` in `policies/molmoact2/modeling_molmoact2.py`; the base layer runs next to each trained layer under `no_grad` |
| Gradients | `_representation_gradients` in `rl/molmoact2/rl_molmoact2_trainer.py`: one more forward + backward on the first microbatch of a logged step |
| Base prompt | `policy.prompt_setup` renders the base setup clause (`policy.prompt_expected_control_mode` exists, not used: user decision 2026-11-23, no control mode clause); `subtask` in `policy.prompt_clauses_off` |
| Configs | `config_rl_rep.yaml`, `config_rl_rep_validate.yaml` (workspace root) |

The base copy reads the trained model's input embeddings. That is the base input only while
the vision tower, the projector and the embeddings are frozen (`vision_from_layer: null`,
`freeze_embedding: true`).

## 3. Logged (Aim, `rep/...`; held-out episodes `val_rep/...`)

| Key | Meaning |
| --- | --- |
| `rep/lm`, `rep/expert` | mean $e$ over layers |
| `rep/{lm,expert}_norm/NN` | norm part $(\rho-1)^2$ at layer NN |
| `rep/{lm,expert}_cos/NN` | cosine part $2\rho(1-\cos)$ at layer NN |
| `rep/grad_norm_loss_{lm,expert}`, `rep/grad_norm_penalty_{lm,expert}` | gradient norm of the loss (without the penalty) and of the penalty (without $\beta$) |
| `rep/grad_cos_{lm,expert}` | cosine between the two gradients; they come from two forwards with different noise draws |

## 4. Memory

The base copy costs 8.8 GB of weights plus its activations (8.4 GB at batch 64). Batch 64 x 2
runs out of memory on the 96 GB card; the configs use 32 x 4 (peak 76 GB).

## 5. Plan

1. Plain run from base, `representation_beta: 0.0`, 2000 steps.
2. Read the drift (`rep/...`) and the gradient norms; choose $\beta$.
3. Penalised run from base, same config with that $\beta$.
4. Compare flow loss on the held-out episodes at equal training loss.
