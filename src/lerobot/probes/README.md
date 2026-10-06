# Probe documentation

The exact MolmoAct2 tensor locations, equations, and dimensions used by the probes are
documented in [MODEL_TENSORS.md](MODEL_TENSORS.md). Read that note before interpreting a
plot labelled with a layer such as `L14` or `L31`.

The important distinctions are:

| Probe family | Tensor read | Shape for this checkpoint | Meaning of `Lℓ` |
|---|---|---:|---|
| `conditions_matrix`, `domain_representations`, `subspace_spans` | Token-group mean of the language block output | `[B,2560]` per group | After the complete encoder block's attention, MLP, and residual updates |
| `conditions_matrix`, `domain_representations`, `subspace_spans` (`action`) | Horizon mean of the action-expert block output | `[B,768]` | After the complete expert block's self-attention, cross-attention, MLP, and residual updates |
| `attention`, `attention_budget` | Action-expert softmax weights | cross `[B,8,30,S]`; self `[B,8,30,30]` | Inside the attention sublayer, before value mixing, output projection, and the residual update |
| `embodiment_swap` | Integrated normalized action chunk | `[B,30,8]` exposed by the mixed policy | No intermediate layer is read; the intervention is measured after the full model and flow sampler |

Layer numbering is zero-based. For example, `L31` is the 32nd of 36 layer pairs. Attention
weights and post-block representations are different measurements; neither should be
described as the other. Attention weights show routing, while Jacobian probes measure
forward sensitivity.

## Key probe: subspace spans

`subspace_spans` is part of the official full validation suite, enabled by
`probe_parameters.enable_subspace_spans: true` in `config_rl_validate.yaml`.
The training configs expose the same switch, disabled alongside the other extended
probes. `run_probes.sh` dispatches it through the standard validation registry and
requires its report in the final completeness check. The probe viewer and checkpoint
comparison discover its `index.json` automatically.

This probe measures mean-centred (per robot) representation rank, cross-robot span overlap,
principal angles, held-out energy outside each span, and the frames defining it.
Numerical rank counts singular values above a fraction of the largest singular value;
a shared mean component can change that count without removing frame differences.
These are geometry measurements, not task-performance scores.

When conditions matrix is also enabled, it runs first and subspace spans reuses its
`conditions_matrix/cache` with no additional model forwards. When subspace spans is
enabled alone, it collects that same cache without running the conditions-matrix
analysis. Sampling remains controlled by the `conditions_*` settings. `mode: collect`
only captures; `mode: plot` reads the existing cache; `mode: all` captures and analyzes.
A missing or incompatible cache fails visibly rather than substituting another step.

| Setting | Default | Meaning |
|---|---|---|
| `subspace_text` | `real` | Cached text condition (`real` or `neutral`) |
| `subspace_tolerances` | `0.3,0.1,0.05` | Relative singular-value cutoffs on the centred spectrum; middle sorted cutoff is the headline. 0.1 keeps k between 1 and the frame count at nearly every layer; 0.05 is frame-limited for state and action_output below ~100 frames |
| `subspace_layers` | `14,15,16,28,32` | Detailed spectra, principal angles, and pivots; last layer is the headline |
| `subspace_n_null` | `100` | Random subspace pairs per dimension tuple |
| `subspace_n_pivots` | `12` | Frame examples shown in the explorer |
| `subspace_pivot_group` | `img_external_0` | Headline token group |

Ranks and overlaps cover every captured layer. The additional detailed layers bracket
the observed middle-depth drop. Reports are written to `subspace_spans/explorer.html`,
with `index.json`, `summary.json`, and CSV downloads alongside it. The standalone
`python -m lerobot.probes.subspace_spans --cache_dir ...` and `--report_only` paths
remain available for saved-cache analysis.

Flow inversion remains an optional curiosity experiment in
[`migration/flow_inversion_2026-09-28/flow_inversion.py`](../../../../migration/flow_inversion_2026-09-28/flow_inversion.py),
with its existing `flow_inversion_report` renderer. It is not registered or enabled
in the official validation suite.
