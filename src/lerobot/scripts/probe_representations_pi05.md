# PI05 Representation Probe

Visualises the internal representations of a trained PI05 policy using PCA + UMAP,
without training any secondary model. The core idea: collect hidden-state activations
from many dataset frames, compress them to 2D/3D with PCA → UMAP, and examine the geometry
to understand what the model has learned to represent.

---

## Activation sites

Two streams are captured per frame:

**`prefix_out`** `(B, prefix_len, 2048)` — PaliGemma's final hidden states after processing
image tokens, task tokens, and subtask tokens. Encodes the model's understanding of the
current situation. Independent of denoising timestep. Mean-pooled to `(B, 2048)`.

**`suffix_out`** `(B, chunk_size, 1024)` — Gemma expert's hidden states after cross-attending
to the prefix and processing noisy action tokens. Encodes the action expert's denoising state.
Collected at multiple noise levels; last `chunk_size` positions sliced then mean-pooled to `(B, 1024)`.

### Denoising timesteps (suffix only)

Flow matching interpolates `x_t = t · noise + (1−t) · actions`, so `t=1` is pure noise and
`t=0` is the clean action.

| t    | What the suffix sees                       |
|------|--------------------------------------------|
| 1.0  | Pure noise — maximum uncertainty           |
| 0.25 | Mostly clean — action nearly decided       |

Comparing geometry across timesteps shows whether semantic clustering appears early (plan-based)
or only late (reaction-based).

---

## Pipeline

```
Dataset frames  (N_FRAMES_PER_EPISODE evenly spaced per episode, MAX_EPISODES episodes)
    │
    ▼  forward pass per frame (GT subtask injected, no FAST tokens)
    ├── prefix_out  (N, 2048)
    └── suffix_out  (N, 1024)  ×  {t=1.0, t=0.25}
    │
    [optional] subtask injection: re-run each frame with model-generated subtask tokens
    │
    ▼
activations_cache.pt   ← saved here; re-use with --probe_mode plot to skip inference
    │
    ▼  PCA (→ probe_pca_dims components)   scree plot saved to pca_variance/
    │
    ▼  UMAP 2D  →  static PNGs
       UMAP 3D  →  interactive HTML (Plotly)
```

**PCA** denoises high-frequency variation before UMAP and makes it substantially faster.
Check the scree plot — if 95% variance is captured at 30 components, lower `probe_pca_dims`;
if it needs 150, increase it.

**UMAP** `n_neighbors` controls local vs. global structure (small = tight local clusters,
large = coarser global layout). `min_dist` controls how tightly points pack within clusters.

---

## Output structure

```
outputs/probe_representations/
├── activations_cache.pt
├── episode_thumbnails/          first-frame images for each sampled episode
├── pca_variance/
│   ├── prefix_pca_scree.png
│   ├── suffix_t1.0_pca_scree.png
│   └── suffix_t0.25_pca_scree.png
├── 2d/
│   ├── prefix/
│   │   ├── by_episode.png       per-episode sequential colormap, dark=early frame light=late
│   │   ├── by_frame.png         all episodes pooled, colour = frame index
│   │   └── by_subtask.png       all episodes pooled, colour = subtask label
│   ├── suffix_t1.0/
│   └── suffix_t0.25/
├── 3d/
│   ├── prefix/
│   │   ├── by_episode.html
│   │   ├── by_subtask.html
│   │   └── ep{A}_vs_ep{B}.html  two-episode comparison with different marker shapes
│   ├── suffix_t1.0/
│   └── suffix_t0.25/
└── subtask_injection/
    ├── generated_subtasks.csv   per-frame GT and model-generated subtask text
    ├── prefix/  2d/gen_vs_gt.png  3d/gen_vs_gt.html
    ├── suffix_t1.0/
    └── suffix_t0.25/
```

---

## How to read the plots

**`by_episode`** — tight per-episode clusters = memorisation; episodes intermixed with subtasks
clustered = generalisation. Most diagnostic plot for overfitting.

**`by_frame`** — smooth colour gradient within clusters = model tracks task progress;
scattered colour = purely reactive to instantaneous observations.

**`by_subtask`** — clear separation = subtask-organised latent space. Compare prefix vs suffix:
if prefix separates but suffix does not, the VLM encodes the subtask but the action expert ignores it.

**`ep{A}_vs_ep{B}`** — two episodes with different marker shapes, coloured by frame index.
Shows whether both episodes trace similar trajectories (generalised structure) or diverge
(episode-specific memorisation).

**`gen_vs_gt`** — GT subtask (●) vs model-generated subtask (✕) in the same UMAP space.
Overlap = generated subtask produces equivalent representations; separation = subtask
generation is shifting representations away from the GT direction.

---

## Usage

```bash
# install once
pip install umap-learn plotly scikit-learn

# full run (collect activations + plot)
python probe_representations_pi05.py config-hiserl.json

# collect only (on GPU node)
python probe_representations_pi05.py config-hiserl.json --probe_mode collect

# plot from existing cache (no GPU needed)
python probe_representations_pi05.py config-hiserl.json \
    --probe_mode plot \
    --probe_cache outputs/probe_representations/activations_cache.pt
```

Key parameters (top of script or via CLI):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `N_FRAMES_PER_EPISODE` | 128 | Frames sampled per episode |
| `MAX_EPISODES` | 5 | Episodes to sample (`None` = all) |
| `PCA_DIMS` | 100 | Pre-UMAP PCA components |
| `UMAP_N_NEIGHBORS` | 15 | Local vs global structure balance |
| `UMAP_MIN_DIST` | 0.1 | Point packing tightness |
| `DENOISING_TIMESTEPS` | 1.0, 0.25 | Noise levels to probe for suffix |
| `EPISODE_3D_A/B` | 0, 1 | Episodes for the two-episode 3D comparison |
| `DO_SUBTASK_INJECTION` | True | Run gen vs GT subtask analysis |
