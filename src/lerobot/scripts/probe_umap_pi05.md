# UMAP Representation Probe — PI05 Policy

## Motivation

Training a Vision-Language-Action model gives you loss curves and action MSE, but neither tells you
*what* the model has actually learned to represent internally. This probe script is a lightweight
diagnostic tool to answer structural questions about the model's representations without training
any secondary model (no sparse autoencoders, no linear classifiers). The core tool is UMAP: collect
activations from many dataset frames, compress to 2D or 3D, and examine the geometry.

The central questions:

- **Memorisation vs. generalisation.** Does the model encode episode identity (memorised), or does
  it encode semantic content that generalises across episodes? If points cluster tightly by episode
  in UMAP space but scatter within subtask, the model has over-indexed on episode-specific
  features.
- **Semantic structure.** Do frames with the same subtask land near each other regardless of which
  episode or robot state produced them? If yes, the model has built a subtask-organised latent
  space — a desirable property for a language-conditioned policy.
- **Temporal organisation.** Does the representation change smoothly as the episode progresses, or
  is it unordered? A smooth temporal gradient suggests the model tracks task progress; random
  ordering suggests it relies on instantaneous observations only.
- **Effect of subtask conditioning.** When we inject the ground-truth subtask instead of the
  model's own generated subtask, does the action expert's representation change meaningfully? A
  large shift indicates that subtask conditioning is doing real work. A small shift suggests the
  action expert is largely ignoring the subtask token.
- **Noise sensitivity across the diffusion trajectory.** The action expert processes noisy actions
  at different denoising timesteps. Does the representation change substantially as the noise level
  drops (t=1 → t=0.25), or is the geometry stable? Stability across timesteps suggests the model
  encodes a task-level plan early and refines locally; instability suggests it works largely
  reactively at each denoising step.

---

## What Gets Collected

### Activation sites

The model has two main representation streams:

**`prefix_out`** — the VLM backbone's (PaliGemma, 2B parameters) final hidden states after
processing the full prefix: SigLIP image tokens, language task tokens, and subtask tokens. Shape
`(B, prefix_len, 2048)`. This encodes the model's understanding of *the current situation* — what
it sees, what it is asked to do, and what subtask it has inferred. Prefix computation is a single
forward pass; it does not depend on the denoising timestep. Mean-pooled over the sequence dimension
to `(B, 2048)` before PCA.

**`suffix_out`** — the action expert's (Gemma 300M) final hidden states after cross-attending to
the prefix and processing the noisy action tokens. Shape `(B, chunk_size=50, 1024)`. This encodes
the action expert's representation of the current denoising state — how it is integrating the
situational context with the noisy action trajectory. The last `chunk_size` positions are sliced
(to remove any prefix-side padding), then mean-pooled to `(B, 1024)`.

Both sites are collected with GT subtask tokens injected during the standard collection phase.
Subtask injection analysis runs an additional forward with generated subtask tokens.

### Denoising timesteps for `suffix_out`

Flow matching in PI05 uses the interpolation:

```
x_t = t · noise + (1 - t) · actions
```

where `t ∈ [0, 1]`, `t=1` is pure noise, and `t=0` is the clean action. The model is probed at
three points:

| t    | x_t content             | What the suffix represents                        |
|------|-------------------------|---------------------------------------------------|
| 1.0  | Pure noise              | Action expert seeing maximum uncertainty          |
| 0.5  | Half noise, half signal | Mid-denoising — coarse action plan forming        |
| 0.25 | Mostly clean signal     | Late denoising — action trajectory nearly decided |

Comparing UMAP geometry across these three timesteps reveals how the action expert's representation
evolves during denoising. If clustering by subtask is already present at t=1.0, the model forms
subtask-specific action plans even before seeing any action signal. If clustering only emerges at
t=0.25, the model's structure is largely driven by the action content rather than the semantic
context.

---

## Pipeline

```
Dataset frames
    │
    ▼  (N_FRAMES_PER_EPISODE evenly spaced per episode)
Activation collection
    │   • Forward hook on PaliGemmaWithExpertModel
    │   • One forward at each t ∈ {1.0, 0.5, 0.25} per frame
    │   • GT subtask tokens injected via preprocessor
    │
    ├──► prefix_out  (N, 2048)  ─────────────┐
    └──► suffix_out  (N, 1024) × 3 timesteps ┘
                                              │
    [optional] Subtask injection              │
    • Additional forward with generated       │
      subtask tokens → prefix_gen, suffix_gen │
                                              ▼
                                     Activation cache (.pt)
                                              │
                                              ▼
                                   PCA  (→ 100 components)
                                   Scree plot saved
                                              │
                                              ▼
                                   UMAP 2D  +  UMAP 3D
                                              │
                          ┌───────────────────┼──────────────────┐
                          ▼                   ▼                  ▼
                   by_episode.png      by_frame.png      by_subtask.png
                   (2D static)         (2D static)       (2D static)

                   by_episode.html     by_subtask.html   ep0_vs_ep1.html
                   (3D interactive)    (3D interactive)  (3D interactive)

                   subtask_injection/
                     gen_vs_gt.png   gen_vs_gt.html
```

### Dimensionality reduction

**PCA** is applied first (default: 100 components) before UMAP. This serves two purposes: it
denoises high-frequency variation that UMAP might otherwise overfit, and it makes UMAP substantially
faster. The scree plot saved to `pca_variance/` shows per-component and cumulative explained
variance, with 90% and 95% thresholds annotated. Use this to judge whether 100 components is
appropriate (e.g. if 95% variance is captured at 30 components, reduce `probe_pca_dims`; if 95%
requires 150 components, increase it).

**UMAP** (`n_neighbors=15`, `min_dist=0.1` by default) is then applied to the PCA-reduced
representations to produce 2D (static plots) and 3D (interactive HTML) embeddings. The
`n_neighbors` parameter controls the balance between local and global structure: small values
(5–10) emphasise tight local clusters, large values (50–100) emphasise coarser global layout. The
defaults are a reasonable starting point; adjust them in `ProbeUmapConfig` if the plots are
uninformative.

---

## Output Structure

```
outputs/probe_umap/
│
├── activations_cache.pt               Raw activation tensors + metadata
│
├── pca_variance/
│   ├── prefix_pca_scree.png           Explained variance for prefix_out PCA
│   ├── suffix_t1.0_pca_scree.png
│   ├── suffix_t0.5_pca_scree.png
│   └── suffix_t0.25_pca_scree.png
│
├── 2d/
│   ├── prefix/
│   │   ├── by_episode.png             Each colour = one episode
│   │   ├── by_frame.png               Colour = frame index within episode
│   │   └── by_subtask.png             Each colour = one subtask label
│   ├── suffix_t1.0/                   Same three plots for suffix at t=1.0
│   ├── suffix_t0.5/
│   └── suffix_t0.25/
│
├── 3d/
│   ├── prefix/
│   │   ├── by_episode.html            Rotatable 3D, hover shows ep/fr/subtask
│   │   ├── by_subtask.html
│   │   └── ep0_vs_ep1.html            Two specific episodes, different marker shapes
│   ├── suffix_t1.0/
│   ├── suffix_t0.5/
│   └── suffix_t0.25/
│
└── subtask_injection/
    ├── prefix/
    │   ├── 2d/gen_vs_gt.png           GT (●) and generated (✕) in same UMAP space
    │   └── 3d/gen_vs_gt.html          Rotatable, 0=GT / 1=generated
    ├── suffix_t1.0/
    ├── suffix_t0.5/
    └── suffix_t0.25/
```

---

## How to Read the Plots

### `by_episode`

Each colour is one episode. Look for:

- **Tight per-episode clusters, subtasks mixed** → memorisation. The model's representations are
  organised primarily by which episode the frame came from, not by what is happening semantically.
- **Episodes intermixed, subtasks clustered** → generalisation. The model has abstracted away
  episode-level variation and organised representations by semantic content.
- **Episodes partially overlapping** → mixed. Some features generalise, others don't.

This is arguably the most diagnostic plot for detecting overfitting to the training set.

### `by_frame`

Colour encodes the temporal position within the episode (early frames are dark, late frames are
light or vice versa depending on the colourmap). Look for:

- **Smooth colour gradient within episode clusters** → the model tracks task progress. Frames from
  early in an episode are near each other and far from late frames in the same episode.
- **Scattered colour within clusters** → the model's representation is not temporally ordered;
  it responds to instantaneous observations rather than tracking progress.
- **Colour gradient that cuts across episode boundaries** → the model has learned a concept of
  "task progress" that generalises across episodes (e.g. "grasping" always looks similar regardless
  of which episode).

### `by_subtask`

Each colour is one subtask label. This is the most direct test of semantic organisation. Look for:

- **Clear separation by subtask** → the model has built subtask-specific representations. Different
  behaviours (e.g. "reach to object" vs. "lift object" vs. "place object") occupy distinct regions
  of the latent space.
- **No subtask separation** → the model is not using the subtask as a meaningful organising
  principle for its representations.
- **Partial separation** → some subtask pairs are well-separated (look at which ones) while others
  overlap (indicating the model treats those subtasks similarly).

Compare this plot between `prefix` and `suffix`. If `prefix` shows strong subtask separation but
`suffix` does not, the VLM is correctly encoding the subtask but the action expert is not using it
to produce subtask-specific action representations.

### `ep0_vs_ep1` (3D)

Two specific episodes with different marker shapes (● and ■), coloured by frame index. This makes
it possible to see:

- Whether the two episodes trace similar trajectories through the representation space (generalised
  task structure) or entirely different paths (episode-specific memorisation).
- Whether the temporal ordering is preserved (frame colours should progress smoothly) or scrambled.
- Where the two episodes diverge — this often corresponds to the moment a subtask transition occurs.

### `gen_vs_gt` (subtask injection)

GT subtask tokens (●) and model-generated subtask tokens (✕) plotted in the same UMAP space. The
UMAP is fit on the combined set so both conditions share coordinates. Coloured by subtask index.
Look for:

- **● and ✕ overlapping** → the model's generated subtask produces nearly the same representation
  as the correct GT subtask. The subtask generation is accurate and the action expert uses it
  consistently.
- **● and ✕ systematically separated** → the generated subtask shifts representations away from
  the GT subtask direction. Whether this is a problem depends on whether the generated subtask is
  semantically wrong or merely phrased differently.
- **Subtask-coloured clusters that split between ● and ✕** → for a given subtask (colour), the
  generated and GT tokens produce representations in different parts of the space. This is a signal
  that the model's self-generated subtask conditioning is meaningfully different from the ground
  truth.

---

## Running the Script

**Install dependencies (once):**
```bash
pip install umap-learn plotly scikit-learn
```

**Full run (collect + plot):**
```bash
python probe_umap_pi05.py config-hiserl.json
```

**Collect only** (useful on GPU node, then plot locally):
```bash
python probe_umap_pi05.py config-hiserl.json --probe_mode collect
```

**Plot from existing cache** (fast, no model needed):
```bash
python probe_umap_pi05.py config-hiserl.json \
    --probe_mode plot \
    --probe_cache outputs/probe_umap/activations_cache.pt
```

**Key parameters to tune** (set at the top of the script or via CLI):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `N_FRAMES_PER_EPISODE` | 50 | Frames per episode. More = richer plot, slower collection. |
| `PCA_DIMS` | 100 | Pre-UMAP PCA components. Check scree plot to tune. |
| `UMAP_N_NEIGHBORS` | 15 | Small = local clusters emphasised; large = global layout. |
| `UMAP_MIN_DIST` | 0.1 | Small = tighter clusters; large = more spread. |
| `DENOISING_TIMESTEPS` | 1.0, 0.5, 0.25 | Which noise levels to probe for suffix. |
| `EPISODE_3D_A/B` | 0, 1 | Which two episodes to compare in the 3D two-episode plot. |
| `DO_SUBTASK_INJECTION` | True | Whether to run the gen vs GT subtask comparison. |

The activations cache (`activations_cache.pt`) stores the raw tensors and all metadata. Once
collected, the plotting phase (PCA + UMAP + all plots) typically takes a few minutes and requires
no GPU. Adjust UMAP parameters and re-plot without re-running the model by using
`--probe_mode plot --probe_cache /path/to/cache.pt`.
