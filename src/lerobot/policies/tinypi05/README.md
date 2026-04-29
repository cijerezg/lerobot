# TinyPI05

`tinypi05` is a scaled-down PI0.5-style policy for training from scratch.
It keeps the PaliGemma prefix model, Gemma action expert, AdaRMS timestep
conditioning, and flow-matching action chunks from PI0.5, but exposes model
dimensions through config presets and direct overrides.

Default presets:

- `debug`: tiny CPU/CUDA smoke-test model.
- `tiny_300m`: smaller training target.
- `small_500m`: default SmolVLA-like target.

Example:

```bash
uv run python -m lerobot.scripts.train_tinypi05_cube \
  --steps 30000 \
  --batch-size 16 \
  --architecture-preset small_500m
```

Use direct policy overrides when calling `lerobot-train`:

```bash
uv run lerobot-train \
  --dataset.repo_id=jackvial/cube-subtasks-e30-base120trim-0-9-101-end-fixed \
  --dataset.root=outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed \
  --dataset.use_imagenet_stats=false \
  --policy.type=tinypi05 \
  --policy.architecture_preset=tiny_300m \
  --policy.vlm_width=512 \
  --policy.expert_width=512 \
  --policy.push_to_hub=false
```
