# RECAP Walkthrough — `jackvial/so101_pickplace_success_120_v2`

End-to-end guide for getting RECAP running on your 120-episode pick-and-place dataset.

---

## Prerequisites

- [x] LeRobot repo on the `carlos-recap-main` branch
- [x] 120-episode dataset at `jackvial/so101_pickplace_success_120_v2`
- [ ] π0.5 base weights downloaded locally (HuggingFace: `lerobot/pi05_base`)

Download base weights if you haven't already:
```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/pi05_base', local_dir='models/pi05_base')"
```

---

## Step 1: Annotate Dataset with Subtasks

RECAP uses π0.5 **full**, which requires subtask annotations on every episode.
The manual Gradio tool is the most reliable option.

- [x] Launch the annotation UI:
```bash
uv run python src/lerobot/policies/pi05_full/annotate/manual_subtask_annotate.py
```

- [x] Open `http://localhost:7860` in your browser
- [x] Enter the repo ID: `jackvial/so101_pickplace_success_120_v2`
- [ ] Set the video key (likely `observation.images.top` or `observation.images.wrist`)
- [ ] For each episode, scrub through the video and mark skill boundaries (e.g. "reach for cube", "grasp cube", "lift cube", "move to bowl", "release cube")
- [ ] Save annotations — this writes `subtask_index` and `meta/subtasks` into the dataset
- [x] Used script to `scripts/bulk_annotate_subtasks.py` to create sub task annotated dataset `outputs/so101_pickplace_success_120_v2_w_subtasks`

> **Tip:** You don't need to annotate all 120 episodes in one sitting. The tool saves progress and can resume.

After annotation, verify the dataset has subtasks:
```bash
uv run python -c "
import json
from pathlib import Path
info = json.loads(Path.home().joinpath('/home/jack/code/lerobot/outputs/so101_pickplace_success_120_v2_w_subtasks/meta/info.json').read_text())
print('subtasks' in info or 'subtasks' in str(info.get('features', {})))
"
```

---

## Step 2: Compute Anchor Action Stats

RECAP works best with anchor actions (a_t − s_0). You need per-timestep normalization stats.

- [x] Run the stats computation script:
```bash
uv run python src/lerobot/scripts/compute_delta_stats.py \
    --root outputs/so101_pickplace_success_120_v2_w_subtasks \
    --encoding anchor \
    --output-dir outputs/stats
```

- [x] Verify the output file exists:
```bash
ls outputs/stats/action_stats_anchor_*.pt
```

Note the full path — you'll need it for the config.

---

## Step 3: Create Your Config

Copy the template and customize it for your setup.

- [x] Copy the template:
```bash
cp src/lerobot/rl/config-hiserl.json config-recap.json
```

- [x] Edit `config-recap.json` with these fields:

| Field | Value |
|---|---|
| `dataset.repo_id` | `outputs/so101_pickplace_success_120_v2_w_subtasks` |
| `dataset.root` | path to annotated dataset (or `null` to use HF cache) |
| `policy.pi05_checkpoint` | `models/pi05_base` (or wherever you downloaded it) |
| `policy.task` | `"Pick up the cube and place it in the bowl"` (your task description) |
| `policy.action_encoding` | `"anchor"` |
| `policy.action_encoding_stats_path` | `"outputs/stats/action_stats_anchor_so101_pickplace_success_120_v2.pt"` |
| `policy.offline_steps` | `10000` (start with 8000–10000) |
| `offline_output_dir` | `"outputs/recap_offline_v1"` |
| `policy.input_features` | match your dataset cameras (see below) |
| `wandb.enable` | `false` (or `true` if you want logging) |

Camera config — make sure `input_features` matches your dataset:
```json
"input_features": {
    "observation.images.top": {
        "type": "VISUAL",
        "shape": [3, 224, 224]
    },
    "observation.images.side": {
        "type": "VISUAL",
        "shape": [3, 224, 224]
    },
    "observation.state": {
        "type": "STATE",
        "shape": [6]
    }
}
```

Also update the `env.features` and `env.robot.cameras` sections to match your hardware, or remove the `env` section entirely if you're only doing offline training.

---

## Step 4: Run Offline Training

This trains the policy (flow matching) and critic (Bellman updates) together. Target ~8000–10000 steps.

- [ ] Start training:
```bash
uv run python -m lerobot.scripts.offline_learner_pi05 --config_path config-recap.json
```

- [ ] Monitor loss curves (if wandb enabled, or check terminal output)
- [ ] Checkpoints saved to `outputs/recap_offline_v1/checkpoints/`

Expected runtime: ~2–4 hours on a single GPU depending on batch size and gradient accumulation.

---

## Step 5: Verify the Checkpoint

- [ ] Confirm checkpoint exists:
```bash
ls outputs/recap_offline_v1/checkpoints/
```

- [ ] The final checkpoint should contain both actor (policy) and critic weights. You can verify:
```bash
uv run python -c "
import torch
ckpt = torch.load('outputs/recap_offline_v1/checkpoints/latest/pretrained_model/model.safetensors', weights_only=False)
has_critic = any('critic' in k for k in ckpt.keys())
print(f'Has critic weights: {has_critic}')
print(f'Total keys: {len(ckpt)}')
"
```

---

## Step 6 (Optional): Online Training

Once offline training is done, you can move to online RL on the real robot. This requires two processes.

- [ ] Update `config-recap.json`:
  - Set `policy.pi05_checkpoint` to your offline checkpoint path
  - Configure `env.robot` for your hardware (ports, cameras)
  - Set `policy.actor_learner_config.learner_host` to your machine's IP

- [ ] Start the learner (terminal 1):
```bash
uv run python -m lerobot.rl.learner_pi05 --config_path config-recap.json
```

- [ ] Start the actor (terminal 2):
```bash
uv run python -m lerobot.rl.actor_pi05_async --config_path config-recap.json
```

- [ ] Use intervention keys during rollouts:
  - `5` — toggle intervention (take/release control with leader arm)
  - `1` — mark success and end episode
  - `0` — mark failure and end episode
  - `2` — start next episode

---

## Troubleshooting

**"Stats mismatch" or normalization errors**
→ Double-check that `action_encoding_stats_path` points to the `.pt` file from Step 2, and that `action_encoding` is set to `"anchor"`.

**Subtask errors during training**
→ The dataset must have `subtask_index` as a feature and `subtasks` in `meta/info.json`. Re-run the annotation tool if missing.

**OOM during training**
→ Reduce `batch_size` (try 8–16) or increase `gradient_accumulation_steps` to compensate. Enable `gradient_checkpointing: true`.

**Checkpoint won't load**
→ If using `pi05_base`, the path should contain the string `pi05_base` so the loader uses dataset stats instead of checkpoint stats. If renamed, the loader may behave differently.
