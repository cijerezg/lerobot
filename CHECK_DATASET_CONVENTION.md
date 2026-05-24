# Diagnose: joint convention + frame-transform correction preview

## Why this matters

MolmoAct2 was pretrained on SO-100/101 data in the **v2.1 joint convention**.
LeRobot v3.0 records and reports angles in the **v3.0 convention**.
Two joints differ between the conventions:

| Joint | v3.0 → v2.1 transform |
|-------|----------------------|
| 1 (shoulder_lift) | `v2.1 = −v3.0 + 90`  (sign-flip + 90° offset) |
| 2 (elbow_flex)    | `v2.1 =  v3.0 + 90`  (+90° offset only)       |
| 0, 3, 4, 5        | unchanged                                      |

If the training dataset is in **v2.1** the model needs the transform at inference.
If it is in **v3.0** the model was fine-tuned without the transform and doesn't need it.

## What triggered this question

The trained checkpoint's embedded norm stats show for the training data:

| Joint | mean | q01 | q99 |
|-------|------|-----|-----|
| 1 (shoulder_lift) | ~126° | ~45° | ~186° |
| 2 (elbow_flex)    | ~120° | ~36° | ~174° |

These values match the base MolmoAct2 HF checkpoint (v2.1 convention).
But an older local dataset (`annotated_dataset-v3`) shows joint 1 mean ~5°,
joint 2 mean ~-9° — which looks like v3.0 convention.
The training dataset `annotated-dataset-v7` is only on the other PC.

---

## Run this script on the other PC

The script does three things:
1. Reads `annotated-dataset-v7` and reports the raw joint angle distribution.
2. Shows what the v3.0→v2.1 transform would do to a sample frame.
3. (Optional, requires GPU + model cache) Runs one inference with and without
   the transform so you can compare the action chunks side-by-side.

```python
#!/usr/bin/env python3
"""
Convention diagnostic for annotated-dataset-v7.
Run from the LeRobot repo root:
    python CHECK_convention.py
Optional GPU inference preview:
    python CHECK_convention.py --infer --prompt "Pick up the red truck and put it in the bowl"
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np

# ── transform constants (from molmoact2-so101/inference.py defaults) ──────────
SIGNS   = np.array([1., -1.,  1.,  1.,  1.,  1.], dtype=np.float32)
OFFSETS = np.array([0.,  90., 90.,  0.,  0.,  0.], dtype=np.float32)

def arm_to_model(state: np.ndarray) -> np.ndarray:
    """v3.0 arm frame → v2.1 model frame."""
    return SIGNS * state + OFFSETS

def model_to_arm(action: np.ndarray) -> np.ndarray:
    """v2.1 model frame → v3.0 arm frame."""
    return (action - OFFSETS) * SIGNS


# ── 1. Dataset stats ──────────────────────────────────────────────────────────

def check_dataset(dataset_path: pathlib.Path):
    print("\n" + "="*60)
    print("STEP 1 — Raw joint angles in the training dataset")
    print("="*60)

    stats_file = dataset_path / "meta" / "stats.json"
    if stats_file.exists():
        s = json.loads(stats_file.read_text())
        for key in ["action", "observation.state"]:
            if key not in s:
                continue
            print(f"\n  {key}:")
            for stat in ["mean", "q01", "q99"]:
                vals = s[key].get(stat)
                if vals is not None:
                    print(f"    {stat:6s}: {[round(v, 2) for v in vals]}")
    else:
        files = sorted(glob.glob(str(dataset_path / "data/**/*.parquet"),
                                 recursive=True))
        if not files:
            print(f"  ERROR: no parquet files in {dataset_path}")
            return
        import pandas as pd
        df = pd.read_parquet(files[0])
        for col in ["action", "observation.state"]:
            if col not in df.columns:
                continue
            arr = np.stack(df[col].values).astype(np.float32)
            print(f"\n  {col} (first chunk):")
            print(f"    mean: {arr.mean(0).round(2).tolist()}")
            print(f"    min : {arr.min(0).round(2).tolist()}")
            print(f"    max : {arr.max(0).round(2).tolist()}")

    print("""
  HOW TO READ:
    v2.1 convention — joint 1 mean ≈ 90–130°, joint 2 mean ≈ 100–130°
                      → transform IS needed at inference
    v3.0 convention — joint 1 mean ≈ −30 to +10°, joint 2 mean ≈ 0–40°
                      → transform is NOT needed at inference
""")


# ── 2. Transform preview on a sample frame ────────────────────────────────────

def show_transform(dataset_path: pathlib.Path):
    print("="*60)
    print("STEP 2 — What the v3.0→v2.1 transform does to a sample frame")
    print("="*60)

    # Try to grab a real frame from the dataset
    sample = None
    files = sorted(glob.glob(str(dataset_path / "data/**/*.parquet"),
                             recursive=True))
    if files:
        try:
            import pandas as pd
            df = pd.read_parquet(files[0])
            if "observation.state" in df.columns:
                sample = np.array(df["observation.state"].iloc[0], dtype=np.float32)
        except Exception:
            pass
    if sample is None:
        # Fallback: typical SO-101 resting position in v3.0 convention
        sample = np.array([2.4, -97.7, 99.3, 76.7, -49.8, 42.0], dtype=np.float32)
        print("  (using fallback resting-position state — no parquet available)")

    transformed = arm_to_model(sample)

    print(f"\n  Raw arm state (v3.0):        {sample.round(2).tolist()}")
    print(f"  After transform (v2.1 model): {transformed.round(2).tolist()}")
    print(f"\n  signs   = {SIGNS.tolist()}")
    print(f"  offsets = {OFFSETS.tolist()}")
    print(f"  formula : model_state = signs * arm_state + offsets")

    print("\n  Per-joint effect:")
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
             "wrist_flex", "wrist_roll", "gripper"]
    for i, name in enumerate(names):
        delta = transformed[i] - sample[i]
        sign_str = f"sign={int(SIGNS[i]):+d}" if SIGNS[i] != 1 else "sign=+1"
        offset_str = f"offset={int(OFFSETS[i]):+d}°" if OFFSETS[i] != 0 else "offset=0°"
        print(f"    [{i}] {name:16s}: {sample[i]:7.2f}° → {transformed[i]:7.2f}°  "
              f"({sign_str}, {offset_str})")

    # Sanity check: if v2.1 model expects values ~90-130° for shoulder/elbow at rest
    # the transformed values should land in that range.
    j1, j2 = float(transformed[1]), float(transformed[2])
    print(f"\n  Joint 1 after transform: {j1:.1f}°  "
          f"({'looks like v2.1 ✓' if 40 < j1 < 200 else 'unexpected — may already be v2.1 or different convention'})")
    print(f"  Joint 2 after transform: {j2:.1f}°  "
          f"({'looks like v2.1 ✓' if 40 < j2 < 200 else 'unexpected — may already be v2.1 or different convention'})")


# ── 3. Optional inference preview ────────────────────────────────────────────

def infer_preview(dataset_path: pathlib.Path, prompt: str):
    print("\n" + "="*60)
    print("STEP 3 — Inference preview: with vs without transform")
    print("="*60)
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from PIL import Image
    except ImportError as e:
        print(f"  Skipping: missing package ({e})")
        return

    # Grab a real frame from the dataset or use a blank image
    files = sorted(glob.glob(str(dataset_path / "data/**/*.parquet"),
                             recursive=True))
    sample_state = None
    if files:
        try:
            import pandas as pd
            df = pd.read_parquet(files[0])
            if "observation.state" in df.columns:
                sample_state = np.array(df["observation.state"].iloc[50],
                                        dtype=np.float32)
        except Exception:
            pass
    if sample_state is None:
        # Typical SO-101 resting position in v3.0 convention
        sample_state = np.array([2.4, -97.7, 99.3, 76.7, -49.8, 42.0], dtype=np.float32)

    # Blank images (we only care about the action distribution, not visual quality)
    dummy_img = Image.fromarray(np.zeros((378, 378, 3), dtype=np.uint8))

    print(f"\n  Sample arm state (v3.0):  {sample_state.round(2).tolist()}")
    state_v21 = arm_to_model(sample_state)
    print(f"  After transform (v2.1):   {state_v21.round(2).tolist()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Loading model (device={device})...")
    try:
        local_dir = snapshot_download("allenai/MolmoAct2-SO100_101")
        processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            local_dir, trust_remote_code=True, dtype=torch.bfloat16,
        ).to(device).eval()
    except Exception as e:
        print(f"  Could not load model: {e}")
        return

    def predict(state_arr):
        with torch.inference_mode():
            out = model.predict_action(
                processor=processor,
                images=[dummy_img, dummy_img],
                task=prompt,
                state=state_arr,
                norm_tag="so100_so101_molmoact2",
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=5,
                normalize_language=True,
                enable_cuda_graph=False,
            )
        acts = out.actions
        if torch.is_tensor(acts):
            acts = acts.detach().to("cpu", dtype=torch.float32).numpy()
        acts = np.asarray(acts, dtype=np.float32)
        if acts.ndim == 3:
            acts = acts[0]
        return acts  # (T, 6) in model frame

    print("\n  Running inference WITHOUT transform (raw v3.0 state → model)...")
    chunk_raw = predict(sample_state)
    chunk_raw_arm = model_to_arm(chunk_raw)  # convert output back to arm frame
    print(f"    first action in model frame: {chunk_raw[0].round(2).tolist()}")
    print(f"    first action in arm frame  : {chunk_raw_arm[0].round(2).tolist()}")
    print(f"    delta from current state   : {(chunk_raw_arm[0] - sample_state).round(2).tolist()}")

    print("\n  Running inference WITH transform (v3.0 → v2.1 → model → arm)...")
    chunk_tfm = predict(state_v21)
    chunk_tfm_arm = model_to_arm(chunk_tfm)
    print(f"    first action in model frame: {chunk_tfm[0].round(2).tolist()}")
    print(f"    first action in arm frame  : {chunk_tfm_arm[0].round(2).tolist()}")
    print(f"    delta from current state   : {(chunk_tfm_arm[0] - sample_state).round(2).tolist()}")

    print("""
  HOW TO READ:
    The delta shows how far each joint would move on the first step.
    With transform: deltas should be small (model "sees" a sensible arm pose).
    Without transform: deltas may be large or nonsensical (model is confused).

    If WITHOUT looks reasonable and WITH looks wrong, the data is v3.0 and no
    transform is needed. If WITH looks better, the data is v2.1 and the transform
    is required.
""")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="outputs/annotated-dataset-v7",
                   help="Path to the training dataset")
    p.add_argument("--infer", action="store_true",
                   help="Run inference preview (requires GPU + HF model cache)")
    p.add_argument("--prompt", default="Pick up the red truck and put it in the bowl",
                   help="Task prompt for inference preview")
    args = p.parse_args()

    dataset_path = pathlib.Path(args.dataset)
    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}\n"
                 f"Pass --dataset <path> to override.")

    check_dataset(dataset_path)
    show_transform(dataset_path)
    if args.infer:
        infer_preview(dataset_path, args.prompt)
    else:
        print("\n  (Run with --infer to also do the inference comparison.)")

if __name__ == "__main__":
    main()
```

Save as `CHECK_convention.py` in the LeRobot root, then run:

```bash
# Just dataset stats + transform preview (no GPU needed):
python CHECK_convention.py

# Full test including inference comparison (needs GPU + model in HF cache):
python CHECK_convention.py --infer
```
