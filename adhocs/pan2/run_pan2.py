"""Generate PAN 2 videos from a prompts JSON.

    venv/bin/python run_pan2.py prompts.json out/ [--steps 50] [--seed 0]

prompts.json: [{"name": ..., "prompt": ..., "negative_prompt": ..., "image": path, "seconds": 3}, ...]
`image`, `negative_prompt` and `seconds` are optional. Finished clips are skipped.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import ModularPipeline
from diffusers.utils import export_to_video, load_image

parser = argparse.ArgumentParser()
parser.add_argument("prompts")
parser.add_argument("out")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--seconds", type=float, default=3.0)
args = parser.parse_args()

records = json.loads(Path(args.prompts).read_text())
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

t0 = time.time()
pipe = ModularPipeline.from_pretrained("IFM/PAN2")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")
print(f"loaded in {time.time() - t0:.0f} s", flush=True)

for record in records:
    path = out / f"{record['name']}.mp4"
    if path.exists():
        continue
    num_frames = int(round(record.get("seconds", args.seconds) * 24 / 4)) * 4 + 1
    kwargs = dict(
        prompt=record["prompt"],
        num_frames=num_frames,
        num_inference_steps=args.steps,
        generator=torch.Generator().manual_seed(args.seed),
        output="videos",
    )
    if "negative_prompt" in record:
        kwargs["negative_prompt"] = record["negative_prompt"]
    for key in ["height", "width"]:
        if key in record:
            kwargs[key] = record[key]
    if "image" in record:
        kwargs["image"] = load_image(record["image"])
    t0 = time.time()
    video = pipe(**kwargs)[0]
    export_to_video(video, str(path), fps=24)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{path}  {num_frames} frames  {time.time() - t0:.0f} s  peak {peak:.1f} GiB", flush=True)
