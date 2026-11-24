"""How far each part of a finetuned checkpoint's weights is from the base model. No model is built, no GPU.

    uv run python lerobot/adhocs/text_probe/layer_weights.py <base dir> <name>=<pretrained_model dir> ... --out=<file.pt>

Per tensor: |finetuned - base| / |base| (base cast to the finetuned dtype first, so a weight that never trained
gives exactly 0). Saved per tensor and summed per part (LM block, action expert block, vision block, ...).
"""
from __future__ import annotations

import json
import os
import re
import sys

import torch
from safetensors import safe_open


def part_of(key: str) -> str:
    match = re.match(r"(.*?\.(?:blocks|resblocks|layers)\.\d+)\.", key)
    return match.group(1) if match else key.rsplit(".", 1)[0]


def main() -> None:
    base_dir = sys.argv[1]
    runs = dict(arg.split("=", 1) for arg in sys.argv[2:] if not arg.startswith("--"))
    out = [arg.split("=", 1)[1] for arg in sys.argv[2:] if arg.startswith("--out=")][0]
    shard_of = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]
    shards = {name: safe_open(os.path.join(base_dir, name), "pt") for name in set(shard_of.values())}
    result = {}
    for name, folder in runs.items():
        tuned = safe_open(os.path.join(folder, "model.safetensors"), "pt")
        tensors, parts, unmatched = {}, {}, []
        for key in tuned.keys():
            base_key = next((k for k in (key, key.removeprefix("model."), key.removeprefix("model.model.")) if k in shard_of), None)
            if base_key is None:
                unmatched.append(key)
                continue
            new = tuned.get_tensor(key)
            old = shards[shard_of[base_key]].get_tensor(base_key).to(new.dtype).float()
            new = new.float()
            change, size = float((new - old).square().sum()), float(old.square().sum())
            tensors[base_key] = (change**0.5 / max(size**0.5, 1e-12), new.numel())
            entry = parts.setdefault(part_of(base_key), [0.0, 0.0])
            entry[0] += change
            entry[1] += size
        result[name] = {"tensors": tensors, "parts": {k: (c / max(s, 1e-24)) ** 0.5 for k, (c, s) in parts.items()}, "unmatched": unmatched}
        print(name, len(tensors), "tensors,", len(unmatched), "without a base tensor", flush=True)
    torch.save(result, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
