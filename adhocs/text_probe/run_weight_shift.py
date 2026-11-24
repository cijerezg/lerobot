"""Weight shift: how far did finetuning move each part of MolmoAct2 away from the base checkpoint?

    uv run python lerobot/adhocs/text_probe/run_weight_shift.py \
        --checkpoints=outputs/runs/rebot-v2-tube-b128-20261114/checkpoints/001200/pretrained_model,outputs/runs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model \
        --out_dir=outputs/analysis/text_probe/weight_shift_2026-11-15

Per tensor shared with the base: delta = checkpoint - base, with the base rounded to bfloat16 first (the
checkpoints are stored in bfloat16, so an untouched tensor gives exactly 0). Tensors are pooled into groups
(language model block k, vision tower, image projector, action expert, ``lm_head`` text rows / action rows,
input embedding) and each group reports

- shift = |delta| / |base|,
- toward zero = -cos(delta, base): the part of the move that is plain shrinking, which weight decay produces,
- for the first two checkpoints, cos(delta_1, delta_2): do the two runs move the same way.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from safetensors import safe_open

from run_text_probe import label

BASE = "outputs/models/MolmoAct2"
ACTION_ROWS = slice(151931, 151934 + 2048)  # action output / start / end + the FAST bins (base config.json)
SUMS = ("delta2", "base2", "delta_base", "cross", "other2")


def group(key: str) -> str:
    block = re.match(r"model\.transformer\.blocks\.(\d+)\.", key)
    if block:
        return f"llm block {int(block.group(1)):02d}"
    for prefix, name in (
        ("model.transformer.wte", "input embedding"),
        ("model.transformer", "llm final norm"),
        ("model.vision_backbone.image_vit", "vision tower"),
        ("model.vision_backbone", "image projector"),
        ("model.action_expert", "action expert"),
    ):
        if key.startswith(prefix):
            return name
    return key


def run(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
        shard_of = json.load(f)["weight_map"]
    shards = {name: safe_open(os.path.join(BASE, name), "pt", device=args.device) for name in set(shard_of.values())}
    paths = args.checkpoints.split(",")
    names = [label(path) for path in paths]
    files = [safe_open(os.path.join(path, "model.safetensors"), "pt", device=args.device) for path in paths]

    # sums[group][checkpoint] -> running sums of squares and dot products
    sums = defaultdict(lambda: [dict.fromkeys(SUMS, 0.0) for _ in names])
    for key, shard in shard_of.items():
        if "rotary_emb" in key:
            continue
        base = shards[shard].get_tensor(key).to(torch.bfloat16).double()
        deltas = [file.get_tensor("model." + key).double() - base for file in files]
        parts = [(group(key), slice(None))]
        if key == "lm_head.weight":
            text_rows = torch.ones(base.shape[0], dtype=torch.bool, device=base.device)
            text_rows[ACTION_ROWS] = False
            parts = [("lm_head text rows", text_rows), ("lm_head action rows", ACTION_ROWS)]
        for name, rows in parts:
            for i, delta in enumerate(deltas):
                s = sums[name][i]
                s["delta2"] += float((delta[rows] ** 2).sum())
                s["base2"] += float((base[rows] ** 2).sum())
                s["delta_base"] += float((delta[rows] * base[rows]).sum())
                if i == 0 and len(deltas) > 1:
                    s["cross"] += float((delta[rows] * deltas[1][rows]).sum())
                    s["other2"] += float((deltas[1][rows] ** 2).sum())

    rows = []
    for name in sorted(sums):
        row = {"group": name}
        for checkpoint, s in zip(names, sums[name]):
            row[checkpoint] = {
                "shift": (s["delta2"] / s["base2"]) ** 0.5,
                "toward_zero": -s["delta_base"] / max((s["delta2"] * s["base2"]) ** 0.5, 1e-30),
            }
        s = sums[name][0]
        if len(names) > 1:
            row["cos_first_two"] = s["cross"] / max((s["delta2"] * s["other2"]) ** 0.5, 1e-30)
        rows.append(row)
    with open(os.path.join(args.out_dir, "weight_shift.json"), "w") as f:
        json.dump(rows, f, indent=2)

    head = "| group | " + " | ".join(f"{n} shift | {n} toward zero" for n in names) + (" | cos(first two) |" if len(names) > 1 else " |")
    lines = [head, "|---|" + "---|---|" * len(names) + ("---|" if len(names) > 1 else "")]
    for row in rows:
        cells = " | ".join(f"{row[n]['shift']:.4f} | {row[n]['toward_zero']:+.2f}" for n in names)
        lines.append(f"| {row['group']} | {cells}" + (f" | {row['cos_first_two']:+.3f} |" if len(names) > 1 else " |"))
    with open(os.path.join(args.out_dir, "weight_shift.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    blocks = [row for row in rows if row["group"].startswith("llm block")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for n in names:
        axes[0].plot([row[n]["shift"] for row in blocks], label=n)
    axes[0].set(xlabel="language model block", ylabel="|delta| / |base|", title="shift from the base")
    axes[0].legend(fontsize=6)
    if len(names) > 1:
        axes[1].plot([row["cos_first_two"] for row in blocks], color="black")
        axes[1].set(xlabel="language model block", ylabel="cosine", title=f"cos(delta {names[0]}, delta {names[1]})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "weight_shift.png"), dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="comma list of pretrained_model dirs; the first two are also compared with each other")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())
