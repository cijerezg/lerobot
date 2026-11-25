"""Relative weight change per layer between base MolmoAct2 and a checkpoint.

For each tensor: |checkpoint - base| / |base| (Frobenius), base cast to the checkpoint's dtype
first so frozen weights give exactly 0. Tensors are grouped per block and per part of the block
(the group value pools the squared norms of its tensors).

Run from the workspace root:
    uv run python lerobot/adhocs/weight_change/weight_change.py <checkpoint_dir> [<checkpoint_dir> ...]
Writes outputs/analysis/weight_change/<run>_<step>.json and prints the table.
"""

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

BASE = "outputs/models/MolmoAct2"
OUT = Path("outputs/analysis/weight_change")

LM = re.compile(r"model\.transformer\.blocks\.(\d+)\.(self_attn|mlp|attn_norm|ff_norm)")
EXPERT = re.compile(r"model\.action_expert\.blocks\.(\d+)\.(self_attn|cross_attn|mlp|modulation)")
PARTS = {"lm": ["self_attn", "mlp", "norms"], "expert": ["self_attn", "cross_attn", "mlp", "modulation"]}


def group(key: str) -> tuple[str, int, str]:
    if m := LM.match(key):
        part = "norms" if m[2].endswith("_norm") else m[2]
        return "lm", int(m[1]), part
    if m := EXPERT.match(key):
        return "expert", int(m[1]), m[2]
    return "other", 0, re.sub(r"\.\d+\.", ".N.", key).rsplit(".", 1)[0].split("resblocks")[0]


def main(checkpoint: str) -> None:
    base = {}
    for path in sorted(glob.glob(f"{BASE}/model-*.safetensors")):
        handle = safe_open(path, "pt")
        for key in handle.keys():
            base[key] = handle
    new = safe_open(f"{checkpoint}/pretrained_model/model.safetensors", "pt")

    diff, norm = defaultdict(float), defaultdict(float)
    missing = []
    for key in new.keys():
        base_key = key.removeprefix("model.")
        if base_key not in base:
            missing.append(key)
            continue
        w = new.get_tensor(key)
        w0 = base[base_key].get_tensor(base_key).to(w.dtype)
        g = group(base_key)
        diff[g] += (w.float() - w0.float()).square().sum().item()
        norm[g] += w0.float().square().sum().item()
    rel = {g: (diff[g] / norm[g]) ** 0.5 for g in diff}

    step = Path(checkpoint).name
    run = Path(checkpoint).parent.parent.name
    print(f"\n{run} step {step}: relative weight change |w - w0| / |w0|")
    for stack, parts in PARTS.items():
        print(f"{stack:>6} " + " ".join(f"{p:>10}" for p in [*parts, "block"]))
        for layer in range(36):
            block = (sum(diff[stack, layer, p] for p in parts) / sum(norm[stack, layer, p] for p in parts)) ** 0.5
            print(f"{layer:>6} " + " ".join(f"{rel[stack, layer, p]:10.4f}" for p in parts) + f" {block:10.4f}")
    print("other")
    for (stack, _, part), value in sorted(rel.items()):
        if stack == "other":
            print(f"  {value:8.4f}  {part}")
    print("not in base:", missing)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"stack": s, "layer": layer, "part": p, "relative_change": v, "base_norm": norm[s, layer, p] ** 0.5} for (s, layer, p), v in sorted(rel.items())]
    (OUT / f"{run}_{step}.json").write_text(json.dumps({"checkpoint": checkpoint, "rows": rows, "not_in_base": missing}, indent=1))


for checkpoint in sys.argv[1:]:
    main(checkpoint.rstrip("/"))
