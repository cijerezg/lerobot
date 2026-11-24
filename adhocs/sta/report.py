"""Tables and figures of ``sta.py``.

    uv run python lerobot/adhocs/sta/report.py outputs/analysis/sta_2026-11-23/m0_joints.pt --blocks 7,14,21,28

Prints the capture table of ``blocks`` (index into the action expert's blocks, from 0) and writes next to the file:
``score.png`` (layer score against depth, one panel per kind of layer), ``capture.png`` (capture against depth, ours
and the top singular vectors of W) and ``spectrum_<model>.png`` (lam of A_D and of A_P2 against B, ``blocks`` only).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

KINDS = ["self", "cross", "mlp"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # one per model, in the file's order
COLUMNS = ["ours", "W singular", "V = A_D top, U ours", "V ours, U random", "random", "random, expected"]

parser = argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument("--blocks", default="7,14,21,28")
args = parser.parse_args()
result, blocks, folder = torch.load(args.file), [int(b) for b in args.blocks.split(",")], os.path.dirname(args.file)
models = result["models"]
depth = len(next(iter(models.values()))["layers"]) // len(KINDS)

print("frames", result["frames"], "| k", result["k"])
for name, model in models.items():
    print(f"{name}: flow loss {model['flow_loss']} | inputs per layer {model['inputs']}")
print("\n| layer | model | kept | score | lam 1 | lam null | " + " | ".join(COLUMNS) + " |\n|" + "---|" * (6 + len(COLUMNS)))
for block in blocks:
    for kind in KINDS:
        for name, model in models.items():
            layer = model["layers"][f"{block}.{kind}"]
            capture = " | ".join(f"{layer['capture'][c]:.1e}" if layer["capture"] else "" for c in COLUMNS)
            print(f"| {block}.{kind} | {name} | {layer['kept']} | {layer['score']:.1f} | {layer['lam'][0]:.1f} | {layer['null'][0]:.2f} | {capture} |")

JOINTS = ["pan", "lift", "elbow", "wrist flex", "wrist yaw", "wrist roll", "gripper"]
SHARES = [0.9, 0.99, 0.999]


def left(old, share, key):
    """What ``key`` keeps once the largest directions of the other data holding ``share`` of its energy are removed."""
    rank = int((old["other"].cumsum(0) < share).sum()) + 1
    return 1 - old[key][..., :rank].sum(-1)


print("\nshare of sum_j |G_j|^2 per joint | of |G|^2 left without the other data's directions holding 0.9 / 0.99 / 0.999 of its energy (task input energy left)")
print("| layer | model | " + " | ".join(JOINTS) + " | " + " | ".join(str(s) for s in SHARES) + " | per joint at 0.99 |\n|" + "---|" * (3 + len(JOINTS) + len(SHARES)))
for block in blocks:
    for kind in KINDS:
        for name, model in models.items():
            layer = model["layers"][f"{block}.{kind}"]
            squared = layer["joint_gradient_norm"] ** 2
            joints = " | ".join(f"{v:.2f}" for v in (squared / squared.sum()).tolist())
            kept = " | ".join(f"{left(layer['old'], s, 'gradient'):.3f} ({left(layer['old'], s, 'task'):.3f})" for s in SHARES)
            per_joint = " ".join(f"{v:.2f}" for v in left(layer["old"], 0.99, "joint_gradient").tolist())
            print(f"| {block}.{kind} | {name} | {joints} | {kept} | {per_joint} |")
for name, model in models.items():
    total = sum(layer["gradient_norm"] ** 2 for layer in model["layers"].values())
    print(name, "all layers:", ", ".join(f"{s}: {sum(float(left(layer['old'], s, 'gradient')) * layer['gradient_norm'] ** 2 for layer in model['layers'].values()) / total:.3f}" for s in SHARES))


def panels(title, value, file, log=True):
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for axis, kind in zip(axes, KINDS):
        for (name, model), color in zip(models.items(), COLORS):
            for style, key in value.items():
                ys = [key(model["layers"][f"{i}.{kind}"]) for i in range(depth)]
                axis.plot(range(depth), ys, style, color=color, linewidth=2, markersize=3, label=f"{name}{'' if style == '-' else ', W singular'}")
        axis.set_title(kind, loc="left", fontsize=10)
        axis.set_xlabel("block of the action expert")
        if log:
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(title)
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(folder, file), dpi=150)
    plt.close(figure)


panels("layer score", {"-": lambda layer: layer["score"] or float("nan")}, "score.png")
panels("capture", {"-": lambda layer: layer["capture"].get("ours", float("nan")), ":": lambda layer: layer["capture"].get("W singular", float("nan"))}, "capture.png")

for name, model in models.items():
    figure, axes = plt.subplots(len(KINDS), len(blocks), figsize=(3.4 * len(blocks), 7.5), sharex=True)
    for row, kind in zip(axes, KINDS):
        for axis, block in zip(row, blocks):
            layer = model["layers"][f"{block}.{kind}"]
            axis.plot(layer["lam"][:64], color=COLORS[0], linewidth=2, label="A_D")
            axis.plot(layer["null"][:64], color="#888888", linewidth=2, label="A_P2")
            axis.axhline(1.5 * float(layer["null"][0]), color="#888888", linewidth=1, linestyle=":")
            axis.set_yscale("log")
            axis.set_title(f"{block}.{kind}, kept {layer['kept']}", loc="left", fontsize=10)
            axis.grid(alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("index, largest first")
    axes[0, 0].set_ylabel("lam against B")
    axes[0, -1].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{name}: dotted line = 1.5 x the largest lam of A_P2", x=0.01, ha="left", fontsize=10)
    figure.tight_layout()
    figure.savefig(os.path.join(folder, f"spectrum_{name}.png"), dpi=150)
    plt.close(figure)
print("\nfigures in", folder)
