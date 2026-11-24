"""Figures of the second round of outputs/analysis/layer_cosine_2026-11-16/REPORT.md, from the files layer_norms.py,
layer_instruction.py and layer_ability.py write there. No model, no GPU.

    uv run python lerobot/adhocs/text_probe/layer_report2.py --config_path=config_rl_validate.yaml \
        --frames=200,600,1000,1400,1800,2200 --clutter_frames=1180,2180,3215,4310,6040,8500
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.probes.utils import register_config_choices
from layer_sweep import SweepConfig

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})
OUT = "outputs/analysis/layer_cosine_2026-11-16"
NAMES = {"base": "base", "rebot-v2-tube-b128-20261114__001200": "tube 1200", "rebot-internal-only-20261021__001200": "internal 1200"}
COLOR = {"base": "#444444", "tube 1200": "#1f77b4", "internal 1200": "#d62728"}
CLASSES = ["high norm from block 0", "high norm later, image", "high norm later, text"]
CLASS_COLOR = {"high norm from block 0": "#d62728", "high norm later, image": "#ff7f0e", "high norm later, text": "#9467bd", "other positions": "#888888"}
LAYERS = np.arange(36)
LENS_LAYERS = [0, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 34, 35]
WIRES = "observation.images.top", "observation.images.wrist"


def read(name: str):
    return torch.load(os.path.join(OUT, name + ".pt"), weights_only=False)


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def rms(x: torch.Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt())


def family(runs: dict, prefix: str) -> list[dict]:
    return [run for name, run in runs.items() if name.startswith(prefix)]


def origin(trace: dict) -> None:
    """Where the norm comes from: vision path, language-model blocks, and which sublayer writes it."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2), gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    stages = [f"vit {i:02d}" for i in range(25)] + ["pooled", "projected"]
    for key, name in NAMES.items():
        photos = family(trace[key], "chat.")
        top = [np.mean([float(run["vision"][s].max()) for run in photos]) for s in stages]
        median = [np.mean([float(run["vision"][s].median()) for run in photos]) for s in stages]
        axes[0].plot(top, color=COLOR[name], label=f"{name}: largest token")
        axes[0].plot(median, color=COLOR[name], ls=":", label=f"{name}: median token")
    axes[0].set_yscale("log")
    axes[0].set_xticks([0, 8, 9, 16, 24, 25, 26], ["0", "8", "9", "16", "24", "pool", "proj"])
    axes[0].set_xlabel("vision block, then pooling and projector")
    axes[0].set_ylabel("L2 norm of a token")
    axes[0].set_title(f"vision path, {len(photos)} photos per checkpoint")
    axes[0].legend(fontsize=6)
    for ax, (key, name) in zip(axes[1:], NAMES.items()):
        runs = family(trace[key], "act.")
        for cls in CLASSES:
            norm = torch.stack([run["norm"][:, run["members"][cls]].mean(dim=1) for run in runs if run["members"].get(cls)]).mean(dim=0)
            mlp = torch.stack([run["mlp"][:, run["members"][cls]].mean(dim=1) for run in runs if run["members"].get(cls)]).mean(dim=0)
            ax.plot(LAYERS, norm, color=CLASS_COLOR[cls], label=cls)
            ax.plot(LAYERS, mlp.clamp_min(1), color=CLASS_COLOR[cls], ls=":", lw=0.9)
        rest = torch.stack([run["norm"][:, [i for c, index in run["members"].items() if c not in CLASSES for i in index]].median(dim=1).values for run in runs]).mean(dim=0)
        ax.plot(LAYERS, rest, color=CLASS_COLOR["other positions"], label="other positions (median)")
        ax.set_yscale("log")
        ax.set_ylim(1, 4e5)
        ax.set_xlabel("language-model block")
        ax.set_title(f"{name}, {len(runs)} action prompts")
    axes[1].legend(fontsize=6)
    fig.suptitle("Language-model panels: solid = norm of the block output, dotted = norm of what the block's MLP adds", fontsize=9, y=1.03)
    save(fig, "fig10_norm_origin.png")


def patch_map(trace: dict, cfg: SweepConfig) -> None:
    """Which image patches have a high norm, drawn on the top and wrist camera frames."""
    tube = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.val_dataset_path)
    frames = [f"robot_{f}" for f in cfg.frames.split(",")][:4]
    fig, axes = plt.subplots(len(NAMES), 2 * len(frames), figsize=(2.3 * 2 * len(frames), 2.0 * len(NAMES)))
    for row, (key, name) in enumerate(NAMES.items()):
        for col, frame in enumerate(frames):
            run = trace[key][f"act.{frame}"]
            groups = run["groups"]
            starts = [i for i, g in enumerate(groups) if g in ("image", "masked") and groups[i - 1] not in ("image", "masked")]
            item = tube[int(frame.split("_")[1])]
            for slot, wire in ((0, WIRES[0]), (2, WIRES[1])):
                ax = axes[row, 2 * col + (slot > 0)]
                picture = (item[wire].permute(1, 2, 0) * 255).round().byte().numpy()
                ax.imshow(picture, extent=(0, 14, 14, 0), aspect="auto")
                for cls in CLASSES[:2]:
                    for at in run["members"].get(cls, []):
                        patch = at - starts[slot]
                        if 0 <= patch < 196:
                            ax.add_patch(plt.Rectangle((patch % 14, patch // 14), 1, 1, fill=False, edgecolor=CLASS_COLOR[cls], lw=1.4))
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"frame {frame.split('_')[1]}, {'top' if slot == 0 else 'wrist'} camera")
            axes[row, 0].set_ylabel(name)
    fig.suptitle("Image patches (14 x 14 per camera) with a high norm: red = from block 0 (same two patches in every image), orange = written at block 16", fontsize=9, y=1.02)
    save(fig, "fig11_patch_map.png")


def attention(trace: dict) -> None:
    """Share of the language model's attention, from the text positions after the last image, per class."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3), sharey=True)
    for ax, (key, name) in zip(axes, NAMES.items()):
        runs = family(trace[key], "act.")
        for cls in CLASSES:
            mass = torch.stack([run["mass"][cls].mean(dim=1) for run in runs if cls in run["mass"]]).mean(dim=0)
            count = np.mean([len(run["members"].get(cls, [])) for run in runs])
            ax.plot(LAYERS, mass, color=CLASS_COLOR[cls], label=f"{cls} ({count:.0f} positions)")
        rest = torch.stack([run["mass"]["image"].mean(dim=1) for run in runs]).mean(dim=0)
        count = np.mean([len(run["members"]["image"]) for run in runs])
        ax.plot(LAYERS, rest, color=CLASS_COLOR["other positions"], label=f"other image positions ({count:.0f})")
        ax.set_title(f"{name}, {len(runs)} action prompts")
        ax.set_xlabel("language-model block")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("share of attention (mean over heads)")
    for key, name in NAMES.items():
        runs = family(trace[key], "text.")
        mass = torch.stack([sum(run["mass"][cls].mean(dim=1) for cls in CLASSES if cls in run["mass"]) for run in runs]).mean(dim=0)
        axes[3].plot(LAYERS, mass, color=COLOR[name], label=name)
    axes[3].set_title(f"text-only prompts ({len(runs)} per checkpoint):\nattention on the 2-3 high-norm tokens")
    axes[3].set_xlabel("language-model block")
    axes[3].legend(fontsize=6)
    save(fig, "fig12_attention.png")


def decoding(trace: dict) -> None:
    """Probability of the action tokens when each block's output is decoded through the final norm and lm_head."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for ax, (prefix, title) in zip(axes, (("act.", "action prompts, text positions"), ("chat.", "'Describe this image.' on photos, text positions"), ("text.", "text-only prompts"))):
        for key, name in NAMES.items():
            runs = family(trace[key], prefix)
            rows = [run["action_mass"][:, [i for i, g in enumerate(run["groups"]) if g not in ("image", "masked")]].mean(dim=1) for run in runs]
            ax.plot(LENS_LAYERS, torch.stack(rows).mean(dim=0), color=COLOR[name], marker="o", ms=3, label=name)
        ax.set_title(f"{title} ({len(runs)} prompts per checkpoint)")
        ax.set_xlabel("language-model block whose output is decoded")
    axes[0].set_ylabel("probability on the action tokens")
    axes[0].legend(fontsize=7)
    save(fig, "fig13_action_token_decoding.png")


def share(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> float:
    """Projection of the change x - a on b - a; 1 = the full change from a to b."""
    d = (b - a).float().flatten()
    return float(((x - a).float().flatten() @ d) / (d @ d).clamp_min(1e-9))


def instruction(result: dict, given: dict | None) -> None:
    """By which block the instruction has left its own tokens, and how far the chunk moves when the difference
    between the two instructions is scaled in what the action expert reads."""
    from layer_instruction import ALPHAS, PAIRS
    from layer_instruction import LAYERS as AT

    fig, axes = plt.subplots(1, 4 if given else 3, figsize=(18 if given else 12, 3.4))
    style = {PAIRS[0]: "-", PAIRS[1]: "--"}
    words = {PAIRS[0]: "move left / right", PAIRS[1]: "grasp tube on the left / right"}
    for key, name in NAMES.items():
        for pair in PAIRS:
            entries = [e for (_, a, b, _), e in result[key].items() if (a, b) == pair]
            tag = f"{name}, {words[pair]}"
            for ax, kind in zip(axes, ("subtask", "after")):
                ax.plot(AT, [np.mean([share(e["residual"][(kind, layer)], e["a"], e["b"]) for e in entries]) for layer in AT], color=COLOR[name], ls=style[pair], marker="o", ms=2.5, label=tag)
            sizes = [np.mean([rms(e["scale"][alpha] - e["a"]) for e in entries]) for alpha in ALPHAS]
            sizes.insert(1, np.mean([rms(e["b"] - e["a"]) for e in entries]))
            axes[2].plot([0.5, 1, 2, 4, 8], sizes, color=COLOR[name], ls=style[pair], marker="o", ms=2.5)
    axes[0].set_title(f"one block's output taken from the other instruction,\nat the instruction tokens ({len(entries)} frames x seeds per line)")
    axes[1].set_title("same, at every position after the instruction")
    for ax in axes[:2]:
        ax.set_xlabel("language-model block that is replaced")
    axes[0].set_ylabel("share of the chunk change")
    axes[0].legend(fontsize=6, loc="lower left")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    axes[2].set_xlabel("factor on the difference (1 = the other instruction)")
    axes[2].set_ylabel("chunk change (RMS, degrees)")
    axes[2].set_title("keys and values the action expert reads =\none instruction + factor x (other - one)")
    if given:
        rows, labels = [], []
        for key, name in NAMES.items():
            if key not in given:
                continue
            for pair in PAIRS:
                entries = [e for (_, a, b, _), e in given[key].items() if (a, b) == pair]
                other = next(iter(entries[0]["donor"]))
                rows.append([np.mean([rms(e["b"] - e["a"]) for e in entries]), np.mean([rms(e["donor"][other]["difference"] - e["a"]) for e in entries]),
                             np.mean([rms(e["donor"][other]["b"] - e["donor"][other]["a"]) for e in entries])])
                labels.append(f"{name} expert,\n{words[pair].replace(' on', chr(10) + 'on')}")
        rows = np.array(rows)
        for i, (label, color) in enumerate((("own language model", "#444444"), ("own + the other checkpoint's difference", "#ff7f0e"), ("the other checkpoint's language model, both runs", "#1f77b4"))):
            axes[3].bar(np.arange(len(rows)) + 0.27 * (i - 1), rows[:, i], 0.27, color=color, label=label)
        axes[3].set_xticks(np.arange(len(rows)), labels, fontsize=6)
        axes[3].set_ylabel("chunk change (RMS, degrees)")
        axes[3].set_title(f"chunk change between the two instructions, by whose\nkeys and values the expert reads ({len(entries)} frames x seeds per bar)")
        axes[3].set_ylim(0, 0.95)
        axes[3].legend(fontsize=6, loc="upper right")
    save(fig, "fig14_instruction_route.png")


def chunk_edits(result: dict) -> None:
    """The chunk when high-norm positions are edited in the language model or hidden from the action expert."""
    from layer_norms import FRACTIONS

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    bars = ["other seed", "lm: norm clamped", "lm: writes undone", "lm: writes undone, image positions only", "lm: writes undone, other positions only",
            "expert: high-norm keys and values from another frame", "expert: as many ordinary image keys and values from another frame"]
    words = ["another flow seed", "LM: norm clamped\nto the median", "LM: block updates that\ncreate a high norm undone", "same, image\npositions only", "same, text\npositions only",
             "expert: keys and values at\nhigh-norm positions\nfrom another frame", "expert: same at as many\nordinary image positions"]
    for i, (key, name) in enumerate(NAMES.items()):
        entries = list(result[key].values())
        change = lambda k: np.mean([rms(e[k] - e["clean"]) for e in entries if k in e])  # noqa: E731
        axes[0].barh(np.arange(len(bars)) + 0.27 * (i - 1), [change(k) for k in bars], 0.27, color=COLOR[name], label=f"{name} ({len(entries)} frames x seeds)")
        ax = axes[1 + i]
        ax.plot(FRACTIONS, [change(f"expert: {f:.2f} of the high-norm positions hidden") for f in FRACTIONS], color="#ff7f0e", marker="o", ms=3, label="high-norm positions")
        ax.plot(FRACTIONS, [change(f"expert: as many ordinary image positions hidden ({f:.2f})") for f in FRACTIONS], color="#888888", marker="o", ms=3, label="as many ordinary image positions")
        ax.axhline(change("other seed"), color="k", lw=0.8, ls=":", label="another flow seed")
        ax.set_title(f"{name}: hidden from the action expert\n({np.mean([e['count'] for e in entries]):.0f} high-norm positions per prompt)")
        ax.set_xlabel("fraction hidden")
    axes[0].set_yticks(np.arange(len(bars)), words, fontsize=6)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("chunk change (RMS, degrees)")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=6)
    axes[1].set_ylabel("chunk change (RMS, degrees)")
    axes[1].legend(fontsize=6)
    save(fig, "fig15_chunk_edits.png")


def ability(result: dict) -> None:
    """Correct answers per checkpoint: unedited, with the vision path of another checkpoint, and under the edits
    of the high-norm positions."""
    kinds = {"yes / no": "is there a ...? (yes / no)", "open": "one-word answer about the photo", "facts": "text-only fact"}
    color = {"yes / no": "#1f77b4", "open": "#ff7f0e", "facts": "#2ca02c"}
    rate = lambda rows, kind: np.mean([x["correct"] for x in rows[kind]])  # noqa: E731
    short = lambda key: "base" if key == "base" else ("tube " if "tube" in key else "internal ") + str(int(key.split("__")[1]))  # noqa: E731
    keys = list(result)
    fig, axes = plt.subplots(1, 3, figsize=(18, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.9]})
    for kind, words in kinds.items():
        count = len(result["base"]["plain"][kind])
        axes[0].plot(range(len(keys)), [rate(result[k]["plain"], kind) for k in keys], color=color[kind], marker="o", ms=3, label=f"{words} ({count} questions)")
        if kind != "facts":
            axes[0].plot(range(1, len(keys)), [rate(result[k]["vision from base"], kind) for k in keys[1:]], color=color[kind], ls="--", marker="o", ms=3, label="same, vision path taken from base")
            axes[1].plot(range(len(keys)), [rate(result["base"]["plain"], kind)] + [rate(result["base"][f"vision from {k}"], kind) for k in keys[1:]], color=color[kind], marker="o", ms=3, label=words)
    axes[0].set_title("each checkpoint's language model")
    axes[1].set_title("base language model, vision path taken from each checkpoint")
    for ax in axes[:2]:
        ax.set_xticks(range(len(keys)), [short(k).replace(" ", "\n") for k in keys], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, loc="lower left")
    axes[0].set_ylabel("share of correct answers (action tokens masked)")
    conditions = [c for c in result["base"] if not c.startswith("vision from")]
    shown = [k for k in keys if k in NAMES]
    for i, key in enumerate(shown):
        for j, (kind, hatch) in enumerate((("yes / no", None), ("facts", "//"))):
            values = [rate(result[key][c], kind) if result[key][c].get(kind) else 0 for c in conditions]
            axes[2].barh(np.arange(len(conditions)) + 0.14 * (2 * i + j - 2.5), values, 0.14, color=COLOR[NAMES[key]], hatch=hatch, alpha=1 if j == 0 else 0.55,
                         label=f"{NAMES[key]}, {'yes / no on photos' if j == 0 else 'text-only facts'}")
    axes[2].set_yticks(np.arange(len(conditions)), ["no edit" if c == "plain" else c.replace(", every block", ",\nevery block").replace("that create", "that\ncreate").replace("undone at", "undone\nat").replace("set to", "\nset to") for c in conditions], fontsize=6)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("share of correct answers")
    axes[2].set_title("edits inside the language model (no bar = not run: the edit needs an image)")
    axes[2].set_xlim(0, 1.42)
    axes[2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[2].legend(fontsize=6, loc="center right")
    save(fig, "fig16_ability.png")


def run(cfg: SweepConfig) -> None:
    trace = read("norms_trace")
    origin(trace)
    patch_map(trace, cfg)
    attention(trace)
    decoding(trace)
    donors = os.path.join(OUT, "instruction_donors.pt")
    instruction(read("instruction"), read("instruction_donors") if os.path.exists(donors) else None)
    chunk_edits(read("norms_chunks"))
    ability(read("ability"))


@parser.wrap()
def cli(cfg: SweepConfig):
    run(cfg)


def main() -> None:
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    register_config_choices()
    main()
