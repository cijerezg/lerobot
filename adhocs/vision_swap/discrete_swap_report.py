"""Table of ``discrete_swap.py`` (no model).

    uv run python lerobot/adhocs/vision_swap/discrete_swap_report.py outputs/analysis/vision_swap_2026-11-19/discrete_swap.pt

Chunks: the tip path of every chunk (filtered as deployed, forward kinematics, cm) against the reference chunk
(continuous: same flow seed; discrete: the one decoded chunk), mean over the chunk. Losses on the recorded chunk:
the mean over frames, and the mean absolute change per frame against the reference. Each column is read against
its own rows "next" and "other" (a real change of the camera images). Writes ``<name>.md`` next to the file.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.action_smoothing import apply_butterworth_filter

NAMES = {"reference": "reference", "base": "base vision path", "random": "random direction, same size", "next": "camera images 1 s later", "other": "camera images of another episode"}


def main(path: str) -> None:
    data = torch.load(path, weights_only=False)
    frames, cells, joints = data["frames"], ["reference", *data["conditions"]], len(data["joints"])
    kinematics = RebotKinematics()

    def tips(chunks):  # [..., steps, joints] degrees -> [..., steps, 3] cm
        smooth = np.moveaxis(apply_butterworth_filter(np.moveaxis(chunks, -2, 0)), 0, -2)
        return kinematics.ee_path(smooth.reshape(-1, joints)).reshape(*smooth.shape[:-1], 3) * 100

    continuous = tips(np.stack([np.stack([f["continuous"][c].double().numpy() for f in frames]) for c in cells])[..., :joints])  # [cells, frames, seeds, steps, 3]
    discrete = tips(np.stack([np.stack([f["discrete"][c].double().numpy() for f in frames]) for c in cells])[..., :joints])  # [cells, frames, steps, 3]
    recorded = tips(data["recorded"].double().numpy()[: len(frames), :, :joints])  # a file of a running job has fewer frames
    seeds = continuous.shape[2]
    pairs = [(i, j) for i in range(seeds) for j in range(seeds) if i < j]
    two_seeds = np.mean([np.linalg.norm(continuous[0][:, i] - continuous[0][:, j], axis=-1).mean() for i, j in pairs])
    failed = np.isnan(discrete).any(axis=(-1, -2))  # [cells, frames]: the generated tokens did not decode
    between = np.nanmean(np.linalg.norm(continuous[0].mean(axis=1) - discrete[0], axis=-1))

    def loss(cell, key):
        return np.array([float(torch.as_tensor(f["losses"][cell][key]).float().mean()) for f in frames])

    def agree(cell):  # share of the generated chunk's joint values equal to the reference's
        return np.mean([float((f["discrete"][cell] == f["discrete"]["reference"]).all()) for f in frames])

    lines = [f"# Discrete and continuous output under a change of the image tokens\n\n{data['model']}, image tokens of the vision path of {data['vision']}; {len(frames)} frames of {data['root']}, {seeds} flow seeds.\n",
             f"Two flow seeds of the reference: {two_seeds:.2f} cm. Reference, discrete chunk against the mean continuous chunk: {between:.2f} cm. "
             f"To the recorded chunk: continuous {np.linalg.norm(continuous[0] - recorded[:, None], axis=-1).mean():.2f} cm, discrete {np.nanmean(np.linalg.norm(discrete[0] - recorded, axis=-1)):.2f} cm. "
             f"Generated tokens that did not decode to a chunk (frames left out of the discrete columns): {', '.join(f'{c} {int(failed[i].sum())}' for i, c in enumerate(cells))}.\n",
             "| | continuous chunk, tip distance to the reference (cm) | discrete chunk, tip distance to the reference (cm) | discrete chunk identical to the reference (share of frames) | flow loss on the recorded chunk | its change per frame | token cross-entropy on the recorded chunk | its change per frame | top-1 on the recorded tokens |",
             "|" + "---|" * 9]
    for i, cell in enumerate(cells):
        flow, ce = loss(cell, "flow"), loss(cell, "ce")
        row = [NAMES[cell]]
        row += ["", "", ""] if i == 0 else [f"{np.linalg.norm(continuous[i] - continuous[0], axis=-1).mean():.2f}", f"{np.nanmean(np.linalg.norm(discrete[i] - discrete[0], axis=-1)):.2f}", f"{agree(cell):.2f}"]
        row += [f"{flow.mean():.4f}", "" if i == 0 else f"{np.abs(flow - loss('reference', 'flow')).mean():.4f}", f"{ce.mean():.3f}", "" if i == 0 else f"{np.abs(ce - loss('reference', 'ce')).mean():.3f}", f"{loss(cell, 'top1').mean():.3f}"]
        lines.append("| " + " | ".join(row) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.splitext(path)[0] + ".md", "w") as file:
        file.write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
