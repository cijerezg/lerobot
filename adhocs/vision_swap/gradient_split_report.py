"""Tables of ``gradient_split.py`` (no model).

    uv run python lerobot/adhocs/vision_swap/gradient_split_report.py outputs/analysis/vision_swap_2026-11-19/gradient_split.pt

Per condition: the share of |delta|^2 inside the span of the K gradients, and what putting in a + delta, a + the
part inside, a + the remainder does to the discrete output against the reference (KL divergence of the next-token
distributions, mean over the recorded action token positions; absolute change of the cross-entropy on the recorded
tokens); all are means over frames. The rows "random directions" are the same split with K random orthonormal
directions. Gradient check: the change of the K scalars along a step, predicted by the gradients against the
forward pass, all frames and scalars together and per frame. Writes ``<name>.md`` next to the file, or the second
argument.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

NAMES = {"base": "base vision path", "random": "random direction, same size", "next": "camera images 1 s later", "other": "camera images of another episode"}


def main(path: str, out: str | None = None) -> None:
    out = out or os.path.splitext(path)[0] + ".md"
    assert not os.path.exists(out), f"{out} exists; pass another name as the second argument"
    data = torch.load(path, weights_only=False)
    frames, conditions = data["frames"], data["conditions"]
    tokens, positions = frames[0]["token_reference"].numel(), frames[0]["positions"]
    size = tokens * 2560

    def mean(values):
        return float(np.mean(values))

    def agreement(predicted, actual):  # the error of the prediction as a share of the change, and their correlation
        predicted, actual = torch.cat(predicted).double().numpy(), torch.cat(actual).double().numpy()
        return f"{np.abs(actual - predicted).sum() / np.abs(actual).sum():.2f}", f"{np.corrcoef(predicted, actual)[0, 1]:.3f}"

    def halves(vectors):  # [K] per frame -> the recorded token's log-probability, the weighted sums
        return [v[: len(v) // 2] for v in vectors], [v[len(v) // 2:] for v in vectors]

    lines = [f"# Change of the image tokens, split by the gradients of the discrete output\n\n{data['model']}, image tokens of the vision path of {data['vision']}; {len(frames)} frames of {data['root']}; "
             f"model run in {'float32' if data['float32'] else 'bfloat16'}. K = {frames[0]['K']} gradients per frame ({positions} recorded action token positions in the first frame), "
             f"image tokens {tokens} x 2560. Reference cross-entropy {mean([f['ce'] for f in frames]):.3f}.\n",
             "## Share of |delta|^2 inside the span\n",
             f"K / (tokens x width) = {frames[0]['K'] / size:.2e}.\n",
             "| | length of delta | share inside the span of the gradients, mean | median | share inside K random directions, mean |", "|" + "---|" * 5]
    for name in conditions:
        share = [f["share"][name] for f in frames]
        lines.append(f"| {NAMES[name]} | {mean([f['norm'][name] for f in frames]):.0f} | {mean(share):.2e} | {np.median(share):.2e} | {mean([f['share_random_directions'][name] for f in frames]):.2e} |")

    top = max(1, tokens // 100)
    heavy = [f["token_delta"]["base"].topk(top).indices for f in frames]
    lines += ["", f"Per image token, base: the {top} tokens with the longest delta hold {mean([float((f['token_delta']['base'][i] ** 2).sum() / (f['token_delta']['base'] ** 2).sum()) for f, i in zip(frames, heavy)]):.3f} of |delta|^2 "
              f"and {mean([float(f['token_gradient'][i].sum() / f['token_gradient'].sum()) for f, i in zip(frames, heavy)]):.4f} of the squared length of the basis of the gradients (even spread: {top / tokens:.4f}).\n",
              "## Parts put in place\n",
              "| | part | KL to the reference | KL, median over frames | change of the cross-entropy on the recorded tokens, absolute, per frame | cast error (length of what the cast to the model's dtype changed / length of the part) |", "|" + "---|" * 6]
    for name in conditions:
        cells = [("gradients", part, [f["put_in"][name][part] for f in frames]) for part in ("full", "inside", "remainder")]
        cells += [("random directions", part, [f["put_in_random_directions"][name][part] for f in frames]) for part in ("inside", "remainder")]
        for basis, part, cell in cells:
            text = "a + delta" if part == "full" else f"a + {part}, {basis}"
            lines.append(f"| {NAMES[name]} | {text} | {mean([c['kl'] for c in cell]):.4f} | {np.median([c['kl'] for c in cell]):.4f} | {mean([abs(c['ce'] - f['ce']) for c, f in zip(cell, frames)]):.4f} | {mean([c['cast'] for c in cell]):.2g} |")

    lines += ["", "## Gradient check\n",
              "Error = sum |forward - predicted| / sum |forward| over frames and scalars; r = their correlation. The prediction uses the step the model received (after the cast). "
              "The last two columns take the error of each frame alone (all its K scalars): frames with large changes do not weigh more there.\n",
              "| direction | step (x delta) | recorded token log-probability: error | r | weighted sums: error | r | error per frame, median | frames with error below 0.2 |", "|" + "---|" * 8]
    for name in frames[0]["check"]:
        for i, step in enumerate(data["steps"]):
            predicted, actual = halves([f["check"][name][i]["predicted"] for f in frames]), halves([f["check"][name][i]["actual"] for f in frames])
            each = np.array([float(agreement([f["check"][name][i]["predicted"]], [f["check"][name][i]["actual"]])[0]) for f in frames])
            lines.append(f"| {NAMES[name]} | {step:g} | " + " | ".join([*agreement(predicted[0], actual[0]), *agreement(predicted[1], actual[1]), f"{np.median(each):.2f}", f"{int((each < 0.2).sum())} of {len(frames)}"]) + " |")

    lines += ["", "## The whole delta: first-order prediction against the forward pass\n",
              "| | recorded token log-probability: error | r | mean absolute predicted | mean absolute forward | weighted sums: error | r |", "|" + "---|" * 7]
    for name in conditions:
        predicted, actual = halves([f["predicted"][name] for f in frames]), halves([f["put_in"][name]["full"]["scalars"] for f in frames])
        lines.append(f"| {NAMES[name]} | " + " | ".join([*agreement(predicted[0], actual[0]), f"{torch.cat(predicted[0]).abs().mean():.3f}", f"{torch.cat(actual[0]).abs().mean():.3f}", *agreement(predicted[1], actual[1])]) + " |")
    text = "\n".join(lines) + "\n"
    with open(out, "w") as file:
        file.write(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
