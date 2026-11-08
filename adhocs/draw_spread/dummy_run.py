"""DUMMY data through the full metric set and report, to align on definitions and presentation
before the real run. Nothing here touches a model or a dataset.

    uv run python lerobot/adhocs/draw_spread/dummy_run.py

Synthetic world: a = f(subtask, progress) + style(episode, subtask) + jitter. The style is a per-episode
offset-plus-shape deviation that is constant within the episode and larger on the big motions. The fake
policy memorizes on train anchors (mean follows the episode's style, draws collapsed) and is calibrated
but biased on val anchors. Noise settings move the train mean more than the val mean.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from metrics import anchor_metrics, bucket_reference, noise_metrics  # noqa: E402
from report import write_report  # noqa: E402

OUT = "migration/draw_spread_dummy_2026-11-07"
T, D, K = 30, 7, 64
SUBTASKS = [("reach", 0.6, 0.15), ("grasp", 0.12, 0.03), ("move", 0.5, 0.12), ("release", 0.1, 0.03), ("home", 0.35, 0.08)]
N_BINS = 4
rng = np.random.default_rng(0)
tt = np.linspace(0, 1, T)


def base_pattern(sub_idx: int, prog: float, amp: float) -> np.ndarray:
    phase = 2 * np.pi * (prog + 0.17 * sub_idx)
    freqs = 0.5 + 0.5 * np.arange(D)
    offs = amp * np.cos(phase + 0.9 * np.arange(D))
    shape = 0.35 * amp * np.sin(np.pi * tt[:, None] * freqs[None] + phase)
    return offs[None] + shape


def episode(length_s: int, style_scale: float):
    """Per-second anchors of one synthetic episode: list of (subtask, progress_bin, demo chunk, f chunk)."""
    style_off = rng.normal(0, 1, D)
    style_shape = rng.normal(0, 0.4, (T, D)) * np.sin(np.pi * tt)[:, None]
    rows = []
    t = 0
    while t < length_s:
        si = len(rows) % len(SUBTASKS) if not rows else (rows[-1][0] + 1) % len(SUBTASKS)
        dur = int(rng.integers(3, 9))
        for j in range(dur):
            if t >= length_s:
                break
            name, amp, sty = SUBTASKS[si]
            prog = (j + 0.5) / dur
            f = base_pattern(si, prog, amp)
            style = sty * style_scale * (style_off[None] + style_shape)
            jitter = rng.normal(0, 0.02, (T, D))
            rows.append((si, int(prog * N_BINS), f + style + jitter, f, style, t))
            t += 1
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    corpus = {f"train_ep{i:02d}": episode(int(rng.integers(70, 130)), 1.0) for i in range(40)}
    probe_train = list(corpus)[:8]
    val = {f"val_ep{i}": episode(int(rng.integers(40, 90)), 1.0) for i in range(9)}

    keys, chunks = [], []
    for rows in corpus.values():
        for si, pb, a, *_ in rows:
            keys.append((si, pb))
            chunks.append(a)
    ref = bucket_reference(np.stack(chunks), keys)

    records, draw_store = [], {}
    for set_name, eps in (("train", {k: corpus[k] for k in probe_train}), ("val", val)):
        for ep_name, rows in eps.items():
            for si, pb, a, f, style, t in rows:
                mu, var3, _ = ref[(si, pb)]
                sigma = np.sqrt(var3[0])
                if set_name == "train":
                    mean = f + 0.8 * style + rng.normal(0, 0.03, (T, D))
                    spread = 0.3 * sigma
                else:
                    mean = f + rng.normal(0, 0.12, (T, D)) + 0.5 * rng.normal(0, 0.08, D)[None]
                    spread = 1.0 * sigma
                draws = {"clean": mean[None] + spread * rng.normal(0, 1, (K, T, D)) * np.sqrt(0.5)}
                for setting, kick in (("state_noise", 0.6 if set_name == "train" else 0.25), ("image_noise", 0.4 if set_name == "train" else 0.15)):
                    draws[setting] = draws["clean"] + kick * sigma * rng.normal(0, 1, D)[None, None] + 0.2 * sigma * rng.normal(0, 1, (K, T, D))
                base = {"set": set_name, "episode": ep_name, "time_s": t, "subtask": SUBTASKS[si][0], "progress_bin": pb}
                for setting, x in draws.items():
                    rec = {**base, "setting": setting, **anchor_metrics(x, a, mu, var3)}
                    if setting != "clean":
                        rec.update(noise_metrics(x, draws["clean"]))
                    records.append(rec)
                draw_store[f"{ep_name}:{t}"] = draws["clean"].astype(np.float32)

    df = pd.DataFrame(records)
    df.to_csv(f"{OUT}/anchors.csv", index=False)
    np.savez_compressed(f"{OUT}/draws_clean.npz", **draw_store)
    preamble = ("<p class=banner>DUMMY DATA. Synthetic demos and a fake policy that memorizes the per-episode style on train anchors "
                "and is calibrated but biased on val anchors. Only the metric definitions and the layout are real.</p>"
                "<p>Space: normalized anchor-encoded actions, chunk 30 x 7, 64 draws per anchor, anchors every 1 s. "
                "offset = per-joint chunk mean, shape = chunk minus its mean; every mean square splits exactly into offset + shape. "
                "Cross-episode reference = mean and variance of the demo chunks in the (subtask, progress quarter) bucket over the whole training corpus.</p>")
    write_report(df, f"{OUT}/report.html", "Draw spread, train vs val (dummy)", preamble,
                 {"train": probe_train[0], "val": "val_ep0"})
    print(df.groupby(["setting", "set"]).size())
    print(f"wrote {OUT}/report.html")


if __name__ == "__main__":
    main()
