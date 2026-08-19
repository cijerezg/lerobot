r"""Does the subtask clause reach the actions at all? (the old plan's P3)

The subtask is an input and nothing else: the policy stopped generating one
(2026-08-01), so no decode stands between the annotation and the behaviour. What is
written in that clause at rollout — by an operator, a script, or a higher-level
planner — is what the model gets. This probe is therefore the whole test of whether
the subtask vocabulary is worth annotating for control: if sweeping it does not move
the action chunk, every subtask label in every dataset is decorative *for control*,
however good its CE looks.

Method: fix a frame, sweep every subtask in the dataset vocabulary through the
action prompt, identical fixed-seed flow noise on every pass, and ask two questions.

**1. Does the clause move the output more than noise does?** Vocabulary spread is
compared against a *seed floor* — the same prompt re-sampled under different flow
seeds. With $a_\ell$ the chunk under label $\ell$ and $a^{(s)}$ the chunk under
seed $s$, both measured as RMSE in normalized action space,

    $$S = \frac{\text{mean}_{\ell \neq \ell'} \lVert a_\ell - a_{\ell'} \rVert}{\text{mean}_{s \neq s'} \lVert a^{(s)} - a^{(s')} \rVert}$$

is the honest statistic. A raw spread of $0.05$ means nothing on its own: against a
seed floor of $0.05$ the clause does nothing, and against a floor of $0.005$ it
dominates. This is the number to read first.

**2. Is the clause used *correctly*?** Moving the chunk is not the same as moving it
somewhere sensible: a model that reacts to the clause without understanding it scores
just as high on readout 1. So readout 2 makes the model identify its own situation.
Hold the frame fixed, put each vocabulary label $\ell$ into the prompt in turn, and
score the chunk it produces against what the human actually demonstrated,

    $$e(\ell) = \frac{1}{TD} \sum_{t,d} \left( a_{\ell,t,d} - a^{\star}_{t,d} \right)^2$$

with $a^{\star}$ the demonstrated chunk and $T$, $D$ the chunk steps and action
dimensions. Sort the vocabulary by $e$ and record the position of the true label,

    $$r = |\{ \ell \in V : e(\ell) \le e(\ell^{\star}) \}|$$

so $r = 1$ means that of every label available, the true one drove the policy closest
to the demonstration. That is the behaviour a model which understands the clause has to
show: told the truth, it should act most like the demonstration.

The null is the clause changing the chunk while carrying nothing about which chunk is
right. Then $r$ is uniform over the $|V|$ labels, giving

    $$E[r] = \frac{|V| + 1}{2}$$

and, over $n$ frames, a standard error on the mean rank of $\sqrt{(|V|^2-1)/(12n)}$ —
compute it before reading anything into a mean that sits a little off uniform. The
top-1 fraction has its own null at $1/|V|$.

Read the histogram, not just its mean: flat-at-uniform is *read but not understood*,
and no single number separates that from *not read at all*. Both failure modes matter
and they are different bugs — the first is a conditioning problem, the second a
plumbing one.

The two readouts are ordered. If $S \approx 1$ the vocabulary never moved the chunk
beyond noise, and readout 2 is then ranking noise: the rank histogram of a policy that
ignores the clause is uniform by construction, so it adds nothing. Read $S$ first, and
only ask about ranks once it clears the floor.

Cost is ``n_frames x (|vocab| + n_seeds)`` forwards, so both are small knobs. A second
figure re-draws the fan as a ``subtask_sweep_fan_grid`` square of joints x frames, off the
same forwards — no extra cost, and it is qualitative: the statistics stay the two panels
above.

Registered probe: enable with ``probe_parameters.enable_subtask_sweep``.
"""

import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.probes.manifest import Panel, write_index
from lerobot.probes.utils import (
    REBOT_JOINT_NAMES,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)


def _vocabulary(dataset, limit: int) -> list[str]:
    """Subtask label texts from meta/subtasks.parquet (the text is the frame index)."""
    table = getattr(getattr(dataset, "meta", None), "subtasks", None)
    if table is None or not hasattr(table, "index"):
        return []
    labels = [str(name) for name in table.index]
    return labels[:limit] if limit and limit > 0 else labels


def _pairwise_rmse(chunks: list[torch.Tensor]) -> tuple[float, float]:
    """Mean and max pairwise RMSE between action chunks."""
    if len(chunks) < 2:
        return 0.0, 0.0
    values = [
        float((chunks[i] - chunks[j]).pow(2).mean().sqrt())
        for i in range(len(chunks))
        for j in range(i + 1, len(chunks))
    ]
    return float(np.mean(values)), float(np.max(values))


def _caption(ax, lines: list[str]) -> None:
    """The computation behind the panel, printed under the panel.

    Pre-wrapped rather than filled, because the lines carry mathtext spans that a
    wrapper would break mid-formula.
    """
    ax.text(0.0, -0.20, "\n".join(lines), transform=ax.transAxes, fontsize=7.4,
            va="top", ha="left", color="#333333", linespacing=1.55)


def _spread_by_joint(examples: list[dict]) -> "np.ndarray":
    """Across-label std per joint, averaged over chunk steps and over the given frames."""
    per_frame = [
        torch.stack(list(example["acts"].values())).float().std(dim=0).mean(dim=0)
        for example in examples
    ]
    return torch.stack(per_frame).mean(dim=0).numpy()


def _render_fan(examples: list[dict], vocabulary: list[str], grid: int, output_dir: str) -> None:
    """The third summary panel, widened: the same fan over several joints and frames.

    One panel of a fan is an illustration; a grid is closer to a look at the whole
    object, because a clause that steers only one joint on only one frame shows up
    here as fifteen flat panels next to one live one.
    """
    columns = [examples[i] for i in np.linspace(0, len(examples) - 1, grid).astype(int)]
    joints = list(np.argsort(-_spread_by_joint(columns))[:grid])
    colors = plt.cm.tab20(np.arange(len(vocabulary)) % 20)

    # Shared y within a row: the fan's width is the thing being read, and per-panel
    # autoscaling would rescale it frame by frame.
    fig, axes = plt.subplots(
        len(joints), len(columns), figsize=(4.1 * len(columns), 2.7 * len(joints)),
        squeeze=False, sharex=True, sharey="row",
    )
    for col, example in enumerate(columns):
        steps = np.arange(example["gt"].shape[0])
        for row, joint in enumerate(joints):
            ax = axes[row][col]
            for color, (label, chunk) in zip(colors, example["acts"].items(), strict=True):
                ax.plot(steps, chunk[:, joint], linewidth=0.85, alpha=0.65, color=color, label=label)
            ax.plot(steps, example["gt"][:, joint], color="black", linewidth=2.4,
                    label="GT (demonstrated)")
            if example["gt_subtask"] in example["acts"]:
                ax.plot(steps, example["acts"][example["gt_subtask"]][:, joint], color="black",
                        linestyle="--", linewidth=1.9, label="chunk under the true label")
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title(
                    f"ep {example['episode_idx']} fr {example['frame_idx']}  ·  "
                    f"$S$ = {example['separation']:.2f}x\ntrue label: {example['gt_subtask']}",
                    fontsize=8.5,
                )
            if row == len(joints) - 1:
                ax.set_xlabel("chunk step", fontsize=9)
            if col == 0:
                name = REBOT_JOINT_NAMES[joint] if joint < len(REBOT_JOINT_NAMES) else joint
                ax.set_ylabel(f"{name}\nnormalized action", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5, frameon=False)
    fig.suptitle(
        f"Subtask clause → action fan ({len(joints)} joints x {len(columns)} frames, "
        f"{len(vocabulary)} labels)",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.955,
        "Every coloured line is the same observation decoded under a different subtask label, "
        "identical flow seed; black is the demonstrated chunk.\n"
        "Rows are the joints the vocabulary moves most, so this is the most favourable view of "
        "the clause; columns are frames spread across the sampled episodes.",
        ha="center", va="top", fontsize=8.5, color="#333333", linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.925))
    fig.savefig(os.path.join(output_dir, "subtask_sweep_fan.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def _render(rows: list[dict], vocabulary: list[str], example: dict | None, output_dir: str) -> None:
    fig = plt.figure(figsize=(17, 6.2))
    grid = fig.add_gridspec(1, 3, wspace=0.26, left=0.045, right=0.985, top=0.80, bottom=0.30)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]

    spread = [row["vocab_spread_mean"] for row in rows]
    floor = [row["seed_floor_mean"] for row in rows]
    axes[0].boxplot([spread, floor], tick_labels=["subtask vocab", "flow seed"])
    axes[0].set_ylabel("pairwise RMSE (normalized actions)")
    axes[0].set_title(
        "Does the clause move the chunk\nmore than noise does?\n"
        f"median separation = {np.median([r['separation'] for r in rows]):.2f}x"
    )
    _caption(axes[0], [
        r"One frame gives one point to each box: mean RMSE over all pairs of the " + str(len(vocabulary)) + r" chunks",
        r"the vocabulary produced (left), and over all pairs of chunks the same clause produced",
        r"under different flow seeds (right). Normalized action space, whole chunk, same frame.",
        r"The left box has no scale of its own — the title reads the per-frame ratio $S$ of the two,",
        r"and $S \approx 1$ means the clause did nothing the noise was not already doing.",
    ])

    ranks = [row["gt_rank"] for row in rows if row["gt_rank"] is not None]
    if ranks:
        n_labels = len(vocabulary)
        uniform_mean = (n_labels + 1) / 2
        uniform_se = np.sqrt((n_labels**2 - 1) / (12 * len(ranks)))
        axes[1].hist(ranks, bins=np.arange(0.5, n_labels + 1.5, 1.0), color="#457B9D",
                     edgecolor="white")
        axes[1].axhline(len(ranks) / max(n_labels, 1), color="#E63946", linestyle="--",
                        linewidth=1.2, label="uniform (clause not understood)")
        axes[1].set_xlabel(f"rank of the GT subtask among {n_labels} labels (1 = best)")
        axes[1].set_ylabel("frames")
        axes[1].set_title(
            f"Is it used correctly?\nmean rank {np.mean(ranks):.1f} vs "
            f"{uniform_mean:.1f} ± {uniform_se:.1f} if uniform  ·  "
            f"top-1 {np.mean([r == 1 for r in ranks]):.0%} vs {1 / n_labels:.0%}"
        )
        axes[1].legend(fontsize=8)
        _caption(axes[1], [
            r"Not an error axis — a position axis. Per frame: hold the observation fixed, put each of",
            r"the " + str(n_labels) + r" labels into the prompt in turn, score its chunk against the demonstrated one by",
            r"MSE, sort ascending, and record where the label the human actually wrote landed.",
            r"A bar is a count of frames, so height 4 at rank 12 = 4 frames where 11 wrong labels beat",
            r"the truth. Piled at rank 1 = understood; flat on the dashed line ($n/|V|$ per bin) = the",
            r"clause moved the chunk but carried nothing about which chunk was right. Only read this",
            r"panel once the left one clears its floor: ranks off an ignored clause are uniform anyway.",
        ])
    else:
        axes[1].text(0.5, 0.5, "no GT subtask on the sampled frames", ha="center", va="center")
        axes[1].axis("off")

    if example is not None:
        joint = example["joint"]
        steps = np.arange(example["gt"].shape[0])
        for label, chunk in example["acts"].items():
            axes[2].plot(steps, chunk[:, joint], linewidth=1.0, alpha=0.75, label=label)
        axes[2].plot(steps, example["gt"][:, joint], color="black", linewidth=2.2, label="GT")
        joint_name = REBOT_JOINT_NAMES[joint] if joint < len(REBOT_JOINT_NAMES) else joint
        axes[2].set_title(
            f"Fan across the vocabulary — ep {example['episode_idx']} fr {example['frame_idx']}\n"
            f"joint: {joint_name}"
        )
        axes[2].set_xlabel("chunk step")
        axes[2].grid(True, alpha=0.25, linestyle=":")
        axes[2].legend(fontsize=5, ncol=2)
        _caption(axes[2], [
            r"One frame, one flow seed, one joint (" + str(joint_name) + r"): every coloured line is the chunk under a",
            r"different label, black is what the human demonstrated. $y$ is normalized action units.",
            r"The vertical spread between coloured lines is the left panel's numerator, drawn — this is",
            r"what a separation of " + f"{np.median([r['separation'] for r in rows]):.2f}" + r"x looks like. Distance from black is the middle panel's",
            r"score. Illustration of one frame, not evidence: the statistics are the other two panels.",
        ])

    fig.suptitle(
        f"Subtask clause → action transfer (n={len(rows)} frames, {len(vocabulary)} labels)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(os.path.join(output_dir, "subtask_sweep.png"), bbox_inches="tight", dpi=125)
    plt.close(fig)


def run(adapter, dataset, cfg, output_dir: str) -> None:
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[subtask_sweep] needs continuous flow actions — skipping.")
        return

    p = cfg.probe_parameters
    vocabulary = _vocabulary(dataset, int(getattr(p, "subtask_sweep_max_labels", None) or p.max_labels))
    if len(vocabulary) < 2:
        logging.info("[subtask_sweep] dataset has no subtask vocabulary — skipping.")
        return

    makedirs(output_dir)
    device = adapter.device
    chunk_size = int(cfg.policy.chunk_size)
    n_seeds = max(int(getattr(p, "subtask_sweep_n_seeds", None) or p.n_seeds), 2)
    n_frames = int(getattr(p, "subtask_sweep_n_frames", None) or p.n_frames_per_episode)

    samples = sample_episodes_evenly(
        dataset, n_frames, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )
    logging.info(
        f"[subtask_sweep] {len(samples)} frames x ({len(vocabulary)} labels + {n_seeds} seeds) = "
        f"{len(samples) * (len(vocabulary) + n_seeds)} forward passes"
    )

    adapter._set_probe_cuda_graph_enabled(False)
    rows: list[dict] = []
    examples: list[dict] = []
    try:
        for ep_idx, fr_idx, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"])

            def predict(subtask: str, seed: int) -> torch.Tensor:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                return adapter.predict_action_chunk(
                    frame["obs"], frame["task"], state=frame["state"], subtask=subtask,
                    metadata=frame["metadata"], generator=generator,
                )[1]

            acts = {label: predict(label, 0) for label in vocabulary}
            # Seed floor: same clause, different noise. Without it the vocabulary
            # spread has no scale and any number looks like evidence.
            gt_subtask = frame["subtask"] or vocabulary[0]
            seed_draws = [predict(gt_subtask, seed) for seed in range(1, n_seeds + 1)]

            vocab_mean, vocab_max = _pairwise_rmse(list(acts.values()))
            seed_mean, seed_max = _pairwise_rmse(seed_draws)
            mse = {label: float((chunk - gt_norm).pow(2).mean()) for label, chunk in acts.items()}
            ordered = sorted(mse, key=lambda label: mse[label])

            gt_rank = ordered.index(frame["subtask"]) + 1 if frame["subtask"] in mse else None
            rows.append(
                {
                    "episode_idx": int(ep_idx), "frame_idx": int(fr_idx), "global_idx": int(global_idx),
                    "gt_subtask": frame["subtask"],
                    "vocab_spread_mean": vocab_mean, "vocab_spread_max": vocab_max,
                    "seed_floor_mean": seed_mean, "seed_floor_max": seed_max,
                    "separation": vocab_mean / max(seed_mean, 1e-9),
                    "gt_rank": gt_rank,
                    "best_label": ordered[0],
                    "gt_mse_by_label": mse,
                }
            )
            # A chunk is (chunk_size, action_dim) floats, so keeping every frame's fan
            # for the grid costs kilobytes and saves a second sweep.
            examples.append(
                {
                    "acts": acts, "gt": gt_norm, "joint": 0,
                    "episode_idx": int(ep_idx), "frame_idx": int(fr_idx),
                    "gt_subtask": frame["subtask"], "separation": rows[-1]["separation"],
                }
            )
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not rows:
        logging.warning("[subtask_sweep] no frames produced measurements.")
        return

    ranks = [row["gt_rank"] for row in rows if row["gt_rank"] is not None]
    separations = [row["separation"] for row in rows]
    summary = {
        "n_frames": len(rows),
        "vocabulary": vocabulary,
        "n_labels": len(vocabulary),
        "vocab_spread_mean": float(np.mean([r["vocab_spread_mean"] for r in rows])),
        "seed_floor_mean": float(np.mean([r["seed_floor_mean"] for r in rows])),
        "separation_mean": float(np.mean(separations)),
        "separation_median": float(np.median(separations)),
        "gt_rank_mean": float(np.mean(ranks)) if ranks else None,
        "gt_rank_uniform_expectation": (len(vocabulary) + 1) / 2,
        "gt_rank_top1_fraction": float(np.mean([r == 1 for r in ranks])) if ranks else None,
        "verdict_note": (
            "separation ~1 => the clause moves the chunk no more than flow noise does "
            "(hop two is dead). separation >> 1 with gt_rank_mean ~ uniform => the clause "
            "is read but not understood."
        ),
        "per_frame": rows,
    }
    with open(os.path.join(output_dir, "subtask_sweep.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _render(rows, vocabulary, examples[0], output_dir)
    grid = min(
        int(getattr(p, "subtask_sweep_fan_grid", 4)), len(examples), examples[0]["gt"].shape[1]
    )
    _render_fan(examples, vocabulary, grid, output_dir)

    write_index(
        output_dir,
        sys.modules[__name__],
        title="Subtask Sweep",
        group="Steering",
        claim="Does swapping the subtask clause move the chunk more than flow noise does?",
        summary=summary,
        see_also=["metadata_steering", "mem_history_influence", "action_trace"],
        # Both readouts are titled onto subtask_sweep.png with their own null lines, and
        # the docstring says how to read them. The values stay in subtask_sweep.json.
        metrics=[],
        panels=[
            Panel(
                "subtask_sweep.png",
                "Vocabulary spread against the seed floor, then where the true label ranks",
                "Left: how far the vocabulary moves the chunk versus how far flow noise alone moves "
                "it — the two boxes only mean something as the ratio $S$, never separately. Middle: "
                "per frame, all labels are scored against the demonstrated chunk and sorted, and the "
                "histogram counts where the true label landed; piled at rank 1 is understood, flat on "
                "the dashed line is read-but-not-understood. Right: one frame's fan, to see what those "
                "numbers look like. Read left to right — the ranks are noise until $S$ clears the floor.",
                primary=True,
                refs=["metadata_steering"],
            ),
            Panel(
                "subtask_sweep_fan.png",
                "The same fan over several joints and frames",
                "The right-hand panel of the summary figure, widened into a grid: rows are the "
                "joints the vocabulary moves most, columns are frames spread across the sampled "
                "episodes, and each panel holds the observation fixed while sweeping every label "
                "under one flow seed. Black is the demonstrated chunk and the dashed red line is "
                "the chunk decoded under the true label, so a clause that is understood shows red "
                "hugging black while the rest of the fan stands off it. Qualitative — the numbers "
                "are still the summary figure's two statistics.",
                refs=["metadata_steering"],
            ),
        ],
    )

    logging.info(
        f"[subtask_sweep] n={len(rows)}  vocab spread={summary['vocab_spread_mean']:.4f}  "
        f"seed floor={summary['seed_floor_mean']:.4f}  separation={summary['separation_median']:.2f}x  "
        + (
            f"GT rank {summary['gt_rank_mean']:.1f}/{len(vocabulary)} "
            f"(uniform {summary['gt_rank_uniform_expectation']:.1f}, "
            f"top-1 {summary['gt_rank_top1_fraction']:.0%})"
            if ranks else "no GT subtask on sampled frames"
        )
    )
