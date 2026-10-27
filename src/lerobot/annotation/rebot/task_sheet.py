"""Write the ReBot task sheet: every task of the training mix with its subtask script.

One markdown file, one section per task, built from the roots listed under
``dataset.sources`` of a training config. Each section shows the subtask sequence of one
episode of that task (the episode with the highest mean segment quality) with the labels
the inference prompt takes (``eval_subtasks``, ``eval_subtask_precisions``,
``eval_subtask_contacts``) plus speed and quality for orientation, and a ready-to-paste
YAML block. Other episodes of the same task sit in a collapsed block below.

    uv run python -m lerobot.annotation.rebot.task_sheet config_rl.yaml lerobot/docs/annotation_notes/rebot_tasks.md
"""

from __future__ import annotations

import glob
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from lerobot.annotation.vocab import CONTACT_VOCAB
from lerobot.rl.offline_dataset_utils import REBOT_SPEED_TABLE

PRECISION_LEGEND = {
    1: "> 10 cm slack or no external target (return to home, drop a sock in a basket)",
    2: "3-10 cm (cup into a basket, block into a bin)",
    3: "1-3 cm (straddle a cup, bottle, handle; mouth a few cm wider than the object)",
    4: "3-10 mm (hex bit, screw; bit into a compartment)",
    5: "< 3 mm or a mating fit (peg, plug, bit into a one-bit slot)",
}
QUALITY_LEGEND = {
    5: "exemplary: best grip point, one clean motion",
    4: "default: straight to the goal, small blemish at most",
    3: "laboured but converges; worse grip point; hover past the norm",
    2: "poor: long hover, search, one mistake",
    1: "failed step, no recovery, or human help",
}


def load_segments(root: str) -> pd.DataFrame:
    meta = Path(root) / "meta"
    tasks = pd.read_parquet(meta / "tasks.parquet").reset_index()
    task_of_index = dict(zip(tasks["task_index"], tasks["task"]))
    data = pd.concat(
        pd.read_parquet(f, columns=["episode_index", "task_index", "index"])
        for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    )
    episode_task = data.groupby("episode_index")["task_index"].agg(lambda s: s.mode().iloc[0]).map(task_of_index)
    episode_frames = data.groupby("episode_index").size()
    episode_start = data.groupby("episode_index")["index"].min()
    keys = ["episode_index", "segment_index"]
    df = pd.read_parquet(meta / "episode_metadata.parquet")
    df = df.merge(pd.read_parquet(meta / "precision.parquet")[keys + ["precision"]], on=keys)
    df = df.merge(pd.read_parquet(meta / "contact.parquet")[keys + ["contact_slug"]], on=keys)
    # speed is frame-level: the step shows the range of the runs it overlaps ("1-3")
    runs = pd.read_parquet(meta / REBOT_SPEED_TABLE)

    def speed_range(a, b):
        hit = runs.speed[(runs.from_index < b) & (runs.to_index > a)]
        return f"{hit.min()}" if hit.min() == hit.max() else f"{hit.min()}-{hit.max()}"

    df["speed"] = [speed_range(a, b) for a, b in zip(df["from_index"], df["to_index"])]
    df["task"] = df["episode_index"].map(episode_task)
    df["frames"] = df["episode_index"].map(episode_frames)
    # from_index/to_index are global dataset indices; show episode-local frames
    df["from_index"] -= df["episode_index"].map(episode_start)
    df["to_index"] -= df["episode_index"].map(episode_start)
    return df.sort_values(keys).reset_index(drop=True)


def episode_table(seg: pd.DataFrame) -> str:
    lines = ["| step | subtask | precision | contact | speed | quality | frames |", "|---|---|---|---|---|---|---|"]
    for n, r in enumerate(seg.itertuples(), 1):
        lines.append(
            f"| {n} | {r.subtask} | {r.precision} | {r.contact_slug} | {r.speed} | {r.quality} | {r.from_index}-{r.to_index} |"
        )
    return "\n".join(lines)


def yaml_block(task: str, seg: pd.DataFrame) -> str:
    lines = [f"task: {yaml.safe_dump(task, width=10**6).splitlines()[0]}", "eval_subtasks:"]
    lines += [f"- {s}" for s in seg["subtask"]]
    lines.append(f"eval_subtask_precisions: [{', '.join(str(int(p)) for p in seg['precision'])}]")
    lines.append(f"eval_subtask_contacts: [{', '.join(seg['contact_slug'])}]")
    return "```yaml\n" + "\n".join(lines) + "\n```"


def anchor(text: str) -> str:
    # github-slugger: drop punctuation, spaces to hyphens
    return re.sub(r"[^a-z0-9 -]", "", text.lower()).replace(" ", "-")


def main(config_path: str, out_path: str) -> None:
    cfg = yaml.safe_load(open(config_path))
    sources = cfg["dataset"]["sources"]
    frames: list[pd.DataFrame] = []
    for src in sources:
        df = load_segments(src["root"])
        df["source"] = src["name"]
        df["root"] = src["root"]
        group_of_episode = {ep: g for g, eps in src.get("episode_groups", {}).items() for ep in eps}
        df["group"] = df["episode_index"].map(group_of_episode).fillna(src["sampling_group"])
        frames.append(df)
    seg = pd.concat(frames, ignore_index=True)

    episodes = (
        seg.groupby(["source", "root", "group", "task", "episode_index"])
        .agg(steps=("segment_index", "size"), quality=("quality", "mean"), frames=("frames", "first"))
        .reset_index()
    )
    task_rows = (
        episodes.groupby(["source", "root", "group", "task"])
        .agg(episodes=("episode_index", list), steps=("steps", "min"), steps_max=("steps", "max"))
        .reset_index()
        .sort_values(["source", "group", "task"])
    )
    source_order = {s["name"]: i for i, s in enumerate(sources)}
    task_rows["order"] = task_rows["source"].map(source_order)
    task_rows = task_rows.sort_values(["order", "group", "task"]).reset_index(drop=True)
    task_rows["id"] = [f"T{i + 1:03d}" for i in range(len(task_rows))]

    out: list[str] = []
    out.append("# ReBot task sheet\n")
    out.append(
        f"Generated {date.today().isoformat()} by `lerobot.annotation.rebot.task_sheet` from the roots under "
        f"`dataset.sources` of `{config_path}`. Regenerate after any root changes:\n"
    )
    out.append(f"```bash\nuv run python -m lerobot.annotation.rebot.task_sheet {config_path} {out_path}\n```\n")
    out.append(
        "Use: find the task in the index, open its section, paste the YAML block into `policy:` of the inference "
        "config (`task`, `eval_subtasks`, `eval_subtask_precisions`, `eval_subtask_contacts`). The table shows the "
        "episode with the highest mean quality for that task; the other episodes of the task sit in the collapsed "
        "block under it. A step that repeats the previous one is a retry in the demonstration; collapse it when writing "
        "the script. Speed and quality are shown for orientation only; the prompt does not take them at inference.\n"
    )
    out.append("## Legend\n")
    out.append("| precision | slack at the commit |\n|---|---|")
    out += [f"| {k} | {v} |" for k, v in PRECISION_LEGEND.items()]
    out.append("\n| contact | definition |\n|---|---|")
    out += [f"| {e.slug} | {e.definition} |" for e in CONTACT_VOCAB]
    out.append("\n| quality | meaning |\n|---|---|")
    out += [f"| {k} | {v} |" for k, v in QUALITY_LEGEND.items()]
    out.append(
        "\nSpeed: frame-level `speed_hybrid_v1` bucket 1 (slowest) to 5 (fastest), 1 s smoothed joint motion against the "
        "ReBot training distribution, in runs of at least 1 s; a step shows the lowest-highest bucket of its frames. "
        "Frames are indices inside the root's episode.\n"
    )

    out.append("## Roots\n")
    out.append("| source | root | episodes | tasks | segments |\n|---|---|---|---|---|")
    for src in sources:
        s = seg[seg["source"] == src["name"]]
        out.append(
            f"| {src['name']} | `{src['root']}` | {s['episode_index'].nunique()} | {s['task'].nunique()} | {len(s)} |"
        )

    out.append("\n## Task index\n")
    out.append("| id | task | source | group | episodes | steps |\n|---|---|---|---|---|---|")
    for r in task_rows.itertuples():
        steps = str(r.steps) if r.steps == r.steps_max else f"{r.steps}-{r.steps_max}"
        eps = ", ".join(str(e) for e in sorted(r.episodes))
        out.append(f"| [{r.id}](#{anchor(r.id + ' ' + r.task)}) | {r.task} | {r.source} | {r.group} | {eps} | {steps} |")

    out.append("\n## Tasks\n")
    for r in task_rows.itertuples():
        out.append(f"### {r.id} {r.task}\n")
        eps = episodes[(episodes["source"] == r.source) & (episodes["task"] == r.task)].sort_values(
            ["quality", "episode_index"], ascending=[False, True]
        )
        best = eps.iloc[0]
        out.append(
            f"Source `{r.source}` (`{r.root}`), group `{r.group}`, episodes {', '.join(str(e) for e in sorted(r.episodes))}. "
            f"Table: episode {int(best.episode_index)} ({int(best.frames)} frames, mean quality {best.quality:.1f}).\n"
        )
        best_seg = seg[(seg["source"] == r.source) & (seg["episode_index"] == best.episode_index)]
        out.append(episode_table(best_seg) + "\n")
        out.append(yaml_block(r.task, best_seg) + "\n")
        if len(eps) > 1:
            out.append(f"<details><summary>Other episodes ({len(eps) - 1})</summary>\n")
            for e in eps.iloc[1:].itertuples():
                e_seg = seg[(seg["source"] == r.source) & (seg["episode_index"] == e.episode_index)]
                out.append(f"\nEpisode {int(e.episode_index)} ({int(e.frames)} frames, mean quality {e.quality:.1f})\n")
                out.append(episode_table(e_seg) + "\n")
            out.append("</details>\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(out) + "\n")
    print(f"{out_path}: {len(task_rows)} tasks, {len(episodes)} episodes, {len(seg)} segments")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
