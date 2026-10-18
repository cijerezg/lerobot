"""Live record of the training data: episodes, frames and hours per root and per sampling group, read from the
training config and the roots themselves. Writes lerobot/dataset_info/DATA.md (overwritten every run).

    uv run python lerobot/dataset_info/data_count.py [--config config_rl.yaml]

Sources: `dataset.sources` (ReBot roots and their episode_groups), `diverse.rebot_group_weights` (shares),
`diverse.rebot_exclusions_manifest` (usefulness cuts: frames whose action targets are never drawn), `diverse.root`
(corpus/ and fmb/ episodes.jsonl minus holdout_episodes.json and excluded_episodes.json), `val_dataset_path`.
Roots built but not yet in the config are listed in lerobot/dataset_info/pending.yaml (`- root: ..., note: ...`).
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

WORKSPACE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def episode_lengths(root):
    eps = pd.concat([pd.read_parquet(p, columns=["episode_index", "length"]) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
    return dict(zip(eps.episode_index, eps.length, strict=True))


def hours(frames, fps=30):
    return f"{frames / fps / 3600:.2f}"


def diverse_rows(root):
    skip = set()
    for name in ("holdout_episodes.json", "excluded_episodes.json"):
        if (root / name).is_file():
            skip |= set(json.loads((root / name).read_text())["episode_ids"])
    by_source = defaultdict(lambda: [0, 0, 0.0])
    for sub in ("corpus", "fmb"):
        for line in (root / sub / "episodes.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["episode_id"] in skip:
                continue
            src = r["source"] if isinstance(r["source"], str) else sub
            s = by_source[src]
            s[0] += 1
            s[1] += r.get("frames", r.get("frame_count", 0))
            s[2] += r.get("duration_s", r.get("duration_s_nominal", 0.0))
    return dict(sorted(by_source.items())), len(skip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_rl.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((WORKSPACE / args.config).read_text())
    shares = cfg["diverse"]["rebot_group_weights"]
    manifest = cfg["diverse"].get("rebot_exclusions_manifest")
    cut = defaultdict(int)  # (root, episode) -> excluded frames
    if manifest:
        for x in json.loads((WORKSPACE / manifest).read_text())["exclusions"]:
            cut[(x["source_root"], x["episode_index"])] += x["to_frame"] - x["from_frame"]

    group = defaultdict(lambda: [0, 0, 0])  # episodes, frames, usable frames
    src_rows = []
    for s in cfg["dataset"]["sources"]:
        lengths = episode_lengths(WORKSPACE / s["root"])
        per = {}
        for g, eps in s["episode_groups"].items():
            frames = sum(lengths[e] for e in eps)
            usable = frames - sum(cut[(s["root"], e)] for e in eps)
            per[g] = (len(eps), frames)
            group[g][0] += len(eps)
            group[g][1] += frames
            group[g][2] += usable if shares.get(g, 0) else 0
        sampled = sum(n for g, (n, _) in per.items() if shares.get(g, 0))
        src_rows.append((s["name"], s["root"], len(lengths), sum(lengths.values()), sampled, per))

    rebot_share = sum(v for g, v in shares.items() if g in group)
    diverse_share = cfg["diverse"]["weight"] if cfg["diverse"].get("enabled") else 0
    div, n_skip = diverse_rows(WORKSPACE / cfg["diverse"]["root"])

    out = [f"# Data record\n\nGenerated {datetime.now():%Y-%m-%d %H:%M} by `uv run python lerobot/dataset_info/data_count.py` from "
           f"`{args.config}`. Do not edit by hand; rerun after every change to the data or the config.\n"]
    out.append("## Training mix by sampling group\n\nShare = fraction of all training samples. Usable = frames minus the usefulness cuts "
               f"(`{manifest}`). ReBot hours at 30 fps.\n")
    out.append("| group | share | episodes | frames | usable frames | hours |\n|---|---|---|---|---|---|")
    for g, (n, f, u) in sorted(group.items(), key=lambda kv: -shares.get(kv[0], 0)):
        out.append(f"| {g} | {shares.get(g, 0):.4g} | {n} | {f:,} | {u:,} | {hours(f)} |")
    tot = [sum(v[i] for g, v in group.items() if shares.get(g, 0)) for i in range(3)]
    out.append(f"| **ReBot sampled** | **{rebot_share:.4g}** | **{tot[0]}** | **{tot[1]:,}** | **{tot[2]:,}** | **{hours(tot[1])}** |")
    dn, df, ds = (sum(v[i] for v in div.values()) for i in range(3))
    out.append(f"| diverse corpus | {diverse_share:.4g} | {dn} | {df:,} | | {ds / 3600:.2f} |\n")

    out.append("## ReBot roots in the config\n\n| source | root | episodes | frames | hours | sampled eps | groups (episodes / frames) |\n|---|---|---|---|---|---|---|")
    for name, root, n, f, sampled, per in src_rows:
        groups = ", ".join(f"{g} {k} / {fr:,}" for g, (k, fr) in per.items())
        out.append(f"| {name} | `{root}` | {n} | {f:,} | {hours(f)} | {sampled} | {groups} |")
    n_all, f_all = sum(r[2] for r in src_rows), sum(r[3] for r in src_rows)
    out.append(f"| **total** | | **{n_all}** | **{f_all:,}** | **{hours(f_all)}** | | |\n")

    out.append(f"## Diverse corpus\n\n`{cfg['diverse']['root']}`: corpus/ + fmb/ episodes, minus {n_skip} holdout or excluded episodes "
               "(anchor-level retention not counted). Hours from each episode's duration.\n")
    out.append("| source | episodes | frames | hours |\n|---|---|---|---|")
    out += [f"| {k} | {n} | {f:,} | {s / 3600:.2f} |" for k, (n, f, s) in div.items()]

    val = cfg.get("val_dataset_path")
    if val:
        lengths = episode_lengths(WORKSPACE / val)
        out.append(f"\n## Validation\n\n`{val}`: {len(lengths)} episodes, {sum(lengths.values()):,} frames, {hours(sum(lengths.values()))} h.")

    pending = yaml.safe_load((HERE / "pending.yaml").read_text()) if (HERE / "pending.yaml").is_file() else None
    out.append("\n## Built, not in the config\n\nFrom `lerobot/dataset_info/pending.yaml`.\n")
    if pending:
        out.append("| root | episodes | frames | hours | note |\n|---|---|---|---|---|")
        for p in pending:
            lengths = episode_lengths(WORKSPACE / p["root"])
            out.append(f"| `{p['root']}` | {len(lengths)} | {sum(lengths.values()):,} | {hours(sum(lengths.values()))} | {p.get('note', '')} |")
    else:
        out.append("None.")
    (HERE / "DATA.md").write_text("\n".join(out) + "\n")
    print((HERE / "DATA.md").read_text())


if __name__ == "__main__":
    main()
