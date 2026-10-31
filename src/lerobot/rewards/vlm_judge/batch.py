"""Shared loop of the batch CLIs (score_episode / assign_episode ``--sim-run``): every (layout folder, episode) of a sim eval
run that has videos and a scorer record, a thread pool across episodes (vllm batches the concurrent requests; within an
episode the queries stay sequential), existing outputs skipped (rerun = resume), one markdown table per run."""

import fnmatch
import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def sim_episodes(sim_run: Path, layouts: str, episodes: list[int] | None) -> list[tuple[Path, int]]:
    out = []
    for d in sorted(p for p in sim_run.iterdir() if (p / "subtask_events.json").exists()):
        if not fnmatch.fnmatch(d.name, layouts):
            continue
        recorded = sorted(int(m[1]) for p in (d / "videos").glob("ep*_top.mp4") if (m := re.fullmatch(r"ep(\d+)_top\.mp4", p.name)))
        out += [(d, ep) for ep in recorded if episodes is None or ep in episodes]
    return out


def run_batch(jobs: list[tuple[Path, int]], out_dir: Path, suffix: str, one: Callable[[Path, int, Path], dict], workers: int) -> list[dict]:
    """``one(layout_dir, episode, out_path) -> metrics`` writes out_path; an existing out_path is reused (its .metrics.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def job(item):
        d, ep = item
        out = out_dir / f"{d.name}_ep{ep}.{suffix}.json"
        metrics_path = out.with_suffix(".metrics.json")
        if metrics_path.exists():
            return dict(layout=d.name, episode=ep, **json.loads(metrics_path.read_text()))
        try:
            m = one(d, ep, out)
        except Exception as e:  # one broken episode does not stop the batch
            print(f"{d.name} ep{ep}: FAILED {type(e).__name__}: {e}")
            return None
        metrics_path.write_text(json.dumps(m, indent=1))
        print(f"{d.name} ep{ep}: " + ", ".join(f"{k}={v}" for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)))
        return dict(layout=d.name, episode=ep, **m)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for r in pool.map(job, jobs) if r is not None]
    return rows


def write_table(rows: list[dict], keys: list[str], path: Path, title: str) -> None:
    """Mean of each numeric key per layout group (layout without its _p<k>), then overall."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(re.sub(r"_[vp]\d+$", "", r["layout"]), []).append(r)

    def mean(rs, k):
        v = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
        return f"{np.mean(v):.3f}" if v else "-"

    lines = [f"# {title}", "", "| layout | episodes | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    lines += [f"| {g} | {len(rs)} | " + " | ".join(mean(rs, k) for k in keys) + " |" for g, rs in sorted(groups.items())]
    lines += [f"| **all** | {len(rows)} | " + " | ".join(mean(rows, k) for k in keys) + " |"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
