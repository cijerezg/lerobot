"""New ReBot root whose policy actions have the chunk-seam steps blended out. The source is read only. Never overwrites.

    uv run python -m lerobot.annotation.rebot.seam_repair_root outputs/<root> --out outputs/<new_root> [--cache_dir <dir>]

Rollouts recorded before the RTC window fix (2026-11-09) carry a one-frame step in the commanded action at every chunk
handover, every ~16 frames; the arm low-passed it, the recorded action did not (lerobot/adhocs/joint_traces). Per arm
joint, inside one stretch of policy frames:

    v[t]    = a[t] - a[t-1]
    step[t] = v[t] - median(v[t-2 .. t+2])
    seam at s when seam_deg < max_joint |step[s]| <= max_deg
    x       = clip((t - s + K + 0.5) / 2K, 0, 1),  S = 10x^3 - 15x^4 + 6x^5
    a'[t]   = a[t] - step[s] * ((t >= s) - S)          summed over the seams

so each step is spread over +-K frames and nothing else moves. An episode is repaired only when it has at least
``min_rate`` seams per policy frame (rollouts: ~0.05, teleop: < 0.001). Intervention frames, the gripper, steps above
``max_deg`` and the frames next to a recorded cut are left as they are.

Every file is hardlinked; only the data files are unlinked and rewritten. The stats (meta/stats.json, episodes table)
stay those of the recorded actions (user 2026-11-10). meta/seam_repair.json and meta/seam_repair.parquet list what was
done. With ``--cache_dir`` the source root's
replay cache is adopted under the new fingerprint: hardlinked memmaps, new actions.bin. verify_root does not apply
(it asserts action == source).
"""

import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

ARM = 6  # joints 0-5; the gripper's fast changes are real closes


def policy_runs(blocked):
    edges = np.flatnonzero(np.diff(np.concatenate([[1], blocked.astype(int), [1]])))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= 8]


def find_seams(a, seam_deg, max_deg):
    """a: one policy run (T, ARM). Seam s = step between frames s-1 and s."""
    d1 = np.diff(a, axis=0)
    step = d1 - median_filter(d1, size=(5, 1), mode="nearest")
    worst = np.abs(step).max(1)
    hit = np.flatnonzero(worst > seam_deg)
    keep = worst[hit] <= max_deg
    return hit[keep] + 1, step[hit[keep]], hit[~keep] + 1, step[hit[~keep]]


def offset_blend(a, seams, steps, half):
    out = a.copy()
    t = np.arange(len(a))
    for s, d in zip(seams, steps, strict=True):
        x = np.clip((t - s + half + 0.5) / (2 * half), 0.0, 1.0)
        ramp = x**3 * (10 - 15 * x + 6 * x**2)
        out -= ((t >= s) - ramp)[:, None] * d
    return out


def repair_episode(action, interv, cut_frames, args):
    """Returns the repaired action and the seam / large-step rows (frame, per-joint step)."""
    blocked = interv.copy()
    for k in cut_frames:
        blocked[max(k - 3, 0) : k + 3] = True
    found = []
    for lo, hi in policy_runs(blocked):
        seams, steps, big, big_steps = find_seams(action[lo:hi, :ARM], args.seam_deg, args.max_deg)
        found.append((lo, hi, seams, steps, big, big_steps))
    n_policy = int((~interv).sum())
    n_seams = sum(len(f[2]) for f in found)
    repaired = n_policy > 0 and n_seams / n_policy >= args.min_rate
    out = action.copy()
    rows = []
    for lo, hi, seams, steps, big, big_steps in found:
        if repaired:
            out[lo:hi, :ARM] = offset_blend(action[lo:hi, :ARM], seams, steps, args.half)
            rows += [(int(s + lo), "seam", *d) for s, d in zip(seams, steps, strict=True)]
        rows += [(int(s + lo), "large_step", *d) for s, d in zip(big, big_steps, strict=True)]
    return out, rows, repaired, n_policy, n_seams


def jerk(action, interv):
    d2 = np.abs(np.diff(action[:, :ARM], 2, axis=0)).max(1)
    return d2[~(interv[1:-1] | interv[:-2] | interv[2:])]


def rewrite(path):
    """Break the hardlink so writing the new file leaves the source untouched."""
    path.unlink()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache_dir", type=Path)
    ap.add_argument("--seam_deg", type=float, default=1.0)
    ap.add_argument("--max_deg", type=float, default=15.0)
    ap.add_argument("--half", type=int, default=8)
    ap.add_argument("--min_rate", type=float, default=0.01)
    args = ap.parse_args()
    root, out = args.root, args.out
    assert not out.exists(), out

    for dirpath, _, files in os.walk(root):
        dst = out / Path(dirpath).relative_to(root)
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = Path(dirpath) / f
            if src.is_symlink():
                os.symlink(os.path.realpath(src), dst / f)
            else:
                os.link(src, dst / f)

    info = json.loads((root / "meta/info.json").read_text())
    names = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]][:ARM]
    labels = pd.read_parquet(root / "meta/online_labels.parquet")[["episode_index", "frame_index", "is_intervention"]]
    provenance = {p["episode_index"]: p for p in json.loads((root / "meta/provenance.json").read_text())}
    eps = pd.read_parquet(out / "meta/episodes/chunk-000/file-000.parquet")
    assert list(eps.episode_index) == list(range(len(eps)))

    seam_rows, ep_report = [], []
    before, after = [], []
    data_files = sorted({info["data_path"].format(chunk_index=c, file_index=f)
                         for c, f in zip(eps["data/chunk_index"], eps["data/file_index"], strict=True)})
    for rel in data_files:
        d = pd.read_parquet(root / rel)
        columns = list(d.columns)
        m = d.merge(labels, on=["episode_index", "frame_index"], how="left")
        assert len(m) == len(d) and (m["index"].to_numpy() == d["index"].to_numpy()).all()
        action = np.stack(d["action"].to_numpy()).astype(np.float64)
        interv_all = m["is_intervention"].eq(True).to_numpy()
        new = action.copy()
        for ep, pos in d.groupby("episode_index").indices.items():
            assert (np.diff(d["frame_index"].to_numpy()[pos]) == 1).all(), (rel, ep)
            p = provenance[ep]
            cut_frames, off = [], p["keep"][0]
            for c in p.get("cuts") or []:
                cut_frames.append(c[0] - off)
                off += c[1] - c[0]
            fixed, rows, repaired, n_policy, n_seams = repair_episode(action[pos], interv_all[pos], cut_frames, args)
            new[pos] = fixed
            seam_rows += [(int(ep), *r) for r in rows]
            ep_report.append(dict(episode_index=int(ep), kind=p.get("kind"), frames=len(pos), policy_frames=n_policy,
                                  seams_found=n_seams, repaired=bool(repaired)))
            if repaired:
                before.append(jerk(action[pos], interv_all[pos]))
                after.append(jerk(fixed, interv_all[pos]))
        assert (new[interv_all] == action[interv_all]).all() and (new[:, ARM:] == action[:, ARM:]).all()
        d["action"] = list(new.astype(np.float32))
        d[columns].to_parquet(rewrite(out / rel), index=False)

    seams = pd.DataFrame(seam_rows, columns=["episode_index", "frame_index", "kind", *[f"step_{n}" for n in names]])
    seams.to_parquet(out / "meta/seam_repair.parquet", index=False)
    before = np.concatenate(before) if before else np.zeros(1)
    after = np.concatenate(after) if after else np.zeros(1)
    report = dict(
        date="2026-11-10", source_root=str(root), method="offset blend (lerobot.annotation.rebot.seam_repair_root)",
        seam_deg=args.seam_deg, max_deg=args.max_deg, half=args.half, min_rate=args.min_rate,
        episodes_repaired=sum(e["repaired"] for e in ep_report), episodes=len(ep_report),
        seams_repaired=int((seams.kind == "seam").sum()), large_steps_left=int((seams.kind == "large_step").sum()),
        jerk_frames_over_3deg_before=float((before > 3).mean()), jerk_frames_over_3deg_after=float((after > 3).mean()),
        jerk_p99_before=float(np.percentile(before, 99)), jerk_p99_after=float(np.percentile(after, 99)),
        stats="unchanged (recorded actions)",
        per_episode=ep_report,
    )

    if args.cache_dir is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.rl.buffer import ReplayBuffer

        caches = {json.loads(m.read_text())["dataset_root"]: m.parent for m in args.cache_dir.glob("*/metadata.json")}
        old_cache = caches[str(root)]
        meta = json.loads((old_cache / "metadata.json").read_text())
        kw = dict(state_keys=meta["state_keys"], image_storage_dtype=meta["image_storage_dtype"],
                  image_storage_size=meta["image_storage_size"], image_stride=meta["image_stride"])
        old = LeRobotDataset("local/" + root.name, root=str(root))
        ds = LeRobotDataset("local/" + out.name, root=str(out))
        assert ReplayBuffer._dataset_fingerprint(old, **kw) == meta["fingerprint"], "fingerprint recipe mismatch"
        fp = ReplayBuffer._dataset_fingerprint(ds, **kw)
        new_cache = args.cache_dir / fp
        assert not new_cache.exists(), new_cache

        def cache_actions(r):
            d = pd.concat(pd.read_parquet(r / rel, columns=["index", "action"]) for rel in data_files).sort_values("index")
            assert list(d["index"]) == list(range(meta["num_transitions"]))
            a = torch.from_numpy(np.stack(d["action"].to_numpy()).astype(np.float32))
            return a.to(torch.bfloat16).view(torch.uint16).numpy()

        cached = np.fromfile(old_cache / "actions.bin", dtype=np.uint16).reshape(-1, len(info["features"]["action"]["names"]))
        assert (cached == cache_actions(root)).all(), "cached actions are not the bf16 of the root's action column"
        new_cache.mkdir()
        for f in old_cache.iterdir():
            if f.name not in ("metadata.json", "actions.bin"):
                os.link(f, new_cache / f.name)
        cache_actions(out).tofile(new_cache / "actions.bin")
        meta.update(fingerprint=fp, dataset_root=str(out), adopted_from=str(old_cache), seam_repair=True)
        (new_cache / "metadata.json").write_text(json.dumps(meta, indent=2))
        assert ReplayBuffer.find_cache(ds, args.cache_dir, **kw) == new_cache
        report["cache"] = str(new_cache)

    (out / "meta/seam_repair.json").write_text(json.dumps(report, indent=1))
    print(f"{out}: {report['episodes_repaired']}/{report['episodes']} episodes, {report['seams_repaired']} seams, "
          f"{report['large_steps_left']} large steps left; frames > 3 deg {100 * report['jerk_frames_over_3deg_before']:.2f}% -> "
          f"{100 * report['jerk_frames_over_3deg_after']:.2f}%, p99 {report['jerk_p99_before']:.2f} -> {report['jerk_p99_after']:.2f}; "
          f"cache {report.get('cache')}")


if __name__ == "__main__":
    main()
