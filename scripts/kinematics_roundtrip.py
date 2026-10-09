"""Phase 1d acceptance test for the kinematics assets: joints -> pose -> joints roundtrip per robot.

For each robot, 200 recorded frames spread over its episodes go through
``q -> fk -> ik(seed = q + noise) -> q'``; the report gives the pose error of ``fk(q')`` vs ``fk(q)``
(position mm, rotation deg), the joint error ``q' - q`` and the frames where IK did not converge.
Two reference checks anchor the chains to something outside this code: ReBot against pinocchio on
the same URDF, Panda against the FMB recorded ``tcp_pose``. The first frame of 20 episodes per
source, plus the frame where the fingertip is lowest (pads on the table: a direct check of the
tool offset), is rendered as the global camera frame next to the FK skeleton with the fingertip marked
(no global-camera extrinsics exist for any source, so the fingertip cannot be projected into the image).

From the workspace root:
  uv run python lerobot/scripts/kinematics_roundtrip.py --out migration/kinematics_roundtrip_2026-10-09
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import av
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.model.fk import get_chain, solve_ik  # noqa: E402
from lerobot.model.fk.chain import rotation_log  # noqa: E402

REBOT_ROOT = "outputs/rebot_training_mix_2026-10-04/clothing_train"
FMB_ROOT = "outputs/diverse_robot_dataset_v3/fmb"
CORPUS_ROOT = "outputs/diverse_robot_dataset_v3/corpus"
N_FRAMES = 200
N_RENDER = 20
SEED_NOISE = {"deg": 5.0, "rad": 0.1}


# ---------------------------------------------------------------- recorded frames per source
def rebot_states() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All frames of the ReBot root: (states (N,7) deg, episode (N,), frame (N,))."""
    files = sorted(glob.glob(f"{REBOT_ROOT}/data/chunk-*/file-*.parquet"))
    df = pd.concat(
        [pd.read_parquet(f, columns=["observation.state", "episode_index", "frame_index"]) for f in files]
    )
    return (
        np.stack(df["observation.state"].to_numpy()).astype(np.float64),
        df["episode_index"].to_numpy(),
        df["frame_index"].to_numpy(),
    )


def rebot_image(episode: int, frame: int) -> np.ndarray:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("cijerezg/rebot-all-v1", root=REBOT_ROOT)
    start = int(ds.meta.episodes[episode]["dataset_from_index"])
    return (ds[start + frame]["observation.images.external_0"].permute(1, 2, 0).numpy() * 255).astype(
        np.uint8
    )


def fmb_episodes() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(f"{FMB_ROOT}/episodes/*") if (Path(p) / "q.npy").exists())


def fmb_states(ep: Path) -> np.ndarray:
    q = np.load(ep / "q.npy")
    g = np.load(ep / "gripper_pose.npy").astype(np.float64)
    return np.concatenate([q, g[:, None]], 1)


def droid_episodes(source: str) -> list[Path]:
    with open(f"{CORPUS_ROOT}/episodes.jsonl") as f:
        rows = [json.loads(line) for line in f]
    return [Path(CORPUS_ROOT) / r["directory"] for r in rows if r["source"] == source]


def video_frames(path: Path, wanted: list[int]) -> list[np.ndarray]:
    """Decode the listed frame indices (ascending order not required) of one mp4."""
    out: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                out[i] = frame.to_ndarray(format="rgb24")
            if len(out) == len(set(wanted)):
                break
    return [out[i] for i in wanted]


def lowest_fingertip(robot: str, states: np.ndarray) -> int:
    """Frame index with the lowest fingertip in one episode (the pads touch the table there)."""
    p, _, _ = get_chain(robot).fk(torch.tensor(states))
    return int(p[:, 2].argmin())


def lowest_table(robot: str, states: np.ndarray, episode: np.ndarray) -> dict:
    """Per-episode minimum fingertip height, in mm, summarized over the source: a tool-offset check."""
    p, _, _ = get_chain(robot).fk(torch.tensor(states))
    z = p[:, 2].numpy() * 1000
    mins = np.array([z[episode == ep].min() for ep in np.unique(episode)])
    return {
        "episodes": int(len(mins)),
        "min_fingertip_z_mm_p10": float(np.percentile(mins, 10)),
        "median": float(np.median(mins)),
        "p90": float(np.percentile(mins, 90)),
        "min": float(mins.min()),
    }


def spread(n_total: int, n: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_total - 1, n).round().astype(int))


def collect(source: str) -> dict:
    """``{"robot": chain key, "states": (N, dof+1), "episode": (N,), "frame": (N,), "render": [(label, image, state)]}``."""
    if source == "rebot":
        s, e, f = rebot_states()
        idx = spread(len(s), N_FRAMES)
        eps = np.unique(e)
        render = []
        for ep in eps[spread(len(eps), N_RENDER)]:
            rows = np.flatnonzero(e == ep)
            low = int(rows[lowest_fingertip("rebot_b601_follower", s[rows])])
            render.append((f"ep{ep:03d} f0", rebot_image(int(ep), 0), s[rows[0]]))
            render.append((f"ep{ep:03d} f{f[low]} lowest", rebot_image(int(ep), int(f[low])), s[low]))
        return {
            "robot": "rebot_b601_follower",
            "states": s[idx],
            "episode": e[idx],
            "frame": f[idx],
            "render": render,
            "lowest_z": lowest_table("rebot_b601_follower", s, e),
        }
    if source == "fmb":
        eps = fmb_episodes()
        states = [fmb_states(ep) for ep in eps]
        s = np.concatenate(states)
        e = np.concatenate([np.full(len(x), i) for i, x in enumerate(states)])
        f = np.concatenate([np.arange(len(x)) for x in states])
        idx = spread(len(s), N_FRAMES)
        render = []
        for i in spread(len(eps), N_RENDER):
            low = lowest_fingertip("fmb", states[i])
            rgb = np.load(eps[i] / "side_1_rgb.npy", mmap_mode="r")
            render.append((eps[i].name[8:30] + " f0", rgb[0], states[i][0]))
            render.append((eps[i].name[8:30] + f" f{low} lowest", rgb[low], states[i][low]))
        return {
            "robot": "fmb",
            "states": s[idx],
            "episode": e[idx],
            "frame": f[idx],
            "render": render,
            "lowest_z": lowest_table("fmb", s, e),
        }
    eps = droid_episodes(source)
    states = [np.load(ep / "state.npy").astype(np.float64) for ep in eps]
    s = np.concatenate(states)
    e = np.concatenate([np.full(len(x), i) for i, x in enumerate(states)])
    f = np.concatenate([np.arange(len(x)) for x in states])
    idx = spread(len(s), N_FRAMES)
    render = []
    for i in spread(len(eps), N_RENDER):
        low = lowest_fingertip(source, states[i])
        first, lowest = video_frames(eps[i] / "videos/left_external.mp4", [0, low])
        render.append((eps[i].name[7:], first, states[i][0]))
        render.append((eps[i].name[7:] + f" f{low} lowest", lowest, states[i][low]))
    return {
        "robot": source,
        "states": s[idx],
        "episode": e[idx],
        "frame": f[idx],
        "render": render,
        "lowest_z": lowest_table(source, s, e),
    }


# ---------------------------------------------------------------- roundtrip
def rot_deg(ra: torch.Tensor, rb: torch.Tensor) -> np.ndarray:
    return np.degrees(rotation_log(ra @ rb.transpose(1, 2)).norm(dim=1).numpy())


def roundtrip(robot: str, states: np.ndarray, seed: int = 0) -> dict:
    chain = get_chain(robot)
    q = torch.tensor(states[:, : chain.dof])
    p, r = chain.pose(q)
    noise = SEED_NOISE[chain.joint_units] * torch.tensor(np.random.default_rng(seed).standard_normal(q.shape))
    ik = solve_ik(chain, p, r, q + noise)
    p2, r2 = chain.pose(ik.q)
    pos_mm = (p2 - p).norm(dim=1).numpy() * 1000
    rot = rot_deg(r2, r)
    dq = (ik.q - q).numpy()
    dq_deg = dq if chain.joint_units == "deg" else np.degrees(dq)
    dq_deg = (dq_deg + 180.0) % 360.0 - 180.0  # revolute joints: a full turn is the same joint value
    return {
        "chain": chain,
        "pos_mm": pos_mm,
        "rot_deg": rot,
        "joint_err_deg": np.abs(dq_deg).max(axis=1),
        "q": q.numpy(),
        "q_ik": ik.q.numpy(),
        "converged": ik.converged.numpy(),
        "iters": ik.iters.numpy(),
        "redundant": chain.dof > 6,
    }


def summary_row(
    name: str, rt: dict, episode: np.ndarray, frame: np.ndarray
) -> tuple[dict, list[dict], list[dict]]:
    ok = rt["converged"]
    row = {
        "robot": name,
        "frames": int(len(ok)),
        "converged": int(ok.sum()),
        "iters_median": float(np.median(rt["iters"][ok])) if ok.any() else None,
        "iters_max": int(rt["iters"][ok].max()) if ok.any() else None,
        "pos_err_mm_median": float(np.median(rt["pos_mm"][ok])),
        "pos_err_mm_max": float(rt["pos_mm"][ok].max()),
        "rot_err_deg_median": float(np.median(rt["rot_deg"][ok])),
        "rot_err_deg_max": float(rt["rot_deg"][ok].max()),
        "joint_err_deg_median": float(np.median(rt["joint_err_deg"][ok])),
        "joint_err_deg_max": float(rt["joint_err_deg"][ok].max()),
        "joint_err_is_pass_criterion": not rt["redundant"],
    }
    failed = [
        {
            "episode": int(episode[i]),
            "frame": int(frame[i]),
            "pos_err_mm": float(rt["pos_mm"][i]),
            "rot_err_deg": float(rt["rot_deg"][i]),
        }
        for i in np.flatnonzero(~ok)
    ]
    # converged to the same pose but not the same joints: a different branch (6-DOF) or a null-space point (Panda)
    other_branch = [
        {
            "episode": int(episode[i]),
            "frame": int(frame[i]),
            "joint_err_deg": float(rt["joint_err_deg"][i]),
            "q": rt["q"][i].round(3).tolist(),
            "q_ik": rt["q_ik"][i].round(3).tolist(),
        }
        for i in np.flatnonzero(ok & (rt["joint_err_deg"] > 1e-3))
    ]
    row["converged_other_branch"] = len(other_branch)
    return row, failed, other_branch


# ---------------------------------------------------------------- reference checks
def reference_rebot() -> dict:
    """Torch FK vs pinocchio on the same URDF at end_link, 500 joint vectors inside the follower limits."""
    import pinocchio as pin

    from lerobot.model.fk import asset_dirs
    from lerobot.model.fk.chain import _tool_link_offset

    d = asset_dirs()["rebot_b601_follower"]
    chain = get_chain("rebot_b601_follower")
    model = pin.buildModelFromUrdf(str(d / "ReBot_Arm_DM.urdf"))
    data = model.createData()
    fid = model.getFrameId("end_link")
    q = np.random.default_rng(0).uniform(
        [-145, -170, -200, -80, -90, -90.0], [145, 1, 1, 90, 90, 90.0], size=(500, 6)
    )
    frames = chain.frames(torch.tensor(q))
    off = _tool_link_offset(json.loads((d / "chain.json").read_text())["urdf"], d)
    dp = dr = 0.0
    for i in range(len(q)):
        qq = np.zeros(model.nq)
        qq[:6] = np.deg2rad(q[i])
        pin.forwardKinematics(model, data, qq)
        pin.updateFramePlacements(model, data)
        t = (frames[i, 5] @ off).numpy()
        dp = max(dp, float(np.abs(t[:3, 3] - data.oMf[fid].translation).max()))
        dr = max(dr, float(np.abs(t[:3, :3] - data.oMf[fid].rotation).max()))
    return {
        "check": "rebot torch FK vs pinocchio end_link, 500 random joint vectors",
        "max_abs_dp_m": dp,
        "max_abs_dR": dr,
    }


def reference_fmb() -> dict:
    """Panda hand-TCP FK vs the FMB recorded tcp_pose (xyzw quaternion), every frame of 10 episodes."""
    from lerobot.annotation.precision_contact.panda_fk import quat_to_rot

    chain = get_chain("fmb")
    eps = fmb_episodes()[::16][:10]
    q = np.concatenate([np.load(e / "q.npy") for e in eps])
    tp = np.concatenate([np.load(e / "tcp_pose.npy") for e in eps])
    p, r = chain.pose(torch.tensor(q))
    rt = torch.tensor(np.stack([quat_to_rot(x, "xyzw") for x in tp[:, 3:7]]))
    dp = (p - torch.tensor(tp[:, :3])).norm(dim=1).numpy() * 1000
    dr = rot_deg(r, rt)
    return {
        "check": "panda_franka_hand FK vs FMB tcp_pose",
        "frames": int(len(q)),
        "pos_err_mm_median": float(np.median(dp)),
        "pos_err_mm_max": float(dp.max()),
        "rot_err_deg_median": float(np.median(dr)),
        "rot_err_deg_max": float(dr.max()),
    }


# ---------------------------------------------------------------- render
def render(source: str, robot: str, items: list, out: Path) -> Path:
    chain = get_chain(robot)
    n = len(items)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(6.4 * cols, 3.0 * rows))
    for k, (label, image, state) in enumerate(items):
        frames = chain.frames(torch.tensor(state[None]))[0].numpy()
        origins = np.concatenate([np.zeros((1, 3)), frames[:, :3, 3]])
        hand = frames[-1]
        p, r, g = chain.fk(torch.tensor(state[None]))
        ax = fig.add_subplot(rows, 2 * cols, 2 * k + 1)
        ax.imshow(image)
        ax.set_title(f"{source} {label}", fontsize=8)
        ax.axis("off")
        ax3 = fig.add_subplot(rows, 2 * cols, 2 * k + 2, projection="3d")
        ax3.plot(origins[:, 0], origins[:, 1], origins[:, 2], "o-", color="0.4", ms=3, lw=1.5)
        ax3.scatter(*hand[:3, 3], color="red", s=30, label="fingertip")
        for col, color in ((1, "green"), (2, "blue")):
            v = hand[:3, 3] + 0.08 * hand[:3, col]
            ax3.plot([hand[0, 3], v[0]], [hand[1, 3], v[1]], [hand[2, 3], v[2]], color=color, lw=2)
        ax3.set_title(f"p=({p[0, 0]:.2f},{p[0, 1]:.2f},{p[0, 2]:.2f}) m  g={g[0] * 1000:.0f} mm", fontsize=7)
        lim = max(0.9, float(np.abs(origins).max()) + 0.1)
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_zlim(0, lim)
        ax3.set_box_aspect((2 * lim, 2 * lim, lim))
        ax3.tick_params(labelsize=5)
        ax3.view_init(elev=25, azim=-135)
    fig.suptitle(
        f"{source}: first frame and lowest-fingertip frame of {n // 2} episodes, global camera (left) and FK skeleton with fingertip (red), hand y closing (green), z approach (blue)",
        fontsize=10,
    )
    fig.tight_layout()
    path = out / f"render_{source}.png"
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="migration/kinematics_roundtrip_2026-10-09")
    parser.add_argument("--sources", nargs="+", default=["rebot", "fmb", "droid", "droid_success"])
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)

    report: dict = {"reference": [], "roundtrip": [], "not_converged": {}, "other_branch": {}, "renders": {}}
    report["reference"].append(reference_rebot())
    report["reference"].append(reference_fmb())
    report["reference"].append(
        {
            "check": "DROID fk(q) vs source cartesian_position",
            "status": "not run: the DROID staging shards with cartesian_position are no longer on disk",
        }
    )
    for ref in report["reference"]:
        print(json.dumps(ref))

    for source in args.sources:
        data = collect(source)
        rt = roundtrip(data["robot"], data["states"])
        report["lowest_fingertip_z"] = report.get("lowest_fingertip_z", {})
        report["lowest_fingertip_z"][source] = data["lowest_z"]
        print("  lowest fingertip per episode (mm):", json.dumps(data["lowest_z"]))
        row, failed, other_branch = summary_row(
            f"{source} ({rt['chain'].name})", rt, data["episode"], data["frame"]
        )
        report["roundtrip"].append(row)
        report["not_converged"][source] = failed
        report["other_branch"][source] = other_branch
        print(json.dumps(row))
        if failed:
            print("  not converged:", json.dumps(failed))
        if other_branch and not rt["redundant"]:
            print("  converged on another branch:", json.dumps(other_branch))
        if not args.no_render:
            report["renders"][source] = str(render(source, data["robot"], data["render"], out))

    lines = [
        "| robot | frames | converged | same joints | iters med/max | pos err mm med/max | rot err deg med/max | joint err deg med/max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["roundtrip"]:
        je = f"{r['joint_err_deg_median']:.2e} / {r['joint_err_deg_max']:.2e}" + (
            "" if r["joint_err_is_pass_criterion"] else " (info only, redundant)"
        )
        lines.append(
            f"| {r['robot']} | {r['frames']} | {r['converged']} | {r['converged'] - r['converged_other_branch']} | {r['iters_median']:.0f} / {r['iters_max']} | "
            f"{r['pos_err_mm_median']:.2e} / {r['pos_err_mm_max']:.2e} | {r['rot_err_deg_median']:.2e} / {r['rot_err_deg_max']:.2e} | {je} |"
        )
    (out / "roundtrip.md").write_text("\n".join(lines) + "\n")
    (out / "roundtrip.json").write_text(json.dumps(report, indent=2))
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
