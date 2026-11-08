"""Draw-spread study: 64 flow draws per anchor along train and val episodes, clean and under input noise.

    uv run python lerobot/adhocs/draw_spread/run_draw_spread.py --config_path=config_rl_validate.yaml \
        --policy.pretrained_path=outputs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model \
        --out_dir=outputs/draw_spread/internal1200_2026-11-07

Per anchor (every ``anchor_seconds`` along the episode, snapped to the image stride) and per setting
(clean, state noise, image noise) the prefix (ViT + LLM) runs once, its KV cache is expanded to
``n_draws`` rows and only the action expert's flow steps run at that batch (the deployed
``num_inference_steps``). Draws are stored absolute (deg) and normalized (the loss space); metrics are
in ``metrics.py``, the page in ``report.py``. Stage 1 builds the cross-episode reference pool: every
training root's demo chunks at the same cadence, normalized, with subtask string and state; each probe anchor's
reference is its K_REF nearest pool anchors by state within the same subtask (verb fallback), other episodes only.

Intermediates (all under out_dir): reference.npz/csv, episodes/<set>_<root>_<ep>.npz (draws, demo
chunks, states, frames), anchors.csv (one row per anchor x setting), run.json, report.html.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from metrics import anchor_metrics, msq3, noise_metrics  # noqa: E402
from report import write_report  # noqa: E402

from lerobot.configs import parser  # noqa: E402
from lerobot.configs.train import TrainRLServerPipelineConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.molmoact2.anchor_encoding import ANCHOR_KEY  # noqa: E402
from lerobot.probes.base import ProbablePolicy  # noqa: E402
from lerobot.probes.utils import (  # noqa: E402
    _dataset_subtask_indices,
    build_episode_index,
    get_subtask_str,
    probe_frame_inputs,
    probe_image_stride,
    register_config_choices,
    subtask_group,
)
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

N_BINS = 4
MIN_POOL = 16  # smallest same-subtask pool before falling back to the verb pool
K_REF = 32  # cross-episode neighbours (by state, within the pool) that form the reference
# Roots of the 8 probed training episodes (all in the internal-only 10-21 run's sources): one per root, the
# episode whose length is the root's median.
DEFAULT_TRAIN_ROOTS = (
    "outputs/rebot_all-annotated-v14-tasks2",
    "outputs/rebot_main_additions-annotated-v3",
    "outputs/rebot_bits_kitfold_2026-10-06-annotated-v2-tasks1",
    "outputs/rebot_additions_2026-10-04-annotated-v12-tasks1",
    "outputs/rebot_teleop_2026-10-08-annotated-v1-tasks1",
    "outputs/rebot_2026-10-14-annotated-v4",
    "outputs/rebot_2026-10-16-annotated-v1",
    "outputs/rebot_2026-10-21-annotated-v1",
)


@dataclass
class DrawSpreadConfig(TrainRLServerPipelineConfig):
    out_dir: str = "outputs/draw_spread/internal1200_2026-11-07"
    train_roots: str = ""  # comma-separated root[:episode]; empty = DEFAULT_TRAIN_ROOTS with the median-length episode
    n_draws: int = 64
    anchor_seconds: float = 1.0
    noise_settings: str = "state1,state3,image0.02,image0.05"  # state<deg> = Gaussian on every joint in degrees; image<s> = pixel std on [0, 1]
    smoke: int = 0  # > 0: only this many anchors per set and the first two training roots
    skip_reference: bool = False  # reuse out_dir/reference.npz


# ── Dataset helpers ────────────────────────────────────────────────────────────


def lowdim_table(ds) -> dict:
    hf = ds.hf_dataset.with_format(None)
    return {
        "action": np.asarray(hf["action"], dtype=np.float32),
        "state": np.asarray(hf["observation.state"], dtype=np.float32),
        "episode": np.asarray(hf["episode_index"], dtype=np.int64).reshape(-1),
        "frame": np.asarray(hf["frame_index"], dtype=np.int64).reshape(-1),
    }


def subtask_runs(ds, table: dict) -> tuple[list[str], np.ndarray]:
    """Per frame: subtask string and progress in [0, 1) inside its contiguous run within the episode."""
    idx = _dataset_subtask_indices(ds)
    n = len(table["episode"])
    sub_idx = np.full(n, -1, dtype=np.int64) if idx is None else np.asarray(idx, dtype=np.int64)[:n]
    names = {i: get_subtask_str(ds, int(i)) for i in np.unique(sub_idx)}
    strings = [names[int(i)] for i in sub_idx]
    progress = np.zeros(n, dtype=np.float32)
    key = sub_idx * 10_000_000 + table["episode"]
    change = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    for a, b in zip(starts, ends):
        progress[a:b] = (np.arange(b - a) + 0.5) / (b - a)
    return strings, progress


def chunk_at(table: dict, gidx: int, chunk: int) -> np.ndarray:
    """Raw demo chunk [chunk, D] from the low-dim table, padded with the last action at the episode end."""
    ep = table["episode"][gidx]
    end = gidx
    while end < len(table["episode"]) and end - gidx < chunk and table["episode"][end] == ep:
        end += 1
    a = table["action"][gidx:end]
    if len(a) < chunk:
        a = np.concatenate([a, np.repeat(a[-1:], chunk - len(a), axis=0)])
    return a


def anchor_frames(table: dict, episode: int, step: int, stride: int) -> list[int]:
    idx = np.flatnonzero(table["episode"] == episode)
    frames = table["frame"][idx]
    grid = (frames % step == 0) & (frames % stride == 0)
    return idx[grid].tolist()


def median_episode(table: dict) -> int:
    eps, counts = np.unique(table["episode"], return_counts=True)
    return int(eps[np.argsort(counts)[len(counts) // 2]])


def bucket_key(subtask: str, progress: float) -> tuple[str, int]:
    return (subtask, min(int(progress * N_BINS), N_BINS - 1))


# ── Policy helpers ─────────────────────────────────────────────────────────────


class Sampler:
    """Prefix once, expanded KV cache, K draws through the action expert."""

    def __init__(self, adapter, cfg, n_draws: int):
        self.adapter = adapter
        self.policy = adapter._policy
        self.cfg = cfg
        self.K = n_draws
        self.device = adapter._device
        self.steps = int(cfg.policy.num_inference_steps)
        self.dtype = torch.bfloat16 if str(cfg.policy.dtype) in ("bfloat16", "torch.bfloat16") else torch.float16 if "16" in str(cfg.policy.dtype) else None

    @torch.no_grad()
    def draws(self, frame: dict, obs: dict, seed: int, K: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (norm [K, T, D], absolute [K, T, D]) for the deployed prompt of *frame* with observation *obs*.
        Keeps the last packed batch and the padded action width on self for affine_map."""
        K = K or self.K
        adapter, policy = self.adapter, self.policy
        batch = adapter._make_batch(obs, frame["task"], subtask=frame["subtask"], metadata=frame["metadata"])
        model_inputs = policy._model_inputs(batch)
        backbone = policy._backbone()
        autocast = torch.autocast(device_type=self.device.type, dtype=self.dtype) if self.dtype is not None else torch.no_grad()
        with autocast:
            outputs = policy._run_prefix_backbone(model_inputs)
            kv = backbone._extract_kv_states(outputs.past_key_values)
            mask = policy._encoder_attention_mask_for_action_expert(
                input_ids=model_inputs.get("input_ids"), attention_mask=model_inputs.get("attention_mask")
            )
            expand = lambda t: None if t is None else t.expand(K, *t.shape[1:]).contiguous()  # noqa: E731
            kv_k = [(expand(k), expand(v)) for k, v in kv]
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            actions = backbone.generate_actions_from_inputs(
                input_ids=expand(model_inputs["input_ids"]),
                attention_mask=expand(model_inputs.get("attention_mask")),
                states=expand(model_inputs.get("states")),
                action_dim_is_pad=expand(batch.get("action_dim_is_pad")),
                action_horizon=policy._generation_action_horizon(),
                num_steps=self.steps,
                generator=generator,
                encoder_kv_states=kv_k,
                encoder_attention_mask=expand(mask),
            )
        action_dim = policy._output_action_dim(batch)
        norm_actions = actions[:, : policy.config.n_action_steps, :action_dim].to(dtype=torch.float32)
        norm = adapter._probe_joint_actions(norm_actions.cpu())
        absolute = self._unnormalize(norm_actions, batch, obs[OBS_STATE], K)
        self.last_batch, self.last_width = batch, int(norm_actions.shape[-1])
        return norm.numpy(), absolute.numpy()

    def _unnormalize(self, norm_actions: torch.Tensor, batch: dict, state: torch.Tensor, K: int) -> torch.Tensor:
        adapter = self.adapter
        action_encoding = getattr(self.cfg.policy, "action_encoding", "absolute")
        if action_encoding in ("anchor", "delta"):
            anchor = adapter._to_action_width(state.to(self.device).reshape(1, -1)).expand(K, -1)
            identity = dict(adapter._identity_columns)
            for index_key in adapter._stats_index_keys:
                if index_key in batch:
                    identity[index_key] = batch[index_key].expand(K, *batch[index_key].shape[1:]) if torch.is_tensor(batch[index_key]) else batch[index_key]
            payload = {ACTION: norm_actions, ANCHOR_KEY: anchor, TransitionKey.COMPLEMENTARY_DATA: identity}
            unnorm = adapter._postprocessor(payload)
        else:
            unnorm = adapter._postprocessor(norm_actions)
        return unnorm.float().cpu()

    @torch.no_grad()
    def reference_draw(self, frame: dict, obs: dict, seed: int) -> np.ndarray:
        """The policy's own single-row path at the same seed: the consistency check for the shared prefix."""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        _, norm, _ = self.adapter.predict_action_chunk_in_mode(
            obs, frame["task"], inference_action_mode="continuous", subtask=frame["subtask"], metadata=frame["metadata"], generator=generator
        )
        return norm.numpy()


def parse_settings(spec: str) -> list[tuple[str, str, float]]:
    """'clean' or '' = no noise settings; otherwise comma-separated state<deg> / image<std> items."""
    out = [("clean", "none", 0.0)]
    if spec.strip().lower() in ("", "clean", "none"):
        return out
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        kind = "state" if item.startswith("state") else "image"
        out.append((item, kind, float(item[len(kind):])))
    return out


def perturb(obs: dict, kind: str, level: float, seed: int) -> dict:
    if kind == "none":
        return obs
    g = torch.Generator().manual_seed(seed)
    out = dict(obs)
    if kind == "state":
        s = obs[OBS_STATE]
        out[OBS_STATE] = s + level * torch.randn(s.shape, generator=g)
    else:
        for key, v in obs.items():
            if key.startswith("observation.images.") and torch.is_tensor(v) and v.is_floating_point():
                out[key] = (v + level * torch.randn(v.shape, generator=g)).clamp(0.0, 1.0)
    return out


def affine_map(sampler: "Sampler", batch_like: dict, state: torch.Tensor, T: int, width: int, D: int) -> tuple[np.ndarray, np.ndarray]:
    """The postprocessor's own map absolute = alpha + sigma * norm at this anchor's state, read off by pushing the
    zero chunk and the unit chunk through it ([T, D] each). Exact for the affine anchor + unnormalize path."""
    zeros = torch.zeros(2, T, width, device=sampler.device)
    zeros[1] = 1.0
    out = sampler._unnormalize(zeros, batch_like, state, 2).numpy()[:, :, :D].astype(np.float64)
    return out[0], out[1] - out[0]


# ── Stages ─────────────────────────────────────────────────────────────────────


def build_reference(cfg, adapter, roots: list[str], out_dir: str, step: int, stride: int) -> tuple[dict, dict, np.ndarray]:
    chunk = int(cfg.policy.chunk_size)
    rows, chunks, states, anchor_states = [], [], [], []
    for root in roots:
        ds = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=root)
        ds.delta_timestamps = None
        ds.delta_indices = None
        table = lowdim_table(ds)
        strings, progress = subtask_runs(ds, table)
        states.append(table["state"])
        for ep in np.unique(table["episode"]):
            for gidx in anchor_frames(table, int(ep), step, stride):
                raw = chunk_at(table, gidx, chunk)
                norm = adapter.normalize_gt_actions(torch.from_numpy(raw), torch.from_numpy(table["state"][gidx])).numpy()
                rows.append({"root": root, "episode": int(ep), "frame": int(table["frame"][gidx]), "subtask": strings[gidx],
                             "group": subtask_group(strings[gidx]), "progress": float(progress[gidx])})
                chunks.append(norm)
                anchor_states.append(table["state"][gidx])
        logging.info(f"[reference] {root}: {len(rows)} anchors so far")
    ref_df = pd.DataFrame(rows)
    ref_chunks = np.stack(chunks).astype(np.float32)
    ref_df.to_csv(os.path.join(out_dir, "reference.csv"), index=False)
    state_std = np.concatenate(states).std(0)
    np.savez_compressed(os.path.join(out_dir, "reference.npz"), chunks=ref_chunks, states=np.stack(anchor_states).astype(np.float32), state_std=state_std)
    return ref_df, ref_chunks, np.stack(anchor_states), state_std


def load_reference(out_dir: str):
    ref_df = pd.read_csv(os.path.join(out_dir, "reference.csv"))
    z = np.load(os.path.join(out_dir, "reference.npz"))
    return ref_df, z["chunks"], z["states"], z["state_std"]


class Reference:
    """Cross-episode reference for one probe anchor: the K_REF corpus anchors nearest in state (per-joint
    standardized) among those with the same subtask string (or the same verb when that pool is smaller than
    MIN_POOL), never from the anchor's own episode. Returns (mean chunk, var3, n, level, mean state distance)."""

    def __init__(self, ref_df: pd.DataFrame, ref_chunks: np.ndarray, ref_states: np.ndarray, state_std: np.ndarray):
        self.chunks = ref_chunks
        self.z = ref_states / np.maximum(state_std, 1e-6)
        self.std = np.maximum(state_std, 1e-6)
        self.ep_key = (ref_df.root.astype(str) + "#" + ref_df.episode.astype(str)).to_numpy()
        sub = ref_df.subtask.fillna("").to_numpy()
        grp = ref_df.group.fillna("?").to_numpy()
        self.by_subtask = {k: np.flatnonzero(sub == k) for k in np.unique(sub)}
        self.by_group = {k: np.flatnonzero(grp == k) for k in np.unique(grp)}

    def lookup(self, subtask: str, state: np.ndarray, own_episode: str):
        pool, level = self.by_subtask.get(subtask, np.zeros(0, dtype=int)), "subtask"
        pool = pool[self.ep_key[pool] != own_episode]
        if len(pool) < MIN_POOL:
            pool, level = self.by_group.get(subtask_group(subtask), np.zeros(0, dtype=int)), "group"
            pool = pool[self.ep_key[pool] != own_episode]
        if len(pool) < 2:
            return None, None, 0, "none", np.nan
        d = np.sqrt(((self.z[pool] - state / self.std) ** 2).sum(axis=1))
        take = pool[np.argsort(d)[:K_REF]]
        sub = self.chunks[take]
        mean = sub.mean(axis=0)
        var3 = msq3(sub - mean).mean(axis=0)
        return mean, var3, len(take), level, float(np.sort(d)[:K_REF].mean())


def checkpoint_roots(pretrained_path: str, cfg) -> list[str]:
    """ReBot roots the checkpoint trained on: dataset.sources of the train_config.json saved next to it, with
    the val root left out; the run config's sources when the checkpoint carries no train_config.json."""
    path = os.path.join(pretrained_path, "train_config.json")
    if os.path.exists(path):
        sources = json.load(open(path)).get("dataset", {}).get("sources", []) or []
        roots = [s["root"] for s in sources if s.get("root") and s.get("enabled", True) and float(s.get("weight", 1.0)) > 0]
    else:
        from lerobot.rl.offline_dataset_utils import get_offline_dataset_sources

        roots = [s.root for s in get_offline_dataset_sources(cfg) if s.root]
    return [r for r in roots if r != cfg.val_dataset_path and os.path.exists(os.path.join(r, "meta", "info.json"))]


def run(cfg: DrawSpreadConfig):
    init_logging()
    os.makedirs(os.path.join(cfg.out_dir, "episodes"), exist_ok=True)
    device = get_safe_torch_device(try_device=cfg.policy.device)
    val_ds = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.val_dataset_path)
    val_ds.delta_timestamps = None
    val_ds.delta_indices = None
    adapter = ProbablePolicy.for_config(cfg, device, dataset=val_ds)
    adapter._set_probe_cuda_graph_enabled(False)
    fps = float(cfg.env.fps)
    stride = probe_image_stride(cfg)
    step = int(round(cfg.anchor_seconds * fps))
    chunk = int(cfg.policy.chunk_size)
    settings = parse_settings(cfg.noise_settings)

    train_specs = []
    for item in [s for s in cfg.train_roots.split(",") if s.strip()] or list(DEFAULT_TRAIN_ROOTS):
        root, _, ep = item.partition(":")
        train_specs.append((root, int(ep) if ep else None))
    if cfg.smoke:
        train_specs = train_specs[:2]

    corpus_roots = checkpoint_roots(str(cfg.policy.pretrained_path), cfg)
    logging.info(f"[reference] pool roots ({len(corpus_roots)}): {corpus_roots}")
    missing = [r for r, _ in train_specs if r not in corpus_roots]
    if missing:
        raise ValueError(f"train episodes must come from the checkpoint's training roots; not in them: {missing}")
    if cfg.skip_reference and os.path.exists(os.path.join(cfg.out_dir, "reference.npz")):
        ref_df, ref_chunks, ref_states, state_std = load_reference(cfg.out_dir)
    else:
        t0 = time.time()
        ref_df, ref_chunks, ref_states, state_std = build_reference(cfg, adapter, corpus_roots[:2] if cfg.smoke else corpus_roots, cfg.out_dir, step, stride)
        logging.info(f"[reference] {len(ref_df)} anchors from {len(corpus_roots)} roots in {time.time() - t0:.0f} s")
    reference = Reference(ref_df, ref_chunks, ref_states, state_std)

    sampler = Sampler(adapter, cfg, cfg.n_draws)
    records, timing = [], []
    consistency = affine_check = None
    jobs = [("val", cfg.val_dataset_path, val_ds, int(ep)) for ep in sorted(build_episode_index(val_ds))]
    for root, ep in train_specs:
        ds = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=root)
        ds.delta_timestamps = None
        ds.delta_indices = None
        jobs.append(("train", root, ds, ep))

    total_anchors = 0
    resolved_train = []
    for set_name, root, ds, ep in jobs:
        table = lowdim_table(ds)
        if ep is None:
            ep = median_episode(table)
        if set_name == "train":
            resolved_train.append((root, int(ep)))
        strings, progress = subtask_runs(ds, table)
        anchors = anchor_frames(table, ep, step, stride)
        if cfg.smoke:
            anchors = anchors[: cfg.smoke]
        tag = f"{set_name}_{os.path.basename(root)}_ep{ep:03d}"
        store = {"norm_clean": [], "abs": [], "gt_abs": [], "gt_norm": [], "state": [], "frame": [], "subtask": [], "bucket_level": []}
        started = time.time()
        for a_i, gidx in enumerate(anchors):
            frame = probe_frame_inputs(ds, cfg, gidx, chunk, with_gripper_event_targets=False)
            gt_abs = frame["gt_actions"].numpy()
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"]).numpy()
            D = gt_abs.shape[-1]
            gt_norm = gt_norm[:, :D]
            sub = strings[gidx]
            mu, var3, ref_n, level, ref_dist = reference.lookup(sub, frame["state"].numpy(), f"{root}#{int(ep)}")
            seed = 1_000_003 * (zlib.crc32(f"{set_name}|{root}|{ep}".encode()) % 10_007) + int(gidx)
            norm_by, abs_by = {}, {}
            t_anchor = time.time()
            for s_i, (name, kind, lvl) in enumerate(settings):
                obs = perturb(frame["obs"], kind, lvl, seed + 17 * s_i)
                norm, absolute = sampler.draws(frame, obs, seed)
                norm_by[name], abs_by[name] = norm[:, :, :D], absolute[:, :, :D]
            timing.append(time.time() - t_anchor)
            alpha, sigma = affine_map(sampler, sampler.last_batch, frame["obs"][OBS_STATE], chunk, sampler.last_width, D)
            if affine_check is None:
                affine_check = float(np.abs((abs_by["clean"] - alpha) / sigma - norm_by["clean"]).max())
                logging.info(f"[draw_spread] affine map check on the clean draws: max |delta| = {affine_check:.2e}")
            if consistency is None:
                single = sampler.reference_draw(frame, frame["obs"], seed)[:, :D]
                mine, _ = sampler.draws(frame, frame["obs"], seed, K=1)
                consistency = float(np.abs(single - mine[0, :, :D]).max())
                logging.info(f"[draw_spread] shared-prefix K=1 vs policy single path at the same seed: max |delta| = {consistency:.2e}")
            base = {"set": set_name, "root": root, "episode": int(ep), "frame": int(table["frame"][gidx]), "global_idx": int(gidx),
                    "time_s": float(table["frame"][gidx] / fps), "subtask": sub, "group": subtask_group(sub),
                    "progress_bin": bucket_key(sub, float(progress[gidx]))[1], "bucket_level": level,
                    "bucket_n": int(ref_n), "ref_state_dist": ref_dist}
            mu = mu[:, :D] if mu is not None else np.full_like(gt_norm, np.nan)
            var3 = var3 if var3 is not None else np.full(3, np.nan)
            for name, _, _ in settings:
                n = norm_by[name] if name == "clean" else (abs_by[name] - alpha) / sigma  # re-anchored to the clean state
                rec = {**base, "setting": name, **anchor_metrics(n, gt_norm, mu, var3)}
                if name != "clean":
                    rec.update(noise_metrics(n, norm_by["clean"]))
                records.append(rec)
            store["norm_clean"].append(norm_by["clean"])
            store["abs"].append(np.stack([abs_by[n] for n, _, _ in settings]))
            store["gt_abs"].append(gt_abs)
            store["gt_norm"].append(gt_norm)
            store["state"].append(frame["state"].numpy())
            store["frame"].append(int(table["frame"][gidx]))
            store["subtask"].append(sub)
            store["bucket_level"].append(level)
            if a_i % 20 == 0 or a_i == len(anchors) - 1:
                per = np.mean(timing[-50:])
                logging.info(f"[draw_spread] {tag} anchor {a_i + 1}/{len(anchors)}  {per:.2f} s/anchor ({len(settings)} settings)  "
                             f"elapsed {time.time() - started:.0f} s")
        total_anchors += len(anchors)
        np.savez_compressed(
            os.path.join(cfg.out_dir, "episodes", f"{tag}.npz"),
            norm_clean=np.stack(store["norm_clean"]).astype(np.float32),
            abs=np.stack(store["abs"]).astype(np.float32),
            gt_abs=np.stack(store["gt_abs"]).astype(np.float32),
            gt_norm=np.stack(store["gt_norm"]).astype(np.float32),
            state=np.stack(store["state"]).astype(np.float32),
            frame=np.asarray(store["frame"]),
            subtask=np.asarray(store["subtask"]),
            bucket_level=np.asarray(store["bucket_level"]),
            settings=np.asarray([s[0] for s in settings]),
        )
        pd.DataFrame(records).to_csv(os.path.join(cfg.out_dir, "anchors.csv"), index=False)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(cfg.out_dir, "anchors.csv"), index=False)
    run_info = {
        "checkpoint": str(cfg.policy.pretrained_path), "val_dataset_path": cfg.val_dataset_path,
        "train_episodes": resolved_train, "n_draws": cfg.n_draws, "anchor_seconds": cfg.anchor_seconds,
        "settings": settings, "num_inference_steps": sampler.steps, "chunk_size": chunk, "image_stride": stride,
        "reference_anchors": int(len(ref_df)), "reference_roots": corpus_roots, "anchors": int(total_anchors),
        "consistency_max_abs_delta": consistency, "affine_check_max_abs_delta": affine_check, "seconds_per_anchor": float(np.mean(timing)) if timing else None,
    }
    json.dump(run_info, open(os.path.join(cfg.out_dir, "run.json"), "w"), indent=2, default=str)
    df["episode"] = df.apply(lambda r: f"{os.path.basename(r.root)} ep{r.episode}", axis=1)
    preamble = (f"<p>Checkpoint {run_info['checkpoint']}. {cfg.n_draws} draws per anchor, anchors every {cfg.anchor_seconds} s, "
                f"{sampler.steps} flow steps, chunk {chunk}. Space: normalized anchor-encoded actions; offset = per-joint chunk mean, "
                f"shape = the rest. Reference per anchor = the {K_REF} training-corpus anchors ({len(ref_df)} candidates from {len(corpus_roots)} roots, "
                f"same cadence) nearest in state among those with the same subtask string (the verb when that pool has fewer than {MIN_POOL}), "
                f"never from the anchor's own episode. Shared-prefix check vs the policy's own path: max |delta| = {consistency:.1e}. "
                f"Noise settings: {cfg.noise_settings} (state = Gaussian in degrees on every joint, image = pixel std on [0, 1]).</p>"
                f"<p>Reference level per set (share of anchors whose pool was the same subtask string / the verb / none): "
                + "; ".join(f"{s}: " + ", ".join(f"{lvl} {share:.0%}" for lvl, share in df[(df.setting == 'clean') & (df.set == s)].bucket_level.value_counts(normalize=True).items()) for s in ("train", "val"))
                + f". Median standardized state distance to the reference neighbours: "
                + ", ".join(f"{s} {df[(df.setting == 'clean') & (df.set == s)].ref_state_dist.median():.2f}" for s in ("train", "val")) + ".</p>")
    write_report(df, os.path.join(cfg.out_dir, "report.html"), "Draw spread, train vs val", preamble, None)
    logging.info(f"[draw_spread] wrote {cfg.out_dir}/report.html ({total_anchors} anchors, {np.mean(timing):.2f} s/anchor)")


@parser.wrap()
def cli(cfg: DrawSpreadConfig):
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
