#!/usr/bin/env python
"""Model explorer: scrub a dataset episode, edit the prompt, watch the chunk and the
attention change live.

    uv run python -m lerobot.scripts.model_explorer --config_path config_rl.yaml

Checkpoint = ``policy.pretrained_path`` from the config (``inference_checkpoint_path``
wins when set, as for standalone inference). One model resident on the GPU, one lock;
every forward runs on the deployed path (`probe_frame_inputs` + the probe adapter), so
what the page shows is what a rollout would compute for that frame and that prompt.

Two attention reads per request, from one capture forward:

* action side: the action expert's cross attention (queries = the 30 action tokens,
  keys = the VLM prefix) at ``action_layer`` — per camera patch grids, the depth block,
  and the mass on every prompt token, all per head;
* VLM side: the language model's own attention at ``vlm_layer`` for every prompt text
  token (query) over every prefix position (key) — one patch grid per token, so
  clicking a word shows where the VLM looked when it read it.

Endpoints are plain JSON; the page is ``model_explorer.html`` next to this file.
"""

from __future__ import annotations

import base64
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.attention import _decode_token_label, _extract_overlay_grids
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import (
    DEPLOYMENT_METADATA,
    _dataset_subtask_indices,
    build_episode_index,
    frame_metadata_lookup,
    get_subtask_str,
    joint_names_for_dim,
    load_extra_dataset,
    load_probe_dataset,
    probe_frame_inputs,
    probe_image_stride,
    register_config_choices,
)
from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.action_smoothing import apply_butterworth_filter
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

HTML_PATH = Path(__file__).with_name("model_explorer.html")


def _plotly_js_path() -> Path:
    import plotly

    return Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


@dataclass
class ExplorerConfig(TrainRLServerPipelineConfig):
    """Explorer-only knobs; everything else is the training config."""

    explorer_roots: list[str] = field(
        default_factory=lambda: [
            "outputs/rebot_val-annotated-v4",
            "outputs/rebot_inference_2026-09-08-v1",
            "outputs/rebot_rollouts-annotated-v2",
        ]
    )
    explorer_host: str = "127.0.0.1"
    explorer_port: int = 7871


class PredictRequest(BaseModel):
    dataset: str
    episode: int
    frame: int
    task: str
    subtask: str = ""
    metadata: dict | None = None
    seeds: int = 1
    action_layer: int = 9
    vlm_layer: int | None = None  # VLM prompt-row attention; the page no longer asks for it
    vlm_head: int = -1
    fast: bool = False


class GenerateRequest(BaseModel):
    dataset: str
    episode: int
    frame: int
    task: str


def _jpeg_data_url(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _model_view_np(img_t: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] in [-1,1] -> uint8 HWC (same de-normalisation as the attention probe)."""
    img_t = img_t.squeeze(0).detach().float().cpu() * 0.5 + 0.5
    return (img_t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _depth_np(img_t: torch.Tensor) -> np.ndarray:
    gray = _model_view_np(img_t)[..., 0]
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def _round(t: torch.Tensor | np.ndarray, digits: int = 4) -> list:
    return np.asarray(t, dtype=np.float64).round(digits).tolist()


def _bands(values: list, labels: dict | None = None) -> list[dict]:
    """Run-length encode a per-frame label list into [{start, end, label}] (end exclusive)."""
    out: list[dict] = []
    for pos, value in enumerate(values):
        if out and out[-1]["value"] == value:
            out[-1]["end"] = pos + 1
        else:
            out.append({"start": pos, "end": pos + 1, "value": value})
    for band in out:
        band["label"] = labels.get(band["value"], str(band["value"])) if labels is not None else band["value"]
    return out


class Explorer:
    def __init__(self, cfg: ExplorerConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.lock = threading.Lock()
        self.primary = load_probe_dataset(cfg)
        self.adapter = ProbablePolicy.for_config(cfg, device, dataset=self.primary)
        # Batch size follows the seed count, so replayed inference graphs cannot be used.
        self.adapter._set_probe_cuda_graph_enabled(False)
        self.stride = probe_image_stride(cfg)
        self.chunk_size = self.adapter.chunk_size

        from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep

        self.pack_step = next(
            s for s in self.adapter.preprocessor.steps if isinstance(s, MolmoAct2PackInputsProcessorStep)
        )
        self.subtask_vocab = list(self.pack_step.subtask_names)

        self.roots: dict[str, str] = {}
        for root in [str(self.primary.root), *cfg.explorer_roots]:
            self.roots.setdefault(Path(root).name, root)
        self.datasets: dict[str, Any] = {Path(str(self.primary.root)).name: self.primary}
        self.episode_indices: dict[str, dict[int, list[int]]] = {}
        self.metadata_lookup: dict[str, dict[int, dict]] = {}
        self.subtask_by_frame: dict[str, Any] = {}
        self.frame_cache: OrderedDict[tuple, dict] = OrderedDict()
        self.capture_cache: OrderedDict[tuple, Any] = OrderedDict()

        self.kin = RebotKinematics()
        self.table_z = float(getattr(cfg.probe_parameters, "trace_table_z", 0.0))

        policy = self.adapter.policy
        self.n_action_layers = len(policy._action_expert().blocks)
        self.n_vlm_layers = int(policy._backbone().transformer.config.num_hidden_layers)
        self.checkpoint = str(getattr(cfg, "inference_checkpoint_path", None) or cfg.policy.pretrained_path)

    # ── datasets ─────────────────────────────────────────────────────────────

    def dataset(self, name: str):
        if name not in self.datasets:
            if name not in self.roots:
                raise KeyError(f"unknown dataset {name!r}")
            self.datasets[name] = load_extra_dataset(self.cfg.dataset.repo_id, self.roots[name])
        return self.datasets[name]

    def episodes(self, name: str) -> dict[int, list[int]]:
        if name not in self.episode_indices:
            self.episode_indices[name] = build_episode_index(self.dataset(name))
        return self.episode_indices[name]

    def metadata_for(self, name: str) -> dict[int, dict]:
        if name not in self.metadata_lookup:
            self.metadata_lookup[name] = frame_metadata_lookup(self.dataset(name))
        return self.metadata_lookup[name]

    def episode_summary(self, name: str, episode: int) -> dict:
        ds = self.dataset(name)
        indices = self.episodes(name)[episode]
        subtask_indices = _dataset_subtask_indices(ds)
        subtask_bands: list[dict] = []
        if subtask_indices is not None:
            per_frame = [int(subtask_indices[g]) for g in indices]
            names = {i: get_subtask_str(ds, i) or "(none)" for i in set(per_frame)}
            subtask_bands = _bands(per_frame, names)
        lookup = self.metadata_for(name)
        mistake_bands = _bands([bool(lookup.get(g, {}).get("mistake", False)) for g in indices])
        return {
            "n_frames": len(indices),
            "subtasks": subtask_bands,
            "mistakes": [b for b in mistake_bands if b["value"]],
        }

    def snap(self, name: str, episode: int, frame: int) -> tuple[int, int]:
        """Clamp to the episode and back onto the image/depth stride grid."""
        indices = self.episodes(name)[episode]
        pos = min(max(int(frame), 0), len(indices) - 1)
        pos -= pos % self.stride
        return pos, indices[pos]

    # ── frame ─────────────────────────────────────────────────────────────────

    def frame(self, name: str, episode: int, frame: int) -> dict:
        pos, global_idx = self.snap(name, episode, frame)
        key = (name, episode, pos)
        if key in self.frame_cache:
            self.frame_cache.move_to_end(key)
            return self.frame_cache[key]
        ds = self.dataset(name)
        inputs = probe_frame_inputs(ds, self.cfg, global_idx, self.chunk_size)
        dataset_meta = self.metadata_for(name).get(global_idx)
        gt = inputs["gt_actions"]
        payload = {
            "frame": pos,
            "global_idx": int(global_idx),
            "task": inputs["task"],
            "subtask": inputs["subtask"],
            "deployment_metadata": dict(DEPLOYMENT_METADATA),
            "dataset_metadata": dataset_meta,
            "state": _round(inputs["state"], 3) if inputs["state"] is not None else None,
            "gt": _round(gt, 3),
            "joint_names": joint_names_for_dim(int(gt.shape[1])),
        }
        with self.lock:
            images, depth = self._model_views(inputs)
        payload["images"] = images
        payload["depth"] = depth
        self.frame_cache[key] = {"inputs": inputs, "payload": payload}
        while len(self.frame_cache) > 64:
            self.frame_cache.popitem(last=False)
        return self.frame_cache[key]

    def _model_views(self, inputs: dict) -> tuple[dict[str, str], str | None]:
        """Molmo's global crop per present camera, plus the display-scaled depth map."""
        adapter = self.adapter
        obs = inputs["obs"]
        obs_dev = {k: v.to(self.device) for k, v in obs.items()}
        obs_dev, presence = adapter._fill_absent_cameras(obs_dev)
        batch = adapter._make_batch(obs, inputs["task"], subtask=inputs["subtask"], metadata=inputs["metadata"])
        seq_len = int(batch["input_ids"].shape[1])
        _, _, _, extras = adapter._image_attention_metadata(batch, obs_dev, seq_len)
        present = self._present_cams(obs_dev, presence)
        images = {
            cam: _jpeg_data_url(_model_view_np(tensor))
            for cam, tensor in extras.get("image_tensors_by_segment", {}).items()
            if cam in present
        }
        depth = None
        depth_extras = adapter._depth_attention_extras(batch, obs_dev, seq_len)
        segment = depth_extras.get("depth_segment")
        if segment and torch.is_tensor(segment.get("image")):
            depth = _jpeg_data_url(_depth_np(segment["image"]))
        return images, depth

    def _present_cams(self, obs_dev: dict, presence: dict) -> set[str]:
        keys = self.adapter._image_keys_for_obs(obs_dev)
        return {
            self.adapter._safe_cam_name(key, idx)
            for idx, key in enumerate(keys)
            if presence.get(f"camera_is_present.{key}", True)
        }

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self, req: PredictRequest) -> dict:
        entry = self.frame(req.dataset, req.episode, req.frame)
        inputs = entry["inputs"]
        obs = inputs["obs"]
        subtask = req.subtask.strip() or None
        metadata = dict(req.metadata) if req.metadata else None
        seeds = max(int(req.seeds), 1)
        adapter = self.adapter

        with self.lock:
            noise = torch.cat([adapter.flow_noise_like(1, seed) for seed in range(seeds)], dim=0)
            unnorm, norm = adapter.predict_action_chunk_batch(
                obs, req.task, [subtask] * seeds, metadatas=[metadata] * seeds if metadata else None,
                noise=noise, inference_action_mode="continuous",
            )
            fast = None
            if req.fast:
                try:
                    fast_unnorm, _ = adapter.predict_action_chunk_batch(
                        obs, req.task, [subtask], metadatas=[metadata] if metadata else None,
                        inference_action_mode="discrete",
                    )
                    fast = fast_unnorm[0]
                except Exception as exc:  # the checkpoint may have no discrete head
                    logger.warning("FAST decode failed: %s", exc)
                    fast = None
            result = self._capture(entry, req.task, subtask, metadata, req.action_layer, req.vlm_layer)

        width = int(inputs["gt_actions"].shape[1])
        gt_norm = adapter.normalize_gt_actions(inputs["gt_actions"], inputs["state"])
        # As in action_trace: every PREDICTED chunk is the filtered command the runtimes
        # would send (zero-phase Butterworth); GT is a recording and stays raw.
        chunks = [apply_butterworth_filter(c[:, :width].double().numpy()) for c in unnorm]
        fast_np = apply_butterworth_filter(fast[:, :width].double().numpy()) if fast is not None else None
        gt_np = inputs["gt_actions"][:, :width].double().numpy()
        state_np = inputs["state"].double().numpy() if inputs["state"] is not None else None
        payload = {
            "chunks": _round(np.stack(chunks), 3),
            "chunks_norm": _round(norm[:, :, :width], 4),
            "gt_norm": _round(gt_norm[:, :width], 4),
            "fast": _round(fast_np, 3) if fast_np is not None else None,
            "ee": {
                "gt": _round(self.kin.ee_path(gt_np), 5),
                "seeds": _round(np.stack([self.kin.ee_path(c) for c in chunks]), 5),
                "fast": _round(self.kin.ee_path(fast_np), 5) if fast_np is not None else None,
                "start": _round(self.kin.ee_path(state_np[None])[0], 5) if state_np is not None else None,
                "table_z": self.table_z,
            },
            "attention": self._attention_payload(result, entry, req),
        }
        return payload

    def _capture(self, entry, task, subtask, metadata, action_layer, vlm_layer):
        key = (entry["payload"]["global_idx"], entry["inputs"]["task"], task, subtask,
               tuple(sorted((metadata or {}).items())), int(action_layer), vlm_layer)
        if key in self.capture_cache:
            self.capture_cache.move_to_end(key)
            return self.capture_cache[key]
        result = self.adapter.capture_attention(
            entry["inputs"]["obs"], task, timestep=0.5, layers=[int(action_layer)],
            vlm_layers=None if vlm_layer is None else [int(vlm_layer)], subtask=subtask, metadata=metadata,
        )
        self.capture_cache[key] = result
        while len(self.capture_cache) > 16:
            self.capture_cache.popitem(last=False)
        return result

    def _attention_payload(self, result, entry, req: PredictRequest) -> dict:
        obs_dev = {k: v.to(self.device) for k, v in entry["inputs"]["obs"].items()}
        obs_dev, presence = self.adapter._fill_absent_cameras(obs_dev)
        present = self._present_cams(obs_dev, presence)

        text_positions = [int(p) for p in result.extras.get("text_token_indices_by_segment", {}).get("language", [])]
        ids = result.task_tokens[0].detach().cpu()
        tokens = [{"pos": int(p), "label": _decode_token_label(result.tokenizer, ids[p])} for p in text_positions]

        # ── action expert → prefix ─────────────────────────────────────────
        layer = int(req.action_layer)
        cross = result.cross_attn_by_layer.get(layer)
        action: dict = {"layer": layer, "cams": {}, "depth": None, "prompt": None, "budget": None}
        if cross is not None:
            attn = torch.nan_to_num(cross[0].float().cpu(), nan=0.0)  # [H, n_act, enc]
            for grid in _extract_overlay_grids(result, layer):
                name = grid["cam_name"]
                item = {"grid_hw": list(grid["grid_hw"]), "heads": _round(grid["per_head_grid"])}
                if name.startswith("depth"):
                    item["null_bank"] = name == "depth_nullbank"
                    action["depth"] = item
                elif name in present:
                    action["cams"][name] = item
            if text_positions:
                idx = torch.as_tensor(text_positions, dtype=torch.long)
                action["prompt"] = _round(attn.index_select(2, idx).mean(dim=1))  # [H, T]
            action["budget"] = self._budget(attn, result, present, text_positions)

        # ── VLM prompt rows → prefix (only when asked for) ─────────────────
        vlm_layer = req.vlm_layer
        vlm_raw = None if vlm_layer is None else result.extras.get("vlm_attn_by_layer", {}).get(int(vlm_layer))
        vlm: dict = {"layer": vlm_layer, "head": int(req.vlm_head), "n_heads": 0, "cams": {}, "depth": None,
                     "budget": None}
        if vlm_raw is not None and text_positions:
            full = torch.nan_to_num(vlm_raw[0].float(), nan=0.0)  # [H, T, S]
            vlm["n_heads"] = int(full.shape[0])
            rows = full.mean(dim=0) if req.vlm_head < 0 else full[int(req.vlm_head)]  # [T, S]
            patch_indices = result.extras.get("image_patch_indices_by_segment", {})
            pooled = result.extras.get("image_pooling_by_segment", {})
            for name, indices in patch_indices.items():
                if name not in present or name in pooled:
                    continue
                n_p = int(len(indices) ** 0.5)
                if n_p * n_p != len(indices):
                    continue
                idx = torch.as_tensor(indices, dtype=torch.long)
                vlm["cams"][name] = {"grid_hw": [n_p, n_p], "tokens": _round(rows.index_select(1, idx))}
            segment = result.extras.get("depth_segment")
            if segment and segment.get("grid_hw") is not None:
                idx = torch.as_tensor(segment["indices"], dtype=torch.long)
                vlm["depth"] = {"grid_hw": list(segment["grid_hw"]), "tokens": _round(rows.index_select(1, idx)),
                                "null_bank": bool(segment.get("is_null_bank"))}
            vlm["budget"] = self._budget(rows.unsqueeze(1), result, present, text_positions, per_query=True)

        return {"tokens": tokens, "action": action, "vlm": vlm}

    def _budget(self, attn: torch.Tensor, result, present: set[str], text_positions: list[int],
                per_query: bool = False) -> dict:
        """Where a softmax row's mass goes: each camera, depth, prompt text, everything else.

        ``attn`` is [H, n_query, S]. Returns per-head totals averaged over queries, or,
        with ``per_query``, per-query totals (the leading axis is then the query)."""
        enc = int(attn.shape[-1])
        groups: dict[str, list[int]] = {}
        for name, indices in result.extras.get("image_patch_indices_by_segment", {}).items():
            if name in present:
                groups[name] = [int(i) for i in indices]
        segment = result.extras.get("depth_segment")
        if segment:
            groups["depth"] = [int(i) for i in segment["indices"] if 0 <= int(i) < enc]
        groups["prompt"] = [int(p) for p in text_positions]
        used = torch.zeros(enc, dtype=torch.bool)
        out: dict[str, list] = {}
        for name, indices in groups.items():
            idx = torch.as_tensor(indices, dtype=torch.long)
            used[idx] = True
            mass = attn.index_select(-1, idx).sum(-1)  # [H, n_query]
            out[name] = _round(mass[:, 0] if per_query else mass.mean(1))
        rest = attn[..., ~used].sum(-1)
        out["other"] = _round(rest[:, 0] if per_query else rest.mean(1))
        return out

    # ── subtask generation ────────────────────────────────────────────────────

    def generate_subtask(self, req: GenerateRequest) -> dict:
        from lerobot.policies.molmoact2.processor_molmoact2 import snap_to_subtask_vocab

        entry = self.frame(req.dataset, req.episode, req.frame)
        max_new_tokens = int(getattr(self.cfg.policy, "subtask_max_new_tokens", 0)) or 64
        with self.lock:
            self.pack_step.prompt_mode = "subtask_generation"
            try:
                batch = self.adapter._make_batch(entry["inputs"]["obs"], req.task)
            finally:
                self.pack_step.prompt_mode = "action"
            with torch.no_grad():
                token_ids = self.adapter.policy.generate_subtask_tokens(batch, max_new_tokens=max_new_tokens)
        tokenizer = self.pack_step.processor.tokenizer
        raw = tokenizer.decode(token_ids[0], skip_special_tokens=True).strip()
        index = snap_to_subtask_vocab(raw, self.subtask_vocab) if self.subtask_vocab else -1
        ids = batch["input_ids"][0].detach().cpu()
        mask = batch["attention_mask"][0].detach().cpu().to(torch.bool)
        prompt = tokenizer.decode([int(t) for t in ids[mask]], skip_special_tokens=False)
        return {"raw": raw, "name": self.subtask_vocab[index] if index >= 0 else raw, "index": index,
                "prompt": prompt.replace("<im_patch>", "")}

    # ── meta ──────────────────────────────────────────────────────────────────

    def meta(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "datasets": list(self.roots.keys()),
            "stride": self.stride,
            "chunk_size": self.chunk_size,
            "n_action_layers": self.n_action_layers,
            "n_vlm_layers": self.n_vlm_layers,
            "subtask_vocab": self.subtask_vocab,
            "deployment_metadata": dict(DEPLOYMENT_METADATA),
            "fps": float(self.cfg.env.fps),
        }


def build_app(explorer: Explorer) -> FastAPI:
    app = FastAPI(title="model explorer")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PATH.read_text()

    @app.get("/static/plotly.min.js")
    def plotly_js():
        return FileResponse(_plotly_js_path(), media_type="application/javascript")

    @app.get("/api/meta")
    def meta():
        return explorer.meta()

    @app.get("/api/episodes")
    def episodes(dataset: str):
        return {"episodes": [
            {"index": int(ep), "n_frames": len(indices)} for ep, indices in sorted(explorer.episodes(dataset).items())
        ]}

    @app.get("/api/episode")
    def episode(dataset: str, episode: int):
        return explorer.episode_summary(dataset, episode)

    @app.get("/api/frame")
    def frame(dataset: str, episode: int, frame: int):
        return explorer.frame(dataset, episode, frame)["payload"]

    @app.post("/api/predict")
    def predict(req: PredictRequest):
        try:
            return explorer.predict(req)
        except Exception as exc:
            logger.exception("predict failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/generate_subtask")
    def generate_subtask(req: GenerateRequest):
        try:
            return explorer.generate_subtask(req)
        except Exception as exc:
            logger.exception("generate_subtask failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


@parser.wrap()
def main(cfg: ExplorerConfig):
    import uvicorn

    init_logging()
    checkpoint = getattr(cfg, "inference_checkpoint_path", None)
    if checkpoint is not None:
        cfg.policy.pretrained_path = checkpoint
    device = get_safe_torch_device(try_device=cfg.policy.device)
    explorer = Explorer(cfg, device)
    logger.info("model explorer: %s on http://%s:%d", explorer.checkpoint, cfg.explorer_host, cfg.explorer_port)
    uvicorn.run(build_app(explorer), host=cfg.explorer_host, port=cfg.explorer_port, log_level="warning")


if __name__ == "__main__":
    register_config_choices()
    main()
