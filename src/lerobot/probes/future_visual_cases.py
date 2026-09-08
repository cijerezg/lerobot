"""Fixed mistake and clean-subtask cases for future prediction and temporal attention.

Each case shows the actual RGB lookback, current and +horizon scenes, predicted
and target latent colors, full-dimensional normalized L1 error, and temporal
attention by camera/layer/age. Colors use one seeded projection and fixed scale;
they are not generated RGB. Raw vectors and per-head attention are saved as NPZ.
Mistake labels select cases, never change the prompt or weight the displayed error.
An after-span frame is not assumed to be recovery. Unrecorded futures are missing,
not replaced by a terminal frame. No target/PCA fitting occurs during this probe.

Standalone: python -m lerobot.probes.future_visual_cases --config config_rl.yaml
Add --preview to select/decode the cases without loading a model. Standalone model
runs require --policy.pretrained_path pointing to the checkpoint being examined.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.manifest import Panel, write_index
from lerobot.probes.utils import as_image, probe_frame_inputs, probe_image_stride, suppress_pack_dropout


def select_cases(root, *, episode=3, horizon_seconds=4.0, history_seconds=(6., 4., 2.), stride=3):
    """Deterministic annotation-only selection, snapped to the episode-local grid."""
    root = Path(root)
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    fps = float(info["fps"])
    offset = round(horizon_seconds * fps)
    if abs(offset - horizon_seconds * fps) > 1e-6 or offset < 1 or offset % stride:
        raise ValueError("Future horizon must land on the image-stride grid.")
    segments = pd.read_parquet(meta / "episode_metadata.parquet").to_dict("records")
    mistakes = [r for r in pd.read_parquet(meta / "mistakes.parquet").to_dict("records") if r["mistake"]]
    bounds = {}
    for row in segments:
        ep = int(row["episode_index"])
        lo, hi = bounds.get(ep, (int(row["from_index"]), int(row["to_index"])))
        bounds[ep] = min(lo, int(row["from_index"])), max(hi, int(row["to_index"]))
    if episode not in bounds:
        raise ValueError(f"Focus episode {episode} is absent; recorded episode IDs are {sorted(bounds)}.")
    cases, omitted = [], []
    lookback = round(max(history_seconds, default=0) * fps)

    def add(index, ep, label, event=None):
        lo, hi = bounds[ep]
        if not lo <= index < hi:
            omitted.append({"label": label, "episode": ep, "reason": "outside recorded episode"})
            return
        index = lo + (int(index) - lo) // stride * stride
        segment = next(r for r in segments if int(r["episode_index"]) == ep and r["from_index"] <= index < r["to_index"])
        active = [r for r in mistakes if int(r["episode_index"]) == ep and r["from_index"] <= index < r["to_index"]]
        future_valid = index + offset < hi
        cases.append({
            "id": f"ep{ep:02d}_fr{index-lo:06d}_{len(cases):02d}",
            "label": label, "episode": ep, "global_index": index, "frame_index": index-lo,
            "seconds": (index-lo)/fps, "subtask": str(segment["subtask"]),
            "mistake": bool(active), "mistake_types": [str(r["mistake_type"]) for r in active],
            "event": event, "future_index": index+offset if future_valid else None,
            "future_status": "recorded" if future_valid else "beyond recorded episode",
            "history_indices": [max(lo, index-round(t*fps)) for t in history_seconds],
            "history_padded": [index-round(t*fps) < lo for t in history_seconds],
        })

    focused = sorted([r for r in mistakes if int(r["episode_index"]) == episode], key=lambda r:r["from_index"])
    other = sorted(
        [r for r in mistakes if int(r["episode_index"]) != episode
         and r["to_index"] + round(2*fps) + offset < bounds[int(r["episode_index"])][1]],
        key=lambda r:(abs(int(r["episode_index"])-episode), r["from_index"]),
    )[:1]
    for event in focused + other:
        ep, start, end = int(event["episode_index"]), int(event["from_index"]), int(event["to_index"])
        event_info = {"start": start, "end": end, "type": str(event["mistake_type"]), "note": str(event.get("note", ""))}
        for phase, anchor in (("before", start-round(2*fps)), ("during", (start+end)//2), ("after span", end+round(2*fps))):
            add(anchor, ep, f"{event['mistake_type']}: {phase}", event_info)

    # Prefer a clean exact-subtask counterpart; verb-level fallback is labelled.
    def clean_candidates(predicate):
        found = []
        for row in segments:
            if int(row.get("quality", 0)) < 4 or not predicate(str(row["subtask"])):
                continue
            ep = int(row["episode_index"])
            lo, hi = bounds[ep]
            anchor = lo + ((int(row["from_index"])+int(row["to_index"]))//2-lo)//stride*stride
            if anchor-lookback < lo or anchor+offset >= hi:
                continue
            if any(int(m["episode_index"]) == ep and m["from_index"] <= anchor+offset
                   and m["to_index"] > anchor-lookback for m in mistakes):
                continue
            found.append((ep != episode, ep, anchor, row))
        return sorted(found, key=lambda r:r[:3])

    used = {c["global_index"] for c in cases}
    mistake_subtasks = sorted({c["subtask"] for c in cases if c["mistake"]})
    for task in mistake_subtasks:
        choices = clean_candidates(lambda s:s == task)
        match = "same subtask"
        if not choices:
            choices = clean_candidates(lambda s:s.split()[0] == task.split()[0])
            match = "same verb; different object/subtask"
        choice = next((c for c in choices if c[2] not in used), None)
        if choice:
            _, ep, anchor, _ = choice
            add(anchor, ep, f"clean comparison ({match})")
            cases[-1]["comparison_for"] = task
            used.add(anchor)
    choice = next((c for c in clean_candidates(lambda s:s.lower().startswith(("move ", "transport ")))
                   if c[2] not in used), None)
    if choice:
        add(choice[2], choice[1], "clean transport")

    digest = hashlib.sha256()
    for name in ("info.json", "episode_metadata.parquet", "mistakes.parquet"):
        digest.update((meta/name).read_bytes())
    return {"dataset": str(root.resolve()), "annotation_sha256": digest.hexdigest(), "fps": fps,
            "horizon_seconds": horizon_seconds, "history_seconds": list(history_seconds),
            "image_stride": stride, "focus_episode": episode, "cases": cases, "omitted": omitted}


def latent_error(prediction, target, eps=1e-6):
    """Unweighted per-patch error in the same normalized space as the auxiliary."""
    p = F.layer_norm(prediction.float(), (prediction.shape[-1],), eps=eps)
    t = F.layer_norm(target.float(), (target.shape[-1],), eps=eps)
    return (p-t).abs().mean(-1)


@torch.no_grad()
def capture_case(policy, batch, *, cameras, future_pixels=None):
    """ViT-only readout; no target updates, VLM, action sampling, or calibration."""
    from lerobot.policies.molmoact2.future_visual import normalize_taps
    from lerobot.policies.molmoact2.modeling_molmoact2 import _MEM_TEMPORAL_CAPTURE

    vision = policy._backbone().vision_backbone
    aux = getattr(policy, "future_visual", None)
    previous_history = getattr(vision, "_lerobot_history", None)
    previous_feature_capture = getattr(vision, "_lerobot_capture_future_features", False)
    previous_capture = dict(_MEM_TEMPORAL_CAPTURE)
    was_training = policy.training
    result = {"prediction_status": "auxiliary absent in checkpoint"}
    try:
        policy.eval()
        vision._lerobot_capture_future_features = False
        _MEM_TEMPORAL_CAPTURE.update(enabled=True, records=[])
        current = batch["pixel_values"].reshape(1, cameras, *batch["pixel_values"].shape[1:])
        history = batch.get("history_images")
        vision._lerobot_history = None if history is None else (
            history, batch["history_image_times"], batch["history_images_mask"]
        )
        pixels = ((current.float()+1)*0.5*255).round().clamp(0,255)/255*2-1
        features = vision.encode_image(pixels.to(dtype=next(vision.image_vit.parameters()).dtype))
        records = _MEM_TEMPORAL_CAPTURE["records"]
        result["layers"] = [i for i in range(len(vision.image_vit.transformer.resblocks))
                            if (i+1) % int(vision._lerobot_temporal_layer_stride) == 0] if records else []
        if records:
            age = torch.stack([r["by_bc_head_age"] for r in records])
            patch_age = torch.stack([r["by_bc_patch_age"] for r in records])
            result["attention"] = torch.cat([age, (1-age.sum(-1,keepdim=True)).clamp_min(0)],-1)
            result["patch_attention"] = torch.cat([patch_age, (1-patch_age.sum(-1,keepdim=True)).clamp_min(0)],-1)
            result["value_magnitude"] = torch.stack([r["by_bc_head_key_value"] for r in records])
        if aux is not None:
            result["prediction_status"] = "ready" if bool(aux.ready) and int(aux.optimizer_steps)>0 else "auxiliary untrained or PCA not ready"
            if result["prediction_status"] == "ready":
                prediction = aux.predictor(normalize_taps(features, aux.tap_width, aux.cfg.eps))
                result["prediction"] = prediction[0]
                result["target_update_step"] = int(aux.last_target_update)
                result["pca_update_step"] = int(aux.last_pca_update)
                if future_pixels is not None:
                    future = future_pixels.reshape_as(current)
                    target = normalize_taps(aux.encode_target(future), aux.tap_width, aux.cfg.eps) @ aux.basis.float()
                    present = normalize_taps(aux.encode_target(current), aux.tap_width, aux.cfg.eps) @ aux.basis.float()
                    result.update(target=target[0], error=latent_error(prediction,target,aux.cfg.eps)[0],
                                  persistence_error=latent_error(present,target,aux.cfg.eps)[0])
    finally:
        vision._lerobot_history = previous_history
        vision._lerobot_capture_future_features = previous_feature_capture
        _MEM_TEMPORAL_CAPTURE.clear()
        _MEM_TEMPORAL_CAPTURE.update(previous_capture)
        policy.train(was_training)
    return {k:v.detach().float().cpu().numpy() if torch.is_tensor(v) else v for k,v in result.items()}


def _image_url(array):
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    image.thumbnail((384,384))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def latent_colors(features):
    """One fixed seeded color projection, never independently rescaled per map."""
    values = np.asarray(features, dtype=np.float32)
    values = (values-values.mean(-1,keepdims=True))/np.sqrt(values.var(-1,keepdims=True)+1e-6)
    projection = np.random.default_rng(0).normal(size=(values.shape[-1],3))/math.sqrt(values.shape[-1])
    return np.rint((np.tanh(values@projection/2)+1)*127.5).clip(0,255).astype(np.uint8)


def render_report(output_dir, manifest, views, *, mode):
    """Self-contained interactive HTML; compact plots plus complete raw NPZ files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    data = {"manifest":manifest,"views":views,"mode":mode}
    template = Path(__file__).with_name("future_visual_cases.html").read_text()
    payload = json.dumps(data, allow_nan=False).replace("<","\\u003c")
    (output_dir/"cases.html").write_text(template.replace("__CASE_DATA__",payload))
    (output_dir/"cases.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    write_index(str(output_dir),sys.modules[__name__],title="Future prediction: fixed cases",group="History",
                claim="What does the model predict and read around mistakes and clean subtasks?",
                metrics=[],panels=[Panel("cases.html",caption="Case browser",primary=True,
                    how="Select a fixed frame, camera and temporal layer. Error uses all latent channels; feature colors use three fixed projections.")],
                summary={"cases":len(views),"mode":mode},see_also=["mem_temporal_attention","mem_history_influence"])


def run(adapter, dataset, cfg, output_dir):
    """With adapter=None, render an explicitly labelled dataset-only case preview."""
    p = cfg.probe_parameters
    history = cfg.policy.memory.history_times_seconds()
    manifest = select_cases(dataset.root,episode=p.future_visual_case_episode,
                            horizon_seconds=cfg.policy.future_visual_loss.horizon_seconds,
                            history_seconds=history,stride=probe_image_stride(cfg))
    checkpoint = getattr(cfg.policy,"pretrained_path",None)
    manifest["checkpoint"] = str(checkpoint) if checkpoint is not None else None
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    views=[]
    preprocessor = adapter._preprocessor if adapter is not None else None
    from contextlib import nullcontext
    with suppress_pack_dropout(preprocessor) if preprocessor is not None else nullcontext():
        for number, case in enumerate(manifest["cases"]):
            logging.info("[future_visual_cases] %s/%s %s",number+1,len(manifest["cases"]),case["id"])
            frame = probe_frame_inputs(dataset,cfg,case["global_index"],1,with_depth=False)
            future = None if case["future_index"] is None else probe_frame_inputs(
                dataset,cfg,case["future_index"],1,with_depth=False,with_history=False)
            keys = [key for key in cfg.policy.image_keys if key in frame["obs"]]
            historical = [probe_frame_inputs(dataset,cfg,i,1,with_depth=False,with_history=False)
                          for i in case["history_indices"]]
            view={"id":case["id"],"cameras":[],"layers":[],"prediction_status":"dataset preview; model not run"}
            readout={}
            if adapter is not None:
                batch=adapter._make_batch(frame["obs"],frame["task"],subtask=frame["subtask"],metadata=frame["metadata"])
                future_batch=None if future is None else adapter._make_batch(future["obs"],future["task"])
                all_keys=adapter._configured_image_keys()
                readout=capture_case(adapter.policy,batch,cameras=len(all_keys),
                                     future_pixels=None if future_batch is None else future_batch["pixel_values"])
                arrays={key:value for key,value in readout.items() if isinstance(value,np.ndarray)}
                np.savez_compressed(output/f"{case['id']}.npz",**arrays)
                view.update(layers=readout["layers"],prediction_status=readout["prediction_status"],raw=f"{case['id']}.npz")
                case["model_status"]={k:v for k,v in readout.items() if not isinstance(v,np.ndarray)}
                view["history_active"]="history_images" in batch
            else:
                all_keys=keys
                view["history_active"]=False
            for key in keys:
                camera={"name":key.rsplit(".",1)[-1],"scenes":[_image_url(as_image(f["obs"][key])) for f in historical+[frame]],
                        "future":None if future is None else _image_url(as_image(future["obs"][key]))}
                index=all_keys.index(key)
                if "attention" in readout:
                    camera["attention"]=readout["attention"][:,index].mean(1).round(6).tolist()
                    camera["patch_attention"]=readout["patch_attention"][:,index].round(5).tolist()
                for name in ("prediction","target"):
                    if name in readout:
                        values=readout[name][index]
                        side=math.isqrt(values.shape[0])
                        camera[name]=_image_url(latent_colors(values).reshape(side,side,3))
                for name in ("error","persistence_error"):
                    if name in readout:
                        values=readout[name][index]
                        camera[name]=values.round(6).tolist()
                        camera[f"mean_{name}"]=float(values.mean())
                view["cameras"].append(camera)
            views.append(view)
    render_report(output,manifest,views,mode="dataset preview" if adapter is None else "checkpoint readout")
    return {"cases":len(views)}


@parser.wrap()
def cli(cfg: TrainRLServerPipelineConfig):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.probes.base import ProbablePolicy
    from lerobot.probes.utils import load_probe_dataset
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.utils import init_logging
    init_logging()
    if not cfg.val_dataset_path:
        raise ValueError("val_dataset_path is required.")
    if not _PREVIEW:
        if not cfg.policy.pretrained_path:
            raise ValueError("Set policy.pretrained_path to the checkpoint to inspect, or use --preview.")
        from lerobot.configs import PreTrainedConfig
        checkpoint = cfg.policy.pretrained_path
        saved = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        saved.pretrained_path = checkpoint
        cfg.policy = saved
    dataset=LeRobotDataset(repo_id=cfg.dataset.repo_id,root=cfg.val_dataset_path)
    dataset.delta_timestamps=dataset.delta_indices=None
    adapter=None
    if not _PREVIEW:
        if not cfg.policy.pretrained_path:
            raise ValueError("Set policy.pretrained_path to the checkpoint to inspect, or use --preview.")
        adapter=ProbablePolicy.for_config(cfg,get_safe_torch_device(try_device=cfg.policy.device),dataset=load_probe_dataset(cfg))
    run(adapter,dataset,cfg,Path(cfg.probe_parameters.output_dir)/"future_visual_cases")


_PREVIEW=False


def main():
    global _PREVIEW
    from lerobot.probes.utils import register_config_choices
    from lerobot.scripts.rl_offline import _extract_config_path_args,_preprocess_config_yaml
    register_config_choices()
    args=sys.argv[1:]
    _PREVIEW="--preview" in args
    args=[arg for arg in args if arg != "--preview"]
    config,remaining=_extract_config_path_args(args)
    sys.argv=[sys.argv[0],*remaining,*([f"--config_path={_preprocess_config_yaml(config)}"] if config else [])]
    cli()


if __name__ == "__main__":
    main()
