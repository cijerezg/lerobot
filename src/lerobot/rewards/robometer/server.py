"""Robometer-4B HTTP service for the Spark (port 8012). Runs in ~/robometer-venv, which holds UPSTREAM lerobot main with
the ``robometer`` extra (this fork predates the integration); run it by file path so ``import lerobot`` resolves to
the upstream package:

    ~/robometer-venv/bin/python <LeRobot>/lerobot/src/lerobot/rewards/robometer/server.py --port 8012

POST /score {"task": str, "frames": [data:image/jpeg;base64,... oldest first, <= max_frames]} ->
{"progress": [per frame in 0..1], "success": [per frame probability], "n_frames": int}. Uses the upstream encoder
(RobometerEncoderProcessorStep.encode_samples) and the model's per-frame heads (_compute_rbm_logits +
the module-level decode_progress_outputs) rather than compute_reward, which keeps only the last frame. GET /health ->
{"ok": true}. Signatures checked against upstream main d3793b7 (2026-10-23, the checkout in the Spark's flux3_lerobot);
the model itself has not been run (UNTESTED end to end)."""

import argparse
import base64
import io
import logging

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
app = FastAPI()
STATE: dict = {}


class ScoreRequest(BaseModel):
    task: str
    frames: list[str]


def load(pretrained: str, device: str, max_frames: int) -> None:
    from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
    from lerobot.rewards.robometer.modeling_robometer import decode_progress_outputs
    from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

    cfg = RobometerConfig(pretrained_path=pretrained, device=device, reward_output="progress", max_frames=max_frames)
    model = RobometerRewardModel.from_pretrained(pretrained, config=cfg).to(device).eval()
    encoder = RobometerEncoderProcessorStep(base_model_id=cfg.base_model_id, use_multi_image=cfg.use_multi_image,
                                            use_per_frame_progress_token=cfg.use_per_frame_progress_token, max_frames=max_frames)
    STATE.update(model=model, encoder=encoder, decode=decode_progress_outputs, device=device, max_frames=max_frames)


def decode(uri: str) -> np.ndarray:
    raw = base64.b64decode(uri.split(",", 1)[1] if uri.startswith("data:") else uri)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


@app.get("/health")
def health() -> dict:
    return dict(ok="model" in STATE, max_frames=STATE.get("max_frames"))


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    frames = np.stack([decode(f) for f in req.frames[-STATE["max_frames"]:]])
    encoded = STATE["encoder"].encode_samples([(frames, req.task)])
    inputs = {k: (v.to(STATE["device"]) if isinstance(v, torch.Tensor) else v) for k, v in encoded.items()}
    model = STATE["model"]
    with torch.no_grad():
        progress_logits, success_logits = model._compute_rbm_logits(inputs)
    # module-level upstream helper: {"progress_pred": [[per frame]], "success_probs": [[per frame]]}, one inner list per sample
    decoded = STATE["decode"](progress_logits, success_logits, is_discrete_mode=model.config.use_discrete_progress)
    progress = [min(max(float(v), 0.0), 1.0) for v in decoded["progress_pred"][0]]
    success = [float(v) for v in decoded["success_probs"][0]]
    return dict(progress=progress[: len(frames)], success=success[: len(frames)], n_frames=int(len(frames)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--pretrained", default="lerobot/Robometer-4B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=8)
    args = ap.parse_args()
    load(args.pretrained, args.device, args.max_frames)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
