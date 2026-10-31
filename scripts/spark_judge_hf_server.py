"""Fallback judge server without vllm: transformers + FastAPI serving the subset of the OpenAI chat-completions API the
vlm_judge client uses (messages with text and image_url data-URI parts, max_tokens, temperature). Runs in ~/judge-venv
built with `spark_judge_setup.sh --fallback`; started by `spark_judge_serve.sh hf gemma|qwen`. Single request at a
time, bf16, greedy when temperature is 0. UNTESTED on the Spark (written 2026-10-30 without the models)."""

import argparse
import base64
import io
import time

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

app = FastAPI()
STATE: dict = {}


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int = 300
    temperature: float = 0.0
    response_format: dict | None = None
    chat_template_kwargs: dict = {}


def to_hf(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append(dict(role=m["role"], content=[dict(type="text", text=content)]))
            continue
        parts = []
        for p in content:
            if p["type"] == "text":
                parts.append(dict(type="text", text=p["text"]))
            elif p["type"] == "image_url":
                url = p["image_url"]["url"]
                raw = base64.b64decode(url.split(",", 1)[1])
                parts.append(dict(type="image", image=Image.open(io.BytesIO(raw)).convert("RGB")))
        out.append(dict(role=m["role"], content=parts))
    return out


@app.get("/v1/models")
def models() -> dict:
    return dict(object="list", data=[dict(id=STATE["name"], object="model")])


@app.post("/v1/chat/completions")
def chat(req: ChatRequest) -> dict:
    processor, model = STATE["processor"], STATE["model"]
    inputs = processor.apply_chat_template(to_hf(req.messages), add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                           **req.chat_template_kwargs).to(model.device)
    n_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_tokens, do_sample=req.temperature > 0, temperature=req.temperature or None)
    text = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
    return dict(id=f"hf-{int(time.time() * 1000)}", object="chat.completion", model=STATE["name"],
                choices=[dict(index=0, message=dict(role="assistant", content=text), finish_reason="stop")],
                usage=dict(prompt_tokens=int(n_in), completion_tokens=int(out.shape[1] - n_in), total_tokens=int(out.shape[1])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()
    STATE["name"] = args.model
    STATE["processor"] = AutoProcessor.from_pretrained(args.model)
    STATE["model"] = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
