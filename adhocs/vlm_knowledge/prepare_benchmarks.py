"""Public benchmarks in one format for ``run_benchmarks.py``.

    uv run python lerobot/adhocs/vlm_knowledge/prepare_benchmarks.py [--only=<name>]

Writes ``outputs/data/vlm_benchmarks/<name>/questions.jsonl`` + ``images/`` (a benchmark whose file exists is skipped):
  mmlu_redux   edinburgh-dawg/mmlu-redux-2.0, all subjects (5,700 text questions, 4 options)
  cv_bench     nyu-visionx/CV-Bench test (2,638, 2D + 3D)
  embspatial   FlagEval/EmbSpatial-Bench test (3,640, 4 options)
  vqa_v2       2,000 questions drawn with seed 0 from the official VQA v2 validation set, COCO val2014 images
One line per question: id, image (relative path or null), question, options (null = short answer), answer (index of
the right option, or the 10 human answers), category, meta.
"""
import argparse
import io
import json
import os
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

from datasets import get_dataset_config_names, load_dataset

OUT = "outputs/data/vlm_benchmarks"
VQA = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_{}_Val_mscoco.zip"
COCO = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"


def save(picture, folder: str, name: str) -> str:
    picture.convert("RGB").save(os.path.join(folder, "images", name))
    return "images/" + name


def mmlu_redux(folder: str) -> list[dict]:
    rows = []
    for subject in get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0"):
        for i, r in enumerate(load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject, split="test")):
            rows.append({"id": f"{subject}_{i}", "image": None, "question": r["question"], "options": r["choices"], "answer": r["answer"], "category": subject,
                         "meta": {"error_type": r["error_type"], "correct_answer": r["correct_answer"]}})
    return rows


def cv_bench(folder: str) -> list[dict]:
    rows = []
    for r in load_dataset("nyu-visionx/CV-Bench", split="test"):
        rows.append({"id": str(r["idx"]) + "_" + r["type"], "image": save(r["image"], folder, f"{len(rows):05d}.png"), "question": r["question"], "options": r["choices"],
                     "answer": "ABCDEF".index(r["answer"][1]), "category": r["task"], "source_prompt": r["prompt"], "meta": {"type": r["type"], "source": r["source"]}})
    return rows


def embspatial(folder: str) -> list[dict]:
    rows = []
    for r in load_dataset("FlagEval/EmbSpatial-Bench", split="test").remove_columns("image_base64"):
        rows.append({"id": r["question_id"], "image": save(r["image"], folder, r["question_id"] + ".jpg"), "question": r["question"], "options": r["answer_options"],
                     "answer": r["answer"], "category": r["relation"], "meta": {"data_source": r["data_source"], "scene_id": r["scene_id"]}})
    return rows


def vqa_v2(folder: str) -> list[dict]:
    def fetch(kind):
        file = os.path.join(OUT, "_downloads", f"v2_{kind}_Val_mscoco.zip")
        if not os.path.exists(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)
            urllib.request.urlretrieve(VQA.format(kind), file)
        with zipfile.ZipFile(file) as z:
            return json.load(io.TextIOWrapper(z.open(z.namelist()[0])))

    questions = {q["question_id"]: q for q in fetch("Questions")["questions"]}
    annotations = sorted(fetch("Annotations")["annotations"], key=lambda a: a["question_id"])
    rows = []
    for a in random.Random(0).sample(annotations, 2000):
        rows.append({"id": str(a["question_id"]), "image": f"images/{a['image_id']:012d}.jpg", "question": questions[a["question_id"]]["question"], "options": None,
                     "answer": [x["answer"] for x in a["answers"]], "category": a["answer_type"],
                     "meta": {"question_type": a["question_type"], "multiple_choice_answer": a["multiple_choice_answer"], "image_id": a["image_id"]}})

    def image(image_id):
        file = os.path.join(folder, "images", f"{image_id:012d}.jpg")
        for _ in range(5):
            if os.path.exists(file):
                return
            try:
                urllib.request.urlretrieve(COCO.format(image_id), file + ".part")
                os.rename(file + ".part", file)
            except OSError:
                pass

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(image, sorted({r["meta"]["image_id"] for r in rows})))
    return rows


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--only", default="mmlu_redux,cv_bench,embspatial,vqa_v2")
    for name in args.parse_args().only.split(","):
        folder = os.path.join(OUT, name)
        if os.path.exists(os.path.join(folder, "questions.jsonl")):
            continue
        os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        rows = globals()[name](folder)
        with open(os.path.join(folder, "questions.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(name, len(rows), "questions")


if __name__ == "__main__":
    main()
