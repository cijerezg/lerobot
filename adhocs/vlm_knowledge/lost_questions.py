"""What kind of questions do the fine-tuned models lose? Reads the answer files of ``run_benchmarks.py``.

    uv run python lerobot/adhocs/vlm_knowledge/lost_questions.py

Writes ``lost_questions.md`` next to the answers folder.
  MMLU-Redux   share of the questions base gets right that tube700 / internal get wrong, by question length, by
               whether Molmo2-ER gets it right, and by the blind type labels in ``question_types/`` (557 questions
               lost by both + 557 of the 2,646 kept by both, labelled by agents with ``question_types/RUBRIC.md``)
  images       share of the questions base gets right that are lost when base's language model reads our vision
               encoders, by source, question type and the number to count; sim scenes per object
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np

OURS = ("tube700", "internal")


def table(header: list, rows: list) -> list[str]:
    return ["", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows] + [""]


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--answers", default="outputs/analysis/vlm_knowledge_2026-11-17/answers")
    args.add_argument("--data", default="outputs/data/vlm_benchmarks")
    args = args.parse_args()
    out = os.path.dirname(args.answers.rstrip("/"))

    def questions(benchmark):
        with open(os.path.join(args.data, benchmark, "questions.jsonl")) as f:
            return {q["id"]: q for q in map(json.loads, f)}

    def right(lm, vision, benchmark, ids):
        with open(os.path.join(args.answers, f"{lm}__vision-{vision}", benchmark + ".jsonl")) as f:
            rows = {r["id"]: r for r in map(json.loads, f)}
        return np.array([float(rows[i]["masked" if benchmark == "vqa_v2" else "letters"]) >= 0.5 for i in ids])

    def lost_table(name, groups, base, lost, least=30):
        rows = [[g, (m & base).sum()] + [f"{100 * (m & lost[k]).sum() / (m & base).sum():.0f}" for k in OURS] for g, m in groups.items() if (m & base).sum() >= least]
        rows.insert(0, ["all", base.sum()] + [f"{100 * lost[k].sum() / base.sum():.0f}" for k in OURS])
        return table([name, "base right", "lost, tube700 %", "lost, internal %"], rows)

    lines = ["# What kind of questions are lost", "", "Written by `lerobot/adhocs/vlm_knowledge/lost_questions.py`.", ""]

    q = questions("mmlu_redux")
    ids = sorted(q)
    base = right("base", "base", "mmlu_redux", ids)
    lost = {m: base & ~right(m, m, "mmlu_redux", ids) for m in OURS}
    er = right("er", "er", "mmlu_redux", ids)
    words = np.array([len(q[i]["question"].split()) for i in ids])
    groups = {"Molmo2-ER right": er, "Molmo2-ER wrong": ~er}
    groups.update({f"question of {low}-{high - 1} words": (words >= low) & (words < high) for low, high in ((0, 15), (15, 30), (30, 60), (60, 120), (120, 10000))})
    lines += ["## MMLU-Redux", "", "Share of the questions base gets right that the model gets wrong:"] + lost_table("questions", groups, base, lost)

    with open(os.path.join(out, "question_types", "groups.json")) as f:
        sample = json.load(f)
    labels = {}
    for file in glob.glob(os.path.join(out, "question_types", "labels_*.jsonl")):
        with open(file) as f:
            labels.update({r["id"]: r for r in map(json.loads, filter(str.strip, f))})
    both = set(sample["lost"])
    kept_all = int((base & ~lost["tube700"] & ~lost["internal"]).sum())
    weight = kept_all / len(sample["kept_sample"])  # each labelled kept question stands for this many
    lines += [f"Blind type labels: {len(both)} questions lost by both models and {len(sample['kept_sample'])} of the {kept_all} kept by both. "
              "Loss rate = lost by both / (lost by both + kept by both), the kept count scaled up from the sample."]
    views = {"skill": lambda r: r["skill"], "steps": lambda r: f"{r['steps']} step" + ("s or more" if r["steps"] == 2 else ""), "skill, steps": lambda r: f"{r['skill']}, {r['steps']}",
             "relation to a robot arm's world (2 = mechanics, geometry, control, circuits, programs; 1 = other physical world; 0 = none)": lambda r: str(r["physical"]),
             "relation, calculation": lambda r: f"{r['physical']}, " + ("calculate" if r["skill"] == "calculate" else "no calculation")}
    for name, view in views.items():
        count = Counter((view(r), i in both) for i, r in labels.items())
        rows = [[g, count[(g, True)], count[(g, False)], f"{100 * count[(g, True)] / (count[(g, True)] + weight * count[(g, False)]):.0f}"] for g in sorted({g for g, _ in count})]
        rows.append(["all", len(both), len(sample["kept_sample"]), f"{100 * len(both) / (len(both) + kept_all):.0f}"])
        lines += table([name, "lost by both", "kept (sample)", "loss rate %"], rows)

    lines += ["## Images: lost to our vision encoders (base language model, right with base vision, wrong with ours)", ""]
    for benchmark, views in (("cv_bench", {"task, source": lambda r: f"{r['category']}, {r['meta']['source']}", "number of options": lambda r: str(len(r["options"]))}),
                             ("embspatial", {"relation": lambda r: r["category"], "source": lambda r: r["meta"]["data_source"]}),
                             ("vqa_v2", {"question type": lambda r: r["meta"]["question_type"]})):
        q = questions(benchmark)
        ids = sorted(q)
        base = right("base", "base", benchmark, ids)
        lost = {m: base & ~right("base", m, benchmark, ids) for m in OURS}
        lines += [f"### {benchmark}"]
        for name, view in views.items():
            value = np.array([view(q[i]) for i in ids])
            lines += lost_table(name, {g: value == g for g in sorted(set(value))}, base, lost)

    cells = (("base", "base"), ("base", "tube700"), ("base", "internal"), ("tube700", "base"), ("internal", "base"), ("tube700", "tube700"), ("internal", "internal"))
    lines += ["## Counting: accuracy by the right number", ""]
    for benchmark, number in (("cv_bench", lambda r: int(r["options"][r["answer"]]) if r["category"] == "Count" else None),
                              ("vqa_v2", lambda r: int(r["meta"]["multiple_choice_answer"]) if r["category"] == "number" and r["meta"]["multiple_choice_answer"].isdigit() else None)):
        q = questions(benchmark)
        ids = [i for i in sorted(q) if number(q[i]) is not None]
        value = np.array([number(q[i]) for i in ids])
        scores = {cell: right(*cell, benchmark, ids) for cell in cells}
        rows = [[f"{low}" if high == low + 1 else f"{low} or more" if high > 99 else f"{low}-{high - 1}", ((value >= low) & (value < high)).sum()]
                + [f"{100 * scores[cell][(value >= low) & (value < high)].mean():.0f}" for cell in cells] for low, high in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10000)) if ((value >= low) & (value < high)).sum() >= 10]
        lines += [f"{benchmark}:"] + table(["right number", "questions"] + ["/".join(cell) for cell in cells], rows)

    q = questions("sim_scenes")
    ids = sorted(q)
    cells = (("er", "er"), *cells, ("er", "tube700"), ("er", "internal"))
    scores = {cell: right(*cell, "sim_scenes", ids) for cell in cells}
    group = [f"{q[i]['meta']['object']}, {q[i]['category']}" for i in ids]
    rows = [[g, group.count(g), {True: "yes", False: "no", None: ""}[q[ids[group.index(g)]]["meta"].get("in_training_data")]] + [int(scores[cell][np.array(group) == g].sum()) for cell in cells] for g in sorted(set(group), key=lambda g: g.split(", ")[::-1])]
    lines += ["## Sim scenes: questions right per object", ""] + table(["object", "questions", "in our training data"] + ["/".join(cell) for cell in cells], rows)

    with open(os.path.join(out, "lost_questions.md"), "w") as f:
        f.write(re.sub(r"\n{3,}", "\n\n", "\n".join(lines)) + "\n")
    print("wrote", os.path.join(out, "lost_questions.md"))


if __name__ == "__main__":
    main()
