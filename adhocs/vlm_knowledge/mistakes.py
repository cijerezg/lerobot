"""Are the mistakes the same? Per-question comparison of the answer files of ``run_benchmarks.py``.

    uv run python lerobot/adhocs/vlm_knowledge/mistakes.py [--answers=outputs/analysis/vlm_knowledge_2026-11-17/answers]

Writes ``mistakes.md`` next to the answers folder. A multiple-choice question counts as right under the ``letters``
reading (best option letter at the first position), a short-answer question under ``masked``. Per benchmark:
accuracy of every (language model, vision encoder) cell, which of the four models are wrong on the same questions,
wrong-question overlap per pair against the overlap of unrelated mistakes (product of the two error rates), where
the questions lost against base come back (base vision, base language model), overlap of the questions lost to
each fine-tuned vision encoder, accuracy per category and per position of the right option, and example questions.
"""
import argparse
import glob
import itertools
import json
import os
import random
from collections import Counter

import numpy as np

MODELS = ("er", "base", "tube700", "internal")
OURS = ("tube700", "internal")
LETTERS = "ABCDEF"


def table(header: list, rows: list) -> list[str]:
    return ["", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows] + [""]


def said(row: dict) -> str:
    return row["masked_answer"].strip().lower().rstrip(".")


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--answers", default="outputs/analysis/vlm_knowledge_2026-11-17/answers")
    args.add_argument("--data", default="outputs/data/vlm_benchmarks")
    args.add_argument("--examples", type=int, default=4)
    args = args.parse_args()
    out = os.path.dirname(args.answers.rstrip("/"))
    lines = ["# Mistakes per question", "", "Written by `lerobot/adhocs/vlm_knowledge/mistakes.py`. Cells are named language model / vision encoder.", ""]
    for benchmark in ("mmlu_redux", "cv_bench", "embspatial", "vqa_v2", "sim_scenes"):
        with open(os.path.join(args.data, benchmark, "questions.jsonl")) as f:
            questions = {q["id"]: q for q in map(json.loads, f)}
        ids = sorted(questions)
        choice = questions[ids[0]]["options"] is not None
        reading = "letters" if choice else "masked"
        cells = {}
        for file in glob.glob(os.path.join(args.answers, "*", benchmark + ".jsonl")):
            with open(file) as f:
                cells[tuple(os.path.basename(os.path.dirname(file)).split("__vision-"))] = {r["id"]: r for r in map(json.loads, f)}
        score = {cell: np.array([float(rows[i][reading]) for i in ids]) for cell, rows in cells.items()}
        right = {cell: s >= 0.5 for cell, s in score.items()}
        own = {m: right[(m, m)] for m in MODELS}
        n = len(ids)
        lines += [f"## {benchmark} ({n} questions, reading `{reading}`)", ""]

        if len(cells) > len(MODELS):
            lines += ["Accuracy, language model (rows) x vision encoder (columns):"]
            lines += table(["language model", *MODELS], [[lm] + [f"{100 * score[(lm, v)].mean():.1f}" for v in MODELS] for lm in MODELS])

        lines += ["Which of the four models (own vision encoder) are wrong on a question:"]
        patterns = Counter(tuple(not own[m][k] for m in MODELS) for k in range(n))
        lines += table([*MODELS, "questions", "share"], [["wrong" if w else "" for w in p] + [c, f"{100 * c / n:.1f}"] for p, c in patterns.most_common()])

        lines += ["Wrong-question overlap per pair (own vision encoder). \"Unrelated\" = both wrong if the mistakes were unrelated (product of the two error rates)."
                  + (" Same wrong option: among questions both answer with a wrong letter; chance = one over the number of wrong options." if choice else " Same wrong answer: among questions both get wrong.")]
        rows = []
        for a, b in itertools.combinations(MODELS, 2):
            wa, wb = ~own[a], ~own[b]
            both = [i for k, i in enumerate(ids) if wa[k] and wb[k]]
            ra, rb = cells[(a, a)], cells[(b, b)]
            if choice:
                wrong = lambda rows, i: len(said(rows[i])) == 1 and LETTERS.find(said(rows[i]).upper()) not in (-1, questions[i]["answer"])  # noqa: E731
                both = [i for i in both if wrong(ra, i) and wrong(rb, i)]
                chance = f" (chance {100 * np.mean([1 / (len(questions[i]['options']) - 1) for i in both]):.0f})" if both else ""
            else:
                chance = ""
            same = np.mean([said(ra[i]) == said(rb[i]) for i in both]) if both else np.nan
            rows.append([a, b, wa.sum(), wb.sum(), (wa & wb).sum(), f"{wa.mean() * wb.mean() * n:.0f}", f"{100 * (wa & wb).sum() / wa.sum():.0f}", f"{100 * (wa & wb).sum() / wb.sum():.0f}", f"{100 * same:.0f}{chance}" if both else "-"])
        lines += table(["A", "B", "A wrong", "B wrong", "both wrong", "unrelated", "% of A's wrong also wrong in B", "% of B's wrong also wrong in A", "same wrong " + ("option %" if choice else "answer %")], rows)

        base = own["base"]
        lost = {m: base & ~own[m] for m in OURS}
        both = lost["tube700"] & lost["internal"]
        lines += [f"Questions base gets right: {base.sum()}. Lost by tube700: {lost['tube700'].sum()}, by internal: {lost['internal'].sum()}, by both: {both.sum()} "
                  f"(unrelated: {lost['tube700'].sum() * lost['internal'].sum() / base.sum():.0f}). Gained (base wrong, model right): tube700 {(own['tube700'] & ~base).sum()}, internal {(own['internal'] & ~base).sum()}.", ""]

        if len(cells) > len(MODELS):
            lines += ["Where the questions lost against base come back (swap one part of the model for base's):"]
            rows = []
            for m in OURS:
                vision, lm, k = right[(m, "base")], right[("base", m)], lost[m]
                rows.append([m, k.sum(), (k & vision & ~lm).sum(), (k & lm & ~vision).sum(), (k & vision & lm).sum(), (k & ~vision & ~lm).sum()])
            lines += table(["model", "lost", "right again only with base vision", "only with base language model", "with either", "with neither"], rows)

            lines += ["Questions lost to a fine-tuned vision encoder (right with base vision, wrong with ours, same language model):"]
            rows = []
            for lm in ("er", "base"):
                ok = right[(lm, "base")]
                a, b = ok & ~right[(lm, "tube700")], ok & ~right[(lm, "internal")]
                rows.append([lm, ok.sum(), a.sum(), b.sum(), (a & b).sum(), f"{a.sum() * b.sum() / ok.sum():.0f}"])
            lines += table(["language model", "right with base vision", "lost with tube700 vision", "lost with internal vision", "lost with both", "unrelated"], rows)

        shown = [(m, m) for m in MODELS] if len(cells) == len(MODELS) else [("er", "er"), ("base", "base"), ("base", "tube700"), ("base", "internal"), ("tube700", "base"), ("tube700", "tube700"), ("internal", "base"), ("internal", "internal")]
        categories = sorted({questions[i]["category"] for i in ids})
        group = {c: np.array([questions[i]["category"] == c for i in ids]) for c in categories}
        lines += ["Accuracy per category:"]
        if len(categories) > 12:  # one row per category, sorted by base minus internal
            rows = [[c, group[c].sum()] + [round(100 * own[m][group[c]].mean()) for m in MODELS] for c in categories]
            lines += table(["category", "questions", *MODELS], sorted(rows, key=lambda r: r[3] - r[5]))
        else:
            lines += table(["cell"] + [f"{c} ({group[c].sum()})" for c in categories], [["/".join(cell)] + [f"{100 * score[cell][group[c]].mean():.0f}" for c in categories] for cell in shown])

        if benchmark == "mmlu_redux":
            kinds = sorted({questions[i]["meta"]["error_type"] for i in ids})
            lines += ["Accuracy per MMLU-Redux annotation of the question (`ok` = no error found in the question):"]
            lines += table(["annotation", "questions", *MODELS], [[t, sum(questions[i]["meta"]["error_type"] == t for i in ids)] + [f"{100 * np.mean([own[m][k] for k, i in enumerate(ids) if questions[i]['meta']['error_type'] == t]):.1f}" for m in MODELS] for t in kinds])
            lines += ["Answers that do not start with an option letter under the `masked` reading (the model starts a sentence; counted wrong there), and accuracy under the three readings:"]
            rows = []
            for m in MODELS:
                r = cells[(m, m)]
                rows.append([m, sum(len(said(r[i])) != 1 for i in ids)] + [f"{100 * np.mean([r[i][key] for i in ids]):.1f}" for key in ("free", "masked", "letters")])
            lines += table(["model", "answers that are not a letter", "free", "masked", "letters"], rows)

        if choice:
            positions = sorted({questions[i]["answer"] for i in ids})
            at = {p: np.array([questions[i]["answer"] == p for i in ids]) for p in positions}
            lines += ["Accuracy by the position of the right option, and share of `masked` answers per letter (the right answers' shares in the first row):"]
            rows = [["right answers"] + [""] * len(positions) + [f"{100 * at[p].mean():.0f}" for p in positions]]
            for cell in shown:
                letters = Counter(said(cells[cell][i]).upper() for i in ids)
                rows.append(["/".join(cell)] + [f"{100 * score[cell][at[p]].mean():.0f}" for p in positions] + [f"{100 * letters[LETTERS[p]] / n:.0f}" for p in positions])
            lines += table(["cell"] + [f"right is {LETTERS[p]}" for p in positions] + [f"answers {LETTERS[p]} %" for p in positions], rows)
        else:
            lines += ["Most frequent answers (count):"]
            rows = [["most common human answer", ", ".join(f"{a} {c}" for a, c in Counter(questions[i]["meta"]["multiple_choice_answer"] for i in ids).most_common(8))]]
            rows += [["/".join(cell), ", ".join(f"{a} {c}" for a, c in Counter(said(cells[cell][i]) for i in ids).most_common(8))] for cell in shown]
            lines += table(["cell", "answers"], rows)

        picks = {"right for er and base, wrong for tube700 and internal": own["er"] & own["base"] & ~own["tube700"] & ~own["internal"],
                 "right for er, base and tube700, wrong for internal": own["er"] & own["base"] & own["tube700"] & ~own["internal"],
                 "wrong for base, right for tube700 and internal": ~own["base"] & own["tube700"] & own["internal"]}
        if len(cells) > len(MODELS):
            picks["lost by internal, wrong again with base vision and with base language model"] = lost["internal"] & ~right[("internal", "base")] & ~right[("base", "internal")]
        rng = random.Random(0)
        for title, mask in picks.items():
            chosen = [i for k, i in enumerate(ids) if mask[k]]
            lines += [f"### Examples: {title} ({len(chosen)} questions)", ""]
            for i in rng.sample(chosen, min(args.examples, len(chosen))):
                q = questions[i]
                if q["image"] is not None:
                    lines += [f'<img src="{os.path.relpath(os.path.join(args.data, benchmark, q["image"]), out)}" width="320">', ""]
                lines += [f"**{i}** ({q['category']}): {q['question']}", ""]
                if choice:
                    lines += [f"- {LETTERS[k]}. {o}" + (" **(right)**" if k == q["answer"] else "") for k, o in enumerate(q["options"])]
                else:
                    lines += ["- human answers: " + ", ".join(f"{a} x{c}" for a, c in Counter(q["answer"]).most_common())]
                lines += ["", "Answers (`masked`): " + "; ".join("/".join(cell) + f" `{cells[cell][i]['masked_answer'].strip()}`" for cell in shown), ""]
    with open(os.path.join(out, "mistakes.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(out, "mistakes.md"))


if __name__ == "__main__":
    main()
