"""Tables from the answer files of ``run_benchmarks.py``.

    uv run python lerobot/adhocs/vlm_knowledge/summary.py [--answers=outputs/analysis/vlm_knowledge_2026-11-17/answers]

Writes ``summary.md`` next to the answers folder. Per benchmark: accuracy of every (language model, vision
encoder) cell under the three readings, the share of free answers that are action tokens, the probability on
action tokens at the first position. Intervals are 95% bootstrap over questions. Differences are paired (the
same questions in both cells) and given against ``--reference`` cells.
"""
import argparse
import glob
import json
import os

import numpy as np

READINGS = ("free", "masked", "letters")


def interval(values: np.ndarray, rng) -> str:
    draws = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return f"{100 * values.mean():.1f} ({100 * low:.1f}, {100 * high:.1f})"


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--answers", default="outputs/analysis/vlm_knowledge_2026-11-17/answers")
    args.add_argument("--reference", default="er__vision-er,base__vision-base")
    args.add_argument("--reading", default="masked", help="reading used for the paired differences and the categories")
    args = args.parse_args()
    rng = np.random.default_rng(0)
    cells = {}  # benchmark -> cell -> {id: row}
    for file in sorted(glob.glob(os.path.join(args.answers, "*", "*.jsonl"))):
        with open(file) as f:
            rows = {row["id"]: row for row in map(json.loads, f)}
        cells.setdefault(os.path.basename(file)[:-6], {})[os.path.basename(os.path.dirname(file))] = rows
    lines = []
    for benchmark, by_cell in cells.items():
        ids = sorted(set.intersection(*[set(rows) for rows in by_cell.values()]))
        column = lambda cell, key: np.array([float(by_cell[cell][i].get(key, np.nan)) for i in ids])  # noqa: E731
        lines += [f"## {benchmark} ({len(ids)} questions)", "", "| language model | vision encoder | free | masked | letters | free answer is action tokens | probability on action tokens |", "|---|---|---|---|---|---|---|"]
        for cell in by_cell:
            lm, vision = cell.split("__vision-")
            scores = [interval(column(cell, r), rng) if r in by_cell[cell][ids[0]] else "" for r in READINGS]
            lines.append(f"| {lm} | {vision} | " + " | ".join(scores) + f" | {100 * column(cell, 'free_in_action_tokens').mean():.1f} | {column(cell, 'action_mass').mean():.3f} |")
        for reference in [r for r in args.reference.split(",") if r in by_cell]:
            lines += ["", f"Paired difference to {reference}, reading `{args.reading}` (points):", "", "| cell | difference | right only here | right only in the reference |", "|---|---|---|---|"]
            base = column(reference, args.reading)
            for cell in by_cell:
                if cell != reference:
                    own = column(cell, args.reading)
                    draws = (own - base)[rng.integers(0, len(ids), (2000, len(ids)))].mean(axis=1)
                    low, high = np.percentile(draws, [2.5, 97.5])
                    lines.append(f"| {cell} | {100 * (own - base).mean():+.1f} ({100 * low:+.1f}, {100 * high:+.1f}) | {int((own > base).sum())} | {int((own < base).sum())} |")
        categories = sorted({by_cell[next(iter(by_cell))][i]["category"] for i in ids})
        if 1 < len(categories) <= 12:
            lines += ["", f"By category, reading `{args.reading}`:", "", "| cell | " + " | ".join(categories) + " |", "|---|" + "---|" * len(categories)]
            first = next(iter(by_cell))
            groups = {c: np.array([by_cell[first][i]["category"] == c for i in ids]) for c in categories}
            for cell in by_cell:
                lines.append(f"| {cell} | " + " | ".join(f"{100 * column(cell, args.reading)[groups[c]].mean():.1f}" for c in categories) + " |")
        lines.append("")
    out = os.path.join(os.path.dirname(args.answers.rstrip("/")), "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
