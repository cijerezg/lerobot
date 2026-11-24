"""Yes/no "is there a ... in this image?" questions on the object-choice sim scenes, in the format run_benchmarks.py reads.

    uv run python lerobot/adhocs/vlm_knowledge/prepare_sim_scenes.py

Writes outputs/data/vlm_benchmarks/sim_scenes/questions.jsonl and images/ (copies of the top-camera start renders, one per
placement). Per image: one question per thing in the scene (the two objects, the basket, the robot arm) and as many about
things that are not: the choice objects of the other pairs, then distractors drawn with a fixed seed.
"""
import json
import os
import random
import shutil

from lerobot.robots.rebot_b601_sim.tasks import _CHOICE_PAIRS, _TWIN_BASKET, SCENES_DIR

RENDERS = "lerobot/scripts/sim_builders/object_choice/renders"
OUT = "outputs/data/vlm_benchmarks/sim_scenes"
TRAINED = {"pill_bottle": True, "roar_bottle": True, "basket": True, "banana": False, "mug": False}  # in the robot training data
EXTRA = {"basket": "the basket", "robot_arm": "the robot arm"}  # in every scene; the choice objects' nouns come from _TWIN_BASKET
DISTRACTORS = ["apple", "spoon", "book", "scissors", "toy car", "sponge", "fork", "plate"]
SEED = 0


def question(noun: str) -> str:
    noun = noun.removeprefix("the ")
    return f"Is there {'an' if noun[0] in 'aeiou' else 'a'} {noun} in this image?"


def main() -> None:
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    rng = random.Random(SEED)
    choice_bodies = sorted({body for bodies in _CHOICE_PAIRS.values() for body in bodies})
    rows = []
    for pair, bodies in _CHOICE_PAIRS.items():
        for k in range(1, 7):
            name = f"choice_{pair}_p{k}"
            shutil.copyfile(os.path.join(RENDERS, f"{name}_ext.png"), os.path.join(OUT, "images", f"{name}.png"))
            truth = json.loads((SCENES_DIR / f"choice_{pair}_ask_pill_p{k}" / "objects.json").read_text())
            assert set(truth["sides"]) == set(bodies)
            present = [(body, _TWIN_BASKET[body][0]) for body in bodies] + list(EXTRA.items())
            absent = [(body, _TWIN_BASKET[body][0]) for body in choice_bodies if body not in bodies]
            absent += [(noun.replace(" ", "_"), noun) for noun in rng.sample(DISTRACTORS, len(present) - len(absent))]
            for category, things in (("present", present), ("absent", absent)):
                for body, noun in things:
                    meta = {"scene": name, "pair": pair, "placement": k, "note": truth["note"], "object": body, "noun": noun.removeprefix("the "),
                            "kind": "choice" if body in choice_bodies else "distractor" if category == "absent" else "fixture",
                            "in_training_data": TRAINED.get(body)}  # None: robot arm and distractors
                    if body in truth["sides"]:
                        meta |= {"side_image": truth["sides"][body]["side_image"], "image_uv": truth["sides"][body]["image_uv"]}
                    rows.append({"id": f"{name}__{body}", "image": f"images/{name}.png", "question": question(noun), "options": ["Yes", "No"],
                                 "answer": 0 if category == "present" else 1, "category": category, "meta": meta})
    with open(os.path.join(OUT, "questions.jsonl"), "x") as f:  # an existing file stops the script
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions on {len(rows) // 8} images to {OUT}")


if __name__ == "__main__":
    main()
