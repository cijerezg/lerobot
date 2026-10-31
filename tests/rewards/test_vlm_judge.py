import json

import numpy as np
import pytest

from lerobot.rewards.vlm_judge import (
    Frame,
    JudgeConfig,
    JudgeParseError,
    JudgeRunner,
    TaskSpec,
    build_messages,
    parse_response,
)
from lerobot.rewards.vlm_judge.compare_to_scorer import compare
from lerobot.rewards.vlm_judge.prompt import build_request, describe_request


def test_every_spec_loads():
    names = TaskSpec.names()
    assert {"bottle_basket", "basket", "fold_kit", "close_lid", "nail_clippers"} <= set(names)
    for name in names:
        if name == "basket":
            for obj in ["pill bottle", "cup", "tape roll", "bottle"]:
                spec = TaskSpec.load("basket", object=obj)
                assert "{object}" not in json.dumps([s.__dict__ for s in spec.subtasks]) and obj in spec.task
            assert TaskSpec.load("basket:tape roll").subtasks[1].contact == "side-pinch"
            with pytest.raises(ValueError):
                TaskSpec.load("basket", object="piano")
        else:
            TaskSpec.load(name)


def test_bottle_basket_matches_sim_scorer_ladder():
    from lerobot.robots.rebot_b601_sim.easy_tasks import BottleBasketScorer

    spec = TaskSpec.load("bottle_basket")
    assert [(s.name, s.value, s.cursor) for s in spec.stages[1:]] == BottleBasketScorer.STAGES


def _frames(n=4, cams=("top", "wrist")):
    rng = np.random.default_rng(0)
    return [Frame(t=0.5 * i, camera=c, image=rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)) for i in range(n) for c in cams]


def test_prompt_builder():
    spec = TaskSpec.load("bottle_basket")
    frames = _frames()
    messages = build_messages(spec, frames, max_side=320)
    assert messages[0]["role"] == "system"
    for s in spec.subtasks:
        assert f"{s.id}: {s.text}" in messages[0]["content"]
    for s in spec.stages:
        assert f"{s.id}: {s.name}" in messages[0]["content"]
    images = [p for p in messages[1]["content"] if p["type"] == "image_url"]
    assert len(images) == 8 and all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in images)
    labels = [p["text"] for p in messages[1]["content"] if p["type"] == "text"]
    assert labels[1] == "top camera, t=0.0 s:" and labels[-2] == "wrist camera, t=1.5 s:"
    req = build_request(spec, frames, "m", 300, 0.0)
    assert req["response_format"] == {"type": "json_object"} and "chars>" in describe_request(req)


def test_parser_accepts_canned_and_fenced():
    spec = TaskSpec.load("bottle_basket")
    j = parse_response('{"subtask_id": 3, "stage_id": 3.0, "progress_0_1": 0.6, "confidence": 0.9, "evidence": "bottle above the rim"}', spec)
    assert (j.subtask, j.stage, j.cursor, j.stage_value) == ("carry the bottle to the basket", "lifted", 1, 0.5)
    fenced = '```json\n{"subtask_id": 6, "stage_id": 0, "progress_0_1": 0.0, "confidence": 0.5, "evidence": "arm idle"}\n```'
    assert parse_response(fenced, spec).cursor == -1
    chatty = 'Sure. {"subtask_id": 0, "stage_id": 1, "progress_0_1": 0.15, "confidence": 0.7, "evidence": "x"} done'
    assert parse_response(chatty, spec).stage == "reach"


@pytest.mark.parametrize("bad", [
    '{"subtask_id": 9, "stage_id": 0, "progress_0_1": 0.0, "confidence": 0.5, "evidence": "x"}',
    '{"subtask_id": 1, "stage_id": 0, "progress_0_1": 1.5, "confidence": 0.5, "evidence": "x"}',
    '{"subtask_id": 1, "stage_id": 0, "progress_0_1": 0.5, "confidence": 0.5}',
    '{"subtask_id": 1.5, "stage_id": 0, "progress_0_1": 0.5, "confidence": 0.5, "evidence": "x"}',
    '{"subtask_id": true, "stage_id": 0, "progress_0_1": 0.5, "confidence": 0.5, "evidence": "x"}',
    "no json here",
])
def test_parser_rejects(bad):
    with pytest.raises(JudgeParseError):
        parse_response(bad, TaskSpec.load("bottle_basket"))


def test_compare_to_scorer_perfect_and_lagged():
    spec = TaskSpec.load("bottle_basket")
    events = [dict(t=2.0, cursor=0, stage="reach", reward=0.1, why="reached reach"),
              dict(t=4.0, cursor=0, stage="held", reward=0.3, why="reached held"),
              dict(t=4.0, cursor=1, stage="held", reward=0.3, why="stage held"),
              dict(t=6.0, cursor=1, stage="lifted", reward=0.5, why="reached lifted")]
    record = dict(episode=0, success=False, reward=0.5, stage="lifted", events=events)

    def row(t, sub, stage, p, pmax):
        s, st = spec.subtasks[sub], spec.stages[stage]
        return dict(t=t, subtask_id=sub, stage_id=stage, progress=p, progress_max=pmax, cursor=s.cursor, stage_value=st.value)

    perfect = dict(steps=[row(1.0, 0, 0, 0.0, 0.0), row(3.0, 1, 1, 0.1, 0.1), row(5.0, 2, 2, 0.3, 0.3), row(7.0, 3, 3, 0.5, 0.5)])
    m = compare(perfect, record, spec)
    assert m["cursor_accuracy"] == 1.0 and m["stage_accuracy"] == 1.0 and m["spearman_progress"] == 1.0
    assert m["stage_lag_s"]["held"]["lag_s"] == 1.0 and m["stage_lag_s"]["lifted"]["lag_s"] == 1.0
    lagged = dict(steps=[row(1.0, 0, 0, 0.0, 0.0), row(3.0, 0, 0, 0.0, 0.0), row(5.0, 1, 1, 0.1, 0.1), row(7.0, 6, 2, 0.3, 0.3)])
    m = compare(lagged, record, spec)
    assert m["cursor_accuracy"] == 0.5 and m["off_task_steps"] == 1 and m["stage_lag_s"]["reach"]["lag_s"] == 3.0
    assert m["stage_lag_s"]["lifted"]["judge_t"] is None and m["confusion"][1][-1] == 1


class _FakeClient:
    def __init__(self):
        self.calls = []

    def query(self, spec, frames):
        self.calls.append([(f.t, f.camera) for f in frames])
        j = parse_response('{"subtask_id": 2, "stage_id": 2, "progress_0_1": 0.35, "confidence": 0.8, "evidence": "held"}', spec)
        return j, dict(latency_s=0.01, raw="", usage=None)


def test_runner_buffers_and_queries():
    cfg = JudgeConfig(frames_per_camera=3, frame_period_s=0.5, every_k_steps=15)
    client = _FakeClient()
    runner = JudgeRunner(cfg, TaskSpec.load("bottle_basket"), client)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for step in range(91):  # 3 s at 30 Hz
        runner.observe(step, step / 30, {"top": img, "wrist": img})
        runner.close()
    assert runner.latest is not None and runner.latest.stage == "held" and runner.progress_max == 0.35
    assert len(client.calls) == 7 and len(client.calls[-1]) == 6  # one query per 15 steps, last strip 3 frames x 2 cameras
    assert [t for t, c in client.calls[-1] if c == "top"] == [2.0, 2.5, 3.0]
    runner.reset()
    assert runner.latest is None and all(len(b) == 0 for b in runner.buffers.values())


def test_assigner_offers_only_the_current_segment():
    from lerobot.rewards.vlm_judge.assigner import offered_steps, segments

    basket = ["grasp the cup", "move the cup to the basket", "release the cup in the basket", "return to home"]
    assert segments(basket) == [(0, 4)]
    assert all(offered_steps(basket, c) == [0, 1, 2, 3] for c in range(4))
    seq = ["grasp the tape roll", "move the tape roll to the basket", "release the tape roll in the basket",
           "grasp the cup", "move the cup to the basket", "release the cup in the basket", "return to home"]
    assert offered_steps(seq, 1) == [0, 1, 2, 3]  # this object + the next grasp
    assert offered_steps(seq, 3) == [3, 4, 5, 6]
    hw = ["grasp the lid", "close the lid"] + [f"{v} the bit {i}" for i in range(6) for v in ("grasp", "move", "release")]
    assert max(len(offered_steps(hw, c)) for c in range(len(hw))) == 4


def test_assigner_parse_maps_offered_ids_to_script():
    from lerobot.rewards.vlm_judge.assigner import parse_assignment

    script = ["grasp the cup", "move the cup to the basket", "release the cup in the basket", "grasp the tape roll", "move the tape roll to the basket"]
    a = parse_assignment('{"step": 1, "confidence": 0.9}', script, [3, 4])
    assert (a.cursor, a.step, a.confidence) == (4, "move the tape roll to the basket", 0.9)
    for bad in ['{"step": 2, "confidence": 0.9}', '{"step": 0}', '{"step": 0.5, "confidence": 0.9}', '{"step": 0, "confidence": 2}']:
        with pytest.raises(JudgeParseError):
            parse_assignment(bad, script, [3, 4])


def test_assigner_request_shape():
    from lerobot.rewards.vlm_judge.assigner import build_request
    from lerobot.rewards.vlm_judge.config import ASSIGNER

    cfg = JudgeConfig(**ASSIGNER)
    script = ["grasp the cup", "move the cup to the basket", "release the cup in the basket", "grasp the tape roll", "move the tape roll to the basket"]
    img = np.zeros((48, 64, 3), np.uint8)
    frames = [Frame(t=0.5 * i, camera=c, image=img) for i in range(3) for c in ("top", "wrist")]
    req = build_request(cfg, "Tidy up", script, [3, 4], frames, with_evidence=False)
    system = req["messages"][0]["content"]
    assert "grasp the tape roll" in system and "grasp the cup" not in system.split("Steps")[1] and "release the cup in the basket" in system
    assert sum(p["type"] == "image_url" for p in req["messages"][1]["content"]) == 6
    assert req["chat_template_kwargs"] == {"enable_thinking": False} and req["model"] == "Qwen/Qwen3.5-9B"


def test_sim_specs_follow_tasks_py():
    from lerobot.robots.rebot_b601_sim import tasks as sim_tasks

    for layout in sim_tasks.TWIN_LAYOUTS + sim_tasks.LAYOUTS:
        spec = TaskSpec.load(f"sim:{layout}_p2")
        task = sim_tasks.TASKS[layout]
        assert list(spec.script) == list(task.eval_subtasks)
        assert [(s.name, s.value, s.cursor) for s in spec.stages[1:]] == [tuple(x) for x in task.scorer.STAGES]
