"""Sim eval registry, staged reward monotonicity and the scorers' predicates on a synthetic MJCF."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from lerobot.robots.rebot_b601_sim.easy_tasks import LID_REACH_M, BottleBasketScorer, CloseLidScorer, LiftBottleScorer, body_surface_distance  # noqa: E402
from lerobot.robots.rebot_b601_sim.nail_task import GRIPPER_CLOSED_DEG, SLOTS, NailTaskScorer  # noqa: E402
from lerobot.robots.rebot_b601_sim.scorer import SUCCESS, TaskScorer  # noqa: E402
from lerobot.robots.rebot_b601_sim import tasks  # noqa: E402
from lerobot.robots.rebot_b601_sim.tasks import LAYOUTS, TASKS, TIERS, task_for  # noqa: E402

# Gripper: mocap gripper_end with two finger boxes 40 mm apart (bottle diameter 55 mm: both touch at +-27.5 mm).
GRIPPER = """
<body name="gripper_end" mocap="true" pos="1 1 1">
  <body name="gripper_left" pos="0 -0.0275 0"><geom type="box" size="0.01 0.003 0.02" /></body>
  <body name="gripper_right" pos="0 0.0275 0"><geom type="box" size="0.01 0.003 0.02" /></body>
</body>"""
EASY_XML = f"""
<mujoco>
  <compiler angle="radian" />
  <option gravity="0 0 0" />
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.01" />
    {GRIPPER}
    <body name="bottle" pos="0.3 0 0.045"><freejoint /><geom type="cylinder" size="0.0275 0.045" mass="0.1" /></body>
    <body name="basket" pos="0.5 0.2 0"><freejoint />
      <geom type="box" size="0.1 0.1 0.005" pos="0 0 0.005" mass="0.1" />
      <geom type="box" size="0.1 0.005 0.05" pos="0 0.1 0.05" mass="0.1" />
    </body>
    <body name="headphone_case" pos="0.3 -0.3 0.02"><freejoint /><geom type="box" size="0.05 0.05 0.02" mass="0.1" />
      <body name="headphone_lid" pos="-0.05 0 0.02">
        <joint name="lid_hinge" type="hinge" axis="0 1 0" range="0 3.0" />
        <geom type="box" size="0.05 0.05 0.005" pos="-0.05 0 0.005" mass="0.05" />
      </body>
    </body>
  </worldbody>
</mujoco>"""
# Nail: case at the origin, slot floors from SLOTS; clippers are 8 mm thick boxes.
NAIL_XML = f"""
<mujoco>
  <compiler angle="radian" />
  <option gravity="0 0 0" />
  <worldbody>
    {GRIPPER}
    <body name="clipper_case" pos="0.3 0 0"><freejoint /><geom type="box" size="0.044 0.06 0.01" mass="0.1" /></body>
    <body name="clipper_large" pos="0.5 0.2 0.004"><freejoint /><geom type="box" size="0.01 0.035 0.004" mass="0.01" /></body>
    <body name="clipper_small" pos="0.5 -0.2 0.004"><freejoint /><geom type="box" size="0.008 0.025 0.004" mass="0.01" /></body>
  </worldbody>
</mujoco>"""


def load(xml):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def place(m, d, body, pos, quat=(1, 0, 0, 0)):
    adr = m.jnt_qposadr[m.body_jntadr[m.body(body).id]]
    d.qpos[adr : adr + 3] = pos
    d.qpos[adr + 3 : adr + 7] = quat
    mujoco.mj_forward(m, d)


def gripper_to(m, d, pos):
    d.mocap_pos[0] = pos
    mujoco.mj_forward(m, d)


def test_registry():
    assert set(LAYOUTS) <= set(TASKS) and set(LAYOUTS) == {"clippers_only", "bottle_basket", "close_lid", "ep0", "ep3"}
    for name, task in TASKS.items():
        assert task.tier in TIERS
        values = [v for _, v, _ in task.stages]
        assert values == sorted(values) and values[-1] == 1.0 and len(set(values)) == len(values)
        assert all(0 <= c < len(task.eval_subtasks) for _, _, c in task.stages)
        assert len(task.eval_subtasks) == len(task.eval_subtask_precisions) == len(task.eval_subtask_contacts)
        assert task.overrides()[0] == f"--policy.task={task.task}"
    assert TASKS["ep0"].sheet_id == "T085" and TASKS["close_lid"].sheet_id == "T071"
    assert task_for("some/dir/scenes/ep3/scene.xml") is TASKS["ep3"]
    assert {t.tier for n, t in TASKS.items() if n in ("ep0", "ep3")} == {"hard"}
    assert {t.tier for n, t in TASKS.items() if n in ("clippers_only", "bottle_basket", "close_lid", "lift_bottle")} == {"easy"}


def test_placement_variants_resolve_and_expand(tmp_path, monkeypatch, capsys):
    assert tasks.layout_name("ep0_v3") == "ep0" and tasks.layout_name("x/scenes/clippers_only_v12/scene.xml") == "clippers_only"
    assert tasks.layout_name("ep3") == "ep3" and tasks.layout_name("some/dir/scenes/ep3/scene.xml") == "ep3"
    assert task_for("close_lid_v1") is TASKS["close_lid"] and task_for("close_lid_v1").overrides() == TASKS["close_lid"].overrides()
    for name in ("ep0_v2", "ep0_v10", "ep0_v1", "ep0_vx", "ep0_v3_old", "ep0"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "scene.xml").write_text("")
    (tmp_path / "ep0_v4").mkdir()  # no scene.xml: not a variant
    monkeypatch.setattr(tasks, "SCENES_DIR", tmp_path)
    monkeypatch.setattr(tasks, "LAYOUTS", ("ep3", "ep0"))
    assert tasks.variants("ep0") == ["ep0_v1", "ep0_v2", "ep0_v10"] and tasks.variants("ep3") == []
    assert tasks.eval_layouts() == ["ep3", "ep0_v1", "ep0_v2", "ep0_v10"]  # variants replace their base, bases without any stay
    monkeypatch.setattr(tasks.sys, "argv", ["tasks", "layouts"])
    tasks.main()
    assert capsys.readouterr().out.split() == ["ep3", "ep0_v1", "ep0_v2", "ep0_v10"]


class Toggle(TaskScorer):
    STAGES = [("a", 0.2, 0), ("b", 0.5, 1), ("c", 1.0, 1)]
    flags = {}

    def _predicates(self):
        return dict(self.flags)


def test_staged_reward_monotone_and_sequential():
    m, d = load(EASY_XML)
    s = Toggle(m, d, lambda: 0.0)
    s.flags = dict(a=False, b=True, c=False)
    assert s.poll() == 0 and s.reward == 0.0  # b without a: no credit
    s.flags = dict(a=True, b=True, c=False)
    assert s.poll() == 1 and s.reward == 0.5 and s.stage_name == "b"
    s.flags = dict(a=False, b=False, c=False)
    assert s.poll() == 1 and s.reward == 0.5  # never decreases
    s.flags = dict(a=False, b=False, c=True)
    assert s.poll() == 1 and s.reward == 1.0 and SUCCESS.is_set()  # the last stage is reachable directly
    assert [e["why"] for e in s.events] == ["reached a", "reached b", "stage b", "reached c"]
    assert all(set(e) == {"t", "cursor", "stage", "reward", "why"} for e in s.events)
    s.reset()
    assert s.reward == 0.0 and s.stage == -1 and not SUCCESS.is_set() and s.events == []


def test_bottle_basket_predicates():
    m, d = load(EASY_XML)
    s = BottleBasketScorer(m, d, lambda: 0.0)
    assert s.poll() == 0 and s.reward == 0.0
    gripper_to(m, d, (0.3, 0, 0.05))  # fingers straddle the bottle: reach + pinched
    st = s._predicates()
    assert st["reach"] and st["held"] and not st["lifted"]
    assert s.poll() == 1 and s.reward == 0.3
    place(m, d, "bottle", (0.3, 0, 0.09)); gripper_to(m, d, (0.3, 0, 0.095))
    assert s.poll() == 1 and s.reward == 0.5
    place(m, d, "bottle", (0.5, 0.2, 0.12)); gripper_to(m, d, (0.5, 0.2, 0.125))  # over the footprint, 7.5 cm up
    assert s.poll() == 2 and s.reward == 0.8
    gripper_to(m, d, (1, 1, 1)); place(m, d, "bottle", (0.5, 0.2, 0.055))  # released on the basket floor (rim 0.10)
    assert s.poll() == 3 and s.reward == 1.0 and SUCCESS.is_set()
    lo, hi = s.aabb(m.body("basket").id)
    assert np.allclose(lo[:2], (0.4, 0.1)) and np.allclose(hi[:2], (0.6, 0.305)) and hi[2] == pytest.approx(0.1)


def test_bottle_basket_outside_is_not_inside():
    m, d = load(EASY_XML)
    s = BottleBasketScorer(m, d, lambda: 0.0)
    place(m, d, "bottle", (0.5, 0.45, 0.045))  # beside the basket wall
    assert not s._predicates()["inside"]
    place(m, d, "bottle", (0.5, 0.2, 0.2))  # above the rim
    assert not s._predicates()["inside"]


def test_lift_bottle_predicates_and_hold():
    m, d = load(EASY_XML)
    s = LiftBottleScorer(m, d, lambda: 0.0)
    gripper_to(m, d, (0.3, 0, 0.05))
    assert s.poll() == 1 and s.reward == 0.5
    place(m, d, "bottle", (0.3, 0, 0.1)); gripper_to(m, d, (0.3, 0, 0.105))  # 5.5 cm up
    assert s.poll() == 1 and s.reward == 0.8 and not SUCCESS.is_set()
    d.time += 1.0; assert s.poll() == 1 and s.reward == 0.8
    d.time += 1.1; assert s.poll() == 1 and s.reward == 1.0 and SUCCESS.is_set()


def test_lift_bottle_hold_restarts_after_a_drop():
    m, d = load(EASY_XML)
    s = LiftBottleScorer(m, d, lambda: 0.0)
    place(m, d, "bottle", (0.3, 0, 0.1)); gripper_to(m, d, (0.3, 0, 0.105)); s.poll()
    d.time += 1.5; s.poll()
    gripper_to(m, d, (1, 1, 1)); s.poll()  # let go
    gripper_to(m, d, (0.3, 0, 0.105)); d.time += 1.0; s.poll()
    assert s.reward == 0.8
    d.time += 2.0; s.poll()
    assert s.reward == 1.0


def test_close_lid_predicates():
    m, d = load(EASY_XML)
    s = CloseLidScorer(m, d, lambda: 0.0)
    assert s.closed == 3.0
    adr = m.jnt_qposadr[m.joint("lid_hinge").id]
    assert s.poll() == 0 and s.reward == 0.0
    # reach is rim-based: the lid tray spans x 0.15-0.25, y -0.35..-0.25, z 0.04-0.05. 7 cm above the COM is not a reach ...
    gripper_to(m, d, d.xipos[m.body("headphone_lid").id] + (0, 0, 0.07))
    assert s.poll() == 0 and s.reward == 0.0
    # ... a pinch on the far corner (midpoint 3.5 cm above the rim corner, 8.1 cm from the COM) is
    gripper_to(m, d, (0.15, -0.35, 0.085))
    assert body_surface_distance(m, d, m.body("headphone_lid").id, s.midpoint()) == pytest.approx(0.035)
    assert np.linalg.norm(d.xipos[m.body("headphone_lid").id] - s.midpoint()) > 0.08
    assert s.poll() == 0 and s.reward == 0.1
    d.qpos[adr] = 0.8; mujoco.mj_forward(m, d)
    assert s.poll() == 1 and s.reward == 0.4
    d.qpos[adr] = 2.4; mujoco.mj_forward(m, d)
    assert s.poll() == 1 and s.reward == 0.7
    d.qpos[adr] = 2.95; gripper_to(m, d, (0.3, -0.3, 0.06))  # closed but the fingers rest on the case
    assert any(b in s.gripper for b in s.touching(m.body("headphone_case").id))
    assert s.poll() == 1 and s.reward == 0.7
    gripper_to(m, d, (1, 1, 1))
    assert s.poll() == 2 and s.reward == 1.0 and SUCCESS.is_set()


def test_close_lid_reach_follows_the_rim():
    """The reach distance is to the lid's collision surface, so it moves with the hinge; LID_REACH_M is the single threshold."""
    m, d = load(EASY_XML)
    s = CloseLidScorer(m, d, lambda: 0.0)
    lid = m.body("headphone_lid").id
    adr = m.jnt_qposadr[m.joint("lid_hinge").id]
    assert body_surface_distance(m, d, lid, np.array([0.2, -0.3, 0.05])) == pytest.approx(0.0)  # on the tray's top face
    assert body_surface_distance(m, d, lid, np.array([0.2, -0.3, 0.05 + LID_REACH_M])) == pytest.approx(LID_REACH_M)
    d.qpos[adr] = np.pi / 2; mujoco.mj_forward(m, d)  # lid vertical: the tray stands on the hinge line x = 0.25
    assert body_surface_distance(m, d, lid, np.array([0.2, -0.3, 0.05])) > 0.03
    assert body_surface_distance(m, d, lid, np.array([0.25, -0.3, 0.09])) < 0.012


def test_nail_predicates_and_cursor():
    m, d = load(NAIL_XML)
    grip = {"deg": -120.0}
    s = NailTaskScorer(m, d, lambda: grip["deg"])
    assert s.poll() == 0 and s.reward == 0.0
    gripper_to(m, d, (0.5, 0.2, 0.03))  # reach the large clipper, gripper open
    assert s.poll() == 0 and s.reward == 0.1
    grip["deg"] = GRIPPER_CLOSED_DEG + 10; place(m, d, "clipper_large", (0.5, 0.2, 0.03))  # closed, lifted, near the midpoint
    assert s.poll() == 1 and s.reward == 0.2
    c = SLOTS["clipper_large"]["center"]
    place(m, d, "clipper_large", (0.3 + c[0], c[1], 0.05)); gripper_to(m, d, (0.3 + c[0], c[1], 0.05))  # over the case
    assert s.poll() == 2 and s.reward == 0.3
    grip["deg"] = -120.0; place(m, d, "clipper_large", (0.3 + c[0], c[1], SLOTS["clipper_large"]["floor_z"] + 0.004)); gripper_to(m, d, (1, 1, 1))
    assert s.poll() == 3  # released in the slot
    assert s.poll() == 4 and s.reward == 0.5  # seated
    # small clipper dropped mid-move sends the cursor back to its grasp step; the reward keeps 0.5
    grip["deg"] = GRIPPER_CLOSED_DEG + 10; place(m, d, "clipper_small", (0.5, -0.2, 0.03)); gripper_to(m, d, (0.5, -0.2, 0.03))
    assert s.poll() == 5 and s.reward == 0.6
    grip["deg"] = -120.0; place(m, d, "clipper_small", (0.5, -0.2, 0.004)); gripper_to(m, d, (1, 1, 1))
    assert s.poll() == 4 and s.reward == 0.6
    c = SLOTS["clipper_small"]["center"]
    place(m, d, "clipper_small", (0.3 + c[0], c[1], SLOTS["clipper_small"]["floor_z"] + 0.004))
    s.poll()
    assert s.reward == 1.0 and SUCCESS.is_set()
    assert [e["cursor"] for e in s.events][-1] == 4
