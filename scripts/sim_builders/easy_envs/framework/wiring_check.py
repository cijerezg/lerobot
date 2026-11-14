"""Connect RebotB601Sim (no cameras) on each on-disk scene with auto_subtasks, step 1 s and poll: the registry picks the scorer."""
import numpy as np

from lerobot.robots.rebot_b601_sim import RebotB601Sim, RebotB601SimRobotConfig, task_for

for scene in ["clippers_only", "ep0", "ep3", "lift_bottle"]:
    robot = RebotB601Sim(RebotB601SimRobotConfig(scene=scene, auto_subtasks=True))
    robot.connect()
    hold = dict(zip(robot.action_features, robot._present_deg(), strict=True))
    for _ in range(30):
        robot.send_action(hold)
    cursor = robot.subtask_scorer.poll()
    task = task_for(scene)
    print(f"{scene}: {type(robot.subtask_scorer).__name__} tier={task.tier} cursor={cursor} reward={robot.subtask_scorer.reward} t={robot.subtask_scorer.t:.2f} predicates={ {k: bool(v) for k, v in robot.subtask_scorer._predicates().items()} }")
    robot.disconnect()
