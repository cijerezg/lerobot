"""Step response of one follower joint at several position-loop gains (KP_APR). MOVES THE ARM.

    uv run python lerobot/adhocs/motor_gains/step_test.py --port=/dev/ttyACM0 --joint=shoulder_pan --kps=54,108,216

Only ``joint`` is moved, by ``step_deg`` from the pose the arm is in when the script starts; every other
joint is held at its measured position. Sequence: connect (torque on), then per gain: write KP_APR to the
one joint (NOT stored, a power cycle restores the motor's own value), step the joint by ``step_deg``, hold,
step back, hold. At the end the gain the motor had is written back and the arm parks (the follower's
normal disconnect, every joint to the park pose). Ctrl-C takes the same exit.

Per gain it prints the time to 63% of the step, the overshoot and the ripple while holding, and saves
every sample (time, command, position, motor velocity, motor torque, and the position of all joints) to ``step_<joint>_<date>.csv``.
"""
import argparse
import csv
import datetime
import os
import time

import numpy as np
from motorbridge.damiao_registers import RID_KP_APR

from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerRobotConfig
from lerobot.robots.rebot_b601_follower.rebot_b601_follower import RebotB601Follower

parser = argparse.ArgumentParser()
parser.add_argument("--port", default="/dev/ttyACM0")
parser.add_argument("--joint", default="shoulder_pan")
parser.add_argument("--kps", default="54,108,216")
parser.add_argument("--step_deg", type=float, default=10.0)
parser.add_argument("--hold_s", type=float, default=3.0)
parser.add_argument("--rate_hz", type=float, default=100.0)
args = parser.parse_args()

# the velocity caps of config_rl.yaml, so the response is the one recording and inference see
robot = RebotB601Follower(RebotB601FollowerRobotConfig(port=args.port, id="rebot_follower_v1", pos_vel_velocity=[40.0] * 6 + [180.0]))
robot.connect()
motor = robot.motors[args.joint]
kp_before = motor.damiao_get_param_f32(RID_KP_APR)
print(f"{args.joint}: KP_APR is {kp_before:g}")

samples = []


def hold(target: dict[str, float], seconds: float, kp: float, phase: str) -> np.ndarray:
    """Send ``target`` every tick for ``seconds``; returns [n, 2] = (time since start, joint position)."""
    trace = []
    start = time.perf_counter()
    while (now := time.perf_counter() - start) < seconds:
        robot.send_action({f"{name}.pos": pos for name, pos in target.items()})
        present = robot._present_pos()
        trace.append((now, present[args.joint]))
        samples.append([kp, phase, now, target[args.joint], present[args.joint], robot._present_vel[args.joint], robot._present_torque[args.joint], *present.values()])
        time.sleep(max(0.0, 1.0 / args.rate_hz - (time.perf_counter() - start - now)))
    return np.array(trace)


try:
    pose = robot._present_pos()  # the pose the user left the arm in; every other joint is held there
    hold(pose, 1.0, kp_before, "hold")

    for kp in (float(k) for k in args.kps.split(",")):
        motor.damiao_write_param_f32(RID_KP_APR, kp)
        kp_read = motor.damiao_get_param_f32(RID_KP_APR)
        if abs(kp_read - kp) > 1e-3:
            raise RuntimeError(f"KP_APR write not accepted: asked {kp:g}, motor reports {kp_read:g}")
        hold(pose, 1.0, kp, "settle")
        base = robot._present_pos()[args.joint]
        for phase, goal in (("step", pose[args.joint] + args.step_deg), ("back", pose[args.joint])):
            trace = hold({**pose, args.joint: goal}, args.hold_s, kp, phase)
            start_pos = base if phase == "step" else pose[args.joint] + args.step_deg
            progress = (trace[:, 1] - start_pos) / (goal - start_pos)
            t63 = trace[np.argmax(progress >= 0.63), 0] if (progress >= 0.63).any() else float("nan")
            overshoot = (progress.max() - 1.0) * abs(goal - start_pos)
            ripple = trace[trace[:, 0] > args.hold_s - 1.0, 1].std()
            print(f"KP_APR {kp:5g} {phase:4s}: 63% at {t63:.3f} s, overshoot {max(overshoot, 0.0):.2f} deg, ripple in the last second {ripple:.3f} deg")
finally:
    motor.damiao_write_param_f32(RID_KP_APR, kp_before)
    print(f"{args.joint}: KP_APR written back to {motor.damiao_get_param_f32(RID_KP_APR):g}")
    out_path = os.path.join(os.path.dirname(__file__), f"step_{args.joint}_{datetime.datetime.now():%Y-%m-%d_%H%M%S}.csv")
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows([["kp", "phase", "t_s", "command_deg", "position_deg", "velocity_deg_s", "torque_nm", *robot.motor_names]] + samples)
    print(f"saved {out_path}")
    robot.disconnect()
