"""Print leader vs follower joint positions every 3s, read-only.

Follower is read limp via motorbridge (NO torque enable) so you can move it by
hand. Leader is read through its LeRobot teleop class (passive by nature).

Move BOTH arms to the same physical pose and compare:
  - follower_deg     : where the follower actually is (calibrated)
  - leader_cmd_deg   : what the leader WOULD command the follower (get_action)
  - leader_raw_deg   : leader's raw servo angle before direction/range mapping

If you put both arms in the same pose, follower_deg and leader_cmd_deg should
match. Any joint where they don't is the mismatch.

Run: uv run lerobot/src/lerobot/robots/rebot_b601_follower/compare_arms.py
"""

import math
import time

from motorbridge import Controller

from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower.rebot_b601_follower import MOTOR_MODELS
from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader import (
    RebotArm102LeaderTeleopConfig,
)
from lerobot.teleoperators.rebot_102_leader.rebot_102_leader import RebotArm102Leader

FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyUSB0"
LEADER_ID = "rebot_leader_v1"
BAUD = 921600

MOTOR_CAN_IDS = RebotB601FollowerConfig.__dataclass_fields__["motor_can_ids"].default_factory()
JOINTS = list(MOTOR_CAN_IDS)

# ---- follower (limp, read-only via motorbridge) ----
fbus = Controller.from_dm_serial(serial_port=FOLLOWER_PORT, baud=BAUD)
fmotors = {n: fbus.add_damiao_motor(s, r, MOTOR_MODELS[n]) for n, (s, r) in MOTOR_CAN_IDS.items()}

# ---- leader (LeRobot teleop class, read-only) ----
leader = RebotArm102Leader(RebotArm102LeaderTeleopConfig(port=LEADER_PORT, id=LEADER_ID))
leader.connect()

print("Move both arms to the same pose. Ctrl-C to stop.\n")
try:
    while True:
        # follower positions
        for m in fmotors.values():
            m.request_feedback()
        fbus.poll_feedback_once()
        follower = {}
        for n, m in fmotors.items():
            st = m.get_state()
            follower[n] = math.degrees(st.pos) if st is not None else None

        # leader positions
        raw = leader._read_raw_positions()
        cmd = leader.get_action()

        print(f"{'joint':<14}{'follower_deg':>13}{'leader_cmd_deg':>16}{'leader_raw_deg':>16}{'diff':>9}")
        print("-" * 68)
        for n in JOINTS:
            f = follower[n]
            c = cmd.get(f"{n}.pos")
            r = raw.get(n)
            diff = (f - c) if (f is not None and c is not None) else float("nan")
            fs = f"{f:8.1f}" if f is not None else "    None"
            print(f"{n:<14}{fs:>13}{c:>16.1f}{r:>16.1f}{diff:>9.1f}")
        print()
        time.sleep(3)
except KeyboardInterrupt:
    pass
finally:
    leader.disconnect()
    fbus.close()
