"""Read the follower motors' control-loop registers. Read only: no enable, no mode change, no write.

    uv run python lerobot/adhocs/motor_gains/read_gains.py --port=/dev/ttyACM0

Prints one row per register and one column per joint, and saves the table next to this file as
``gains_<date>.csv`` so a later write can be compared against (and reverted to) what the motors had.
"""
import argparse
import csv
import datetime
import os

from motorbridge import Controller
from motorbridge.damiao_registers import DAMIAO_RW_REGISTERS

from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower.rebot_b601_follower import MOTOR_MODELS

# acceleration / deceleration / speed limit, control mode, mapping ranges, current, speed and position loops
REGISTERS = (4, 5, 6, 10, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34)

parser = argparse.ArgumentParser()
parser.add_argument("--port", default="/dev/ttyACM0")
args = parser.parse_args()

can_ids = RebotB601FollowerConfig(port=args.port).motor_can_ids
bus = Controller.from_dm_serial(serial_port=args.port)
motors = {name: bus.add_damiao_motor(send_id, recv_id, MOTOR_MODELS[name]) for name, (send_id, recv_id) in can_ids.items()}

rows = []
for rid in REGISTERS:
    spec = DAMIAO_RW_REGISTERS[rid]
    read = "get_register_u32" if spec.data_type == "u32" else "get_register_f32"
    rows.append([rid, spec.variable, spec.description] + [getattr(motor, read)(rid) for motor in motors.values()])
bus.close()

header = ["rid", "name", "description"] + [f"{name} ({MOTOR_MODELS[name]})" for name in motors]
print(" | ".join(header))
for row in rows:
    print(" | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in row))

out_path = os.path.join(os.path.dirname(__file__), f"gains_{datetime.date.today().isoformat()}.csv")
with open(out_path, "w", newline="") as f:
    csv.writer(f).writerows([header] + rows)
print(f"saved {out_path}")
