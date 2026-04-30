#!/bin/bash

set -euo pipefail

# Use stable by-id paths when possible to avoid ttyACM renumbering.
# You can override these without editing the file:
#   FOLLOWER_PORT=/dev/serial/by-id/<...> LEADER_PORT=/dev/serial/by-id/<...> ./teleoperate.sh
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5876043763-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A057748-if00}"
export FOLLOWER_PORT
export LEADER_PORT

# Preflight: verify the configured ports actually respond with Feetech servo IDs.
.venv/bin/python - <<'PY'
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode
import os, sys

def ping_bus(port: str) -> bool:
    bus = FeetechMotorsBus(
        port=port,
        motors={f"m{i}": Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i in range(1, 7)},
        calibration=None,
    )
    try:
        bus.connect(handshake=False)
        ids = bus.broadcast_ping(raise_on_error=False)
        ok = bool(ids) and len(ids) > 0
        print(f"[preflight] {port}: broadcast_ping={ids}")
        return ok
    except Exception as e:
        print(f"[preflight] {port}: ERROR {e!r}")
        return False
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass

follower = os.getenv("FOLLOWER_PORT")
leader = os.getenv("LEADER_PORT")
if not follower or not leader:
    raise RuntimeError("FOLLOWER_PORT and LEADER_PORT must be set (they are normally set by teleoperate.sh).")

ok_follower = ping_bus(follower)
ok_leader = ping_bus(leader)

if not ok_follower or not ok_leader:
    print(
        "\n[preflight] One or more ports did not respond with any servo IDs.\n"
        "Common causes:\n"
        "  - wrong port (ttyACM swapped)\n"
        "  - controller/servo bus not powered\n"
        "  - USB cable issue\n"
        "  - ModemManager interference (try stopping it)\n",
        file=sys.stderr,
    )
    sys.exit(2)
PY

.venv/bin/python ./src/lerobot/scripts/lerobot_teleoperate.py \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.cameras="{ screwdriver: {type: opencv, index_or_path: /dev/video0, width: 800, height: 600, fps: 30, fourcc: MJPG}, side: {type: opencv, index_or_path: /dev/video4, width: 800, height: 600, fps: 30, fourcc: MJPG}}" \
    --robot.id=so101_follower_2026_04_12 \
    --teleop.type=so101_leader \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id=so101_leader_2026_04_12 \
    --fps=30 \
    --display_data=false