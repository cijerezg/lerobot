#!/usr/bin/env python3
"""Inspect the SO-101 leader servos referenced by ``record.sh``.

The Feetech STS3215 bus reports model number and firmware information, but the
standard registers used by LeRobot do not expose the C001/C044/C046 suffix. This
script therefore reports the bus-readable identity plus the SO-101 leader's
expected variant and gear ratio for each joint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.motors import Motor, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402
from lerobot.motors.feetech.tables import (  # noqa: E402
    MODEL_NUMBER_TABLE,
    STS_SMS_SERIES_BAUDRATE_TABLE,
)
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS  # noqa: E402


DEFAULT_RECORD_SH = REPO_ROOT / "record.sh"

DEFAULT_LEADER_MOTORS = {
    "shoulder_pan": (1, MotorNormMode.RANGE_M100_100),
    "shoulder_lift": (2, MotorNormMode.RANGE_M100_100),
    "elbow_flex": (3, MotorNormMode.RANGE_M100_100),
    "wrist_flex": (4, MotorNormMode.RANGE_M100_100),
    "wrist_roll": (5, MotorNormMode.RANGE_M100_100),
    "gripper": (6, MotorNormMode.RANGE_0_100),
}

EXPECTED_SO101_LEADER_VARIANTS = {
    "shoulder_pan": ("STS3215-C044", "1:191"),
    "shoulder_lift": ("STS3215-C001", "1:345"),
    "elbow_flex": ("STS3215-C044", "1:191"),
    "wrist_flex": ("STS3215-C046", "1:147"),
    "wrist_roll": ("STS3215-C046", "1:147"),
    "gripper": ("STS3215-C046", "1:147"),
}

MODEL_NAME_BY_NUMBER = {model_number: model for model, model_number in MODEL_NUMBER_TABLE.items()}
BAUDRATE_BY_CODE = {code: baudrate for baudrate, code in STS_SMS_SERIES_BAUDRATE_TABLE.items()}


@dataclass
class RecordShConfig:
    leader_id: str
    leader_port: str
    teleop_type: str | None


def _strip_shell_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_record_sh(path: Path) -> RecordShConfig:
    text = path.read_text()

    leader_id_match = re.search(r"--teleop\.id=(?P<value>[^\s\\]+)", text)
    if leader_id_match is None:
        raise ValueError(f"Could not find --teleop.id in {path}")
    leader_id = _strip_shell_quotes(leader_id_match.group("value"))

    teleop_type_match = re.search(r"--teleop\.type=(?P<value>[^\s\\]+)", text)
    teleop_type = _strip_shell_quotes(teleop_type_match.group("value")) if teleop_type_match else None

    leader_port = os.getenv("LEADER_PORT")
    if not leader_port:
        env_default_match = re.search(
            r'^\s*LEADER_PORT="\$\{LEADER_PORT:-(?P<value>[^}]+)\}"',
            text,
            flags=re.MULTILINE,
        )
        plain_match = re.search(
            r"^\s*LEADER_PORT=(?P<value>[^\n#]+)",
            text,
            flags=re.MULTILINE,
        )
        if env_default_match is not None:
            leader_port = env_default_match.group("value").strip()
        elif plain_match is not None:
            leader_port = _strip_shell_quotes(plain_match.group("value"))

    if not leader_port:
        raise ValueError(f"Could not find LEADER_PORT in {path}; pass --port explicitly")

    return RecordShConfig(leader_id=leader_id, leader_port=leader_port, teleop_type=teleop_type)


def _load_calibration(leader_id: str) -> tuple[dict[str, Any], Path | None]:
    candidates = [
        HF_LEROBOT_CALIBRATION / TELEOPERATORS / "so_leader" / f"{leader_id}.json",
        HF_LEROBOT_CALIBRATION / TELEOPERATORS / "so101_leader" / f"{leader_id}.json",
    ]
    for path in candidates:
        if path.is_file():
            with path.open() as f:
                return json.load(f), path
    return {}, None


def _build_leader_motors(calibration: dict[str, Any]) -> dict[str, Motor]:
    motors = {}
    for name, (default_id, norm_mode) in DEFAULT_LEADER_MOTORS.items():
        motor_id = int(calibration.get(name, {}).get("id", default_id))
        motors[name] = Motor(motor_id, "sts3215", norm_mode)
    return motors


def _safe_read(bus: FeetechMotorsBus, register: str, motor: str, retries: int) -> Any:
    try:
        return bus.read(register, motor, normalize=False, num_retry=retries)
    except Exception as exc:
        return f"ERR: {exc.__class__.__name__}"


def _fmt_model(model_number: Any) -> str:
    if not isinstance(model_number, int):
        return str(model_number)
    model_name = MODEL_NAME_BY_NUMBER.get(model_number, "unknown")
    return f"{model_name} ({model_number})"


def _fmt_voltage(raw_voltage: Any) -> str:
    if isinstance(raw_voltage, int):
        return f"{raw_voltage / 10:.1f} V"
    return str(raw_voltage)


def _fmt_temperature(raw_temperature: Any) -> str:
    if isinstance(raw_temperature, int):
        return f"{raw_temperature} C"
    return str(raw_temperature)


def _fmt_baud(raw_baud_code: Any) -> str:
    if isinstance(raw_baud_code, int):
        baudrate = BAUDRATE_BY_CODE.get(raw_baud_code)
        if baudrate is not None:
            return f"{baudrate} ({raw_baud_code})"
    return str(raw_baud_code)


def _print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    widths = {}
    for key, title in columns:
        widths[key] = max(len(title), *(len(str(row.get(key, ""))) for row in rows))

    header = "  ".join(title.ljust(widths[key]) for key, title in columns)
    divider = "  ".join("-" * widths[key] for key, _ in columns)
    print(header)
    print(divider)
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns))


def inspect_leader(port: str, leader_id: str, retries: int) -> int:
    calibration, calibration_path = _load_calibration(leader_id)
    motors = _build_leader_motors(calibration)
    bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)

    print(f"leader_id: {leader_id}")
    print(f"leader_port: {port}")
    if calibration_path is not None:
        print(f"calibration: {calibration_path}")
    else:
        print("calibration: not found; using default SO leader motor ids 1-6")
    print()

    try:
        bus.connect(handshake=False)
        found_models = bus.broadcast_ping(num_retry=retries, raise_on_error=False) or {}
        if not found_models:
            print("broadcast_ping returned no motors; trying expected IDs one by one")
            for motor in motors.values():
                model_number = bus.ping(motor.id, num_retry=retries, raise_on_error=False)
                if model_number is not None:
                    found_models[motor.id] = model_number

        expected_ids = {motor.id for motor in motors.values()}
        found_ids = set(found_models)
        missing_ids = sorted(expected_ids - found_ids)
        extra_ids = sorted(found_ids - expected_ids)

        rows = []
        raw_rows = []
        for name, motor in motors.items():
            expected_variant, expected_ratio = EXPECTED_SO101_LEADER_VARIANTS[name]
            model_number = found_models.get(motor.id)
            if model_number is None:
                model_number = _safe_read(bus, "Model_Number", name, retries)
            firmware_major = _safe_read(bus, "Firmware_Major_Version", name, retries)
            firmware_minor = _safe_read(bus, "Firmware_Minor_Version", name, retries)
            firmware = (
                f"{firmware_major}.{firmware_minor}"
                if isinstance(firmware_major, int) and isinstance(firmware_minor, int)
                else f"{firmware_major}/{firmware_minor}"
            )
            present_voltage = _safe_read(bus, "Present_Voltage", name, retries)
            present_temperature = _safe_read(bus, "Present_Temperature", name, retries)
            baud_rate = _safe_read(bus, "Baud_Rate", name, retries)
            angular_resolution = _safe_read(bus, "Angular_Resolution", name, retries)
            velocity_unit_factor = _safe_read(bus, "Velocity_Unit_factor", name, retries)
            max_velocity_limit = _safe_read(bus, "Maximum_Velocity_Limit", name, retries)

            rows.append(
                {
                    "joint": name,
                    "id": motor.id,
                    "bus_model": _fmt_model(model_number),
                    "firmware": firmware,
                    "voltage": _fmt_voltage(present_voltage),
                    "temp": _fmt_temperature(present_temperature),
                    "expected_variant": expected_variant,
                    "expected_ratio": expected_ratio,
                    "actual_suffix": "not exposed",
                }
            )
            raw_rows.append(
                {
                    "joint": name,
                    "id": motor.id,
                    "baud": _fmt_baud(baud_rate),
                    "angular_resolution": angular_resolution,
                    "velocity_unit_factor": velocity_unit_factor,
                    "max_velocity_limit": max_velocity_limit,
                }
            )

        _print_table(
            rows,
            [
                ("joint", "joint"),
                ("id", "id"),
                ("bus_model", "bus model"),
                ("firmware", "firmware"),
                ("voltage", "voltage"),
                ("temp", "temp"),
                ("expected_variant", "expected variant"),
                ("expected_ratio", "expected ratio"),
                ("actual_suffix", "actual suffix"),
            ],
        )
        print()
        _print_table(
            raw_rows,
            [
                ("joint", "joint"),
                ("id", "id"),
                ("baud", "baud"),
                ("angular_resolution", "angular_resolution"),
                ("velocity_unit_factor", "velocity_unit_factor"),
                ("max_velocity_limit", "max_velocity_limit"),
            ],
        )

        print()
        print(
            "Note: STS3215 C001/C044/C046 and gear ratio are not exposed by the standard Feetech "
            "registers used here. The expected variant/ratio columns are the SO-101 leader allocation; "
            "verify the actual C-code on the servo label or order invoice."
        )

        status = 0
        if missing_ids:
            print(f"ERROR: expected IDs not found on the leader bus: {missing_ids}", file=sys.stderr)
            status = 2
        if extra_ids:
            print(f"WARNING: extra IDs found on the leader bus: {extra_ids}", file=sys.stderr)
        wrong_models = {
            id_: model_number
            for id_, model_number in found_models.items()
            if id_ in expected_ids and model_number != MODEL_NUMBER_TABLE["sts3215"]
        }
        if wrong_models:
            print(f"ERROR: expected STS3215 model number 777, got: {wrong_models}", file=sys.stderr)
            status = 2
        return status
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-sh",
        type=Path,
        default=DEFAULT_RECORD_SH,
        help=f"Path to record.sh to read --teleop.id and LEADER_PORT from (default: {DEFAULT_RECORD_SH})",
    )
    parser.add_argument("--leader-id", help="Override the leader id parsed from record.sh")
    parser.add_argument("--port", help="Override LEADER_PORT parsed from record.sh or the environment")
    parser.add_argument("--retries", type=int, default=2, help="Read retry count per register (default: 2)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_cfg = _parse_record_sh(args.record_sh)
    leader_id = args.leader_id or record_cfg.leader_id
    port = args.port or record_cfg.leader_port

    if record_cfg.teleop_type and record_cfg.teleop_type != "so101_leader":
        print(f"WARNING: record.sh teleop type is {record_cfg.teleop_type!r}, expected 'so101_leader'")

    try:
        return inspect_leader(port=port, leader_id=leader_id, retries=args.retries)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
