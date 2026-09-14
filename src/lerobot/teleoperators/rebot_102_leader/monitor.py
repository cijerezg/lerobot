"""Bounded FashionStar monitor transactions, without the SDK's per-servo timeout.

The caller owns the UART lock. Reads must be nonblocking and writes must have a
finite timeout. Only packets validated in this transaction count as replies;
the SDK's cached servo objects are deliberately not used.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import serial


@dataclass(frozen=True)
class MonitorSample:
    angle_monitor: float
    current: int
    power: int
    voltage: int
    temp: float
    status: int


def read_servo_monitor(
    uart: serial.Serial, servo_ids: list[int], timeout_s: float
) -> dict[int, MonitorSample]:
    """Send once; return valid replies received before one total deadline.

    Missing IDs are absent from the result. Discard bytes left from an earlier
    transaction before sending. The protocol has no transaction IDs, so a delayed
    old reply arriving *after* the new request cannot be distinguished on the wire.
    """
    deadline = time.monotonic() + timeout_s
    uart.reset_input_buffer()
    params = bytes([22, 1, len(servo_ids), *servo_ids])
    request = b"\x12\x4c" + bytes([25, len(params)]) + params
    request += bytes([sum(request) % 256])
    if uart.write(request) != len(request):
        raise OSError("Incomplete leader monitor request write")

    expected = set(servo_ids)
    received: dict[int, MonitorSample] = {}
    buffer = bytearray()
    while time.monotonic() < deadline:
        waiting = uart.in_waiting
        if not waiting:
            time.sleep(min(0.0002, max(0.0, deadline - time.monotonic())))
            continue
        buffer.extend(uart.read(min(waiting, 4096)))
        while len(buffer) >= 4 and time.monotonic() < deadline:
            if buffer[:4] != b"\x05\x1c\x16\x10":
                del buffer[0]
                continue
            if len(buffer) < 21:
                break
            packet = bytes(buffer[:21])
            if sum(packet[:-1]) % 256 != packet[-1]:
                del buffer[0]
                continue
            del buffer[:21]
            servo_id, voltage, current, power, temp_raw, status, angle, turn = struct.unpack(
                "<BHHHHBih", packet[4:-1]
            )
            if servo_id not in expected or (angle == -235929599 and turn == 0):
                continue
            if not 0 <= temp_raw < 4096:
                continue
            # Match the vendor SDK's thermistor conversion.
            temp = (
                1 / (math.log(temp_raw / (4096 - temp_raw)) / 3435 + 1 / 298.15) - 273.15
                if temp_raw else 0.0
            )
            received[servo_id] = MonitorSample(angle / 10.0, current, power, voltage, temp, status)
            if received.keys() == expected:
                return received
    return received
