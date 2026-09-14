"""Exercise packet boundaries and the total deadline without hardware or real sleeps."""

import struct

import pytest

from lerobot.teleoperators.rebot_102_leader import monitor


def packet(servo_id, angle=100, status=0, temp_raw=2048):
    params = struct.pack("<BHHHHBih", servo_id, 12000, 30, 360, temp_raw, status, angle, 0)
    body = b"\x05\x1c" + bytes([22, len(params)]) + params
    return body + bytes([sum(body) % 256])


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.now += delay


class UART:
    def __init__(self, clock, batches):
        self.clock = clock
        self.batches = list(batches)
        self.pending = []
        self.buffer = bytearray()
        self.writes = []

    def deliver(self):
        while self.pending and self.pending[0][0] <= self.clock.now:
            _, data = self.pending.pop(0)
            self.buffer.extend(data)

    def reset_input_buffer(self):
        self.deliver()
        self.buffer.clear()

    def write(self, data):
        self.writes.append(data)
        self.pending.extend((self.clock.now + delay, payload) for delay, payload in self.batches.pop(0))
        self.pending.sort(key=lambda item: item[0])
        return len(data)

    @property
    def in_waiting(self):
        self.deliver()
        return len(self.buffer)

    def read(self, count):
        out = bytes(self.buffer[:count])
        del self.buffer[:count]
        return out


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(monitor, "time", value)
    return value


def test_valid_replies_return_early_and_request_matches_vendor_protocol(clock):
    uart = UART(clock, [[(0.004, b"".join(packet(i) for i in reversed(range(7))))]])
    result = monitor.read_servo_monitor(uart, list(range(7)), 0.020)
    assert set(result) == set(range(7))
    assert clock.now == pytest.approx(0.004)
    assert result[0].angle_monitor == 10.0
    assert result[0].temp == pytest.approx(25.0)
    from fashionstar_uart_sdk.uservo import Packet, UartServoManager
    expected = Packet.pack(UartServoManager.CODE_SYNC_COMMAND, bytes([22, 1, 7, *range(7)]))
    assert uart.writes == [expected]


@pytest.mark.parametrize("count", [0, 1, 6])
def test_missing_replies_use_one_20ms_budget_not_one_per_servo(clock, count):
    uart = UART(clock, [[(0.004, b"".join(packet(i) for i in range(count)))]])
    result = monitor.read_servo_monitor(uart, list(range(7)), 0.020)
    assert set(result) == set(range(count))
    assert clock.now == pytest.approx(0.020)
    assert len(uart.writes) == 1


def test_fragmented_noisy_and_duplicate_packets_are_parsed_by_valid_id(clock):
    first = packet(0)
    uart = UART(clock, [[
        (0.001, b"noise\x05" + first[:5]),
        (0.003, first[5:] + first + packet(99)),
        (0.006, b"".join(packet(i) for i in range(1, 7))),
    ]])
    result = monitor.read_servo_monitor(uart, list(range(7)), 0.020)
    assert set(result) == set(range(7))
    assert clock.now < 0.007


def test_duplicate_frames_cannot_substitute_for_a_missing_servo(clock):
    uart = UART(clock, [[(0.004, packet(0) * 7)]])
    result = monitor.read_servo_monitor(uart, list(range(7)), 0.020)
    assert set(result) == {0}
    assert clock.now == pytest.approx(0.020)


def test_bad_checksum_does_not_return_previous_transaction_position(clock):
    full = b"".join(packet(i) for i in range(7))
    damaged = full[:-1] + bytes([full[-1] ^ 1])
    uart = UART(clock, [[(0.004, full)], [(0.004, damaged)]])
    assert len(monitor.read_servo_monitor(uart, list(range(7)), 0.020)) == 7
    result = monitor.read_servo_monitor(uart, list(range(7)), 0.020)
    assert 6 not in result
    assert clock.now == pytest.approx(0.024)


@pytest.mark.parametrize("last", [packet(6, angle=-235929599), packet(6, temp_raw=4096)])
def test_invalid_monitor_payload_is_not_a_fresh_sample(clock, last):
    uart = UART(clock, [[(0.004, b"".join(packet(i) for i in range(6)) + last)]])
    assert 6 not in monitor.read_servo_monitor(uart, list(range(7)), 0.020)


def test_total_deadline_does_not_restart_as_replies_arrive(clock):
    uart = UART(clock, [[(0.005, packet(0)), (0.019, packet(1)), (0.021, packet(2))]])
    result = monitor.read_servo_monitor(uart, [0, 1, 2], 0.020)
    assert set(result) == {0, 1}
    assert clock.now == pytest.approx(0.020)


def test_old_bytes_waiting_before_next_request_are_discarded(clock):
    full = b"".join(packet(i) for i in range(7))
    uart = UART(clock, [[(0.025, full)], []])
    assert monitor.read_servo_monitor(uart, list(range(7)), 0.020) == {}
    clock.sleep(0.014)  # next control tick; old replies have arrived
    assert monitor.read_servo_monitor(uart, list(range(7)), 0.020) == {}


def test_short_write_is_an_io_error(clock):
    uart = UART(clock, [])
    uart.write = lambda data: len(data) - 1
    with pytest.raises(OSError, match="Incomplete"):
        monitor.read_servo_monitor(uart, [0], 0.020)
