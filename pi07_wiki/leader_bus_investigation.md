# 102HD leader bus investigation (2026-09-06)

Status: **both root causes found. Overvoltage worked around (highV 13000); the byte loss is the
chipset USB controller and is gone on a CPU-controller port.** §0 is the resolution; §1-§5 are
the trail as written before it. Companion: `leader_102hd_actuation.md` (driver design),
`06_inference.md` (runtime).

## 0. Resolution (12:30)

**Root cause of the unreliable read path: the chipset xHCI controller (PCI `0000:0e:00.0`,
AMD 43fc, USB buses 001/002, which includes the PC-case port) drops bytes from the CH340 when
the adapter sits on one of its root ports.** Same adapter, same cable, same arm on the CPU's own
controller (PCI `0000:10:00.3`, buses 003/004): zero loss.

Evidence, all with `migration/leader_bus_capture.py` (raw bytes of every burst saved as JSON in
`migration/leader_bus_capture_*.json`, re-analysable offline with `analyze --json`):

| condition (7-servo sync burst, 33 ms cadence) | clean bursts |
|---|---|
| chipset root port 1-1, idle | 46/200, 52/200 |
| same, 32 spinning processes (no C-states) | 37/200 |
| same, webcam streaming on the same controller | 47/200 |
| same, 3-servo bursts (0,1,2) and (3,4,5) | 200/200 each |
| same, 4-servo burst (0..3) | 121/200, slot 3 damaged |
| single monitor queries (code 22), each servo x5 | 35/35 |
| CPU controller port 3-1, idle | 200/200 |
| CPU controller port 3-1, 16 spinning processes | 1000/1000 |

- Every loss on the bad port is a **clean deletion of 1-13 bytes starting at stream byte 64, 96
  or 128** (154 of 158 single-cut cases; the rest are tail truncations near byte 134). Those are
  the CH340's 32-byte bulk-packet boundaries. Nothing before byte 64 was ever lost in ~900
  bursts, hence the perfect 63-byte 3-servo bursts. Reversed order (6..0) and gripper-first
  (6,0..5) damage the same byte positions with different servos in them: **position, not servo.**
  No bytes ever precede the first header (no echo). Checksums of the surviving frame parts are
  intact, i.e. the bytes were dropped between the adapter and the kernel, not garbled on the wire.
- The two clean runtime runs (10:47, 11:28: 0 of 2186 reads at the SDK's 100 ms wall) were on
  the CPU controller behind a hub (`3-1.3`). Every lossy capture, including the 11:5x "47/50",
  came after the 11:45 move to the chipset root port `1-1`. That was the whole "standalone vs
  runtime" difference.
- On the good port: first reply byte 0.75 ms after the request, wire done by ~2.3 ms; the CH340
  holds the final partial USB packet ~2 ms, so the last frame is readable at ~4.3 ms and the
  SDK's fast exit lands at ~4.5 ms.

**Port rule.** Leader on a port of `0000:10:00.3` or `0000:10:00.4` (they enumerate as
`usb3`/`usb4` and `usb5`/`usb6`; check with `lsusb -t`, the ch341 must hang under Bus 003 or
005). Never under Bus 001. Free CPU-controller ports right now: 3-2 (USB3 side 4-2), 5-1, 5-2
(6-1, 6-2). The D405 (was 4-1, unplugged at 12:25) goes on any of those directly; the hub is not
needed for it. Acceptance check after any re-plug, before a run:

```
.venv/bin/python migration/leader_bus_capture.py burst --trials 200 --spacing 0.033 --window 0.025
```
must print `200/200 bursts clean`.

**Quirks found on the way (keep):**
- A sync burst naming **id 6 alone** makes the id-7 button board answer too and the two replies
  collide: 162/200 bursts came back 42 bytes (id 7's frame appended), 17/200 empty, the rest
  garbled. Never sync-query the gripper alone; the direct code-22 query is fine (5/5).
- The SDK collector (§3.3) is unchanged: a damaged burst still means a 100 ms wall and **stale,
  unflagged values** for the damaged servos; a burst that ends short by a multiple of 21 bytes
  means the 0.8 s loss path. On the good port this is latent, not active.

**Hypotheses (§4) closed:** H2 confirmed and located at the host controller; H1 true only as
the amplifier (loss -> 100 ms + stale); H3 no echo, collision only in the id-6-alone case; H4
no; H5 no.

**Open (driver, not bus):** replace the collector with our own (validate every frame per read,
re-query missing servos within ~10 ms, trip only on a servo that stays silent); then the
watchdog can go back from 2.0 s toward 0.5 s; a connect-time burst self-test would have named
this port in one second.

## 1. TL;DR

- The morning's "leader cannot follow" trips were **servo overvoltage protection**, not gravity
  and not a bad servo. The "12 V" supplies measure 12.54 / 12.58 V; the servos' high-voltage
  protection was 12600 mV; a tripped servo releases torque and ignores commands until re-power.
  Worked around by writing 13000 mV to data_id 40 on all seven (user's decision, ~3% over the
  vendor's 9.0-12.6 V range). First run after: 54 s, no protection bits, bus peak 12679 mV,
  leader shadowed the whole reach (shoulder_lift 819 mA, 18 deg max error).
- That run ended on the per-step ceiling after an **834 ms leader read stall**. The 834 ms is
  **not USB**: it is the vendor SDK's `monitor_update` loss path (`0.1 s x 7 + 0.1 s`) waiting
  for a sync-monitor burst that never completed.
- Standalone, with nothing else running and the adapter on a direct root port, **47 of 50
  sync-monitor bursts came back incomplete**, every burst took exactly 100 ms, and the missing
  set is structured (servo 6 missing in all 47, then {3} / {4,5} / {0,1} / all). Single-servo
  queries never fail but always take 100 ms (the SDK waits its full window by design).
- The runtime loop nevertheless ran at 33 ms with the read costing ~5 ms, so in the runtime the
  burst usually completes. **Why standalone and runtime differ is the open question.** Keep an
  open mind: SDK collector, CH340 receive path, bus echo/collision, and servo-order effects are
  all live hypotheses; nothing is proven against the wire yet.

## 2. Timeline (all 2026-09-06)

| time | event | outcome |
|---|---|---|
| 10:15 | run trips: watchdog "no feedback command for 0.823s" right after "monitor read 1/3: no reply from all seven" | watchdog raised 0.5 -> 2.0 s (the read holds `io_lock` through the retry) |
| 10:25 | tracking trip, elbow_flex 41.4 raw deg / 0.77 s; follower elbow moved -16 -> -42 at ~7 deg/s | leader joint stationary, not slow |
| 10:37 | tracking trip, shoulder_lift 48.0 | same pattern |
| 10:47 | first run with telemetry: shoulder_lift status 1 -> 9 (executing + OVERVOLTAGE) at t=11.9 s while at rest, 30 mA; raw frozen at 0.0 for 7.5 s while target ramped 0 -> 47 | protection, not load |
| 11:0x | bus read at idle: 12.31-12.61 V per servo; multimeter 12.58 V (other supply 12.54) | data_id 40 = 12600 on all seven |
| 11:2x | wrote data_id 40 = 13000 on all seven; persisted across power cycle | 480-690 mV margin |
| 11:28 | run: 1633 steps / 54 s, no error bits, bus peak 12679 mV (t=14 s), min 11.8 V; two 834 ms read stalls (41.8 s, 59.7 s); ended on ceiling "shoulder_lift asked to move -12.1 raw deg" | ceiling made time-scaled (4x cap) |
| 11:45 | leader moved from a hub to a direct root port (Bus 001 port 1, C-to-C cable enumerates fine); D405 alone on Bus 004 | no config change (still ttyUSB0) |
| 11:5x | standalone burst test: 47/50 incomplete, 100 ms each; 7 single queries 700 ms | see §4 |

## 3. Established facts (with the evidence)

### 3.1 Hardware and protection
- 102-HD = 12 V-class U45H-M UART servos, working 9.0-12.6 V, standby < 40 mA (vendor wiki).
  Vendor text on voltage protection: outside the set range the servo "automatically releases
  its lock force, outputs no torque, enters a free state; recovery requires re-powering with the
  voltage back in range". Status BIT3 clears by itself when the voltage is normal; the torque
  release does not.
- Protocol user area (protocol PDF v0.5.0.12092024, appendix B): data_id 37 stall mode, 38 stall
  power cap mW, 39 low-voltage mV, 40 high-voltage mV, 41 temperature ADC, 42 power mW, 43
  current mA, 46 power-on brake, 48-52 angle limits. All R/W via SDK `read_data` / `write_data`
  (little-endian). **Never call `reset_user_data` (cmd 2): it wipes servo IDs and baud.**
  Factory values read: highV 12600 everywhere; power/stall caps 50 W (ids 1,2), 30 W (3), 10/6 W
  (0,4,5), 4/2 W (gripper). The `power` field we send is clamped to these caps.
- Web config tool exposes only teleop parameters; the PC config software or `write_data` is
  needed for protection values.
- USB: leader = CH340 (`1a86 USB Serial`, `ch341`, full-speed 12M) at `/dev/ttyUSB0`, now on
  Bus 001 port 1 directly under the root; D405 alone on Bus 004; follower `/dev/ttyACM0` (HDSC
  CDC) behind a hub on Bus 001 shared with a USB2 webcam. Vendor README lists "Communication hub:
  UC-01, UART" for all variants; the boxed UC-01 is unused here, so ttyUSB0 is either the
  base-integrated equivalent or another CH340 board (not yet physically traced).

### 3.2 Driver behaviour (as built today, all in `hd_controller.py`)
- Watchdog 2.0 s. Tracking fault 40 raw deg / 0.75 s. Per-step ceiling 8 raw deg x
  clamp(elapsed / (1/30 s), 1, 4). Monitor read retries 3x.
- Telemetry: status/voltage/current/power/temperature kept from the monitor reply; status-bit
  transitions logged once ("Leader <joint> status: <flags>"); `{output_dir}/leader_trace.csv`
  per read (`time, torqued, intervening`, per joint `target, raw, ma, mv, mw, c, status`);
  a tracking/current trip dumps the joint's last 1.0 s.

### 3.3 SDK sync-monitor collector (`fashionstar_uart_sdk/uservo.py`, read today)
- `send_sync_servo_monitor(ids, realtime=True)` -> `monitor_update(ids, timeout=0.1)`.
- `monitor_update` is a byte-level state machine: scan byte-by-byte for `05 1c`, then read
  exactly 19 bytes, then 21-byte chunks **only when `in_waiting >= 21`**; exits early only when
  `len(buffer) >= 21 * n`. With bytes waiting past 0.1 s it breaks (100 ms). With no bytes
  waiting it enters the loss path and only gives up after `0.1 * n + 0.1` = **0.8 s** with
  `loss_count > n` -> that is the 834 ms stall.
- Per-servo `is_response` is computed from parsed frames **only on the loss path**; otherwise it
  is set to all-True regardless, and `realtime=True` then returns the cached servo objects. The
  controller detects a missing servo via `angle_monitor is None`, which only works for servos
  that have never answered. A dropped frame on a non-loss-path exit therefore yields a **stale
  reading** for that servo, unflagged.
- `update(wait_response=True, timeout=0.1)` (used by `read_data`, `query_servo_monitor`) reads
  until no bytes are waiting AND the timeout has elapsed: **every single query costs 100 ms** by
  design, even when the reply arrived in 2 ms.
- Response frame: header `05 1c`, code, size, params (`<BHHHHBih`: id, mV, mA, mW, tempADC,
  status, angle x10, turns), checksum = 21 bytes. Servo id is the byte at header offset 4.

### 3.4 Standalone burst test (adapter direct, arm idle, torque off, ~8 Hz cadence)
```
sync burst: median 100.1 ms  max 100.2 ms  bursts with missing frames: 47/50
missing sets: [3,6] x19, [4,5,6] x15, [6] x9, [0..6] x3, [0,1,6] x2  (servo 6 in all 47)
7 single queries: 700.5 ms total
```
Reading: the collector hit its 100 ms wall with bytes still waiting (the `if bytes_waiting`
branch is the only 100 ms exit), i.e. it stopped consuming input — consistent with a leftover
of < 21 bytes at the tail, which the chunker never reads. The structure of the missing sets
(always the last servo, then contiguous groups) points at chunk misalignment more than at random
line loss, but a raw capture (§5.1) is needed to say which.

### 3.5 Runtime contrast
- 11:28 run: loop at 33 ms, `action_proc` (contains the leader read) ~5 ms average, two 834 ms
  stalls in 1634 reads. So the burst completed on most steps in the runtime.
- Stale-row check on that trace (rows identical to the previous row in raw, mA, mV, mW, degC):
  gripper 9.0 %, shoulder_pan 4.2 %, wrist_yaw 3.2 %, wrist_roll 2.9 %, others ~1 %; only 1 row
  identical across all seven; shoulder_lift mV exact repeats 261/1634. **Inconclusive**: at rest
  a genuine repeat of five quantised fields is plausible; the gripper (servo 6, last in the burst)
  being highest is suggestive but not proof. A capture under runtime conditions decides it.
- Differences between the two situations that could matter: cadence (33 ms vs ~120 ms), torque
  on/off and commands interleaved on the bus, port freshly opened (CH340 DTR/RTS toggle on open),
  the retry masking losses in the runtime (attempt 2 usually succeeds).

## 4. Hypotheses (all open; test, do not assume)

- **H1 SDK chunker**: the 19/21-byte fixed chunking mis-handles the burst under some byte-arrival
  pattern (partial tail < 21 bytes never read; exits at 100 ms). Predicts: a raw capture shows
  all 147 bytes present and well-formed.
- **H2 CH340 receive path**: bytes lost when the 147-byte burst at 1 Mbaud outruns the adapter's
  RX buffer / USB polling. Predicts: the raw capture is short, with corrupted checksums, and
  loss depends on burst size (ids 0-3 vs 0-6).
- **H3 half-duplex echo or collision**: the request is echoed into RX, or two servos overlap.
  Predicts: extra bytes before the first `05 1c`, or overlapping/garbled frames at a fixed
  position.
- **H4 servo order**: servo 6 (gripper, 4 W cap) is always missing; something about the last
  responder (reply latency, the button on id 7 also answering, gripper firmware). Predicts:
  reordering ids in the request moves which servo is missing.
- **H5 open-port transient**: the first reads after `serial.Serial(...)` are bad, and the runtime
  survives because it keeps the port open. Predicts: losses fall after a warm-up.

## 5. Next steps (in order; each is a standalone script with the arm powered, torque off)

### 5.1 Raw capture of one burst (decides H1 vs H2/H3)
```
.venv/bin/python - <<'EOF'
import time, struct, serial, fashionstar_uart_sdk as uservo
u = serial.Serial("/dev/ttyUSB0", 1_000_000, timeout=0)
c = uservo.UartServoManager(u, srv_num=8)
ids = [0, 1, 2, 3, 4, 5, 6]
param = struct.pack("<BBB", c.CODE_QUERY_SERVO_MONITOR, 1, len(ids)) + bytes(ids)
for trial in range(8):
    u.reset_input_buffer()
    t0 = time.perf_counter(); c.send_request(c.CODE_SYNC_COMMAND, param)
    buf = bytearray(); first = None; last = t0
    while time.perf_counter() - t0 < 0.05:
        n = u.in_waiting
        if n:
            buf += u.read(n); last = time.perf_counter()
            if first is None: first = last
    hdr = [i for i in range(len(buf) - 1) if buf[i] == 0x05 and buf[i + 1] == 0x1C]
    print(f"trial {trial}: {len(buf)} bytes (147 expected), first byte {1000*((first or t0)-t0):.1f} ms, "
          f"last {1000*(last-t0):.1f} ms, headers at {hdr}, ids {[buf[i+4] for i in hdr if i+4 < len(buf)]}")
    print("   ", buf.hex(" "))
    time.sleep(0.05)
EOF
```
Read: 147 bytes with headers at 0,21,...,126 and ids 0..6 = the wire is fine and H1 holds.
Fewer bytes / bad spacing = H2 or H3; bytes before the first header = echo (H3).

### 5.2 Burst-size dependence (H2) — same script with `ids = [0,1,2,3]` and `[6,5,4,3,2,1,0]` (H4).

### 5.3 True reply latency of a single query — same loop with
`param = struct.pack("<B", sid)` sent via `c.send_request(c.CODE_QUERY_SERVO_MONITOR, param)`;
report first/last byte times. This sets the budget for a per-servo fallback (expected ~1-3 ms,
not the SDK's 100 ms).

### 5.4 Warm-up and cadence (H5) — run 5.1 for 200 trials at 33 ms spacing; report loss rate
in the first 20 vs the rest.

### 5.5 Then decide the driver change. Candidate (drafted, not built): own collector in
`hd_controller._read_raw_locked`: send the sync request, collect for <= ~25 ms, parse with the
SDK `PacketBuffer` + `response_query_servo_monitor` (so servo objects update as before), mark a
servo answered **only if its frame parsed this call**, single-query the missing ones within the
same step using our own short window, trip only if a servo stays silent. If 5.1 shows the wire
is lossy at burst size 7, fall back to per-servo queries (7 x ~2 ms) instead of the burst.
Baud reduction (data_id 36) is a last resort: writing it wrong bricks addressing.

### 5.6 Physically trace the ttyUSB0 board (UC-01 or base-integrated?), photo the bus wiring.

## 6. Standing rules from today
- After any leader trip: **unplug/replug the arm** before the next run (released servos do not
  re-engage on their own).
- Leave data_id 40 at 13000 unless the supply is brought to <= 12.0 V at the servos; undo is a
  write of 12600, never `reset_user_data`.
- Read the trace after each run: peak mV per servo, error bits, count of read intervals > 60 ms.
