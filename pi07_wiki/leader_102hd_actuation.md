# reBot 102HD leader: actuation, shadowing, and policy preview

Plan of record for making the second leader arm (a **Star Arm 102HD**, acquired 2026-09-02)
drive, and for the three features that unlocks. The original leader is a **102LD** and is
encoder-only; both arms must keep working from the same driver.

Status: **measurements complete, implementation not started.**

---

## 1. Established facts

All measured on the hardware 2026-09-02 with `probe_new_leader.py` and `probe_hd_vendor.py`
(repo root). Vendor source of truth: `github.com/servodevelop/Star-Arm-102`,
`Python_SDK/stararm102_ro.py` (button/lock) and `stararm102_ro_hover.py` (powered hold).

### 1.1 Bus and identity

| | |
|---|---|
| Port | `/dev/ttyUSB0`, CH340 (`1a86:7523`), `ch341-uart`, 1 Mbaud |
| Joints | ids 0-6 = pan, lift, elbow, wrist_flex, wrist_yaw, wrist_roll, gripper |
| Button board | **id 7 — present on HD only.** LD enumerates 0-6 |
| Follower | `/dev/ttyACM0` — a different device class, never confuse the two |

`sync_monitor` misparses id 7 (turn 255, 22136 mA, 4660 mV, temp 13398) because it is not a
joint servo. Read only its `angle_deg`, which is a state flag:
**0.0 = unlocked, 180.0 = lock requested, 90.0 = unlock requested** (vendor `Button.key_state_*`).

### 1.2 The actuation rule

Stop modes, from the SDK source (`stop_on_control_mode(servo_id, method, power)`):

| method | meaning |
|---|---|
| `0x10` | 停止后卸力 — unload after stop |
| `0x11` | 停止后保持锁力 — hold locking force after stop |
| `0x12` | 停止后进入阻尼状态 — damping after stop |

Two command paths, isolated by A/B on wrist_roll:

- **Sync path drives from any stop mode.** `send_sync_multiturnanglebyinterval(14, n, [...])`
  where each entry is `struct.pack("<BlLHHH", id, angle*10, interval_ms, t_acc, t_dec, power)`.
  Measured +14.90 deg of +15.0 commanded with stop mode still `0x10`.
- **Single path is silently ignored unless `0x11` was set first.**
  `set_servo_angle(id, angle, is_mturn=True, interval=..., power=...)` moved +14.70 of +15.0
  after `stop_on_control_mode(id, 0x11, power)`, and 0.00 without it.

The payloads are identical (`<BiIHHH` under command 14 either way), so this is a servo-state
gate, not a packet-format difference. **Use the sync path** — it is the one that works
unconditionally, and it commands all joints in one frame, which is what a shadow loop wants.

### 1.3 Two traps

**Current telemetry is useless as a drive indicator.** `current`/`power` read 30 mA / 374 mW
idle, while holding the arm's weight, and while moving (peak 33 mA). Several wrong conclusions
during bring-up came from watching that field. Verify actuation by measured angle change or by
hand — never by current.

**`motorbridge_smart_servo` cannot drive this arm.** Its `set_angle` exposes no `power` field,
it has no sync-with-power call, and the PyO3 `_native.abi3.so` offers no raw-packet escape.
Driving requires `fashionstar-uart-sdk` (installed 1.3.12 into `LeRobot/.venv` 2026-09-02).

### 1.4 Load

The leader carries only its own weight — no payload, ever. The gravity-margin question that
would apply to a follower does not apply here; ids 1 and 5 tracked commanded targets with the
arm raised. A 15 deg move over a 3000 ms interval vibrates audibly (stick-slip at ~5 deg/s);
the vendor uses `interval=100` and re-sends every loop. Match that cadence, do not use long
intervals.

---

## 2. The capability contract in this repo

Four members, duck-typed at [`dagger.py:180`](../src/lerobot/rollout/strategies/dagger.py):

```python
def _teleop_supports_feedback(teleop):
    return (bool(teleop.feedback_features)
            and hasattr(teleop, "disable_torque")
            and hasattr(teleop, "enable_torque"))
```

Reference implementation: [`so_leader.py:337`](../src/lerobot/teleoperators/so_leader/so_leader.py),
including an `is_intervening` guard that suppresses feedback while the operator holds the arm.

### 2.1 What already calls it

| Site | Behaviour |
|---|---|
| [`inference_utils.py:970`](../src/lerobot/rl/inference_utils.py) | Every non-intervention rollout step: builds `{k: float(v) for k in obs if k.endswith(".pos")}` and calls `send_feedback` |
| [`rtc_actor_runtime.py:891`](../src/lerobot/rl/rtc_actor_runtime.py) | Same, in the RTC actor |
| [`dagger.py:766`](../src/lerobot/rollout/strategies/dagger.py) | On AUTONOMOUS → PAUSED, interpolates the leader onto the follower's pose over 2 s so handover has no jerk. Falls back to dragging the *follower* to the leader when the teleop is not actuated |

**Feature A is therefore already written.** The only reason nothing moves is that
`RebotArm102Leader.send_feedback` raises `NotImplementedError`.

### 2.2 Latent crash

Both rollout call sites invoke `send_feedback` unguarded. `config_rl.yaml` sets
`control_mode: leader` with `type: rebot_102_leader`, so the first non-intervention step of an
online run raises and kills the actor. It has never fired because online runs are still blocked
on `fixed_reset_joint_positions` (`config_rl.yaml:665`). Any option chosen below must close this.

---

## 3. Scope

Three features. A and B share the driver migration; **C is independent of both and can be built first.**

**A — Shadow mirroring.** Leader continuously tracks the follower during autonomous rollout, so
grabbing it is always possible. Plumbing exists; needs the driver only.

**B — Grab-to-takeover.** Deflection past a threshold releases torque and switches control to the
leader. Vendor's tuned threshold is a summed absolute error of 5 deg across joints 0-5; the
button on id 7 is a second, explicit trigger.

**C — Policy preview on the leader.** The follower is *not* commanded; the leader plays an
untested checkpoint's action chunk so it can be watched on the cheap arm first. A standalone
script talking to the vendor SDK directly — no driver changes, no rollout-path changes. See §5.

---

## 4. Design decisions

### 4.1 One library owns the port

A single process cannot read through `motorbridge_smart_servo` and write through
`fashionstar_uart_sdk`: even where Linux permits the double open, interleaving two drivers on a
half-duplex bus corrupts responses. Since driving *requires* the vendor SDK, the read path moves
to it as well. `_read_raw_positions` swaps `bus.sync_monitor(ids)` for
`send_sync_servo_monitor(ids)` + `servos[i].angle_monitor`; unwrap and clamp logic is untouched.

Blast radius is small: `motorbridge_smart_servo` appears in exactly two files,
`rebot_102_leader.py` and `utils/import_utils.py`.

### 4.2 Variant flag, defaulting to LD

Both arms speak the same protocol; the LD simply ignores drive commands. Without a
discriminator, a migrated driver would expose `enable_torque` on an LD, `_teleop_supports_feedback`
would return `True`, and DAgger would take the actuated branch and silently fail to move the arm
instead of falling back to moving the follower. `dagger.py:764` already carries a comment about
exactly this failure shape.

Add `variant: Literal["102LD", "102HD"] = "102LD"` to `RebotArm102LeaderConfig`, mirroring the
vendor's own `--leader_type`. The four actuated members are exposed only when `variant == "102HD"`.
Default LD means the existing arm and every existing config behave exactly as they do today.

### 4.3 Mapping algebra

`get_action` currently computes, per joint, with $d$ = `joint_directions`, $[r_{\min}, r_{\max}]$ = `joint_ranges`:

$$p = \mathrm{clamp}(\mathrm{unwrap}(a_{\text{raw}}) \cdot d,\; r_{\min},\; r_{\max})$$

`send_feedback` is its inverse:

$$a_{\text{raw}} = \mathrm{clamp}(p,\; r_{\min},\; r_{\max}) \; / \; d$$

Here $a_\text{raw}$ is the servo's own reported angle in degrees, $p$ the LeRobot joint position
in follower convention, $d$ the per-joint sign (and, for the gripper, the travel scale $-6$).
The clamp happens **before** the division so a policy that commands past the leader's mechanical
range is bounded, not wrapped. Gripper check: $p = -270$ (open) gives $a_\text{raw} = 45$, which
is the ~45 deg of raw travel the leader gripper actually has.

**Prerequisite fix.** `_round_to_valid_range` builds its unwrap window as `range * sign` instead
of `range / direction`. For the gripper ($d = -6$) that centres the window on 135 deg instead of
22.5. Latent today; it bites as soon as the gripper's raw origin shifts, and driving the gripper
makes that far more likely. Fix it in the same change as `send_feedback`, with a test on the
gripper's constants.

### 4.4 Safety invariants

1. **Unload on every exit path.** `stop_on_control_mode(0xFF, 0x10, 0x00)` in `disconnect`, in
   `__del__`, and in an exception handler around the shadow loop. A crashed actor must not leave
   a rigid arm.
2. **Watchdog.** If no `send_feedback` arrives for N ms (start at 500), unload. This covers the
   actor hanging rather than crashing.
3. **Clamp before send**, per §4.3.
4. **Never torque on connect.** `configure()` unlocks all servos today; keep that. Torque is
   enabled only by an explicit `enable_torque`.
5. **Suppress feedback while intervening**, as `so_leader` does — otherwise the arm fights the
   operator's hand.

---

## 5. Feature C: policy preview (safety)

**Purpose: watch a new checkpoint's actions play out on the cheap arm before letting it touch the
follower.** Confirmed 2026-09-02: a static visual observation and a static follower state are
*acceptable and intended*. This is not an evaluation and makes no claim to be one — it is a
pre-flight look at what the policy wants to do.

### 5.1 Shape

Open-loop chunk playback:

1. Capture one observation (cameras + follower state read, never commanded).
2. Run the policy once. `chunk_size` is 30.
3. Map each action in the chunk to leader raw angles (§4.3) and play the whole chunk on the
   leader at the configured control rate.
4. Watch. Repeat on demand.

Play the **whole chunk**, not one action per inference step. Because the observation is frozen,
re-inferring each step returns approximately the same chunk, so a step-wise loop would command
the same first action repeatedly and the arm would stall a few degrees from where it started —
showing nothing. The full chunk is the trajectory worth seeing.

This is the physical counterpart of the existing 3D action-trace probe: same open-loop
pre-flight intent, executed on hardware instead of drawn.

### 5.2 It needs no changes to the rollout path

Earlier framing of this plan assumed a `send_action` suppression flag in
[`gym_manipulator.py:285`](../src/lerobot/rl/gym_manipulator.py). **Not required.** A standalone
script that never constructs the follower robot never calls `send_action` — there is nothing to
suppress. The follower stays powered and connected only so its state can be read for the
observation; it is never commanded.

Consequently feature C is **independent of the driver migration** (§4.1, stages 1-3). The script
owns `/dev/ttyUSB0` and talks to the leader through `fashionstar_uart_sdk` directly, exactly as
`probe_hd_vendor.py` already does. It needs only the mapping algebra of §4.3, importing
`joint_directions` / `joint_ranges` from `RebotArm102LeaderConfig` rather than duplicating them.

Per the standing preference for inference-side scripts over pipeline-config fields: standalone
script plus a config subclass, no new fields on `TrainRLServerPipelineConfig`.

### 5.3 Safety rules specific to preview

The arm executes a policy that has never been validated. On top of §4.4:

- **Clamp every action to the leader's `joint_ranges` before sending**, and log any clamp — a
  clamp is itself a finding about the checkpoint.
- **Rate-limit per step.** Reject any inter-step jump beyond a configured degrees-per-step
  ceiling rather than commanding it; a wild action should stop playback, not be executed fast.
- **Abort key.** Playback must be interruptible mid-chunk, unloading on the way out.
- Clear space around the arm before playback. The leader has no collision awareness.

---

## 6. Staged plan

**Two independent tracks.** Track C carries no risk to data collection and can start immediately;
track A/B rewrites a working read path and is gated behind a verification step.

### Track C — safety preview (independent, start here)

| Stage | Change | Verification |
|---|---|---|
| **C1** | Mapping helper: LeRobot joint position -> leader raw angle (§4.3), importing the config's `joint_directions` / `joint_ranges` | Unit test the round-trip against `get_action` for all 7 joints, gripper included |
| **C2** | Playback script: vendor SDK, sync path, clamp + rate-limit + abort key, unload on every exit | Play a hand-written chunk (e.g. a slow sine on one joint), confirm the arm tracks and the abort key works mid-motion |
| **C3** | Wire the policy: capture observation, infer once, play the chunk | Compare the played trajectory against the 3D action-trace probe's output for the same observation |

### Track A/B — driver actuation

| Stage | Change | Verification | Rollback |
|---|---|---|---|
| **0** | Baseline capture: record one short episode on the current driver; keep the parquet | — | — |
| **1** | Add `variant` field, default `"102LD"`. No behaviour change | Existing teleop session unchanged | Delete the field |
| **2** | Port `_read_raw_positions` to `fashionstar_uart_sdk`; drop the `motorbridge` import | **Replay stage 0's motion and diff joint traces against the baseline.** Angles must agree within encoder noise. Then one full recording session | Revert one file |
| **3** | Add `feedback_features`, `send_feedback`, `enable_torque`, `disable_torque`, gated on `variant == "102HD"`. Fix `_round_to_valid_range` (§4.3). Add watchdog + unload-on-exit | Unit test the mapping round-trip; drive each of the 7 joints to a known target and measure | Members are additive; gate off |
| **4** | Feature A. No code changes outside the driver — verify the existing call sites drive the arm | Teleoperate the follower by hand, confirm the leader tracks; then an autonomous segment | Set `variant` back to LD |
| **5** | Feature B: deflection detection (start at the vendor's 5 deg summed threshold) + id 7 button as explicit trigger | Grab the arm mid-shadow; confirm release is prompt and handover has no jerk | Disable the trigger |

**Stage 2 is the risk in this plan.** It rewrites the read path all data collection depends on,
and recording failures are quiet — the follower already writes `0.0` into `observation.state` on
a dropout rather than raising. Do not proceed past stage 2 until a recorded episode has been
diffed against the stage 0 baseline. Track C never touches this path.

The latent crash of §2.2 is closed by stage 3 (a real `send_feedback` on HD) but remains open for
the LD arm until then; if an online run is attempted before stage 3, guard the two call sites.

---

## 7. Open questions

1. **Shadow rate.** The rollout loop calls `send_feedback` every step. What that rate is in
   practice, and whether the sync packet at that cadence stays smooth, is unmeasured. Vendor
   hover runs at ~1 kHz loop with `interval=100`.
2. **Gripper in shadow.** The vendor never locks or drives the gripper (their loop covers ids 0-5
   only). Whether to mirror it — and whether the $-6$ scale behaves under continuous drive —
   is untested.
3. **Both arms at once.** If the LD stays connected for recording while the HD shadows, they are
   two ports (`ttyUSB0`, `ttyUSB1`) and two driver instances. Not required by anything above,
   but it is the configuration the vendor's own teleop scripts assume.
