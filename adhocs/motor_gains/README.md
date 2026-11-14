# motor_gains (2026-11-13)

Question: what control-loop gains do the follower's Damiao motors run, and can the shoulder / elbow
response (0.77 s time constant measured from the datasets, wrist 0.23 s) be shortened by raising them?

## Files

- `read_gains.py`: reads the loop registers of the seven motors (acceleration, deceleration, speed limit,
  control mode, current / speed / position loop gains). Read only: no enable, no mode change, no write.

- `step_test.py`: MOVES THE ARM. Steps one joint at each KP_APR of a list and measures the response. Gains are written without storing and the motor's own value is written back.

- `plot_recording.py`: reads a recording and writes `plots/<root name>.html`: command vs state per joint over
  time, and the speed spectrum of both (optionally against a reference recording). Prints the rms joint
  speed in the 4 to 12 Hz tremor band.

- `trace_recording.py`: the Action Inspector's 3D page for a recording, no model: per anchor frame the next
  second of the leader command (black) and the path the follower took (in the sample slot). Writes
  `plots/<root name>_trace.html`.

## Run

    uv run python lerobot/adhocs/motor_gains/trace_recording.py --root=<recording> --n=12 --episode=0

    uv run python lerobot/adhocs/motor_gains/plot_recording.py --root=<recording> --ref=<recording at the old gain>

    uv run python lerobot/adhocs/motor_gains/read_gains.py --port=/dev/ttyACM0
    uv run python lerobot/adhocs/motor_gains/step_test.py --port=/dev/ttyACM0 --joint=shoulder_pan --kps=54,108,216

The follower must be powered and nothing else may hold the port (no recording or inference running).
`step_test.py` flags: `--step_deg=10`, `--hold_s=3`, `--rate_hz=100`. It moves only the tested joint, from
the pose the arm is in at the start (pose the arm first); the automatic lift of the first version was
removed 2026-11-13 after it sent the arm somewhere the user did not expect.

## Outputs

- `gains_<date>.csv` in this folder: the values the motors had, kept as the reference to revert to.
- `step_<joint>_<date>_<time>.csv`: every sample of a step test (command, position, motor velocity, torque).

## Found

2026-11-13, register dump (`gains_2026-11-13.csv`): KP_APR = 54 and KI_APR = 0 on all seven motors, ACC = 2,
DEC = -2, speed-loop filter 40 Hz. The joints differ in gearbox, not in gain: 54 / 40 = 1.35 /s for the 4340P
joints (1.30 /s measured from the datasets), 54 / 10 = 5.4 /s for the 4310 wrist joints (3.8 to 5.0 /s
measured). The gearbox ratios are read off the model names.

2026-11-13, step tests (10 deg, 40 deg/s cap, first version of `step_test.py`), time to 63% of the step:

| joint | KP_APR 54 | 108 | 216 |
|---|---|---|---|
| shoulder_pan | 0.76 s | 0.38 s | 0.21 s |
| shoulder_lift | 0.75 s | 0.37 s | 0.20 s |
| elbow_flex | 0.76 s | 0.38 s | 0.20 s |

Overshoot at most 0.04 deg, no vibration while holding. The response time halves when the gain doubles, so
the gearbox explanation holds. Not tested: teleoperation, moves of several joints, a load in the gripper,
gains above 216.

2026-11-13, first teleoperation at 216 on the three big joints (`gain216-test-v1_20261113_142107`, one
120 s episode) against `insert-floss-v1_20261111_132721` at 54: time constant 0.21 / 0.21 / 0.22 s (was
0.76 / 0.75 / 0.77), tip gap while moving 8 / 16 / 26 mm at p50 / p90 / p99 (was 16 / 38 / 71), frames
with gap above half the 1 s chunk path 1% (was 50%). State speed in 4 to 12 Hz: pan 0.38 deg/s (was 0.36),
lift 0.61 (0.65), elbow 1.49 (1.25); different tasks, so a rough comparison.

The follower config has `position_kp` (motor name to KP_APR, written at connect, not stored) since
2026-11-13; 216 on the three big joints is on trial in teleoperation, no decision yet.
