# reBot B601-DM robot description

Vendored from [Seeed-Projects/reBotArm_control_py](https://github.com/Seeed-Projects/reBotArm_control_py)
(`urdf/reBot-DevArm_fixend_description/`, fetched 2026-07-27). That repo carries **no
LICENSE file**; the parent [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm)
repo does. Resolve licensing before redistributing these assets.

## Why the `fixend` variant

`reBotArm_control_py` ships two descriptions. `config/rebotarm_dm.yaml` — the config for
the Damiao-motor B601-DM, which is our arm — selects `reBot-DevArm_fixend.urdf` with
`end_effector_frame: end_link`. The other (`00-arm-rs_asm-v3`) is the RS variant and has
**joint2/joint3 limits mirrored to [0, +π]**; our dataset's `shoulder_lift ∈ [-170°, 0°]`
and `elbow_flex ∈ [-196°, 1°]` fall entirely outside that range, so the RS URDF is the
wrong one and would silently produce mirrored traces.

## Joint mapping (verified, not assumed)

Identity, in degrees, no sign flips or offsets:

| dataset dim | name             | URDF joint | URDF limit (deg) | observed range (deg) |
|-------------|------------------|------------|------------------|----------------------|
| 0           | `shoulder_pan`   | `joint1`   | ±160.4           | [-63.5, 60.5]        |
| 1           | `shoulder_lift`  | `joint2`   | [-180, 0]        | [-170.0, 0.0]        |
| 2           | `elbow_flex`     | `joint3`   | [-180, 0]        | [-196.4, 1.0]        |
| 3           | `wrist_flex`     | `joint4`   | [-107, 90]       | [-13.2, 88.6]        |
| 4           | `wrist_yaw`      | `joint5`   | ±90              | [-24.0, 33.9]        |
| 5           | `wrist_roll`     | `joint6`   | ±180             | [-66.2, 49.0]        |
| 6           | `gripper`        | —          | —                | [-238.8, 0.0]        |

Observed ranges are `meta/stats.json` of `rebot-socks-annotated-v2`. `elbow_flex` exceeds
the URDF's -180° limit (the robot's own soft limit is -200°); FK does not clamp, so this
is harmless, but IK against this URDF would need the limit widened.

The gripper has no joint here — `end_joint` is fixed and the description terminates at a
rigid `end_link`. Gripper state is carried as a scalar channel, not as geometry.

## Files

- `reBot-DevArm_fixend.urdf` — upstream, unmodified. Its `meshes/*.STL` references are
  **not** vendored (25 MB); only `pin.buildModelFromUrdf` (kinematics, no geometry) is
  used, which ignores them.
- `link_hulls.npz` — per-link convex hulls of those STLs, `<link>.v` float32 `(V,3)`
  vertices in link frame and `<link>.f` int32 `(F,3)` faces. The hull's minimum z equals
  the mesh's exactly (the lowest vertex is always a hull vertex), so clearance numbers
  are not approximated; only the rendered silhouette is slightly inflated, which is the
  conservative direction for a collision check.
