# SO101PlaceCubeMarker-v1 Gripper Physics Refinement Plan

This checklist is for refining `SO101PlaceCubeMarker-v1` so the cube is grasped by contact-rich rigid-body physics instead of broad-link contact checks or marker-specific hard attachment.

Primary files:

- `src/lerobot/robots/squint_so101/sim/envs/place.py`
- `src/lerobot/robots/squint_so101/sim/envs/robot/so101.py`
- `src/lerobot/robots/squint_so101/sim/envs/robot/so101.urdf`
- `src/lerobot/robots/squint_so101/sim/envs/base_random_env.py`

## Goal

- [ ] The cube can be lifted and placed on the marker using only simulated fingertip contact.
- [ ] Side, palm, or jaw-body contact is not classified as a stable grasp.
- [ ] The gripper closes without launching or jittering the cube.
- [ ] Success is reported only after the cube was lifted, released on the marker, and is static.
- [ ] Domain randomization is added only after the nominal deterministic setup works.

## Phase 0: Baseline And Instrumentation

- [ ] Run the current `SO101PlaceCubeMarker-v1` environment with a scripted or teleop grasp attempt.
- [ ] Save a short video or rollout trace before changing physics.
- [ ] Log these values per step:
  - [ ] gripper joint position
  - [ ] item pose
  - [ ] item linear velocity
  - [ ] item height
  - [ ] `is_item_grasped`
  - [ ] `success`
  - [ ] left/right gripper contact force magnitudes
- [ ] Record the current failure mode:
  - [ ] never contacts
  - [ ] contacts with one side only
  - [ ] false grasp from palm/side contact
  - [ ] cube launches on close
  - [ ] cube slips during lift
  - [ ] cube places but does not settle

Done criteria:

- [ ] There is a repeatable baseline run to compare against later changes.
- [ ] The baseline confirms whether the marker-specific attachment path is masking physical grasp failures.

## Phase 1: Verify And Add Fingertip Collision Geometry

Current local finding: `finger1_tip` and `finger2_tip` exist in `so101.urdf`, but they are dummy fixed links with no collision geometry. Do not rely on `scene.get_pairwise_contact_forces(self.finger1_tip, obj)` until these links actually have collision bodies.

- [ ] Inspect `so101.urdf` fingertip links:
  - [ ] `finger1_tip`
  - [ ] `finger2_tip`
- [ ] Decide whether fingertip contact should be modeled by:
  - [ ] adding primitive collision geometry to `finger1_tip` and `finger2_tip`, or
  - [ ] adding dedicated fixed fingertip pad links, or
  - [ ] keeping broad collision links but creating narrower contact-specific pad links.
- [ ] Prefer simple convex geometry:
  - [ ] small box pads
  - [ ] capsules
  - [ ] convex mesh only if primitives are insufficient
- [ ] Avoid:
  - [ ] concave fingertip collisions
  - [ ] very thin collision shells
  - [ ] high-poly fingertip meshes
  - [ ] visual-only fingertip markers
- [ ] Verify in the viewer that fingertip collision pads are located at the actual contact patches.
- [ ] Verify that pairwise contact forces are nonzero when the cube touches each fingertip pad.

Suggested initial check:

```python
lcf = self.scene.get_pairwise_contact_forces(self.agent.finger1_tip, self.item)
rcf = self.scene.get_pairwise_contact_forces(self.agent.finger2_tip, self.item)
```

Done criteria:

- [ ] Each fingertip pad can independently report contact force against the cube.
- [ ] Contact force does not appear when only the palm or jaw body touches the cube.

## Phase 2: Replace Broad-Link Grasp Detection

Current local finding: `SO101.is_grasping()` checks `gripper_link` and `moving_jaw_so101_v1_link`, which can classify side or palm contact as a grasp.

- [ ] Update `SO101.is_grasping()` in `robot/so101.py`.
- [ ] Use contact-specific fingertip links after Phase 1 is complete:
  - [ ] `self.finger1_tip`
  - [ ] `self.finger2_tip`
- [ ] Require both fingertip forces above threshold.
- [ ] Require contact force direction to be plausible for the gripper closing direction.
- [ ] Require left/right contact force vectors to oppose each other.
- [ ] Keep thresholds configurable:
  - [ ] `min_force`
  - [ ] `max_angle`
  - [ ] `opposition_dot`
  - [ ] `require_opposing`
- [ ] Keep `is_touching()` broader than `is_grasping()` so non-grasp contact is still detectable.

Suggested grasp predicate shape:

```python
def is_grasping(
    self,
    object: Actor,
    min_force: float = 0.35,
    max_angle: float = 85,
    opposition_dot: float = -0.25,
    require_opposing: bool = True,
):
    left_forces = self.scene.get_pairwise_contact_forces(self.finger1_tip, object)
    right_forces = self.scene.get_pairwise_contact_forces(self.finger2_tip, object)

    left_mag = torch.linalg.norm(left_forces, dim=1)
    right_mag = torch.linalg.norm(right_forces, dim=1)

    left_dir = self.finger1_tip.pose.to_transformation_matrix()[..., :3, 1]
    right_dir = -self.finger2_tip.pose.to_transformation_matrix()[..., :3, 1]

    left_angle = common.compute_angle_between(left_dir, left_forces)
    right_angle = common.compute_angle_between(right_dir, right_forces)

    grasp = (
        (left_mag >= min_force)
        & (right_mag >= min_force)
        & (torch.rad2deg(left_angle) <= max_angle)
        & (torch.rad2deg(right_angle) <= max_angle)
    )

    if require_opposing:
        left_norm = left_forces / torch.clamp(left_mag[:, None], min=1e-6)
        right_norm = right_forces / torch.clamp(right_mag[:, None], min=1e-6)
        opposing = torch.sum(left_norm * right_norm, dim=1) < opposition_dot
        grasp = grasp & opposing

    return grasp
```

Done criteria:

- [ ] Palm-only contact returns `False`.
- [ ] One-finger contact returns `False`.
- [ ] Two-finger opposing fingertip contact returns `True`.
- [ ] A lifted cube with low slip remains classified as grasped.

## Phase 3: Remove Marker-Specific Hard Attachment

Current local finding: `place.py` has marker-specific logic in `_after_control_step()` that moves the item with the gripper after `marker_grasp_active` is set. This makes the marker task behave like an attachment benchmark instead of physical gripping.

- [ ] Remove or feature-flag the marker grasp assist in `Place._after_control_step()`.
- [ ] Remove or feature-flag:
  - [ ] `marker_grasp_active`
  - [ ] `marker_grasp_offset`
  - [ ] direct `self.item.set_pose(...)` while grasped
  - [ ] forced zero item velocity while grasped
- [ ] Remove the marker override in `evaluate()`:

```python
is_item_grasped = is_item_grasped | self.marker_grasp_active
```

- [ ] If preserving backward compatibility, introduce an explicit option such as:

```python
use_marker_grasp_assist: bool = False
```

Done criteria:

- [ ] The cube moves only because of physics contacts.
- [ ] Turning the marker assist on/off is explicit, not hidden in the default task.
- [ ] The physical-grasp path is the default for refinement work.

## Phase 4: Tune Contact Materials And Object Mass

Current local finding: item friction defaults to `0.1-0.5`, item density defaults to `200`, and SO101 gripper friction is `2.0`.

- [ ] Update `PlaceRandomizationConfig` nominal item physics:
  - [ ] item friction range: start around `0.6-1.2`
  - [ ] item density range: start around `500-1200`
  - [ ] restitution remains `0.0`
- [ ] Keep the cube deterministic while debugging:
  - [ ] fixed cube size
  - [ ] fixed friction
  - [ ] fixed density
- [ ] Revisit gripper material in `SO101.urdf_config`:
  - [ ] static friction
  - [ ] dynamic friction
  - [ ] restitution
  - [ ] `patch_radius`
  - [ ] `min_patch_radius`
- [ ] Check whether `patch_radius=0.1` is too large for a 24 mm cube.
- [ ] Avoid using extreme gripper friction to hide poor grasp geometry.

Starting values to test:

```python
item_friction_range = (0.8, 0.8)
item_density_range = (800, 800)
```

Done criteria:

- [ ] A correctly pinched cube does not slip immediately.
- [ ] A poorly pinched cube still fails instead of being held by unrealistic friction.
- [ ] The cube does not bounce noticeably during placement.

## Phase 5: Reduce Gripper Aggressiveness

Current local finding: the SO101 controller applies high stiffness, high damping, and high force limit to all six active joints, including the gripper.

- [ ] Split gripper actuator values from arm actuator values in `SO101._controller_configs`.
- [ ] Keep arm values near the current defaults initially.
- [ ] Lower only the gripper joint values first.
- [ ] Reduce gripper delta action bounds.
- [ ] Update `BaseRandomEnv._randomize_gripper_speed()` so randomized drive properties do not restore `force_limit=100`.

Suggested starting controller values:

```python
stiffness = [1e3, 1e3, 1e3, 1e3, 1e3, 150]
damping = [1e2, 1e2, 1e2, 1e2, 1e2, 20]
force_limit = [100, 100, 100, 100, 100, 10]
lower = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.05]
upper = [0.1, 0.1, 0.1, 0.1, 0.1, 0.05]
```

If the cube is ejected:

- [ ] Lower gripper `force_limit` toward `5`.
- [ ] Lower gripper delta bound toward `0.02-0.03`.
- [ ] Lower gripper stiffness.
- [ ] Increase damping only if it reduces oscillation without making contact unstable.

If the cube slips despite a good pinch:

- [ ] Slightly increase gripper force limit.
- [ ] Slightly increase fingertip friction.
- [ ] Check fingertip collision alignment before increasing reward.

Done criteria:

- [ ] Closing from open to contact is slow and stable.
- [ ] The gripper holds without behaving like an infinite-force clamp.
- [ ] The object is not launched by a close command.

## Phase 6: Increase Contact Solver Stability

Current local finding: the default sim config is `sim_freq=100` and `control_freq=10`.

- [ ] Test a higher simulation frequency:
  - [ ] `sim_freq=200`
  - [ ] `sim_freq=240`
- [ ] Test a higher control frequency:
  - [ ] `control_freq=20`
  - [ ] `control_freq=50`
- [ ] Keep one controlled experiment per change.
- [ ] Prefer stable contact over faster rollout speed while debugging.
- [ ] Document the chosen frequency pair and why it was chosen.

Done criteria:

- [ ] Contact response is stable at grasp closure.
- [ ] Increasing sim/control frequency improves or preserves grasp reliability.
- [ ] Runtime is still acceptable for the intended training setup.

## Phase 7: Add Debug Contact Diagnostics

- [ ] Add diagnostic values to `evaluate()` or `info`.
- [ ] Optionally expose them in `_get_obs_extra()` only for debug or privileged training.
- [ ] Track:
  - [ ] left fingertip force magnitude
  - [ ] right fingertip force magnitude
  - [ ] minimum two-finger force
  - [ ] force opposition dot product
  - [ ] item linear speed
  - [ ] item angular speed if available
  - [ ] `is_slipping`
  - [ ] `grasp_quality`
- [ ] Do not expose privileged contact-force signals to a real-world policy unless comparable sensing exists.

Suggested diagnostic terms:

```python
left_force = torch.linalg.norm(left_contact_forces, dim=1)
right_force = torch.linalg.norm(right_contact_forces, dim=1)
two_finger_force = torch.minimum(left_force, right_force)

left_norm = left_contact_forces / torch.clamp(left_force[:, None], min=1e-6)
right_norm = right_contact_forces / torch.clamp(right_force[:, None], min=1e-6)
force_opposition = -torch.sum(left_norm * right_norm, dim=1)

item_speed = torch.linalg.norm(self.item.linear_velocity, dim=1)
is_slipping = item_speed > 0.25
grasp_quality = torch.clamp(two_finger_force / 1.0, 0.0, 1.0) * torch.clamp(force_opposition, 0.0, 1.0)
```

Done criteria:

- [ ] Rollouts clearly show why each grasp succeeds or fails.
- [ ] False positives in `is_item_grasped` are easy to diagnose.

## Phase 8: Tighten Evaluation And Success

Current marker success already uses target height, lifted-once, no robot-item contact, and item static. Keep that structure, but verify it still works after hard attachment is removed.

- [ ] For marker target success, require:
  - [ ] item is within marker XY bounds
  - [ ] item is at marker/cube target height
  - [ ] item was lifted once
  - [ ] robot has released the item
  - [ ] item is static
- [ ] Consider adding stricter release detection:
  - [ ] no gripper/object contact, or
  - [ ] gripper qpos above release threshold, or
  - [ ] both
- [ ] Keep `is_item_static` in the success condition.
- [ ] Confirm the cube cannot receive success while falling or bouncing.

Done criteria:

- [ ] Success is impossible while the robot is still holding or pushing the cube.
- [ ] Success is impossible while the cube is moving quickly.
- [ ] Success matches visual inspection.

## Phase 9: Refine Dense Reward After Physics Works

Current local finding: reward jumps when `is_item_grasped` is true. Keep staged reward, but add a continuous contact-quality term after physical contact works.

- [ ] Keep reaching reward.
- [ ] Add continuous two-finger contact reward.
- [ ] Add opposition reward.
- [ ] Add lift reward.
- [ ] Add placement distance reward.
- [ ] Add object velocity or slip penalty.
- [ ] Add action penalty if policies learn aggressive close motions.
- [ ] Avoid making the reward compensate for broken grasp physics.

Suggested grasp quality reward:

```python
contact_reward = torch.clamp(torch.minimum(left_force, right_force) / 1.0, 0.0, 1.0)
opposition_reward = torch.clamp(force_opposition, 0.0, 1.0)
grasp_quality_reward = contact_reward * opposition_reward
reward += 1.0 * grasp_quality_reward
```

Done criteria:

- [ ] Reward increases smoothly as the gripper reaches, pinches, lifts, transports, releases, and settles.
- [ ] Reward does not report a high grasp score for side contact.
- [ ] Reward does not encourage slamming the gripper shut.

## Phase 10: Scripted Regression Tests

- [ ] Create or run a scripted policy with these stages:
  - [ ] move above cube
  - [ ] descend
  - [ ] close gripper slowly
  - [ ] lift
  - [ ] move above marker
  - [ ] descend
  - [ ] open gripper
  - [ ] retreat
- [ ] Test deterministic nominal physics first.
- [ ] Run enough episodes to see common failures, not just one success.
- [ ] Log success rate and failure classes.

Minimum acceptance tests:

- [ ] Open gripper touching cube does not count as grasp.
- [ ] One fingertip touching cube does not count as grasp.
- [ ] Two opposing fingertip contacts count as grasp.
- [ ] Hard attachment disabled still allows physical lift.
- [ ] Object does not launch on close.
- [ ] Released cube settles on marker and reports success.

## Phase 11: Domain Randomization

Add randomization only after the nominal setup is stable.

- [ ] Randomize object mass or density:
  - [ ] start with `+/-20%`
  - [ ] expand only if training remains stable
- [ ] Randomize object friction:
  - [ ] start with `+/-20%`
  - [ ] avoid extreme low/high values initially
- [ ] Randomize fingertip friction:
  - [ ] start with `+/-10%`
- [ ] Randomize gripper force limit:
  - [ ] start with `+/-10%`
- [ ] Randomize gripper stiffness/damping:
  - [ ] start with `+/-10%`
- [ ] Add small object pose perturbations.
- [ ] Add small control noise or latency only after contact behavior is reliable.

Done criteria:

- [ ] Nominal success remains high.
- [ ] Randomized training does not learn a fragile exploit.
- [ ] Failure logs still indicate real physical causes.

## Recommended Work Order

- [ ] 1. Baseline the current marker task.
- [ ] 2. Add real fingertip collision pads.
- [ ] 3. Switch `SO101.is_grasping()` to fingertip/opposition contact.
- [ ] 4. Disable marker hard attachment by default.
- [ ] 5. Tune item friction/density.
- [ ] 6. Lower gripper gains, force limit, and close delta.
- [ ] 7. Increase sim/control frequency if contact remains unstable.
- [ ] 8. Add contact diagnostics.
- [ ] 9. Tighten evaluation and success.
- [ ] 10. Refine reward.
- [ ] 11. Add domain randomization.

## Notes

- Do not model grasping as an object attachment unless the benchmark intentionally wants simplified grasp behavior.
- Treat reward as a measurement and shaping layer, not as a replacement for contact physics.
- If a reward change appears to fix grasping before fingertip contact and actuator settings are corrected, inspect whether it is hiding a physics problem.
- Keep deterministic nominal physics until the scripted grasp is reliable.
