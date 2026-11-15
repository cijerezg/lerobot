# adhocs

One-off analyses that answered a real question and are worth running again. One folder per analysis, each with
a README that states the question, how to run it from the workspace root, and what it found. Scripts take the
workspace layout for granted (`config_rl.yaml`, `outputs/`), like `CHEAT_SHEET.md`. Nothing here is on the
training or probe path; the probe suite does not import it.

| folder | question | date |
|---|---|---|
| `joint_onset_trace/` | does the model know a joint move (wrist roll at the clipper grasp), and does it apply it on the robot? per-checkpoint | 2026-10-17 |
| `intervention_episode_reps/` | along one robot episode, how do representations and the model's intended chunk change across an off phase, an operator intervention and the recovery? representations + action-trace probes on the episode with the deployed prompt | 2026-10-20 |
| `object_coverage/` | where on the table, and at what angle, have the clippers, headphone case, bit kit and bits been seen in the internal data? grid readout of one top frame per grasp segment by vision agents, coverage plots + empty cells | 2026-10-26 |
| `camera_preview/` | what do the wrist and top cameras see right now, without connecting the follower or the leader? live Rerun stream from the same camera classes | 2026-10-28 |
| `torque_viewer/` | does the motor torque show contact events the joint positions and cameras do not? torque vs state, action and videos for one inference episode | 2026-11-12 |
| `motor_viewer/` | what do motor velocity and torque look like along a recorded episode, and does the gripper torque show the grasp of a small object? position / velocity / torque per joint, synced to the videos | 2026-11-14 |
| `torque_levels/` | does the smoothed gripper torque say whether the gripper holds something (low / high)? latch rule (torque after a close, until the gripper opens) or threshold with a hysteresis margin, set in the page, synced to the videos | 2026-11-14 |
| `tip_offset/` | can a per-frame vector from the gripper tip to the next grasp / release point be built from the recording alone (forward kinematics + subtask labels), as the target of a depth adapter? targets table + check sheet | 2026-11-14 |
| `loss_share/` | does one data group take more than its share of the flow loss and of the action expert's gradient? same frames across the checkpoints of a run | 2026-11-12 |
| `grad_drift/` | how fast does the gradient of a fixed set of frames change while the model trains? per data group: cosine to the start gradient, norm, and first-order loss prediction after 1, 2, 4, ... training steps | 2026-11-12 |
| `quality_sweep_trace/` | what does the quality clause do to the commanded path? one frame, one flow-noise draw, quality asked at 1..5, drawn in the Action Inspector's 3D scene at approach / near grasp / move / near release | 2026-11-13 |
| `motor_gains/` | what control-loop gains do the follower motors run, and can the slow shoulder / elbow response be shortened? read-only dump of the Damiao loop registers | 2026-11-13 |
| `grasp_reference/` | how do the good human demonstrations grasp each object? tool pose, arrival, gripper angles and approach speed per object from the annotated internal roots (the object table of the correction rubric) | 2026-11-09 |
| `sim_branch/` | from frame t of a recorded sim rollout, does a hand-written corrective plan beat the policy's own next actions? deterministic restore + side-by-side video and metrics | 2026-11-09 |
| `pan2/` | can the PAN 2 video world model continue a real top-camera frame into a plausible 10 s manipulation clip? prompt builder + generation script, blind comparison key | 2026-11-11 |
