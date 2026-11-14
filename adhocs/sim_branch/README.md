# sim_branch (2026-11-09)

Question: from frame t of a recorded sim rollout where the policy has gone astray, does a hand-written (or
LLM-written) corrective plan do better than the policy's own next actions? The sim is deterministic, so frame t is
restored by replaying the recorded actions from the scene reset.

## Files

- `simlab.py look <layout> <t>`: dumps what a labeller gets on the real robot at frame t (top / wrist RGB plain and
  with a pixel grid, wrist depth, joint state, camera calibration; no object poses).
- `simlab.py run <layout> <t> <plan.py> ...`: executes the plan files in order from frame t and the policy's own
  recorded actions from the same frame; writes a side-by-side video (policy | plan) per camera, `actions_plan.npy`
  and `metrics.json` (lid closed fraction, tip-to-lid distance, gripper, max joint speed).
- `motion.py`: joint targets from tool-frame waypoints (straight tip line, slerped orientation, one speed ramp).
- `p2_a_hover.py`, `p2_b_pinch.py`, `p2_c_close.py`: example plan stages for `twin_l3_close_lid_p2`. A plan file
  exposes `plan(obs, gripper) -> (N, 7)` joint targets in degrees.
- `diag.py`: debugging only, reads the sim's ground truth through the stages (hinge angle, base lift, finger contacts).

## Run

Needs a rollout recorded with `--record_inference_dataset=true` under
`outputs/llm_corrections_sim_2026-11-09/rollouts/validation/step_00000000/sim/<layout>/inference_dataset`
(`OUT` / `ROLLOUTS` at the top of `simlab.py`). The metrics are written for the close-lid layouts.

    MUJOCO_GL=egl PYTHONPATH=lerobot/src .venv/bin/python lerobot/adhocs/sim_branch/simlab.py look twin_l3_close_lid_p2 600
    MUJOCO_GL=egl PYTHONPATH=lerobot/src .venv/bin/python lerobot/adhocs/sim_branch/simlab.py run twin_l3_close_lid_p2 450 \
        lerobot/adhocs/sim_branch/p2_a_hover.py lerobot/adhocs/sim_branch/p2_b_pinch.py lerobot/adhocs/sim_branch/p2_c_close.py

## Findings

Runs of 2026-11-09 are under `outputs/llm_corrections_sim_2026-11-09/runs/` (`metrics.json` per run). No summary
was written. The real-robot correction pass that followed is `src/lerobot/annotation/corrections/`.
