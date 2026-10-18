"""Action-trace probe on the 31 clipper frames with the deployed prompt metadata. On the robot the console sent
"grasp the large nail clipper [precision 4, contact: a top pinch]" (policy.eval_subtask_precisions/contacts, step 1);
the stock probe sends only quality/mistake/speed, so its prompt lacks those two clauses. Everything else is the stock probe.

    uv run python lerobot/adhocs/joint_onset_trace/trace_clippers.py --config config_rl.yaml \
        --dataset.root outputs/rebot_rollouts_2026-10-17-probe-v1 \
        --probe_parameters.output_dir outputs/probe_runs/<name> \
        --probe_parameters.trace_n_samples 16 \
        --probe_parameters.trace_frames "$(cat lerobot/adhocs/joint_onset_trace/reach_anchors.txt)"
"""

import functools

from lerobot.annotation.vocab import code_for
from lerobot.probes import action_trace_probe as A
from lerobot.probes.utils import DEPLOYMENT_METADATA, probe_frame_inputs

METADATA = {**DEPLOYMENT_METADATA, "precision": 4, "contact": code_for("top-pinch")}

if __name__ == "__main__":
    # The gripper-event loss targets never reach the forward; the probe root carries no labels for them.
    A.probe_frame_inputs = functools.partial(probe_frame_inputs, metadata=METADATA, with_gripper_event_targets=False)
    A.register_config_choices()
    A.main()
