"""Action-trace probe on the episode's sampled frames with the deployed prompt metadata per step (the stock probe
sends only quality/mistake/speed). Anchors come from anchors.txt (anchors.py). Everything else is the stock probe:
action_trace_records.npz holds state, the executed chunk (the operator's during an intervention) and every flow sample.

    uv run python lerobot/adhocs/intervention_episode_reps/trace_episode.py --config config_rl.yaml \
        --dataset.root outputs/rebot_rollouts_2026-10-20-probe-v1 \
        --probe_parameters.output_dir migration/inference_review_2026-10-20/trace_probe \
        --probe_parameters.trace_n_samples 8 \
        --probe_parameters.trace_frames "$(cat lerobot/adhocs/intervention_episode_reps/anchors.txt)"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deployed_prompt import deployed_metadata  # noqa: E402

from lerobot.probes import action_trace_probe as A  # noqa: E402
from lerobot.probes.utils import probe_frame_inputs  # noqa: E402


def _inputs(dataset, cfg, global_idx, chunk_size):
    fr = probe_frame_inputs(dataset, cfg, global_idx, chunk_size, with_gripper_event_targets=False)
    fr["metadata"] = deployed_metadata(cfg, fr["subtask"])
    return fr


if __name__ == "__main__":
    A.probe_frame_inputs = _inputs
    A.register_config_choices()
    A.main()
