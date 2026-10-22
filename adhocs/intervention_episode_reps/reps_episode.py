"""Representations probe on one robot episode only (the probe root), every frame at repr_stride_s, with the deployed
prompt metadata per step. The stock probe would also sample every training source in config_rl.yaml; this wrapper
restricts the sampling plan to --dataset.root (and REPS_EPISODES="1,4" to those episodes). Everything else is the stock probe (trajectories.html, temporal.csv,
cache/<group>.npy).

    uv run python lerobot/adhocs/intervention_episode_reps/reps_episode.py --config config_rl.yaml \
        --dataset.root outputs/rebot_rollouts_2026-10-20-probe-v1 \
        --probe_parameters.output_dir migration/inference_review_2026-10-20/reps_probe \
        --probe_parameters.repr_stride_s 0.5 --probe_parameters.repr_constant_text false \
        --probe_parameters.repr_max_frames 600 --probe_parameters.repr_layers 8,14,20,28,32 \
        --probe_parameters.repr_groups action,img_external_0,img_wrist_0,state,subtask,action_output
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deployed_prompt import deployed_metadata  # noqa: E402

import lerobot.probes.representations as R  # noqa: E402
from lerobot.probes.conditions_matrix import _rebot_inputs as _stock_inputs  # noqa: E402


def _inputs(dataset, cfg, sample, chunk_size):
    if os.environ.get("REPS_NO_DEPTH"):  # prompt without the wrist depth (the pointmap encoder is untrained at step 0)
        from lerobot.probes.utils import probe_frame_inputs
        frame = probe_frame_inputs(dataset, cfg, sample["index"], chunk_size, with_depth=False, with_gripper_event_targets=False)
        r = {"obs": frame["obs"], "task": frame["task"], "subtask": frame["subtask"], "metadata": {},
             "extra": {"embodiment_index": sample["embodiment_index"]}}
    else:
        r = _stock_inputs(dataset, cfg, sample, chunk_size)
    r["metadata"] = deployed_metadata(cfg, r["subtask"])
    return r


def _sampling_plan(dataset, cfg):
    p = cfg.probe_parameters
    root = str(Path(dataset.root).resolve())
    candidates = R._lerobot_candidates(dataset, cfg, root, False)
    wanted = os.environ.get("REPS_EPISODES")  # "1,4": only these episode indices of the root
    if wanted:
        keep = {f"{root}::{int(e)}" for e in wanted.split(",")}
        candidates = [frames for frames in candidates if frames[0]["episode"] in keep]
    samples = R.select_sequences(candidates, p.max_episodes, p.repr_max_frames, p.random_seed,
                                 p.repr_min_frames_per_episode, p.repr_holdout_episodes)
    return samples, {root: dataset}, {}


if __name__ == "__main__":
    R._rebot_inputs = _inputs
    R.sampling_plan = _sampling_plan
    R.main()
