"""The prompt metadata the console sent on the robot: quality/mistake/speed constants plus the step's precision
and contact (policy.eval_subtask_precisions / eval_subtask_contacts). The stock probes send only the constants."""

from lerobot.annotation.vocab import code_for
from lerobot.probes.utils import DEPLOYMENT_METADATA


def deployed_metadata(cfg, subtask: str) -> dict:
    if subtask not in cfg.policy.eval_subtasks:  # a dataset subtask outside the robot script: constants only
        return dict(DEPLOYMENT_METADATA)
    step = list(cfg.policy.eval_subtasks).index(subtask)
    return {**DEPLOYMENT_METADATA, "precision": int(cfg.policy.eval_subtask_precisions[step]),
            "contact": code_for(cfg.policy.eval_subtask_contacts[step])}
