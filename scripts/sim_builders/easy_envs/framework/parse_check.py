"""Parse config_rl_sim.yaml + CLI overrides through lerobot's parser exactly as inference_async does; print the policy task fields."""
import lerobot.rl.gym_manipulator  # noqa: F401
import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig) -> None:
    print("task:", cfg.policy.task)
    print("eval_subtasks:", cfg.policy.eval_subtasks)
    print("precisions:", cfg.policy.eval_subtask_precisions, "contacts:", cfg.policy.eval_subtask_contacts)
    print("scene:", cfg.env.robot.scene, "episodes:", cfg.env.robot.episodes, "control_time_s:", cfg.env.processor.reset.control_time_s)


main()
