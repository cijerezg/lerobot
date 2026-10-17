"""Leader stand-in for the MuJoCo rig (robots/rebot_b601_sim).

The RTC env worker waits for START_EPISODE between episodes (the operator's key 2 with the real
leader) and reads the teleop's action and events on every step. Here there is no operator: no
intervention, no feedback (so the runtime never ramps a leader), and START_EPISODE stays true until
the sim robot has run its configured episodes (``STOP_EPISODES``), after which the worker idles until
the robot's SIGINT lands.
"""

from typing import Any

from lerobot.robots.rebot_b601_sim import STOP_EPISODES
from lerobot.types import RobotAction

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_sim_autostart import SimAutostartTeleopConfig


class SimAutostartTeleop(Teleoperator):
    config_class = SimAutostartTeleopConfig
    name = "sim_autostart"

    def __init__(self, config: SimAutostartTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

    @property
    def action_features(self) -> dict:
        return {}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        return {}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.START_EPISODE: not STOP_EPISODES.is_set(),
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def disconnect(self) -> None:
        self._connected = False
