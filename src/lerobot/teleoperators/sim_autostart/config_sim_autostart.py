from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("sim_autostart")
@dataclass
class SimAutostartTeleopConfig(TeleoperatorConfig):
    """Leader stand-in for the sim rig: no intervention, every episode starts at once."""
