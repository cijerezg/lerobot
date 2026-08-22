from __future__ import annotations

from types import SimpleNamespace

import pytest

from lerobot.common.aim_utils import AimLogger
from lerobot.configs.default import AimConfig
from lerobot.configs.train import _migrate_legacy_wandb_config


class _FakeRun:
    def __init__(self) -> None:
        self.tracked = []
        self.params = {}
        self.close_calls = 0

    def track(self, value, *, name: str, step: int) -> None:
        self.tracked.append((value, name, step))

    def __setitem__(self, key, value) -> None:
        self.params[key] = value

    def close(self) -> None:
        self.close_calls += 1


class _FakeDistribution:
    def __init__(self, distribution, *, bin_count: int) -> None:
        self.distribution = distribution
        self.bin_count = bin_count


def _logger() -> AimLogger:
    logger = AimLogger.__new__(AimLogger)
    logger.cfg = SimpleNamespace(histogram_bins=32)
    logger._run = _FakeRun()
    logger._distribution_type = _FakeDistribution
    logger._closed = False
    return logger


def test_log_dict_preserves_metric_names_mode_and_step() -> None:
    logger = _logger()

    logger.log_dict({"loss": 1.25, "learning_rate": 3e-4}, step=17, mode="train")

    assert logger._run.tracked == [
        (1.25, "train/loss", 17),
        (3e-4, "train/learning_rate", 17),
    ]


def test_log_dict_uses_and_logs_custom_optimizer_step() -> None:
    logger = _logger()

    logger.log_dict(
        {"loss_flow": 0.2, "Optimization step": 41},
        mode="train",
        custom_step_key="Optimization step",
    )

    assert logger._run.tracked == [
        (0.2, "train/loss_flow", 41),
        (41, "train/Optimization step", 41),
    ]


def test_log_dict_preserves_latest_string_as_run_metadata() -> None:
    logger = _logger()

    logger.log_dict({"phase": "warmup"}, step=3, mode="eval")

    assert logger._run.params[("latest", "eval", "phase")] == "warmup"


def test_log_distribution_preserves_name_step_and_bin_budget() -> None:
    logger = _logger()
    values = [0.1, 0.2, 0.3]

    logger.log_distribution("actor_loss_histogram", values, step=23)

    distribution, name, step = logger._run.tracked[0]
    assert isinstance(distribution, _FakeDistribution)
    assert distribution.distribution is values
    assert distribution.bin_count == 32
    assert name == "train/actor_loss_histogram"
    assert step == 23


def test_close_is_idempotent() -> None:
    logger = _logger()

    logger.close()
    logger.close()

    assert logger._run.close_calls == 1


@pytest.mark.parametrize("bins", [0, 513])
def test_aim_config_rejects_invalid_histogram_bin_counts(bins: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 512"):
        AimConfig(histogram_bins=bins)


def test_legacy_wandb_config_migrates_without_losing_run_grouping() -> None:
    config = {
        "wandb": {
            "enable": True,
            "project": "rebot-online",
            "offline_project": "rebot-offline",
            "notes": "baseline",
            "add_tags": False,
            "disable_artifact": True,
        },
        "steps": 100,
    }

    migrated = _migrate_legacy_wandb_config(config)

    assert migrated == {
        "aim": {
            "enable": True,
            "experiment": "rebot-online",
            "offline_experiment": "rebot-offline",
            "notes": "baseline",
            "add_tags": False,
        },
        "steps": 100,
    }
    assert "wandb" in config
