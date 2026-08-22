#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import json
import logging
import sys
from numbers import Real
from pathlib import Path
from typing import Any

from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig


def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """Return stable run tags derived from the trainable, seed, dataset, and environment."""
    if cfg.is_reward_model_training:
        trainable_tag = f"reward_model:{cfg.reward_model.type}"
    else:
        trainable_tag = f"policy:{cfg.policy.type}"
    tags = [trainable_tag, f"seed:{cfg.seed}"]
    if cfg.dataset is not None:
        tags.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        tags.append(f"env:{cfg.env.type}")
    return tags if return_list else "-".join(tags)


def _local_repo_path(repo: str) -> Path | None:
    """Resolve a local Aim repo path while leaving aim:// remote URLs untouched."""
    if repo.startswith("aim://"):
        return None
    return Path(repo).expanduser().resolve()


class AimLogger:
    """Small Aim adapter for LeRobot scalar metrics and distributions.

    Automatic terminal capture, system metrics, and system-parameter collection are
    disabled deliberately. The Aim repository therefore contains only run metadata,
    the training configuration, explicitly logged metrics, and distributions.
    """

    def __init__(self, cfg: TrainPipelineConfig):
        if sys.version_info >= (3, 13):
            raise RuntimeError(
                "Aim 3.29.1 supports Python 3.12 but is not compatible with Python 3.13. "
                "Run LeRobot with Python 3.12 when Aim logging is enabled."
            )

        from aim import Distribution, Run

        self.cfg = cfg.aim
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self._group = cfg_to_group(cfg)
        self._distribution_type = Distribution
        self._closed = False

        repo = self.cfg.repo
        local_repo = _local_repo_path(repo)
        if local_repo is not None:
            local_repo.mkdir(parents=True, exist_ok=True)
            repo = str(local_repo)

        run_hash = self.cfg.run_hash if cfg.resume else None
        self._run = Run(
            repo=repo,
            experiment=self.cfg.experiment,
            run_hash=run_hash,
            system_tracking_interval=None,
            log_system_params=False,
            capture_terminal_logs=False,
        )
        self._run.name = self.job_name
        if self.cfg.notes:
            self._run.description = self.cfg.notes
        if self.cfg.add_tags and run_hash is None:
            for tag in cfg_to_group(cfg, return_list=True):
                self._run.add_tag(tag)

        # Round-trip through JSON to normalize Paths and other config leaf values
        # into Aim's supported parameter types.
        self._run["hparams"] = json.loads(json.dumps(cfg.to_dict(), default=str))
        self._run["group"] = self._group

        self.cfg.run_hash = self._run.hash
        atexit.register(self.close)

        logging.info(colored("Metrics will be stored in Aim.", "blue", attrs=["bold"]))
        if local_repo is not None:
            logging.info(
                "Aim run %s. View results with: aim up --repo %s",
                self._run.hash,
                local_repo,
            )
        else:
            logging.info("Aim run %s is tracking to %s", self._run.hash, repo)

    @property
    def run_hash(self) -> str:
        return self._run.hash

    def log_dict(
        self,
        d: dict[str, Any],
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        """Track scalar metrics using either the supplied or dictionary-provided step."""
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")
        if custom_step_key is not None:
            if custom_step_key not in d:
                raise KeyError(f'Custom step key "{custom_step_key}" is missing from metrics.')
            step = int(d[custom_step_key])
        assert step is not None

        for key, value in d.items():
            if isinstance(value, Real):
                self._run.track(value, name=f"{mode}/{key}", step=int(step))
            elif isinstance(value, str):
                # Aim metric sequences are numeric. Preserve the latest string as
                # searchable run metadata instead of silently dropping it.
                self._run["latest", mode, key] = value
            else:
                logging.warning(
                    'Aim logging of key "%s" was ignored because type "%s" is not scalar.',
                    key,
                    type(value),
                )

    def log_distribution(
        self,
        name: str,
        values: Any,
        step: int,
        mode: str = "train",
    ) -> None:
        """Track a bounded-size histogram while retaining the existing metric name."""
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        distribution = self._distribution_type(
            values,
            bin_count=self.cfg.histogram_bins,
        )
        self._run.track(distribution, name=f"{mode}/{name}", step=int(step))

    def close(self) -> None:
        """Finalize the run once so Aim can index it for comparisons."""
        if self._closed:
            return
        self._closed = True
        self._run.close()
