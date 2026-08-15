#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Small, policy-agnostic distributed runtime for RL trainers."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from accelerate import Accelerator


class TrainingRuntime:
    """Hide Accelerate mechanics behind the generic Trainer seam.

    ``accelerator=None`` deliberately provides a complete single-process
    implementation. This keeps direct Trainer calls and CPU unit tests from
    needing to construct a process group (or even import Accelerate).
    """

    def __init__(
        self,
        accelerator: Accelerator | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self._accelerator = accelerator
        self._device = torch.device(device or "cpu")

    @property
    def device(self) -> torch.device:
        if self._accelerator is not None:
            return self._accelerator.device
        return self._device

    @property
    def is_main_process(self) -> bool:
        return self._accelerator is None or self._accelerator.is_main_process

    @property
    def num_processes(self) -> int:
        return 1 if self._accelerator is None else self._accelerator.num_processes

    @property
    def process_index(self) -> int:
        return 0 if self._accelerator is None else self._accelerator.process_index

    def unwrap_model(self, policy: nn.Module) -> nn.Module:
        if self._accelerator is None:
            return policy
        return self._accelerator.unwrap_model(policy, keep_fp32_wrapper=True)

    def prepare_model_and_optimizers(
        self,
        policy: nn.Module,
        optimizers: Mapping[str, torch.optim.Optimizer],
    ) -> tuple[nn.Module, dict[str, torch.optim.Optimizer]]:
        """Prepare a model and named optimizers without losing optimizer keys."""
        names = list(optimizers)
        if self._accelerator is None:
            return policy, dict(optimizers)

        prepared = self._accelerator.prepare(policy, *(optimizers[name] for name in names))
        if not isinstance(prepared, tuple):
            prepared = (prepared,)
        prepared_policy = prepared[0]
        prepared_optimizers = dict(zip(names, prepared[1:], strict=True))
        return prepared_policy, prepared_optimizers

    def backward(self, loss: torch.Tensor) -> None:
        if self._accelerator is None:
            loss.backward()
        else:
            self._accelerator.backward(loss)

    def clip_grad_norm_(
        self,
        parameters: Iterable[torch.Tensor],
        max_norm: float,
    ) -> torch.Tensor:
        parameters = list(parameters)
        if self._accelerator is None:
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        return self._accelerator.clip_grad_norm_(parameters, max_norm)

    def no_sync(
        self,
        policy: nn.Module,
        accumulation_index: int,
        gradient_accumulation_steps: int,
    ) -> contextlib.AbstractContextManager[None]:
        """Skip DDP gradient synchronization before the final microbatch."""
        is_final = accumulation_index == gradient_accumulation_steps - 1
        if self._accelerator is None or self.num_processes == 1 or is_final:
            return contextlib.nullcontext()
        return self._accelerator.no_sync(policy)

    def wait_for_everyone(self) -> None:
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()

    def any_process(self, value: bool) -> bool:
        if self.num_processes == 1:
            return bool(value)
        flag = torch.tensor(int(value), device=self.device)
        assert self._accelerator is not None
        return bool(self._accelerator.reduce(flag, reduction="sum").item() > 0)

    def reduce_scalar(self, value: float, *, reduction: str = "mean") -> float:
        if self.num_processes == 1:
            return float(value)
        scalar = torch.tensor([float(value)], dtype=torch.float64, device=self.device)
        assert self._accelerator is not None
        if reduction == "max":
            return float(self._accelerator.gather(scalar).max().item())
        return float(self._accelerator.reduce(scalar, reduction=reduction).item())

    @staticmethod
    def _metric_descriptor(value: Any) -> tuple[str, str, tuple[int, ...] | None]:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return ("scalar", "tensor", None)
            return ("array", "tensor", tuple(value.shape))
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return ("scalar", "numpy_scalar", None)
            return ("array", "numpy", tuple(value.shape))
        if isinstance(value, np.number):
            return ("scalar", "numpy_scalar", None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return ("scalar", type(value).__name__, None)
        return ("metadata", type(value).__name__, None)

    def _gather_object(self, value: Any) -> list[Any]:
        if self.num_processes == 1:
            return [value]
        # Accelerate flattens one iterable from every rank, so wrap each object
        # in a one-element list to preserve dictionaries as dictionaries.
        wrapped = [value]
        if self._accelerator is not None and hasattr(self._accelerator, "gather_object"):
            return self._accelerator.gather_object(wrapped)
        from accelerate.utils import gather_object

        return gather_object(wrapped)

    def gather_equal(self, value: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Gather equal-shaped histogram tensors by concatenating dimension zero."""
        if self.num_processes == 1:
            return value
        is_numpy = isinstance(value, np.ndarray)
        tensor = torch.as_tensor(value, device=self.device) if is_numpy else value.to(self.device)
        assert self._accelerator is not None
        gathered = self._accelerator.gather(tensor.contiguous())
        if is_numpy:
            return gathered.detach().cpu().numpy()
        return gathered

    def reduce_metrics(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Mean scalars and concatenate equal-shaped histogram values across ranks.

        Rank-local scalar keys are allowed: their mean is computed over ranks that
        emitted the key. Histogram keys must exist with the same shape everywhere,
        since padding would change their statistical meaning.
        """
        if self.num_processes == 1:
            return dict(metrics)

        local_schema = {key: self._metric_descriptor(value) for key, value in metrics.items()}
        schemas = self._gather_object(local_schema)
        all_keys = sorted({key for schema in schemas for key in schema})
        reduced: dict[str, Any] = {}

        for key in all_keys:
            descriptors = [schema[key] for schema in schemas if key in schema]
            categories = {descriptor[0] for descriptor in descriptors}
            if len(categories) != 1:
                raise TypeError(f"Metric {key!r} has incompatible rank-local types: {descriptors}")
            category = descriptors[0][0]

            if key == "Optimization step":
                if key in metrics:
                    value = metrics[key]
                    reduced[key] = int(value.item() if isinstance(value, torch.Tensor) else value)
                continue

            if category == "scalar":
                present = key in metrics
                local_value = metrics.get(key, 0.0)
                if isinstance(local_value, (torch.Tensor, np.ndarray)):
                    local_value = local_value.item()
                pair = torch.tensor(
                    [float(local_value), float(present)], dtype=torch.float64, device=self.device
                )
                assert self._accelerator is not None
                total, count = self._accelerator.reduce(pair, reduction="sum")
                mean = total / count.clamp_min(1.0)
                if present and isinstance(metrics[key], torch.Tensor):
                    original = metrics[key]
                    reduced[key] = mean.to(dtype=original.dtype).reshape(original.shape)
                elif present and isinstance(metrics[key], np.generic):
                    reduced[key] = type(metrics[key])(mean.item())
                else:
                    reduced[key] = float(mean.item())
                continue

            if category == "array":
                shapes = {descriptor[2] for descriptor in descriptors}
                if len(descriptors) != self.num_processes or len(shapes) != 1:
                    raise ValueError(
                        f"Histogram metric {key!r} must have the same shape on every rank; got {descriptors}"
                    )
                reduced[key] = self.gather_equal(metrics[key])
                continue

            # Metadata is intentionally rank-local. Only the main-process value
            # can reach the sole logger.
            if self.is_main_process and key in metrics:
                reduced[key] = metrics[key]

        return reduced
