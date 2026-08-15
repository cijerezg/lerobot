from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.rl.training_runtime import TrainingRuntime
from lerobot.scripts.rl_offline import (
    _build_on_each_process_main_first,
    _validate_distributed_config,
)


class _FakeAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.num_processes = 2
        self.process_index = 0
        self.is_main_process = True
        self.no_sync_entries = 0
        self.barriers = 0
        self.remote_schema = {}

    def unwrap_model(self, model, keep_fp32_wrapper=True):
        del keep_fp32_wrapper
        return getattr(model, "raw_model", model)

    def prepare(self, model, *optimizers):
        return (model, *(f"prepared-{index}" for index, _ in enumerate(optimizers)))

    @contextlib.contextmanager
    def no_sync(self, model):
        del model
        self.no_sync_entries += 1
        yield

    def wait_for_everyone(self):
        self.barriers += 1

    def reduce(self, tensor, reduction):
        if tensor.ndim == 0:
            if reduction == "sum":
                return tensor + 1
            return tensor
        if tensor.numel() == 2:
            value = float(tensor[0].item())
            remote = {
                2.0: torch.tensor([4.0, 1.0], dtype=tensor.dtype),
                10.0: torch.tensor([0.0, 0.0], dtype=tensor.dtype),
            }.get(value, torch.zeros_like(tensor))
            return tensor + remote
        return tensor

    def gather(self, tensor):
        if tensor.numel() == 1:
            return torch.cat((tensor, tensor + 1))
        return torch.cat((tensor, tensor + 2))

    def gather_object(self, wrapped):
        return [wrapped[0], self.remote_schema]

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def test_single_process_runtime_is_a_complete_noop() -> None:
    runtime = TrainingRuntime(device="cpu")
    policy = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(policy.parameters())

    prepared_policy, prepared = runtime.prepare_model_and_optimizers(
        policy,
        {"policy": optimizer},
    )
    assert prepared_policy is policy
    assert prepared == {"policy": optimizer}
    assert runtime.unwrap_model(policy) is policy
    assert runtime.is_main_process
    assert runtime.num_processes == 1
    assert runtime.process_index == 0
    assert runtime.device == torch.device("cpu")
    assert not runtime.any_process(False)
    assert runtime.reduce_scalar(3.0, reduction="max") == 3.0

    with runtime.no_sync(policy, 0, 2):
        loss = policy(torch.ones(1, 2)).sum()
        runtime.backward(loss)
    assert all(parameter.grad is not None for parameter in policy.parameters())

    metrics = {
        "loss": 1.5,
        "histogram": np.array([1.0, 2.0]),
        "Optimization step": 3,
    }
    reduced = runtime.reduce_metrics(metrics)
    assert reduced["loss"] == 1.5
    np.testing.assert_array_equal(reduced["histogram"], metrics["histogram"])
    assert reduced["Optimization step"] == 3


def test_prepare_preserves_named_optimizer_order() -> None:
    accelerator = _FakeAccelerator()
    runtime = TrainingRuntime(accelerator)
    policy = nn.Linear(2, 1)
    optimizers = {
        "policy": torch.optim.AdamW(policy.parameters()),
        "depth": torch.optim.AdamW(policy.parameters()),
    }

    prepared_policy, prepared = runtime.prepare_model_and_optimizers(policy, optimizers)

    assert prepared_policy is policy
    assert prepared == {"policy": "prepared-0", "depth": "prepared-1"}


def test_no_sync_is_used_only_before_final_microbatch() -> None:
    accelerator = _FakeAccelerator()
    runtime = TrainingRuntime(accelerator)
    policy = nn.Linear(2, 1)

    with runtime.no_sync(policy, 0, 2):
        pass
    with runtime.no_sync(policy, 1, 2):
        pass

    assert accelerator.no_sync_entries == 1


def test_distributed_metric_reduction_preserves_shapes_and_step_type() -> None:
    accelerator = _FakeAccelerator()
    runtime = TrainingRuntime(accelerator)
    remote_metrics = {
        "loss": 4.0,
        "tensor_loss": torch.tensor(4.0),
        "histogram": np.array([3.0, 4.0]),
        "Optimization step": 7,
        "tag": "remote",
    }
    accelerator.remote_schema = {
        key: runtime._metric_descriptor(value) for key, value in remote_metrics.items()
    }

    reduced = runtime.reduce_metrics(
        {
            "loss": 2.0,
            "tensor_loss": torch.tensor(2.0),
            "optional": 10.0,
            "histogram": np.array([1.0, 2.0]),
            "Optimization step": 7,
            "tag": "main",
        }
    )

    assert reduced["loss"] == pytest.approx(3.0)
    assert torch.is_tensor(reduced["tensor_loss"])
    assert reduced["tensor_loss"].shape == torch.Size([])
    assert reduced["tensor_loss"].item() == pytest.approx(3.0)
    assert reduced["optional"] == pytest.approx(10.0)
    np.testing.assert_array_equal(
        reduced["histogram"],
        np.array([1.0, 2.0, 3.0, 4.0]),
    )
    assert reduced["Optimization step"] == 7
    assert isinstance(reduced["Optimization step"], int)
    assert reduced["tag"] == "main"


def test_shutdown_and_max_elapsed_are_collective() -> None:
    runtime = TrainingRuntime(_FakeAccelerator())

    assert runtime.any_process(False)
    assert runtime.reduce_scalar(2.0, reduction="max") == pytest.approx(3.0)


def _offline_cfg(**overrides):
    policy = SimpleNamespace(
        type="molmoact2_rl",
        gradient_accumulation_steps=2,
        subtask_loss_weight=0.0,
        storage_device="cpu",
    )
    cfg = SimpleNamespace(policy=policy, batch_size=32, skip_critic=True)
    for key, value in overrides.items():
        if key.startswith("policy__"):
            setattr(policy, key.removeprefix("policy__"), value)
        else:
            setattr(cfg, key, value)
    return cfg


def test_supported_distributed_config_passes_validation() -> None:
    runtime = TrainingRuntime(_FakeAccelerator())

    _validate_distributed_config(_offline_cfg(), runtime)


def test_unsupported_distributed_config_reports_every_guard() -> None:
    runtime = TrainingRuntime(_FakeAccelerator())
    cfg = _offline_cfg(
        policy__type="pi05_rl",
        policy__subtask_loss_weight=1.0,
        policy__storage_device="cuda",
        skip_critic=False,
    )

    with pytest.raises(ValueError) as exc_info:
        _validate_distributed_config(cfg, runtime)

    message = str(exc_info.value)
    assert "policy.type" in message
    assert "skip_critic" in message
    assert "subtask_loss_weight" in message
    assert "storage_device" in message


def test_main_first_factory_has_barriers_on_both_sides() -> None:
    accelerator = _FakeAccelerator()
    runtime = TrainingRuntime(accelerator)
    calls = []

    result = _build_on_each_process_main_first(runtime, lambda: calls.append("built") or 7)

    assert result == 7
    assert calls == ["built"]
    assert accelerator.barriers == 2


def test_molmo_actor_uses_wrapped_forward_and_unwrapped_attributes() -> None:
    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer
    from lerobot.utils.constants import ACTION

    class _RawPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.depth_visual = None
            self.forward_calls = 0

        def forward(self, batch, **kwargs):
            del kwargs
            self.forward_calls += 1
            per_sample = self.scale * batch["action"].float().square().mean(dim=(1, 2))
            return per_sample, {
                "loss": per_sample.mean().item(),
                "action_flow_loss": per_sample.mean().item(),
                "loss_raw": per_sample.detach(),
                "flow_loss_per_sample": per_sample.detach(),
            }

    class _WrappedPolicy(nn.Module):
        def __init__(self, raw_model) -> None:
            super().__init__()
            self.raw_model = raw_model
            self.forward_calls = 0

        def forward(self, *args, **kwargs):
            self.forward_calls += 1
            return self.raw_model(*args, **kwargs)

    raw_policy = _RawPolicy()
    wrapped_policy = _WrappedPolicy(raw_policy)
    accelerator = _FakeAccelerator()
    runtime = TrainingRuntime(accelerator)
    optimizer = torch.optim.AdamW(raw_policy.parameters(), lr=0.1)
    disabled = SimpleNamespace(enabled=False)
    cfg = SimpleNamespace(
        log_freq=1,
        policy=SimpleNamespace(
            gradient_accumulation_steps=2,
            optimizer_grad_clip_norm=1.0,
            output_features={},
            pointmap_config=None,
            subtask_loss_weight=0.0,
            task="test",
            action_auxiliary_loss=disabled,
            discrete_action_auxiliary_loss=disabled,
            depth_gripper_event_loss=disabled,
        ),
    )
    batch = {
        ACTION: torch.ones(2, 1, 6),
        "reward": torch.zeros(2),
        "done": torch.zeros(2),
        "state": {},
        "truncated": torch.zeros(2),
        "next_state": {},
        "complementary_info": {},
    }
    iterator = iter((batch, batch))
    initial_scale = raw_policy.scale.detach().clone()

    metrics = MolmoAct2Trainer().update_actor(
        policy=wrapped_policy,
        optimizers={"policy": optimizer},
        online_iter=iterator,
        offline_iter=None,
        preprocessor=lambda value: value,
        dataset=None,
        device="cpu",
        cfg=cfg,
        optimization_step=1,
        training_runtime=runtime,
    )

    assert wrapped_policy.forward_calls == 2
    assert raw_policy.forward_calls == 2
    assert accelerator.no_sync_entries == 1
    assert raw_policy.scale.item() < initial_scale.item()
    assert "loss_actor" in metrics
