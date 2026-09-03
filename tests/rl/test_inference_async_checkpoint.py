from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.rl.inference_async import _configure_inference_checkpoint


def _cfg(path=None, policy_path=None):
    return SimpleNamespace(
        inference_checkpoint_path=path,
        policy=SimpleNamespace(pretrained_path=policy_path),
    )


def _checkpoint(path: Path, *, weights: str = "model.safetensors") -> Path:
    path.mkdir()
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json", weights):
        (path / name).touch()
    return path


def test_checkpoint_path_is_required_for_standalone_hardware_inference() -> None:
    with pytest.raises(RuntimeError, match="refusing to run"):
        _configure_inference_checkpoint(_cfg())


def test_checkpoint_must_be_a_complete_saved_policy(tmp_path) -> None:
    checkpoint = tmp_path / "incomplete"
    checkpoint.mkdir()
    (checkpoint / "config.json").touch()

    with pytest.raises(FileNotFoundError, match="policy_postprocessor.*policy_preprocessor.*model"):
        _configure_inference_checkpoint(_cfg(checkpoint))


@pytest.mark.parametrize("weights", ["model.safetensors", "model.safetensors.index.json"])
def test_checkpoint_accepts_single_or_sharded_weights_and_sets_policy_path(
    tmp_path, weights: str
) -> None:
    checkpoint = _checkpoint(tmp_path / weights.replace(".", "_"), weights=weights)
    cfg = _cfg(checkpoint)

    resolved = _configure_inference_checkpoint(cfg)

    assert resolved == checkpoint.resolve()
    assert cfg.policy.pretrained_path == checkpoint.resolve()


def test_standard_policy_pretrained_path_remains_a_supported_fallback(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path / "standard")
    cfg = _cfg(policy_path=checkpoint)

    _configure_inference_checkpoint(cfg)

    assert cfg.policy.pretrained_path == checkpoint.resolve()
