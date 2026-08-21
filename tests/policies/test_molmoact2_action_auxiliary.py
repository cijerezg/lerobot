import math

import pytest
import torch

from lerobot.policies.molmoact2.configuration_molmoact2 import ActionAuxiliaryLossConfig
from lerobot.policies.molmoact2.modeling_molmoact2 import (
    _action_band_auxiliary_loss,
    _parse_action_frequency_bands,
    _tempered_action_band_beta,
)


def _config(**overrides) -> ActionAuxiliaryLossConfig:
    values = {
        "enabled": True,
        "weight": 1.0,
        "gamma": 0.25,
        "band_spec": "dc=0;k1=1;k2=2;k3=3;detail=4-9;high=10-20",
        "band_powers": (
            1.0086008381,
            0.0411540009,
            0.0067903467,
            0.0012663327,
            0.0001327078,
            0.0000066883,
        ),
    }
    values.update(overrides)
    return ActionAuxiliaryLossConfig(**values)


def test_action_band_config_validates_fixed_metric_parameters():
    config = _config(path_weight=0.5, path_threshold=0.2)
    assert config.gamma == 0.25
    assert config.path_weight == 0
    assert config.path_threshold is None
    legacy = ActionAuxiliaryLossConfig(enabled=True, path_weight=0.5)
    assert not legacy.enabled
    with pytest.raises(ValueError, match="gamma must be in"):
        _config(gamma=1.1)
    with pytest.raises(ValueError, match="strictly positive"):
        _config(band_powers=(1.0, 0.0))
    with pytest.raises(ValueError, match="band_spec and fixed band_powers"):
        ActionAuxiliaryLossConfig(enabled=True, band_spec="dc=0")
    with pytest.raises(ValueError, match="band_spec and fixed band_powers"):
        ActionAuxiliaryLossConfig(enabled=True)


def test_band_parser_allows_only_a_contiguous_low_frequency_prefix():
    assert _parse_action_frequency_bands("dc=0;low=1-2", horizon=5) == [
        ("dc", 0, 1),
        ("low", 1, 3),
    ]
    with pytest.raises(ValueError, match="ordered, contiguous"):
        _parse_action_frequency_bands("dc=0;high=2-", horizon=5)


def test_tempering_endpoints_are_mse_width_and_inverse_power():
    spec = "first=0;rest=1-3"
    _, mse_beta = _tempered_action_band_beta(spec, [4.0, 1.0], 0.0, horizon=4)
    _, inverse_beta = _tempered_action_band_beta(spec, [4.0, 1.0], 1.0, horizon=4)

    torch.testing.assert_close(mse_beta, torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(inverse_beta, torch.tensor([0.2, 0.8]))


def test_configured_gamma_quarter_budgets_match_the_diagnostic():
    config = _config()
    _, beta = _tempered_action_band_beta(
        config.band_spec,
        config.band_powers,
        config.gamma,
        horizon=30,
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        beta,
        torch.tensor(
            [
                0.0059934008,
                0.0133352185,
                0.0209232992,
                0.0318394970,
                0.2145321961,
                0.7133763884,
            ],
            dtype=torch.float64,
        ),
        rtol=1e-7,
        atol=1e-9,
    )


def test_gamma_zero_full_partition_reconstructs_time_domain_mse():
    residual = torch.randn(4, 3, 8, 5, requires_grad=True)
    config = _config(
        gamma=0.0,
        band_spec="low=0-2;high=3-7",
        band_powers=(2.0, 0.5),
    )

    auxiliary, _, _ = _action_band_auxiliary_loss(residual, config)
    expected = residual.square().mean(dim=(1, 2, 3))
    torch.testing.assert_close(auxiliary, expected, rtol=1e-5, atol=1e-6)

    auxiliary.mean().backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()


def test_untrusted_tail_is_absent_only_from_the_auxiliary():
    horizon = 6
    k = 5
    n = torch.arange(horizon, dtype=torch.float64)
    tail_mode = math.sqrt(2.0 / horizon) * torch.cos(math.pi * k * (n + 0.5) / horizon)
    residual = tail_mode.view(1, 1, horizon, 1)
    config = _config(
        gamma=0.0,
        band_spec="signal=0-4",
        band_powers=(1.0,),
    )

    auxiliary, _, _ = _action_band_auxiliary_loss(residual, config)
    torch.testing.assert_close(auxiliary, torch.zeros_like(auxiliary), atol=1e-12, rtol=0)
    assert residual.square().mean() > 0  # The unchanged base flow MSE still sees this mode.


def test_padded_dimensions_are_ignored_and_tail_padded_chunks_skip_auxiliary():
    residual = torch.zeros(2, 1, 4, 3)
    residual[0, :, :, 2] = 1000.0
    residual[1, :, :, :2] = 1.0
    config = _config(
        gamma=0.0,
        band_spec="all=0-3",
        band_powers=(1.0,),
    )

    auxiliary, components, active = _action_band_auxiliary_loss(
        residual,
        config,
        action_horizon_is_pad=torch.tensor(
            [[False, False, False, False], [False, False, False, True]]
        ),
        action_dim_is_pad=torch.tensor([[False, False, True], [False, False, True]]),
    )

    torch.testing.assert_close(auxiliary, torch.zeros(2))
    assert active["full_chunk_active"].tolist() == [True, False]
    assert components["band_all_mse"][0] == 0
    assert torch.isnan(components["band_all_mse"][1])
