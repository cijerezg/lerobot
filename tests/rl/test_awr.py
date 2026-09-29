"""Behavioral checks for calibrated AWR, independent of a downloaded model."""

import json
import math
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from lerobot.rl.awr import AWRCalibration, group_key, normalize_log_weights, td_advantage, weight_telemetry
from lerobot.rl.awr_calibration import expected_training_groups, prepare_calibration
from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer
from lerobot.rl.training_runtime import TrainingRuntime


def test_td_keeps_discount_reward_mistake_penalty_and_terminal_mask():
    current = torch.full((5,), -1., requires_grad=True)
    successor = torch.tensor([-.8, -.8, -9., float("nan"), -20.], requires_grad=True)
    reward = torch.tensor([-1/12, -6/12, 0., -5/12, -1/12])
    done = torch.tensor([False, False, True, True, False])
    a = td_advantage(current, successor, reward, done, .97, -2., 0.)
    assert a.tolist() == pytest.approx([1 - 1/12 - .97*.8, 1 - .5 - .97*.8, 1., 1 - 5/12, -1.])
    assert not a.requires_grad
    assert a[0] - a[1] == pytest.approx(5/12)


def test_group_offsets_cancel_and_each_group_has_mean_one():
    first, second = group_key(4, "grasp"), group_key(4, "move")
    a = torch.tensor([-1., 0., 1., 20., 21., 22.])
    groups = [first] * 3 + [second] * 3
    calibration = AWRCalibration.fit(a, groups, .5, 3)
    log_q, z = calibration.log_weights(a, groups)
    w = normalize_log_weights(log_q)
    torch.testing.assert_close(z[:3], z[3:])
    torch.testing.assert_close(w[:3], w[3:])
    assert w[:3].mean() == pytest.approx(1.)
    assert w[3:].mean() == pytest.approx(1.)
    assert w[2]/w[0] == pytest.approx(math.exp(4 / calibration.scale), rel=1e-5)


def test_exponential_normalizer_balances_groups_with_different_variances():
    a = torch.tensor([-2., 0., 2., -.01, 0., .01])
    groups = ["wide"] * 3 + ["narrow"] * 3
    calibration = AWRCalibration.fit(a, groups, .5, 3)
    log_q, z = calibration.log_weights(a, groups)
    w = normalize_log_weights(log_q)
    assert w[:3].mean() == pytest.approx(1.)
    assert w[3:].mean() == pytest.approx(1.)
    assert (z[2] - z[0]) / (z[5] - z[3]) == pytest.approx(200.)
    assert w[5]/w[3] < 1.1  # tiny group variance is not amplified


def test_batch_normalization_preserves_calibrated_ratios_and_frozen_scale():
    calibration = AWRCalibration.fit(torch.tensor([-1., 0., 1.]), ["a"]*3, .5, 3)
    first = torch.tensor([-.4, .4])
    larger = torch.tensor([-.4, .4, 100.])
    small_q, small_z = calibration.log_weights(first, ["a"]*2)
    large_q, large_z = calibration.log_weights(larger, ["a"]*3)
    small_w, large_w = normalize_log_weights(small_q), normalize_log_weights(large_q)
    torch.testing.assert_close(small_z, large_z[:2])
    assert small_w[1]/small_w[0] == pytest.approx(float(large_w[1]/large_w[0]))
    assert large_w.mean() == pytest.approx(1.)
    assert normalize_log_weights(large_q, lam=0).tolist() == [1., 1., 1.]


def test_degenerate_sparse_and_unknown_groups_are_explicit():
    calibration = AWRCalibration.fit(torch.tensor([1., 7., 7.]), ["one", "two", "two"], .5, 3)
    q, z = calibration.log_weights(torch.tensor([1., 7.]), ["one", "two"])
    assert torch.isfinite(z).all()
    assert normalize_log_weights(q).tolist() == [1., 1.]
    with pytest.raises(ValueError, match="missing"):
        calibration.log_weights(torch.tensor([0.]), ["unknown"])
    with pytest.raises(ValueError, match="finite"):
        calibration.log_weights(torch.tensor([float("nan")]), ["one"])
    with pytest.raises(ValueError, match="no critic-valid"):
        AWRCalibration.fit(torch.tensor([]), [], .5, 3)


def test_serialization_keeps_training_samples_and_recomputes_temperature(tmp_path):
    calibration = AWRCalibration.fit(torch.tensor([-1., 0., 1.]), ["a"]*3, .5, 3, {"critic": "fixed"})
    path = tmp_path / "calibration.json"
    calibration.save(path)
    loaded = AWRCalibration.load(path, provenance={"critic": "fixed"})
    assert loaded.centers == calibration.centers
    assert loaded.log_normalizers == calibration.log_normalizers
    warmer = AWRCalibration.load(path, beta=2)
    q, _ = warmer.log_weights(torch.tensor([-1., 0., 1.]), ["a"]*3)
    assert q.exp().mean() == pytest.approx(1.)
    with pytest.raises(ValueError, match="does not match"):
        AWRCalibration.load(path, provenance={"critic": "different"})


@pytest.mark.parametrize("beta,clip,lam", [(0,3,1), (-1,3,1), (float("nan"),3,1), (1,0,1), (1,3,2)])
def test_invalid_weight_parameters(beta, clip, lam):
    from lerobot.rl.awr import validate_weight_parameters
    with pytest.raises(ValueError):
        validate_weight_parameters(beta, clip, lam)


def test_effective_batch_not_microbatch_and_skipped_rows_stay_one():
    calibration = AWRCalibration.fit(torch.tensor([-1., 0., 1.]), ["a"]*3, .5, 3)
    cfg = NS(policy=NS(advantage_normalization="subtask", advantage_lambda=1.))
    advantages = [torch.tensor([-1., 999.]), torch.tensor([0., 1.])]
    kepts = [torch.tensor([True, False]), torch.tensor([True, True])]
    weights, a, w = MolmoAct2Trainer._advantage_weights(
        advantages, kepts, cfg, groups=["a"]*3, calibration=calibration,
    )
    assert weights[0][1] == 1
    assert a.tolist() == [-1., 0., 1.]
    assert torch.cat(weights).mean() == pytest.approx(1.)
    assert weights[0].mean() < weights[1].mean()  # no microbatch renormalization
    assert w[-1] > 2.5
    weights, a, w = MolmoAct2Trainer._advantage_weights(
        advantages, [torch.zeros(2, dtype=torch.bool)]*2, cfg, groups=[], calibration=calibration,
    )
    assert not a.numel() and not w.numel()
    assert torch.cat(weights).tolist() == [1., 1., 1., 1.]


def test_global_normalization_includes_other_ranks_and_empty_rank():
    global_q = torch.tensor([-1000., -999., -998.], dtype=torch.float64)
    class Runtime:
        def __init__(self):
            self.calls = []
        def reduce_scalar(self, value, reduction):
            self.calls.append(reduction)
            return {1: 3., 2: -998., 3: float((global_q + 998).exp().sum())}[len(self.calls)]
    runtime = Runtime()
    actual = normalize_log_weights(global_q[:1], runtime=runtime)
    expected = normalize_log_weights(global_q)
    torch.testing.assert_close(actual, expected[:1])
    empty_runtime = Runtime()
    assert not normalize_log_weights(torch.tensor([]), runtime=empty_runtime).numel()
    assert empty_runtime.calls == ["sum", "max", "sum"]


def test_telemetry_distinguishes_frequency_from_weight_mass():
    keys = [group_key(4, "grasp")]*2 + [group_key(0, "move")]*2
    metrics, rows = weight_telemetry(
        [0., 1., 0., 1.], [.1, 1.9, .2, 1.8], [0., 4., 0., 2.], keys,
        [False, True, False, False], [False, False, True, False], 5, 3,
    )
    assert metrics["awr/kept_fraction"] == .8
    assert metrics["awr/terminal_fraction"] == .25
    assert metrics["awr/terminal_weight_share"] == pytest.approx(1.9/4)
    assert metrics["awr/mistake_onset_weight_share"] == pytest.approx(.2/4)
    assert metrics["awr/clip_fraction"] == .25
    assert all(row["weight_mean"] == pytest.approx(1.) for row in rows)
    assert "awr/clip_fraction" in MolmoAct2Trainer._aim_metrics(metrics)


def test_expected_groups_respect_stride_episode_boundaries_and_diverse_skip():
    buffer = NS(size=9, capacity=9, image_stride=2, storage_device="cpu", optimize_memory=True,
                actions=torch.zeros(9, 7), dones=torch.tensor([0,0,1,0,0,0,0,0,1], dtype=torch.bool),
                complementary_info={"subtask_index": torch.arange(9), "embodiment_index": torch.full((9,), 4)})
    trainer = NS(subtask_vocabulary=lambda _: [str(i) for i in range(9)])
    diverse = NS(_identity=[NS(subtask_index=3, embodiment_index=0), NS(subtask_index=5, embodiment_index=0)],
                 _critic=NS(skip=[False, True]))
    groups = expected_training_groups([buffer], diverse, trainer, None, NS(policy=NS(n_action_steps=3)))
    assert groups == {group_key(4, str(i)) for i in (0,4,6)} | {group_key(0, "3")}


def test_frozen_critic_scoring_uses_eval_without_changing_actor_training():
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.critic = nn.Dropout(.9)
        def forward_critic(self, batch):
            assert not self.critic.training
            return {"value": self.critic(batch)}
    policy = Policy().train()
    trainer = MolmoAct2Trainer()
    trainer._critic_batches = lambda *args: (torch.tensor([-1.]), torch.tensor([-.8]), torch.tensor([-1/12]), torch.tensor([False]))
    cfg = NS(skip_critic=True, policy=NS(discount=.97, value_support_min=-2, value_support_max=0))
    a, kept = trainer._advantages(policy, {}, None, cfg)
    assert policy.training and not policy.critic.training
    assert a.item() == pytest.approx(1 - 1/12 - .97*.8)
    assert kept.all()


def test_actor_optimizer_receives_calibrated_weights_and_writes_telemetry(tmp_path):
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.))
            self.critic = nn.Linear(1, 1).requires_grad_(False)
            self.depth_visual = None
        def forward(self, batch, **kwargs):
            loss = self.scale * batch["action"].square().mean(dim=(1,2))
            return loss, {"loss": loss.mean().item(), "loss_raw": loss.detach()}
    policy = Policy()
    trainer = MolmoAct2Trainer()
    key = group_key(4, "grasp")
    trainer._awr_calibration = AWRCalibration.fit(torch.tensor([-1., 0., 1.]), [key]*3, .5, 3)
    trainer._advantages = lambda policy, raw, *args: (raw["reward"], torch.ones(2, dtype=torch.bool))
    trainer._advantage_groups = lambda raw, *args: [key]*2
    batches = [{"action": torch.tensor(values).reshape(2,1,1), "reward": torch.tensor(a),
                "done": torch.zeros(2), "truncated": torch.zeros(2), "state": {}, "next_state": {}, "complementary_info": {}}
               for values, a in [([1.,2.], [-1.,0.]), ([3.,4.], [0.,1.])]]
    disabled = NS(enabled=False)
    p = NS(advantage_weighting=True, advantage_normalization="subtask", advantage_lambda=1., advantage_clip=3.,
           reward_normalization_constant=12., gradient_accumulation_steps=2, optimizer_grad_clip_norm=100.,
           output_features={}, pointmap_config=None, subtask_loss_weight=0., task="test",
           action_auxiliary_loss=disabled, discrete_action_auxiliary_loss=disabled, depth_gripper_event_loss=disabled)
    cfg = NS(policy=p, log_freq=1, output_dir=str(tmp_path))
    optimizer = torch.optim.SGD([policy.scale], lr=.01)
    log_q, _ = trainer._awr_calibration.log_weights(torch.tensor([-1.,0.,0.,1.]), [key]*4)
    expected_loss = (normalize_log_weights(log_q) * torch.tensor([1.,4.,9.,16.])).mean().item()
    metrics = trainer.update_actor(policy, {"policy": optimizer}, iter(batches), None, lambda x:x, None, "cpu", cfg,
                                   optimization_step=1)
    assert policy.scale.item() == pytest.approx(1 - .01 * expected_loss)
    assert metrics["loss_actor_weighted"] == pytest.approx(expected_loss)
    assert metrics["awr/weight_mean"] == pytest.approx(1.)
    audit = json.loads((tmp_path / "awr_subtask_weights.jsonl").read_text())
    assert audit["step"] == 1 and audit["groups"][0]["subtask"] == "grasp"


@pytest.mark.parametrize("skip_critic,awr,expected", [(True,True,True), (True,False,False), (False,False,True)])
def test_diverse_loader_serves_real_successors_for_frozen_awr(monkeypatch, skip_critic, awr, expected):
    from lerobot.rl.data_sources import diverse_integration as integration
    monkeypatch.setattr(integration, "sample_spec_from_config", lambda cfg: object())
    monkeypatch.setattr(integration, "open_federated_corpus", lambda root: object())
    monkeypatch.setattr(integration, "select_actor_anchors", lambda corpus: object())
    monkeypatch.setattr(integration, "resolve_cache", lambda *args, **kwargs: object())
    monkeypatch.setattr(integration, "extend_dataset_vocabulary", lambda *args, **kwargs: ({},{}))
    monkeypatch.setattr(integration, "HierarchicalAnchorSampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(integration, "DiverseActorBuffer", lambda *args, **kwargs: kwargs)
    cfg = NS(skip_critic=skip_critic, policy=NS(advantage_weighting=awr, reward_normalization_constant=12.))
    diverse = NS(validate=lambda: None, root="unused", cache_dir="unused", group_weight="uniform",
                 group_weight_overrides={}, render_automatic_quality=False)
    result = integration.build_diverse_buffer(cfg, diverse, main_dataset=None, device="cpu", seed=42, is_main_process=False)
    assert result["serve_critic"] is expected


def test_calibration_preparation_is_training_only_persisted_and_coverage_checked(tmp_path, monkeypatch):
    from lerobot.rl import awr_calibration as module
    monkeypatch.setattr(module, "calibration_provenance", lambda cfg: {"critic": "fixed"})
    key = group_key(4, "grasp")
    class Trainer:
        def _advantages(self, policy, raw, preprocessor, cfg):
            return raw["reward"], torch.tensor([True, False])
        def _advantage_groups(self, raw, preprocessor):
            return [key, group_key(4, "skipped")]
    trainer = Trainer()
    p = NS(advantage_beta=.5, advantage_clip=3., advantage_lambda=1., advantage_calibration_path=None,
           advantage_calibration_batches=2, reward_normalization_constant=12., pretrained_path=None)
    cfg = NS(policy=p, skip_critic=True, output_dir=str(tmp_path))
    policy = NS(critic=nn.Linear(1,1).requires_grad_(False))
    def batch(reward):
        return {"reward": torch.tensor(reward), "done": torch.zeros(2), "action": torch.zeros(2,1,1),
                "truncated": torch.zeros(2), "state": {}, "next_state": {}, "complementary_info": {}}
    calibration = prepare_calibration(trainer, policy, iter([batch([-1.,9.]), batch([1.,9.])]),
                                      None, cfg, TrainingRuntime(), {key})
    assert calibration.n == 2 and calibration.samples[key] == [-1.,1.]
    assert trainer._awr_calibration is calibration
    coverage = json.loads((tmp_path / "awr_calibration_coverage.json").read_text())
    assert coverage["missing_groups"] == []
    assert coverage["shared_scale"] == 1.
    # Loading a pinned calibration must not draw more data or fit to another batch.
    loaded = prepare_calibration(trainer, policy, iter(()), None, cfg, TrainingRuntime(), {key})
    assert loaded.centers == calibration.centers
    with pytest.raises(ValueError, match="missing 1/2"):
        prepare_calibration(trainer, policy, iter(()), None, cfg, TrainingRuntime(), {key, "missing"})
    cfg.skip_critic = False
    with pytest.raises(ValueError, match="frozen critic"):
        prepare_calibration(trainer, policy, iter(()), None, cfg, TrainingRuntime(), {key})


def test_probe_resampling_and_beta_sweep_use_frozen_training_calibration(tmp_path):
    import numpy as np
    from lerobot.probes.critic import _weight_plots
    train = AWRCalibration.fit(torch.tensor([-1.,0.,1.,2.,3.,4.]), ["a"]*3+["b"]*3, .5, 3.)
    # Deliberately different means in validation: these must not be re-centered.
    validation = torch.tensor([.4,.7,2.7,3.7])
    groups = ["a","a","b","b"]
    tables = {.5: train}
    calls = []
    def weights(indices, beta):
        calls.append((len(indices), beta))
        if beta not in tables:
            tables[beta] = train.with_beta(beta)
        q, z = tables[beta].log_weights(validation[indices], [groups[int(i)] for i in indices])
        return normalize_log_weights(q), z.clamp(-3,3)
    w, z = weights(np.arange(4), .5)
    summary = _weight_plots(validation.numpy(), z.numpy(), w.numpy(), .5, 3., 1., 8, 42, tmp_path, weight_fn=weights)
    assert summary["adv_weight_ess_frac"] < 1
    assert sum(n == 8 for n, beta in calls) == 200
    assert len(tables) > 1
    assert all(table.centers == train.centers and table.scale == train.scale for table in tables.values())
    assert (tmp_path / "advantage_weights.png").is_file()


def test_coverage_samples_preserve_mixture_scale_and_survive_beta_reload(tmp_path):
    original = AWRCalibration.fit([-1., 1.], ["common"] * 2, .5, 3.)
    filled = original.with_coverage({"rare": [-100., 100.]})
    assert filled.scale == original.scale == 1.
    assert filled.centers["common"] == original.centers["common"]
    assert filled.log_normalizers["common"] == original.log_normalizers["common"]
    assert filled.n == 4 and filled.mixture_n == 2
    path = tmp_path / "calibration.json"
    filled.save(path)
    loaded = AWRCalibration.load(path, beta=1.)
    assert loaded.scale == 1. and loaded.coverage_samples == {"rare": [-100., 100.]}
    assert loaded.log_normalizers == filled.with_beta(1.).log_normalizers
    with pytest.raises(ValueError, match="absent from the mixture"):
        original.with_coverage({"common": [3.]})
    # The already-computed schema-1 artifacts remain readable.
    original.save(path)
    old = json.loads(path.read_text())
    old["schema"] = 1
    old.pop("coverage_groups")
    old.pop("mixture_n")
    path.write_text(json.dumps(old))
    assert AWRCalibration.load(path).samples == original.samples


def test_incomplete_calibration_recovers_only_missing_groups_and_resumes_each_batch(tmp_path, monkeypatch):
    from lerobot.rl import awr_calibration as module
    monkeypatch.setattr(module, "calibration_provenance", lambda cfg: {"critic": "fixed"})
    common, rare = group_key(4, "common"), group_key(0, "rare")
    original = AWRCalibration.fit([-1., 1.], [common] * 2, .5, 3., {"critic": "fixed"})
    path = tmp_path / "awr_calibration.json"
    original.save(path)
    policy = NS(critic=nn.Linear(1, 1).requires_grad_(False))
    p = NS(advantage_beta=.5, advantage_clip=3., advantage_lambda=1., advantage_calibration_path=None,
           advantage_calibration_batches=2048, reward_normalization_constant=12., pretrained_path=None)
    cfg = NS(policy=p, skip_critic=True, output_dir=str(tmp_path), batch_size=3, seed=42)
    trainer = NS(_advantages=lambda policy, raw, *args: (raw["reward"], torch.ones(len(raw["reward"]), dtype=torch.bool)),
                 _advantage_groups=lambda raw, *args: [rare] * len(raw["reward"]))
    requests_seen = []

    def sampler(requests, seed):
        requests_seen.append(list(requests))
        for start in range(0, len(requests), cfg.batch_size):
            if len(requests_seen) == 1 and start == 3:
                raise RuntimeError("interrupted recovery")
            n = min(cfg.batch_size, len(requests) - start)
            yield {"reward": torch.arange(n).float(), "done": torch.zeros(n), "action": torch.zeros(n, 1, 1),
                   "truncated": torch.zeros(n), "state": {}, "next_state": {}, "complementary_info": {}}

    with pytest.raises(RuntimeError, match="interrupted recovery"):
        prepare_calibration(trainer, policy, iter(()), None, cfg, TrainingRuntime(), {common, rare}, coverage_sampler=sampler)
    partial = AWRCalibration.load(path)
    assert partial.mixture_samples == original.samples
    assert len(partial.coverage_samples[rare]) == 3
    recovered = prepare_calibration(trainer, policy, iter(()), None, cfg, TrainingRuntime(), {common, rare}, coverage_sampler=sampler)
    assert requests_seen == [[rare] * 8, [rare] * 5]
    assert recovered.mixture_samples == original.samples
    assert len(recovered.coverage_samples[rare]) == 8
    assert recovered.scale == original.scale
    coverage = json.loads(path.with_name("awr_calibration_coverage.json").read_text())
    assert coverage["missing_groups"] == [] and coverage["coverage_samples"] == 8
    prepare_calibration(trainer, policy, iter(()), None, cfg, TrainingRuntime(), {common, rare}, coverage_sampler=sampler)
    assert len(requests_seen) == 2  # A complete artifact performs no new critic evaluations.


def test_conditional_coverage_keeps_source_episode_anchor_probabilities():
    import numpy as np
    from lerobot.rl.awr_coverage import TrainingCoverageSampler
    from lerobot.rl.data_sources.diverse_mixture import MixtureGroup
    key = group_key(0, "rare")
    diverse = NS(size=4, _identity=[NS(embodiment_index=0, subtask_index=0)] * 4,
                 _critic=NS(skip=[True, False, False, False]), collate=lambda rows: rows,
                 _sampler=NS(groups=["a", "b"], probabilities=[.75, .25],
                             episode_rows={"a": [np.array([0, 1]), np.array([2])], "b": [np.array([3])]}))
    sampler = TrainingCoverageSampler([MixtureGroup("diverse", [diverse])],
                                      NS(subtask_vocabulary=lambda _: ["rare"]), None,
                                      NS(batch_size=32, policy=NS(n_action_steps=30)))
    entries = sampler._candidates({key})[key]
    assert [row for _, row, _ in entries] == [1, 2, 3]
    assert [mass for _, _, mass in entries] == pytest.approx([.1875, .375, .25])
    with pytest.raises(ValueError, match="No eligible training rows"):
        sampler._candidates({group_key(0, "unreachable")})


def test_explicit_replay_starts_use_normal_chunks_and_reject_invalid_boundaries():
    from lerobot.rl.buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=10, device="cpu", state_keys=["observation.state"],
                          optimize_memory=True, use_drq=False)
    for index in range(10):
        buffer.add({"observation.state": torch.tensor([[float(index)]])}, torch.tensor([[float(index)]]),
                   1., next_state=None, done=index in (4, 9), truncated=False)
    batch = buffer.sample(2, action_chunk_size=3, indices=[0, 2])
    assert batch["action"].reshape(2, 3).tolist() == [[0., 1., 2.], [2., 3., 4.]]
    assert batch["next_state"]["observation.state"].reshape(-1).tolist() == [3., 5.]
    with pytest.raises(ValueError, match="physical episode boundary"):
        buffer.sample(1, action_chunk_size=3, indices=[3])
    with pytest.raises(ValueError, match="in range"):
        buffer.sample(1, action_chunk_size=3, indices=[-1])
    buffer.image_stride = 2
    with pytest.raises(ValueError, match="stride aligned"):
        buffer.sample(1, action_chunk_size=2, indices=[1])


def test_conditional_rebot_coverage_preserves_source_mass_when_rows_are_skipped():
    from lerobot.rl.awr_coverage import TrainingCoverageSampler
    from lerobot.rl.data_sources.diverse_mixture import MixtureGroup
    def buffer(skips):
        n = len(skips)
        return NS(size=n, capacity=n, optimize_memory=True, storage_device="cpu", image_stride=1,
                  actions=torch.zeros(n, 1), dones=torch.zeros(n, dtype=torch.bool),
                  complementary_info={"subtask_index": torch.zeros(n, dtype=torch.long),
                                      "embodiment_index": torch.zeros(n, dtype=torch.long),
                                      "critic_skip": torch.tensor(skips)})
    first, second = buffer([True, True, True, False]), buffer([False])
    sampler = TrainingCoverageSampler([MixtureGroup("rebot", [first, second])],
                                      NS(subtask_vocabulary=lambda _: ["rare"]), None,
                                      NS(batch_size=4, policy=NS(n_action_steps=1)))
    entries = sampler._candidates({group_key(0, "rare")})[group_key(0, "rare")]
    assert [(row, mass) for _, row, mass in entries] == [(3, .125), (0, .5)]
