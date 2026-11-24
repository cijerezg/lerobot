"""Is a change of the image tokens read by the discrete action output through a few directions only?

    uv run python lerobot/adhocs/vision_swap/gradient_split.py --config_path=config_rl_validate.yaml

The checkpoint ``model`` is loaded whole, as in ``discrete_swap.py``. Per frame, with its own image tokens ``a``
(the reference) and the training forward on the recorded action tokens (dropout off):

    gradients   of K scalars with respect to the image tokens, each one vector [tokens, width]: per recorded
                action token position the log-probability of the recorded token, and per position one random
                weighted sum of the log-probabilities of the action token vocabulary (weights of seed 0).
                QR of the K flattened vectors (float64) gives an orthonormal basis Q of their span.
    split       delta = (tokens of a condition) - a, for the conditions of ``discrete_swap.py`` (base, random,
                next, other); the part of delta inside the span of Q and the remainder; the share of |delta|^2
                inside. Control: the same with K random orthonormal directions in place of Q.
    put in      a + delta, a + inside part, a + remainder (forward only). Against the reference: the KL divergence
                between the next-token distributions (mean over the recorded action token positions) and the
                cross-entropy on the recorded tokens.
    check       the change of the K scalars along ``steps`` * delta (random and base), predicted by the gradients
                and from a forward pass.

``--float32=true`` runs the whole model in float32 (the weights are bfloat16). ``gradient_split_report.py`` reads
the file. The training forward runs under no_grad in the adapter and returns no logits: here it is called without
that decorator, and the hidden states it hands to the token loss are kept and sent through the LM head again.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import torch

from discrete_swap import CONDITIONS, SwapConfig, image_tokens, with_images

from lerobot.configs import PreTrainedConfig, parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.probes.adapters.molmoact2 import MolmoAct2Adapter
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.objective import flow_timestep_grid
from lerobot.probes.utils import probe_frame_inputs, register_config_choices, sample_episodes_evenly
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from backbone_cross import parts
from run_text_probe import label

training_losses = MolmoAct2Adapter.training_losses.__wrapped__  # without its no_grad


@dataclass
class SplitConfig(SwapConfig):
    frames_per_episode: int = 5
    episodes: int = 8
    steps: list[float] = field(default_factory=lambda: [0.01, 0.03, 0.1, 0.3, 1.0])
    float32: bool = False
    out: str = "outputs/analysis/vision_swap_2026-11-19/gradient_split.pt"


def run(cfg: SplitConfig) -> None:
    init_logging()
    assert not os.path.exists(cfg.out), f"{cfg.out} exists; pass another --out"
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    device = get_safe_torch_device(try_device=cfg.policy.device)
    base_path, tokenizer = cfg.policy.base_path, cfg.policy.discrete_action_tokenizer
    cfg.policy = PreTrainedConfig.from_pretrained(cfg.model)
    # the saved config has the paths of the day it was written
    cfg.policy.base_path, cfg.policy.discrete_action_tokenizer, cfg.policy.pretrained_path = base_path, tokenizer, cfg.model

    dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.root)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    adapter._set_probe_cuda_graph_enabled(False)
    policy = adapter.policy
    assert not policy.training
    policy.requires_grad_(False)  # the only gradient asked for is the one of the image tokens
    if cfg.float32:
        policy.float()
        policy.config.dtype = "float32"
    own_vision, other_vision = parts(cfg.model, base_path)["vision"], parts(cfg.vision, base_path)["vision"]
    assert set(own_vision) == set(other_vision)
    grid = torch.from_numpy(flow_timestep_grid(cfg.policy, int(cfg.policy.num_flow_timesteps))).float()
    action_ids = torch.tensor(sorted(policy._action_token_id_to_bin()), device=device)
    discrete_loss = policy._discrete_loss_from_backbone_outputs

    def frame(index):
        return probe_frame_inputs(dataset, cfg, index, cfg.policy.chunk_size)

    def forward(f, obs, fn, grad=False):
        """Log-probabilities [positions, vocabulary] at the recorded action token positions, the recorded tokens,
        and the image tokens as the vision backbone computed them."""
        kept = {}

        def keep(batch, outputs, **kwargs):
            kept["hidden"], kept["labels"] = outputs.last_hidden_state, batch["labels"]
            return discrete_loss(batch, outputs, **kwargs)

        policy._discrete_loss_from_backbone_outputs = keep
        with image_tokens(policy, fn) as vision, torch.set_grad_enabled(grad):
            out = training_losses(adapter, {**f, "obs": obs}, flow_timesteps=grid, flow_noise_seed=int(f["global_idx"]))
        del policy._discrete_loss_from_backbone_outputs
        assert len(vision.seen) == 1, f"vision backbone ran {len(vision.seen)} times in the training forward"
        shifted = torch.nn.functional.pad(kept["labels"], (0, 1), value=-100)[..., 1:]
        logp = torch.nn.functional.linear(kept["hidden"][shifted != -100], policy.model.lm_head.weight).float().log_softmax(-1)
        labels = shifted[shifted != -100]
        ce = -logp.detach().gather(1, labels[:, None])[:, 0].cpu()
        assert torch.allclose(ce, torch.from_numpy(out["discrete_token_ce"]).flatten(), atol=1e-4), "not the logits of the token loss"
        return logp, labels, vision.seen[0]

    samples = sample_episodes_evenly(dataset, cfg.frames_per_episode, cfg.episodes, 0, stride=cfg.policy.image_stride)
    frames = [frame(index) for _, _, index in samples]
    shift = round(cfg.next_seconds * cfg.env.fps / cfg.policy.image_stride) * cfg.policy.image_stride
    later = []
    for f in frames:
        candidate = frame(f["global_idx"] + shift) if f["global_idx"] + shift < dataset.num_frames else None
        if candidate is None or candidate["episode_idx"] != f["episode_idx"]:
            candidate = frame(f["global_idx"] - shift)
        assert candidate["episode_idx"] == f["episode_idx"]
        later.append(candidate)

    # the other vision path's tokens of every frame, with its weights in place
    policy.load_state_dict(other_vision, strict=False)
    swapped = [forward(f, f["obs"], None)[2] for f in frames]
    policy.load_state_dict(own_vision, strict=False)

    result = {
        "model": label(cfg.model), "vision": label(cfg.vision), "root": cfg.root, "conditions": CONDITIONS, "steps": cfg.steps,
        "float32": cfg.float32, "next_seconds": cfg.next_seconds, "episode": [f["episode_idx"] for f in frames],
        "frame": [f["frame_idx"] for f in frames], "index": [f["global_idx"] for f in frames], "frames": [],
    }
    start = time.time()
    for at, f in enumerate(frames):
        other = frames[(at + cfg.frames_per_episode) % len(frames)]
        assert other["episode_idx"] != f["episode_idx"]
        reference, labels, a = forward(f, f["obs"], None)
        images = {"next": forward(f, with_images(f["obs"], later[at]["obs"]), None), "other": forward(f, with_images(f["obs"], other["obs"]), None)}
        generator = torch.Generator(device=device).manual_seed(at)
        direction = torch.nn.functional.normalize(torch.randn(a.shape, generator=generator, device=device), dim=-1)
        noised = a + (direction * (swapped[at] - a).float().norm(dim=-1, keepdim=True)).to(a.dtype)
        tokens = {"base": swapped[at].to(a.dtype), "random": noised, "next": images["next"][2], "other": images["other"][2]}
        weights = (torch.randn(len(labels), len(action_ids), generator=torch.Generator().manual_seed(0)) / len(action_ids) ** 0.5).to(device)

        def scalars(logp):  # [K]: the recorded token's log-probability per position, then the weighted sums
            return torch.cat([logp.gather(1, labels[:, None])[:, 0], (logp[:, action_ids] * weights).sum(1)])

        def measure(part):  # a + part put in place, against the reference
            new = (a.double() + part).to(a.dtype)
            logp = forward(f, f["obs"], lambda _tokens: new)[0]
            return {"kl": float((reference.exp() * (reference - logp)).sum(1).mean()), "logp": logp,
                    "ce": float(-logp.gather(1, labels[:, None]).mean()), "scalars": (scalars(logp) - scalars(reference)).cpu(),
                    "applied": (new.double() - a.double()).flatten(),  # what the model received, after the cast to the weights' dtype
                    "cast": float((new.double() - a.double() - part).norm() / part.norm())}

        leaf = a.clone().requires_grad_(True)
        with_grad = scalars(forward(f, f["obs"], lambda _tokens: leaf, grad=True)[0])
        gradients = torch.stack([torch.autograd.grad(s, leaf, retain_graph=True)[0].flatten().double() for s in with_grad])  # [K, tokens * width]
        del with_grad
        Q, R = torch.linalg.qr(gradients.T)
        chance = torch.linalg.qr(torch.randn(Q.shape, generator=torch.Generator(device=device).manual_seed(at), device=device, dtype=torch.float64))[0]
        entry = {"other_episode": other["episode_idx"], "positions": len(labels), "K": len(gradients), "ce": float(-reference.gather(1, labels[:, None]).mean()),
                 "gradient_norm": gradients.norm(dim=1).cpu(), "singular": torch.linalg.svdvals(R).cpu(), "share": {}, "share_random_directions": {},
                 "norm": {}, "predicted": {}, "put_in": {}, "put_in_random_directions": {}, "check": {}, "token_delta": {},
                 "token_reference": a.float().norm(dim=-1).cpu(),  # per image token: the length of a, below of each delta
                 "token_gradient": (Q.view(*a.shape, -1) ** 2).sum((1, 2)).cpu()}  # per image token: |Q|^2 on it, sums to K
        for name in CONDITIONS:
            delta = tokens[name].double() - a.double()

            def split(basis):
                inside = (basis @ (basis.T @ delta.flatten())).view_as(delta)
                return inside, delta - inside

            (inside, remainder), (chance_inside, chance_remainder) = split(Q), split(chance)
            entry["norm"][name], entry["token_delta"][name] = float(delta.norm()), delta.norm(dim=-1).float().cpu()
            entry["share"][name] = float(inside.norm() ** 2 / delta.norm() ** 2)
            entry["share_random_directions"][name] = float(chance_inside.norm() ** 2 / delta.norm() ** 2)
            entry["predicted"][name] = (gradients @ delta.flatten()).cpu()
            full = measure(delta)
            if name in images and at == 0:
                print(f"{name}: tokens put in place against the camera images changed, largest log-probability difference", float((full["logp"] - images[name][0]).abs().max()), flush=True)
            entry["put_in"][name] = {"full": full, "inside": measure(inside), "remainder": measure(remainder)}
            entry["put_in_random_directions"][name] = {"inside": measure(chance_inside), "remainder": measure(chance_remainder)}
            if name in ("base", "random"):
                entry["check"][name] = []
                for step in cfg.steps:
                    stepped = measure(step * delta)
                    entry["check"][name].append({"predicted": (gradients @ stepped["applied"]).cpu(), "actual": stepped["scalars"]})
        for cells in (entry["put_in"], entry["put_in_random_directions"]):
            for parts_of in cells.values():
                for cell in parts_of.values():
                    del cell["logp"], cell["applied"]
        result["frames"].append(entry)
        if at == 0:
            print("image tokens", tuple(a.shape), a.dtype, "| recorded action tokens", len(labels), "| action token vocabulary", len(action_ids), "| K", len(gradients), flush=True)
            print("gradient lengths", [round(float(v), 4) for v in entry["gradient_norm"]], flush=True)
            print("singular values of the K gradients, largest and smallest", float(entry["singular"][0]), float(entry["singular"][-1]), flush=True)
            print("memory in use at the peak (GB)", torch.cuda.max_memory_allocated() / 1e9, flush=True)
        print(f"frame {at + 1} of {len(frames)}, {time.time() - start:.0f} s | share inside", {name: f"{entry['share'][name]:.2e}" for name in CONDITIONS},
              "| KL base full / inside / remainder", " / ".join(f"{entry['put_in']['base'][part]['kl']:.4f}" for part in ("full", "inside", "remainder")), flush=True)
        if at % 10 == 9 or at == len(frames) - 1:
            torch.save(result, cfg.out)
    print("wrote", cfg.out)


@parser.wrap()
def cli(cfg: SplitConfig):
    run(cfg)


def main() -> None:
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    register_config_choices()
    main()
