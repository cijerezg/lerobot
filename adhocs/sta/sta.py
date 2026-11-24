"""Does the gradient of the tube flow loss on a write layer of the action expert sit in a few directions that can be
found without any fitting? (M0 of the subspace-translated adaptation spec, README.)

    uv run python lerobot/adhocs/sta/sta.py --config_path=config_rl_validate.yaml

The policy of ``model`` is loaded once (its config, prompt and tip adapter are the ones in every run) and run in
float32; the weights of each entry of ``models`` are then put in place. Layers: per block of the action expert the
three Linear modules that write into its residual stream, ``self`` (self_attn.out_proj), ``cross``
(cross_attn.out_proj) and ``mlp`` (mlp.down_proj). Frames: one every ``seconds`` of ``episodes`` episodes of
``task_root`` (D) and of every episode of ``other_root`` (P; its episodes alternate between P1 and P2). Every frame
goes through the training forward of the flow loss on the recorded chunk (dropout off, the flow time grid, noise
seeded by the frame's index), so one frame gives num_flow_timesteps * chunk_size inputs h of a layer.

Per layer, in float64:

    A_D, A_P1, A_P2   sum(h^T h) / inputs
    V                 B = A_P1 + eps I, eps = 1e-3 trace(A_P1) / d_in. lam, vecs of A_D v = lam B v (v^T B v = 1);
                      lam_null = the largest lam of A_P2 against B. V = the vecs with lam > 1.5 lam_null, largest
                      first, at most k. score = sum(lam - lam_null) over them.
    U                 the top k eigenvectors of W B W^T
    G                 the gradient of the summed flow loss of the D frames with respect to W. A token of the expert is
                      one time step of the chunk with all joints in it, so the inputs h cannot be split by joint; the
                      loss can: G_j is the gradient through the predicted velocity of joint j alone, G = sum_j G_j.
                      Capture is also given per G_j (same U and V), with |G_j| and the cosines between the G_j.
    capture           |U^T G Vq|^2 / |G|^2, Vq = V made orthonormal (QR). The same for other choices of (U, V) with
                      the same numbers of columns: random; the top singular vectors of W; V = the top eigenvectors
                      of A_D with U as above; V as above with U random.

``report.py`` reads the file.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from dataclasses import dataclass

import torch
from safetensors.torch import load_file

from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.objective import flow_timestep_grid
from lerobot.probes.utils import build_episode_index, probe_frame_inputs, register_config_choices, suppress_pack_dropout
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

TUBE = "outputs/runs/rebot-v2-tube-b128-20261114/checkpoints/000700/pretrained_model"
INTERNAL = "outputs/runs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model"
KINDS = {"self": lambda block: block.self_attn.out_proj, "cross": lambda block: block.cross_attn.out_proj,
         "mlp": lambda block: block.mlp.down_proj}


@dataclass
class StaConfig(TrainRLServerPipelineConfig):
    model: str = TUBE
    models: str = f"base,internal={INTERNAL},tube700={TUBE}"  # comma list of name=path; "base" = policy.base_path
    task_root: str = "outputs/data/rebot_teleop_2026-11-15-tube3-annotated-v1"
    episodes: int = 4
    other_root: str = "outputs/data/rebot_2026-10-14-val-v1"
    seconds: float = 2.0
    k: int = 16
    out: str = "outputs/analysis/sta_2026-11-23/m0_joints.pt"


def weights(path: str, base_path: str) -> dict[str, torch.Tensor]:
    """The weights of one checkpoint under the names the loaded policy uses."""
    if path == "base":
        # the released files: several shards, names without the policy wrapper's leading "model."
        files = sorted(glob.glob(os.path.join(base_path, "model-*.safetensors")))
        return {"model." + name: value for file in files for name, value in load_file(file).items()}
    return load_file(os.path.join(path, "model.safetensors"))


def generalized(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """lam (largest first) and vecs of A v = lam B v, with v^T B v = 1."""
    L = torch.linalg.cholesky(B)
    C = torch.linalg.solve_triangular(L, torch.linalg.solve_triangular(L, A, upper=False).mT, upper=False)  # L^-1 A L^-T
    lam, Y = torch.linalg.eigh((C + C.mT) / 2)
    return lam.flip(0), torch.linalg.solve_triangular(L.mT, Y.flip(1), upper=True)


def top(S: torch.Tensor, k: int) -> torch.Tensor:
    return torch.linalg.eigh(S)[1].flip(1)[:, :k]


def random_basis(d: int, k: int, generator: torch.Generator) -> torch.Tensor:
    return torch.linalg.qr(torch.randn(d, k, generator=generator, dtype=torch.float64))[0]


def analyse(W, joints, A_D, A_P1, A_P2, k: int, generator: torch.Generator) -> dict:
    d_out, d_in = W.shape
    G = joints.sum(0)
    flat = torch.nn.functional.normalize(joints.flatten(1), dim=1)
    B = A_P1 + 1e-3 * A_P1.trace() / d_in * torch.eye(d_in, dtype=W.dtype, device=W.device)
    lam, vecs = generalized(A_D, B)
    null = generalized(A_P2, B)[0]
    kept = int((lam[:k] > 1.5 * null[0]).sum())
    out = {"lam": lam.cpu(), "null": null.cpu(), "kept": kept, "score": float((lam[:kept] - null[0]).sum()),
           "gradient_norm": float(G.norm()), "capture": {},
           "joint_gradient_norm": joints.flatten(1).norm(dim=1).cpu(), "joint_cosine": (flat @ flat.T).cpu()}
    # along the eigenvectors q_i of the other data's second moment, largest first: the share of the other inputs'
    # energy, of the task inputs' energy, and of |G|^2 and of each |G_j|^2 (G q_i is what the layer's gradient
    # asks for when the input is q_i). Leaving out the first r gives what G (I - P_r) keeps.
    old, Q = torch.linalg.eigh((A_P1 + A_P2) / 2)
    old, Q = old.flip(0), Q.flip(1)
    out["old"] = {"other": (old / old.sum()).cpu(), "task": ((Q * (A_D @ Q)).sum(0) / A_D.trace()).cpu(),
                  "gradient": ((G @ Q).norm(dim=0) ** 2 / G.norm() ** 2).cpu(),
                  "joint_gradient": ((joints @ Q).norm(dim=1) ** 2 / (joints.flatten(1).norm(dim=1) ** 2)[:, None]).cpu()}
    if kept == 0:
        return out
    Vq = torch.linalg.qr(vecs[:, :kept])[0]
    U = top(W @ B @ W.T, k)
    left, _, right = torch.linalg.svd(W, full_matrices=False)
    U_random, V_random = random_basis(d_out, k, generator).to(W.device), random_basis(d_in, kept, generator).to(W.device)
    bases = {"ours": (U, Vq), "random": (U_random, V_random), "W singular": (left[:, :k], right[:kept].T),
             "V = A_D top, U ours": (U, top(A_D, kept)), "V ours, U random": (U_random, Vq)}
    out["capture"] = {name: float((u.T @ G @ v).norm() ** 2 / G.norm() ** 2) for name, (u, v) in bases.items()}
    out["capture"]["random, expected"] = k * kept / (d_out * d_in)
    out["joint_capture"] = {name: [float((u.T @ g @ v).norm() ** 2 / g.norm() ** 2) for g in joints] for name, (u, v) in bases.items()}
    return out


def run(cfg: StaConfig) -> None:
    init_logging()
    assert not os.path.exists(cfg.out), f"{cfg.out} exists; pass another --out"
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    device = get_safe_torch_device(try_device=cfg.policy.device)
    base_path, tokenizer = cfg.policy.base_path, cfg.policy.discrete_action_tokenizer
    cfg.policy = PreTrainedConfig.from_pretrained(cfg.model)
    # the saved config has the paths of the day it was written
    cfg.policy.base_path, cfg.policy.discrete_action_tokenizer, cfg.policy.pretrained_path = base_path, tokenizer, cfg.model

    task, other = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.task_root), LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.other_root)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=task)
    adapter._set_probe_cuda_graph_enabled(False)
    policy = adapter.policy
    assert not policy.training
    policy.requires_grad_(False)
    policy.float()
    policy.config.dtype = "float32"
    layers = {f"{i}.{kind}": pick(block) for i, block in enumerate(policy._backbone()._require_action_expert().blocks) for kind, pick in KINDS.items()}
    for layer in layers.values():
        layer.weight.requires_grad_(True)
    grid = torch.from_numpy(flow_timestep_grid(cfg.policy, int(cfg.policy.num_flow_timesteps))).float().to(device)

    def frames(dataset, episodes):
        """{episode: frames}, one frame every ``seconds`` from the episode's start."""
        index = build_episode_index(dataset)
        every = round(cfg.seconds * cfg.env.fps)
        assert every % cfg.policy.image_stride == 0
        return {e: [probe_frame_inputs(dataset, cfg, g, cfg.policy.chunk_size) for g in index[e][::every]] for e in sorted(index)[:episodes]}

    D = [f for part in frames(task, cfg.episodes).values() for f in part]
    P = list(frames(other, None).values())
    P1, P2 = [f for part in P[0::2] for f in part], [f for part in P[1::2] for f in part]
    print(f"frames: D {len(D)}, P1 {len(P1)}, P2 {len(P2)}", flush=True)

    stats = None  # {layer: [sum of h^T h, inputs]} of the set going through

    def keep(name):
        def hook(_module, inputs):
            h = inputs[0].detach().flatten(0, -2).double()
            if name not in stats:
                stats[name] = [torch.zeros(h.shape[1], h.shape[1], dtype=torch.float64, device=device), 0]
            stats[name][0] += h.T @ h
            stats[name][1] += len(h)
        return hook

    for name, layer in layers.items():
        layer.register_forward_pre_hook(keep(name))

    seen = []  # the output of the expert's last layer, the predicted velocity [flow times, chunk, action width]
    policy._backbone()._require_action_expert().final_layer.register_forward_hook(lambda _module, _inputs, output: seen.append(output))
    names = list(task.meta.features["action"]["names"])
    params = [layer.weight for layer in layers.values()]
    joints = None  # [joint][layer]: the gradient of the summed flow loss through that joint's output alone

    def flow_loss(f):
        gt_actions = torch.as_tensor(f["gt_actions"])
        gt_actions = (gt_actions[0] if gt_actions.ndim == 3 else gt_actions)[: adapter.chunk_size, : adapter.action_dim].unsqueeze(0).to(device)
        with suppress_pack_dropout(adapter._preprocessor):
            batch = adapter._make_batch(f["obs"], f["task"], gt_actions=gt_actions, subtask=f["subtask"], metadata=f["metadata"])
        noise = torch.randn((1, len(grid), *batch[ACTION].shape[1:]), generator=torch.Generator(device=device).manual_seed(int(f["global_idx"])), device=device)
        vision = policy._backbone().vision_backbone
        vision._lerobot_future_features, vision._lerobot_capture_future_features = None, False
        losses, _ = policy._compute_flow_matching_loss_joint_per_layer(
            batch=batch, model_inputs=policy._model_inputs(batch), timesteps=grid[None], noise=noise, reduction="none")
        return losses.sum(), seen.pop()

    def second_moments(part, grad=False):
        nonlocal stats, joints
        if grad:
            joints = [[torch.zeros_like(p) for p in params] for _ in names]
        stats, total = {}, 0.0
        for f in part:
            with torch.set_grad_enabled(grad):
                loss, velocity = flow_loss(f)
            if grad:
                (d,) = torch.autograd.grad(loss, velocity, retain_graph=True)
                assert float(d[..., len(names):].abs().max()) == 0, "the loss reaches outputs past the joints"
                for j in range(len(names)):
                    through = torch.zeros_like(d)
                    through[..., j] = d[..., j]
                    for total_j, g in zip(joints[j], torch.autograd.grad(velocity, params, grad_outputs=through, retain_graph=j + 1 < len(names)), strict=True):
                        total_j += g
            total += float(loss.detach())
        return {name: (A / n).cpu() for name, (A, n) in stats.items()}, total / len(part), next(iter(stats.values()))[1]

    result = {"model": cfg.model, "task_root": cfg.task_root, "other_root": cfg.other_root, "seconds": cfg.seconds, "k": cfg.k,
              "frames": {"D": len(D), "P1": len(P1), "P2": len(P2)}, "joints": names, "models": {}}
    for entry in cfg.models.split(","):
        name, path = entry.split("=") if "=" in entry else (entry, entry)
        start = time.time()
        policy.load_state_dict(weights(cfg.model, base_path), strict=False)  # what a checkpoint's file does not have is the one of ``model``
        policy.load_state_dict(weights(path, base_path), strict=False)
        A_D, loss_D, inputs_D = second_moments(D, grad=True)
        A_P1, loss_P1, inputs_P1 = second_moments(P1)
        A_P2, loss_P2, inputs_P2 = second_moments(P2)
        generator = torch.Generator().manual_seed(0)
        found = {layer_name: analyse(layer.weight.detach().double(), torch.stack([joints[j][at] for j in range(len(names))]).double(), A_D[layer_name].to(device), A_P1[layer_name].to(device),
                                     A_P2[layer_name].to(device), cfg.k, generator) for at, (layer_name, layer) in enumerate(layers.items())}
        result["models"][name] = {"path": path, "flow_loss": {"D": loss_D, "P1": loss_P1, "P2": loss_P2},
                                  "inputs": {"D": inputs_D, "P1": inputs_P1, "P2": inputs_P2}, "layers": found}
        print(f"{name}: {time.time() - start:.0f} s | flow loss D {loss_D:.4f}, P1 {loss_P1:.4f}, P2 {loss_P2:.4f} | inputs per layer D {inputs_D}, P1 {inputs_P1}, P2 {inputs_P2}", flush=True)
        for layer_name in ("7.self", "7.cross", "7.mlp"):
            print(" ", layer_name, "kept", found[layer_name]["kept"], "score", round(found[layer_name]["score"], 2), {key: f"{value:.2e}" for key, value in found[layer_name]["capture"].items()}, flush=True)
        torch.save(result, cfg.out)
    print("wrote", cfg.out)


@parser.wrap()
def cli(cfg: StaConfig):
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
