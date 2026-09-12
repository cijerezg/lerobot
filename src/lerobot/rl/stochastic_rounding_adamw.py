"""AdamW whose bf16 weight write rounds stochastically instead of to nearest.

Why: with bf16 parameters and no fp32 master copy, nearest rounding drops every Adam step
smaller than half an ulp of the weight (|w| >= 1.56e-2 at lr 5e-5 never moves; see the
2026-09-11 weight-delta audit, migration/weight_deltas_2026-09-11). Stochastic rounding makes
the stored weight an unbiased estimate of the fp32 update, so small steps land in expectation.

Recipe follows Ozkara, Yu, Park (AISTATS 2025, arXiv 2502.20566): moments stay in the parameter
dtype and only the final weight write is rounded stochastically. Weight decay and the Adam step
are applied in fp32 and rounded together, once. The state layout matches torch.optim.AdamW, so
a checkpoint from either optimizer loads into the other.

Knob: policy.optimizer_stochastic_rounding (see build_named_adamw_optimizers in rl/utils.py).
"""

import math

import torch


def stochastic_round_bf16_bits_(x_f32: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """In place: add uniform noise to the low 16 bits of x's fp32 pattern and clear them.

    Afterwards x.to(torch.bfloat16) is exact and equals stochastic rounding of the original
    value: the truncated tail rounds the magnitude up with probability tail / ulp. Integer
    addition carries into the exponent correctly and two's complement makes it dtype-symmetric,
    so negative values round away from zero with the same probability.
    """
    bits = x_f32.view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, device=bits.device, dtype=torch.int32, generator=generator)
    bits.add_(noise).bitwise_and_(-65536)  # -65536 == 0xFFFF0000 as int32
    return x_f32


class StochasticRoundingAdamW(torch.optim.AdamW):
    """torch.optim.AdamW math (decoupled weight decay, bias correction, eps outside the sqrt),
    computed in fp32 per tensor and written back to bf16 parameters with stochastic rounding.
    Non-bf16 parameters are written normally.

    Telemetry: set ``collect_stats`` before ``step()`` and read ``step_stats`` after it, a dict
    param -> (moved, rn_moved) 0-d int64 tensors: the number of elements whose stored bf16
    value changed, and the number nearest rounding would have changed on the same update.
    """

    def __init__(self, params, seed: int = 0, **kwargs):
        super().__init__(params, **kwargs)
        device = self.param_groups[0]["params"][0].device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.collect_stats = False
        self.step_stats: dict[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_stats = {}
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"].item()

                grad = p.grad
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2_sqrt = math.sqrt(1 - beta2**step)
                # copy=True: .float() on an fp32 tensor is the tensor itself, not a copy.
                denom = exp_avg_sq.to(torch.float32, copy=True).sqrt_().div_(bias_correction2_sqrt).add_(eps)
                new = p.to(torch.float32, copy=True).mul_(1 - lr * weight_decay)
                new.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
                del denom

                if p.dtype == torch.bfloat16:
                    if self.collect_stats:
                        rn_moved = new.to(torch.bfloat16).ne(p).sum()
                    stochastic_round_bf16_bits_(new, self.generator)
                    if self.collect_stats:
                        self.step_stats[p] = (new.to(torch.bfloat16).ne(p).sum(), rn_moved)
                p.copy_(new)
        return loss
