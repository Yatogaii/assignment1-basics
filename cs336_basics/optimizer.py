from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # g
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # state 是 p 对应的历史状态，需要手动初始化
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["step"] += 1
                t = state["step"]

                # Decay the first and second moment running average coefficient
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(grad, alpha=1 - b1) # m
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2) # v

                # Bias correction
                bias_correction1 = 1 - b1 ** state["step"]
                bias_correction2 = 1 - b2 ** state["step"]

                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = torch.sqrt(v) + eps
                
                p.addcdiv_(m, denom, value=-alpha_t)
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss 

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int ,
):
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    else:
        return min_learning_rate