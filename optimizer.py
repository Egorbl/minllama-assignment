from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                # Update first and second moments of the gradients
                current_m1 = state.get("m1", torch.zeros_like(p))
                current_m2 = state.get("m2", torch.zeros_like(p))
                state["m1"] = beta1 * current_m1 + (1 - beta1) * grad
                state["m2"] = beta2 * current_m2 + (1 - beta2) * grad ** 2

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                corr = alpha * ((1 - beta2 ** state["step"]) ** 0.5) / (1 - beta1 ** state["step"])

                # Update parameters
                p.data = p.data - corr * state["m1"] / (state["m2"] ** 0.5 + eps)

                # Add weight decay after the main gradient-based updates.
                p.data = p.data - alpha * group["weight_decay"] * p.data
                # Please note that the learning rate should be incorporated into this update.

        return loss