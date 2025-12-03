"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""
import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        sharded_meta: list[tuple[Tensor, dict, Tensor, int]] = []

        # Kick off gradient reductions; fall back to all_reduce when parameters
        # are not evenly divisible across ranks (e.g., world size 24).
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                p = params[base_i]
                grad = p.grad
                if grad is None:
                    continue
                if grad.shape[0] % world_size == 0 and grad.shape[0] > 0:
                    rank_size = grad.shape[0] // world_size
                    grad_slice = torch.empty_like(grad[:rank_size])
                    reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                    sharded_meta.append((p, group, grad_slice, rank_size))
                else:
                    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
                    beta1, beta2 = group['betas']
                    eps = group['eps']
                    wd = group['weight_decay']
                    lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                    state = self.state[p]
                    if not state:
                        state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    state['step'] += 1
                    t = state['step']
                    if wd != 0:
                        eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                        p.mul_(1 - eff_weight_decay)
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias1 = 1 - beta1 ** t
                    bias2 = 1 - beta2 ** t
                    denom = exp_avg_sq.sqrt().add_(eps)
                    step_size = lr * (torch.sqrt(bias2) / bias1)
                    update = exp_avg.div(denom).mul_(step_size)
                    p.add_(other=update, alpha=-1.0)

        # Handle sharded parameters after collectives finish.
        for idx in range(len(sharded_meta)):
            p, group, g_slice, rank_size = sharded_meta[idx]
            reduce_scatter_futures[idx].wait()
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            lr = group['lr'] * getattr(p, "lr_mul", 1.0)
            state = self.state[p]
            if not state:
                state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1
            t = state['step']
            if wd != 0:
                eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                p_slice.mul_(1 - eff_weight_decay)
            exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
            bias1 = 1 - beta1 ** t
            bias2 = 1 - beta2 ** t
            denom = exp_avg_sq.sqrt().add_(eps)
            step_size = lr * (torch.sqrt(bias2) / bias1)
            update = exp_avg.div(denom).mul_(step_size)
            p_slice.add_(other=update, alpha=-1.0)
            all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()
