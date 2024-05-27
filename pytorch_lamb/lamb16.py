"""Lamb16 optimizer by Jyn"""

import collections

import torch
from tensorboardX import SummaryWriter
from torch.optim import Optimizer


def log_lamb_rs(optimizer: Optimizer, event_writer: SummaryWriter, token_count: int):
    """Log a histogram of trust ratio scalars in across layers."""
    results = collections.defaultdict(list)
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for i in ('weight_norm', 'adam_norm', 'trust_ratio'):
                if i in state:
                    results[i].append(state[i])

    for k, v in results.items():
        event_writer.add_histogram(f'lamb/{k}', torch.tensor(v), token_count)


class Lamb16(Optimizer):
    r"""Implements Lamb16 algorithm.

    LAMB16 is based LAMB algorithm, which is in turn based on Adam.
    LAMB16 introduced 2 scalar states variables to enable store optimizer
    states in 8bit float format.
    
    This is a Proof of Concept implemention.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float8_e5m2)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float8_e5m2)
                    state['exp_avg_norm'] = 1.0
                    state['exp_avg_sq_norm'] = 1.0

                # Restore or "decompress" the state
                # In real world implementation, this should be done in the kernel
                exp_avg = state['exp_avg'].to(torch.float32, copy=True) \
                           * state['exp_avg_norm']
                exp_avg_sq =  state['exp_avg_sq'].to(torch.float32, copy=True) \
                               * state['exp_avg_sq_norm']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                exp_avg_norm = exp_avg.pow(2).clip(0, 10).sum().sqrt() / exp_avg.numel() + group['eps']
                exp_avg_sq_norm = exp_avg_sq.pow(2).clip(0, 10).sum().sqrt() / exp_avg_sq.numel() + group['eps']
                state['exp_avg_norm'] = exp_avg_norm
                state['exp_avg_sq_norm'] = exp_avg_sq_norm
                # if state['step'] % 100 == 0:
                    # print(f"Max exp_avg: {exp_avg.max()}, min exp_avg: {exp_avg.min()}")
                    # print(f"Max exp_avg_sq: {exp_avg_sq.max()}, min exp_avg_sq: {exp_avg_sq.min()}")
                    # print(f"Norm exp_avg: {exp_avg_norm}, norm exp_avg_sq: {exp_avg_sq_norm}")
                state['exp_avg'] = (exp_avg / exp_avg_norm).clip(-15, 15).to(torch.float8_e4m3fn, copy=True)
                state['exp_avg_sq'] = (exp_avg_sq.clip(0, 10) / exp_avg_sq_norm).to(torch.float8_e5m2, copy=True)

                step_size = group['lr']

                weight_norm = (p.data.pow(2).sum().sqrt() / p.data.numel()).clip(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt() / adam_step.numel()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                trust_ratio = trust_ratio.clamp(0, 1)
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                # if state['step'] % 100 == 0:
                    # print(f"Weight norm: {weight_norm}, Adam norm: {adam_norm}, trust ratio: {trust_ratio}")

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss
