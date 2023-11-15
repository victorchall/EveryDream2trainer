import torch
from torch.optim.optimizer import Optimizer

class AdamCM(Optimizer):
    def __init__(self, params, lr=1e-6, betas=(0.9, 0.999), epsilon=1e-8,
                 weight_decay=0, buffer_capacity=10, decay_lambda=0.8):
        if lr < 0.0 or lr is None:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, epsilon=epsilon,
                        weight_decay=weight_decay, buffer_capacity=buffer_capacity,
                        decay_lambda=decay_lambda)
        super(AdamCM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamCM, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.cuda.amp.autocast():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('AdamCM does not support sparse gradients')

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['buffer'] = []
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.bfloat16)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Update buffer
                    priority = grad.norm()
                    if len(state['buffer']) < group['buffer_capacity']:
                        state['buffer'].append((priority, exp_avg.bfloat16().clone(), exp_avg_sq.bfloat16().clone()))
                    else:
                        # Find and replace the gradient with the smallest priority
                        min_priority, min_idx = min((buf[0], idx) for idx, buf in enumerate(state['buffer']))
                        if priority > min_priority:
                            state['buffer'][min_idx] = (priority, exp_avg.bfloat16().clone(), exp_avg_sq.bfloat16().clone())

                    # Decay priorities
                    #for i, buf in enumerate(state['buffer']):
                    #    buf[0] *= group['decay_lambda']

                    # Aggregate momenta
                    critical_exp_avg = torch.zeros_like(exp_avg, dtype=torch.bfloat16)
                    critical_exp_avg_sq = torch.zeros_like(exp_avg_sq, dtype=torch.bfloat16)

                    for i, (priority, buf_exp_avg, buf_exp_avg_sq) in enumerate(state['buffer']):
                        decayed_priority = priority * group['decay_lambda']
                        critical_exp_avg.add_(buf_exp_avg)
                        critical_exp_avg_sq.add_(buf_exp_avg_sq)
                        state['buffer'][i] = (decayed_priority, buf_exp_avg, buf_exp_avg_sq)
                        
                    denom = critical_exp_avg_sq.sqrt().add_(group['epsilon'])

                    step_size = group['lr']
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                    # Parameter update
                    p.data.addcdiv_(critical_exp_avg, denom, value=-step_size)

        return loss
