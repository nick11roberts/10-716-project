import torch
from torch.optim.optimizer import Optimizer, required

def L1(params, lr=required, reg=required, beta=required,
          momentum=0, dampening=0, weight_decay=0, nesterov=False):
    return BetaLasso(params, lr, reg, 0.0, 
        momentum, dampening, weight_decay, nesterov)

def Lasso(params, lr=required, reg=required, beta=required,
          momentum=0, dampening=0, weight_decay=0, nesterov=False):
    return BetaLasso(params, lr, reg, 1.0, 
        momentum, dampening, weight_decay, nesterov)

class BetaLasso(Optimizer):
    def __init__(self, params, lr=required, reg=required, beta=required,
                 momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, reg=reg, beta=beta, 
                        momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(BetaLasso, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BetaLasso, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            reg = group['reg']
            beta = group['beta']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                if weight_decay != 0:
                    d_p = d_p.add(param, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]

                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                # Add the L1 penalty subgradient
                d_p = d_p.add(torch.sign(param), alpha=reg)
                param.add_(d_p, alpha=-lr)

                # Proximal operator
                param.mul_(torch.abs(param) >= (beta * reg))

            # update momentum_buffers in state
            for p, momentum_buffer in zip(
                    params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
        