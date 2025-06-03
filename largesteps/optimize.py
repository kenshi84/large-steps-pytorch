import torch

class AdamUniform(torch.optim.Optimizer):
    """
    Variant of Adam with uniform scaling by the second moment.

    Instead of dividing each component by the square root of its second moment,
    we divide all of them by the max.
    """
    def __init__(self, params, lr=0.1, betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamUniform, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamUniform, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state)==0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)
                    state["m2max"] = 0

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                g1.mul_(b1).add_(grad, alpha=1-b1)
                g2.mul_(b2).add_(grad.square(), alpha=1-b2)
                m1 = g1 / (1-(b1**state["step"]))
                m2 = g2 / (1-(b2**state["step"]))
                # This is the only modification we make to the original Adam algorithm
                if amsgrad:
                    state["m2max"] = max(state["m2max"], m2.sqrt().max())
                    gr = m1 / (eps + state["m2max"])
                else:
                    gr = m1 / (eps + m2.sqrt().max())
                p.data.sub_(gr, alpha=lr)
