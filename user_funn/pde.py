import torch

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)

def diff(y, x, degree=1):
    if degree == 1:
        return torch.autograd.grad(y, x,
                               grad_outputs=torch.ones_like(y),
                               create_graph=True)[0]
    else:
        dydx = diff(y,x,degree-1)
        return torch.autograd.grad(dydx, x,
                               grad_outputs=torch.ones_like(dydx),
                               create_graph=True)[0]