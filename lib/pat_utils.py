import torch

'''
To do:
Rewrite so that forward and backward function can take multiple inputs:
https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd
'''

def generate_func(pnn):
    class func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return pnn.exp_normalized(x)
        def backward(ctx, grad_output):
            x = ctx.saved_tensors
            torch.set_grad_enabled(True)
            y = torch.autograd.functional.vjp(pnn.dt, x, v=grad_output)
            torch.set_grad_enabled(False)
            return y[1]
    return func.apply 