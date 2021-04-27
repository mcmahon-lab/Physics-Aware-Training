"""
This module contains the backbone code of physics-aware-training, a custom
auto-differentiable function whose forward pass is through an experiment and
whose backward pass is through a differentiable digital model.

Functions:
--------
    def generate_func(pnn):
        Generates a custom auto-differentiable function with the
        pnn.exp_normalized function as forward pass and pnn.dt function
        as backward pass.
    def custom_autodiff_function(forward_func, backward_func):
        Generates a custom auto-differentiable function with forward
        pass forward_func and backward pass backward_func.
"""

import torch

def generate_func(pnn):
    """
    Generates a custom auto-differentiable function with the
    pnn.exp_normalized function as forward pass and pnn.dt function
    as backward pass.
    """
    return custom_autodiff_function(
        forward_func = pnn.exp_normalized, 
        backward_func = pnn.dt
    )

def custom_autodiff_function(forward_func, backward_func):
    """
    Generates a custom auto-differentiable function with forward
    pass forward_func and backward pass backward_func.
    
    Both forward_func and backward_func have to have the signature 
    forward_func(x) and backward_func(x).
    Additional arguments are not possible in this implementation.
    """
    class func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return forward_func(x)
        def backward(ctx, grad_output):
            x = ctx.saved_tensors
            torch.set_grad_enabled(True)
            y = torch.autograd.functional.vjp(backward_func, x, v=grad_output)
            torch.set_grad_enabled(False)
            return y[1]
    return func.apply 