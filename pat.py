import torch

#this code is used to construct the custom autograd function for PAT
def make_pat_func(f_forward, f_backward):
    """
    A function that constructs and returns the custom autograd function for physics−aware training.

    Parameters:
    f_forward: The function that is applied in the forward pass.
    Typically, the computation is performed on a physical system that is connected and controlled by
    the computer that performs the training loop. For this expandable code, we use a simulation for convenience.
    f_backward: The function that is employed in the backward pass to propagate estimators of gradients.

    Returns:
    f_pat: The custom autograd function for physics−aware training.
    
    Note:
    The arguments for f_forward, f_backward, f_pat are left quite generic. 
    For the demo code, it follows the form of f_forward(x, theta_1, theta_2)
    where x is the input data and theta_1 and theta_2 are two different parameters of this function
    """
    class func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args): 
            ctx.save_for_backward(*args)
            return f_forward(*args)
        def backward(ctx, grad_output):
            args = ctx.saved_tensors
            torch.set_grad_enabled(True)
            y = torch.autograd.functional.vjp(f_backward, args, v=grad_output)
            torch.set_grad_enabled(False)
            return y[1]
    return func.apply