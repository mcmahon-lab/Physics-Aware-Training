############## code for solving ODEs with pytorch ###################
# The following code will solve ODEs with the 4th order Runge–Kutta (RK4) method

import torch

def rk_loop(ode, x, dt, *args):
    """
    This functions runs the internal loop of the RK4 method
    """
    k1 = ode(x, *args)
    k2 = ode(x+dt*k1/2, *args)
    k3 = ode(x+dt*k2/2, *args)
    k4 = ode(x+dt*k3, *args)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def make_ode_map(ode, Nt, dt):
    """
    The output of this function is the ode_map function, which takes
    in the initial condition x and returns the final state of the 
    ODE after evolving by time Nt*dt. 
    """
    def ode_map(x, *args):
        for i in range(Nt):
            x = rk_loop(ode, x, dt, *args)
        return x
    return ode_map

def ode_map_all_points(ode, x0, Nt, dt, *args):
    """
    This function solves an ODE and also returns the 
    intermediate state vectors during the ODE evolution.
    This is a non-essential function that is only used
    to visualize the dynamics of the model after training.
    """
    xlist = torch.empty([Nt, *x0.shape]).to(x0.device)
    x = x0
    for i in range(Nt):
        x = rk_loop(ode, x, dt, *args)
        xlist[i] = x
    return xlist

