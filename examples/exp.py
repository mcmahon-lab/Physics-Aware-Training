import torch
import numpy as np
from scipy import integrate
from torchdiffeq import odeint
    
class CoupledPendula():
    '''
    This class propagates inputs through a system of coupled pendula.
    Machine learning inputs will be encoded in the inital positions of the
    pendula, the position of the pendula at time T corresponds to the outputs.
    In this simulation, the natural frequency, damping, drive amplitude, drive phase,
    coupling, and intial velocity are in principle all trainable parameters.
    For the sake of experimental realizability and the difficulty of building digital
    models for all parameters at once, we restrict ourselves to training the initial
    angle of all pendula.
    '''
    def __init__(self, Tmax, rtol = 1e-4, atol = 1e-6, **kwargs):
        '''
        All hyperparameters go into the __init__ subroutine and all preparations
        that need to be done once before the experiment starts, like connecting
        and setting up external devices.
        Because this experiment is just a simulation, we only need to set hyperparameters
        
        Parameters:
        -----------
        Tmax : float
            Simulation time after which final position is measured [s].
        rtol : float
            Relative tolerance of ODE solver (larger = faster).
        atol : float
            Aboslute tolerance of ODE solver (larger = faster).
        '''
        
        self.Tmax = Tmax
        self.rtol = rtol
        self.atol = atol
        
    def __call__(self, 
                 
                 x_input, 
                 
                 ω0, # eigenfrequency of each pendulum [s]
                 ωd, # driving frequency of each pendulum [s]
                 Ad, # driving amplitude of each pendulum [rad]
                 v0, # inital velocity of each pendulum [rad/s]
                 coupling, # universal coupling between pendula [rad/s^2 / rad]
                 γ, # universal damping of pendula [(rad/s) / (rad/s) / s]
                 encoding_amplitude, # multiplier for inputs
                 phid, # driving phase
                 
                 full = False,
                 **kwargs): # optionally save the full time evolution of all pendula
        '''
        All trainable parameters go into the __call__ subroutine.
        The experimental function should be able to handle batched inputs and parameters,
        i.e. inputs of the shape [batch_size, input_dim] and parameters of the shape 
        [batch_size, parameter dimension]. In this case, each row of inputs will be
        matched up with their respective row of parameters and each input should be
        propagated through the experiment with their respective parameters.
        
        Alternatively, parameters should be able to be of size [1, parameter dimension].
        In this case all parameters should be propagated through the experiment with the
        same set of parameters.
        
        The simulation of this experiments follows equation (1) of:
        Thakur, R. B., English, L. Q., & Sievers, A. J. (2007). 
        Driven Intrinsic Localized Modes in a Coupled Pendulum Array. 
        arXiv preprint arXiv:0710.3373.
        https://arxiv.org/abs/0710.3373
        
        Parameters:
        -----------
        x_input : torch.tensor
            Tensor of shape [batch_size, number of pendula] containing initial angles of all
            pendula.
        ω0 : torch.tensor
            Eigenfrequency of each pendulum [s]. 
        ωd : torch.tensor
            Driving frequency of each pendulum [s]
        Ad : torch.tensor
            Driving amplitude of each pendulum [rad]
        v0 : torch.tensor
            Inital angular velocity of each pendulum [rad/s]
        coupling : torch.tensor 
            Set of coupling constants that specified bidirectional coupling between each pair
            of neighboring pendula [rad/s^2 / rad]
        γ : torch.tensor
            Unique damping constants for each pendulum [(rad/s) / (rad/s) / s]
        encoding_amplitude : torch.tensor
            Universal multiplier for initial amplitude that rescales initial angles and hence
            controls the amount of nonlinearity.
        phid : torch.tensor
            Phase of drive [rad].
        full : bool
            If "True", not only the final position of each pendulum is returned but rather the
            full time evolution of the angle and angular velocity.
            
        Outputs:
        -----------
        final_position : torch.tensor
            Tensor containing the angle of each pendulum at time T of size [batch_size, number of pendula].
            If full==True, then it is of shape [time_step, batch_size, number of pendula, 2] where index 0 
            of the last dimension is the final angle and index 1 is angular velocity.
        '''
        
        '''
        # Numpy version if we decide to switch to numpy!
        self.ω0 = torch.from_numpy(ω0)
        self.ωd = torch.from_numpy(ωd)
        self.Ad = torch.from_numpy(Ad)
        self.coupling = torch.from_numpy(coupling)
        self.γ = torch.from_numpy(γ)
        self.phid = torch.from_numpy(phid)
        
        x_input = torch.from_numpy(x_input)
        v0 = torch.from_numpy(v0)
        '''
        
        self.ω0 = ω0
        self.ωd = ωd
        self.Ad = Ad
        self.coupling = coupling
        self.γ = γ
        self.phid = phid
        
        # define time grid
        t = torch.arange(0.,self.Tmax, 1/30)

        batch_size, _ = x_input.shape

        if len(v0.shape) == 1:
            v0 = v0.repeat(batch_size,1)
        # combine initial positions and velocities in initial conditions for ODE solver
        x0 = torch.stack((x_input*encoding_amplitude,v0), dim = -1)
        # solve ODE
        x = odeint(self.func, x0, t, method = 'dopri5', rtol = self.rtol, atol = self.atol)
        
        if full:
            return x
        
        # return pendula position at latest time step
        final_position = x[-1, ..., 0]
        return final_position
    
    def func(self,t,x):
        # The coupled pendula are described by a second order ODE. Convert into two first order ODEs
        # represeting position (angle) and angular velocity:
        
        q = x[..., 0] # position (angle)
        p = x[..., 1] # angular velocity
        
        # x dot = v
        dxdt = p
        # x dot dot = ...
        dvdt = -torch.sin(q)*self.ω0**2 \
        + self.coupling.roll(shifts = 1) * (q.roll(dims=-1, shifts = 1) - q) \
        + self.coupling * (q.roll(dims=-1, shifts = -1) - q) \
        - self.γ*p \
        + self.Ad * torch.cos(self.ωd*t - self.phid) * torch.cos(q)

        # concatenate position and angular velocity and return
        return torch.cat((dxdt.unsqueeze(-1), dvdt.unsqueeze(-1)), dim = -1)
    
    def find_nearest(self, tensor, target):
        '''
        Find index of value in tensor that is closest to each value in target.
        
        Parameters:
        -----------
        tensor : tensor
            Tensor of values, Shape [n].
        target : tensor
            Tensor of targets, [1, batch_size].
            
        Outputs:
        -----------
        indices : tensor [long]
            Index of element in tensor that is closest to each element in target tensor. 
            Shape: [batch_size].
        '''
        tensor = tensor.repeat(len(target), 1)
        target = target.repeat(1, tensor.shape[1])
        indices = torch.argmin(np.abs(tensor - target), dim = -1)
        return indices.long()