import torch
import torch.nn as nn
import physics_aware_training.digital_twin_utils 
import physics_aware_training.params_utils
from torchdiffeq import odeint

class DiscoverODE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 nparams,
                 output_dim,
                 A,
                 B,
                 input_range,
                 ODEtmax = 2.,
                 ODEtimesteps = 60): 
        '''
        An ODE with trainable parameters as a digital model.
        
        Parameters:
        -----------
        input_dim : int
            Input dimension of experiment/digital model.
        nparams : int
            Number of trainable parameters in experiment/digital model.
        output_dim : int
            Output dimension of experiment/digital model.
        A : float
            Normalization parameter for experimental data (standard deviation of all training samples).
        B : float
            Normalization parameter for experimental data (mean of all training samples).
        input_range : list
            List containing mininimal and maximal value inputs can take (in physical units).
        ODEtmax : float, default: 2.0
            Time for which the digital model ODE is simulated.
        ODEtimesteps: float, default: 60
            Number of timesteps for which the digital model ODE returns state variable.
        '''
        super(DiscoverODE, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nparams = nparams
        self.ODEtime = torch.linspace(0, ODEtmax, ODEtimesteps)
        self.A = A
        self.B = B
        self.input_range = input_range
        
        # For simplicity, natural frequency and the coupling parameter are described by a scalar.
        # Can be straigtforwardly extended to a vector.
        self.ω0 = nn.Parameter(torch.tensor(10.), requires_grad = True) 
        self.coupling = nn.Parameter(torch.tensor(20.), requires_grad = True) 
                
    def forward(self, x):
        '''
        The digital model's forward pass. It lets inputs x propagates through an ODE with
        trainable coefficients for time ODEtmax.
        
        Parameters:
        -----------
        x : torch.tensor
            Tensor of shape [batch_size, input_dim] containing the initial angles of all
            pendula in normalized units for one batch.
            
        Returns:
        outputs : torch.tensor
            Tensor of shape [batch_size, output_dim] containing the final angles of all pendula
            for the whole batch after being propagated through the trainable ODE for time ODEtmax.
        '''
        batch_size, _ = x.shape
        
        # split x into physical system inputs and parameters
        inputs = x[:, :self.input_dim]
        
        # convert normalized inputs back to physical units
        inputs = physics_aware_training.params_utils.inputsNormalized2Physical(inputs, self.input_range)
        
        # create tensor for initial velocity
        v0 = torch.zeros_like(inputs)
        # combine initial positions and velocities in initial conditions for ODE solver
        x0 = torch.stack((inputs, v0), dim = -1)
        
        # solve the discovered ODE
        outputs = odeint(self.func, x0.type(torch.float64), self.ODEtime.to(inputs.device))[-1,...]
        
        # convert outputs in physical units back to normalized units
        outputs= physics_aware_training.digital_twin_utils.outputsPhysical2Normalized(outputs, self.A, self.B)
        return outputs[...,0].float()
    
    def func(self,t,x):
        # The coupled pendula are described by a second order ODE. Convert into two first order ODEs
        # represeting position (angle) and angular velocity:
        
        # x dot = v
        dxdt = x[...,1]
        # x dot dot = ...
        dvdt = -torch.sin(x[...,:,0])*self.ω0**2 \
        + self.coupling * (x.roll(dims=-2, shifts = 1)[...,:,0] - x[...,:,0]) \
        + self.coupling * (x.roll(dims=-2, shifts = -1)[...,:,0] - x[...,:,0]) #\
        #dvdt = -torch.sin(x[...,:,0])*self.parameters[...,:self.input_dim] \
        #+ self.parameters[...,self.input_dim:2*self.input_dim].roll(shifts = 1) * (x.roll(dims=-2, shifts = 1)[...,:,0] - x[...,:,0]) \
        #+ self.parameters[...,self.input_dim:2*self.input_dim] * (x.roll(dims=-2, shifts = -1)[...,:,0] - x[...,:,0]) #\
        #- self.parameters[...,2*input_dim:3*input_dim]*x[...,:,1] \
        #+ self.parameters[...,3*input_dim:4*input_dim]* torch.cos(self.parameters[...,4*input_dim:5*input_dim]*t - self.parameters[...,5*input_dim:6*input_dim]) * torch.cos(x[...,:,0])

        # concatenate position and angular velocity and return
        return torch.cat((dxdt.unsqueeze(-1), dvdt.unsqueeze(-1)), dim = -1)