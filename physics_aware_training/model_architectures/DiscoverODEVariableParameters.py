'''
NOTE: This digital model architecture uses a trainable 'calibration' model that maps control parameters to the parameters of a system of coupled differential equations, here the equations for the coupled pendula system. 

This version of the code has not been artificially modified to demonstrate auto-calibration - it assumes that our best initial guess for the calibration is that the control parameters map 1:1 to the ODE parameters.
'''

import torch
import torch.nn as nn
import physics_aware_training.digital_twin_utils 
import physics_aware_training.params_utils
from torchdiffeq import odeint


class DiscoverODEVariableParameters(nn.Module):
    def __init__(self, 
                 input_dim, 
                 nparams,
                 output_dim,
                 A,
                 B,
                 input_range,
                 nODEcoefficients,
                 params_range,
                 parameterNunits = [10,10,10],
                 ODEtmax = 2,
                 ODEtimesteps = 60,
                 rtol = 1e-4, 
                 atol = 1e-6,
                encoding_amplitude=1,
                NoCouplingInParameterNet=False): 
        '''
        An ODE with parameter predicted by a neural network as a digital model.
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
            List containing mininimal and maximal value inputs can assume (in physical units).
        nODEcoefficients: int
            Number of ODE coefficients that the parameter network has to predict.
        parameterNunits : list of ints, default: [10,10,10]
            Hidden neurons per layer of the parameter network that predicts the coefficients of the ODE.
            The inputs to the parameter network are the trainable parameters of the experiment,
            the outputs are the coefficients of the ODE.
        ODEtmax : float, default: 2.0
            Time for which the digital model ODE is simulated.
        ODEtimesteps: float, default: 60
            Number of timesteps for which the digital model ODE returns state variable.
        '''
        super(DiscoverODEVariableParameters, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nparams = nparams
        self.parameterNunits = parameterNunits
        self.nODEcoefficients = nODEcoefficients
        self.params_range = params_range
        self.ODEtime = torch.arange(0.,ODEtmax, 1/30)
        self.A = A
        self.B = B
        self.input_range = input_range
        self.rtol = rtol
        self.atol = atol
        self.encoding_amplitude=encoding_amplitude
        self.NoCouplingInParameterNet = NoCouplingInParameterNet
        # parameterNet is a submodel that predicts a matrix of dimensions 
        if len(parameterNunits)>0:
            self.parameterNet = torch.nn.Sequential()
            self.parameterNet.add_module("fcIn", torch.nn.Linear(nparams, parameterNunits[0]))
            for i in range(len(parameterNunits)):
                if i<len(parameterNunits)-1:
                    self.parameterNet.add_module(f"relu{i}", torch.nn.ReLU())
                    self.parameterNet.add_module(f"fc{i}", torch.nn.Linear(parameterNunits[i], parameterNunits[i+1]))
                else:
                    self.parameterNet.add_module(f"relu{i}", torch.nn.ReLU())
                    self.parameterNet.add_module(f"fcOut", torch.nn.Linear(parameterNunits[i], nODEcoefficients))
            self.parameterNN=True
        else:
             #just do a linear model to predict coefficients
            self.parameterNet = torch.nn.Linear(nparams, nODEcoefficients)
            self.parameterNet.weight.data = torch.diag(torch.ones(nparams))
            self.parameterNet.bias.data = torch.zeros(nparams)
        
        if NoCouplingInParameterNet:
            self.parameterNet = torch.nn.Parameter(torch.ones(nparams))
            self.parameterNet_offset = torch.nn.Parameter(torch.zeros(nparams))            
            
           
    def forward(self, x):
        '''
        The digital model's forward pass. It splits the tensor x into inputs and parameters,
        let's the parameters predict the coefficients of an ODE and propagates the inputs 
        through said ODE.
        Parameters:
        -----------
        x : torch.tensor
            Tensor of shape [batch_size, input_dim + nparams] containing the initial angles 
            and trainable parameters of the pendulum experiment in normalized units for one 
            batch.
        Returns:
        outputs : torch.tensor
            Tensor of shape [batch_size, output_dim] containing the final angles of all pendula
            for the whole batch after being propagated through the trainable ODE for time ODEtmax.
        '''

        
        # split x into physical system inputs and parameters

        inputs = x[:, :self.input_dim]
        parameters = x[:, self.input_dim:]
        
                # AUXILIARY PARAMETER NETWORK
        if self.NoCouplingInParameterNet:
            ODEcoefficients = self.parameterNet*parameters+self.parameterNet_offset #element-wise multiplication and offset
        else:
            ODEcoefficients = self.parameterNet(parameters)            
        
        # convert normalized inputs back to physical units
        inputs = physics_aware_training.params_utils.inputsNormalized2Physical(inputs, self.input_range)
        # create tensor for initial velocity and combine with initial position
        v0 = torch.zeros_like(inputs)
        # combine initial positions and velocities in initial conditions for ODE solver

        # extract coefficients from output
        self.ω0 = ODEcoefficients[...,:self.input_dim]
        self.coupling = ODEcoefficients[...,self.input_dim:2*self.input_dim]
        # scale coefficients to physical units
        self.ω0 = self.ω0 * (self.params_range['ω0'][1]-self.params_range['ω0'][0]) + self.params_range['ω0'][0]
        self.coupling = self.coupling * (self.params_range['coupling'][1]-self.params_range['coupling'][0]) + self.params_range['coupling'][0]

        # solve the discovered ODE
        
        x0=torch.stack((inputs*self.encoding_amplitude, v0), dim = -1)
        outputs = odeint(self.func,x0, self.ODEtime.to(inputs.device), method = 'dopri5', rtol = self.rtol, atol = self.atol)[-1,...]
        
        
        
        # convert outputs in physical units back to normalized units
        outputs= physics_aware_training.digital_twin_utils.outputsPhysical2Normalized(outputs, self.A, self.B)
        return outputs[...,0].float()
    
    def func(self,t,x):
        # The coupled pendula are described by a second order ODE. Convert into two first order ODEs
        # represeting position (angle) and angular velocity:
        # x dot = v
        dxdt = x[...,1]
        # x dot dot = ...
        dvdt = -torch.sin(x[...,:,0])*self.ω0 ** 2 \
        + self.coupling.roll(shifts = 1) * (x.roll(dims=-2, shifts = 1)[...,:,0] - x[...,:,0]) \
        + self.coupling * (x.roll(dims=-2, shifts = -1)[...,:,0] - x[...,:,0]) #\
        # So far I have only tried to fit the natural frequencies and coupling constants of the experiment.
        # If other parameters of the ODE shall be fitted, uncomment the following lines.
        #- self.parameters[...,2*input_dim:3*input_dim]*x[...,:,1] \
        #+ self.parameters[...,3*input_dim:4*input_dim]* torch.cos(self.parameters[...,4*input_dim:5*input_dim]*t - self.parameters[...,5*input_dim:6*input_dim]) * torch.cos(x[...,:,0])
        # concatenate position and angular velocity and return
        return torch.cat((dxdt.unsqueeze(-1), dvdt.unsqueeze(-1)), dim = -1)
    
    