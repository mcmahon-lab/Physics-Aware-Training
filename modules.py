from lib.pat_utils import generate_func
import lib.params_utils as pu
from digitaltwin import DigitalTwin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F    # needed for softplus for lagrangian loss

import ipdb

class ExpModule(torch.nn.Module):
    """
    A class that makes experiments act like a torch.nn.Module.

    Attributes:
    -----------
    exp : callable 
        Instance of callable for experiment that the digital model should emulate.
    dt : DigitalTwin
        Instance of DigitalTwin class, representing the digital model used for gradient estimation during
        the backward pass.
    f : torch.autograd.Function
        Autograd function whose forward pass calls ExpModule.exp_normalized and backward pass calls 
        ExpModule.dt
    input_dim : int
        Input dimension of exp.
    output_dim : int
        Output dimension of exp.
    input_range : list
        List containing mininimal and maximal value inputs can take (in physical units). If during
        training inputs exceed this range, a "lagrangian" loss term adds loss to
        keep inputs within the range. The accuracy of the digital model will also most likely
        deteriorate outside this range, as the model is only trained within the range.
    params_reg : dict
        Dictionary with keys corresponding to parameters needed to call exp, and values corresponding
        to the dimension of respective parameter. The keys have to exactly match the name of arguments
        that exp requires.
    hparams : dict
        Dictionary with keys corresponding to hyperparameters needed to call exp and their values.
        The keys have to exactly match the name of arguments that are needed to initialize exp as they will
        be directly passed to exp.__init___.
    params_range : dict
        Dictionary with keys corresponding to parameters needed to call exp, and values corresponding
        to mininimal and maximal value each parameter can assume (in physical units). If during
        training any parameter exceeds this range, a "lagrangian" loss term adds loss to
        keep the parameter within the range. The accuracy of the digital model will also most likely
        deteriorate outside this range, as the model is only trained within the range. If a parameter's
        minimal value equals its maximal value, it will be fixed at this value during training.
    device : string
        Name of device on which digital model is trained ('cpu' or 'cuda')
    differentiable : bool
        Inidicates whether the callable experiment function is auto-differentiable. If "True", gradients
        are calculated by calling the exp function's backward method during trainign. If "False", gradients
        are estimated using a digital model that needs to be trained.
    params : dict
        Dictionary with keys corresponding to trainable parameters of exp, and values corresponding
        to nn.Parameter storing the current value of each parameter. The values are randomly initialized 
        from a uniform distribution of the parameter ranges specified in params_range.
    fixed_params : dict
        Dictionary with keys corresponding to parameters of exp whose range was intialized to zero.
        The dict values are tensor's storing the fixed value of each parameter and intialized to the 
        miminal/maximal value from params_range.
    nparams : int
        Number of parameters stored in all values of params and fixed_params combined.
    ntrainableparams : int
        Number of parameters stored in all values of params. Corresponds to the overall number of trainable
        parameters.
    x_lagrangian : torch.tensor
        A copy of the last input to Exp.forward() which can be used to add loss for all inputs outside of
        input_range during the training loop.
    exp_on_cpu: bool
        Inidicates whether the callable experiment function will be executed on the CPU or not. If True,
        all tensors will be pulled on the CPU before passing them to the experiment. If False, they remain
        on the device specified in the device attribute.
        
    Methods:
    -----------
    __init__(Exp, input_dim, output_dim, input_range, params_reg, params_range, device, hparams, load_dt, differentiable, **kwargs)
        Initializes experiment Exp with given hyperparameters hparams, fills trainable parameters with random
        values according to params_reg and params_range, 
    cat_input_params(xinput):
        Concatenates normalized inputs with current values of normalized parameters (as saved in Exp.params).
    dt_physical(xinput):
        Concatenates xinput in physical units with current PNN module parameters and feeds into digital model.
    exp_normalized(xfull):
        Passes a tensor containing inputs and parameters in normalized units to experiment.
    current_exp(xinput, **kwargs):
        Passes tensor containing inputs in physical to experiment with parameter values as in self.params.
    current_exp_normalized(xinput, **kwargs):
        Passes tensor containing inputs in normalized units to experiment with parameter values as in self.params.
    forward(x):
        Passes tensor containing inputs in physical units through experiment during a forward pass
        or digital model during a backward pass.
    lagrangian(lag_amp = 1.):
        The lagrangian function that can be added to the loss during a training loop.
    """
        
    def __init__(
            self,
            Exp,
            input_dim: int,
            output_dim: int,
            input_range: list = [0,1],
            params_reg: dict = {},
            params_range: dict = {},
            device = 'cuda:0',
            hparams = {},
            load_dt = True,
            differentiable = False,
            exp_on_cpu = True,
            **kwargs):
        """
        Initializes experiment Exp with given hyperparameters hparams, fills trainable parameters with random
        values according to params_reg and params_range, and initializes digital model from DigitalTwin.
            
        To do: Replace hparams by ordered dictionary so filename is uniquely specified
        
        Parameters:
        -----------
        Exp : callable
            Callable class that propagates batched inputs through the experiment. 
            The class needs to have the signature:
                class Exp():
                    __init__(self, [all hyperparameters as arguments], **kwargs)
                    __call__(self, x, [all trainable parameters as arguments])
                        where x is a torch tensor of size [batch_size, input_dim]
                        return torch.tensor of size [batch_size, output_dim]
        input_dim : int
            dimension of inputs to Exp
        output_dim : int
            dimension of outputs from Exp
        input_range : list, default: [0,1]
            List containing mininimal and maximal value inputs can take (in physical units). If during
            training inputs exceed this range, a "lagrangian" loss term adds loss to
            keep inputs within the range. The accuracy of the digital model will also most likely
            deteriorate outside this range, as the model is only trained within the range.
        params_reg : dict, default: {}
            Dictionary with keys corresponding to parameters needed to call exp, and values corresponding
            to the dimension of respective parameter. The keys have to exactly match the name of arguments
            that exp requires.
        params_range : dict, default: {}
            Dictionary with keys corresponding to parameters needed to call exp, and values corresponding
            to mininimal and maximal value each parameter can assume (in physical units). If during
            training any parameter exceeds this range, a "lagrangian" loss term adds loss to
            keep the parameter within the range. The accuracy of the digital model will also most likely
            deteriorate outside this range, as the model is only trained within the range. If a parameter's
            minimal value equals its maximal value, it will be fixed at this value during training.
        device : string, default: 'cuda:0'
            Name of device on which digital model is trained ('cpu' or 'cuda')
        hparams : dict, default: {}
            Dictionary with keys corresponding to hyperparameters needed to call exp and their values.
            The keys have to exactly match the name of arguments that are needed to initialize exp as they will
            be directly passed to exp.__init___.
        load_dt : bool, default: True
            Determines whether to load the digital twin immediately during initialization.
        differentiable : bool, default: False
            Inidicates whether the callable experiment function is auto-differentiable. If "True", gradients
            are calculated by calling the exp function's backward method during trainign. If "False", gradients
            are estimated using a digital model that needs to be trained.
        exp_on_cpu: bool, default: True
            Inidicates whether the callable experiment function will be executed on the CPU or not. If True,
            all tensors will be pulled on the CPU before passing them to the experiment. If False, they remain
            on the device specified in the device attribute.
            
        **kwargs:
        -----------
        retrain : bool, default: False
            Determines whether the digital twin will be retrained (even if a digital twin
            already exists for the given hyperparameters of the experiment). This should for example be
            called when changing the range of trainable parameters.
        retake_data : bool, default: False
            Determines whether new training data for a digital twin will be recorded 
            (even if data already exists for the given hyperparameters of the experiment). 
            This should for example be called when changing the range of trainable parameters.
        **trainargs:
            Any other arguments that specify details about the digital model architecture and how
            it should be trained. Detailed documentation under DigitalTwin.train_mean_dt.
        """
        super(ExpModule, self).__init__() # Initialize self._modules as OrderedDict
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_range = input_range
        self.params_reg = params_reg
        self.hparams = hparams
        self.params_range = params_range
        self.device = device
        self.differentiable = differentiable
        self.exp_on_cpu = exp_on_cpu
        
        # initialize trainable parameters in a parameter dictionary
        self.params = nn.ParameterDict()
        self.fixed_params = {}
        for key, val in self.params_reg.items():
            # fill nn.Parameter of size specified in params_reg with values uniformly distributed over
            # range specified in params_range
            init_values = torch.FloatTensor(*params_reg[key]).uniform_(*self.params_range[key])
            # if parameters have a zero range, put them into a separate dictionary
            if self.params_range[key][0] == self.params_range[key][1]:
                self.fixed_params[key] = init_values.to(device)
            else:
                self.params[key] = nn.Parameter(init_values.to(device), requires_grad = True)
            
        # count and save number of parameters for this module
        self.nparams = pu.count_params(self.params_reg)
        self.ntrainableparams = pu.count_params(self.params_reg, self.params_range, only_trainable = True)
        
        # create experiment 
        self.exp = Exp(**self.hparams, **kwargs)
        
        # load a digital twin for the PNN module. 
        # The digital twin needs to know about  the parameter register beforehand, 
        # so that it can train for random inputs and parameters
        self.dt = DigitalTwin(exp = self.exp,
                        input_dim = self.input_dim,
                        output_dim = self.output_dim,
                        input_range = self.input_range,
                        params_reg = self.params_reg,
                        params_range = self.params_range,
                        fixed_params = self.fixed_params,
                        hparams = self.hparams,
                        device = self.device,
                        **kwargs)
        
        # create auto-differentiable function
        self.f = generate_func(self)

        # optionally load the digital model immediately on initialization of the Exp module
        if load_dt:
            self.dt.load_digital_twin()
    
    
    def cat_input_params(self, xinput):
        '''
        Concatenates normalized inputs with current values of normalized parameters (as saved in Exp.params).
        
        Parameters:
        -----------
        xinput : torch.tensor
            Tensor of size [batch_size, input_dim] containing a batch of normalized inputs (range 0 to 1).
        
        Outputs:
        -----------
        xfull: torch.tensor
            Tensor of size [batch_size, input_dim + ntrainableparameters] containing a batch of given inputs 
            concatenated with batch_size copies of current parametesrs in normalized units (range 0 to 1).
        '''
        batch_size = xinput.shape[0]
        
        if not self.params:
            # in case there are no parameters
            xfull = xinput
        else:
            # convert parameters to normalized units
            paramsDict = pu.paramsDictPhysical2Normalized(self.params, self.params_range)
            # extract all parameters into a one-dimensional tensor
            paramsTensor = pu.paramsDict2Tensor(paramsDict).to(self.device)
            # repeat parameters for each element in batch
            paramsTensor = paramsTensor.repeat(batch_size, 1)
            # concatenate inputs and parameters to one tensor
            xfull = torch.cat((xinput, paramsTensor), dim = 1)
        
        return xfull
    
    def dt_physical(self, xinput):
        '''
        Concatenates xinput in physical units with current PNN module parameters and feeds into digital model.
        
        Parameters:
        -----------
        xinput : torch.tensor
            Tensor of size [batch_size, input_dim] containing a batch of inputs in physical units.
        
        Outputs:
        -----------
        outputs: torch.tensor
            Tensor of size [batch_size, output_dim], containing predictions by digital twin in physical units.
        '''
        xinput = pu.inputsPhysical2Normalized(xinput, self.input_range)
        outputs = self.dt(self.cat_input_params(xinput))
        return outputs 
    
    def exp_normalized(self, xfull):
        '''
        Passes a tensor containing inputs and parameters in normalized units to experiment.
        
        All inputs to experiment are also moved to cpu before passing them along.
        
        Parameters:
        -----------
        xfull : torch.tensor
            Tensor of size [batch_size, input_dim + ntrainableparameters] containing a batch of inputs and
            parameters.
        
        Outputs:
        -----------
        outputs: torch.tensor
            Tensor of size [batch_size, output_dim] containing experimental outcomes in physical units.
        '''
        if not self.differentiable:
            # detach and pull on cpu
            xfull = xfull.detach()
        if self.exp_on_cpu:
            xfull = xfull.cpu()
        
        # split xfull into inputs and parameters
        xinput = xfull[:, :self.input_dim]
        xparams = xfull[:, self.input_dim:]
        
        # scale into physical units
        xinput = pu.inputsNormalized2Physical(xinput, self.input_range)
        paramsDict = pu.paramsTensor2Dict(
            xparams, self.params_reg, self.params_range, 
            only_trainable = True, requires_grad = False)
        paramsDict = pu.paramsDictNormalized2Physical(paramsDict, self.params_range)
        
        if self.exp_on_cpu:
            fixed_params= pu.paramsDict2cpu(self.fixed_params)
        else: 
            fixed_params = self.fixed_params
            
        outputs = self.exp(xinput, **paramsDict, **fixed_params).to(self.device)
        return outputs
    
    def current_exp(self, xinput, **kwargs):
        '''
        Passes tensor containing inputs in physical to experiment with parameter values as in self.params.
        
        All inputs to the experiment are moved to the cpu before passing them on.
        If self.differentiable is "False", parameters and inputs are detached at this point.        
        
        To do: Multiplex parameters to match dimensions of inputs
        
        Parameters:
        -----------
        xinput : torch.tensor
            Tensor of size [batch_size, input_dim] containing a batch of inputs in physical units.
        
        Outputs:
        -----------
        outputs: torch.tensor
            Tensor of size [batch_size, output_dim], containing experimental outcomes in physical units.
        '''
        if self.exp_on_cpu:
            # move all inputs to cpu
            xinput = xinput.cpu()
            params= pu.paramsDict2cpu(self.params)
            fixed_params= pu.paramsDict2cpu(self.fixed_params)
        else:
            params = self.params
            fixed_params = self.fixed_params
        
        # detach for experiment (if not a differentiable experiment)
        if not self.differentiable:
            xinput = xinput.detach()
            params= pu.paramsDictDetach(params)
            
        # pass through experiment
        outputs = self.exp(xinput, **params, **fixed_params, **kwargs).to(self.device)
        return outputs
    
    def current_exp_normalized(self, xinput, **kwargs):
        '''
        Passes tensor containing inputs in normalized units to experiment with parameter values as in self.params.
        
        All inputs to the experiment are moved to the cpu before passing them on.
        If self.differentiable is "False", parameters and inputs are detached at this point.        
        
        To do: Multiplex parameters to match dimensions of inputs
        
        Parameters:
        -----------
        xinput : torch.tensor
            Tensor of size [batch_size, input_dim] containing a batch of inputs in normalized units.
        
        Outputs:
        -----------
        outputs: torch.tensor
            Tensor of size [batch_size, output_dim], containing experimental outcomes in physical units.
        '''
        # scale into physical units
        xinput = pu.inputsNormalized2Physical(xinput, self.input_range)
        outputs = self.current_exp(xinput, **kwargs)
        return outputs
        
    def forward(self, x):
        '''
        Passes tensor containing inputs in physical units through experiment during a forward pass
        or digital model during a backward pass.
        
        Inputs are saved in self.x_lagrangian so any inputs out of input_range can be "punished" during
        the training loop.
        If self.differentiable is "True", then the backward pass is also directed towards the experiment
        instead of the digital model.
        
        Parameters:
        -----------
        x: torch.tensor
            Tensor of size [batch_size, input_dim] containing a batch of inputs in physical units.
        
        Outputs:
        -----------
        outputs: torch.tensor
            Tensor of size [batch_size, output_dim], containing experimental outcomes in physical units.
        '''
        # convert inputs to normalized units
        x = pu.inputsPhysical2Normalized(x, self.input_range)
        # save for lagrangian
        self.x_lagrangian = x
        
        # convert parameters to normalized units
        paramsDict = pu.paramsDictPhysical2Normalized(self.params, self.params_range)
        # convert to tensor and save for lagrangian
        self.params_lagrangian = pu.paramsDict2Tensor(paramsDict)
        if torch.is_tensor(self.params_lagrangian):
            self.params_lagrangian.to(self.device)
        
        x = self.cat_input_params(x)
        if not self.differentiable:
            return self.f(x).to(self.device)
        else:
            return self.exp_normalized(x).to(self.device)
    
    def lagrangian(self, lag_amp = 1.):
        """
        The lagrangian function that can be added to the loss during a training loop.
        
        A softened ReLu function is applied to each input. The loss is applied in normalized inputs
        where the lower bound is always 0 and the upper bound always 1. For each distance delta that
        the inputs are outside of that range, approximately delta*lag_amp loss is added.
        
        Parameters:
        -----------
        lag_amp : float, default: 1.
            Multiplier that determines how much loss is added for inputs outside of input_range.
        
        Outputs:
        -----------
        loss : torch.tensor
            Tensor containing the value of the added loss.
        """
        loss = lag_amp*(clamp_lag(self.x_lagrangian))
        if torch.is_tensor(self.params_lagrangian):
            loss += lag_amp*(clamp_lag(self.params_lagrangian))
        return loss
        

def relu_approx(x, factor=20.0):
    """
    A soft-relu function.
    https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    
    Parameters:
    -----------
    factor : float, default: 20.
        Factor that determines shape of softplus. The larger, the closer to relu, the smaller, the smoother.
    """
    return F.softplus(x*factor)/factor

def clamp_lag(x, low=0.0, high=1.0, factor=20.): 
    """
    Returns loss that is higher the more the values of x exceed the lower and upper threshold.
    
    Approximately returns mean distance of the elements of x from the interval [low, high]. The
    higher factor, the more accurate the returned value is to the mean distance.
    
    Parameters:
    -----------
    x : torch.tensor
        Tensor for which loss is calculated.
    low : float, default: 0.0
        Lower boundary. Any values of x that are lower than low by a value of delta acquire a loss of
        approximately delta.
    high: float, default: 1.0
        Upper boundary. Any values of x that are higher than high by a value of delta acquire a loss of
        approximately delta.
    factor: float, default: 20.0
        Factor determining exact shape of loss function. The larger factor the closer the loss follows 
        a ReLu function.
    """
    return torch.mean(relu_approx(-(x-low), factor) + relu_approx(x-high, factor)) 

class Manifold(torch.nn.Module):
    """
    nn.Module that implements a trainable, elementwise, linear rescaling of data
    of the form x_i -> a_i*x_i + b_i
    """
    def __init__(self, dim, device, factor_init=0.9, offset_init=0.1):
        """
        Parameters:
        -----------
        dim : int
            Dimension of tensors that will be rescaled with this module.
        device : string
            Device on which parameters will be stored. Choices: "cuda" or "cpu".
        factor_init : float
            Sets initialization of a
        offset_init : float
            Sets initialization of b
        """
        super().__init__()
        self.factors = nn.Parameter(factor_init*torch.ones(dim, device=device))
        self.offsets = nn.Parameter(offset_init*torch.zeros(dim, device=device))
    def forward(self, x):
        """
        Returns elementwise rescaling according to factors and offsets.
        
        Parameters:
        -----------
        x : torch.tensor
            Tensor of shape [batch_size, dim] whose elements will be rescaled.
        """
        return x*self.factors + self.offsets

    
class Downsample():
    '''
    Untrainable transformation that downsamples data from input_dim to output_dim where 
    input_dim has to be a multiple of output_dim.
    '''
    def __init__(self, output_dim, input_dim, device):
        """
        This calculates the matrix that takes the parameter vector to a 196-D block-structured 
        vector to modulate the input to the first layer
        
        Parameters:
        -----------
        output_dim : int
            Dimension to which inputs are downsampled.
        input_dim : int
            Dimension to from which inputs are downsampled. Has to be a multiple of output_dim.
        device : string
            Device on which parameters will be stored. Choices: "cuda" or "cpu".
        """
        self.M = torch.zeros([input_dim,output_dim], device = device)
        idx2=0
        idx3=0
        for idx in range(input_dim):
            self.M[idx,idx3]=1.
            idx2+=1
            if np.mod(idx2,input_dim//output_dim)==0:
                idx3+=1
                
    def __call__(self, x):
        return x @ self.M