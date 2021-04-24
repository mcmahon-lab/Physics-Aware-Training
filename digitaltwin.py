import numpy as np
import torch
import os
from lib.digital_twin_utils import *
import lib.params_utils as pu

class DigitalTwin(torch.nn.Module):
    """ 
    A class to handle data acquisition, training procedure and usage of differentiable digital model.
    
    Attributes:
    -----------
    exp : callable 
        Callable for experiment that the digital model should emulate.
    input_dim : int
        Input dimension of exp.
    output_dim : int
        Output dimension of exp.
    input_range : list
        List containing mininimal and maximal value inputs can take (in physical units).
    params_reg : dict
        Dictionary with keys corresponding to trainable parameters needed to call exp and values
        corresponding to the dimension of respective parameter.
    fixed_params : dict
        Dictionary with keys corresponding to fixed parameters needed to call exp and the values
        of those fixed parameters as torch.FloatTensor.
    hparams : dict
        Dictionary with keys corresponding to hyperparameters needed to call exp and their values.
    Nrepeat : int
        Number of repetitions from which each mean output value is determined.
    nparams : int
        Number of trainable parameters as specified in params_reg
    dt_path : string
        Path to saved model.
    dt_data_path : string
        Path to .npz file that contains training data.
    dt_AB_path : string
        Path to .npz file that contains normalization constants for training data and digital model.
    device : string
        Name of device on which digital model is trained ('cpu' or 'cuda')
    mean_dt : nn.Module
        Digital model that emulates the input-output transformation of exp.
    A : float
        Standard deviation of exp output values over training data.
    B : float
        Mean of exp output values over training data.
    trainargs:
        Any other arguments that specify details about the digital model architecture and how
        it should be trained. Detailed documentation under DigitalTwin.train_mean_dt.
        
    Methods:
    -----------
    forward(x):
        Performs a forward pass through the digital model.
    load_digital_twin:
        Loads digital model from self.dt_path. 
    load_AB:
        Loads normalization constants for digital model.
    train_mean_dt(retake_data = False, Nx = 1e4, **trainargs)
        Trains a mean digital twin for the current set of hyperparameters.
    take_dt_data(Nx = 1e4):
        Acquires data to train a digital twin.
    get_dt_path:
        Returns the path to the digital twin model.
    get_dt_data_path:
        Returns the path to the digital twin data.
    get_dt_AB_path
        Returns the path for the digital twin normalization constants.
    hparams2filename:
        Concatenates all hyperparameter keys and values in self.hparams to filename in alphabetical order.
    remove_optuna_study:
        Returns exisiting optuna study with the corresponding name.
    """
    def __init__(self,
                 exp,
                 input_dim: int,
                 output_dim: int,
                 input_range: list,
                 params_reg: dict,
                 params_range: dict,
                 fixed_params: dict,
                 hparams: dict,
                 retrain = False,
                 retake_data = False,
                 Nrepeat = 2,
                 device = 'cuda',
                 **trainargs):
        """
        Parameters:
        -----------
        exp : callable 
            Callable for experiment that the digital model should emulate.
        input_dim : int
            Input dimension of exp.
        output_dim : int
            Output dimension of exp.
        input_range : list
            List containing mininimal and maximal value inputs can take (in physical units).
        params_reg : dict
            Dictionary with keys corresponding to trainable parameters needed to call exp and values
            corresponding to the dimension of respective parameter.
        fixed_params : dict
            Dictionary with keys corresponding to fixed parameters needed to call exp and the values
            of those fixed parameters as torch.FloatTensor.
        hparams : dict
            Dictionary with keys corresponding to hyperparameters needed to call exp and their values.
        retrain : bool
            Determines whether to retrain a digital model regardless of whether a model with the specified name
            could be loaded. If True, self.train_mean_dt will be called on initialization.
        retake_data : bool
            Determines whether to retake training data for digital model regardless of whether a data file with
            the specified name could be loaded.
        Nrepeat : int
            Number of repetitions from which each mean output value is determined.
        device : string, default: 'cuda'
            Name of device on which digital model is trained ('cpu' or 'cuda')
        **trainargs:
            Any other arguments that specify details about the digital model architecture and how
            it should be trained. Detailed documentation under DigitalTwin.train_mean_dt.
        """
        super(DigitalTwin, self).__init__() 
        self.exp = exp
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_range = input_range 
        self.params_reg = params_reg
        self.params_range = params_range
        self.fixed_params = fixed_params
        self.hparams = hparams 
        self.Nrepeat = Nrepeat
        self.trainargs = trainargs
        self.nparams = pu.count_params(self.params_reg, self.params_range, only_trainable = True)
        
        self.dt_path = self.get_dt_path()
        self.dt_data_path = self.get_dt_data_path()
        self.dt_AB_path = self.get_dt_AB_path()
        
        self.device = device
        
        if retrain:
            try:
                print('Trying to remove previous optuna studies...')
                os.remove(f"{self.dt_path}.db")
                print(f'Successfully removed {self.dt_path}.db.')
            except:
                print(f"{self.dt_path}.db don't exist")
            self.train_mean_dt(retake_data)
        
        
    def forward(self, x):
        """
        Performs a forward pass through the digital model.
        
        Parameters:
        -----------
        x : torch.tensor: 
            Normalized inputs of shape [batch_size, input_dim].
            
        Returns:
        -----------
        y : torch.tensor
            Outputs of shape [batch_size, output_dim] in physical units.
        """
        y = outputsNormalized2Physical(self.mean_dt(x), self.A, self.B)
        return y
    
            
    def load_digital_twin(self):
        """
        Loads digital model from self.dt_path. 
        
        If loading fails, calls self.train_mean_dt to train a digital model.
        
        Parameters:
        -----------
        None
            
        Returns:
        -----------
        None
        """
        try:
            self.mean_dt = torch.load(self.dt_path, map_location='cpu').to(self.device)
            self.mean_dt.eval() 
            # load normalization constants
            self.load_AB()
        except:
            print('Could not load digital twin. Proceeding to train digital twin...')
            self.train_mean_dt()
            self.mean_dt = torch.load(self.dt_path, map_location='cpu').to(self.device)
            
        print('Loaded digital twin ', self.dt_path)
            
    
    def load_AB(self):
        """Loads normalization constants for digital model."""
        self.A = torch.from_numpy(np.load(self.dt_AB_path)['A'])
        self.B = torch.from_numpy(np.load(self.dt_AB_path)['B'])
            
            
    def train_mean_dt(self, retake_data = False, Nx = 10000, **trainargs):
        """
        Trains a mean digital twin for the current set of hyperparameters.
        
        Parameters:
        -----------
        retake_data : bool
            Determines whether to retake training data for digital model regardless of whether a data file with
            the specified name could be loaded.
        Nx : int, default: 10000
            Amount of training samples that is acquired if data is retaken.
            
        **trainargs:
        -----------
        Model : nn.Module, default: DNN
            Module defining architecture of digital model. 
        lr : float, default: 0.1
            Learning rate used during digital model training.
        epochs : int, default: 300
            Number of epochs to train (each) digital model.
        NAS : bool, default: False
            Determines whether to perform a Neural Architecture Search (NAS) during digital model training.
            If True, requires optuna to be installed and an Objective to be specified.
        Objective : object, default: None
            Optuna training objective that defines superarchitecture over which NAS is performed.
        Ntrials : int, default: 10
            Specifies how many architectures are trained during the NAS.
        **modelargs:
            Any keyword arguments that are needed to initialize the Model. E.g. for the default DNN, the
            dimension and number of hidden layers can be specified by a list Nunits.
        """
        if retake_data:
            self.take_dt_data(Nx)
            self.dt_data = np.load(self.dt_data_path)
            print('Saved data', self.dt_data_path)
        else:
            try:
                self.dt_data = np.load(self.dt_data_path)
                # load normalization constants
                self.load_AB()
            except:
                print('Could not find training data for digital twin. Proceeding to take training data for digital twin...')
                self.take_dt_data()
                self.dt_data = np.load(self.dt_data_path)
        print('Loaded data', self.dt_data_path)
        
        if not os.path.exists('dt_models'):
            os.makedirs('dt_models')
        
        train_dig_twin(self.dt_data_path,
                          self.dt_AB_path,
                          self.dt_path,
                          self.input_dim,
                          self.nparams,
                          self.output_dim,
                          device = self.device,
                          **self.trainargs,
                          **trainargs)
        
        print('Trained digital twin and saved at ', self.dt_path)
        self.mean_dt = torch.load(self.dt_path, map_location='cpu').to(self.device)
        
    
    def take_dt_data(self, Nx = 10000):
        """
        Acquires data to train a digital twin.
        
        The input data to the experiment is uniformly distributed over all inputs and parameters.
        Each input is passed through the experiment self.Nrepeats times.
        The acquired data is saved in a .npz file named after the experiment and the values of
        hyperparameters as specified in the doc string of DigitalTwin.hparams2filename with the
        attached ending "_data.npz".
        The file contains two arrays:
            xlist of shape [Nx, number of inputs + trainable parameters], containing a single
                copy of the inputs to the experiment in physical units
            exp_out_list of shape [self.Nrepeats, Nx, number of outputs], containing self.Nrepeats
                copies of the experimental output in physical units
        A second .npz file with the same naming convention but the ending "_AB.npz" is saved and 
        contains the mean and standard deviation of the training data to faciliatate normalization
        when training the digital model. This file contains two variables:
            A: standard deviation of exp_out_list
            B: mean of exp_out_list
        
        Parameters:
        -----------
        Nx : int, default: 10000
            Amount of training samples that is acquired if data is retaken.
        """
        if not os.path.exists('dt_data'):
            os.makedirs('dt_data')
            
        # create unformly distributed inputs
        xlist = torch.FloatTensor(Nx, self.input_dim).uniform_(0, 1)
        
        # create uniformly distributed parameters
        paramstensor = torch.FloatTensor(Nx, self.nparams).uniform_(0, 1)
        
        # convert parameter tensor to dictionary
        paramsdict = pu.paramsTensor2Dict(
            paramstensor, self.params_reg, self.params_range, 
            only_trainable = True, requires_grad = False)
        # convert normalized to physical units
        paramsdict = pu.paramsDictNormalized2Physical(paramsdict, self.params_range)
        xlistPhysical = pu.inputsNormalized2Physical(xlist, self.input_range)
        
        # pull fixed parameters on cpu
        fixed_params = pu.paramsDict2cpu(self.fixed_params)
        
        # take experimental data
        exp_out_list = []
        for i in range(self.Nrepeat):
            exp_out_list.append(
                self.exp(xlistPhysical, **paramsdict, **fixed_params))
        exp_out_list = torch.stack(exp_out_list)
        xlist = torch.cat((xlist, paramstensor), dim = 1)
        
        # save experimental data
        np.savez(self.dt_data_path,
                 xlist = xlist,
                 exp_out_list = exp_out_list)
        
        print('Saved data', self.dt_data_path)
        
        # save normalization constants
        self.B = torch.mean(exp_out_list).item()
        self.A = torch.std(exp_out_list-self.B).item()
        np.savez(self.dt_AB_path,
                 A = self.A,
                 B = self.B)
        
            
    def get_dt_path(self):
        """Returns the path to the digital twin model."""
        filename = []
        filename.append(self.hparams2filename())
        filename.append('.p')
        filename = ''.join(filename)
        dt_path= os.path.join('dt_models', filename)
        
        return dt_path
    
    def get_dt_data_path(self):
        """Returns the path to the digital twin data."""
        filename = []
        filename.append(self.hparams2filename())
        filename.append('_data.npz')
        filename = ''.join(filename)
        dt_data_path= os.path.join('dt_data', filename)
        
        return dt_data_path
    
    def get_dt_AB_path(self):
        """Returns the path for the digital twin normalization constants."""
        filename = []
        filename.append(self.hparams2filename())
        filename.append('_AB.npz')
        filename = ''.join(filename)
        dt_data_path= os.path.join('dt_data', filename)
        
        return dt_data_path
    
    def hparams2filename(self):
        """
        Concatenates all hyperparameter keys and values in self.hparams to filename in alphabetical order.
        
        The form of the returned string is:
        [name of exp class]_in[input_dim]_out[output_dim]_[key0][val0]_ ... _[key-1][val-1]
        where key and val are the keys and values in self.hparams.
        """
        filename = []
        filename.append(self.exp.__class__.__name__)
        filename.append(f'_mean_in{self.input_dim}_out{self.output_dim}')
        
        for key, val in sorted(self.hparams.items()):
            filename.append('_')
            filename.append(str(key))
            filename.append(str(val))
            
        return ''.join(filename)

    def remove_optuna_study(self):
        """Returns exisiting optuna study with the corresponding name."""
        print('Trying to remove previous optuna studies...')
        os.remove(f"{self.dt_path}.db")
        print(f'Successfully removed {self.dt_path}.db.')
