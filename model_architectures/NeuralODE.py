import torch
import torch.nn as nn
import lib.digital_twin_utils
from torchdiffeq import odeint

class NeuralODE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 nparams,
                 output_dim, 
                 parameterNunits = [10,10,10],
                 internalDim = 10,
                 ODEtmax = 1,
                 ODEtimesteps = 2): 
        '''
        Defines neural ODE digital twin
        
        Args:
            input_dim (int): dimension of physical system inputs
            output_dim (int): dimension of physical system outputs 
            nparams (int): dimension of all trainable physical system parameters combined
            internalDim (int): internal dimension in which the ODE works
            T (int): Time over which to solve neural ODE
        '''
        super(NeuralODE, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nparams = nparams
        self.parameterNunits = parameterNunits
        self.internalDim = internalDim
        self.ODEtime = torch.linspace(0, ODEtmax, ODEtimesteps)
        
        # parameterNet is a submodel that predicts a matrix of dimensions 
        self.parameterNet = torch.nn.Sequential()
        self.parameterNet.add_module("fcIn", torch.nn.Linear(nparams, parameterNunits[0]))
        for i in range(len(parameterNunits)):
            if i<len(parameterNunits)-1:
                self.parameterNet.add_module(f"relu{i}", torch.nn.ReLU())
                self.parameterNet.add_module(f"fc{i}", torch.nn.Linear(parameterNunits[i], parameterNunits[i+1]))
            else:
                self.parameterNet.add_module(f"relu{i}", torch.nn.ReLU())
                self.parameterNet.add_module(f"fcOut", torch.nn.Linear(parameterNunits[i], internalDim**2))
                
        # two fully connected input and output layers adjust the input and output dimenstion to
        # the internal dimension
        self.fcIn = nn.Linear(input_dim, internalDim) 
        self.fcOut = nn.Linear(internalDim, output_dim)
        

    def forward(self, x):
        batch_size, _ = x.shape
        
        # split x into physical system inputs and parameters
        inputs = x[:, :self.input_dim]
        parameters = x[:, self.input_dim:]
        
        # AUXILIARY PARAMETER NETWORK
        parameters = self.parameterNet(parameters)
        
        # Neural ODE with parameter kernel
        inputs = self.fcIn(inputs).unsqueeze(-1)
        # create coupling matrix
        self.A = parameters.reshape(batch_size, self.internalDim, self.internalDim)
        self.A = self.A.type(torch.float64)
        # solve ODE with above coupling matrix over timegrid ODEtime
        inputs = odeint(self.func, inputs.type(torch.float64), self.ODEtime.to(inputs.device))[-1,...]
        return self.fcOut(inputs.squeeze(-1).float())
    
    def func(self,t,x):
        return torch.tanh(torch.bmm(self.A, x))
    
class NeuralODEObjective(object):
    # define class to smuggle additional arguments into objective function
    def __init__(self, train_loader, test_loader, dt_path, 
                 input_dim, nparams, output_dim, **modelargs):
        self.modelargs = modelargs
        self.dt_path = dt_path
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.input_dim = input_dim
        self.nparams = nparams
        self.output_dim = output_dim

    def __call__(self, trial):
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
        
        parameterNlayers = trial.suggest_categorical("parameterNlayers", [1, 2, 3, 4, 5])
        parameterNunits = []
        if parameterNlayers == 1:
            parameterNunits.append(int(trial.suggest_loguniform("Nunits1", 50, 500)))
        if parameterNlayers == 2:
            parameterNunits.append(int(trial.suggest_loguniform("Nunits1", 50, 500)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits2", 50, 500)))
        if parameterNlayers == 3:
            parameterNunits.append(int(trial.suggest_loguniform("Nunits1", 50, 500)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits2", 50, 500)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits3", 50, 500)))
        if parameterNlayers == 4:
            parameterNunits.append(int(trial.suggest_loguniform("Nunits1", 50, 500)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits2", 50, 500)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits3", 50, 500)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits4", 50, 500)))
        if parameterNlayers == 5:
            parameterNunits.append(int(trial.suggest_loguniform("Nunits1", 50, 500)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits2", 50, 500)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits3", 50, 500)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits4", 50, 500)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits5", 50, 500)))
            
        internalDim = int(trial.suggest_loguniform("internalDim", 10, 300))

        name = f"{self.dt_path}_v{trial.number}" #create name with trial index
        value, model_path = lib.digital_twin_utils.train_loop_reg_model(
                                               self.train_loader,
                                               self.test_loader,
                                               name,
                                               self.input_dim,
                                               self.nparams,
                                               self.output_dim,
                                               Model = NeuralODE,
                                               parameterNunits = parameterNunits, 
                                               internalDim = internalDim,
                                               lr = lr, 
                                               **self.modelargs)
        trial.set_user_attr('model_path', model_path) #save the model path string in NAS study
        return value