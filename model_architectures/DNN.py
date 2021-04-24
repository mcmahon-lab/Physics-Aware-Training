import torch
import torch.nn as nn
import lib.digital_twin_utils

class DNN(nn.Module):
    def __init__(self, input_dim, nparams, output_dim, Nunits = None, batchnorm = False, nlaf = 'relu', **kwargs):
        '''
        Defines configurable deep neural network with fully connected layers and a choice of 
        nonlinear activation functions.

        Args:
            input_dim (int): dimension of input layer
            output_dim (int): dimension of output layer
            Nunits (list of int): dimensions of hidden layers
            batchnorm (bool): determines whether to use batchnorm between each hidden layer.
                The order in which batchnorm is applied is:
                fully connected layer - batchnorm - nonlinear activation function
            nlaf (string): determines the nonlinear activation function. Choices:
                'relu', 'tanh', 'sigmoid'
        '''
        super(DNN, self).__init__()
        
        if Nunits == None:
            Nunits = [100, 100]
        self.batchnorm = batchnorm
        self.nlaf = nlaf
        
        Nunits.insert(0, input_dim + nparams)
        
        self.layers = nn.ModuleList([])
        for i in range(len(Nunits) - 1):
            self.layers.append(nn.Linear(Nunits[i], Nunits[i+1]))
        self.outputlayer = nn.Linear(Nunits[-1], output_dim)
        
        if batchnorm:
            self.batchnorms = nn.ModuleList([])
            for i in range(len(Nunits)-1):
                self.batchnorms.append(nn.BatchNorm1d(Nunits[i+1]))
         

    def forward(self, x):
        '''
        Performs the forward pass through the network.
        
        Args:
            x (float tensor): inputs of dimension [batch_size, input_dim + nparams]
        '''
        
        if self.nlaf == 'relu':
            nlaf = torch.relu
        elif self.nlaf == 'tanh':
            nlaf = torch.tanh
        elif self.nlaf == 'sigmoid':
            nlaf = torch.sigmoid
            
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batchnorm:
                x = self.batchnorms[i](x)
            x = nlaf(x)
        return self.outputlayer(x)

    
class DNNObjective(object):
    # define class to smuggle additional arguments into objective function
    def __init__(self, train_loader, test_loader, dt_path,
                 input_dim, nparams, output_dim, **modelargs):
        '''
        Defines an optuna objective which optimizes hyperparameters drawn from the 
        distribution defined in __call__. 
        
        Args:
            dt_path (string): Location at which best model will be saved
        '''
        self.modelargs = modelargs
        self.dt_path = dt_path
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.input_dim = input_dim
        self.nparams = nparams
        self.output_dim = output_dim

    def __call__(self, trial):
        Nlayers = trial.suggest_categorical("Nlayers", [1, 2, 3, 4, 5])
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
        Nunits = []
        if Nlayers == 1:
            Nunits.append(int(trial.suggest_loguniform("Nunits1", 50, 1000)))
        if Nlayers == 2:
            Nunits.append(int(trial.suggest_loguniform("Nunits1", 50, 1000)))
            Nunits.append(int(trial.suggest_loguniform("Nunits2", 50, 1000)))
        if Nlayers == 3:
            Nunits.append(int(trial.suggest_loguniform("Nunits1", 50, 1000)))
            Nunits.append(int(trial.suggest_loguniform("Nunits2", 50, 1000)))
            Nunits.append(int(trial.suggest_loguniform("Nunits3", 50, 1000)))
        if Nlayers == 4:
            Nunits.append(int(trial.suggest_loguniform("Nunits1", 50, 1000)))
            Nunits.append(int(trial.suggest_loguniform("Nunits2", 50, 1000)))
            Nunits.append(int(trial.suggest_loguniform("Nunits3", 50, 1000)))
            Nunits.append(int(trial.suggest_loguniform("Nunits4", 50, 1000)))
        if Nlayers == 5:
            Nunits.append(int(trial.suggest_loguniform("Nunits1", 50, 1000)))
            Nunits.append(int(trial.suggest_loguniform("Nunits2", 50, 1000)))
            Nunits.append(int(trial.suggest_loguniform("Nunits3", 50, 1000)))
            Nunits.append(int(trial.suggest_loguniform("Nunits4", 50, 1000)))
            Nunits.append(int(trial.suggest_loguniform("Nunits5", 50, 1000)))


        name = f"{self.dt_path}_v{trial.number}" #create name with trial index
        value, model_path = lib.digital_twin_utils.train_loop_reg_model(
                                               self.train_loader,
                                               self.test_loader,
                                               name,
                                               self.input_dim,
                                               self.nparams,
                                               self.output_dim,
                                               Model = DNN,
                                               Nunits = Nunits,
                                               Nlayers = Nlayers,
                                               lr = lr,
                                               trial = trial,
                                               **self.modelargs)
        trial.set_user_attr('model_path', model_path) #save the model path string in NAS study
        return value