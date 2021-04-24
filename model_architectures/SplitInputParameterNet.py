import torch
import torch.nn as nn
import lib.digital_twin_utils

class SplitInputParameterNet(nn.Module):
    def __init__(self, 
                 input_dim, 
                 nparams, 
                 output_dim, 
                 parameterNunits = [100,100,100], 
                 internalNunits = [10,10,10]): 
        '''
        Defines network that splits inputs x into physical system input and parameters.
        Inputs are propagated through a "main" neural network whose weights are predicted by an
        auxiliary neural network whose inputs are the parameters. 
        
        Args:
            inputDim (int): dimension of physical system inputs
            outputDim (int): dimension of physical system outputs 
            parameterDim (int): dimension of all physical system parameters combined
            parameterNunits (list of int): defines the number of hidden units per layer in the
                auxiliary parameter network.
            internalDim (int): number of hidden units per layer of the main neural network that 
                propagates physical system inputs
            inputNlayers (int): number of hidden layers of main neural network
        '''
        super(SplitInputParameterNet, self).__init__()
        
        self.input_dim = input_dim
        self.nparams = nparams 
        self.output_dim = output_dim
        self.internalNunits = internalNunits 
        self.inputNlayers = len(internalNunits) 
        
        nparameters = 0
        for i in range(len(internalNunits)-1):
            nparameters += internalNunits[i]*internalNunits[i+1]
            nparameters += internalNunits[i+1]
        
        # parameterNet is a submodel that predicts a matrix of dimensions 
        self.parameterNet = torch.nn.Sequential()
        self.parameterNet.add_module("fcIn", torch.nn.Linear(nparams, parameterNunits[0]))
        for i in range(len(parameterNunits)):
            if i<len(parameterNunits)-1:
                self.parameterNet.add_module(f"relu{i}", torch.nn.ReLU())
                self.parameterNet.add_module(f"fc{i}", torch.nn.Linear(parameterNunits[i], parameterNunits[i+1]))
            else:
                self.parameterNet.add_module(f"relu{i}", torch.nn.ReLU())
                self.parameterNet.add_module(f"fcOut", torch.nn.Linear(parameterNunits[i], nparameters))
                
        # two fully connected input and output layers adjust the input and output dimenstion to
        # the internal dimension
        self.fcIn = nn.Linear(input_dim, internalNunits[0])
        self.fcOut = nn.Linear(internalNunits[-1], output_dim)
        

    def forward(self, x):
        batch_size, _ = x.shape
        
        # initialize matrices for inputNet
        inputNetMatrices = []
        inputNetBiases = []
        for i in range(len(self.internalNunits)-1):
            inputNetMatrices.append([torch.zeros(batch_size, self.internalNunits[i], self.internalNunits[i+1])])
            inputNetBiases.append([torch.zeros(batch_size, self.internalNunits[i+1], 1)])

        # split x into physical system inputs and parameters
        inputs = x[:, :self.input_dim]
        parameters = x[:, self.input_dim:]
        
        # AUXILIARY PARAMETER NETWORK
        parameters = self.parameterNet(parameters)
        
        # fill inputNetMatrices with outputs from parameterNet
        index = 0
        for i in range(len(self.internalNunits)-1):
            index_temp = index
            index += self.internalNunits[i] * self.internalNunits[i+1]
            inputNetMatrices[i] = parameters[:, index_temp:index].reshape(batch_size, self.internalNunits[i+1], self.internalNunits[i])

        # fill inputNetBiases with outputs from parameterNet
        for i in range(len(self.internalNunits)-1):
            index_temp = index
            index += self.internalNunits[i+1]
            inputNetBiases[i] = parameters[:, index_temp:index].reshape(batch_size, self.internalNunits[i+1], 1)
        
        # MAIN INPUT NETWORK
        inputs = self.fcIn(inputs).unsqueeze(-1)
        # MAIN INPUT NETWORK
        for i in range(len(self.internalNunits)-1):
            # apply matrices and biases just filled with outputs from parameterNet
            inputs = torch.bmm(inputNetMatrices[i], inputs)
            inputs += inputNetBiases[i]
            inputs = torch.relu(inputs)
            
        return self.fcOut(inputs.squeeze(-1))
    
class SplitInputParameterObjective(object):
    # define class to smuggle additional arguments into objective function
    def __init__(self, train_loader, test_loader, dt_path, input_dim, nparams, output_dim, **modelargs):
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
            parameterNunits.append(int(trial.suggest_loguniform("Nunits1", 50, 1000)))
        if parameterNlayers == 2:
            parameterNunits.append(int(trial.suggest_loguniform("Nunits1", 50, 1000)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits2", 50, 1000)))
        if parameterNlayers == 3:
            parameterNunits.append(int(trial.suggest_loguniform("Nunits1", 50, 1000)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits2", 50, 1000)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits3", 50, 1000)))
        if parameterNlayers == 4:
            parameterNunits.append(int(trial.suggest_loguniform("Nunits1", 50, 1000)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits2", 50, 1000)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits3", 50, 1000)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits4", 50, 1000)))
        if parameterNlayers == 5:
            parameterNunits.append(int(trial.suggest_loguniform("Nunits1", 50, 1000)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits2", 50, 1000)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits3", 50, 1000)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits4", 50, 1000)))
            parameterNunits.append(int(trial.suggest_loguniform("Nunits5", 50, 1000)))
            
        internalNlayers = trial.suggest_categorical("internalNlayers", [1, 2, 3, 4, 5])
        internalNunits = []
        if parameterNlayers == 1:
            internalNunits.append(int(trial.suggest_loguniform("iNunits1", 10, 100)))
        if parameterNlayers == 2:
            internalNunits.append(int(trial.suggest_loguniform("iNunits1", 10, 100)))
            internalNunits.append(int(trial.suggest_loguniform("iNunits2", 10, 100)))
        if parameterNlayers == 3:
            internalNunits.append(int(trial.suggest_loguniform("iNunits1", 10, 100)))
            internalNunits.append(int(trial.suggest_loguniform("iNunits2", 10, 100)))
            internalNunits.append(int(trial.suggest_loguniform("iNunits3", 10, 100)))
        if parameterNlayers == 4:
            internalNunits.append(int(trial.suggest_loguniform("iNunits1", 10, 100)))
            internalNunits.append(int(trial.suggest_loguniform("iNunits2", 10, 100)))
            internalNunits.append(int(trial.suggest_loguniform("iNunits3", 10, 100)))
            internalNunits.append(int(trial.suggest_loguniform("iNunits4", 10, 100)))
        if parameterNlayers == 5:
            internalNunits.append(int(trial.suggest_loguniform("iNunits1", 10, 100)))
            internalNunits.append(int(trial.suggest_loguniform("iNunits2", 10, 100)))
            internalNunits.append(int(trial.suggest_loguniform("iNunits3", 10, 100)))
            internalNunits.append(int(trial.suggest_loguniform("iNunits4", 10, 100)))
            internalNunits.append(int(trial.suggest_loguniform("iNunits5", 10, 100)))

        name = f"{self.dt_path}_v{trial.number}" #create name with trial index
        value, model_path = lib.digital_twin_utils.train_loop_reg_model(
                                               self.train_loader,
                                               self.test_loader,
                                               name,
                                               self.input_dim,
                                               self.nparams,
                                               self.output_dim,
                                               Model = SplitInputParameterNet,
                                               parameterNunits = parameterNunits, 
                                               internalNunits = internalNunits, 
                                               lr = lr, 
                                               **self.modelargs)
        trial.set_user_attr('model_path', model_path) #save the model path string in NAS study
        return value