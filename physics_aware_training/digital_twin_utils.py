"""
This module contains several functions to facilitate training of the differentiable
digital model.

Functions:
--------
train_reg(model, device, train_loader, optimizer, epoch):
    A single training epoch for a regression model.
test_reg(model, device, test_loader):
    A single testing epoch for a regression model.
np2loaders(x, y, train_ratio=0.9, Nbatch = 200,intout=False):
    Creates training and testing dataloaders for numpy arrays x and y.
outputsPhysical2Normalized(y, A, B):
    Rescales outputs y from physical to normalized units.
outputsNormalized2Physical(y, A, B):
    Rescales outputs y from normalized to physical units.
train_dig_twin(dt_data_path, dt_AB_path, dt_path, input_dim, nparams, 
        output_dim, Model = DNN, device = 'cuda', lr = 0.1, NAS = False, 
        Objective = None, Ntrials = 10, **modelargs):
    Trains and saves a digital model for an experiment.
train_loop_reg_model(train_loader, test_loader, dt_path, input_dim, nparams, 
        output_dim, Model, lr, device, X, Y, gamma = 0.5, step_size = 50, 
        epochs = 300, train_ratio = 0.9, trial = None, **modelargs):
    Training loop for regression model.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader, random_split
from physics_aware_training.model_architectures.DNN import DNN

def train_reg(model, device, train_loader, optimizer, epoch):
    """
    A single training epoch for a regression model.
    
    Parameters:
    -----------
    model : nn.Module
        Regression model.
    device : string
        Determines to which device data is moved. Choices: "cuda" or "cpu".
    train_loader : torch.utils.data.DataLoader
        Dataloader containg the training data.
    optimizer : torch.optim.Optimizer
        Optimizer that determines weight update step.
    epoch : int
        Current epoch.
        
    Returns:
    -----------
    epoch : int
        Current epoch.
    batch_idx : int
        Number of batches that were processed.
    loss : float
        MSE loss on the last processesd training batch (before last update step).
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        #if batch_idx % log_interval == 0:
    return epoch, batch_idx, loss.item()

def test_reg(model, device, test_loader):
    """
    A single testing epoch for a regression model.
    
    Parameters:
    -----------
    model : nn.Module
        Regression model.
    device : string
        Determines to which device data is moved. Choices: "cuda" or "cpu".
    test_loader : torch.utils.data.DataLoader
        Dataloader containg the training data.
        
    Returns:
    -----------
    test_loss : float
        Current MSE loss on the testing dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='mean').item()  # sum up batch loss

    return test_loss

    
def np2loaders(x, y, train_ratio=0.9, Nbatch = 200,intout=False):
    """
    Creates training and testing dataloaders for numpy arrays x and y.
    
    The inputs x and outputs y are chopped into a training and testing set
    
    
    Parameters:
    -----------
    x : np.array 
        Input data of shape [number of samples, input dimension]
    y : np.array
        Output data of shape [number of samples, output dimension]
    train_ratio : float, default: 0.9 
        The ratio of dataset that will be training data (between 0 and 1).
    Nbatch : int, default: 200
        Batch size for the training set. The test batch size is the size of the
        of the whole test set.
    intout : bool, default: False
        If "False", explicitly converts output tensors to float.
        
    Returns:
    -----------
    train_loader : torch.utils.data.DataLoader
        Dataloader containg the training data.
    val_loader : torch.utils.data.DataLoader
        Dataloader containg the testing data.
    """
    Ntotal = x.shape[0]
    Ntrain = int(np.floor(Ntotal*train_ratio))
    train_inds = np.arange(Ntrain)
    val_inds = np.arange(Ntrain, Ntotal)

    X_train = torch.tensor(x[train_inds]).float()
    X_val = torch.tensor(x[val_inds]).float()
    
    if intout == False:
        Y_train = torch.tensor(y[train_inds]).float()
        Y_val = torch.tensor(y[val_inds]).float()
    else:
        Y_train = torch.tensor(y[train_inds])
        Y_val = torch.tensor(y[val_inds])
        
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, Nbatch)
    val_loader = DataLoader(val_dataset, val_dataset.tensors[0].shape[0])

    return train_loader, val_loader

def outputsPhysical2Normalized(y, A, B):
    """Rescales outputs y from physical to normalized units (if A and B correspond to saved normalization constants."""
    return (y-B)/A

def outputsNormalized2Physical(y, A, B):
    """Rescales outputs y from normalized to physical units (if A and B correspond to saved normalization constants."""
    return A*y + B

def train_dig_twin(
        dt_data_path,
        dt_AB_path,
        dt_path,
        input_dim,
        nparams,
        output_dim,
        Model = DNN,
        device = 'cuda',
        lr = 0.1,
        NAS = False,
        Objective = None,
        Ntrials = 10,
        **modelargs):
    """
    Trains and saves a digital model for an experiment.
    
    The digital model will attempt to learn a map from the inputs to the outputs as specified in the data provided
    in dt_data_path. To facilitate learning, the map will be learned in normalized units, where the data is normalized
    to mean = 0 = B and standard deviation = 1 = A with the normalization constants saved in dt_AB_path.
    If NAS is "True", the function will try to import optuna
    which needs to be installed first (https://optuna.readthedocs.io/en/stable/installation.html). The function
    will then create an optuna study with name dt_path + ".db" and save the best found model at dt_path.
    
    Parameters:
    -----------
    dt_data_path : string
        path to ..._mean.npz file with training data.
    dt_AB_path : string
        path to ..._AB.npz file with normalization constants for training data.
    dt_path : string
        path to file in which trained model will be saved.
    input_dim : int
        number of inputs to the experiment per sample
    nparams : int
        number of parameters in the experiment
    output_dim : int
        number of outputs from the experiment per sample
    Model : nn.Module, default: DNN
        Model architecture for digital model. The parameters needed to specify Model completely can be passed
        through **modelargs (documentation in the respective definition, for the default DNN in DNN.py).
    device : string, default: 'cuda'
        Name of device on which digital model is trained ('cpu' or 'cuda')
    lr : float
        Learning rate if using a fixed architecture. May be overwritten by neural architecture search objective.
    NAS : bool
        Determines whether to perform neural architecture search. 
    Objective : Optuna object
        Optuna objective defining a superarchitecture from which architectures are sampled for the neural 
        architecture search.
    Ntrials : int
        Number of architectures sampled from Objective when performing NAS.
    **modelargs:
        Any keyword arguments that are needed to initialize the Model. E.g. for the default DNN, the
        dimension and number of hidden layers can be specified by a list Nunits. All modelargs
        will be passed to the Model constructor (documentation in the respective definition, 
        for the default DNN in DNN.py).
        
    Returns:
    -----------
    None
    """
                   
    # load normalization constants
    A = np.load(dt_AB_path)['A']
    B = np.load(dt_AB_path)['B'] 
    
    # load data
    X = np.load(dt_data_path)['xlist']
    Y = np.load(dt_data_path)['exp_out_list'].mean(axis = 0)
    Y = outputsPhysical2Normalized(Y, A, B)
    
    # create dataloaders
    train_loader, test_loader = np2loaders(X,Y)
    
    if NAS:
        import optuna 

        def create_study(NAS_name, sampler=optuna.samplers.TPESampler(), direction="minimize"):
            #pruner = optuna.pruners.NopPruner()
            pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource = int(10))
            storage = f'sqlite:///{NAS_name}.db' #way to specify an SQL database
            study = optuna.create_study(pruner=pruner, sampler=sampler, 
                    storage=storage, study_name="", load_if_exists=True, direction=direction)
            return study
        
        sampler = optuna.samplers.RandomSampler() #choose the random sampler
        study = create_study(dt_path, sampler=sampler) #create the NAS with the chosen sampler
        
        # perform the architecture search
        objective = Objective(
            train_loader, test_loader, dt_path, input_dim, nparams, output_dim, 
            device = device, X = X, Y = Y, **modelargs)
        study.optimize(objective, n_trials=Ntrials) #run 10 trials

        # find and save the best model
        best_model_path = study.best_trial.user_attrs['model_path']
        best_model = torch.load(best_model_path)
        
        print('saving model ' + dt_path)
        torch.save(best_model, dt_path) 
    else:
        value, model_path = train_loop_reg_model(train_loader, test_loader, dt_path, 
                                                 input_dim, nparams, output_dim, Model,
                                                 lr, device, X, Y, **modelargs)
        
def train_loop_reg_model(
        train_loader, 
        test_loader, 
        dt_path, 
        input_dim,
        nparams,
        output_dim,
        Model,
        lr, 
        device,
        X,
        Y,
        gamma = 0.5, 
        step_size = 50,
        epochs = 300,
        train_ratio = 0.9,
        trial = None,
        **modelargs):
    """
    Training loop for regression model.
    
    Trains Model for given number of epochs on training data provided in train_loader with a step learning rate
    scheduler. Trained model is saved at dt_path and the final testing loss and the model's location is returned.
    Three plots are produced and shown: The first one shows training and testing loss over epoch, the second one 
    shows a scatter plot of ground truth vs model prediction for training data, the third one shows the same for
    testing data.
    
    To do: Remove inputs X, Y and rely on train and test_loader to visualize model performance.
    
    Parameters:
    -----------
    train_loader : torch.utils.data.DataLoader
        Dataloader containg the training data.
    test_loader : torch.utils.data.DataLoader
        Dataloader containg the testing data.
    dt_path : string
        path to file in which trained model will be saved.
    input_dim : int
        number of inputs to the experiment per sample
    nparams : int
        number of parameters in the experiment
    output_dim : int
        number of outputs from the experiment per sample
    Model : nn.Module, default: DNN
        Model architecture for digital model. All parameters needed to specify Model completely can be passed
        through **kwargs.
    lr : float
        Learning rate. 
    device : string
        Determines to which device data is moved. Choices: "cuda" or "cpu".
    X : np.array 
        Input data of shape [number of samples, input dimension] (used for visualizing the model performance)
    Y : np.array
        Output data of shape [number of samples, output dimension] (used for visualizing the model performance)
    gamma : float, default: 0.5
        Learning rate reduction after step_size epochs (StepLR learning rate scheduler)
    step_size : int, default: 50
        Number of epochs after which StepLR reduces learning rate by a factor of gamma
    epochs : int, default: 300
        Number of training epochs
    train_ratio : float, default: 0.9 
        The ratio of dataset that will be training data (between 0 and 1).
    trial : Optuna trial
    **modelargs:
        Any keyword arguments that are needed to initialize the Model. E.g. for the default DNN, the
        dimension and number of hidden layers can be specified by a list Nunits. All modelargs
        will be passed to the Model constructor.
        
    Returns:
    -----------
    None
    """
    # import optuna for NAS
    if trial is not None:
        import optuna
    
    train_ls = []
    test_ls = []
    
    # intialize model, Adam optimizer and learning rate scheduler
    model = Model(input_dim, nparams, output_dim, **modelargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # training epoch
    for epoch in range(1, epochs + 1):
        epoch, batch_idx, loss = train_reg(model, device, train_loader, optimizer, epoch)
        test_loss = test_reg(model, device, test_loader)
        scheduler.step()
        train_ls.append(loss)
        test_ls.append(test_loss)
        
        
        if trial is not None:
            # if doing NAS, report information back to optuna and decide whether to prune
            trial.report(test_loss, epoch)
            if trial.should_prune():
                print('Pruning after ', epoch)
                raise optuna.TrialPruned()
        else:        
            # only print training information when not doing NAS:
            print(f'Epoch: {epoch}, \t Train loss: {loss:.2f}, \t Test loss: {test_loss:.2f}')
    
    torch.save(model, dt_path)
    
    # retrieve number of training and total samples to separate training and testing data for the visualization 
    Ntotal = X.shape[0]
    Ntrain = int(np.floor(Ntotal*train_ratio))
    
    plt.figure(figsize = [12,3])
    # plot training and testing loss over epoch
    plt.subplot(1,3,1)
    plt.title(f'Final test loss: {test_ls[-1]:.2f}')
    plt.plot(train_ls, label = 'train loss')
    plt.plot(test_ls, label = 'test loss')
    plt.legend()
    
    # plot training ground truth vs prediction
    plt.subplot(1,3,2)
    with torch.no_grad():
        plt.plot(Y[:Ntrain].flatten(), model(torch.from_numpy(X[:Ntrain]).float().to(device)).flatten().cpu(), '.', alpha = 0.01)
    plt.title('Train performance')
    plt.xlabel('experimental outcome')
    plt.ylabel('digital twin prediction')
    plt.plot(Y.flatten(), Y.flatten())
    
    # plot testing ground truth vs prediction
    plt.subplot(1,3,3)
    with torch.no_grad():
        plt.plot(Y[Ntrain:].flatten(), model(torch.from_numpy(X[Ntrain:]).float().to(device)).flatten().cpu(), '.', alpha = 0.1)
    plt.title('Test performance')
    plt.xlabel('experimental outcome')
    plt.ylabel('digital twin prediction')
    plt.plot(Y.flatten(), Y.flatten())
    
    # clear model and variables from memory as a precaution to prevent memory buildup
    del model
    del loss
    del optimizer
    del scheduler
    
    return test_loss, dt_path