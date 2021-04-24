import torch
import torch.nn as nn
from modules import ExpModule
import torch.nn.functional as F

class Module(nn.Module):
    """
    A subclass of nn.Module with additional methods for PNN training.
    
    Methods:
    -----------
    
    """
    def __init__(self):
        pass
    
    def pnnlagrangian(self):
        """
        Finds all ExpModules in pnn.Module and calculates lagrangian loss.
        
        Outputs:
        -----------
        lg_loss : torch.tensor
            Lagrangian loss.
        """
        lg_loss = 0
        for name, module in self.__dict__['_modules'].items():
            if type(module) == ExpModule:
                lg_loss += module.lagrangian(lag_amp = 5)
        return lg_loss

    def train_epoch(self, device, train_loader, optimizer, epoch, log_interval = 1):
        """
        One training epoch for a classifictaion task with NLL and lagrangian loss.
        
        Parameter:
        -----------
        device : string
            Name of device on which digital model is trained ('cpu' or 'cuda')
        train_loader : torch.utils.data.DataLoader
            Dataloader containg the training data.
        optimizer : torch.optim.Optimizer
            Optimizer that determines weight update step.
        epoch : int
            Current training epoch.
        log_interval : int, default: 1
            Determines after how many batches training losses and accuracy are logged.
        
        Outputs:
        -----------
        batch_idx : int
            Number of last batches.
        acc_list : list of floats
            List of training accuracies every log_interval-batches.
        cr_loss_list : list of floats
            List of NLL loss every log_interval-batches.
        lg_loss_list : list of floats
            List of Lagrangian loss every log_interval-batches.
        """
        self.train()
        batch_idx_list = []
        acc_list = []
        cr_loss_list = []
        lg_loss_list = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)

            # the only addition to a conventional pytorch training loop is the lagrangian loss term
            # that keeps parameters and inputs within the user-defined ranges
            lg_loss = self.pnnlagrangian()
            cr_loss = F.nll_loss(output, target)
            loss = lg_loss + cr_loss

            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                batch_idx_list.append(epoch + batch_idx / len(train_loader))
                cr_loss_list.append(cr_loss)
                lg_loss_list.append(lg_loss)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                acc_list.append(correct / len(target))
                
        return batch_idx_list, acc_list, cr_loss_list, lg_loss_list


    def test_epoch(self, device, test_loader):
        """
        One testing epoch for a classifictaion task with NLL and lagrangian loss.
        
        Parameter:
        -----------
        device : string
            Name of device on which digital model is trained ('cpu' or 'cuda')
        test_loader: torch.utils.data.DataLoader
            Dataloader containg the testing data.
        
        Outputs:
        -----------
        test_loss : float
            Average NLL loss over whole testing dataset.
        test_acc : float
            Fraction of correctly classified samples in testing dataset.
        """
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)

        return test_loss, test_acc

    
class Parameter(nn.Parameter):
    """
    Subclass of nn.Parameter that additionally stores 
    an upper and lower bound for each parameter.
    """
    def __new__(cls, data=None, requires_grad=True, limits=None):
        """
        Parameter:
        -----------
        data : torch.tensor
            Tensor containing the values of the parameter.
        requires_grad : bool
            Determines whether computation tree is built for par
        limits : list of floats
            Specifies the lower and upper bound of the parameter that can be
            used during PNN training to keep the parameter within those limits.
        """
        param = nn.Parameter.__new__(cls, data = data, requires_grad=requires_grad)
        param.limits = limits
        return param
    
    def __repr__(self):
        return 'Parameter containing:\n' + super(nn.Parameter, self).__repr__() + '\tLimits: ' + str(self.limits)

def get_cpu_state_dict(model):
    """Save all state dict variables except the digital twins ones."""
    cpu_state_dict = {}
    for key, val in model.state_dict().items():
        if not '.dt' in key:
            cpu_state_dict[key] = val.cpu()
    return cpu_state_dict

def set_cpu_state_dict(model, pretrained_dict):
    """Set all state dict variables except the digital twins ones."""
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model