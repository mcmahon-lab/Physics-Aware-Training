import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Dataset2D(Dataset):
    """A prototye for datasets akin to http://playground.tensorflow.org/"""
    
    def __init__(self, seed = 0, n = 100):
        """
        Args:
            seed (int): Seed for repeatable performance.
            n (int): Number of different samples.
        """
        self.seed = seed
        self.n = n
        
        self.X = torch.rand([4*self.n, 2]) * 2 - 1
        
    def plot(self):
        #x = torch.linspace(-1, 1, 50)
        #y = torch.linspace(-1, 1, 50)
        #xx, yy = torch.meshgrid(x, y)
        #zz = self.potential_func(xx, yy)
        
        plt.figure(dpi = 150)
        plt.title('Dataset')
        #plt.contourf(xx, yy, self.potential_func(xx, yy), 100, cmap = cm.PuOr, alpha = 0.5, vmin = -1.5, vmax = 1.5)
        plt.scatter(self.X[:,0], self.X[:,1], s = 20, c = self.Y, cmap = cm.PuOr, alpha = 0.5, vmin = -0.2, vmax = 1.2)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.Y[idx]

    
class Quadrants(Dataset2D):
    '''A linearly inseparable datasets assigning different classes to different qudrants.'''
    
    def __init__(self, seed = 0, n = 100):
        super().__init__(seed, n)
        self.Y = self.potential_func(self.X[:, 0], self.X[:, 1])
        
        # separate points by some margin
        self.X = self.X[(torch.abs(self.Y)) > 0.2]
        print(self.X.shape)
        self.X = self.X[:self.n]
        self.Y = self.Y[(torch.abs(self.Y)) > 0.2][:self.n] 
        
        # label points
        self.Y = (self.Y > 0).long()
        
    def potential_func(self, x, y):
        return x*y
    
class AngledLine(Dataset2D):
    '''A linearly separable datasets splitting the cartesian plane by a straight line.'''
    
    def __init__(self, seed = 0, n = 100, angle = 45):
        super().__init__(seed, n)
        self.angle = torch.tensor(np.pi*angle/180.).float()
        self.Y = self.potential_func(self.X[:, 0], self.X[:, 1])
        
        # separate points by some margin
        self.X = self.X[(torch.abs(self.Y)) > 0.2][:self.n]
        self.Y = self.Y[(torch.abs(self.Y)) > 0.2][:self.n] 
        
        # label points
        self.Y = (self.Y > 0).long()
        
    def potential_func(self, x, y):
        return x*(torch.cos(self.angle) + torch.sin(self.angle)) + y*(torch.cos(self.angle) - torch.sin(self.angle)) 
    
class Stripe(Dataset2D):
    def __init__(self, seed = 0, n = 100, angle = 0):
        super().__init__(seed, n)
        self.angle = torch.tensor(np.pi*angle/180.).float()
        self.Y = self.potential_func(self.X[:, 0], self.X[:, 1])
        
        # separate points by some margin
        self.X = self.X[(torch.abs(self.Y)) > 0.2][:self.n]
        self.Y = self.Y[(torch.abs(self.Y)) > 0.2][:self.n] 
        
        # label points
        self.Y = (self.Y > 0).long()
        
    def potential_func(self, x, y):
        x = x*(torch.cos(self.angle) + torch.sin(self.angle))
        y = y*(torch.cos(self.angle) - torch.sin(self.angle))
        return torch.abs(x + y) - 0.7
    
class Circle(Dataset2D):
    def __init__(self, seed = 0, n = 100):
        super().__init__(seed, n)
        self.Y = self.potential_func(self.X[:, 0], self.X[:, 1])
        
        # separate points by some margin
        self.X = self.X[(torch.abs(self.Y)) > 0.2][:self.n]
        self.Y = self.Y[(torch.abs(self.Y)) > 0.2][:self.n] 
        
        # label points
        self.Y = (self.Y > 0).long()
        
    def potential_func(self, x, y):
        return x**2 + y**2 - 0.7
