"""
This module contains code to create simple 2-dimensional machine learning datasets.

Classes:
--------
    Dataset2d(Dataset):
        A prototye for datasets akin to http://playground.tensorflow.org/
    Quadrants(Dataset2D):
        A linearly inseparable datasets assigning different classes to 
        different qudrants in the [-1,1] x [-1,1] plane.
    AngledLine(Dataset2D):
        A linearly separable datasets splitting the cartesian plane by a straight line
        and assigning classes to each point depending on which side of the line they
        are on.
    Stripe(Dataset2D):
        A linearly inseparable datasets containing of three stripes of points
        of class 1, 0, and again 1, crossing the plane at an angle.
    Circle(Dataset2D):
        A linearly inseparable datasets containing circular point cloud in 
        the cartesian plane surrounded by points of a different class.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Dataset2D(Dataset):
    """
    A prototye for datasets akin to http://playground.tensorflow.org/
    
    Attributes:
    -----------
    n : int, default: 100
        Number of points in dataset.
    X : torch.tensor
        Tensor of shape [4*n, 2] containing uniformly distributed points.
    
    Methods:
    plot():
        Produces a scatterplot of the dataset with points colored according
        to their classification.
    -----------
    
    """
    
    def __init__(self, n = 100):
        """Randomly samples 4*n uniformly distributed points in the [-1,1]^2 plane."""
        self.n = n
        
        self.X = torch.rand([4*self.n, 2]) * 2 - 1
        
    def plot(self):
        """Produces a scatterplot of the dataset with points colored according
        to their classification."""
        plt.figure(dpi = 150)
        plt.title('Dataset')
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
    """
    A linearly inseparable datasets assigning different classes to 
    different qudrants in the [-1,1] x [-1,1] plane.
    n points (x,y) are randomly sampled over the plane and for classification,
    each point is assigned a z-value of x*y. If x*y is positive, the point is 
    assigned to class 1, if it is negative, the point is assigned to class 0
    
    
    Attributes:
    -----------
    n : int, default: 100
        Number of points in dataset.
    X : torch.tensor
        Tensor of shape [n, 2] containing the coordinates of each point.
    Y : torch.tensor
        Tensor of shape [n] containing the class of each point.
    
    Methods:
    potential_func(x,y):
        The function according to which the points are classified.
    """
    def __init__(self, n = 100):
        """Randomly initializes n points of the dataset."""
        super().__init__(n)
        self.Y = self.potential_func(self.X[:, 0], self.X[:, 1])
        
        # separate points by some margin
        self.X = self.X[(torch.abs(self.Y)) > 0.2]
        print(self.X.shape)
        self.X = self.X[:self.n]
        self.Y = self.Y[(torch.abs(self.Y)) > 0.2][:self.n] 
        
        # label points
        self.Y = (self.Y > 0).long()
        
    def potential_func(self, x, y):
        """The function according to which the points are classified."""
        return x*y
    
class AngledLine(Dataset2D):
    """
    A linearly separable datasets splitting the cartesian plane by a straight line
    and assigning classes to each point depending on which side of the line they
    are on.
    
    Attributes:
    -----------
    n : int, default: 100
        Number of points in dataset.
    angle : float 
        Angle (in degrees, with respect to the x-axis) at which the line splits 
        the cartesian plane.
    X : torch.tensor
        Tensor of shape [n, 2] containing the coordinates of each point.
    Y : torch.tensor
        Tensor of shape [n] containing the class of each point.
    
    Methods:
    potential_func(x,y):
        The function according to which the points are classified.
    """
    
    def __init__(self, n = 100, angle = 45):
        """
        Randomly initializes n points of the dataset.
        
        Parameters:
        -----------
        n : int, default: 100
            Number of points in dataset.
        angle : float 
            Angle (in degrees, with respect to the x-axis) at which the line splits 
            the cartesian plane.
        """
        super().__init__(n)
        self.angle = torch.tensor(np.pi*angle/180.).float()
        self.Y = self.potential_func(self.X[:, 0], self.X[:, 1])
        
        # separate points by some margin
        self.X = self.X[(torch.abs(self.Y)) > 0.2][:self.n]
        self.Y = self.Y[(torch.abs(self.Y)) > 0.2][:self.n] 
        
        # label points
        self.Y = (self.Y > 0).long()
        
    def potential_func(self, x, y):
        """The function according to which the points are classified."""
        return x*(torch.cos(self.angle) + torch.sin(self.angle)) + y*(torch.cos(self.angle) - torch.sin(self.angle)) 
    
class Stripe(Dataset2D):
    """
    A linearly inseparable datasets containing of three stripes of points
    of class 1, 0, and again 1, crossing the plane at an angle.
    
    Attributes:
    -----------
    n : int, default: 100
        Number of points in dataset.
    angle : float 
        Angle (in degrees, with respect to the x-axis) at which the stripes cross
        the cartesian plane.
    X : torch.tensor
        Tensor of shape [n, 2] containing the coordinates of each point.
    Y : torch.tensor
        Tensor of shape [n] containing the class of each point.
    
    Methods:
    potential_func(x,y):
        The function according to which the points are classified.
    """
    def __init__(self, n = 100, angle = 0):
        """
        Randomly initializes n points of the dataset.
        
        Parameters:
        -----------
        n : int, default: 100
            Number of points in dataset.
        angle : float 
            Angle (in degrees, with respect to the x-axis) at which the stripes cross
            the cartesian plane.
        """
        super().__init__(n)
        self.angle = torch.tensor(np.pi*angle/180.).float()
        self.Y = self.potential_func(self.X[:, 0], self.X[:, 1])
        
        # separate points by some margin
        self.X = self.X[(torch.abs(self.Y)) > 0.2][:self.n]
        self.Y = self.Y[(torch.abs(self.Y)) > 0.2][:self.n] 
        
        # label points
        self.Y = (self.Y > 0).long()
        
    def potential_func(self, x, y):
        """The function according to which the points are classified."""
        x = x*(torch.cos(self.angle) + torch.sin(self.angle))
        y = y*(torch.cos(self.angle) - torch.sin(self.angle))
        return torch.abs(x + y) - 0.7
    
class Circle(Dataset2D):
    """
    A linearly inseparable datasets containing circular point cloud in 
    the cartesian plane surrounded by points of a different class.
    
    n points (x,y) are randomly sampled over the plane and for classification,
    each point is assigned a z-value of x^2 + y^2. If this value is larger than
    0.7, the point is assigned to class 1, if it is negative, the point is 
    assigned to class 0.
    
    Attributes:
    -----------
    n : int, default: 100
        Number of points in dataset.
    X : torch.tensor
        Tensor of shape [n, 2] containing the coordinates of each point.
    Y : torch.tensor
        Tensor of shape [n] containing the class of each point.
    
    Methods:
    potential_func(x,y):
        The function according to which the points are classified.
    """
    def __init__(self, n = 100):
        """Randomly initializes n points of the dataset."""
        super().__init__(n)
        self.Y = self.potential_func(self.X[:, 0], self.X[:, 1])
        
        # separate points by some margin
        self.X = self.X[(torch.abs(self.Y)) > 0.2][:self.n]
        self.Y = self.Y[(torch.abs(self.Y)) > 0.2][:self.n] 
        
        # label points
        self.Y = (self.Y > 0).long()
        
    def potential_func(self, x, y):
        """The function according to which the points are classified."""
        return x**2 + y**2 - 0.7
