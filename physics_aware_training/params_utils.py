"""
This module contains several functions helping in transforming parameters between normalized
and physical units, and between dictionaries and tensors.

Functions:
--------
    count_params(params_reg, params_range = None, only_trainable = False):
        Counts the number of parameters in a parameter register.
    paramsDict2Tensor(params):
        Concatenates all parameters in a ParameterDict to a single onedimensional tensor.
    paramsTensor2Dict(x, params_reg, params_range = None, only_trainable = True, requires_grad = True):
        Chops up a twodimensional tensor and places each variable into a ParameterDict.
    paramsDictNormalized2Physical(params, params_range):
        Converts a dictionary of normalized parameters into a dictionary of physical parameters.
    paramsDictPhysical2Normalized(params, params_range):
        Converts a dictionary of physical parameters into a dictionary of normalized parameters.
    paramsDictDetach(paramsDict):
        Returns a copy of paramsDict with all elements detached.
    paramsDict2cpu(paramsDict):
        Returns a copy of paramsDict with all elements on CPU.
    inputsPhysical2Normalized(x, input_range):
        Converts inputs x from physical to normalized units.
    inputsNormalized2Physical(x, input_range):
        Converts inputs x from normalized to physical units.
"""

import torch
import torch.nn as nn
import numpy as np

def count_params(params_reg, params_range = None, only_trainable = False):
    """
    Counts the number of parameters in a parameter register,
    or counts the number of trainable parameters, where trainable
    parameters are defined by having a non-zero range.
    """
    n = 0
    for name, shape in params_reg.items():
        if only_trainable:
            if not params_range[name][0] == params_range[name][1]:
                n += np.prod(shape)
        else:
            n += np.prod(shape)
    return n

def paramsDict2Tensor(params):
    """Concatenates all parameters in a ParameterDict to a single onedimensional tensor."""
    
    # awkward way to get the device of parameters in params
    if params:
        device = list(params.values())[0].device
        x = torch.tensor([], device = device)
    else:
        return None
    
    for key, val in params.items():
        x = torch.cat((x, val.flatten()))
    return x

def paramsTensor2Dict(x, params_reg, params_range = None, only_trainable = True, requires_grad = True):
    """
    Chops up a twodimensional tensor of the shape:
    
    [batch_size, number of parameters]
    
    and places each variable into a ParameterDict
    according to the parameter register.
    """
    batch_size = x.shape[0]
    
    params = nn.ParameterDict()
    idx = 0
    for key, shape in params_reg.items():
        if only_trainable:
            if params_range[key][0] == params_range[key][1]:
                continue
        idy = idx + np.prod(shape)
        params[key] = nn.Parameter(x[:, idx:idy].reshape([batch_size, *shape]), requires_grad = requires_grad)
        idx = idy
    return params

def paramsDictNormalized2Physical(params, params_range):
    """Converts a dictionary of normalized parameters into a dictionary of physical parameters."""
    
    # create copy of dictionary
    params = dict(params)
    
    for key, val in params.items():
        params[key] = val * (params_range[key][1]-params_range[key][0]) + params_range[key][0]
    return params

def paramsDictPhysical2Normalized(params, params_range):
    """Converts a dictionary of physical parameters into a dictionary of normalized parameters."""
    
    # create copy of dictionary
    params = dict(params)
    
    for key, val in params.items():
        params[key] = (val - params_range[key][0]) / (params_range[key][1] - params_range[key][0])
    return params

def paramsDictDetach(paramsDict):
    """Returns a copy of paramsDict with all elements detached."""
    detachedDict = {}
    for key, val in paramsDict.items():
        detachedDict[key] = val.detach()
    return detachedDict

def paramsDict2cpu(paramsDict):
    """Returns a copy of paramsDict with all elements on CPU."""
    cpuDict = {}
    for key, val in paramsDict.items():
        cpuDict[key] = val.to('cpu')
    return cpuDict

def inputsPhysical2Normalized(x, input_range):
    """Converts inputs x from physical to normalized units."""
    return (x - input_range[0]) / (input_range[1]-input_range[0])

def inputsNormalized2Physical(x, input_range):
    """Converts inputs x from normalized to physical units."""
    return x * (input_range[1]-input_range[0]) + input_range[0]