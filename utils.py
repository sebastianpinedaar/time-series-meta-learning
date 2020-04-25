import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def rmse_loss(y_pred, y_true):
    loss = ((y_pred - y_true)**2).mean()
    return loss



def smape(y_true, y_pred):
    
    #y_true = torch.from_numpy(y_true)
    num=torch.abs(y_true-y_pred)
    den=torch.abs(y_true)+torch.abs(y_pred)
    
    return torch.sum(torch.mul(num, 1/den))


def scale(x, x_max_train):
    
    x_max_train = np.tile(x_max_train, x.shape[-1]).reshape(x.shape, order="F")
    
    return np.divide(x, x_max_train)


def rescale (x, x_max_train):
    
    x_max_train = np.tile(x_max_train, x.shape[-1]).reshape(x.shape, order="F")
    return np.multiply(x, x_max_train)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

