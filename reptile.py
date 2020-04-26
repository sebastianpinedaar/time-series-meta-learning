import load_data as ld
import os
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable#
import matplotlib.pyplot as plt
import torch
from pytorchtools import EarlyStopping
from utils import *


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def batcher(dataset, batch_size, infinite=False):
    while True:
        x, y = dataset
        for x_, y_ in zip(split(x, batch_size), split(y, batch_size)):
            yield x_, y_
        if not infinite:
            break

def get_optimizer(net, lr, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def do_learning(model, loss, optimizer, train_gen, max_iter, device):

    model.train()
    for iteration, data in enumerate(train_gen):
        # Sample minibatch
        x, y = torch.from_numpy(data[0]).float().to(device), torch.from_numpy(data[1]).float().to(device)

        x_max = torch.max(x, axis=1)[0]
        x = torch.div(x, x_max[:,None])

        # Forward pass

        prediction = model(x.squeeze())
        prediction = prediction * x_max[:, None]
        
        # Get loss
        criteria = loss(prediction, y)

        # Backward pass - Update fast net
        optimizer.zero_grad()
        criteria.backward()
        optimizer.step()
        
        if (iteration> max_iter):
            break
        
    return criteria.data




def rmse_loss(y_pred, y_true):
    loss = ((y_pred - y_true)**2).mean()
    return loss

def do_evaluation(model, loss, data_gen, batch_size, device):
    
    #data_gen = batcher((x, y), batch_size=batch_size, infinite=False)
    cum_loss = 0.0
  
    model.eval()
    for j, data in enumerate(data_gen):
        x, y = torch.from_numpy(data[0]).float().to(device), torch.from_numpy(data[1]).float().to(device)
        x_max = torch.max(x, axis=1)[0]
        x = torch.div(x, x_max[:,None])
        y_pred = model(x.squeeze()).reshape(y.shape)
        y_pred = y_pred * x_max[:, None]
        cum_loss += loss(y_pred, y).data 
    
    return cum_loss


class ReptileModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class Model(ReptileModel):

    def __init__(self, backcast_length, forecast_length, hidden):
        ReptileModel.__init__(self)

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.hidden = hidden
        self.predictor = nn.Sequential(
            
            nn.Linear(backcast_length, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], forecast_length)
    
        )

    def forward(self, x):
        out = self.predictor(x)

        return out


    def clone(self):
        clone = Model(self.backcast_length, self.forecast_length, self.hidden)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

class ResnetBlock(nn.Module):
    
    def __init__(self, n_input, kernel_sizes, n_filters):
        super(ResnetBlock, self).__init__()
        self.total_filters = n_filters*len(kernel_sizes)
        self.conv = nn.Conv1d(n_input, self.total_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(self.total_filters)
        self.conv_list1 = nn.ModuleList([  nn.Conv1d(n_input, n_filters, kernel_size, padding=int((kernel_size-1)/2)) for kernel_size in kernel_sizes])
        self.conv_list2 = nn.ModuleList([  nn.Conv1d(self.total_filters, n_filters, kernel_size,  padding=int((kernel_size-1)/2))  for kernel_size in kernel_sizes])
        self.bn1 = nn.BatchNorm1d(self.total_filters)
        self.bn2= nn.BatchNorm1d(self.total_filters)
        
    def forward(self, x):
        
        residual = self.bn(self.conv(x))
        x = torch.cat([nn.functional.pad(self.conv_list1[i](x), (0,1)) for i in range(len(self.conv_list1))], axis=1)
        x = F.relu(self.bn1(x))
        x = torch.cat([nn.functional.pad(self.conv_list2[i](x), (0,1)) for i in range(len(self.conv_list2))], axis=1)
        x = F.relu(self.bn2(x))
        x+= residual
        
        return x
    
    def to_cuda(self):
        
        for conv in self.conv_list1:
            conv.cuda()
            
        for conv in self.conv_list2:
            conv.cuda()



class ResnetRegressor(reptile.ReptileModel):
    
    def __init__(self, n_input, forecast_length, backcast_length, kernel_sizes, n_filters = 33, hidden = 512, n_blocks = 4):
        
        super(reptile.ReptileModel, self).__init__()
        self.n_input = n_input
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.hidden = hidden
        self.n_blocks = n_blocks
        
        total_filters = n_filters*len(kernel_sizes)
        self.total_filters = total_filters
        self.block1 = ResnetBlock(n_input, kernel_sizes, n_filters)
        self.blocks = nn.ModuleList()
        for block in range(n_blocks):
            self.blocks.append(ResnetBlock(total_filters, kernel_sizes, n_filters))

        self.feed_forward = nn.Linear(backcast_length*total_filters, hidden)
        self.feed_forward2 = nn.Linear(hidden, forecast_length)
                                          
        
        
    def forward(self, x):
        
        inp = x
        x = x.unsqueeze(1)
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        x = x.view(inp.shape[0], -1)
        x = F.relu(self.feed_forward(x))
        x = self.feed_forward2(x)
        return x
    

    def clone(self):
        clone = ResnetRegressor(self.n_input, 
                                self.forecast_length, 
                                self.backcast_length, 
                                self.kernel_sizes, 
                                self.n_filters, 
                                self.hidden,
                                self.n_blocks)
    
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
            clone.to_cuda()
        return clone

    def to_cuda(self):
        
        self.block1.cuda()
        self.block1.to_cuda()
        for block in self.blocks:
            block.cuda()
            block.to_cuda()

