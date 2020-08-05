import torch

def mae(y_pred, y_true):
    loss = (torch.abs(y_pred - y_true)).mean()
    return loss  