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
import n_beats as beats
import pandas as pd
from utils import *
import reptile
import sys
from scipy import signal
import argparse
import pickle

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    subdataset = args.subdataset
    samples_per_task = args.samples_per_task
    backcast_length = args.backcast_length
    forecast_length = args.forecast_length
    model_file = args.model_file
    transform = args.transform

    print(args)
    n_stacks = 20
    lr = 0.000001
    patience = 20
    stack_types = ["generic"]*n_stacks
    theta_dims = [512]*(n_stacks)
    iterations = 10000
    batch_size = 1024
    pct = (0.8, 0.9, 1.0)
    std_freq, std_noise, slope_factor = (0.3, 0.01, 0.1)

    pickle_name = "temp/"+model_file.split(".")[0]+"-"+dataset+"-sub"+str(subdataset)+".pickle"

    #ts_sample, new_ts_ampl, new_ts_gauss, new_ts_phase, new_ts_f1, new_ts_f2, new_ts_f3
    x_train, y_train, x_test, y_test, x_val, y_val = ld.reload_and_transform_meta_datasets(pickle_name, [transform], std_freq, std_noise, slope_factor)

    print(x_train[0])
    print(x_val.shape)
    x_max = []
    scaled = []
    for x, y in [(x_train, y_train), (x_test, y_test), (x_val, y_val)]:
        
        x_max.append(np.max(x, axis=2))
        scaled.append(scale(x, x_max[-1]))
        scaled.append(scale(y, x_max[-1]))

    x_max_train, x_max_test, x_max_val = x_max
    x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, x_val_scaled, y_val_scaled = scaled

    
    model = beats.NBeatsNet(device, tuple(stack_types),                 
                    nb_blocks_per_stack=1,
                    forecast_length=forecast_length,
                    backcast_length=backcast_length,
                    thetas_dims=theta_dims,
                    share_weights_in_stack=False,
                    hidden_layer_units=512)

      
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_train = x_train_scaled.reshape(-1, backcast_length)
    y_train = y_train.reshape(-1, forecast_length)
    x_val = x_val.reshape(-1, backcast_length)
    y_val = y_val.reshape(-1, forecast_length)
    x_max_train = x_max_train.reshape(-1)
    train_samples = x_train.shape[0]
    val_samples = x_val.shape[0]
    early_stopping = EarlyStopping(patience=patience, model_file=model_file, verbose=False)

    loss_list = []

    for i in range(iterations):
        
        idx = np.random.randint(0, train_samples, batch_size)
        x, y = torch.from_numpy(x_train[idx]).float().to(device), torch.from_numpy(y_train[idx]).float().to(device)
        x_max =  torch.from_numpy(x_max_train[idx]).float().to(device)
        prediction = model(x.squeeze())
        prediction = prediction * x_max[:, None]
        
        # Get loss
        loss = rmse_loss(prediction, y.squeeze())

        # Backward pass - Update fast net
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        max_val = (val_samples//batch_size)*batch_size
        data_gen_val = ld.batcher((x_val, y_val), batch_size=batch_size, infinite=False)
        temp_loss = reptile.do_evaluation(model, smape, data_gen_val, batch_size, device )#reuse from reptile
        val_loss = 200*temp_loss/(forecast_length*val_samples)
        loss_list.append(val_loss.cpu().data.numpy())
        
        if(i%100==0):
            print(i)
            print(val_loss)
            
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(model_file))

    model.load_state_dict(torch.load(model_file))
    x_test = x_test.reshape(-1, backcast_length)
    y_test = y_test.reshape(-1, forecast_length)
    test_samples = x_test.shape[0]
    max_test = (test_samples//batch_size)*batch_size
    data_gen_test = ld.batcher((x_test, y_test), batch_size=batch_size, infinite=False)
    temp_loss = reptile.do_evaluation(model, smape, data_gen_test, batch_size, device )#reuse from reptile
    test_loss = 200*temp_loss/(forecast_length*test_samples)

    print("Test loss", test_loss)
    
    np.save("test_loss", test_loss.cpu().numpy())
    np.save("loss_history", loss_list)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use, possible: M4, M3, TOURISM', default="M4")
    argparser.add_argument('--subdataset', type=int, help='subdataset number indicanding seasonality', default=1)
    argparser.add_argument('--model_file', type=str, help='name for the model in memory', default="n_beats.pt")
    argparser.add_argument('--samples_per_task', type=int, help='number of samples per time series', default=2)
    argparser.add_argument('--backcast_length', type=int, help='number of points to use in input', default=42)
    argparser.add_argument('--forecast_length', type=int, help='number of points to predict', default=6)
    argparser.add_argument('--transform', type=int, help='whether to perfom data augmentation or not', default=1)
    
    args = argparser.parse_args()

    main(args)