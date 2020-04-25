import load_data as ld
import os
from data import get_m4_data, dummy_data_generator, get_m4_data_multivariate
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
import reptile
import argparse

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)
    dataset = args.dataset
    subdataset = args.subdataset
    samples_per_task = args.samples_per_task
    backcast_length = args.backcast_length
    forecast_length = args.forecast_length
    inner_iterations = args.inner_iterations # 
    meta_batch_size = args.meta_batch_size #B
    model_file = args.model_file
    model_type = args.model_type

    #fixed
    update_batch_size = samples_per_task 
    outer_iterations = 1000 # M
    meta_lr = 0.01#/meta_batch_size
    update_lr = 0.01#/samples_per_task
    patience = 20
    loss = rmse_loss
    pct = (0.8, 0.9, 1.0)

    if(model_type == "RESNET"):

        #fixed parameters
        n_filters = 33
        kernel_sizes = [32, 16, 8, 4]
        n_input = 1
        n_output = forecast_length
        n_blocks = 3
        hidden = 512

        model =  reptile.ResnetRegressor(n_input, forecast_length, backcast_length, kernel_sizes, n_filters, hidden, n_blocks)
        model.cuda()
        model.to_cuda()

    else: 

        hidden = [40,20,10]
        model = reptile.Model(backcast_length, forecast_length, hidden).to(device)

    print("Model parameters:", count_parameters(model))



    if dataset== "M4":
        
        path = "M4DataSet/"
        files = os.listdir(path)
        print(files)
        x_train, y_train, x_test, y_test, x_val, y_val = ld.load_M4_meta_datasets(path+files[subdataset], 
                                                                                backcast_length, 
                                                                                forecast_length, 
                                                                                n_tasks=samples_per_task,
                                                                                pct = pct)

    elif dataset == "TOURISM":
        
        path = "TourismDataSet/"
        files = os.listdir(path)

        x_train, y_train, x_test, y_test, x_val, y_val = ld.load_Tourism_meta_datasets(path, 
                                                                                    backcast_length, 
                                                                                    forecast_length, 
                                                                                    n_tasks=samples_per_task,
                                                                                    pct = pct)
    
    else:
        
        filename = "M3DataSet/M3C.xls"
        df = pd.read_excel(filename, sheet_name= None)
        sheet_names = list(df.keys())
        sheet_name = sheet_names[subdataset]
        
        x_train, y_train, x_test, y_test, x_val, y_val = ld.load_M3_meta_datasets(filename, 
                                                                                sheet_name, 
                                                                                backcast_length, 
                                                                                forecast_length, 
                                                                                n_tasks=samples_per_task,
                                                                                pct = pct)

    meta_optimizer = torch.optim.SGD(model.parameters(), lr=meta_lr)
    data_gen_train = ld.batcher((x_train, y_train), batch_size=meta_batch_size, infinite=True)

    x_max = []
    scaled = []
    for x, y in [(x_train, y_train), (x_test, y_test), (x_val, y_val)]:
        
        x_max.append(np.max(x, axis=2))
        scaled.append(scale(x, x_max[-1]))
        scaled.append(scale(y, x_max[-1]))

    x_max_train, x_max_test, x_max_val = x_max
    x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, x_val_scaled, y_val_scaled = scaled

    x_val = x_val.reshape(-1, backcast_length)
    y_val = y_val.reshape(-1, forecast_length)
    x_max_train = x_max_train.reshape(-1)
    train_samples = x_train.shape[0]
    val_samples = x_val.shape[0]

    early_stopping = EarlyStopping(patience=patience, model_file=model_file, verbose=False)
    state = None
    meta_loss_list = []
    val_samples = x_val.shape[0]

    for i in range(outer_iterations):

        model.train()
        new_model = model.clone()
        optimizer = reptile.get_optimizer(new_model, update_lr, state)

        #inner loop
        for j in range(meta_batch_size):

            t = np.random.randint(x_train.shape[0])
            data_gen2 = ld.batcher((x_train[t,...], y_train[t,...]), batch_size=update_batch_size, infinite = True)
            meta_loss = reptile.do_learning(new_model, loss, optimizer, data_gen2, inner_iterations, device)
            state = optimizer.state_dict()

        #meta update
        state = optimizer.state_dict()  # save optimizer state

        model.point_grad_to(new_model)
        meta_optimizer.step()

        data_gen_val = ld.batcher((x_val, y_val), batch_size=meta_batch_size, infinite=False)
        val_loss = reptile.do_evaluation(model, smape, data_gen_val, meta_batch_size, device)
        val_loss = 200*val_loss/(forecast_length*val_samples)

        if(i%100==0):
            print("Currently in iteration ",i, " with validation loss: ", val_loss)

        early_stopping(val_loss, model)
        meta_loss_list.append(val_loss.cpu().data.numpy())

        if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(model_file))

    x_test = x_test.reshape(-1, backcast_length)
    y_test = y_test.reshape(-1, forecast_length)
    test_samples = x_test.shape[0]
    data_gen_test = ld.batcher((x_test, y_test), batch_size=meta_batch_size, infinite=False)
    temp_loss = reptile.do_evaluation(model, smape, data_gen_test, meta_batch_size, device )#reuse from reptile
    test_loss = 200*temp_loss/(forecast_length*test_samples)

    print("Test loss", test_loss)
    
    np.save("test_loss", test_loss.cpu().numpy())
    np.save("loss_history", meta_loss_list)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use, possible: M4, M3, TOURISM', default="M4")
    argparser.add_argument('--subdataset', type=int, help='subdataset number indicanding seasonality', default=1)
    argparser.add_argument('--model_file', type=str, help='name for the model in memory', default="reptile.pt")
    argparser.add_argument('--samples_per_task', type=int, help='number of samples per time series', default=2)
    argparser.add_argument('--backcast_length', type=int, help='number of points to use in input', default=42)
    argparser.add_argument('--forecast_length', type=int, help='number of points to predict', default=6)
    argparser.add_argument('--meta_batch_size', type=int, help='number of samples to use in the outer iteration', default=5)
    argparser.add_argument('--inner_iterations', type=int, help ='number of updates in the inner loop', default=5)
    argparser.add_argument('--model_type', type=str, help='type of model to train with Reptile', default='FEED')
    args = argparser.parse_args()

    main(args)