import random
import os
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
from pytorchtools import count_parameters
import torch.nn as nn
import pickle
from base_models import LSTMModel, FCN
from torch.utils.data import Dataset, DataLoader
from metrics import torch_mae as mae
import argparse
from pytorchtools import EarlyStopping
from ts_dataset import TSDataset, SimpleDataset



def step(model, data_iter, len_dataloader, optimizer = None, loss = mae, is_train=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_train:
        model.train()
    else:
        model.eval()

    accum_err = 0
    accum_size = 0
    i = 0
    while i < len_dataloader:

        # training model using source data
        data_source = data_iter.next()
        x, y = data_source

        if is_train:
            model.zero_grad()
        
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)

        y_pred = model(x)      
        err = loss(y, y_pred)

        if is_train:
            err.backward()
            optimizer.step()

        #print(err)
        accum_err +=err*x.shape[0]
        accum_size += x.shape[0]
        i += 1

    return float(accum_err/accum_size)
        


def train(model, train_loader, val_loader, early_stopping, learning_rate = 0.001, epochs = 500, monitor_stopping = True):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = early_stopping.patience//4, verbose=True)

    for epoch in range(epochs):

        len_train_loader = len(train_loader)
        train_iter = iter(train_loader)

        len_val_loader = len(val_loader)
        val_iter = iter(val_loader)

        mean_err = step(model, train_iter, len_train_loader, optimizer, is_train=True)

        with torch.no_grad():
            mean_err_val = step(model, val_iter, len_val_loader)

        print ('epoch: %d, \n TRAINING -> mean_err: %f' % (epoch, mean_err))
        print ('epoch: %d, \n VAL -> mean_err: %f' % (epoch, mean_err_val))

        scheduler.step(mean_err_val)

        if monitor_stopping:
            early_stopping(mean_err_val, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('done')

    return epoch+1


def test(model, test_loader, output_directory, model_file):

    len_test_loader = len(test_loader)
    test_iter = iter(test_loader)

    model.load_state_dict(torch.load(model_file))

    with torch.no_grad():
        mean_err = step(model, test_iter, len_test_loader)

    print("Regression error on test: %f"%(mean_err))

    f=open(output_directory+"/results.txt", "a+")
    f.write("Test error :%f"% mean_err)
    f.write("\n")
    f.close()

    return mean_err




def main(args):

    meta_info = {"POLLUTION": [5, 50, 14],
                 "HR": [32, 50, 13],
                 "BATTERY": [20, 50, 3] }

    output_directory = "output/"
    verbose=True
    batch_size=256


    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0}

    dataset_name = args.dataset 
    mode = args.mode 
    save_model_file = args.save_model_file
    load_model_file = args.load_model_file
    lower_trial = args.lower_trial
    upper_trial = args.upper_trial
    learning_rate = args.learning_rate
    regularization_penalty = args.regularization_penalty
    model_name = args.model
    is_test = args.is_test
    patience_stopping = args.patience_stopping
    epochs = args.epochs

    assert mode in ("WFT", "WOFT", "50"), "Mode was not correctly specified"
    assert model_name in ("FCN", "LSTM"), "Model was not correctly specified"
    assert dataset_name in ("POLLUTION", "HR", "BATTERY")

    window_size, task_size, input_dim = meta_info[dataset_name]

    train_data = pickle.load(  open( "../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    validation_data = pickle.load( open( "../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    validation_data_ML = pickle.load( open( "../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    test_data = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    test_data_ML = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    for trial in range(lower_trial, upper_trial):

        output_directory = "../Models/"+dataset_name+"_"+model_name+"/"+str(trial)+"/"
        save_model_file_ = output_directory + save_model_file
        load_model_file_ = output_directory + load_model_file

        try:
            os.mkdir(output_directory)
        except OSError as error: 
            print(error)

        if mode == "WOFT":
            
            f=open(output_directory+"/results.txt", "a+")
            f.write("Dataset :%s \n"% dataset_name)
            f.write("Learning rate:%f \n"%learning_rate)
            f.close()

            if model_name == "FCN":

                kernels = [8,5,3] if dataset_name!= "POLLUTION" else [4,2,1]
                train_data.x = np.transpose(train_data.x, [0,2,1])
                test_data.x = np.transpose(test_data.x, [0,2,1])
                validation_data.x = np.transpose(validation_data.x, [0,2,1])

            early_stopping = EarlyStopping(patience=patience_stopping, model_file=save_model_file_, verbose=verbose)

            if model_name == "LSTM":
                model = LSTMModel( batch_size=batch_size, seq_len = window_size, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =1)
            elif model_name == "FCN":
                model = FCN(time_steps = window_size,  channels=[input_dim, 128, 128, 128] , kernels=kernels)


            model.cuda()

            train_loader = DataLoader(train_data, **params)
            val_loader = DataLoader(validation_data, **params)

            if is_test:
                test_loader = DataLoader(test_data, **params)
            else:
                test_loader = DataLoader(validation_data, **params)

            train(model, train_loader, val_loader, early_stopping, learning_rate, epochs) 
            test(model, test_loader, output_directory, save_model_file_)

        elif mode == "WFT":

            f=open(output_directory+"/results.txt", "a+")
            f.write("Dataset :%s"% dataset_name)
            f.write("\n")
            f.close()

            save_model_file_ = output_directory+save_model_file
            load_model_file_ = output_directory+load_model_file

            assert save_model_file!=load_model_file, "Files cannot be the same"

            n_tasks, task_size, dim, channels = test_data_ML.x.shape
            horizon = 10
            #epochs = 20
            test_loss_list = []
            initial_test_loss_list = []

            for task_id in range(0, (n_tasks-horizon-1), n_tasks//100):
                

                early_stopping = EarlyStopping(patience=patience_stopping, model_file=save_model_file_, verbose=verbose)

                if model_name == "LSTM":
                    model = LSTMModel( batch_size=batch_size, seq_len = window_size, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =1)
                elif model_name == "FCN":
                    model = FCN(time_steps = window_size, channels=[input_dim, 128, 128, 128] )
                    
                model.cuda()
                model.load_state_dict(torch.load(load_model_file_))

                #freeze model here if necessary

                if is_test: 
                    temp_x_train = test_data_ML.x[task_id][:int(task_size*0.8)]
                    temp_y_train = test_data_ML.y[task_id][:int(task_size*0.8)]
                    
                    temp_x_val = test_data_ML.x[task_id][int(task_size*0.8):]
                    temp_y_val = test_data_ML.y[task_id][int(task_size*0.8):]

                    temp_x_test = test_data_ML.x[(task_id+1):(task_id+horizon+1)].reshape(-1, dim, channels)
                    temp_y_test = test_data_ML.y[(task_id+1):(task_id+horizon+1)].reshape(-1, 1)

                else:
                    temp_x_train = validation_data_ML.x[task_id][:int(task_size*0.8)]
                    temp_y_train = validation_data_ML.y[task_id][:int(task_size*0.8)]
                    
                    temp_x_val = validation_data_ML.x[task_id][int(task_size*0.8):]
                    temp_y_val = validation_data_ML.y[task_id][int(task_size*0.8):]

                    temp_x_test = validation_data_ML.x[(task_id+1):(task_id+horizon+1)].reshape(-1, dim, channels)
                    temp_y_test = validation_data_ML.y[(task_id+1):(task_id+horizon+1)].reshape(-1, 1)               

                if model_name == "FCN":

                    kernels = [8,5,3] if dataset_name!= "POLLUTION" else [4,2,1]
                    temp_x_train = np.transpose(temp_x_train, [0,2,1])
                    temp_x_test = np.transpose(temp_x_test, [0,2,1])
                    temp_x_val = np.transpose(temp_x_val, [0,2,1])


                train_loader = DataLoader(SimpleDataset(x=temp_x_train, y=temp_y_train), **params)
                val_loader = DataLoader(SimpleDataset(x=temp_x_val, y=temp_y_val), **params)
                test_loader = DataLoader(SimpleDataset(x=temp_x_test, y=temp_y_test), **params)


                initial_loss = test(model, test_loader, output_directory, load_model_file_)
                train(model, train_loader, val_loader, early_stopping, learning_rate, epochs) 
                loss = test(model, test_loader, output_directory, save_model_file_)

                test_loss_list.append(loss)
                initial_test_loss_list.append(initial_loss)

            f=open(output_directory+"/results.txt", "a+")
            f.write("Initial Test error :%f \n"% np.mean(initial_test_loss_list))
            f.write("Test error: %f \n"% np.mean(test_loss_list))
            f.write("\n")
            f.close()       


        elif mode == "50":
            pass
    


    
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use, possible: POLLUTION, HR, BATTERY', default="POLLUTION")
    argparser.add_argument('--mode', type=str, help='evaluation mode, possible: WOFT, WFT, 50, HYP', default="WOFT")
    argparser.add_argument('--model', type=str, help='model architecture, possible: FCN, LSTM', default="LSTM")
    argparser.add_argument('--save_model_file', type=str, help='name to save the model in memory', default="model.pt")
    argparser.add_argument('--load_model_file', type=str, help='name to load the model in memory', default="model.pt")
    argparser.add_argument('--lower_trial', type=int, help='identifier of the lower trial value', default=0)
    argparser.add_argument('--upper_trial', type=int, help='identifier of the upper trial value', default=3)
    argparser.add_argument('--learning_rate', type=float, help='learning rate', default=0.01)
    argparser.add_argument('--regularization_penalty', type=float, help='regularization penaly', default=0.0001)
    argparser.add_argument('--is_test', type=int, help='whether apply on test (1) or validation (0)', default=0)
    argparser.add_argument('--patience_stopping', type=int, help='patience for early stopping', default=20)
    argparser.add_argument('--epochs', type=int, help='epochs', default=500)

    args = argparser.parse_args()

    main(args)