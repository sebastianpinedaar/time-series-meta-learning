import learn2learn as l2l
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import sys
import argparse
import os
from run_MAML_04 import test2 as test_maml

sys.path.insert(1, "..")

from ts_dataset import TSDataset
from base_models import LSTMModel, FCN
from metrics import torch_mae as mae
import copy
from pytorchtools import EarlyStopping
from eval_base_models import test, train, freeze_model
from torch.utils.data import Dataset, DataLoader
from ts_dataset import DomainTSDataset, SimpleDataset
from ts_transform import split_idx_50_50
from pytorchtools import to_torch

def main(args):

    
    meta_info = {"POLLUTION": [5, 50, 14],
                 "HR": [32, 50, 13],
                 "BATTERY": [20, 50, 3] }

    output_directory = "output/"
    verbose=True
    batch_size=64
    freeze_model_flag = True

    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0}

    dataset_name = args.dataset 
    model_name = args.model 
    learning_rate = args.learning_rate 
    save_model_file = args.save_model_file
    load_model_file = args.load_model_file
    lower_trial = args.lower_trial
    upper_trial = args.upper_trial
    is_test = args.is_test
    epochs = args.epochs
    experiment_id = args.experiment_id
    adaptation_steps = args.adaptation_steps
   
    assert model_name in ("FCN", "LSTM"), "Model was not correctly specified"
    assert dataset_name in ("POLLUTION", "HR", "BATTERY")

    window_size, task_size, input_dim = meta_info[dataset_name]
    batch_size = 64

    train_data = pickle.load(  open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    train_data_ML = pickle.load( open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    validation_data = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    validation_data_ML = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    test_data = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    test_data_ML = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )



    if is_test == 0:
        test_data = validation_data

    train_idx, val_idx, test_idx = split_idx_50_50(test_data.file_idx) if is_test else split_idx_50_50(validation_data.file_idx)
    n_domains_in_test = np.max(test_data.file_idx)+1

    test_loss_list = []
    initial_test_loss_list = []

    trials_loss_list = []

    #trial = 0
    for trial in range(lower_trial, upper_trial):
            
        output_directory = "../../Models/"+dataset_name+"_"+model_name+"_MAML/"+str(trial)+"/"

        
        #save_model_file_ = output_directory + "encoder_"+save_model_file
        #save_model_file_2 = output_directory + save_model_file
        save_model_file_ = output_directory + experiment_id + "_encoder_model.pt"
        save_model_file_2 = output_directory + experiment_id + "_model.pt"
        load_model_file_ = output_directory + load_model_file

        model = LSTMModel( batch_size=batch_size, seq_len = window_size, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =1)
        model2 = nn.Linear(120, 1)

        model.cuda()
        model2.cuda()

        maml = l2l.algorithms.MAML(model2, lr=learning_rate, first_order=False)
        model.load_state_dict(torch.load(save_model_file_))
        maml.load_state_dict(torch.load(save_model_file_2))

            
        n_domains_in_test = np.max(test_data.file_idx)+1

        error_list = []

        y_list = []

        for domain in range(n_domains_in_test):
            x_test = test_data.x
            y_test = test_data.y


            temp_train_data = SimpleDataset(x=np.concatenate([x_test[ np.concatenate([train_idx[domain],val_idx[domain]])][np.newaxis,:], x_test[ test_idx[domain]][np.newaxis,:]]), 
                                                                    y =np.concatenate([ y_test[np.concatenate([train_idx[domain],val_idx[domain]])][np.newaxis,:], y_test[test_idx[domain]][np.newaxis,:]]))


            total_tasks_test = len(test_data_ML)
            

            learner = maml.clone()  # Creates a clone of model
            learner.cuda()
            accum_error = 0.0
            accum_std = 0.0
            count = 0.0

            input_dim = test_data_ML.x.shape[-1]
            window_size = test_data_ML.x.shape[-2]
            output_dim = test_data_ML.y.shape[-1]

            task = 0

            model2 = nn.Linear(120, 1)
            model2.load_state_dict(copy.deepcopy(maml.module.state_dict()))

            model.cuda()
            model2.cuda()

            x_spt, y_spt = temp_train_data[task]
            x_qry = temp_train_data.x[(task+1)]
            y_qry = temp_train_data.y[(task+1)]

            if model_name == "FCN":
                x_qry = np.transpose(x_qry, [0,2,1])
                x_spt = np.transpose(x_spt, [0,2,1])

            x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
            x_qry = to_torch(x_qry)
            y_qry = to_torch(y_qry)

            opt2 = optim.SGD(list(model2.parameters()), lr=learning_rate)
            #learner.module.train()
            size_back = 300
            step_size =  task_size*size_back
            
            #model2.eval()
            for step in range(adaptation_steps):

                print(step)
                #model2.train()
                for idx in range(x_spt.shape[0]-task_size*size_back, x_spt.shape[0], step_size):

                    pred = model2(model.encoder(x_spt[idx:idx+step_size]))
                    print(pred.shape)
                    print(step_size)
                    error = mae(pred, y_spt[idx:idx+step_size])
                    print(error)
                    opt2.zero_grad()
                    error.backward()

                    #learner.adapt(error)
                    opt2.step()

            #model2.eval()
            #learner.module.eval()
            step = x_qry.shape[0]//255
            for idx in range(0, x_qry.shape[0], step):
                pred = model2(model.encoder(x_qry[idx:idx+step]))
                error = mae(pred, y_qry[idx:idx+step])

                accum_error += error.data
                accum_std += error.data**2
                count += 1

            error = accum_error/count
    
            y_list.append(y_qry.cpu().numpy())
            error_list.append(float(error.cpu().numpy()))
            print(np.mean(error_list))
            print(error_list)

            trials_loss_list.append(np.mean(error_list))

        print("mean:", np.mean(trials_loss_list))
        print("std:", np.std(trials_loss_list))

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use, possible: POLLUTION, HR, BATTERY', default="POLLUTION")
    argparser.add_argument('--model', type=str, help='base model, possible: LSTM, FCN', default="LSTM")
    argparser.add_argument('--adaptation_steps', type=int, help='number of updates in the inner loop', default=1)
    argparser.add_argument('--learning_rate', type=float, help='learning rate for the inner loop', default=0.0001)
    argparser.add_argument('--save_model_file', type=str, help='name to save the model in memory', default="model.pt")
    argparser.add_argument('--load_model_file', type=str, help='name to load the model in memory', default="model.pt")
    argparser.add_argument('--lower_trial', type=int, help='identifier of the lower trial value', default=0)
    argparser.add_argument('--upper_trial', type=int, help='identifier of the upper trial value', default=3)
    argparser.add_argument('--is_test', type=int, help='whether apply on test (1) or validation (0)', default=0)

    argparser.add_argument('--epochs', type=int, help='epochs', default=20000)
    argparser.add_argument('--experiment_id', type=str, help='experiment_id for the experiments list', default="DEFAULT-ID")

    args = argparser.parse_args()

    main(args)