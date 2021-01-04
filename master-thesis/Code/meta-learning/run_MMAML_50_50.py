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
from run_MMAML_04 import test as test_maml
from multimodallearner import get_task_encoder_input
from multimodallearner import LSTMDecoder, Lambda, MultimodalLearner
from metalearner import MetaLearner
from meta_base_models import LinearModel, Task

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = mae 

    train_data = pickle.load(  open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    train_data_ML = pickle.load( open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    validation_data = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    validation_data_ML = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    test_data = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    test_data_ML = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )

    # paramters wto increase capactiy of the model
    n_layers_task_net = 2
    n_layers_task_encoder = 1
    n_layers_task_decoder = 1

    hidden_dim_task_net = 120
    hidden_dim_encoder = 120
    hidden_dim_decoder = 120

    input_dim_task_net = input_dim
    input_dim_task_encoder = input_dim + 1
    output_dim_task_net = 1
    output_dim_task_decoder = input_dim + 1
    output_dim = 1

    if is_test == 0:
        test_data = validation_data

    train_idx, val_idx, test_idx = split_idx_50_50(test_data.file_idx) if is_test else split_idx_50_50(validation_data.file_idx)
    n_domains_in_test = np.max(test_data.file_idx)+1

    test_loss_list = []
    initial_test_loss_list = []

    trials_loss_list = []
    modulate_task_net = True
    

    #trial = 0
    for trial in range(lower_trial, upper_trial):
            
        output_directory = "../../Models/"+dataset_name+"_"+model_name+"_MMAML/"+str(trial)+"/"

        
        #save_model_file_ = output_directory + "encoder_"+save_model_file
        #save_model_file_2 = output_directory + save_model_file
        save_model_file_encoder = output_directory + experiment_id + "_encoder_model.pt"
        save_model_file_ = output_directory + experiment_id + "_model.pt"
        load_model_file_ = output_directory + load_model_file

        ##creating the network

        task_net = LSTMModel(batch_size=batch_size,
                             seq_len=window_size,
                             input_dim=input_dim_task_net,
                             n_layers=n_layers_task_net,
                             hidden_dim=hidden_dim_task_net,
                             output_dim=output_dim_task_net)

        task_encoder = LSTMModel(batch_size=batch_size,
                                 seq_len=task_size,
                                 input_dim=input_dim_task_encoder,
                                 n_layers=n_layers_task_encoder,
                                 hidden_dim=hidden_dim_encoder,
                                 output_dim=1)

        task_decoder = LSTMDecoder(batch_size=1,
                                   n_layers=n_layers_task_decoder,
                                   seq_len=task_size,
                                   output_dim=output_dim_task_decoder,
                                   hidden_dim=hidden_dim_encoder,
                                   latent_dim=hidden_dim_decoder,
                                   device=device)
        lmbd = Lambda(hidden_dim_encoder, hidden_dim_task_net)

        multimodal_learner = MultimodalLearner(task_net, task_encoder, task_decoder, lmbd, modulate_task_net)
        multimodal_learner.to(device)

        output_layer = nn.Linear(120, 1)
        output_layer.to(device)

        maml = l2l.algorithms.MAML(output_layer, lr=learning_rate, first_order=False)
   

        multimodal_learner.load_state_dict(torch.load(save_model_file_encoder))
        maml.load_state_dict(torch.load(save_model_file_))
        

        n_domains_in_test = np.max(test_data.file_idx)+1

        error_list = []

        y_list = []

        for domain in range(n_domains_in_test):
            print("Domain:", domain)
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

            task_id = 0

            #model2 = nn.Linear(120, 1)
            #model2.load_state_dict(copy.deepcopy(maml.module.state_dict()))

            #model.cuda()
            #model2.cuda()
            output_layer = nn.Linear(120, 1)
            output_layer.load_state_dict(copy.deepcopy(maml.module.state_dict()))
            output_layer.to(device)
            
            x_spt, y_spt = temp_train_data[task_id]
            x_qry = temp_train_data.x[(task_id+1)]
            y_qry = temp_train_data.y[(task_id+1)]

            task = get_task_encoder_input(SimpleDataset(x=x_spt[-50:][np.newaxis,:], y=y_spt[-50:][np.newaxis, :]))
            task = to_torch(task)

            if model_name == "FCN":
                x_qry = np.transpose(x_qry, [0,2,1])
                x_spt = np.transpose(x_spt, [0,2,1])

            x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
            x_qry = to_torch(x_qry)
            y_qry = to_torch(y_qry)

            opt2 = optim.SGD(list(output_layer.parameters()), lr=learning_rate)
            #learner.module.train()
            size_back = 200
            step_size =  task_size*size_back
            
            multimodal_learner.train()
            #model2.eval()
            for step in range(adaptation_steps):

                step_size = 1
                error_accum = 0
                count = 0
                #model2.train()
                for idx in range(0, x_spt.shape[0], step_size):
                
                    x_spt_encoding, (vrae_loss, _, _) = multimodal_learner(x_spt[idx:idx+step_size], task,output_encoding=True)
                    pred = output_layer(x_spt_encoding)
                    error_accum+= mae(pred, y_spt[idx:idx+step_size])
                    count += 1
                
                
                opt2.zero_grad()
                error = error_accum/count
                error.backward()

                #learner.adapt(error)
                opt2.step()

            #model2.eval()
            #learner.module.eval()

            multimodal_learner.eval()
            step = x_qry.shape[0]//255
            error_accum = 0
            count = 0
            for idx in range(0, x_qry.shape[0], step):

                x_qry_encoding, (vrae_loss, _, _) = multimodal_learner(x_qry[idx:idx+step], task,output_encoding=True)
                pred = output_layer(x_qry_encoding)
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