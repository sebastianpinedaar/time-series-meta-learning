##own implementation with data augmentation, fine-tuning only the last layer, 

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import sys
from collections import defaultdict, namedtuple
from metalearner import MetaLearner
from meta_base_models import LinearModel, Task
import os


sys.path.insert(1, "..")

from utils import progressBar
from ts_dataset import TSDataset
from metrics import torch_mae as mae
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
from pytorchtools import EarlyStopping, to_torch, get_grad_norm
from base_models import LSTMModel
import argparse



def test (test_data_ML, meta_learner, model, device, noise_level ,noise_type = "additive", horizon = 10):

    total_tasks_test = len(test_data_ML)
    task_size = test_data_ML.x.shape[-3]
    input_dim = test_data_ML.x.shape[-1]
    window_size = test_data_ML.x.shape[-2]
    output_dim = test_data_ML.y.shape[-1]
    grid = [0., noise_level]

    accum_error = 0.0
    count = 0



    for task in range(0, (total_tasks_test-horizon-1), total_tasks_test//100):

        x_spt, y_spt = test_data_ML[task]
        x_qry = test_data_ML.x[(task+1):(task+1+horizon)].reshape(-1, window_size, input_dim)
        y_qry = test_data_ML.y[(task+1):(task+1+horizon)].reshape(-1, output_dim)
        
        x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
        x_qry = to_torch(x_qry)
        y_qry = to_torch(y_qry)

        epsilon = grid[np.random.randint(0,len(grid))]

        if noise_type == "additive":
            y_spt = y_spt+epsilon
            y_qry = y_qry+epsilon

        else:
            y_spt = y_spt*(1+epsilon)
            y_qry = y_qry*(1+epsilon)


        train_task = [Task(model.encoder(x_spt), y_spt)]
        val_task = [Task(model.encoder(x_qry), y_qry)]

        adapted_params = meta_learner.adapt(train_task)
        mean_loss = meta_learner.step(adapted_params, val_task, is_training = 0)

        count += 1
        accum_error += mean_loss.cpu().detach().numpy()

    return accum_error/count




def main(args):



    dataset_name = args.dataset
    model_name = args.model 
    n_inner_iter = args.adaptation_steps
    batch_size = args.batch_size
    save_model_file = args.save_model_file
    load_model_file = args.load_model_file
    lower_trial = args.lower_trial
    upper_trial = args.upper_trial
    is_test = args.is_test
    stopping_patience = args.stopping_patience
    epochs = args.epochs
    fast_lr = args.learning_rate
    slow_lr = args.meta_learning_rate
    noise_level = args.noise_level
    noise_type = args.noise_type
    resume = args.resume

    first_order = False
    inner_loop_grad_clip = 20
    task_size = 50
    output_dim = 1
    checkpoint_freq = 10
    horizon = 10
    ##test


    meta_info = {"POLLUTION": [5, 50, 14],
                 "HR": [32, 50, 13],
                 "BATTERY": [20, 50, 3] }

    assert model_name in ("FCN", "LSTM"), "Model was not correctly specified"
    assert dataset_name in ("POLLUTION", "HR", "BATTERY")

    window_size, task_size, input_dim = meta_info[dataset_name]

    grid = [0., noise_level]
    output_directory = "output/"

    train_data_ML = pickle.load( open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    validation_data_ML = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    test_data_ML = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )


    for trial in range(lower_trial, upper_trial):

        output_directory = "../../Models/"+dataset_name+"_"+model_name+"_MAML/"+str(trial)+"/"
        save_model_file_ = output_directory + save_model_file
        save_model_file_encoder = output_directory + "encoder_" + save_model_file
        load_model_file_ = output_directory + load_model_file
        checkpoint_file = output_directory + "checkpoint_" + save_model_file.split(".")[0]

        try:
            os.mkdir(output_directory)
        except OSError as error: 
            print(error)

        with open(output_directory+"/results2.txt", "a+") as f:
            f.write("Learning rate :%f \n"% fast_lr)
            f.write("Meta-learning rate: %f \n" % slow_lr)
            f.write("Adaptation steps: %f \n" % n_inner_iter)
            f.write("Noise level: %f \n" % noise_level)

        if model_name == "LSTM":
            model = LSTMModel( batch_size=batch_size, seq_len = window_size, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =output_dim)
            model2 = LinearModel(120,1)
        optimizer = torch.optim.Adam(list(model.parameters())+list(model2.parameters()), lr = slow_lr)
        loss_func = mae
        #loss_func = nn.SmoothL1Loss()
        #loss_func = nn.MSELoss()
        initial_epoch = 0

        #torch.backends.cudnn.enabled = False

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        meta_learner = MetaLearner(model2, optimizer, fast_lr , loss_func,
                        first_order, n_inner_iter, inner_loop_grad_clip,
                        device)
        model.to(device)

        early_stopping = EarlyStopping(patience=stopping_patience, model_file=save_model_file_encoder, verbose=True)
        early_stopping2 = EarlyStopping(patience=stopping_patience, model_file=save_model_file_, verbose=True)

        if resume:
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint["model"])
            meta_learner.load_state_dict(checkpoint["meta_learner"])
            initial_epoch = checkpoint["epoch"]
            best_score = checkpoint["best_score"]
            counter = checkpoint["counter_stopping"]

            early_stopping.best_score = best_score
            early_stopping2.best_score = best_score

            early_stopping.counter = counter
            early_stopping2.counter = counter



        total_tasks, task_size, window_size, input_dim = train_data_ML.x.shape
        accum_mean =0.0

        for epoch in range(initial_epoch, epochs):


            model.zero_grad()
            meta_learner._model.zero_grad()

            #train
            batch_idx = np.random.randint(0, total_tasks-1, batch_size)

            #for batch_idx in range(0, total_tasks-1, batch_size):


            x_spt, y_spt = train_data_ML[batch_idx]
            x_qry, y_qry = train_data_ML[batch_idx+1]

            x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
            x_qry = to_torch(x_qry)
            y_qry = to_torch(y_qry)

            # data augmentation
            epsilon = grid[np.random.randint(0, len(grid))]

            if noise_type == "additive":
                y_spt = y_spt + epsilon
                y_qry = y_qry + epsilon
            else:
                y_spt = y_spt * (1 + epsilon)
                y_qry = y_qry * (1 + epsilon)

            train_tasks = [Task(model.encoder(x_spt[i]), y_spt[i]) for i in range(x_spt.shape[0])]
            val_tasks = [Task(model.encoder(x_qry[i]), y_qry[i]) for i in range(x_qry.shape[0])]

            adapted_params = meta_learner.adapt(train_tasks)
            mean_loss = meta_learner.step(adapted_params, val_tasks, is_training = True)
            #accum_mean += mean_loss.cpu().detach().numpy()

            #progressBar(batch_idx, total_tasks, 100)

            #print(accum_mean/(batch_idx+1))

            #test

            val_error = test(validation_data_ML, meta_learner, model, device, noise_level)
            test_error = test(test_data_ML, meta_learner, model, device, 0.0)
            print("Epoch:", epoch)
            print("Val error:", val_error)
            print("Test error:", test_error)

            early_stopping(val_error, model)
            early_stopping2(val_error, meta_learner)

  
              #checkpointing
            if epochs % checkpoint_freq ==0:
                torch.save({ "epoch" : epoch,
                             "model" : model.state_dict(),
                             "meta_learner": meta_learner.state_dict(),
                             "best_score" : early_stopping2.best_score,
                             "counter_stopping": early_stopping2.counter},
                             checkpoint_file)


    
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("hallo")
        model.load_state_dict(torch.load(save_model_file_encoder))
        model2.load_state_dict(torch.load(save_model_file_)["model_state_dict"])
        meta_learner = MetaLearner(model2, optimizer, fast_lr , loss_func,
                        first_order, n_inner_iter, inner_loop_grad_clip,
                        device)


        validation_error = test(validation_data_ML, meta_learner, model, device, noise_level = 0.0)
        test_error = test(test_data_ML, meta_learner, model, device, noise_level = 0.0)

        validation_error_h1 = test(validation_data_ML, meta_learner, model, device, noise_level = 0.0, horizon = 1)
        test_error_h1 = test(test_data_ML, meta_learner, model, device, noise_level = 0.0, horizon = 1)

        model.load_state_dict(torch.load(save_model_file_encoder))
        model2.load_state_dict(torch.load(save_model_file_)["model_state_dict"])
        meta_learner2 = MetaLearner(model2, optimizer, fast_lr ,loss_func,
                first_order,0, inner_loop_grad_clip,
                device)


        validation_error_h0 = test(validation_data_ML, meta_learner2, model, device, noise_level = 0.0, horizon = 1)
        test_error_h0 = test(test_data_ML, meta_learner2, model, device, noise_level = 0.0, horizon = 1)

        model.load_state_dict(torch.load(save_model_file_encoder))
        model2.load_state_dict(torch.load(save_model_file_)["model_state_dict"])
        meta_learner2 = MetaLearner(model2, optimizer, fast_lr ,loss_func,
                first_order, n_inner_iter, inner_loop_grad_clip,
                device)
        validation_error_mae = test(validation_data_ML, meta_learner2, model, device, 0.0)
        test_error_mae = test(test_data_ML, meta_learner2, model, device, 0.0)
        print("test_error_mae", test_error_mae)

        with open(output_directory+"/results2.txt", "a+") as f:
            f.write("Test error: %f \n" % test_error)
            f.write("Validation error: %f \n" %validation_error)
            f.write("Test error h1: %f \n" % test_error_h1)
            f.write("Validation error h1: %f \n" %validation_error_h1)
            f.write("Test error h0: %f \n" % test_error_h0)
            f.write("Validation error h0: %f \n" %validation_error_h0) 
            f.write("Test error mae: %f \n" % test_error_mae)
            f.write("Validation error mae: %f \n" %validation_error_mae)     
            
        print(test_error)
        print(validation_error)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use, possible: POLLUTION, HR, BATTERY', default="POLLUTION")
    argparser.add_argument('--model', type=str, help='base model, possible: LSTM, FCN', default="LSTM")
    argparser.add_argument('--adaptation_steps', type=int, help='number of updates in the inner loop', default=5)
    argparser.add_argument('--learning_rate', type=float, help='learning rate for the inner loop', default=0.01)
    argparser.add_argument('--meta_learning_rate', type=float, help='learning rate for the outer loop', default=0.005)
    argparser.add_argument('--batch_size', type=int, help='batch size for the meta-upates (outer loop)', default=20)
    argparser.add_argument('--save_model_file', type=str, help='name to save the model in memory', default="model02.pt")
    argparser.add_argument('--load_model_file', type=str, help='name to load the model in memory', default="model02.pt")
    argparser.add_argument('--lower_trial', type=int, help='identifier of the lower trial value', default=0)
    argparser.add_argument('--upper_trial', type=int, help='identifier of the upper trial value', default=3)
    argparser.add_argument('--is_test', type=int, help='whether apply on test (1) or validation (0)', default=0)
    argparser.add_argument('--stopping_patience', type=int, help='patience for early stopping', default=30)
    argparser.add_argument('--epochs', type=int, help='epochs', default=2000)
    argparser.add_argument('--noise_level', type=float, help='epochs', default=0.00)
    argparser.add_argument('--noise_type', type=str, help='epochs', default="additive")
    argparser.add_argument('--resume', type=int, help='whether load last checkpoint', default=0)

    args = argparser.parse_args()

    main(args)
