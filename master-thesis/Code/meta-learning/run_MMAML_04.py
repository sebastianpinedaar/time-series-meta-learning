# multimodal learning (with maml), using 

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
import os
import learn2learn as l2l
from multimodallearner import get_task_encoder_input
from multimodallearner import LSTMDecoder, Lambda, MultimodalLearner
from metalearner import MetaLearner
from meta_base_models import LinearModel, Task
from torch.utils.tensorboard import SummaryWriter

import numpy as np

sys.path.insert(1, "..")

from ts_dataset import TSDataset
from base_models import LSTMModel, FCN
from metrics import torch_mae as mae
from pytorchtools import EarlyStopping, to_torch


def test(loss_fn, maml, multimodal_model, model_name, dataset_name, data_ML, adaptation_steps, learning_rate, noise_level, noise_type, is_test = True, horizon = 10):
    
    total_tasks = len(data_ML)
    task_size = data_ML.x.shape[-3]
    input_dim = data_ML.x.shape[-1]
    window_size = data_ML.x.shape[-2]
    output_dim = data_ML.y.shape[-1]

    if is_test:
        step = total_tasks//100

    else:
        step = 1

    step = 1 if step == 0 else step

     for task in range(0, (total_tasks_test-horizon-1), step):

        temp_file_idx = test_data_ML.file_idx[task:task+horizon+1]
        if(len(np.unique(temp_file_idx))>1):
            continue
        
        if model_name == "LSTM":
            model2 = LSTMModel( batch_size=None, seq_len = None, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =1)
        elif model_name == "FCN":
            kernels = [8,5,3] if window_size != 5 else [4,2,1]
            model2 = FCN(time_steps = window_size,  channels=[input_dim, 128, 128, 128] , kernels=kernels)

        
        #model2.cuda()
        #model2.load_state_dict(copy.deepcopy(maml.module.state_dict()))
        #opt2 = optim.Adam(model2.parameters(), lr=learning_rate)
        learner = maml.clone() 

        x_spt, y_spt = test_data_ML[task]
        x_qry = test_data_ML.x[(task+1):(task+1+horizon)].reshape(-1, window_size, input_dim)
        y_qry = test_data_ML.y[(task+1):(task+1+horizon)].reshape(-1, output_dim)
        #x_qry = test_data_ML.x[(task+1)].reshape(-1, window_size, input_dim)
        #y_qry = test_data_ML.y[(task+1)].reshape(-1, output_dim)

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

        for step in range(adaptation_steps):


            pred = learner(model.encoder(x_spt))
            error = loss_fn(pred, y_spt)
            learner.adapt(error)

    

        y_pred = learner(model.encoder(x_qry))
        
        y_pred = torch.clamp(y_pred, 0, 1)
        error = mae(y_pred, y_qry)
        
        accum_error += error.data
        accum_std += error.data**2
        count += 1
        
    error = accum_error/count

    print("std:", accum_std/count)   

    return error


def main(args):
    dataset_name = args.dataset
    model_name = args.model
    adaptation_steps = args.adaptation_steps
    meta_learning_rate = args.meta_learning_rate
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    save_model_file = args.save_model_file
    load_model_file = args.load_model_file
    lower_trial = args.lower_trial
    upper_trial = args.upper_trial
    task_size = args.task_size
    noise_level = args.noise_level
    noise_type = args.noise_type
    epochs = args.epochs
    loss_fcn_str = args.loss
    modulate_task_net = args.modulate_task_net
    weight_vrae = args.weight_vrae
    stopping_patience = args.stopping_patience

    meta_info = {"POLLUTION": [5, 14],
                 "HR": [32, 13],
                 "BATTERY": [20, 3]}

    assert model_name in ("FCN", "LSTM"), "Model was not correctly specified"
    assert dataset_name in ("POLLUTION", "HR", "BATTERY")

    window_size, input_dim = meta_info[dataset_name]

    grid = [0., noise_level]

    train_data_ML = pickle.load(
        open("../../Data/TRAIN-" + dataset_name + "-W" + str(window_size) + "-T" + str(task_size) + "-ML.pickle", "rb"))
    validation_data_ML = pickle.load(
        open("../../Data/VAL-" + dataset_name + "-W" + str(window_size) + "-T" + str(task_size) + "-ML.pickle", "rb"))
    test_data_ML = pickle.load(
        open("../../Data/TEST-" + dataset_name + "-W" + str(window_size) + "-T" + str(task_size) + "-ML.pickle", "rb"))

    total_tasks = len(train_data_ML)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = mae if loss_fcn_str == "MAE" else nn.SmoothL1Loss()

    ##multimodal learner parameters
    # paramters wto increase capactiy of the model
    n_layers_task_net = 2
    n_layers_task_encoder = 1
    n_layers_task_decoder = 1

    hidden_dim_task_net = 120
    hidden_dim_encoder = 120
    hidden_dim_decoder = 120

    # fixed values
    input_dim_task_net = input_dim
    input_dim_task_encoder = input_dim + 1
    output_dim_task_net = 1
    output_dim_task_decoder = input_dim + 1
    output_dim = 1

    first_order = False
    inner_loop_grad_clip = 20

    for trial in range(lower_trial, upper_trial):

        output_directory = "../../Models/" + dataset_name + "_" + model_name + "_MMAML/" + str(trial) + "/"
        save_model_file_ = output_directory + save_model_file
        save_model_file_encoder = output_directory + "encoder_" + save_model_file
        load_model_file_ = output_directory + load_model_file
        checkpoint_file = output_directory + "checkpoint_" + save_model_file.split(".")[0]

        writer = SummaryWriter()

        try:
            os.mkdir(output_directory)
        except OSError as error:
            print(error)

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
        opt = optim.Adam(list(maml.parameters()) + list(multimodal_learner.parameters()), lr=meta_learning_rate)

        early_stopping = EarlyStopping(patience=stopping_patience, model_file=save_model_file_, verbose=True)
        early_stopping_encoder = EarlyStopping(patience=stopping_patience, model_file=save_model_file_encoder, verbose=True)

        task_data_train = torch.FloatTensor(get_task_encoder_input(train_data_ML))
        task_data_validation = torch.FloatTensor(get_task_encoder_input(validation_data_ML))
        task_data_test = torch.FloatTensor(get_task_encoder_input(test_data_ML))

        val_loss_hist = []
        test_loss_hist = []
        total_num_tasks = train_data_ML.x.shape[0]

        for epoch in range(epochs):

            opt.zero_grad()
            iteration_error = 0.0
            vrae_loss_accum = 0.0

            multimodal_learner.train()

            for _ in range(batch_size):
                learner = maml.clone()
                task_idx = np.random.randint(0,total_num_tasks-1)
                task = task_data_train[task_idx].cuda()

                if train_data_ML.file_idx[task+1] != train_data_ML.file_idx[task]:
                    continue

                x_spt, y_spt = train_data_ML[task_idx]
                x_qry, y_qry = train_data_ML[task_idx + 1]
                x_qry = x_qry.reshape(-1, window_size, input_dim)
                y_qry = y_qry.reshape(-1, output_dim)

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

                vrae_loss_accum = 0.0

                x_spt_encoding, (vrae_loss, _, _) = multimodal_learner(x_spt, task,output_encoding=True)
                
                for _ in range(adaptation_steps):
                
                    pred = learner(x_spt_encoding)
                    error = loss_fn(pred, y_spt)
                    learner.adapt(error)#, allow_unused=True)#, allow_nograd=True)
                                                                                        
     
                vrae_loss_accum += vrae_loss

                x_qry_encoding, _ = multimodal_learner(x_qry, task, output_encoding=True)
                pred = learner(x_qry_encoding)
                evaluation_error = loss_fn(pred, y_qry)
                iteration_error += evaluation_error

            iteration_error /= batch_size
            vrae_loss_accum /= batch_size
            iteration_error += vrae_loss_accum
            iteration_error.backward()
            opt.step()

            # print(vrae_loss)

            ##plotting grad of output layer
            #for tag, parm in output_layer.linear.named_parameters():
            #    writer.add_histogram("Grads_output_layer_" + tag, parm.grad.data.cpu().numpy(), epoch)

            multimodal_learner.eval()
            val_loss = test(validation_data_ML, multimodal_learner, meta_learner, task_data_validation)
            test_loss = test(test_data_ML, multimodal_learner, meta_learner, task_data_test)

            print("Epoch:", epoch)
            print("Train loss:", mean_loss)
            print("Val error:", val_loss)
            print("Test error:", test_loss)

            early_stopping(val_loss, meta_learner)
            early_stopping_encoder(val_loss, multimodal_learner)

            val_loss_hist.append(val_loss)
            test_loss_hist.append(test_loss)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            writer.add_scalar("Loss/train", mean_loss.cpu().detach().numpy(), epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)

        multimodal_learner.load_state_dict(torch.load(save_model_file_encoder))
        output_layer.load_state_dict(torch.load(save_model_file_)["model_state_dict"])
        meta_learner = MetaLearner(output_layer, opt, learning_rate, loss_fn, first_order, n_inner_iter,
                                   inner_loop_grad_clip, device)

        val_loss = test(validation_data_ML, multimodal_learner, meta_learner, task_data_validation)
        test_loss = test(test_data_ML, multimodal_learner, meta_learner, task_data_test)

        with open(output_directory + "/results3.txt", "a+") as f:
            f.write("Dataset :%s \n" % dataset_name)
            f.write("Test error: %f \n" % test_loss)
            f.write("Val error: %f \n" % val_loss)
            f.write("\n")

        writer.add_hparams({"fast_lr": learning_rate,
                            "slow_lr": meta_learning_rate,
                            "adaption_steps": n_inner_iter,
                            "patience": stopping_patience,
                            "weight_vrae": weight_vrae,
                            "noise_level": noise_level,
                            "dataset": dataset_name,
                            "trial": trial},
                           {"val_loss": val_loss,
                            "test_loss": test_loss})


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use, possible: POLLUTION, HR, BATTERY',
                           default="POLLUTION")
    argparser.add_argument('--model', type=str, help='base model, possible: LSTM, FCN', default="LSTM")
    argparser.add_argument('--adaptation_steps', type=int, help='number of updates in the inner loop', default=5)
    argparser.add_argument('--learning_rate', type=float, help='learning rate for the inner loop', default=0.01)
    argparser.add_argument('--meta_learning_rate', type=float, help='learning rate for the outer loop', default=0.005)
    argparser.add_argument('--batch_size', type=int, help='batch size for the meta-upates (outer loop)', default=20)
    argparser.add_argument('--save_model_file', type=str, help='name to save the model in memory', default="model.pt")
    argparser.add_argument('--load_model_file', type=str, help='name to load the model in memory', default="model.pt")
    argparser.add_argument('--lower_trial', type=int, help='identifier of the lower trial value', default=0)
    argparser.add_argument('--upper_trial', type=int, help='identifier of the upper trial value', default=3)
    argparser.add_argument('--is_test', type=int, help='whether apply on test (1) or validation (0)', default=0)
    argparser.add_argument('--stopping_patience', type=int, help='patience for early stopping', default=30)
    argparser.add_argument('--epochs', type=int, help='epochs', default=2000)
    argparser.add_argument('--noise_level', type=float, help='noise level', default=0.0)
    argparser.add_argument('--noise_type', type=str, help='noise type', default="additive")
    argparser.add_argument('--task_size', type=int, help='Task size', default=50)
    argparser.add_argument('--loss', type=str, help='Loss used in training, possible: MAE, SmoothL1', default="SmoothL1")
    argparser.add_argument('--modulate_task_net', type=int,
                           help='Whether to use conditional layer for modulation or not', default=1)
    argparser.add_argument('--weight_vrae', type=float, help='Weight for VRAE', default=1.0)

    args = argparser.parse_args()

    main(args)
