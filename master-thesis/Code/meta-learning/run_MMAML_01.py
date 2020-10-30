#multimodal learning (without apply maml, just task network with modulation network)

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
from multimodallearner import get_task_encoder_input
from multimodallearner import LSTMDecoder, Lambda, MultimodalLearner
import numpy as np

sys.path.insert(1, "..")

from ts_dataset import TSDataset
from base_models import LSTMModel, FCN
from metrics import torch_mae as mae
from pytorchtools import EarlyStopping


def test(data_ML, multimodal_learner, loss_fn):
    total_tasks, task_size, window_size, input_dim = data_ML.x.shape

    task_data = torch.FloatTensor(get_task_encoder_input(data_ML))
    x_tensor = torch.FloatTensor(data_ML.x)
    y_tensor = torch.FloatTensor(data_ML.y)

    count = 0.0
    accum_loss = 0.0

    for task_id in range(0, total_tasks, total_tasks // 100):
        task = task_data[task_id:task_id + 1].cuda()
        x = x_tensor[task_id + 1].cuda()
        y = y_tensor[task_id + 1].cuda()

        y_pred, (vrae_loss, kl_loss, rec_loss) = multimodal_learner(x, task)

        loss = loss_fn(y_pred, y)
        accum_loss += loss.cpu().detach().numpy()
        count += 1

    return accum_loss / count


def main(args):

    dataset_name = args.dataset
    model_name = args.model
    adaptation_steps = args.adaptation_steps
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    save_model_file = args.save_model_file
    load_model_file = args.load_model_file
    lower_trial = args.lower_trial
    upper_trial = args.upper_trial
    task_size = args.task_size
    noise_level = args.noise_level
    epochs = args.epochs
    loss_fcn_str = args.loss
    modulate_task_net = args.modulate_task_net
    weight_vrae = args.weight_vrae
    stopping_patience = args.stopping_patience

    meta_info = {"POLLUTION": [5, 14],
                 "HR": [32, 13],
                 "BATTERY": [20, 3] }


    assert model_name in ("FCN", "LSTM"), "Model was not correctly specified"
    assert dataset_name in ("POLLUTION", "HR", "BATTERY")

    window_size, input_dim = meta_info[dataset_name]

    grid = [0., noise_level]

    train_data_ML = pickle.load( open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    validation_data_ML = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    test_data_ML = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )

    total_tasks = len(train_data_ML)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = mae if loss_fcn_str == "MAE" else nn.SmoothL1Loss(size_average=False)

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

    for trial in range(lower_trial, upper_trial):

        output_directory = "../../Models/"+dataset_name+"_"+model_name+"_MMAML/"+str(trial)+"/"
        save_model_file_ = output_directory + save_model_file
        load_model_file_ = output_directory + load_model_file
        checkpoint_file = output_directory + "checkpoint_" + save_model_file.split(".")[0]

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

        opt = torch.optim.Adam(multimodal_learner.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=stopping_patience, model_file = save_model_file_, verbose = True)

        task_data = torch.FloatTensor(get_task_encoder_input(train_data_ML))
        x_tensor = torch.FloatTensor(train_data_ML.x)
        y_tensor = torch.FloatTensor(train_data_ML.y)

        val_loss_hist = []

        print("Total tasks:", total_tasks)

        for epoch in range(epochs):

            multimodal_learner.train()

            task_id = np.random.randint(0, total_tasks-1)
            task =task_data[task_id:task_id+1].cuda()
            x = x_tensor[task_id+1].cuda()
            y = y_tensor[task_id+1].cuda()

            y_pred, (vrae_loss, kl_loss, rec_loss) = multimodal_learner(x, task)

            loss = loss_fn(y_pred, y) + vrae_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            multimodal_learner.eval()
            with torch.no_grad():
                val_error = test(validation_data_ML, multimodal_learner, mae)
                test_error = test(test_data_ML, multimodal_learner, mae)

            print("Epoch:", epoch)
            print("Vrae loss:", vrae_loss)
            print("Train loss:", loss)
            print("Val error:", val_error)
            print("Test error:", test_error)

            early_stopping(val_error, multimodal_learner)
            val_loss_hist.append(val_error)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            fig = plt.figure()
            plt.plot(val_loss_hist)
            plt.savefig("val_hist.png")

        multimodal_learner.load_state_dict(torch.load(save_model_file_))

        val_error = test(validation_data_ML, multimodal_learner, mae)
        test_error = test(test_data_ML, multimodal_learner, mae)

        with open(output_directory+"/results3.txt", "a+") as f:
            f.write("Dataset :%s \n"% dataset_name)
            f.write("Test error: %f \n" % test_error)
            f.write("Val error: %f \n" % val_error)
            f.write("\n")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use, possible: POLLUTION, HR, BATTERY', default="POLLUTION")
    argparser.add_argument('--model', type=str, help='base model, possible: LSTM, FCN', default="LSTM")
    argparser.add_argument('--adaptation_steps', type=int, help='number of updates in the inner loop', default=5)
    argparser.add_argument('--learning_rate', type=float, help='learning rate for the inner loop', default=0.0001)
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
    argparser.add_argument('--modulate_task_net', type=int, help='Whether to use conditional layer for modulation or not', default=1)
    argparser.add_argument('--weight_vrae', type=float, help='Weight for VRAE', default=1.0)

    args = argparser.parse_args()

    main(args)