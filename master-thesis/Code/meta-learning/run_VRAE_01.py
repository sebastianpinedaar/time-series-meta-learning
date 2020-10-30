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
from metalearner import MetaLearner
from meta_base_models import LinearModel, Task
from torch.utils.tensorboard import SummaryWriter

import numpy as np

sys.path.insert(1, "..")

from ts_dataset import TSDataset
from base_models import LSTMModel, FCN
from metrics import torch_mae as mae
from pytorchtools import EarlyStopping, to_torch


def test(data_ML, multimodal_learner, task_data, horizon=10):
    total_tasks = len(data_ML)
    input_dim = data_ML.x.shape[-1]
    window_size = data_ML.x.shape[-2]

    accum_error = 0.0
    count = 0

    for task_id in range(0, (total_tasks - horizon - 1), total_tasks // 100):
        x_spt, y_spt = data_ML[task_id]
        x_qry = data_ML.x[(task_id + 1):(task_id + 1 + horizon)].reshape(-1, window_size, input_dim)
        task = task_data[task_id:task_id + 1].cuda()

        x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)

        x_spt_encod, (vrae_loss, _, _) = multimodal_learner(x_spt, task, output_encoding=True)

        count += 1
        accum_error += vrae_loss.data

    return accum_error.cpu().detach().numpy() / count


def main(args):
    dataset_name = args.dataset
    model_name = args.model
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    save_model_file = args.save_model_file
    load_model_file = args.load_model_file
    task_size = args.task_size
    noise_level = args.noise_level
    noise_type = args.noise_type
    epochs = args.epochs
    loss_fcn_str = args.loss
    modulate_task_net = args.modulate_task_net
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
    n_layers_task_encoder = 2
    n_layers_task_decoder = 2

    hidden_dim_task_net = 120
    hidden_dim_encoder = 120
    hidden_dim_decoder = 120

    # fixed values
    input_dim_task_net = input_dim
    input_dim_task_encoder = input_dim + 1
    output_dim_task_net = 1
    output_dim_task_decoder = input_dim + 1

    first_order = False
    inner_loop_grad_clip = 20


    trial = 0
    output_directory = "../../Models/" + dataset_name + "_" + model_name + "_MMAML/" + str(trial) + "/"
    save_model_file_ = output_directory + "vrae_"+ save_model_file
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

    output_layer = LinearModel(120, 1)
    opt = torch.optim.Adam(list(multimodal_learner.parameters()),
                           lr=learning_rate)



    early_stopping = EarlyStopping(patience=stopping_patience, model_file=save_model_file_, verbose=True)

    task_data_train = torch.FloatTensor(get_task_encoder_input(train_data_ML))
    task_data_validation = torch.FloatTensor(get_task_encoder_input(validation_data_ML))
    task_data_test = torch.FloatTensor(get_task_encoder_input(test_data_ML))

    val_loss_hist = []
    test_loss_hist = []

    for epoch in range(epochs):

        multimodal_learner.train()

        #batch_idx = np.random.randint(0, total_tasks - 1, batch_size)

        for idx in range(0, total_tasks-1-batch_size, batch_size):

            batch_idx = np.arange(idx, idx+batch_size)
            task = task_data_train[batch_idx].cuda()

            x_spt, y_spt = train_data_ML[batch_idx]
            x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)

            x_spt_encodings = []
            x_qry_encodings = []
            vrae_loss_accum = 0.0
            for i in range(batch_size):
                x_spt_encoding, (vrae_loss, kl_loss, rec_loss) = multimodal_learner(x_spt[i], task[i:i + 1],
                                                                                    output_encoding=True)
                x_spt_encodings.append(x_spt_encoding)
                vrae_loss_accum += vrae_loss


           ##plotting grad of output layer


            vrae_loss_accum /= batch_size

            opt.zero_grad()
            vrae_loss_accum.backward()
            opt.step()

        #for tag, parm in multimodal_learner.task_net.named_parameters():
        #    writer.add_histogram("Grads_" + tag, parm.grad.data.cpu().numpy(), epoch)
        val_loss = test(validation_data_ML, multimodal_learner, task_data_validation)
        test_loss = test(test_data_ML, multimodal_learner, task_data_test)

        print("Epoch:", epoch)
        print("Train loss:", vrae_loss_accum)
        print("Test loss:", test_loss)
        print("Val loss:", val_loss)


        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.add_scalar("Loss/train", vrae_loss_accum.cpu().detach().numpy(), epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)


    with open(output_directory + "/results3.txt", "a+") as f:
        f.write("VRAE pre-training...")
        f.write("Dataset :%s \n" % dataset_name)
        f.write("Test error: %f \n" % test_loss)
        f.write("Val error: %f \n" % val_loss)
        f.write("\n")



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
