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
from meta_base_models import LSTMModel, Task
import os

sys.path.insert(1, "..")

from ts_dataset import TSDataset
from metrics import torch_mae as mae
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
from pytorchtools import EarlyStopping, to_torch, get_grad_norm
import argparse



def test (test_data_ML, meta_learner, device, horizon = 10):

    total_tasks_test = len(test_data_ML)
    task_size = test_data_ML.x.shape[-3]
    input_dim = test_data_ML.x.shape[-1]
    window_size = test_data_ML.x.shape[-2]
    output_dim = test_data_ML.y.shape[-1]

    accum_error = 0.0
    count = 0

    for task in range(0, (total_tasks_test-horizon-1), total_tasks_test//100):

        x_spt, y_spt = test_data_ML[task]
        x_qry = test_data_ML.x[(task+1):(task+1+horizon)].reshape(-1, window_size, input_dim)
        y_qry = test_data_ML.y[(task+1):(task+1+horizon)].reshape(-1, output_dim)
        
        x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
        x_qry = to_torch(x_qry)
        y_qry = to_torch(y_qry)

        train_task = [Task(x_spt, y_spt)]
        val_task = [Task(x_qry, y_qry)]

        adapted_params = meta_learner.adapt(train_task)
        mean_loss = meta_learner.step(adapted_params, val_task, is_training = 0)
        #y_pred = metalearner._model(x_qry, adapted_params[0])

        #error = mae(y_pred, y_qry)

        count += 1
        #accum_error += error.cpu().data
        accum_error += mean_loss.cpu().data

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

    first_order = False
    inner_loop_grad_clip = 20
    task_size = 50
    output_dim = 1

    horizon = 10
    ##test


    meta_info = {"POLLUTION": [5, 50, 14],
                 "HR": [32, 50, 13],
                 "BATTERY": [20, 50, 3] }

    assert model_name in ("FCN", "LSTM"), "Model was not correctly specified"
    assert dataset_name in ("POLLUTION", "HR", "BATTERY")

    window_size, task_size, input_dim = meta_info[dataset_name]

    output_directory = "output/"

    train_data_ML = pickle.load( open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    validation_data_ML = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    test_data_ML = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )


    for trial in range(lower_trial, upper_trial):

        output_directory = "../../Models/"+dataset_name+"_"+model_name+"_MAML/"+str(trial)+"/"
        save_model_file_ = output_directory + save_model_file
        load_model_file_ = output_directory + load_model_file


        try:
            os.mkdir(output_directory)
        except OSError as error: 
            print(error)

        f=open(output_directory+"/results.txt", "a+")
        f.write("Learning rate :%f \n"% fast_lr)
        f.write("Meta-learning rate: %f \n" % slow_lr)
        f.write("Adaptation steps: %f \n" % n_inner_iter)
        f.write("\n")
        f.close()


        if model_name == "LSTM":
            model = LSTMModel( batch_size=batch_size, seq_len = window_size, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =output_dim)

        optimizer = torch.optim.Adam(model.parameters(), lr = slow_lr)
        loss_func = mae

        torch.backends.cudnn.enabled = False

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        meta_learner = MetaLearner(model, optimizer, fast_lr , loss_func,
                        first_order, n_inner_iter, inner_loop_grad_clip,
                        device)

        total_tasks, task_size, window_size, input_dim = train_data_ML.x.shape
   

        early_stopping = EarlyStopping(patience=stopping_patience, model_file=save_model_file_, verbose=True)

        for _ in range(epochs):

            
            #train
            batch_idx = np.random.randint(0, total_tasks-1, batch_size)
            x_spt, y_spt = train_data_ML[batch_idx]
            x_qry, y_qry = train_data_ML[batch_idx+1]
            
            x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
            x_qry = to_torch(x_qry)
            y_qry = to_torch(y_qry)

            train_tasks = [Task(x_spt[i], y_spt[i]) for i in range(x_spt.shape[0])]
            val_tasks = [Task(x_qry[i], y_qry[i]) for i in range(x_qry.shape[0])]

            adapted_params = meta_learner.adapt(train_tasks)
            mean_loss = meta_learner.step(adapted_params, val_tasks, is_training = True)
            print(mean_loss)

            #test
            val_error = test(validation_data_ML, meta_learner, device)
            print(val_error)

            early_stopping(val_error, meta_learner)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.load_state_dict(torch.load(save_model_file_)["model_state_dict"])
        meta_learner = MetaLearner(model, optimizer, fast_lr , loss_func,
                        first_order, n_inner_iter, inner_loop_grad_clip,
                        device)

        validation_error = test(validation_data_ML, meta_learner, device)
        test_error = test(test_data_ML, meta_learner, device)

        with open(output_directory+"/results.txt", "a+") as f:
            f.write("Test error: %f \n" % test_error)
            f.write("Validation error: %f \n" %validation_error)
        
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
    argparser.add_argument('--save_model_file', type=str, help='name to save the model in memory', default="model.pt")
    argparser.add_argument('--load_model_file', type=str, help='name to load the model in memory', default="model.pt")
    argparser.add_argument('--lower_trial', type=int, help='identifier of the lower trial value', default=0)
    argparser.add_argument('--upper_trial', type=int, help='identifier of the upper trial value', default=3)
    argparser.add_argument('--is_test', type=int, help='whether apply on test (1) or validation (0)', default=0)
    argparser.add_argument('--stopping_patience', type=int, help='patience for early stopping', default=30)
    argparser.add_argument('--epochs', type=int, help='epochs', default=2000)

    args = argparser.parse_args()

    main(args)
