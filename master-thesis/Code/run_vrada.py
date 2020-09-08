import random
import os
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
from pytorchtools import count_parameters
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from metrics import torch_mae as mae
from VariationalRecurrentNeuralNetwork.model import VRNN
import sys
from vrada import VRADA 
from ts_dataset import DomainTSDataset
from pytorchtools import EarlyStopping
from ts_transform import split_idx_50_50
import argparse

sys.path.insert(1, "DANN_py3")



def step(vrada, data_iter, len_dataloader, epochs = 500, epoch = 0, lambda1 = 0.1,  optimizer = None, loss = mae, is_train=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_regression = loss
    loss_domain = torch.nn.NLLLoss()

    if is_train:
        vrada.train()
    else:
        vrada.eval()

    accum_err_reg = 0
    accum_size = 0
    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_iter.next()
        x, y, d = data_source

        if is_train:
            vrada.zero_grad()
        
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)
        d = torch.tensor(d).long().to(device)
        batch_size = x.shape[0]

        regressor_output, domain_class_output, kld_loss, nll_loss = vrada(x, alpha)       
        err_reg = loss_regression(regressor_output, y.to(device) )
        err_dom = loss_domain(domain_class_output.float(), d.long().to(device))

        # training model using target data
        err = err_reg + lambda1*err_dom + (kld_loss+ nll_loss)*(1/batch_size)

        if is_train:
            err.backward()
            optimizer.step()

        #print(err)
        accum_err_reg +=err_reg*x.shape[0]
        accum_size += x.shape[0]
        i += 1

    return alpha, float(accum_err_reg/accum_size)
        


def train(vrada, domain_train_loader, domain_val_loader, lambda1 = 0.1, early_stopping = None, learning_rate = 0.001, epochs = 500 , monitor_stopping = True):

    optimizer = optim.Adam(vrada.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = early_stopping.patience//4, verbose=True)

    for epoch in range(epochs):

        len_train_loader = len(domain_train_loader)
        train_iter = iter(domain_train_loader)

        len_val_loader = len(domain_val_loader)
        val_iter = iter(domain_val_loader)

        alpha, mean_err_reg_train = step(vrada, train_iter, len_train_loader, epochs, epoch, lambda1, optimizer, mae, True)

        with torch.no_grad():
            alpha, mean_err_reg_val= step(vrada, val_iter, len_val_loader, epochs)

        print ('epoch: %d, \n TRAINING -> mean_err: %f' % (epoch, mean_err_reg_train))
        print ('epoch: %d, \n VAL -> mean_err: %f' % (epoch, mean_err_reg_val))

        scheduler.step(mean_err_reg_val)

        #torch.save(my_net, '{0}/mnist_mnistm_model_epoch_{1}.pth'.format(model_root, epoch))
        #test(source_dataset_name, epoch)
        #test(target_dataset_name, epoch)

        if monitor_stopping:
            early_stopping(mean_err_reg_val, vrada)
            
            #vrada.load_state_dict(torch.load(model_file))

            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('done')

    return epoch+1


def test(vrada, domain_test_loader,  output_directory, model_file, verbose = True):

    len_test_loader = len(domain_test_loader)
    test_iter = iter(domain_test_loader)

    vrada.load_state_dict(torch.load(model_file))

    with torch.no_grad():
        alpha, mean_err_reg_test = step(vrada, test_iter, len_test_loader)

    print("Regression error on test: %f"%(mean_err_reg_test))

    if verbose:
        f=open(output_directory+"/results.txt", "a+")
        f.write("Test error :%f"% mean_err_reg_test)
        f.write("\n")
        f.close()

    return mean_err_reg_test


def freeze_vrada(vrada):
    for params in vrada.named_parameters():
        if(params[0][:9]!="regressor"):
            params[1].requires_grad=False
    


def main(args):

    meta_info = {"POLLUTION": [5, 50, 14],
                "HR": [32, 50, 13],
                "BATTERY": [20, 50, 3] }

    #variables
    dataset_name = args.dataset 
    mode = args.mode 
    save_model_file = args.save_model_file
    load_model_file = args.load_model_file
    lower_trial = args.lower_trial
    upper_trial = args.upper_trial
    learning_rate = args.learning_rate
    regularization_penalty = args.regularization_penalty
    is_test = args.is_test
    patience_stopping = args.patience_stopping
    capacity = args.capacity
    lambda1 = args.lambda1
    epochs = args.epochs
    learning_rate = args.learning_rate

    lambda_per_dataset = {"POLLUTION-LOW": 0.1, "HR-LOW": 0.01, "BATTERY-LOW":0.01,
                    "POLLUTION-HIGH":0.001, "HR-HIGH": 0.1, "BATTERY-HIGH":0.0001}

    lambda1 = lambda_per_dataset[dataset_name+"-"+capacity]

    if capacity == "HIGH":
        h_dim = 100
        z_dim = 100
        h_dim_reg = 50
    elif capacity == "LOW":
        h_dim = 50 #100
        z_dim = 16 #100
        h_dim_reg = 25 #50

    else:
        raise Exception("Capacity is not well specified.")
    
    n_layers =  1
    out_dim = 1
    clip = 10
    batch_size = 256
    params = {'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 0}

    horizon = 10
    window_size, task_size, x_dim = meta_info[dataset_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for trial in range(lower_trial,upper_trial):
        
        output_directory = "../Models/"+dataset_name+"_H"+str(h_dim)+"_Z"+str(z_dim)+"/"+str(trial)+"/"
        try:
            os.mkdir(output_directory)
        except OSError as error: 
            print(error)  

        
        train_data = pickle.load(  open( "../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
        validation_data = pickle.load( open( "../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
        validation_data_ML = pickle.load( open( "../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
        test_data = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
        test_data_ML = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )

        domain_train_data = DomainTSDataset(train_data)
        domain_train_loader = DataLoader(domain_train_data, **params)
        domain_val_data = DomainTSDataset(validation_data)
        domain_val_loader = DataLoader(domain_val_data, **params)
        domain_test_data = DomainTSDataset(test_data)
        domain_test_loader = DataLoader(domain_test_data, **params)

        f=open(output_directory+"/results.txt", "a+")
        f.write("Dataset :%s"% dataset_name)
        f.write("\n")
        f.close()

        x_dim = train_data.x.shape[-1]
        n_domains = np.max(train_data.file_idx)+1
        lambda1 = lambda_per_dataset[dataset_name+"-"+capacity]
        
        loss_regression = mae
        loss_domain = torch.nn.NLLLoss()

        if mode =="WOFT":

            print("MODE:", mode)
            f=open(output_directory+"/results.txt", "a+")
            f.write("Mode :%s"% mode)
            f.write("\n")
            f.close()

            model_file = output_directory+"model.pt"
            early_stopping = EarlyStopping(patience=patience_stopping, model_file=model_file, verbose=True)
            vrada = VRADA(x_dim, h_dim, h_dim_reg, z_dim, out_dim, n_domains, n_layers, device)
            vrada.cuda()

            optimizer = optim.Adam(vrada.parameters(), lr=learning_rate)

            train(vrada, domain_train_loader, domain_val_loader, lambda1, early_stopping, learning_rate, epochs )
            test(vrada, domain_test_loader,output_directory, new_model_file)
            test(vrada, domain_test_loader,output_directory, new_model_file)
            test(vrada, domain_test_loader,output_directory, new_model_file)

        elif mode == "50":

            assert save_model_file!=load_model_file, "Files cannot be the same"

            print("MODE:", mode)
            f=open(output_directory+"/results.txt", "a+")
            f.write("Mode :%s"% mode)
            f.write("\n")
            f.close()

            new_model_file = output_directory+save_model_file
            model_file = output_directory+load_model_file
            train_idx, val_idx, test_idx = split_idx_50_50(test_data.file_idx)
            n_domains_in_test = np.max(test_data.file_idx)+1
            n_domains_in_train = np.max(train_data.file_idx)+1

            test_loss_list = []
            initial_test_loss_list = []

            for domain in range(n_domains_in_test):
                
                print("Domain:", domain)
                f=open(output_directory+"/results.txt", "a+")
                f.write("Mode :%d"% domain)
                f.write("\n")
                f.close()

                x_test = test_data.x
                y_test = test_data.y
                temp_train_data = DomainTSDataset(x=x_test[train_idx[domain]], 
                                                                        y= y_test[train_idx[domain]],
                                                                        d = [n_domains_in_train-1]*len(train_idx[domain]))
                
                domain_val_data = DomainTSDataset(x= x_test[val_idx[domain]],
                                                    y= y_test[val_idx[domain]],
                                                    d = [n_domains_in_train-1]*len(val_idx[domain]))

                                
                domain_test_data = DomainTSDataset(x= x_test[test_idx[domain]],
                                                    y= y_test[test_idx[domain]],
                                                    d = [n_domains_in_train-1]*len(test_idx[domain]))
                

                domain_train_loader = DataLoader(temp_train_data, **params)
                domain_val_loader = DataLoader(domain_val_data, **params)
                domain_test_loader = DataLoader(domain_test_data, **params)

                early_stopping = EarlyStopping(patience=patience_stopping, model_file=new_model_file, verbose=True)
                vrada = VRADA(x_dim, h_dim, h_dim_reg, z_dim, out_dim, n_domains_in_train, n_layers, device)
                vrada.cuda()

                vrada.load_state_dict(torch.load(model_file))
                freeze_vrada(vrada)
                optimizer = optim.Adam(vrada.parameters(), lr=learning_rate)
                initial_test_loss_list.append(test(vrada, domain_test_loader, output_directory, model_file))
                initial_test_loss_list.append(test(vrada, domain_test_loader, output_directory, model_file))
                initial_test_loss_list.append(test(vrada, domain_test_loader, output_directory, model_file)) 
                early_stopping(initial_test_loss_list[-1], vrada)             
                train(vrada, domain_train_loader, domain_val_loader, lambda1, early_stopping, learning_rate, epochs )
                test_loss_list.append(test(vrada, domain_test_loader, output_directory, new_model_file))
                test_loss_list.append(test(vrada, domain_test_loader, output_directory, new_model_file))
                test_loss_list.append(test(vrada, domain_test_loader, output_directory, new_model_file))

            total_loss = np.mean(test_loss_list)
            initial_loss = np.mean(initial_test_loss_list)
            f=open(output_directory+"/results.txt", "a+")
            f.write("Total error :%f"% total_loss)
            f.write("Initial Total error :%f"% initial_loss)
            f.write("\n")
            f.close()

        elif mode == "WFT":

            print("MODE:", mode)
            f=open(output_directory+"/results.txt", "a+")
            f.write("Mode :%s"% mode)
            f.write("\n")
            f.close()

            assert save_model_file!=load_model_file, "Files cannot be the same"

            new_model_file = output_directory+save_model_file
            model_file = output_directory+load_model_file

            n_tasks, task_size, dim, channels = test_data_ML.x.shape
            test_loss_list = []
            initial_test_loss_list = []

            for task_id in range(0, (n_tasks-horizon-1), n_tasks//100):
                

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

                early_stopping = EarlyStopping(patience=patience_stopping, model_file=new_model_file, verbose=True)
                vrada = VRADA(x_dim, h_dim, h_dim_reg, z_dim, out_dim, n_domains, n_layers, device)
                vrada.cuda()
                vrada.load_state_dict(torch.load(model_file))
                freeze_vrada(vrada)

                domain_train_data = DomainTSDataset(x = temp_x_train, y= temp_y_train, d = [0]*temp_x_train.shape[0])
                domain_val_data = DomainTSDataset(x = temp_x_val, y=temp_y_val, d=[0]*temp_x_val.shape[0])
                domain_test_data = DomainTSDataset(x = temp_x_test, y= temp_y_test, d= [0]*temp_x_test.shape[0])

                domain_train_loader = DataLoader(domain_train_data, **params)
                domain_val_loader = DataLoader(domain_val_data, **params)
                domain_test_loader = DataLoader(domain_test_data, **params)              

                optimizer = optim.Adam(vrada.parameters(), lr=learning_rate)

                initial_test_loss_list.append( test(vrada, domain_test_loader, output_directory, model_file, False))
                early_stopping(initial_test_loss_list[-1], vrada)  
                train(vrada, domain_train_loader, domain_val_loader, lambda1, early_stopping, learning_rate, epochs )
                test_loss_list.append( test(vrada, domain_test_loader, output_directory, new_model_file, False))
                print(np.mean(test_loss_list))

            total_loss = np.mean(test_loss_list)
            initial_total_loss = np.mean(initial_test_loss_list)
            f=open(output_directory+"/results.txt", "a+")
            f.write("Learning rate: %f \n" % learning_rate)
            f.write("Initial Total error :%f \n"% initial_total_loss)
            f.write("Total error :%f \n"% total_loss)
            f.write("Standard deviation: %f \n" %np.std(test_loss_list))
            f.write("\n")
            f.close()
    
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset to use, possible: POLLUTION, HR, BATTERY', default="POLLUTION")
    argparser.add_argument('--mode', type=str, help='evaluation mode, possible: WOFT, WFT, 50, HYP', default="WOFT")
    argparser.add_argument('--save_model_file', type=str, help='name to save the model in memory', default="model.pt")
    argparser.add_argument('--load_model_file', type=str, help='name to load the model in memory', default="model.pt")
    argparser.add_argument('--lower_trial', type=int, help='identifier of the lower trial value', default=0)
    argparser.add_argument('--upper_trial', type=int, help='identifier of the upper trial value', default=3)
    argparser.add_argument('--learning_rate', type=float, help='learning rate', default=3e-4)
    argparser.add_argument('--regularization_penalty', type=float, help='regularization penaly', default=0.0001)
    argparser.add_argument('--is_test', type=int, help='whether apply on test (1) or validation (0)', default=0)
    argparser.add_argument('--patience_stopping', type=int, help='patience for early stopping', default=20)
    argparser.add_argument('--lambda1', type=float, help='lambda value for the loss function', default=0.01)
    argparser.add_argument('--capacity', type=str, help='Capacity of the model', default="LOW")
    argparser.add_argument('--epochs', type=int, help='epochs', default=500)

    args = argparser.parse_args()

    main(args)