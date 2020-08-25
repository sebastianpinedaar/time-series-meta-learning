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

sys.path.insert(1, "DANN_py3")

mode = "val"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_names = [("POLLUTION", 5, 50)]

output_dim = 1
x_dim = 14
h_dim = 50
z_dim = 16
h_dim_reg = 25
n_layers =  1
out_dim = 1
n_epochs = 100
clip = 10
learning_rate = 3e-4
batch_size = 128
#seed = 128
print_every = 100
save_every = 10
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0}


dataset_name = "POLLUTION"
task_size = 50
window_size = 5


output_directory = "../Models/"+dataset_name+"_H"+str(h_dim)+"_Z"+str(z_dim)+"/"


def step(vrada, data_iter, len_dataloader, is_train=False):

    i = 0
    while i < len_train_loader:

        p = float(i + epoch * len_dataloader) / n_epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = train_iter.next()
        x, y, d = data_source

        vrada.zero_grad()
        
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)
        d = torch.tensor(d).long().to(device)

        regressor_output, domain_class_output, kld_loss, nll_loss = vrada(x, alpha)       
        err_reg = loss_regression(regressor_output, y.to(device) )
        err_dom = loss_domain(domain_class_output.float(), d.long().to(device))

        # training model using target data
        err = err_reg + 0.1*err_dom + 0.000001*kld_loss+ 0.000001*nll_loss

        if is_train:
            err.backward()
            optimizer.step()

        print(err)

        i += 1

    return alpha, (float(err.data), float(err_reg.data), float(err_dom.data), float(kld_loss.data), float(nll_loss.data))
        

try:
    os.mkdir(output_directory)
except OSError as error: 
    print(error)  

f=open(output_directory+"results.txt", "a+")
f.write("Dataset :%s"% dataset_name)
f.write("\n")
f.close()

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

x_dim = train_data.x.shape[-1]
n_domains = np.max(train_data.file_idx)+1

vrada = VRADA(x_dim, h_dim, h_dim_reg, z_dim, out_dim, n_domains, n_layers, device)
vrada.cuda()

optimizer = optim.Adam(vrada.parameters(), lr=0.0001)

loss_regression = mae
loss_domain = torch.nn.NLLLoss()

for epoch in range(n_epochs):

    len_train_loader = len(domain_train_loader)
    train_iter = iter(domain_train_loader)

    len_val_loader = len(domain_val_loader)
    val_iter = iter(domain_val_loader)

    alpha, err_reg_train, err_dom_train, _ , _ = step(vrada, train_iter, len_train_loader, True)
    alpha, err_reg_val, err_dom_val, _ , _ = step(vrada, train_iter, len_train_loader, True)

    print ('epoch: %d, \n TRAINING -> err_reg: %f, err_domain: %f' % (epoch, err_reg_train, err_dom_train))
    print ('epoch: %d, \n VAL -> err_reg: %f, err_domain: %f' % (epoch, err_reg_val, err_dom_val))

    #torch.save(my_net, '{0}/mnist_mnistm_model_epoch_{1}.pth'.format(model_root, epoch))
    #test(source_dataset_name, epoch)
    #test(target_dataset_name, epoch)

print('done')