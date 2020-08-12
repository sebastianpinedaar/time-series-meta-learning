import pickle
import random
import sys
from ts_dataset import TSDataset
from sklearn.metrics import mean_absolute_error as mae #
from sklearn.metrics import mean_squared_error  as mse #
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel,  DotProduct,  RationalQuadratic 
from utils import progressBar



#dataset_name = "POLLUTION"
#window_size = 5
#task_size = 25
#hyperparams = [0.1, 2, 3, 100]

dataset_name = "HR"
window_size = 32
task_size = 25
hyperparams =  [0.05, 2, 3, 100]

#dataset_name = "BATTERY"
#window_size = 20
#task_size = 25
#hyperparams = [0.05, 7, 1, 100]

train_data = pickle.load(  open( "../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
validation_data = pickle.load( open( "../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
test_data = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
test_data_ML = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )

new_train_data = train_data + validation_data
test_size_ML = test_data_ML.x.shape[0]
dim = train_data.x.shape[-1]

new_train_data.x = new_train_data.x.reshape(-1, dim*window_size)

mse_list = []
mae_list = []

accumulated_mae = 0
accumulated_mse = 0

print(test_size_ML, " points to process...")

for i in range(test_size_ML-3):


    x_train = test_data_ML.x[i].reshape(-1, window_size*dim)
    y_train = test_data_ML.y[i]

    temp_kernel = ConstantKernel(1.0, (1e-3, 1e3))* RationalQuadratic () + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=temp_kernel, random_state=0)
    gpr.fit(x_train, y_train)

    y_pred = gpr.predict(test_data_ML.x[i+3].reshape(-1, window_size*dim))

    temp_mae = mae(y_pred, test_data_ML.y[i+3])
    temp_mse = mse(y_pred, test_data_ML.y[i+3])

    mae_list.append(temp_mae)
    mse_list.append(temp_mse)

    accumulated_mae = (i/(i+1))*accumulated_mae+temp_mae/(i+1)
    accumulated_mse = (i/(i+1))*accumulated_mse+temp_mse/(i+1)

    progressBar(i, test_size_ML, 100)

print("Accumulated_mae ", accumulated_mae)
print("Accumulated_mse ", accumulated_mse)