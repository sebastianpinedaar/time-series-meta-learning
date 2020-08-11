from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel,  DotProduct,  RationalQuadratic 
import xgboost as xgb
import pickle

from ts_dataset import TSDataset
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import pandas as pd

dataset_names = [ ("HR", 32, 25), ("BATTERY", 20, 25)]#, ("POLLUTION", 5, 25)]

for dataset, window_size, task_size in dataset_names:

    validation_data = pickle.load( open( "../Data/VAL-"+dataset+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    test_data = pickle.load( open( "../Data/TEST-"+dataset+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )


    dim = validation_data.dim

    C= ConstantKernel(1.0, (1e-3, 1e3)) 
    noise = WhiteKernel()
    kernels = [  RBF(10, (1e-2, 1e2)), DotProduct(), RationalQuadratic () ]
    kernel_trials = 20

    used_kernels = []
    scores = []

    for trial in range(kernel_trials):

        temp_scores = []
        temp_kernel = noise

        for j in range(trial%(len(kernels))+1):
            temp_kernel += C*kernels[int(np.random.randint(0,len(kernels)))]

        for i in range( validation_data.x.shape[0]-1):
            x_train = validation_data.x[i]
            y_train = validation_data.y[i]

            x_test = validation_data.x[i+1]
            y_test = validation_data.y[i+1]
            
            gpr = GaussianProcessRegressor(kernel=temp_kernel,
                random_state=0)
            gpr.fit(x_train.reshape(-1, dim*window_size) , y_train)
            y_pred = gpr.predict(x_test.reshape(-1, dim*window_size))
            score= mae(y_pred , y_test)
            temp_scores.append(score)

        scores.append(np.mean(temp_scores))
        used_kernels.append(temp_kernel)
        print(np.mean(temp_scores))

    pd.DataFrame({"Kernel":used_kernels, "Score": scores}).to_csv("../Data/results/"+dataset+"_hyper_gp.csv")