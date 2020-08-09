from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import xgboost as xgb
import pickle
from ts_dataset import TSDataset
from sklearn.metrics import mean_absolute_error as mae #
import numpy as np
import pandas as pd

dataset_names = [ ("HR", 32, 25), ("BATTERY", 20, 25)]

hyperparams_grid = {"length_scale"    : [0.1, 1.0, 2.0, 10.0 ] ,
                "vertical_scale"      : [ 0.1, 1.0, 2.0, 10.0]}


validation_data = pickle.load( open( "../Data/VAL-POLLUTION-W5-T25-ML.pickle", "rb" ) )
test_data = pickle.load( open( "../Data/TEST-POLLUTION-W5-T25-ML.pickle", "rb" ) )

window_size = 5
dim = validation_data.dim

x_train= validation_data.x[:,:25//2,...]
y_train = validation_data.y[:,:25//2,...]

x_test = validation_data.x[:,25//2:,...]
y_test = validation_data.y[:,25//2:,...]

scores = []
for i in range(500, x_train.shape[0]):
    
    kernel = 1.0 * RBF(1.0)
    gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0)
    gpr.fit(x_train[i].reshape(-1, dim*window_size) , y_train[i])
    y_pred = gpr.predict(x_test[i].reshape(-1, dim*window_size))
    score= mae(y_pred , y_test[i])
    scores.append(scores)
print(np.mean(scores))