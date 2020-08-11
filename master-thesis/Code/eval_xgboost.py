import xgboost as xgb
import pickle
import random
import sys
from ts_dataset import TSDataset
from sklearn.metrics import mean_absolute_error as mae #
from sklearn.metrics import mean_squared_error  as mse #
import pandas as pd
import numpy as np

def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

#dataset_name = "POLLUTION"
#window_size = 5
#task_size = 25
#hyperparams = [0.1, 2, 3, 100]

#dataset_name = "HR"
#window_size = 32
#task_size = 25
#hyperparams =  [0.05, 2, 3, 100]

dataset_name = "BATTERY"
window_size = 20
task_size = 25
hyperparams = [0.05, 7, 1, 100]

train_data = pickle.load(  open( "../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
validation_data = pickle.load( open( "../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
test_data = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
test_data_ML = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )

new_train_data = train_data + validation_data
test_size_ML = test_data_ML.x.shape[0]
dim = train_data.x.shape[-1]

new_train_data.x = new_train_data.x.reshape(-1, dim*window_size)
test_data.x = test_data.x.reshape(-1, dim*window_size)

evaluation_list = ["WO-RT", "50"]# W-RT, WO-RT, 50

#50% evaluation

if "50" in evaluation_list:
        
    x_train = np.concatenate([new_train_data.x, test_data_ML.x[:test_size_ML//2].reshape(-1, window_size*dim)], axis=0)
    y_train = np.concatenate([new_train_data.y, test_data_ML.y[:test_size_ML//2].reshape(-1,1)], axis=0)

    x_test = test_data_ML.x[test_size_ML//2:].reshape(-1, window_size*dim)
    y_test = test_data_ML.y[test_size_ML//2:].reshape(-1, 1)

    model = xgb.XGBRegressor(learning_rate=hyperparams[0],
                        max_depth=hyperparams[1],
                        min_child_weight=hyperparams[2],
                        n_estimators=hyperparams[3])

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae_50pct = mae(y_pred, y_test)
    mse_50pct = mse(y_pred, y_test)

    print("Mae withtout retraining (50pct):", mae_50pct)
    print("mse without retraining (50pct):", mse_50pct)

#evaluation without retraining

if "WO-RT" in evaluation_list:
        
    model = xgb.XGBRegressor(learning_rate=hyperparams[0],
                        max_depth=hyperparams[1],
                        min_child_weight=hyperparams[2],
                        n_estimators=hyperparams[3])

    model.fit(new_train_data.x, new_train_data.y)
    y_pred = model.predict(test_data_ML.x.reshape(-1, dim*window_size))

    mae_wo_retraining = mae(y_pred, test_data_ML.y.reshape(-1, 1))
    mse_wo_retraining = mse(y_pred, test_data_ML.y.reshape(-1, 1))

    print("Mae withtout retraining (wo-rt):", mae_wo_retraining)
    print("mse without retraining (wo-rt):", mse_wo_retraining)


#evaluation with retraining

if "W-RT" in evaluation_list:

    mse_list = []
    mae_list = []

    accumulated_mae = 0
    accumulated_mse = 0

    print(test_size_ML, " points to process...")

    for i in range(test_size_ML-1):

        model = xgb.XGBRegressor(learning_rate=hyperparams[0],
                            max_depth=hyperparams[1],
                            min_child_weight=hyperparams[2],
                            n_estimators=hyperparams[3])    


        x_train = np.concatenate([new_train_data.x, test_data_ML.x[i].reshape(-1, window_size*dim)], axis=0)
        y_train = np.concatenate([new_train_data.y, test_data_ML.y[i]], axis=0)

        model.fit(x_train, y_train)
        y_pred = model.predict(test_data_ML.x[i+1].reshape(-1, window_size*dim))

        temp_mae = mae(y_pred, test_data_ML.y[i+1])
        temp_mse = mse(y_pred, test_data_ML.y[i+1])

        print("Temp mae:", temp_mae)
        print("Temp mse:", temp_mse)

        mae_list.append(temp_mae)
        mse_list.append(temp_mse)

        accumulated_mae = (i/(i+1))*accumulated_mae+temp_mae/(i+1)
        accumulated_mse = (i/(i+1))*accumulated_mse+temp_mse/(i+1)

        #  progressBar(i, test_size_ML, 100)
        print("Iteration ", i)
        print("Accumulated_mae ", accumulated_mae)
        print("Accumulated_mse ", accumulated_mse)
