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

def split_idx_50_50(domain_idx):

    domain_idx = np.array(domain_idx)
    n_domains = np.max(domain_idx)+1
    print(n_domains)

    domain_change = [0]
    for d in range(n_domains):

        domain_change.append(np.sum(domain_idx==d))

    domain_change = np.cumsum(domain_change)
    print(domain_change)
    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(len(domain_change)-1):

        pos1 = domain_change[i]
        pos2 = (domain_change[i]+ int(0.9*(domain_change[i+1]-domain_change[i])/2))
       # pos22 = (domain_change[i]+ int(0.9*(domain_change[i+1]-domain_change[i])/2))
        pos3 = (domain_change[i]+domain_change[i+1])/2
        pos4 = domain_change[i+1]
        
        print(pos1)
        train_idx.append(np.arange(pos1, pos2).astype(int))
        val_idx.append(np.arange(pos2, pos3).astype(int))
        test_idx.append(np.arange(pos3, pos4).astype(int))

    
    return train_idx, val_idx, test_idx

dataset_name = "BATTERY"
task_size = 50

if dataset_name == "POLLUTION":
    window_size = 5
    hyperparams = [0.1, 2, 3, 100]

elif dataset_name == "HR":
    window_size = 32
    hyperparams =  [0.05, 2, 3, 100]

elif dataset_name == "BATTERY":
    window_size = 20
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

evaluation_list = ["50"]# W-RT, WO-RT, 50

#50% evaluation

if "50" in evaluation_list:
        
    #x_train = np.concatenate([new_train_data.x, test_data_ML.x[:test_size_ML//2].reshape(-1, window_size*dim)], axis=0)
    #y_train = np.concatenate([new_train_data.y, test_data_ML.y[:test_size_ML//2].reshape(-1,1)], axis=0)

    #x_test = test_data_ML.x[test_size_ML//2:].reshape(-1, window_size*dim)
    #y_test = test_data_ML.y[test_size_ML//2:].reshape(-1, 1)

    train_idx, val_idx, test_idx = split_idx_50_50(test_data.file_idx)

    n_domains = np.max(test_data.file_idx)+1
    domain_mae_list = []
    domain_mse_list = []

    for domain in range(n_domains):
        print(domain)
        train_idx_ = np.concatenate([train_idx[domain], val_idx[domain]])

        x_train = np.concatenate([train_data.x.reshape(-1, window_size*dim), test_data.x[train_idx_].reshape(-1, window_size*dim)])
        y_train = np.concatenate([train_data.y.reshape(-1,1), test_data.y[train_idx_].reshape(-1,1)])

        #x_train = test_data.x[train_idx_].reshape(-1, window_size*dim)
        #y_train = test_data.y[train_idx_].reshape(-1,1)

        x_test = test_data.x[test_idx[domain]].reshape(-1, window_size*dim)
        y_test = test_data.y[test_idx[domain]].reshape(-1,1)

        model = xgb.XGBRegressor(learning_rate=hyperparams[0],
                            max_depth=hyperparams[1],
                            min_child_weight=hyperparams[2],
                            n_estimators=hyperparams[3])

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        pd.DataFrame({"TRUTH":list(y_test.reshape(-1)), "PRED_XGBOOST": list(y_pred.reshape(-1))}).to_csv("PRED_50_XGBOOST_D"+str(domain)+"_"+dataset_name+".csv")

        domain_mae_list.append(mae(y_pred, y_test))
        domain_mse_list.append(mse(y_pred, y_test))
        print(mae(y_pred, y_test))

    print("Mae withtout retraining (50pct):", np.mean(domain_mae_list))
    print("mse without retraining (50pct):", np.mean(domain_mse_list))

#evaluation without retraining

if "WO-RT" in evaluation_list:
        
    model = xgb.XGBRegressor(learning_rate=hyperparams[0],
                        max_depth=hyperparams[1],
                        min_child_weight=hyperparams[2],
                        n_estimators=hyperparams[3])

    #model.fit(new_train_data.x, new_train_data.y)
    model.fit(train_data.x.reshape(-1, dim*window_size), train_data.y.reshape(-1, 1))
    #y_pred = model.predict(test_data_ML.x.reshape(-1, dim*window_size))
    y_pred = model.predict(test_data.x.reshape(-1, dim*window_size))

    #mae_wo_retraining = mae(y_pred, test_data_ML.y.reshape(-1, 1))
    #mse_wo_retraining = mse(y_pred, test_data_ML.y.reshape(-1, 1))
    mae_wo_retraining = mae(y_pred, test_data.y.reshape(-1, 1))
    mse_wo_retraining = mse(y_pred, test_data.y.reshape(-1, 1))

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
