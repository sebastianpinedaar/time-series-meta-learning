import xgboost as xgb
import pickle
import random
import sys
from ts_dataset import TSDataset
from sklearn.metrics import mean_absolute_error as mae #
import pandas as pd

#dataset_name = "POLLUTION"
#window_size = 5
#task_size = 25

#dataset_names = [("BATTERY", 20, 25),  ("HR", 32, 25)]
dataset_names = [("POLLUTION", 5, 25)]


hyperparams_grid = {"learning_rate"    : [0.05, 0.1, 0.3 ] ,
                "max_depth"        : [ 2, 7, 10],
                "min_child_weight" : [ 1, 3, 7 ],
                "n_estimators": [10, 100, 1000]}

for dataset_name, window_size, task_size in dataset_names:

    train_data = pickle.load(  open( "../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    validation_data = pickle.load( open( "../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    test_data = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )


    dim = train_data.dim*window_size

    train_data.x = train_data.x.reshape(-1, dim)
    validation_data.x = validation_data.x.reshape(-1, dim)


    max_hyperparam_iter = 20
    hyperparam_history = []
    mae_history = []

    for i in range(max_hyperparam_iter):

        print("Fitting model ", i , " ...")

        random_hyperp_idx = [random.randint(0,2), random.randint(0,2), random.randint(0,2), random.randint(0,2)]
        random_hyper = [hyperparams_grid[value][random_hyperp_idx[idx]] for idx, value in enumerate(hyperparams_grid.keys())]

        print("Hyperparams:", random_hyper)

        model = xgb.XGBRegressor(learning_rate=random_hyper[0],
                                max_depth=random_hyper[1],
                                min_child_weight=random_hyper[2],
                                n_estimators=random_hyper[3])

        model.fit(train_data.x, train_data.y)
        y_pred = model.predict(validation_data.x)
        mae_val = mae(y_pred, validation_data.y)

        print("MSE:", mae_val, " with hyper:", random_hyper)

        hyperparam_history.append(random_hyper)
        mae_history.append(mae_val)

    pd.DataFrame({"Hyperparams": hyperparam_history, "MSE": mae_history}).to_csv("../Data/results/"+dataset_name+"_xgboost.csv")