import numpy as np
import pickle
import sys
import os
import keras
import copy
import pandas as pd
from utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from models.deep_learning import resnet
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from keras import regularizers

sys.path.insert(1, "TSRegression")

#dataset_names = [("POLLUTION", 5, 50), ("HR", 32, 50), ("BATTERY", 20, 50)]
#dataset_names = [("HR", 32, 50)]
#dataset_names = [("POLLUTION", 5, 50)]
dataset_names = [("BATTERY", 20, 50)]
output_directory = "output/"
regressor_name = "resnet"
verbose=True
epochs=500
batch_size=64
loss="mean_absolute_error"
metrics = ["mae"]
n_trials = 3
lr =  0.000010
mode = "WRT-HYP"
l2 = 0.0

def freeze_model (model, no_frozen=[-1]):

    for layer in model.model.layers:
        layer.trainable = False

    
    num_layers = len(model.model.layers)

    no_frozen_layers_weights = list()
    for idx in no_frozen:
        model.model.layers[(num_layers+idx)%num_layers].trainable = True
        no_frozen_layers_weights.append(model.model.layers[(num_layers+idx)%num_layers].get_weights())

    return no_frozen_layers_weights



def eval(model, x_test, y_test, mode = "WORT", l2=0.0, lr = 0.005, horizon = 10, epochs = 5, monitor_val = True):

    if mode == "WORT":
        
        y_pred = model.predict(x_test)
        df_metrics = calculate_regression_metrics(y_test, y_pred)


    elif mode == "WRT":

        n_tasks = x_test.shape[0]
        no_frozen_weigths = freeze_model(model, [-1])
        model.epochs = epochs
        min_iter_list = []
        mae_list = []
        mse_list = []
        last_metric_list = []
        original_mae_list = []
        min_metric = []
        #model.lr = 0.000005
        model.batch_size = 50
        model.best_model_file = "temp_resnet_"+str(trial)+"_"+dataset_name+".h5"
        model.model_init_file = "temp_resnet_init_"+str(trial)+"_"+dataset_name+".h5"

        for task_id in range(0, (n_tasks-horizon-1), n_tasks//100):
            
            model.lr = lr#lr*1.1**(task_id/50)
            _ = freeze_model(model, [-1])
            model.model.layers[-1].set_weights(no_frozen_weigths[0])
            model.model.layers[-1].kernel_regularizer = regularizers.l2(l2)
            model.model.compile(loss=model.loss,
                      optimizer=keras.optimizers.Adam(model.lr),
                      metrics=model.metrics)

            task_size = x_test[task_id].shape[0]

            temp_x_train = x_test[task_id][:int(task_size*0.9)]
            temp_y_train = y_test[task_id][:int(task_size*0.9)]
            
            temp_x_val = x_test[task_id][int(task_size*0.9):]
            temp_y_val = y_test[task_id][int(task_size*0.9):]

            temp_x_test = x_test[(task_id+1):(task_id+horizon+1)].reshape(-1, x_test.shape[-2], x_test.shape[-1])
            temp_y_test = y_test[(task_id+1):(task_id+horizon+1)].reshape(-1, 1)
            temp_mae = mae(model.model.predict(temp_x_test), temp_y_test)
            original_mae_list.append(temp_mae)

            print("First mae:", temp_mae)
            hist= model.fit(temp_x_train, temp_y_train, temp_x_val, temp_y_val, monitor_val=True, stopping_patience=5, lr_patience=2)

            model.model = keras.models.load_model(output_directory + model.best_model_file)
            print(mae(model.model.predict(temp_x_test), temp_y_test))
            mae_list.append(mae(model.model.predict(temp_x_test), temp_y_test))
            mse_list.append(mse(model.model.predict(temp_x_test), temp_y_test))
            last_metric_list.append(hist.history["val_loss"][-1])
            min_iter_list.append(np.argmin(temp_mae+hist.history["val_loss"]))
            min_metric.append(np.min([temp_mae]+hist.history["val_loss"]))

        model.model.layers[-1].set_weights(no_frozen_weigths[0])
        df_metrics = pd.DataFrame({"rmse":[0], "mae":[np.median(min_metric)]})

        print("MEAN of Last metric:",  np.mean(last_metric_list))
        print("MAX of min_iter_list:", np.max(min_iter_list))
        print("MEDIAN of min_iter_list:", np.median(min_iter_list))
        print("MEDIAN of min_metric:", np.median(min_metric))
        print("MEAN of original mae:", np.mean(original_mae_list) )
        print("MEAN of MAE", np.mean(mae_list))
        print("MEAN of MSE", np.mean(mse_list))
        print("STD of MAE", np.std(mae_list))

        f=open(model.output_directory+"results.txt", "a+")

        f.write("MEAN of last metric list:  %5.6f \n" %  np.mean(last_metric_list))
        f.write("MAX of min_iter_list: %5.6f \n" % np.max(min_iter_list) )
        f.write("MEDIAN of min_iter_list: %5.6f \n" % np.median(min_iter_list))
        f.write("MEDIAN of min_metric: %5.6f \n" % np.median(min_metric))
        f.write("MEAN mae: %5.6f \n" % np.mean(mae_list))
        f.write("MEAN mse: %5.6f \n" % np.mean(mse_list))
        f.write("MEAN original mae: %5.6f \n" % np.mean(original_mae_list))
        f.write("STD of MAE: %5.6f \n" % np.std(mae_list))
        f.close()

        
        
    elif mode == "WRT-HYP":

        n_tasks = x_test.shape[0]
        no_frozen_weigths = freeze_model(model, [-1])
        model.epochs = epochs
        min_iter_list = []
        mae_list = []
        mse_list = []
        last_metric_list = []
        original_mae_list = []
        min_metric = []
        #model.lr = 0.000005
        model.batch_size = 50
        model.best_model_file = "temp_resnet_"+str(trial)+"_"+dataset_name+".h5"
        model.model_init_file = "temp_resnet_init_"+str(trial)+"_"+dataset_name+".h5"

        for task_id in range(0, (n_tasks-horizon-1), n_tasks//500):
            
            model.lr = lr#lr*1.1**(task_id/50)
            no_frozen_weigths = freeze_model(model, [-1])
            model.model.layers[-1].set_weights(no_frozen_weigths[0])
            model.model.layers[-1].kernel_regularizer = regularizers.l2(0.0001)
            model.model.compile(loss=model.loss,
                      optimizer=keras.optimizers.Adam(model.lr),
                      metrics=model.metrics)

            task_size = x_test[task_id].shape[0]

            temp_x_train = x_test[task_id][:int(task_size*0.8)]
            temp_y_train = y_test[task_id][:int(task_size*0.8)]
            
            temp_x_val = x_test[task_id][int(task_size*0.8):]
            temp_y_val = x_test[task_id][:int(task_size*0.8)]

            temp_x_test = x_test[(task_id+1):(task_id+horizon+1)].reshape(-1, x_test.shape[-2], x_test.shape[-1])
            temp_y_test = y_test[(task_id+1):(task_id+horizon+1)].reshape(-1, 1)
            temp_mae = mae(model.model.predict(temp_x_val), temp_y_val)
            original_mae_list.append(temp_mae)

            print("First mae:", temp_mae)
            hist= model.fit(temp_x_train, temp_y_train, temp_x_val, temp_y_val, monitor_val=True, stopping_patience=20, lr_patience=20)

            mae_list.append(mae(model.model.predict(temp_x_val), temp_y_val))
            mse_list.append(mse(model.model.predict(temp_x_val), temp_y_val))
            last_metric_list.append(hist.history["val_loss"][-1])
            min_iter_list.append(np.argmin(temp_mae+hist.history["val_loss"]))
            min_metric.append(np.min([temp_mae]+hist.history["val_loss"]))

        model.model.layers[-1].set_weights(no_frozen_weigths[0])
        df_metrics = pd.DataFrame({"rmse":[0], "mae":[np.median(min_metric)]})

        print("MEAN of Last metric:",  np.mean(last_metric_list))
        print("MAX of min_iter_list:", np.max(min_iter_list))
        print("MEDIAN of min_iter_list:", np.median(min_iter_list))
        print("MEDIAN of min_metric:", np.median(min_metric))
        print("MEAN of original mae:", np.mean(original_mae_list) )

        f=open(model.output_directory+"results.txt", "a+")

        f.write("MEAN of last metric list:  %5.6f \n" %  np.mean(last_metric_list))
        f.write("MAX of min_iter_list: %5.6f \n" % np.max(min_iter_list) )
        f.write("MEDIAN of min_iter_list: %5.6f \n" % np.median(min_iter_list))
        f.write("MEDIAN of min_metric: %5.6f \n" % np.median(min_metric))
        f.write("MEAN mae: %5.6f \n" % np.mean(mae_list))
        f.write("MEAN mse: %5.6f \n" % np.mean(mse_list))
        f.write("MEAN original mae: %5.6f \n" % np.mean(original_mae_list))
        f.close()

        
        print(df_metrics)

    return df_metrics


for dataset_name, window_size, task_size in dataset_names:



    train_data = pickle.load(  open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    validation_data = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    validation_data_ML = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
    test_data = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
    test_data_ML = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )


    x_train_size = train_data.x.shape[0]
    val_idx = int(x_train_size*0.9)

    x_train = train_data.x[:val_idx]
    y_train = train_data.y[:val_idx]
    x_val = train_data.x[val_idx:]
    y_val = train_data.y[val_idx:]
    x_test = test_data.x
    y_test = test_data.y

    x_val_ml = validation_data_ML.x
    y_val_ml = validation_data_ML.y
    x_test_ml = test_data_ML.x
    y_test_ml = test_data_ML.y 

    #regressor = fit_regressor(output_directory, regressor_name, x_train, y_train, x_val, y_val, itr=itr)
    input_shape = x_train.shape[1:]


    for trial in range(4,7):
    #trial = 8
    #for l2 in [0.001,  0.00001, 0.0001, 0.0]:

        #lr = 1 /(10**lr)

        best_model_file = "resnet_"+str(trial)+"_"+dataset_name+".h5"
        model_init_file = "resnet_init_"+str(trial)+"_"+dataset_name+".h5"
        output_directory = "output/resnet_"+str(trial)+"_"+dataset_name+"_"+mode+"/"

        #try:
        #    os.remove(output_directory+"results.txt")
        #except:
        #    pass       

        try:
            os.mkdir(output_directory)
        except OSError as error: 
            print(error)  

        regressor = resnet.ResNetRegressor(output_directory, 
                                                input_shape, 
                                                verbose, 
                                                epochs, 
                                                batch_size, 
                                                loss, 
                                                metrics, 
                                                lr,
                                                best_model_file,
                                                model_init_file)

        #regressor.fit(x_train, y_train, x_val, y_val, monitor_val=True, stopping_patience=20, lr_patience=10)
        df_metrics = eval(regressor, x_test, y_test, mode="WORT")

        f=open(output_directory+"results.txt", "a+")
        f.write("Test: \n")
        f.write("L2 %5.6f"% l2)
        f.write("\n")
        f.write("RMSE: %5.6f MAE: %5.6f \n" % ( df_metrics["rmse"], df_metrics["mae"]))
        print(df_metrics)
        f.close()

        regressor.model = keras.models.load_model(output_directory + best_model_file)
        df_metrics = eval(regressor, x_test_ml, y_test_ml, mode="WRT",l2 = l2, lr=lr, epochs=20)
        print(df_metrics)

