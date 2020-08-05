import os
import numpy as np
import pandas as pd
import pickle
from ts_dataset import TSDataset
from ts_transform import sliding_window

def preprocess_pollution(data, var):

    cbwd_dict = {"NE":1, "SE":2, "SW":3, "NW":4, "cv":0}

    #imputing target
    data[("PM_US Post")].loc[np.isnan(data["PM_US Post"])] = 0

    #imputing numerical vars
    for v in var:    
        try:
            data[v] = data[v].interpolate()
        except:
            pass

    #new varibale for calm and variable wind
    data["cv_wd"] = (data["cbwd"]=="cv").astype(int)

    #inputting wind directioin
    data["cbwd"] = data["cbwd"].replace(cbwd_dict)
    
    data["na_cbwd"] = np.isnan(data["cbwd"]).astype(int)
    data[("cbwd")].loc[np.isnan(data["cbwd"])] = -1
    
    data["na_precipitation"] = np.isnan(data["precipitation"]).astype(int)
    data[("precipitation")].loc[np.isnan(data["precipitation"])] = -1
    
    data["na_Iprec"] = np.isnan(data["Iprec"]).astype(int)
    data[("Iprec")].loc[np.isnan(data["Iprec"])] = -1
   
    #var = var[:-1] + ["na_cbwd", "na_precipitation", "na_Iprec", "cv_wd"] +[var[-1]]
    var = var[:-1] + ["na_cbwd", "cv_wd"] +[var[-1]]

    return data[var]

def load_data_pollution(window_size, task_size, stride=1, mode="meta-learning", standarize=True, normalize=True):


    path = "C:/Users/Sebastian/Documents/Data Analytics Master/Semester4-Thesis/Datasets/FiveCitiePMData/"
    var = ['month', 'day', 'hour', 'season', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'cbwd', 'Iws', 'precipitation', 'Iprec', 'PM_US Post']


    files = os.listdir(path)
    files = [file for file in filter(lambda x: x[-4:] ==".csv", files)]

    split_info = pd.read_excel("../Data/Data-set-split.ods", engine="odf", sheet_name="Pollution")
    split_info = split_info.iloc[:5, [0,2]]
    split_info["City_abb"] = [city[:3] for city in split_info.City]
    split_info = split_info.set_index("City_abb")

    training_files = []
    validation_files = []
    test_files = []

    for file_name in files:

        meta_set = split_info.loc[file_name[:3], "Meta-set"]

        if meta_set == "Meta-train":
            training_files.append(file_name)
        elif meta_set == "Meta-val":
            validation_files.append(file_name)
        else:
            test_files.append(file_name)

    #training dataset
    data = pd.read_csv(path+training_files[0])
    data = np.array(preprocess_pollution(data, var))
    train_dataset = TSDataset(data, window_size, task_size, stride, mode)
    
    for file_name in training_files[1:]:

        data = pd.read_csv(path+file_name)       
        data = np.array(preprocess_pollution(data, var))
        train_dataset += TSDataset(data, window_size, task_size, stride, mode)

    #validation dataset
    data = pd.read_csv(path+validation_files[0])
    data = np.array(preprocess_pollution(data, var))
    validation_dataset = TSDataset(data, window_size, task_size, stride, mode)
    
    for file_name in validation_files[1:]:

        data = pd.read_csv(path+file_name)       
        data = np.array(preprocess_pollution(data, var))
        validation_dataset += TSDataset(data, window_size, task_size, stride, mode)    
    
    #test dataset
    data = pd.read_csv(path+test_files[0])
    data = np.array(preprocess_pollution(data, var))
    test_dataset = TSDataset(data, window_size, task_size, stride, mode)
    
    for file_name in test_files[1:]:

        data = pd.read_csv(path+file_name)       
        data = np.array(preprocess_pollution(data, var))
        test_dataset += TSDataset(data, window_size, task_size, stride, mode)  


    #scaling

    if normalize:
        train_dataset.compute_min_max_params()
        max_list, min_list = train_dataset.get_min_max_params()
        validation_dataset.set_min_max_params(max_list, min_list)
        test_dataset.set_min_max_params(max_list, min_list)

        train_dataset.normalize()
        validation_dataset.normalize()
        test_dataset.normalize()
        
    if standarize:
        train_dataset.compute_standard_params()
        mean, std = train_dataset.get_standard_params()
        validation_dataset.set_standard_params(mean, std)
        test_dataset.set_standard_params(mean, std)

        train_dataset.standarize()
        validation_dataset.standarize()
        test_dataset.standarize()

    return train_dataset, validation_dataset, test_dataset


###heart rate data
def create_signals(data, signals, names, min_sampling, freq_ECG, window_size_secs, step_size_secs):
    
    #cleaning signals
    signals_sep = [sliding_window(data["signal"][source][signal_name], freq//min_sampling, freq//min_sampling) for source, signal_name, freq, _ in signals]
    signals_agg = [np.mean(signal, axis=1) for signal in signals_sep]
    signals_agg = np.concatenate(signals_agg, axis=1)
    
    #couting peaks for heart rate computation
    step = (freq_ECG)*step_size_secs
    peaks = data["rpeaks"]
    peaks_df = pd.DataFrame({"Peak": peaks})
    peaks_df["Lag"] = peaks_df["Peak"].shift()
    peaks_df["Diff"] = peaks_df["Peak"]- peaks_df["Lag"]
    peaks_on_step = peaks//step
    no_peaks_on_step = set(list(np.arange(np.max(peaks_on_step)))).difference(set(list(peaks_on_step)))

    peaks_count = pd.DataFrame({"Peak_group": peaks_on_step, "Count": np.ones(len(peaks_on_step)), "Diff": peaks_df["Diff"]}).groupby("Peak_group").sum().reset_index()

    no_peaks_count = pd.DataFrame({"Peak_group": list(no_peaks_on_step), "Count": np.zeros(len(no_peaks_on_step)), "Diff": np.zeros(len(no_peaks_on_step))})
    peaks_count = pd.concat([peaks_count, no_peaks_count]).sort_values(by=["Peak_group"]).reset_index()
 

    #computing heart rate ground_truth
    step = int(window_size_secs/step_size_secs)
    heart_rate_count = []
    heart_rate_diff = []
    heart_rate_count.append(np.sum(peaks_count["Count"][:step])-1)
    heart_rate_diff.append(np.sum(peaks_count["Diff"][:step]))


    for idx in range(1,peaks_count.shape[0]-step):
        heart_rate_count.append(heart_rate_count[-1]-peaks_count["Count"][idx]+peaks_count["Count"][idx+step])
        heart_rate_diff.append(heart_rate_diff[-1]-peaks_count["Diff"][idx]+peaks_count["Diff"][idx+step])
    
    heart_rate = np.array(heart_rate_count)/np.array(heart_rate_diff)*(700)*60
    allowed_size = heart_rate.shape[0]
    
    data_matrix = pd.DataFrame(signals_agg[:allowed_size, :])
    data_matrix.columns = names

    data_matrix["HR"] = heart_rate
    data_matrix["Act"] = data["activity"][:allowed_size]
    
    
    return data_matrix


def get_signals_names (signals):
    
    names = []
    for source, signal, freq, n_channels in signals:

        names.append(source+"-"+signal)
        if (n_channels>1):
            names+=[source+"-"+signal+str(i) for i in range(1, n_channels)]
            
    return names

def preprocess_heart_rate (path, folders, names):
    
    data_signals_list = []
    
    for folder in folders:

        print("Processing subject ", folder, "...")
        
        file_pkl = path+folder+"/"+folder+".pkl"
        file_csv = path+folder+"/"+folder+"_quest.csv"

        with open(file_pkl, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()

        data_signals = create_signals(data, signals, names, min_sampling, freq_ECG, window_size_secs, step_size_secs)
        subject_info = pd.read_csv(file_csv)
        data_signals["AGE"] = float(subject_info.iloc[0,1])
        data_signals ["GENDER"] = 1.0 if subject_info.iloc[1,1]=="m" else 0.0
        data_signals ["HEIGHT"] = float(subject_info.iloc[2,1])
        data_signals ["WEIGHT"] = float(subject_info.iloc[3,1])
        data_signals ["SKIN"] = float(subject_info.iloc[4,1])
        data_signals ["SPORT"] = float(subject_info.iloc[5,1])
        data_signals_list.append(data_signals)

    data_merged = pd.concat(data_signals_list)
    
    print("Processed data with shape:", data_merged.shape)
    
    return data_merged


def load_data_heart_rate(min_sampling = 4, freq_ECG = 700, window_size_secs = 8, step_size_secs = 0.25):

    
    signals = [("chest","ACC", 700, 3), ("chest","ECG", 700, 1), ("chest", "EMG", 700, 1), ("chest", "EDA", 700, 1), 
                ("chest", "Temp", 700,1), ("chest","Resp", 700, 1), ("wrist", "ACC", 32, 3), ("wrist", "BVP", 64, 1), 
                ("wrist", "EDA", 4, 1), ("wrist", "TEMP", 4, 1)]
    path  = "C:/Users/Sebastian/Documents/Data Analytics Master/Semester4-Thesis/Datasets/PPG_FieldStudy/"
    
    folders = [i for i in os.listdir(path) if i[0]=="S"]
    names = get_signals_names(signals)

    #var = ['chest-ACC', 'chest-ACC1', 'chest-ACC2', 'chest-EMG', 'chest-ECG',
    #   'chest-EDA', 'chest-Temp', 'chest-Resp', 'wrist-ACC', 'wrist-ACC1',
    #   'wrist-ACC2', 'wrist-BVP', 'wrist-EDA', 'wrist-TEMP', 'Act', 
    #   'AGE', 'GENDER', 'HEIGHT', 'WEIGHT', 'SKIN', 'SPORT', 'HR']

    var = ['chest-ACC', 'chest-ACC1', 'chest-ACC2', 'chest-EMG', 'chest-ECG',
       'chest-EDA', 'chest-Temp', 'chest-Resp', 'wrist-ACC', 'wrist-ACC1',
       'wrist-ACC2', 'wrist-BVP', 'wrist-EDA', 'wrist-TEMP', 'HR']

    split_info = pd.read_excel("../Data/Data-set-split.ods", engine="odf", sheet_name="Heart-Rate")
    split_info = split_info.iloc[:15, [0,2]]
    split_info = split_info.set_index("Subject")

    training_folders = []
    validation_folders = []
    test_folders = []

    for folder in folders:

        meta_set = split_info.loc[folder, "Meta-set"]

        if meta_set == "Meta-train":
            training_folders.append(folder)
        elif meta_set == "Meta-val":
            validation_folders.append(folder)
        else:
            test_folders.append(folder)
    
    #training dataset

    data = np.array(get_heart_rate_data([[0]], names)[var])

    print(split_info)

if __name__ == "__main__":

    #data1, data2, data3 = load_data_pollution(100,100,1,"meta-learning")

    min_sampling = 4 # Hz
    freq_ECG = 700
    window_size_secs = 8
    step_size_secs = 0.25
    load_data_heart_rate()

    print(data1.x.shape, data2.x.shape, data3.x.shape)