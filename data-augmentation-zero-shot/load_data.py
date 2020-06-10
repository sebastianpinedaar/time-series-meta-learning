import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from scipy import signal
import pickle

def load_M4_train(filename, backcast_length, forecast_length, ts_batch= 1024,  n_tasks=1, out_type = 0):

    #out_type 0 is for metalearning (Reptile), where there is an additional axis for the same time series
    
    x_tl = []

    x = np.array([]).reshape( 0, n_tasks, backcast_length)
    y = np.array([]).reshape( 0, n_tasks, forecast_length)

    headers = True#

    print("Processing ", filename)
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            line = line[1:]
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False
                
    x_tl_tl = np.array(x_tl)
    n_time_series = x_tl_tl.shape[0]

    if(ts_batch > n_time_series or ts_batch == 0):
        ts_batch = n_time_series

    idx = np.arange(x_tl_tl.shape[0])
    np.random.shuffle(idx)
    ts_idx = idx[:ts_batch]

    for i in ts_idx:
        if len(x_tl_tl[i]) < backcast_length + forecast_length:
            continue
        time_series = x_tl_tl[i]
        time_series = np.array([float(s) for s in time_series if s != ''])

        x_temp = np.array([]).reshape(0, backcast_length)
        y_temp = np.array([]).reshape(0, forecast_length)

        for t in range(n_tasks):
            time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
            time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
            j = np.random.randint(backcast_length, time_series.shape[0] + 1 - 2*forecast_length)
            time_series_cleaned_forlearning_x[0, :] = time_series[j - backcast_length: j]
            time_series_cleaned_forlearning_y[0, :] = time_series[j:j + forecast_length]

            x_temp = np.vstack((x_temp, time_series_cleaned_forlearning_x))
            y_temp = np.vstack((y_temp, time_series_cleaned_forlearning_y))

        time_series_cleaned_forlearning_x = x_temp[np.newaxis,...]
        time_series_cleaned_forlearning_y = y_temp[np.newaxis,...]


        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))
        
    x_max = np.max(x)
    x = x/x_max
    y = y/x_max

    if(out_type!=0):
        x = x.reshape((-1, backcast_length))
        y = y.reshape((-1, forecast_length))
    
    return x, y, x_max



def load_M4_val(filename, backcast_length, forecast_length, x_max):

    #out_type 0 is for metalearning (Reptile), where there is an additional axis for the same time series
    
    x_tl = []

    x = np.array([]).reshape( 0, backcast_length)
    y = np.array([]).reshape( 0, forecast_length)

    headers = True#

    print("Processing ", filename)
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            line = line[1:]
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False


    x_tl_tl = np.array(x_tl)

    for i in range(x_tl_tl.shape[0]):
        if len(x_tl_tl[i]) < backcast_length + forecast_length:
            continue
        time_series = x_tl_tl[i]
        time_series = np.array([float(s) for s in time_series if s != ''])

        x_temp = np.array([]).reshape(0, backcast_length)
        y_temp = np.array([]).reshape(0, forecast_length)

        
        time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
        time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
        j =  time_series.shape[0] + 1 - forecast_length #the last time horizon
        time_series_cleaned_forlearning_x[0, :] = time_series[j - backcast_length: j]
        time_series_cleaned_forlearning_y[0, :] = time_series[j:j + forecast_length]

        x_temp = np.vstack((x_temp, time_series_cleaned_forlearning_x))
        y_temp = np.vstack((y_temp, time_series_cleaned_forlearning_y))

        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))
        
    x = x/x_max
    y = y/x_max

    return x, y

def data_proc(data, ts_batch, n_time_series, set_type=0):
    
    if(ts_batch > n_time_series or ts_batch==None):
        ts_batch = n_time_series
    
    idx = np.arange(n_time_series)
    np.random.shuffle(idx)
    ts_idx = idx[:ts_batch]

    x = np.array([]).reshape( 0, n_tasks, backcast_length)
    y = np.array([]).reshape( 0, n_tasks, forecast_length)

    for i in ts_idx:
        time_series = data.iloc[:, i].dropna()
        ts_length = time_series.shape[0]

        if(ts_length==0 or ts_length < backcast_length + 2*forecast_length):
            continue


        x_temp = np.array([]).reshape(0, backcast_length)
        y_temp = np.array([]).reshape(0, forecast_length)
        for t in range(n_tasks):
            time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
            time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
            
            if(set_type == 0):
                j = np.random.randint(backcast_length, ts_length + 1 - 2*forecast_length)
            elif (set_type == 1):
                j = ts_length + 1 - 2*forecast_length
            else:
                j = ts_length + 1 - forecast_length
            
            time_series_cleaned_forlearning_x[0, :] = time_series[j - backcast_length: j]
            time_series_cleaned_forlearning_y[0, :] = time_series[j:j + forecast_length]

            x_temp = np.vstack((x_temp, time_series_cleaned_forlearning_x))
            y_temp = np.vstack((y_temp, time_series_cleaned_forlearning_y))

        time_series_cleaned_forlearning_x = x_temp[np.newaxis,...]
        time_series_cleaned_forlearning_y = y_temp[np.newaxis,...]

        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))

    return x, y


def load_Tourism_train(filenames, backcast_length, forecast_length, ts_batch= 1024,  n_tasks=1, out_type = 0, set_type = 0):

    #set type 0 = training
    #set type 1 = validaiton
    #set type 2 = test

    data0 = pd.read_csv(path+filenames[0])
    data1 = pd.read_csv(path+filenames[1])
    
    
    x0, y0 = data_proc(data0, ts_batch, data0.shape[1])
    x1, y1 = data_proc(data1, ts_batch, data1.shape[1])
    
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([y0, y1], axis=0)

    x_max = np.max(x)
    x = x/x_max
    y = y/x_max

    if(out_type!=0):
        x = x.reshape((-1, backcast_length))
        y = y.reshape((-1, forecast_length))
        
    
    return x, y, x_max


def load_Tourism_val(filenames, backcast_length, forecast_length, x_max, out_type = 1):

    #set type 0 = training
    #set type 1 = validaiton
    #set type 2 = test

    data0 = pd.read_csv(path+filenames[0])
    data1 = pd.read_csv(path+filenames[1])
    
    
    x0, y0 = data_proc(data0, data0.shape[1], data0.shape[1], set_type = 1)
    x1, y1 = data_proc(data1, data0.shape[1], data1.shape[1], set_type = 1)
    
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([y0, y1], axis=0)


    if(out_type!=0):
        x = x.reshape((-1, backcast_length))
        y = y.reshape((-1, forecast_length))
        
    x = x/x_max
    y = y/x_max

    return x, y


def load_Tourism_test(filenames, backcast_length, forecast_length, x_max, out_type = 1):

    #set type 0 = training
    #set type 1 = validaiton
    #set type 2 = test

    data0 = pd.read_csv(path+filenames[0])
    data1 = pd.read_csv(path+filenames[1])
    
    
    x0, y0 = data_proc(data0, data0.shape[1], data0.shape[1], set_type = 2)
    x1, y1 = data_proc(data1, data0.shape[1], data1.shape[1], set_type = 2)
    
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([y0, y1], axis=0)


    if(out_type!=0):
        x = x.reshape((-1, backcast_length))
        y = y.reshape((-1, forecast_length))
        
    x = x/x_max
    y = y/x_max

    return x, y


def load_M3_train (filename, sheet_name, backcast_length, forecast_length, ts_batch= 1024,  n_tasks=1, out_type = 0):
    
    #possible sheet names = 'M3Year', 'M3Quart', 'M3Month', 'M3Other'

    df = pd.read_excel(filename, sheet_name= None)

    data = df[sheet_name].iloc[:,6:].T
    x, y = data_proc(data, ts_batch, data.shape[1])
    x_max = np.max(x)
    x = x/x_max
    y = y/x_max

    if(out_type!=0):
        x = x.reshape((-1, backcast_length))
        y = y.reshape((-1, forecast_length))
    
    return x, y, x_max
    

def load_M3_val (filename, sheet_name, backcast_length, forecast_length, x_max, out_type = 1):
    
    #possible sheet names = 'M3Year', 'M3Quart', 'M3Month', 'M3Other'

    df = pd.read_excel(filename, sheet_name= None)

    data = df[sheet_name].iloc[:,6:].T
    x, y = data_proc(data, data.shape[1], data.shape[1], set_type = 1)
    x = x/x_max
    y = y/x_max

    if(out_type!=0):
        x = x.reshape((-1, backcast_length))
        y = y.reshape((-1, forecast_length))
    
    return x, y

def load_M3_test (filename, sheet_name, backcast_length, forecast_length, x_max, out_type = 1):
    
    #possible sheet names = 'M3Year', 'M3Quart', 'M3Month', 'M3Other'

    df = pd.read_excel(filename, sheet_name= None)

    data = df[sheet_name].iloc[:,6:].T
    x, y = data_proc(data, data.shape[1], data.shape[1], set_type = 2)
    x = x/x_max
    y = y/x_max

    if(out_type!=0):
        x = x.reshape((-1, backcast_length))
        y = y.reshape((-1, forecast_length))
    
    return x, y


def load_M4_meta_datasets(filename, backcast_length, forecast_length, n_tasks=2, pct = (0.6, 0.8, 1.0)):

    #out_type 0 is for metalearning (Reptile), where there is an additional axis for the same time series
    
    x_tl = []

    headers = True#

    print("Processing ", filename)
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            line = line[1:]
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False
                
    x_tl_tl = np.array(x_tl)

    n_time_series = x_tl_tl.shape[0]
    
    idx = np.arange(n_time_series)
    np.random.shuffle(idx)
    
    ts_idx_train = idx[:int(pct[0]*n_time_series)]
    ts_idx_test = idx[int(pct[0]*n_time_series):int(pct[1]*n_time_series)]
    ts_idx_val = idx[int(pct[1]*n_time_series):int(pct[2]*n_time_series)]

    print(x_tl_tl.shape)
    x_meta_train, y_meta_train = split_data(x_tl_tl, ts_idx_train, backcast_length, forecast_length, n_tasks=n_tasks)
    x_meta_test, y_meta_test = split_data(x_tl_tl, ts_idx_test, backcast_length, forecast_length, n_tasks=n_tasks)
    x_meta_val, y_meta_val = split_data(x_tl_tl, ts_idx_val, backcast_length, forecast_length, n_tasks=n_tasks)
   
    x_max = np.max(x_meta_train)
    
    x_meta_train /= x_max
    y_meta_train /= x_max
    x_meta_test /= x_max
    y_meta_test /= x_max
    x_meta_val /= x_max
    y_meta_val /= x_max
    
    
    return x_meta_train, y_meta_train, x_meta_test, y_meta_test, x_meta_val, y_meta_val

def load_M3_meta_datasets (filename, sheet_name, backcast_length, forecast_length, n_tasks=2, pct = (0.6, 0.8, 1.0)):
    
    df = pd.read_excel(filename, sheet_name= None)

    data = df[sheet_name].iloc[:,6:].T

    n_time_series = data.shape[1]
    idx = np.arange(n_time_series)
    np.random.shuffle(idx)
    ts_idx_train = idx[:int(pct[0]*n_time_series)]
    ts_idx_test = idx[int(pct[0]*n_time_series):int(pct[1]*n_time_series)]
    ts_idx_val = idx[int(pct[1]*n_time_series):int(pct[2]*n_time_series)]

    x_tl = []

    for i in range(data.shape[1]):
        x_tl.append(np.array(data[i].dropna()))


    x_meta_train, y_meta_train = split_data(x_tl, ts_idx_train, backcast_length, forecast_length, n_tasks=n_tasks)
    x_meta_test, y_meta_test = split_data(x_tl, ts_idx_test, backcast_length, forecast_length, n_tasks=n_tasks)
    x_meta_val, y_meta_val = split_data(x_tl, ts_idx_val, backcast_length, forecast_length, n_tasks=n_tasks)
    
    print(x_meta_train.shape)
    x_max = np.max(x_meta_train)
    
    x_meta_train /= x_max
    y_meta_train /= x_max
    x_meta_test /= x_max
    y_meta_test /= x_max
    x_meta_val /= x_max
    y_meta_val /= x_max
    
    return x_meta_train, y_meta_train, x_meta_test, y_meta_test, x_meta_val, y_meta_val


def load_Tourism_meta_datasets (path, backcast_length, forecast_length, n_tasks=2, pct = (0.6, 0.8, 1.0)):

    filenames = os.listdir(path)
    data0 = pd.read_csv(path+filenames[0])
    data1 = pd.read_csv(path+filenames[1])
    data = pd.concat([data0, data1])


    n_time_series = data.shape[1]
    idx = np.arange(n_time_series)
    np.random.shuffle(idx)
    ts_idx_train = idx[:int(pct[0]*n_time_series)]
    ts_idx_test = idx[int(pct[0]*n_time_series):int(pct[1]*n_time_series)]
    ts_idx_val = idx[int(pct[1]*n_time_series):int(pct[2]*n_time_series)]

    x_tl = []

    for i in data.columns:
        x_tl.append(np.array(data[i].dropna()))

    x_meta_train, y_meta_train = split_data(x_tl, ts_idx_train, backcast_length, forecast_length, n_tasks=n_tasks)
    x_meta_test, y_meta_test = split_data(x_tl, ts_idx_test, backcast_length, forecast_length, n_tasks=n_tasks)
    x_meta_val, y_meta_val = split_data(x_tl, ts_idx_val, backcast_length, forecast_length, n_tasks=n_tasks)

    x_max = np.max(x_meta_train)

    x_meta_train /= x_max
    y_meta_train /= x_max
    x_meta_test /= x_max
    y_meta_test /= x_max
    x_meta_val /= x_max
    y_meta_val /= x_max
    
    return x_meta_train, y_meta_train, x_meta_test, y_meta_test, x_meta_val, y_meta_val


def split_data_deprecated(x_tl_tl, ts_idx, backcast_length, forecast_length, n_tasks):

    x_meta = np.array([]).reshape( 0, n_tasks, backcast_length)
    y_meta = np.array([]).reshape( 0, n_tasks, forecast_length)

    for i in ts_idx:

        time_series = x_tl_tl[i]
        time_series = np.array([float(s) for s in time_series if s != ''])
        
        ts_length =  time_series.shape[0]

        if ts_length < (backcast_length + forecast_length) or ts_length==0:
            continue
 


        x_temp = np.array([]).reshape(0, backcast_length)
        y_temp = np.array([]).reshape(0, forecast_length)

        for t in range(n_tasks):
            time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
            time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
            j = np.random.randint(backcast_length, ts_length + 1 - forecast_length)
            time_series_cleaned_forlearning_x[0, :] = time_series[j - backcast_length: j]
            time_series_cleaned_forlearning_y[0, :] = time_series[j:j + forecast_length]

            x_temp = np.vstack((x_temp, np.copy(time_series_cleaned_forlearning_x)))
            y_temp = np.vstack((y_temp, np.copy(time_series_cleaned_forlearning_y)))

        time_series_cleaned_forlearning_x = x_temp[np.newaxis,...]
        time_series_cleaned_forlearning_y = y_temp[np.newaxis,...]

        x_meta = np.vstack((x_meta, time_series_cleaned_forlearning_x))
        y_meta = np.vstack((y_meta, time_series_cleaned_forlearning_y))
        
    return x_meta, y_meta


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def batcher(dataset, batch_size, infinite=False):
    while True:
        x, y = dataset
        for x_, y_ in zip(split(x, batch_size), split(y, batch_size)):
            yield x_, y_
        if not infinite:
            break

def split_data(x_tl_tl, ts_idx, backcast_length, forecast_length, n_tasks, transform = 0, *args):
    
    
    if(transform):
        n_tasks_ = n_tasks*6
    else:
        n_tasks_ = n_tasks

    x_meta = np.array([]).reshape( 0, n_tasks_, backcast_length)
    y_meta = np.array([]).reshape( 0, n_tasks_, forecast_length)

    for i in ts_idx:
        

        time_series = x_tl_tl[i]
        time_series = np.array([float(s) for s in time_series if s != ''])

        ts_length =  time_series.shape[0]

        if ts_length < (backcast_length + forecast_length) or ts_length==0:
            continue
        x_temp = np.array([]).reshape(0, backcast_length)
        y_temp = np.array([]).reshape(0, forecast_length)

        for t in range(n_tasks):

            j = np.random.randint(backcast_length, ts_length + 1 - forecast_length)
            
            if(transform):
                std_freq, std_noise, slope_factor = args

                new_ts = transform_time_series(time_series[(j - backcast_length): (j + forecast_length)], std_freq, std_noise, slope_factor)
                time_series_cleaned_forlearning_x = new_ts[:, :backcast_length]
                time_series_cleaned_forlearning_y = new_ts[:, backcast_length:]             
            else:
                time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
                time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
                time_series_cleaned_forlearning_x[0, :] = time_series[j - backcast_length: j]
                time_series_cleaned_forlearning_y[0, :] = time_series[j:j + forecast_length]

            x_temp = np.vstack((x_temp, time_series_cleaned_forlearning_x))
            y_temp = np.vstack((y_temp, time_series_cleaned_forlearning_y))

        time_series_cleaned_forlearning_x = x_temp[np.newaxis,...]
        time_series_cleaned_forlearning_y = y_temp[np.newaxis,...]

        x_meta = np.vstack((x_meta, time_series_cleaned_forlearning_x))
        y_meta = np.vstack((y_meta, time_series_cleaned_forlearning_y))
        
    return x_meta, y_meta
        

def amplitude_phase_peturbation(ts_sample, noise_std):
    
    f_ts_sample = np.fft.fft(ts_sample)
    amplitude = np.abs(f_ts_sample)
    phase = np.angle(f_ts_sample)
    amplitude = np.abs(f_ts_sample)
    phase = np.angle(f_ts_sample)
    gaussian_amp = np.random.normal(0, noise_std, len(ts_sample)).astype(np.float32)
    gaussian_phase = np.random.normal(0, noise_std, len(ts_sample)).astype(np.float32)

    perturbed_amplitude = amplitude*(gaussian_amp+1)
    new_ts_f_ampl = perturbed_amplitude*np.exp(np.array([np.complex(0,i) for i in phase]))
    new_ts_ampl = np.fft.ifft(new_ts_f_ampl)
    
    perturbed_phase = phase*(gaussian_phase+1)
    new_ts_f_phase = amplitude*np.exp(np.array([np.complex(0,i) for i in perturbed_phase]))
    new_ts_phase = np.fft.ifft(new_ts_f_phase)
    
    return new_ts_ampl.real, new_ts_phase.real
    
    
def add_gaussian_noise(ts_sample, noise_std):
  
    gaussian = np.random.normal(0, noise_std, len(ts_sample)).astype(np.float32)
    return ts_sample*(1+gaussian)


def transform_time_series(ts_sample, std_freq, std_noise, slope_factor):
    
    ts_length = len(ts_sample)
    new_ts_ampl, new_ts_phase = amplitude_phase_peturbation(ts_sample, std_freq)
    new_ts_gauss = add_gaussian_noise(ts_sample, std_noise)

    new_ts_f1 = signal.convolve(ts_sample, np.array([0.2,0.8]), "same")
    new_ts_f1[-1] = ts_sample[-1]
    new_ts_f1[0] = ts_sample[0]
    
    new_ts_f2 = signal.convolve(ts_sample, np.array([1/2, 1/2]), "same")
    new_ts_f2[-1] = ts_sample[-1]
    new_ts_f2[0] = ts_sample[0]

    new_ts_f3 = signal.convolve(ts_sample, np.array([1/3, 1/3, 1/3]), "same")
    new_ts_f3[-1] = ts_sample[-1]
    new_ts_f3[0] = ts_sample[0]

    trend_slope = np.polyfit(np.arange(ts_length), ts_sample,1)
    modified_slope = np.arange(0, ts_length)*trend_slope[0]*slope_factor
    new_ts_slope = ts_sample + modified_slope

    return np.vstack([ts_sample, new_ts_ampl, new_ts_gauss, new_ts_phase, new_ts_f1, new_ts_f2, new_ts_f3])


def load_and_transform_M4_meta_datasets(filename, backcast_length, forecast_length, n_tasks=2, pct = (0.6, 0.8, 1.0)):

    #out_type 0 is for metalearning (Reptile), where there is an additional axis for the same time series
    
    x_tl = []

    headers = True#

    print("Processing ", filename)
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            line = line[1:]
            if not headers:
                x_tl.append(line)
            if headers:
                headers = False
                
    x_tl_tl = np.array(x_tl)

    n_time_series = x_tl_tl.shape[0]
    
    idx = np.arange(n_time_series)
    np.random.shuffle(idx)
    
    ts_idx_train = idx[:int(pct[0]*n_time_series)]
    ts_idx_test = idx[int(pct[0]*n_time_series):int(pct[1]*n_time_series)]
    ts_idx_val = idx[int(pct[1]*n_time_series):int(pct[2]*n_time_series)]

    print(x_tl_tl.shape)
    x_meta_train, y_meta_train = split_data(x_tl_tl, ts_idx_train, backcast_length, forecast_length, n_tasks, 1,0.3, 0.05, 0.1)
    x_meta_test, y_meta_test = split_data(x_tl_tl, ts_idx_test, backcast_length, forecast_length, n_tasks)
    x_meta_val, y_meta_val = split_data(x_tl_tl, ts_idx_val, backcast_length, forecast_length, n_tasks)
   
    x_max = np.max(x_meta_train)
    
    x_meta_train /= x_max
    y_meta_train /= x_max
    x_meta_test /= x_max
    y_meta_test /= x_max
    x_meta_val /= x_max
    y_meta_val /= x_max
    
    
    return x_meta_train, y_meta_train, x_meta_test, y_meta_test, x_meta_val, y_meta_val


def load_and_transform_M3_meta_datasets (filename, sheet_name, backcast_length, forecast_length, n_tasks=2, pct = (0.6, 0.8, 1.0)):
    
    df = pd.read_excel(filename, sheet_name= None)

    data = df[sheet_name].iloc[:,6:].T

    n_time_series = data.shape[1]
    idx = np.arange(n_time_series)
    np.random.shuffle(idx)
    ts_idx_train = idx[:int(pct[0]*n_time_series)]
    ts_idx_test = idx[int(pct[0]*n_time_series):int(pct[1]*n_time_series)]
    ts_idx_val = idx[int(pct[1]*n_time_series):int(pct[2]*n_time_series)]

    x_tl = []

    for i in range(data.shape[1]):
        x_tl.append(np.array(data[i].dropna()))


    x_meta_train, y_meta_train = split_data(x_tl, ts_idx_train, backcast_length, forecast_length, n_tasks, 1,0.3, 0.05, 0.1)
    x_meta_test, y_meta_test = split_data(x_tl, ts_idx_test, backcast_length, forecast_length, n_tasks)
    x_meta_val, y_meta_val = split_data(x_tl, ts_idx_val, backcast_length, forecast_length, n_tasks)
    
    x_max = np.max(x_meta_train)
    
    x_meta_train /= x_max
    y_meta_train /= x_max
    x_meta_test /= x_max
    y_meta_test /= x_max
    x_meta_val /= x_max
    y_meta_val /= x_max
    
    return x_meta_train, y_meta_train, x_meta_test, y_meta_test, x_meta_val, y_meta_val


def load_and_transform_Tourism_meta_datasets (path, backcast_length, forecast_length, n_tasks=2, pct = (0.6, 0.8, 1.0)):

    filenames = os.listdir(path)
    data0 = pd.read_csv(path+filenames[0])
    data1 = pd.read_csv(path+filenames[1])
    data = pd.concat([data0, data1])


    n_time_series = data.shape[1]
    idx = np.arange(n_time_series)
    np.random.shuffle(idx)
    ts_idx_train = idx[:int(pct[0]*n_time_series)]
    ts_idx_test = idx[int(pct[0]*n_time_series):int(pct[1]*n_time_series)]
    ts_idx_val = idx[int(pct[1]*n_time_series):int(pct[2]*n_time_series)]

    x_tl = []

    for i in data.columns:
        x_tl.append(np.array(data[i].dropna()))

    x_meta_train, y_meta_train = split_data(x_tl, ts_idx_train, backcast_length, forecast_length, n_tasks, 1,0.3, 0.05, 0.1)
    x_meta_test, y_meta_test = split_data(x_tl, ts_idx_test, backcast_length, forecast_length, n_tasks)
    x_meta_val, y_meta_val = split_data(x_tl, ts_idx_val, backcast_length, forecast_length, n_tasks)

    x_max = np.max(x_meta_train)

    x_meta_train /= x_max
    y_meta_train /= x_max
    x_meta_test /= x_max
    y_meta_test /= x_max
    x_meta_val /= x_max
    y_meta_val /= x_max
    
    return x_meta_train, y_meta_train, x_meta_test, y_meta_test, x_meta_val, y_meta_val


def reload_and_transform_meta_datasets(pickle_filename,  transforms = [], *args):

    #out_type 0 is for metalearning (Reptile), where there is an additional axis for the same time series
    std_freq, std_noise, slope_factor = args
    n_transforms = len(transforms)+1
    pickle_in = open(pickle_filename,"rb")
    data = pickle.load(pickle_in)
    transforms = [0]+transforms
    
    x_meta_train, y_meta_train, x_meta_test, y_meta_test, x_meta_val, y_meta_val = data

    
    if(len(transforms)>0):
        x_meta_train_transformed = np.zeros((x_meta_train.shape[0], x_meta_train.shape[1]*n_transforms , x_meta_train.shape[2]))
        y_meta_train_transformed = np.zeros((y_meta_train.shape[0], y_meta_train.shape[1]*n_transforms , y_meta_train.shape[2]))

        for i in range(x_meta_train.shape[0]):
            for j in range(x_meta_train.shape[1]):
            #for j, tr in enumerate(transforms):
                temp_ts = np.hstack((x_meta_train[i,j,:],y_meta_train[i,j,:]))
                #temp_ts = x_meta_train[i,j,:]
                temp_ts = transform_time_series(temp_ts, std_freq, std_noise, slope_factor)
                
                x_meta_train_transformed[i,n_transforms*j,:] = temp_ts[0,:x_meta_train.shape[2]]
                y_meta_train_transformed[i,n_transforms*j,:] = temp_ts[0,x_meta_train.shape[2]:]

                for k, tr in enumerate(transforms[1:]):
                    x_meta_train_transformed[i,(n_transforms*j)+k+1,:] = temp_ts[tr,:x_meta_train.shape[2]]
                    y_meta_train_transformed[i,(n_transforms*j)+k+1,:] = temp_ts[tr,x_meta_train.shape[2]:]                    

        return x_meta_train_transformed, y_meta_train_transformed, x_meta_test, y_meta_test, x_meta_val, y_meta_val

    else:
        return x_meta_train, y_meta_train, x_meta_test, y_meta_test, x_meta_val, y_meta_val

