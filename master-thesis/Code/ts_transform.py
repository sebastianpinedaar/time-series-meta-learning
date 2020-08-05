import numpy as np

def sliding_window(x, window_size, stride):
    
    windows = []
    for idx in range(0, len(x), stride):
        windows.append(x[np.newaxis, idx:idx+window_size,:])
        #print(windows[0].shape)
    return np.vstack(windows)
