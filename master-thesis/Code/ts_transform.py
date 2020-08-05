import numpy as np

def sliding_window(x, window_size, stride):
    
    windows = []
    for idx in range(0, len(x), stride):
        windows.append(x[np.newaxis, idx:idx+window_size,:])
        #print(windows[0].shape)
    return np.vstack(windows)


def scale_datasets(standarize = True, normalize = True, *datasets):

    train_dataset, validation_dataset, test_dataset = datasets

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
