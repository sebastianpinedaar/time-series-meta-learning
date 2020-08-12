from ts_transform import dataset_to_sktime, apply_rocket_kernels
import pickle
import numpy as np
from sktime.transformers.series_as_features.rocket import Rocket
from utils import progressBar
from sklearn.metrics import mean_absolute_error as mae #
from sklearn.metrics import mean_squared_error  as mse #
from sklearn.linear_model import RidgeCV

num_kernels = 10000
dataset_names = [("POLLUTION", 5, 25)]
dataset_name, window_size, task_size = dataset_names[0]

train_data = pickle.load(  open( "../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
validation_data = pickle.load( open( "../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
test_data = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
test_data_ML = pickle.load( open( "../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )

new_train_data = train_data + validation_data

train_data_sktime = dataset_to_sktime(new_train_data)
test_data_sktime = dataset_to_sktime(test_data)

rocket = Rocket(num_kernels=num_kernels)
rocket.fit(train_data_sktime[0])

train_features = apply_rocket_kernels(train_data_sktime, rocket)
del train_data_sktime

pickle_out = open("../Data/NEW-TRAIN-POLLUTION-W5-T25-NOML-ROCKET-10000.pickle", "wb")
pickle.dump(train_features, pickle_out)
pickle_out.close()

test_features = apply_rocket_kernels(test_data_sktime, rocket)
del test_data_sktime
pickle_out = open("../Data/TEST-POLLUTION-W5-T25-NOML-ROCKET-10000.pickle", "wb")
pickle.dump(test_features, pickle_out)
pickle_out.close()

model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(train_features, new_train_data.y)

y_pred = model.predict(test_features)
mae_test = mae(y_pred, test_data.y)
mse_test = mse(y_pred, test_data.y)

print("MAE:", mae_test)
print("MSE:", mse_test)
