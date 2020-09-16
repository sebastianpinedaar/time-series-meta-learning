import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import sys
from collections import defaultdict, namedtuple
from collections import OrderedDict
from torch.nn import _VF
from torch.nn.utils.rnn import PackedSequence

sys.path.insert(1, "..")

from ts_dataset import TSDataset
from metrics import torch_mae as mae
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F


dataset_name = "HR"
dataset_name = "POLLUTION"
model_name = "LSTM"

task_size = 50
batch_size = 64
output_dim = 1

batch_size = 20
horizon = 10
slow_lr = 10e-5
fast_lr = 10e-4
n_inner_iter = 1
##test

if dataset_name == "HR":
    window_size = 32
    input_dim = 13
elif dataset_name == "POLLUTION":
    window_size = 5
    input_dim = 14
Task = namedtuple('Task', ['x', 'y'])


def to_torch(numpy_tensor):
    
    return torch.tensor(numpy_tensor).float().cuda()

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm




class CustomLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(CustomLSTM, self).__init__( *args, **kwargs)  
        
    def forward(self, input, params = None, hx=None, embeddings = None):  # noqa: F811
        
            if params is None:
                params = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn)] 
                
            
            orig_input = input
            # xxx: isinstance check needs to be in conditional for TorchScript to compile
            if isinstance(orig_input, PackedSequence):
                input, batch_sizes, sorted_indices, unsorted_indices = input
                max_batch_size = batch_sizes[0]
                max_batch_size = int(max_batch_size)
            else:
                batch_sizes = None
                max_batch_size = input.size(0) if self.batch_first else input.size(1)
                sorted_indices = None
                unsorted_indices = None

            if hx is None:
                num_directions = 2 if self.bidirectional else 1
                zeros = torch.zeros(self.num_layers * num_directions,
                                    max_batch_size, self.hidden_size,
                                    dtype=input.dtype, device=input.device)
                hx = (zeros, zeros)
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)

            self.check_forward_args(input, hx, batch_sizes)
            if batch_sizes is None:
                result = _VF.lstm(input, hx, params, self.bias, self.num_layers,
                                  self.dropout, self.training, self.bidirectional, self.batch_first)
            else:
                result = _VF.lstm(input, batch_sizes, hx, params, bias,
                                  self.num_layers, self.dropout, self.training, self.bidirectional)
            output = result[0]
            hidden = result[1:]
            # xxx: isinstance check needs to be in conditional for TorchScript to compile
            if isinstance(orig_input, PackedSequence):
                output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
                return output_packed, self.permute_hidden(hidden, unsorted_indices)
            else:
                return output, self.permute_hidden(hidden, unsorted_indices)



class LSTMModel(nn.Module):
    
    def __init__(self, batch_size, seq_len, input_dim, n_layers, hidden_dim, output_dim, lin_hidden_dim = 100):
        super(LSTMModel, self).__init__()

        #self.lstm = nn.CustomLSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        #self.linear = nn.Linear(hidden_dim, output_dim)#
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_layers = n_layers
        #self.hidden = self.init_hidden()
        self.input_dim = input_dim
        self.features = torch.nn.Sequential(OrderedDict([
            ("lstm",  CustomLSTM(input_dim, hidden_dim, n_layers, batch_first=True)),
            ("linear", nn.Linear(hidden_dim, output_dim))]))
        

    def forward(self, x, params):
        
        if params is None:
            params = OrderedDict(self.named_parameters())


        input = x
        for layer_name, layer in self.features.named_children():

            if layer_name=="lstm":
                #names = ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']
                
                temp_params = []
                for name, _ in layer.named_parameters():
                   temp_params.append(params.get("features."+ layer_name +"."+name))

                output, (hn, cn) = layer(x, temp_params)
                x = hn[-1].view(len(input),-1)
            
            elif layer_name=="linear":
                weight = params.get('features.' + layer_name + '.weight', None)
                bias = params.get('features.' + layer_name + '.bias', None)
                x = F.linear(x, weight = weight, bias = bias)
        

        return x

    @property
    def param_dict(self):
        return OrderedDict(self.named_parameters())

class MetaLearner(object):
    def __init__(self, model, optimizer, fast_lr, loss_func,
                 first_order, num_updates, inner_loop_grad_clip,
                 device):

        self._model = model
        self._fast_lr = fast_lr
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._first_order = first_order
        self._num_updates = num_updates
        self._inner_loop_grad_clip = inner_loop_grad_clip
        self._device = device
        self._grads_mean = []

        self.to(device)


    def update_params(self, loss, params):
        """Apply one step of gradient descent on the loss function `loss`,
        with step-size `self._fast_lr`, and returns the updated parameters.
        """
        create_graph = not self._first_order
        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=create_graph, allow_unused=True)
        for (name, param), grad in zip(params.items(), grads):
            if self._inner_loop_grad_clip > 0 and grad is not None:
                grad = grad.clamp(min=-self._inner_loop_grad_clip,
                                  max=self._inner_loop_grad_clip)
            if grad is not None:
              params[name] = param - self._fast_lr * grad

        return params

    def adapt(self, train_tasks):
        adapted_params = []

        for task in train_tasks:
            params = self._model.param_dict

            for i in range(self._num_updates):
                preds = self._model(task.x, params=params)
                loss = self._loss_func(preds, task.y)
                params = self.update_params(loss, params=params)

            adapted_params.append(params)

        return adapted_params

    def step(self, adapted_params_list, val_tasks,
             is_training):
        
        self._optimizer.zero_grad()
        post_update_losses = []

        for adapted_params, task in zip(
                adapted_params_list, val_tasks):
            preds = self._model(task.x, params=adapted_params)
            loss = self._loss_func(preds, task.y)
            post_update_losses.append(loss)

        mean_loss = torch.mean(torch.stack(post_update_losses))
        if is_training:
            mean_loss.backward()
            self._optimizer.step()


        return mean_loss

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)

    def state_dict(self):
        state = {
            'model_state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict() 
        }

        return state


def test (test_data_ML, model, device):

    total_tasks_test = len(test_data_ML)
    task_size = test_data_ML.x.shape[-3]
    input_dim = test_data_ML.x.shape[-1]
    window_size = test_data_ML.x.shape[-2]
    output_dim = test_data_ML.y.shape[-1]

    accum_error = 0.0
    count = 0

    for task in range(0, (total_tasks_test-horizon-1), total_tasks_test//100):

        x_spt, y_spt = test_data_ML[task]
        x_qry = test_data_ML.x[(task+1):(task+1+horizon)].reshape(-1, window_size, input_dim)
        y_qry = test_data_ML.y[(task+1):(task+1+horizon)].reshape(-1, output_dim)
        
        x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
        x_qry = to_torch(x_qry)
        y_qry = to_torch(y_qry)

        train_task = [Task(x_spt, y_spt)]

        adapted_params = meta_learner.adapt(train_task)
        y_pred = model(x_qry, adapted_params[0])

        error = mae(y_pred, y_qry)

        count += 1
        accum_error += error.cpu().data

    return accum_error/count


train_data = pickle.load(  open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
train_data_ML = pickle.load( open( "../../Data/TRAIN-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
validation_data = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
validation_data_ML = pickle.load( open( "../../Data/VAL-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )
test_data = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-NOML.pickle", "rb" ) )
test_data_ML = pickle.load( open( "../../Data/TEST-"+dataset_name+"-W"+str(window_size)+"-T"+str(task_size)+"-ML.pickle", "rb" ) )

model = LSTMModel( batch_size=batch_size, seq_len = window_size, input_dim = input_dim, n_layers = 2, hidden_dim = 120, output_dim =1)


torch.backends.cudnn.enabled = False
optimizer = torch.optim.Adam(model.parameters(), lr = slow_lr)
loss_func = mae
first_order = False
num_updates = 5
epochs = 5
inner_loop_grad_clip = 20
batch_size = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

meta_learner = MetaLearner(model, optimizer, fast_lr , loss_func,
                 first_order, num_updates, inner_loop_grad_clip,
                 device)

total_tasks, task_size, window_size, input_dim = test_data_ML.x.shape
n_batches = train_data_ML.x.shape[0] // batch_size


for epoch in range(epochs):

    
    #train
    batch_idx = np.random.randint(0,n_batches)
    x_spt, y_spt = train_data_ML[batch_idx:batch_size+batch_idx]
    x_qry, y_qry = train_data_ML[batch_idx+1 : batch_idx+batch_size+1]
    
    x_spt, y_spt = to_torch(x_spt), to_torch(y_spt)
    x_qry = to_torch(x_qry)
    y_qry = to_torch(y_qry)

    train_tasks = [Task(x_spt[i], y_spt[i]) for i in range(x_spt.shape[0])]
    val_tasks = [Task(x_qry[i], y_qry[i]) for i in range(x_qry.shape[0])]

    adapted_params = meta_learner.adapt(train_tasks)
    mean_loss = meta_learner.step(adapted_params, val_tasks, is_training = True)
    print(mean_loss)

    #test
    val_error = test(validation_data_ML, model, device)
    print(val_error)


