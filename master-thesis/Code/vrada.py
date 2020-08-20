import sys
sys.path.insert(1, "DANN_py3")
from functions import ReverseLayerF
import torch.nn as nn
from VariationalRecurrentNeuralNetwork.model import VRNN

class VRADA(nn.Module):
    
    def __init__(self, x_dim, h_dim, h_dim_reg, z_dim, out_dim, n_domains, n_layers, device, bias=False):
        super().__init__()
        
        self.device = device
        self.vrnn = VRNN(x_dim, h_dim, z_dim, n_layers, device)
        self.linear = nn.Linear (h_dim, out_dim)
        
        self.regressor = nn.Sequential()
        self.regressor.add_module('c_fc1', nn.Linear(h_dim, h_dim_reg))
        self.regressor.add_module('c_bn1', nn.BatchNorm1d(h_dim_reg))
        self.regressor.add_module('c_relu1', nn.ReLU(True))
        self.regressor.add_module('c_drop1', nn.Dropout2d())
        self.regressor.add_module('c_fc2', nn.Linear(h_dim_reg, out_dim))
        
        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('c_fc1', nn.Linear(h_dim, h_dim_reg))
        self.domain_classifier.add_module('c_bn1', nn.BatchNorm1d(h_dim_reg))
        self.domain_classifier.add_module('c_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('c_drop1', nn.Dropout2d())
        self.domain_classifier.add_module('c_fc2', nn.Linear(h_dim_reg, n_domains))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        
    def forward(self, x, alpha):
        
        x = x.to(self.device)
        x = x.squeeze().transpose(0, 1)
        kld_loss, nll_loss, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std), x1, h = self.vrnn(x)
        reverse_feature = ReverseLayerF.apply(h.squeeze(), alpha)
        regressor_output = self.regressor(reverse_feature)
        domain_class_output = self.domain_classifier(reverse_feature)
        
        return regressor_output, domain_class_output, kld_loss, nll_loss
        
    def cuda(self):
        
        self.vrnn.cuda()
        
        super().cuda()