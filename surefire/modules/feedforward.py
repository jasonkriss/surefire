from torch.nn import Module, Sequential, Linear, ReLU

from surefire.modules import Combine, LinearBlock
from surefire.utils import init_all_weights_


class Feedforward(Module):
    def __init__(self, features, out_features, layers=[], activation='relu', normalization=None, dropout=None):
        super().__init__()
        self._combine = Combine(features)
        self._sequential = Sequential()
        num_in = self._combine.out_features
        for idx, num_out in enumerate(layers):
            self._sequential.add_module(str(idx), LinearBlock(num_in, num_out, activation, normalization=normalization, dropout=dropout))
            num_in = num_out
        self._sequential.add_module('final', Linear(num_in, out_features))
        init_all_weights_(self, activation)
        
    def forward(self, x):
        return self._sequential(self._combine(x))
