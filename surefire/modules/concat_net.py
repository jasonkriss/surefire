import torch
from torch.nn import Module, ReLU, ModuleList, Linear

from surefire.modules import Combine, LinearBlock
from surefire.utils import init_all_weights_


class ConcatNet(Module):
    def __init__(self, features, out_features, layers=[], activation='relu', normalization=None, dropout=None):
        super().__init__()
        self._combine = Combine(features)
        self._blocks = ModuleList()
        num_in = self._combine.out_features
        in_features = num_in
        for num_out in layers:
            self._blocks.append(LinearBlock(num_in, num_out, activation, normalization=normalization, dropout=dropout))
            in_features += num_out
            num_in = num_out
        self._final = Linear(in_features, out_features)
        init_all_weights_(self, activation)
        
    def forward(self, x):
        features = [self._combine(x)]
        for block in self._blocks:
            features.append(block(features[-1]))
        return self._final(torch.cat(features, dim=1))
