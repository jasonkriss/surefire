import torch
from torch.nn import ModuleList

from surefire.combiners import Combiner
from surefire.shared import LinearBlock


class CatCombiner(Combiner):
    def __init__(self, layers, dropout=None):
        super().__init__()
        self._layers = layers
        self._blocks = ModuleList()
        in_features = layers[0]
        for out_features in layers[1:]:
            self._blocks.append(LinearBlock(in_features, out_features, dropout=dropout))
            in_features = out_features
        
    def forward(self, x):
        features = [x]
        for block in self._blocks:
            features.append(block(features[-1]))
        return torch.cat(features, dim=1)

    def num_features(self):
        return sum(self._layers)
