from torch.nn import Module, Sequential, Dropout, ReLU, Linear

from surefire.combiners import Combiner


class _LinearBlock(Module):
    def __init__(self, in_features, out_features, dropout=None):
        super().__init__()
        self._sequential = Sequential(Linear(in_features, out_features), ReLU())
        if dropout:
            self._sequential.add_module('dropout', Dropout(dropout))
        
    def forward(self, x):
        return self._sequential(x)


class FeedforwardCombiner(Combiner):
    def __init__(self, layers, dropout=None):
        super().__init__()
        self._layers = layers
        self._sequential = Sequential()
        in_features = layers[0]
        for idx, out_features in enumerate(layers):
            self._sequential.add_module(str(idx), _LinearBlock(in_features, out_features, dropout=dropout))
            in_features = out_features
        
    def forward(self, x):
        return self._sequential(x)

    def num_features(self):
        return self._layers[-1]
