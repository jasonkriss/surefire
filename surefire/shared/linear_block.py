from torch.nn import Module, Sequential, Dropout, ReLU, Linear


class LinearBlock(Module):
    def __init__(self, in_features, out_features, dropout=None):
        super().__init__()
        self._sequential = Sequential(Linear(in_features, out_features), ReLU())
        if dropout:
            self._sequential.add_module('dropout', Dropout(dropout))
        
    def forward(self, x):
        return self._sequential(x)
