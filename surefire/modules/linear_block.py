from torch.nn import Module, Sequential, ReLU, SELU, Linear, BatchNorm1d, LayerNorm, Dropout


class LinearBlock(Module):
    def __init__(self, in_features, out_features, activation='relu', normalization=None, dropout=None):
        super().__init__()
        activation = ReLU if activation == 'relu' else SELU
        self._sequential = Sequential(Linear(in_features, out_features), activation())
        if normalization == 'batch':
            self._sequential.add_module('normalization', BatchNorm1d(out_features))
        elif normalization == 'layer':
            self._sequential.add_module('normalization', LayerNorm(out_features))
        if dropout is not None:
            self._sequential.add_module('dropout', Dropout(dropout))
        
    def forward(self, x):
        return self._sequential(x)
