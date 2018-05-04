from torch.nn import Module, Sequential, ReLU, SELU, Linear


class LinearBlock(Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super().__init__()
        activation = ReLU if activation == 'relu' else SELU
        self._sequential = Sequential(Linear(in_features, out_features), activation())
        
    def forward(self, x):
        return self._sequential(x)
