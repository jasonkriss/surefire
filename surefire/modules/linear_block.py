from torch.nn import Module, Sequential, ReLU, Linear


class LinearBlock(Module):
    def __init__(self, in_features, out_features, activation=ReLU):
        super().__init__()
        self._sequential = Sequential(Linear(in_features, out_features), activation())
        
    def forward(self, x):
        return self._sequential(x)
