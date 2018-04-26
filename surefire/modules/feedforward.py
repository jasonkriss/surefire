from torch.nn import Module, Sequential, Linear, ReLU

from surefire.modules import Combine, LinearBlock


class Feedforward(Module):
    def __init__(self, features, out_features, layers=[], activation=ReLU):
        super().__init__()
        self._combine = Combine(features)
        self._sequential = Sequential()
        num_in = self._combine.out_features
        for num_out in layers:
            self._sequential.append(LinearBlock(num_in, num_out, activation))
            num_in = num_out
        self._sequential.append(Linear(num_in, out_features))
        
    def forward(self, x):
        return self._sequential(self._combine(x))
