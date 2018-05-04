import math

from torch.nn import Linear
from torch.nn.init import kaiming_uniform_


def init_all_weights_(module, activation):
    def init_weights_(m):
        if isinstance(m, Linear):
            if activation == 'relu':
                kaiming_uniform_(m.weight.data, nonlinearity='relu')
            else:
                m.weight.data.normal_(0, 1 / math.sqrt(m.weight.data.size(1)))
            m.bias.data.zero_()
    module.apply(init_weights_)
