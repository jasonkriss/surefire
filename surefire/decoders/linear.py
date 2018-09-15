from torch.nn import Linear

from surefire.decoders import Decoder


class LinearDecoder(Decoder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._linear = Linear(*args, **kwargs)
        
    def forward(self, x):
        return self._linear(x)
