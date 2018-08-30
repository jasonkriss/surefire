from torch.nn import BatchNorm1d

from surefire.encoders import Encoder


class BatchNormEncoder(Encoder):
    def __init__(self, **kwargs):
        super().__init__()
        self._batch_norm = BatchNorm1d(1, **kwargs)
        
    def forward(self, x):
        return self._batch_norm(x)

    def num_features(self):
        return 1
