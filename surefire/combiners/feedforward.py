from torch.nn import Sequential

from surefire.combiners import Combiner
from surefire.shared import LinearBlock


class FeedforwardCombiner(Combiner):
    def __init__(self, layers, dropout=None):
        super().__init__()
        self._layers = layers
        self._sequential = Sequential()
        in_features = layers[0]
        for idx, out_features in enumerate(layers[1:]):
            self._sequential.add_module(str(idx), LinearBlock(in_features, out_features, dropout=dropout))
            in_features = out_features
        
    def forward(self, x):
        return self._sequential(x)

    def num_features(self):
        return self._layers[-1]
