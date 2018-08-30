from surefire.combiners import Combiner


class IdentityCombiner(Combiner):
    def __init__(self, num_features):
        super().__init__()
        self._num_features = num_features
        
    def forward(self, x):
        return x

    def num_features(self):
        return self._num_features
