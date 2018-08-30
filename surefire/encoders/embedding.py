from torch.nn import Embedding

from surefire.encoders import Encoder


class EmbeddingEncoder(Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._embedding = Embedding(*args, **kwargs)
        
    def forward(self, x):
        return self._embedding(x).squeeze()

    def num_features(self):
        return self._embedding.embedding_dim
