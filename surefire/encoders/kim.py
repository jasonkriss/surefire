import torch
from torch.nn import Embedding, ModuleList, Conv1d
from torch.nn.functional import relu, max_pool1d

from surefire.encoders import Encoder


def _max_over_time_pool(input):
    return max_pool1d(input, input.shape[-1]).squeeze()


class KimEncoder(Encoder):
    def __init__(self, num_embeddings, embedding_dim, num_features=100, kernel_sizes=[3, 4, 5]):
        super().__init__()
        self._embedding = Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self._convs = ModuleList([Conv1d(embedding_dim, num_features, kernel_size) for kernel_size in kernel_sizes])
    
    def forward(self, x):
        embedded = torch.transpose(self._embedding(x), 1, 2)
        features = torch.cat([_max_over_time_pool(conv(embedded)) for conv in self._convs], dim=1)
        return relu(features)

    def num_features(self):
        return sum((conv.out_channels for conv in self._convs))
