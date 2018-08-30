import torch
from torch.nn import Module, ModuleDict


class ECD(Module):
    def __init__(self, encoders, combiner, decoders):
        super().__init__()
        self._encoders = ModuleDict(encoders)
        self._combiner = combiner
        self._decoders = ModuleDict(decoders)

    def forward(self, x):
        encodings = [self._encoders[key](value) for key, value in x.items()]
        features = self._combiner(torch.cat(encodings, dim=1))
        return {key: decoder(features) for key, decoder in self._decoders}
