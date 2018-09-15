import torch
from torch.nn import Module, ModuleDict


class EncoderDict(ModuleDict):
    @property
    def out_features(self):
        return sum((encoder.out_features for encoder in self.values()))


class DecoderDict(ModuleDict):
    pass


class ECD(Module):
    def __init__(self, encoders, combiner, decoders):
        super().__init__()
        self._encoders = encoders
        self._combiner = combiner
        self._decoders = decoders

    def forward(self, x):
        encodings = [encoder(x[key]) for key, encoder in self._encoders.items()]
        features = self._combiner(torch.cat(encodings, dim=1))
        return {key: decoder(features) for key, decoder in self._decoders.items()}
