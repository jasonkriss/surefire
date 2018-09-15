from torch.nn import Module


class Encoder(Module):
    @property
    def out_features(self):
        return self.num_features()

    def num_features(self):
        raise NotImplementedError
