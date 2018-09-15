from surefire.decoders import Decoder


class IdentityDecoder(Decoder):
    def forward(self, x):
        return x
