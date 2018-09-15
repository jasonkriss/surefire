from surefire.encoders import Encoder


class IdentityEncoder(Encoder):
    def forward(self, x):
        return x

    def num_features(self):
        return 1
