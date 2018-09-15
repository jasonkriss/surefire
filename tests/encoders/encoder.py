from unittest import TestCase

from surefire.encoders import Encoder


class DummyEncoder(Encoder):
    def num_features(self):
        return 42


class TestEncoder(TestCase):
    def test_out_features(self):
        encoder = DummyEncoder()
        self.assertEqual(encoder.out_features, 42)
