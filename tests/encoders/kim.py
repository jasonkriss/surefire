from unittest import TestCase

import torch

from surefire.encoders import KimEncoder


class TestKimEncoder(TestCase):
    def test_out_features(self):
        encoder = KimEncoder(8, 4, num_features=32, kernel_sizes=[3,4])
        self.assertEqual(encoder.out_features, 64)
