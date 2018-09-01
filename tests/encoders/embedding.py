from unittest import TestCase

import torch

from surefire.encoders import EmbeddingEncoder


class TestEmbeddingEncoder(TestCase):
    def test_out_features(self):
        encoder = EmbeddingEncoder(8, 4)
        self.assertEqual(encoder.out_features, 4)

    def test_forward(self):
        encoder = EmbeddingEncoder(8, 4)
        result = encoder(torch.tensor([[1], [2]]))
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 4)
