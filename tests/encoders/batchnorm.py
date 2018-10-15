from unittest import TestCase

import torch

from surefire.encoders import BatchNormEncoder


class TestBatchNormEncoder(TestCase):
    def test_forward_with_1d(self):
        encoder = BatchNormEncoder()
        result = encoder(torch.tensor([1., 2., 3.]))
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 1)

    def test_forward_with_2d(self):
        encoder = BatchNormEncoder()
        result = encoder(torch.tensor([[1.], [2.], [3.]]))
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 1)
