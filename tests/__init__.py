from unittest import TestCase

import torch
from surefire import EncoderDict, DecoderDict, ECD
from surefire.encoders import IdentityEncoder
from surefire.combiners import IdentityCombiner
from surefire.decoders import IdentityDecoder


class TestEncoderDict(TestCase):
    def test_out_features(self):
        encoders = EncoderDict({'a': IdentityEncoder(), 'b': IdentityEncoder()})
        self.assertEqual(encoders.out_features, 2)


class TestECD(TestCase):
    def test_forward(self):
        encoders = EncoderDict({'a': IdentityEncoder(), 'b': IdentityEncoder()})
        combiner = IdentityCombiner(encoders.out_features)
        decoders = DecoderDict({'output': IdentityDecoder()})
        ecd = ECD(encoders, combiner, decoders)
        result = ecd({'a': torch.tensor([[1.], [2.]]), 'b': torch.tensor([[3.], [4.]])})
        self.assertTrue(torch.equal(result['output'], torch.tensor([[1., 3.], [2., 4.]])))
