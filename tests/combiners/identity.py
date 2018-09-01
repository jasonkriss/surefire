from unittest import TestCase

from surefire.combiners import IdentityCombiner


class TestIdentityCombiner(TestCase):
    def test_out_features(self):
        combiner = IdentityCombiner(8)
        self.assertEqual(combiner.out_features, 8)
