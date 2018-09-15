from unittest import TestCase

from surefire.combiners import CatCombiner


class TestCatCombiner(TestCase):
    def test_out_features(self):
        combiner = CatCombiner([8, 4, 2])
        self.assertEqual(combiner.out_features, 14)
