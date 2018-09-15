from unittest import TestCase

from surefire.combiners import Combiner


class DummyCombiner(Combiner):
    def num_features(self):
        return 42


class TestCombiner(TestCase):
    def test_out_features(self):
        combiner = DummyCombiner()
        self.assertEqual(combiner.out_features, 42)
