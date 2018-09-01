from unittest import TestCase

from surefire.combiners import FeedforwardCombiner


class TestFeedforwardCombiner(TestCase):
    def test_out_features(self):
        combiner = FeedforwardCombiner([8, 4, 2])
        self.assertEqual(combiner.out_features, 2)
