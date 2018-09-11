from unittest import TestCase

from surefire.data import Dataset


_RECORDS = [{
    'featurea': 1,
    'featureb': 2,
    'targeta': 3,
    'targetb': 4
}, {
    'featurea': 5,
    'featureb': 6,
    'targeta': 7,
    'targetb': 8
}]


class TestDataset(TestCase):
    def test_len(self):
        dataset = Dataset(_RECORDS)
        self.assertEqual(len(dataset), 2)

    def test_without_transforms(self):
        dataset = Dataset(_RECORDS)
        features, targets = dataset[0]
        self.assertEqual(features['featurea'], 1)
        self.assertEqual(targets['targeta'], 3)

    def test_with_transforms(self):
        dataset = Dataset(_RECORDS, transforms={
            'featurea': lambda v: v + 1,
            'targeta': lambda v: v + 2
        })
        features, targets = dataset[0]
        self.assertEqual(features['featurea'], 2)
        self.assertEqual(targets['targeta'], 5)
