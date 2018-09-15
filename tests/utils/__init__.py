from unittest import TestCase

import torch
from torch.nn.functional import l1_loss

from surefire.utils import multi_task_loss

class TestUtils(TestCase):
    def test_multi_task_loss(self):
        y = {'targeta': torch.tensor([1.0, 2.0])}
        y_pred = {'targeta': torch.tensor([3.0, 4.0])}
        loss_without_weights = multi_task_loss({'targeta': l1_loss})
        self.assertEqual(loss_without_weights(y_pred, y).item(), 2.0)
        loss_with_weights = multi_task_loss({'targeta': l1_loss}, weights={'targeta': 2.0})
        self.assertEqual(loss_with_weights(y_pred, y).item(), 4.0)
