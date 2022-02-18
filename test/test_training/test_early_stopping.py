import logging
from itertools import product

import numpy as np
import pytest
import torch
from torch import nn

from neuralpredictors.training import early_stopping

logger = logging.getLogger(__name__)


class CounterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.counter = nn.Parameter(torch.zeros(1))


class CounterClosure:
    def __init__(self, objective_values):
        self.objective_values = objective_values
        self.counter = 0

    def __call__(self, model):
        logger.info("Called")
        ret = self.objective_values[self.counter]
        self.counter += 1
        model.counter.data += 1
        return ret


objective_vals = [5, 4, 0.1, 1, 1, 1, 1, 1, 1, 1]

test_data = [
    (patience, interval, maximize, (np.argmax if maximize else np.argmin)(objective_vals))
    for patience, interval, maximize in product(range(1, 4), range(1, 8), [True, False])
]


class TestEarlyStopping:
    @pytest.mark.parametrize("patience,interval,maximize,expected_epoch", test_data)
    def test_stop_minimize(self, patience, interval, maximize, expected_epoch):
        logger.info(patience, interval, maximize, expected_epoch)
        model = CounterModel()

        closure = CounterClosure(objective_vals)

        for epoch, val in early_stopping(model, closure, maximize=maximize, patience=patience, interval=interval):
            pass
        assert (
            int(model.counter) == expected_epoch + 2
        )  # +2 because the closure is called two more times after convergence
        assert epoch == (expected_epoch + patience) * interval
