from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from itertools import cycle
import numpy as np
import time


def copy_state(model):
    """
    Given PyTorch module `model`, makes a copy of the state onto CPU.
    Args:
        model: PyTorch module to copy state dict of

    Returns:
        A copy of state dict with all tensors allocated on the CPU
    """
    copy_dict = OrderedDict()
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        copy_dict[k] = v.cpu() if v.is_cuda else v.clone()

    return copy_dict

class Tracker:

    def log_objective(self, obj):
        raise NotImplementedError()

    def finalize(self, obj):
        pass


class TimeObjectiveTracker(Tracker):
    def __init__(self):
        self.tracker = np.array([[time.time(), 0.0]])

    def log_objective(self, obj):
        new_track_point = np.array([[time.time(), obj]])
        self.tracker = np.concatenate(
            (self.tracker, new_track_point), axis=0)

    def finalize(self):
        self.tracker[:, 0] -= self.tracker[0, 0]

class MultipleObjectiveTracker(Tracker):

    def __init__(self, **objectives):
        self.objectives = objectives
        self.log = defaultdict(list)
        self.time = []

    def log_objective(self, obj):
        t = time.time()
        for name, objective in self.objectives.items():
            self.log[name].append(objective())
        self.time.append(t)

    def finalize(self):
        self.time = np.array(self.time)
        self.time -= self.time[0]
        for k, l in self.log.items():
            self.log[k] = np.array(l)

@contextmanager
def eval_state(model):
    training_status = model.training

    try:
        model.eval()
        yield model
    finally:
        model.train(training_status)

def early_stopping(model, objective_closure, interval=5, patience=20, start=0, max_iter=1000,
                   maximize=True, tolerance=1e-5, switch_mode=True, restore_best=True,
                   tracker=None):
    """
    Early stopping iterator. When it stops, it restores the best previous state of the model.

    Args:
        model:     model that is being optimized
        objective_closue: objective function that is used for early stopping. Must be of the form objective() and
                          encapsulate the model. Should return the best objective
        interval:  interval at which objective is evaluated to consider early stopping
        patience:  number of times the objective is allow to not become better before the iterator terminates
        start:     start value for iteration (used to check against `max_iter`)
        max_iter:  maximum number of iterations before the iterator terminated
        maximize:  whether the objective is maximized of minimized
        tolerance: margin by which the new objective score must improve to be considered as an update in best score
        switch_mode: whether to switch model's train mode into eval prior to objective evaluation. If True (default),
                     the model is switched to eval mode before objective evaluation and restored to its previous mode
                     after the evaluation.
        restore_best: whether to restore the best scoring model state at the end of early stopping
        tracker (Tracker):
            for tracking training time & stopping objective

    """
    training_status = model.training

    def _objective():
        if switch_mode:
            model.eval()
        ret = objective_closure(model)
        if switch_mode:
            model.train(training_status)
        return ret

    def finalize(model, best_state_dict):
        old_objective = _objective()
        if restore_best:
            model.load_state_dict(best_state_dict)
            print(
                'Restoring best model! {:.6f} ---> {:.6f}'.format(
                    old_objective, _objective()))
        else:
            print('Final best model! objective {:.6f}'.format(
                _objective()))

    epoch = start
    maximize = float(maximize)
    best_objective = current_objective = _objective()
    best_state_dict = copy_state(model)
    patience_counter = 0
    while patience_counter < patience and epoch < max_iter:
        for _ in range(interval):
            epoch += 1
            if tracker is not None:
                tracker.log_objective(current_objective)
            if (~np.isfinite(current_objective)).any():
                print('Objective is not Finite. Stopping training')
                finalize(model, best_state_dict)
                return
            yield epoch, current_objective

        current_objective = _objective()

        if current_objective * (-1) ** maximize < best_objective * (-1) ** maximize - tolerance:
            print('[{:03d}|{:02d}/{:02d}] ---> {}'.format(epoch, patience_counter, patience, current_objective),
                  flush=True)
            best_state_dict = copy_state(model)
            best_objective = current_objective
            patience_counter = 0
        else:
            patience_counter += 1
            print('[{:03d}|{:02d}/{:02d}] -/-> {}'.format(epoch, patience_counter, patience, current_objective),
                  flush=True)
    finalize(model, best_state_dict)


def alternate(*args):
    """
    Given multiple iterators, returns a generator that alternatively visit one element from each iterator at a time.

    Examples:
        >>> list(alternate(['a', 'b', 'c'], [1, 2, 3], ['Mon', 'Tue', 'Wed']))
        ['a', 1, 'Mon', 'b', 2, 'Tue', 'c', 3, 'Wed']

    Args:
        *args: one or more iterables (e.g. tuples, list, iterators) separated by commas

    Returns:
        A generator that alternatively visits one element at a time from the list of iterables
    """
    for row in zip(*args):
        yield from row


def cycle_datasets(trainloaders):
    """
    Cycles through datasets of train loaders.

    Args:
        trainloaders: OrderedDict with trainloaders as values

    Yields:
        readout key, input, targets

    """
    assert isinstance(trainloaders, OrderedDict), 'trainloaders must be an ordered dict'
    for readout_key, outputs in zip(cycle(trainloaders.keys()), alternate(*trainloaders.values())):
        yield readout_key, outputs