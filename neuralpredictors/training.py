from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import numpy as np
import time
import warnings
import torch


def cycle(iterable):
    # see https://github.com/pytorch/pytorch/issues/23900
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


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
    """
    Abstract Tracker class to serve as the bass class for all trackers.
    Defines the two interfacing methods of `log_objective` and `finalize`.

    """

    def log_objective(self, obj=None):
        """
        Logs the provided object

        Args:
            obj (Any, optional): Object to be logged

        Raises:
            NotImplementedError: Override this method to provide a functional behavior.
        """
        raise NotImplementedError("Please override this method to provide functional behavior")

    def finalize(self, obj):
        pass


class TimeObjectiveTracker(Tracker):
    """
    Provides basic tracking of any object with a timestamp. Invoking `finalize()` will
    make all recorded timestamps relative to the first event tracked unless a specific
    time value to relativize against is provided.
    """

    def __init__(self, add_creation_event=False):
        """
        Initializes the tracker. If `add_creation_event` is True, then an entry is created with value
        `0.0` with the timestamp corresponding to the creation fo this tracker object.

        Args:
            add_creation_event (bool, optional): If set to True, an event for creation with value of 0.0 is added to the log. Defaults to False.
        """
        self.tracker = np.array([[time.time(), 0.0]]) if add_creation_event else np.empty((0, 0))

    def log_objective(self, obj):
        """
        Logs the provided object paired with the timestamp. Before finalizing by invoking `finalize()`, all events
        are logged with absolute time in epoch time in seconds.

        Args:
            obj (Any): Object to be logged.
        """
        new_track_point = np.array([[time.time(), obj]])
        self.tracker = np.concatenate((self.tracker, new_track_point), axis=0)

    def finalize(self, reference=None):
        """
        When invoked, all logs time entries are relativized against the first log entry time.
        Pass value `reference` to set the time relative to the passed-in value instead.

        Args:
            reference (float, optional): Timestamp to relativize all logged even times to. If None, relativizes all
                time entries to first time entry of the log. Defaults to None.
        """
        # relativize to the first entry of the tracked events unless reference provided
        reference = self.tracker[0, 0] if reference is None else reference
        # relativize the entry
        self.tracker[:, 0] -= reference


class MultipleObjectiveTracker(Tracker):
    """
    Given a dictionary of functions, this will invoke all functions and log the returned values against
    the invocation timestamp. Calling `finalize` relativizes the timestamps to the first entry (i.e. first
    log_objective call) made.
    """

    def __init__(self, default_name=None, **objectives):
        """
        Initializes the tracker. Pass any additional objective functions as keywords arguments.

        Args:
            default_name (string, optional): Name under which the objective value passed into `log_objective` is saved under.
                If set to None, the passed in value is NOT saved. Defaults to None.
        """
        self._default_name = default_name
        self.objectives = objectives
        self.log = defaultdict(list)
        self.time = []

    def log_objective(self, obj=None):
        """
        Log the objective and also returned values of any list of objective functions this tracker
        is configured with. The passed in value of `obj` is logged only if `default_name` was set to
        something except for None.

        Args:
            obj (Any, optional): Value to be logged if `default_name` is not None. Defaults to None.
        """
        t = time.time()
        # iterate through and invoke each objective function and accumulate the
        # returns. This is performed separately from actual logging so that any
        # error in one of the objective evaluation will halt the whole timestamp
        # logging, and avoid leaving partial results
        values = {}
        # log the passed in value only if `default_name` was given
        if self._default_name is not None:
            values[self._default_name] = obj

        for name, objective in self.objectives.items():
            values[name] = objective()

        # Actually log all values
        self.time.append(t)
        for name, value in values.items():
            self.log[name].append(value)

    def finalize(self, reference=None):
        """
        When invoked, all logs are convereted into numpy arrays and are relativized against the first log entry time.
        Pass value `reference` to set the time relative to the passed-in value instead.

        Args:
            reference (float, optional): Timestamp to relativize all logged even times to. If None, relativizes all
                time entries to first time entry of the log. Defaults to None.
        """
        self.time = np.array(self.time)
        reference = self.time[0] if reference is None else reference
        self.time -= reference
        for k, l in self.log.items():
            self.log[k] = np.array(l)

    def asdict(self, time_key="time", make_copy=True):
        """
        Output the cotent of the tracker as a single dictionary. The time value


        Args:
            time_key (str, optional): Name of the key to save the time information as. Defaults to "time".
            make_copy (bool, optional): If True, the returned log times will be a (shallow) copy. Defaults to True.

        Returns:
            dict: Dictionary containing tracked values as well as the time
        """
        log_copy = {k: np.copy(v) if make_copy else v for k, v in self.log.items()}
        log_copy[time_key] = np.copy(self.time) if make_copy else self.time
        return log_copy


@contextmanager
def eval_state(model):
    """
    Context manager, within which the model will be under `eval` mode.
    Upon existing, the model will return to whatever training state it
    was as it entered into the context.

    Args:
        model (PyTorch Module): PyTorch Module whose train/eval state is to be managed.

    Yields:
        PyTorch Module: The model switched to eval state.
    """
    training_status = model.training

    try:
        model.eval()
        yield model
    finally:
        model.train(training_status)


@contextmanager
def device_state(model, device):
    """
    Within the context, attemps to place the `model` onto the specified
    `device`. If `device` is CUDA and the specified device does not exist,
    the context falls back to using `cpu`. Upon existing the context, the model
    will be placed back on to the original device inferred based on the first entry
    of the model's parameter.

    Args:
        model (PyTorch Module): PyTorch Module object to swtich device.
        device (Any): target device descriptor. Any valid PyTorch device descriptor may be used.

    Yields:
        PyTorch Module: Model placed on the new device
    """
    # infer the original device based on the device the first parameter is placed on
    original_device = next(model.parameters()).device
    # create device spec
    # if device is simply "cuda", then device.index will evaluate to one, and the if statement will error out.
    device = torch.device("cuda:0") if device == "cuda" else torch.device(device)
    if device.type == "cuda" and device.index >= torch.cuda.device_count():
        # fall back to using CPU
        warnings.warn("Incompatible CUDA spec. Falling back to CPU usage")
        device = "cpu"

    try:
        model.to(device)
        yield model
    finally:
        model.to(original_device)


def early_stopping(
    model,
    objective,
    interval=5,
    patience=20,
    start=0,
    max_iter=1000,
    maximize=True,
    tolerance=1e-5,
    switch_mode=True,
    restore_best=True,
    tracker=None,
    scheduler=None,
    lr_decay_steps=1,
):

    """
    Early stopping iterator. Keeps track of the best model state during training. Resets the model to its
        best state, when either the number of maximum epochs or the patience [number of epochs without improvement)
        is reached.
    Also includes a convenient way to reduce the learning rate. Takes as an additional input a PyTorch scheduler object
        (e.g. torch.optim.lr_scheduler.ReduceLROnPlateau), which will automatically decrease the learning rate.
        If the patience counter is reached, the scheduler will decay the LR, and the model is set back to its best state.
        This loop will continue for n times in the variable lr_decay_steps. The patience and tolerance parameters in
        early stopping and the scheduler object should be identical.


    Args:
        model:     model that is being optimized
        objective: objective function that is used for early stopping. The function must accept single positional argument `model`
            and return a single scalar quantity.
        interval:  interval at which objective is evaluated to consider early stopping
        patience:  number of continuous epochs the objective could remain without improvement before the iterator terminates
        start:     start value for iteration (used to check against `max_iter`)
        max_iter:  maximum number of iterations before the iterator terminated
        maximize:  whether the objective is maximized of minimized
        tolerance: margin by which the new objective score must improve to be considered as an update in best score
        switch_mode: whether to switch model's train mode into eval prior to objective evaluation. If True (default),
                     the model is switched to eval mode before objective evaluation and restored to its previous mode
                     after the evaluation.
        restore_best: whether to restore the best scoring model state at the end of early stopping
        tracker (Tracker):
            Tracker to be invoked for every epoch. `log_objective` is invoked with the current value of `objective`. Note that `finalize`
            method is NOT invoked.
        scheduler:  scheduler object, which automatically reduces decreases the LR by a specified amount.
                    The scheduler's `step` method is invoked, passing in the current value of `objective`
        lr_decay_steps: Number of times the learning rate should be reduced before stopping the training.

    """
    training_status = model.training

    def _objective():
        if switch_mode:
            model.eval()
        ret = objective(model)
        if switch_mode:
            model.train(training_status)
        return ret

    def decay_lr(model, best_state_dict):
        old_objective = _objective()
        if restore_best:
            model.load_state_dict(best_state_dict)
            print("Restoring best model after lr decay! {:.6f} ---> {:.6f}".format(old_objective, _objective()))

    def finalize(model, best_state_dict):
        old_objective = _objective()
        if restore_best:
            model.load_state_dict(best_state_dict)
            print("Restoring best model! {:.6f} ---> {:.6f}".format(old_objective, _objective()))
        else:
            print("Final best model! objective {:.6f}".format(_objective()))

    epoch = start
    # turn into a sign
    maximize = -1 if maximize else 1
    best_objective = current_objective = _objective()
    best_state_dict = copy_state(model)

    for repeat in range(lr_decay_steps):
        patience_counter = 0

        while patience_counter < patience and epoch < max_iter:

            for _ in range(interval):
                epoch += 1
                if tracker is not None:
                    tracker.log_objective(current_objective)
                if (~np.isfinite(current_objective)).any():
                    print("Objective is not Finite. Stopping training")
                    finalize(model, best_state_dict)
                    return
                yield epoch, current_objective

            current_objective = _objective()

            # if a scheduler is defined, a .step with the current objective is all that is needed to reduce the LR
            if scheduler is not None:
                scheduler.step(current_objective)

            if current_objective * maximize < best_objective * maximize - tolerance:
                print(
                    "[{:03d}|{:02d}/{:02d}] ---> {}".format(epoch, patience_counter, patience, current_objective),
                    flush=True,
                )
                best_state_dict = copy_state(model)
                best_objective = current_objective
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    "[{:03d}|{:02d}/{:02d}] -/-> {}".format(epoch, patience_counter, patience, current_objective),
                    flush=True,
                )

        if (epoch < max_iter) & (lr_decay_steps > 1) & (repeat < lr_decay_steps):
            decay_lr(model, best_state_dict)

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


def cycle_datasets(loaders):
    """
    Given a dictionary mapping data_key into dataloader objects, returns a generator that alternately yields
    output from the loaders in the dictionary. The order of data_key traversal is determined by the first invocation to `.keys()`.
    To obtain deterministic behavior of key traversal, recommended to use OrderedDict.

    The generator terminates as soon as any one of the constituent loaders is exhausted.

    Args:
        loaders (dict): Dict mapping a data_key to a dataloader object.

    Yields:
        string, Any: data_key  and and the next output from the data loader corresponding to the data_key
    """
    keys = list(loaders.keys())
    # establish a consistent ordering across loaders
    ordered_loaders = [loaders[k] for k in keys]
    for data_key, outputs in zip(cycle(loaders.keys()), alternate(*ordered_loaders)):
        yield data_key, outputs


class Exhauster:
    """
    Given a dictionary of data loaders, mapping data_key into a data loader, steps through each data loader, moving onto the next data loader
    only upon exhausing the content of the current data loader.
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        for data_key, loader in self.loaders.items():
            for batch in loader:
                yield data_key, batch

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])


class LongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders.values()])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, _ in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.max_batches),
        ):
            yield k, next(loader)

    def __len__(self):
        return len(self.loaders) * self.max_batches


class ShortCycler:
    """
    Cycles through trainloaders until the loader with smallest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.min_batches = min([len(loader) for loader in self.loaders.values()])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, _ in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.min_batches),
        ):
            yield k, next(loader)

    def __len__(self):
        return len(self.loaders) * self.min_batches
