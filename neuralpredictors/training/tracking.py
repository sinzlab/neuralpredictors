"""
This module implements tracker classes that can be used to keep track
of statistics that are generated throughout training.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import abc, defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from .utils import deep_update

logger = logging.getLogger(__name__)


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


class AdvancedTracker(Tracker):
    """
    This class implements a more advanced, universal tracker that offers many useful features:
     - Store logging information in an arbitrary hierarchical structure, e.g.:
       ```
       {
        "learning_rate" : [0.1, 0.08],
        "Training":
        {
            "img_classification": {
               "accuracy" : [345.234234, 242.34],
               "loss" : [0.342342, 0.78432],
               "normalization": [4,3],
            }
            "neural_prediction": {
               "accuracy" : [345.234234, 242.34],
               "loss" : [0.342342, 0.78432],
               "normalization": [4,3],
            }
        }
        "Validation": {
             "img_classification": {
               "accuracy" : [348.234234, 0.0],
               "loss" : [0.42342, 0.0],
               "normalization": [2,1],
             },
             "patience": [0],
         },
       }
       ```
     - Retrieving and manipulating information via a simple hierarchical key,
       e.g. `("Training","img_classification","loss")`
     - Scores for the same epoch are automatically accumulated via `log_objective(...)` (usually unnormalized)
     - Scores can be automatically normalized if `"normalization"` (e.g. total examples) is tracked on the same hierarchy level
     - Automatic per-epoch initialization for each objective when a new epoch is started via `start_epoch()`
     - Automatic output of current logs to a logger or even tqdm
     - Easy extraction and reloading of tracker state via `state_dict()`, `from_dict(sd)` and `load_state_dict(sd)`


     Example usage:
     1. Specify the log structure and initial values:
         ```
        objectives = {
            "lr": 0,
            "training": {
                "img_classification": {"loss": 0, "accuracy": 0, "normalization": 0}
            },
            "validation": {
                "img_classification": {
                    "loss": 0,
                    "accuracy": 0,
                    "normalization": 0,
                },
                "patience": 0,
            },
        }
        tracker = AdvancedTracker(
            main_objective=("img_classification", "accuracy"), **objectives
        )
         ```
     2. Start a new epoch:
     ```
        self.tracker.start_epoch()
     ```
     3. Log the objectives:
     ```
        tracker.log_objective(
                100 * predicted.eq(targets).sum().item(),
                keys=(mode, task_key, "accuracy"),
            )
        tracker.log_objective(
            batch_size,
            keys=(mode, task_key, "normalization"),
        )
        tracker.log_objective(
            loss.item() * batch_size,
            keys=(mode, task_key, "loss"),
        )
    ```
    4. Display the current objective values, e.g. everything related to training:
    ```
        self.tracker.display_log(tqdm_iterator=t, keys=("training",))
    ```
    5. Save the tracker (i.e. save the training progress):
    ```
        self.tracker.state_dict()
    ```
    """

    def __init__(self, main_objective: Tuple[str, ...] = (), **objectives: Mapping) -> None:
        """
        In principle, `objectives` is expected to be a dictionary of dictionaries
        The hierarchy can in principle be arbitrarily deep.
        The only restriction is that the lowest level has to be a dictionary with values being
        either a numerical value which will be interpreted as the initial value for this objective
        (to be accumulated manually) or a callable (e.g. a function) that returns the objective value.

        Args:
            main_objective: key of the main objective that is used to e.g. decide lr reductions
                            can be something like `("img_classification","accuracy")` to always look at accuracy.
                            Can be combined with a setting specific key in `get_current_main_objective(...)`
            **objectives: e.g. {"dataset": {"objective1": o_fct1, "objective2": 0, "normalization": 0},...}
                           or {"dataset": {"task_key": {"objective1": o_fct1, "objective2": 0},...},...}.
                          Here the key "normalization" is optional on each hierarchy level.
                          If "normalization" exists, then this entry is expected to contain the value that
                          is used to normalize all other values on the same level,
                          and the normalization will be applied whenever the log is returned to the outside.
                          E.g. if a loss is supposed to be tracked, then the "loss" entry will contain the
                          un-normalized accumulation of loss for all inputs in that epoch at any point.
                          To get the normalized loss that is commonly used in practice, simply accumulate the
                          total number of inputs in the "normalization" entry and let AdvancedTracker do the rest.


        """
        self.objectives = objectives
        self.log = self._initialize_log(objectives)
        self.time = []
        self.main_objective = main_objective
        self.epoch = -1  # will be increased to 0 in next line
        self.start_epoch()

    def add_objectives(self, objectives: Mapping, init_epoch: bool = False) -> None:
        """
        Add new objectives (with initial values) to the logs.
        Args:
            objectives: dictionary that needs to follow the same structure as self.log
            init_epoch: flag that decides whether the a new epoch is initialized for this objective
        """
        deep_update(self.objectives, objectives)
        new_log = self._initialize_log(objectives)
        if init_epoch:
            self._initialize_epoch(new_log, objectives)
        deep_update(self.log, new_log)

    def start_epoch(self, append_epoch: bool = True) -> None:
        """Start a new epoch. Initialize each accumulation with its default value."""
        t = time.time()
        self.time.append(t)
        if append_epoch:
            self.epoch += 1
        else:
            self.epoch = 0
        self._initialize_epoch(self.log, self.objectives, append_epoch=append_epoch)

    def log_objective(self, value: float, key: Tuple[str, ...] = ()) -> None:
        """
        Add a new entry to the logs
        Args:
            value: objective score
            key: hierarchical key to match in `self.log`
        """
        if key:
            self._log_objective_value(value, self.log, key)
        else:
            self._log_objective_callables(self.log, self.objectives)

    def display_log(self, key: Tuple[str, ...] = (), tqdm_iterator: Optional[tqdm] = None) -> None:
        """
        Display the current objective value of everything under `key`
        Args:
            key: Tuple describing the objective to display.
                 This could be something like `("Training","img_classification","loss")`
                 to display the current classification loss or something like `("Training","img_classification")`
                 to display everything we save for image classifcation training.
            tqdm_iterator: A tqdm object that the log could be displayed on.
        """
        # normalize (if applicable) and turn into np.arrays:
        n_log = self._normalize_log(self.log)
        # flatten a subset of the dictionary:
        current_log = self._gather_log(n_log, key, index=-1)
        if tqdm_iterator:
            tqdm_iterator.set_postfix(**current_log)
        else:
            logger.info(current_log)

    def get_objective(self, log: Optional[Union[Mapping, Sequence]] = None, key: Tuple[str, ...] = ()) -> np.array:
        """
        Get the value of the objective that corresponds to the key.
        Args:
            log: log to retrieve objective from
            key: key to match with the log

        Returns:
            value array (across epochs and normalized) for a specific objective key
        """
        if log is None:
            log = self._normalize_log(self.log)
        if len(key) > 1:
            return self.get_objective(log[key[0]], key[1:])
        else:
            if isinstance(log[key[0]], (list, np.ndarray)):
                return log[key[0]]
            else:
                raise ValueError("The key does not fully match an objective. Try specifying the complete key sequence.")

    def get_current_objective(self, key: Tuple[str, ...]) -> float:
        return self.get_objective(self._normalize_log(self.log), key)[-1]

    def get_current_main_objective(self, key: Tuple[str, ...]) -> float:
        """
        Main objective is saved in tracker to make it convenient to get the main objective
        that would be used e.g. for learning-rate reduction or similar.
        """
        combined_key = key + self.main_objective if isinstance(key, tuple) else (key,) + self.main_objective
        return self.get_current_objective(combined_key)

    def check_isfinite(self, log: Optional[Union[Mapping, Sequence]] = None) -> bool:
        """
        Checks if all entries in `log` or (normalized) `self.log` are finite.
        Args:
            log: dict that is recursively searched for infinite entries

        Returns: True if all entries are finite.
        """
        if log is None:
            log = self._normalize_log(self.log)
        if isinstance(log, abc.Mapping):
            for k, l in log.items():
                if not self._check_isfinite(l):
                    return False
        else:
            return np.isfinite(log).any()
        return True

    def finalize(self) -> None:
        """After training, normalize the log and save the total time."""
        self.time = np.array(self.time)
        self.time -= self.time[0]
        self.log = self._normalize_log(self.log)

    def state_dict(self) -> Dict[str, any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            Dict[str, any]: All attributes that make up this configuration instance
        """
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, tracker_dict: Dict) -> None:
        """
        Loads given state_dict from another tracker.
        Args:
            tracker_dict: state that should override this tracker

        """
        self.main_objective = tracker_dict["main_objective"]
        self.objectives = tracker_dict["objectives"]
        self.log = tracker_dict["log"]
        self.time = tracker_dict["time"]
        self.epoch = tracker_dict["epoch"]

    @classmethod
    def from_dict(cls, tracker_dict: Dict) -> AdvancedTracker:
        """
        Same as `load_state_dict`, but creates a new tracker from scratch.
        Args:
            tracker_dict: state of tracker that should be loaded in new tracker

        Returns:
            new tracker with `tracker_dict` loaded

        """
        tracker = cls(main_objective=tracker_dict["main_objective"], **tracker_dict["objectives"])
        tracker.log = tracker_dict["log"]
        tracker.time = tracker_dict["time"]
        tracker.epoch = tracker_dict["epoch"]
        return tracker

    def _initialize_log(self, objectives: Mapping[str, any]) -> Dict:
        log = {}
        for key, objective in objectives.items():
            if isinstance(objective, abc.Mapping):
                log[key] = self._initialize_log(objective)
            elif not callable(objective):
                log[key] = []
        return log

    def _initialize_epoch(
        self, log: Union[Mapping, Sequence], objectives: Mapping[str, any], append_epoch: bool = True
    ) -> None:
        """
        For each key in `objectives`, go through log and append a new entry to its list.
        The entry reflects the default value saved in `objectives`.
        Args:
            log: sub dictionary to add the objectives to
            objectives: dictionary of default values
        """
        for key, objective in objectives.items():
            if isinstance(objective, abc.Mapping):
                self._initialize_epoch(log[key], objective)
            elif not callable(objective):
                if append_epoch or len(log[key]) == 0:
                    while len(log[key]) <= self.epoch:
                        log[key].append(objective)
                else:
                    log[key][0] = objective

    def _log_objective_value(self, value: float, log: Union[Mapping, Sequence], key: Tuple[str, ...] = ()) -> None:
        """
        Recursively walk through the log dictionary to get to follow `key`.
        When on lowest level: add `value` to entry at current epoch.
        Args:
            value: objective value to log
            log: log subdict to add value to
            key: key for where `value` is saved
        """
        if len(key) > 1:
            self._log_objective_value(value, log[key[0]], key[1:])
        else:
            log[key[0]][self.epoch] += value

    def _log_objective_callables(self, log: Union[Mapping, Sequence], objectives: Mapping[str, any]) -> None:
        """
        Log all objectives that are specified as callables
        Disclaimer: this is not very well tested!
        Args:
            log:
            objectives:
        """
        for key, objective in objectives.items():
            if isinstance(objective, abc.Mapping):
                self._log_objective_callables(log[key], objective)
            elif callable(objective):
                log[key][self.epoch] += objective()

    def _normalize_log(self, log: Union[Mapping, Sequence]) -> Union[Mapping, np.array]:
        """
        Recursively go through the log and normalize the entries that
        have `"normalization"` information on the same level.
        Args:
            log: subdict on which to apply normalization or list (if on lowest level)

        Returns:
            Normalized log dictionary that has numpy arrays on the lowest level

        """
        if isinstance(log, abc.Mapping):
            n_log = {}
            norm = None
            for key, l in log.items():
                res = self._normalize_log(l)  # to turn into arrays
                if key == "normalization":
                    assert isinstance(res, np.ndarray)
                    norm = res
                else:
                    n_log[key] = res
            self._normalize(n_log, norm)
            return n_log
        else:
            return np.array(log)

    def _normalize(self, n_log: Mapping, norm: np.ndarray):
        """
        Normalizes sub-dict `n_log`
        """
        if norm is not None:
            nonzero_start = (norm != 0).argmax(axis=0)
            norm = norm[nonzero_start:]
            for key, l in n_log.items():
                l = l[nonzero_start:]
                if isinstance(l, np.ndarray):
                    n_log[key] = l / np.where(norm > 0, norm, np.ones_like(norm))

    def _gather_log(self, log: Union[Mapping, Sequence], key: Tuple[str, ...], index: int = -1) -> Dict[str, str]:
        """
        Get a flattened and print-ready version of the log dictionary for a given key
        Args:
            log: subdict to retrieve values from
            key: tuple describing on which level to retrieve from
            index: which epoch to retrieve from

        Returns:
            Flattened dictionary, e.g. `{"img_clasisification accuracy": 98.5, "img_classsifcation loss": 0.456}
        """
        if len(key) > 1:
            return self._gather_log(log[key[0]], key[1:], index)
        elif key:
            return self._gather_log(log[key[0]], (), index)
        elif isinstance(log, abc.Mapping):
            gathered = {}
            for key, l in log.items():
                logs = self._gather_log(l, (), index)
                for k, v in logs.items():
                    gathered[key + " " + k] = v
            return gathered
        else:
            return {"": "{:03.5f}".format(log[index])}
