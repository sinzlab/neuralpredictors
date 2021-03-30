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
