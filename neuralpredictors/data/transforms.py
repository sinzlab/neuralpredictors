import numpy as np
import torch
from collections import namedtuple


class Invertible:
    def inv(self, y):
        raise NotImplemented("Subclasses of Invertible must implement an inv method")


class DataTransform:
    def __repr__(self):
        return self.__class__.__name__

    def id_transform(self, id_map):
        """
        Given a dictionary mapping from data group name to
        a numpy array of appropriate size containing identity information
        for that particular group name, the transform is expected to return
        the expected identity modification (e.g. dropping or duplication of entries, reordering, etc)
        """
        return id_map

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class MovieTransform(DataTransform):
    pass


class StaticTransform(DataTransform):
    pass


class Subsequence(MovieTransform):
    def __init__(self, frames, channel_first=("inputs",)):
        self.frames = frames
        # for special group, the time slicing is applied on
        self.channel_first = channel_first

    def __call__(self, x):
        first_group = x._fields[0]

        # get the length of the full sequence. Note that which axis to pick depends on if it's
        # channel first (i.e. time second) group
        t = getattr(x, first_group).shape[int(first_group in self.channel_first)]

        i = np.random.randint(0, t - self.frames)
        return x.__class__(
            **{
                k: getattr(x, k)[:, i : i + self.frames, ...]
                if k in self.channel_first
                else getattr(x, k)[i : i + self.frames, ...]
                for k in x._fields
            }
        )

    def id_transform(self, id_map):
        # until a better solution is reached, skipping this
        return id_map

        new_map = {}
        first_group = list(id_map.keys())[0]
        v_fg = id_map[first_group]
        t = v_fg.shape[int(first_group in self.channel_first)]
        i = np.random.randint(0, t - self.frames)
        for k, v in id_map.items():
            if k in self.channel_first:
                new_map[k] = v[:, i : i + self.frames]
            else:
                new_map[k] = v[i : i + self.frames]

        return new_map

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.frames)


class Delay(MovieTransform):
    """
    Delay the specified target gorups by delay frames. In other words,
    given non-delayed group g(t) and delayed group d(t), 
    g(T:N-delay) will be returned with d(T+delay:N) where `delay` is the specified amount
    of delay, and N is the last frame in the dataset.
    """

    def __init__(self, delay, delay_groups=("responses",), channel_first=("inputs",)):
        self.delay = delay
        self.delay_groups = delay_groups
        self.channel_first = channel_first

    def __call__(self, x):
        first_group = x._fields[0]
        t = getattr(x, first_group).shape[int(first_group in self.channel_first)]
        assert t > self.delay, "The sequence length {} has to be longer than the delay {}".format(t, self.delay)
        key_entry = {}
        for k in x._fields:
            if k in self.delay_groups:
                start, stop = self.delay, t
            else:
                start = 0
                stop = t - self.delay

            key_entry[k] = getattr(x, k)[:, start:stop] if k in self.channel_first else getattr(x, k)[start:stop]

        return x.__class__(**key_entry)

    def id_transform(self, id_map):
        # until a better solution is reached, skipping this
        return id_map

        new_map = {}
        first_group = list(id_map.keys())[0]
        v_fg = id_map[first_group]
        t = v_fg.shape[int(first_group in self.channel_first)]
        assert t > self.delay, "The sequence length {} has to be longer than the delay {}".format(t, self.delay)

        for k, v in id_map.items():
            if k in self.delay_groups:
                start, end = self.delay, t
            else:
                start, end = 0, t - self.delay

            new_map[k] = v[:, start:end] if k in self.channel_first else v[start:end]

        return new_map

    def __repr__(self):
        return self.__class__.__name__ + "({} on {})".format(self.delay, self.delay_groups)


class Subsample(MovieTransform, StaticTransform):
    def __init__(self, idx, target_group="responses"):
        self.idx = idx
        self.target_group = target_group
        assert np.ndim(self.idx) == 1, "Dimensionality of index array has to be 1"

    def __call__(self, x):
        return x.__class__(
            **{k: getattr(x, k)[..., self.idx] if k == self.target_group else getattr(x, k) for k in x._fields}
        )

    def __repr__(self):
        return self.__class__.__name__ + "(n={})".format(len(self.idx))

    def id_transform(self, id_map):
        return {k: v[self.idx] if k == self.target_group else v for k, v in id_map.items()}


class ToTensor(MovieTransform, StaticTransform, Invertible):
    def __init__(self, cuda=False):
        self.cuda = cuda

    def inv(self, y):
        return y.numpy()

    def __call__(self, x):
        return x.__class__(
            *[
                torch.from_numpy(elem.astype(np.float32)).cuda()
                if self.cuda
                else torch.from_numpy(elem.astype(np.float32))
                for elem in x
            ]
        )


class Identity(MovieTransform, StaticTransform, Invertible):
    def __call__(self, x):
        return x

    def inv(self, y):
        return y


class Rename(MovieTransform, StaticTransform, Invertible):
    def __init__(self, name_map):
        self.name_map = name_map
        self.rev_map = {v: k for k, v in name_map.items()}
        self.tuple_class = None
        self.origin_tuple_class = None

    def __call__(self, x):
        if self.tuple_class is None:
            if self.origin_tuple_class is None:
                self.origin_tuple_class = x.__class__
            renamed_fields = [self.name_map.get(f, f) for f in x._fields]
            self.tuple_class = namedtuple("RenamedDataPoint", renamed_fields)
        return self.tuple_class(*x)

    def inv(self, y):
        if self.origin_tuple_class is None:
            renamed_fields = [self.rev_map.get(f, f) for f in y._fields]
            self.origin_tuple_class = namedtuple("OriginalDataPoint", renamed_fields)
        return self.origin_tuple_class(*y)

    def id_transform(self, id_map):
        return {self.name_map.get(k, k): v for k, v in id_map.items()}


class NeuroNormalizer(MovieTransform, StaticTransform, Invertible):
    """
    Note that this normalizer only works with MovieDataset of very specific formulation

    Normalizes a trial with fields: inputs, behavior, eye_position, and responses. The pair of
    behavior and eye_position can be missing. The following normalizations are applied:

    - inputs are scaled by the training std of the stats_source and centered on the mean of the movie
    - behavior is divided by the std if the std is greater than 1% of the mean std (to avoid division by 0)
    - eye_position is z-scored
    - reponses are divided by the per neuron std if the std is greater than
            1% of the mean std (to avoid division by 0)
    """

    def __init__(self, data, stats_source="all", exclude=None):

        self.exclude = exclude or []

        in_name = "images" if "images" in data.statistics.keys() else "inputs"

        self._inputs_mean = data.statistics[in_name][stats_source]["mean"][()]
        self._inputs_std = data.statistics[in_name][stats_source]["std"][()]

        s = np.array(data.statistics["responses"][stats_source]["std"])

        # TODO: consider other baselines
        threshold = 0.01 * s.mean()
        idx = s > threshold
        self._response_precision = np.ones_like(s) / threshold
        self._response_precision[idx] = 1 / s[idx]
        transforms, itransforms = {}, {}

        # -- inputs
        transforms[in_name] = lambda x: (x - self._inputs_mean) / self._inputs_std
        itransforms[in_name] = lambda x: x * self._inputs_std + self._inputs_mean

        # -- responses
        transforms["responses"] = lambda x: x * self._response_precision
        itransforms["responses"] = lambda x: x / self._response_precision

        if "eye_position" in data.data_keys:
            # -- eye position
            self._eye_mean = np.array(data.statistics["eye_position"][stats_source]["mean"])
            self._eye_std = np.array(data.statistics["eye_position"][stats_source]["std"])
            transforms["eye_position"] = lambda x: (x - self._eye_mean) / self._eye_std
            itransforms["eye_position"] = lambda x: x * self._eye_std + self._eye_mean

            s = np.array(data.statistics["behavior"][stats_source]["std"])

            # TODO: same as above - consider other baselines
            threshold = 0.01 * s.mean()
            idx = s > threshold
            self._behavior_precision = np.ones_like(s) / threshold
            self._behavior_precision[idx] = 1 / s[idx]

            # -- behavior
            transforms["behavior"] = lambda x: x * self._behavior_precision
            itransforms["behavior"] = lambda x: x / self._behavior_precision

        self._transforms = transforms
        self._itransforms = itransforms

    def __call__(self, x):
        """
        Apply transformation
        """
        return x.__class__(
            **{k: (self._transforms[k](v) if k not in self.exclude else v) for k, v in zip(x._fields, x)}
        )

    def inv(self, x):
        return x.__class__(
            **{k: (self._itransforms[k](v) if k not in self.exclude else v) for k, v in zip(x._fields, x)}
        )

    def __repr__(self):
        return super().__repr__() + ("(not {})".format(", ".join(self.exclude)) if self.exclude is not None else "")


class AddBehaviorAsChannels(MovieTransform, StaticTransform, Invertible):
    """
    Given a StaticImage object that includes "images", "responses", and "behavior", it returns three variables:
        - input image concatinated with behavior as new channel(s)
        - responses
        - behavior
    """

    def __init__(self):
        self.transforms, self.itransforms = {}, {}
        self.transforms["images"] = lambda img, behavior: np.concatenate(
            (
                img,
                np.ones((1, *img.shape[-(len(img.shape) - 1) :]))
                * np.expand_dims(behavior, axis=((len(img.shape) - 2), (len(img.shape) - 1))),
            ),
            axis=len(img.shape) - 3,
        )
        self.transforms["responses"] = lambda x: x
        self.transforms["behavior"] = lambda x: x

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        dd = {
            "images": self.transforms["images"](key_vals["images"], key_vals["behavior"]),
            "responses": self.transforms["responses"](key_vals["responses"]),
            "behavior": self.transforms["behavior"](key_vals["behavior"]),
        }
        return x.__class__(**dd)


class SelectInputChannel(StaticTransform):
    """
    Given a StaticImage object that includes "images", it will select a particular input channel.
    """

    def __init__(self, grab_channel):
        self.grab_channel = grab_channel

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals["images"]
        key_vals["images"] = img[:, (self.grab_channel,)] if len(img.shape) == 4 else img[(self.grab_channel,)]
        return x.__class__(**key_vals)
