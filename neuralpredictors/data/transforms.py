import numpy as np
import torch
from collections import namedtuple, Iterable
from skimage.transform import rescale


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
    """
    Abstract class to certify that the transform is valid for sequence like (e.g. movie) datasets.
    """


class StaticTransform(DataTransform):
    """
    Abstract class to certify that the transform is valid for non-sequential (e.g. image) datasets.
    """


class Subsequence(MovieTransform):
    def __init__(self, frames, channel_first=("inputs",), offset=-1):
        """
        Given a sequential (movie like) data, subselect a consequent `frames` counts of frames, starting with
        `offset` frames skipped. If `offset`< 0, then the subsequence is taken with a random (but valid) offset each iteration.

        Args:
            frames (int): Length of subsequence to be selected from each sample
            channel_first (tuple, optional): A list of data key names where the channel (and thus not time) dimension occurs on the first dimension (dim=0). Otherwise, it's assumed
            that the time dimesion occurs on the first dimension. Defaults to ("inputs",).
            offset (int, optional): Offset to start the subsequence from. Defaults to -1, corresponding to random but valid offset at each iteration.
        """
        self.frames = frames
        # for special group, the time slicing is applied on
        self.channel_first = channel_first
        self.offset = offset

    def __call__(self, x):
        first_group = x._fields[0]

        # get the length of the full sequence. Note that which axis to pick depends on if it's
        # channel first (i.e. time second) group
        t = getattr(x, first_group).shape[int(first_group in self.channel_first)]

        if self.offset < 0:
            i = np.random.randint(0, t - self.frames)
        else:
            i = self.offset
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


class Stack(MovieTransform):
    def __init__(
        self,
        target="inputs",
        sources=("eye_pos", "behavior"),
        concat_axis=0,
        transpose=True,
    ):
        """
        Stack source data elements into the target data elements. In stacking, the source data elements are
        left aligned with the target, and it's dimensions expanded as necessary before stacking along the
        specified existing axis.

        Examples:
        If target is an array of shape [1, 150, 36, 64], and one of the sources is of shape [3, 150], then
        the source data is first expanded into [3, 150, 1, 1], followed by tiling to achieve [3, 150, 36, 64],
        and this is finally stacked with the target to yield a new output of shape [4, 150, 36, 64], where the
        output[0, ...] is the original target, and output[1:4] is the expanded source data. If `tranpose=True`,
        the source is transposed first before performing dimension alignment and expansions.

        Args:
            target (str, optional): Data key for the target to be modified with stacking. Defaults to "inputs".
            sources (str or tuple, optional): A single source or atuple of sources to be stacked into the target.
                Defaults to ("eye_pos", "behavior").
            concat_axis (int, optional): Axis along which sources are concatenated into the target. Defaults to 0.
            transpose (bool, optional): Whether to transpose the sources first. Defaults to True.
        """
        self.target = target
        if isinstance(sources, str):
            sources = (sources,)
        self.sources = sources
        self.concat_axis = concat_axis
        self.transpose = transpose

    def __call__(self, x):
        x_dict = x._asdict()
        target = x_dict[self.target]
        groups = [target]
        for source in [x_dict[s] for s in self.sources]:
            if self.transpose:
                source = source.T
            n_target = len(target.shape)
            n_source = len(source.shape)
            dims = list(range(-n_target + n_source, 0))
            groups.append(np.ones((1,) * n_source + target.shape[n_source:]) * np.expand_dims(source, axis=dims))
        x_dict[self.target] = np.concatenate(groups, axis=self.concat_axis)
        return x.__class__(**x_dict)

    def id_transform(self, id_map):
        # until a better solution is reached, skipping this
        return id_map

    def __repr__(self):
        items = ", ".join(s + ".T" if self.transpose else s for s in self.sources)
        return self.__class__.__name__ + "(stack [{}] on {} along axis={})".format(items, self.target, self.concat_axis)


class Subsample(MovieTransform, StaticTransform):
    def __init__(self, idx, target_group="responses", target_index=None):
        """
        Subselects samples for data_key specified by `target_group`. By default, the subselection is performed on
        the last index of the tensor, but this behavior may be modified by passing in `target_index` a dictionary
        mapping the position of the index to the name of the data_key.

        Args:
            idx (numpy index specifier): Indices to be selected. Must be a valid NumPy index specification (e.g. list of indicies, boolean array, etc.)
            target_group (string or iterable of strings): Specifies the taget data key to perform subselection on. If given a string, it is assumed as the direct name of data_key
                Otherwise, it is assumed to be an iterable over string values of all data_keys
            target_index (optional, dict): If provided a dictionary, the key is asssumed to be the name of data_key and value the index position to perform subselection on. If not provided, index position of -1 (last position) is used.
        """

        self.idx = idx
        if isinstance(target_group, str):
            self.target_groups = (target_group,)
        else:
            self.target_groups = target_group

        if target_index is None:
            target_index = {k: -1 for k in self.target_groups}
        elif isinstance(target_index, int):
            target_index = {k: target_index for k in self.target_groups}
        self.target_index = target_index

        assert np.ndim(self.idx) == 1, "Dimensionality of index array has to be 1"

    def __call__(self, x):
        return x.__class__(
            **{
                k: np.take(getattr(x, k), self.idx, self.target_index[k]) if k in self.target_groups else getattr(x, k)
                for k in x._fields
            }
        )

    def __repr__(self):
        return self.__class__.__name__ + "(n={})".format(len(self.idx))

    def id_transform(self, id_map):
        return {k: v[self.idx] if k in self.target_groups else v for k, v in id_map.items()}


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
    Note that this normalizer only works with datasets that provide specific attributes information
    of very specific formulation

    Normalizes a trial with fields: inputs, behavior, eye_position, and responses. The pair of
    behavior and eye_position can be missing. The following normalizations are applied:

    - inputs are scaled by the training std of the stats_source and centered on the mean of the movie
    - behavior is divided by the std if the std is greater than 1% of the mean std (to avoid division by 0)
    - eye_position is z-scored
    - reponses are divided by the per neuron std if the std is greater than
            1% of the mean std (to avoid division by 0)
    """

    def __init__(self, data, stats_source="all", exclude=None, inputs_mean=None, inputs_std=None):

        self.exclude = exclude or []

        in_name = "images" if "images" in data.statistics.keys() else "inputs"
        out_name = "responses" if "responses" in data.statistics.keys() else "targets"
        eye_name = "pupil_center" if "pupil_center" in data.data_keys else "eye_position"

        self._inputs_mean = data.statistics[in_name][stats_source]["mean"][()] if inputs_mean is None else inputs_mean
        self._inputs_std = data.statistics[in_name][stats_source]["std"][()] if inputs_mean is None else inputs_std

        s = np.array(data.statistics[out_name][stats_source]["std"])

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
        transforms[out_name] = lambda x: x * self._response_precision
        itransforms[out_name] = lambda x: x / self._response_precision

        # -- behavior
        transforms["behavior"] = lambda x: x

        # -- trial_idx
        trial_idx_mean = np.arange(data._len).mean()
        trial_idx_std = np.arange(data._len).std()
        transforms["trial_idx"] = lambda x: (x - trial_idx_mean) / trial_idx_std
        itransforms["trial_idx"] = lambda x: x * trial_idx_std + trial_idx_mean

        if eye_name in data.data_keys:
            self._eye_mean = np.array(data.statistics[eye_name][stats_source]["mean"])
            self._eye_std = np.array(data.statistics[eye_name][stats_source]["std"])
            transforms[eye_name] = lambda x: (x - self._eye_mean) / self._eye_std
            itransforms[eye_name] = lambda x: x * self._eye_std + self._eye_mean

        if "behavior" in data.data_keys:
            s = np.array(data.statistics["behavior"][stats_source]["std"])

            self._behavior_precision = 1 / s
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
        self.transforms["pupil_center"] = lambda x: x
        self.transforms["trial_idx"] = lambda x: x

    def __call__(self, x):

        key_vals = {k: v for k, v in zip(x._fields, x)}
        dd = {
            "images": self.transforms["images"](key_vals["images"], key_vals["behavior"]),
            "responses": self.transforms["responses"](key_vals["responses"]),
            "behavior": self.transforms["behavior"](key_vals["behavior"]),
        }
        if "pupil_center" in key_vals:
            dd["pupil_center"] = self.transforms["pupil_center"](key_vals["pupil_center"])
        if "trial_idx" in key_vals:
            dd["trial_idx"] = self.transforms["trial_idx"](key_vals["trial_idx"])
        return x.__class__(**dd)


class AddPupilCenterAsChannels(MovieTransform, StaticTransform, Invertible):
    """
    Given a StaticImage object that includes "images", "responses", and "pupil center", it returns three variables:
        - input image concatenated with eye position as new channel(s)
        - responses
        - behavior
        - pupil center
    """

    def __init__(self):
        self.transforms, self.itransforms = {}, {}
        self.transforms["images"] = lambda img, pupil_center: np.concatenate(
            (
                img,
                np.ones((1, *img.shape[-(len(img.shape) - 1) :]))
                * np.expand_dims(pupil_center, axis=((len(img.shape) - 2), (len(img.shape) - 1))),
            ),
            axis=len(img.shape) - 3,
        )
        self.transforms["responses"] = lambda x: x
        self.transforms["behavior"] = lambda x: x
        self.transforms["pupil_center"] = lambda x: x

    def __call__(self, x):

        key_vals = {k: v for k, v in zip(x._fields, x)}
        dd = {
            "images": self.transforms["images"](key_vals["images"], key_vals["pupil_center"]),
            "responses": self.transforms["responses"](key_vals["responses"]),
        }
        if "behavior" in key_vals:
            dd["behavior"] = self.transforms["behavior"](key_vals["behavior"])
        dd["pupil_center"] = self.transforms["pupil_center"](key_vals["pupil_center"])

        if "trial_idx" in key_vals:
            dd["trial_idx"] = self.transforms["trial_idx"](key_vals["trial_idx"])
        return x.__class__(**dd)


class SelectInputChannel(StaticTransform):
    """
    Given a StaticImage object that includes "images", it will select a particular input channel.
    """

    def __init__(self, grab_channel):
        self.grab_channel = grab_channel if isinstance(grab_channel, Iterable) else [grab_channel]

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals["images"]
        key_vals["images"] = img[:, (self.grab_channel,)] if len(img.shape) == 4 else img[self.grab_channel, ...]
        return x.__class__(**key_vals)


class ReshapeImages(StaticTransform):
    """
    Given a StaticImage object that includes "images", it will select a particular input channel.
    """

    def __init__(self, reshape_list):
        self.reshape_list = reshape_list

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals["images"]
        key_vals["images"] = img.transpose(self.reshape_list)
        return x.__class__(**key_vals)


class AddPositionAsChannels(StaticTransform):
    """
    Given a StaticImage object that includes "images", it will add two channels,
    which contain the pixel positions from -1 to 1.
    """

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals["images"]
        img_shape = img.shape
        full_img = (
            np.ones((img_shape[0] + 2, img_shape[1], img_shape[2]))
            if len(img_shape) == 3
            else np.ones((img_shape[0], img_shape[1] + 2, img_shape[2], img_shape[3]))
        )
        positions = np.stack(np.meshgrid(np.linspace(-1, 1, img_shape[-1]), np.linspace(-1, 1, img_shape[-2])))
        if len(img_shape) == 3:
            full_img[:-2, ...] = img
            full_img[-2:, ...] = positions
        else:
            full_img[:, :-2, ...] = img
            full_img[:, -2:, ...] = positions
        key_vals["images"] = full_img
        return x.__class__(**key_vals)


class ScaleInputs(StaticTransform, Invertible):
    """
    Applies skimage.transform.rescale to the data_key "images".
    """

    def __init__(
        self,
        scale,
        mode="reflect",
        multichannel=False,
        anti_aliasing=False,
        preserve_range=True,
        clip=True,
        in_name="images",
    ):

        self.scale = scale
        self.mode = mode
        self.multichannel = multichannel
        self.anti_aliasing = anti_aliasing
        self.preserve_range = preserve_range
        self.clip = clip
        self.in_name = in_name

    def __call__(self, x):
        key_vals = {k: v for k, v in zip(x._fields, x)}
        img = key_vals[self.in_name]
        key_vals[self.in_name] = rescale(
            img,
            scale=self.scale,
            mode=self.mode,
            multichannel=self.multichannel,
            anti_aliasing=self.anti_aliasing,
            clip=self.clip,
            preserve_range=self.preserve_range,
        )
        return x.__class__(**key_vals)
