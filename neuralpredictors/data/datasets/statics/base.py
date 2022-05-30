from collections import namedtuple

import h5py

from ...transforms import StaticTransform
from ...utils import recursively_load_dict_contents_from_group
from ..base import TransformDataset, default_image_datapoint


class StaticSet(TransformDataset):
    def __init__(self, *data_keys, transforms=None):
        """
        Abstract class for static datasets. Defines data_keys and a corresponding datapoint.
        """
        super().__init__(transforms=transforms)

        self.data_keys = data_keys
        if set(data_keys) == {"images", "responses"}:
            # this version IS serializable in pickle
            self.data_point = default_image_datapoint
        else:
            # this version is NOT - you cannot use this with a dataloader with num_workers > 1
            self.data_point = namedtuple("DataPoint", data_keys)


class H5ArraySet(StaticSet):
    def __init__(self, filename, *data_keys, transforms=None):
        """
        Dataset for static data stored in hdf5 files.
        Args:
            filename:      filename of the hdf5 file
            *data_keys:    data keys to be read from the file
            transforms:    list of transforms applied to each datapoint
        """
        super().__init__(*data_keys, transforms=transforms)

        self._fid = h5py.File(filename, "r")
        self.data = self._fid
        self.data_loaded = False
        m = None
        for key in data_keys:
            assert key in self.data, "Could not find {} in file".format(key)
            if m is None:
                m = len(self.data[key])
            else:
                assert m == len(self.data[key]), "Length of datasets do not match"
        self._len = m

    def load_content(self):
        self.data = recursively_load_dict_contents_from_group(self._fid)
        self.data_loaded = True

    def unload_content(self):
        self.data = self._fid
        self.data_loaded = False

    def __getitem__(self, item):
        x = self.data_point(*(self.data[g][item] for g in self.data_keys))
        for tr in self.transforms:
            assert isinstance(tr, StaticTransform)
            x = tr(x)
        return x

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __len__(self):
        return self._len

    def __repr__(self):
        return "\n".join(
            ["Tensor {}: {} ".format(key, self.data[key].shape) for key in self.data_keys]
            + ["Transforms: " + repr(self.transforms)]
        )

    def __getattr__(self, item):
        if item in self.data:
            item = self.data[item]
            if isinstance(item, h5py.Dataset):
                dtype = item.dtype
                item = item[()]
                if dtype.char == "S":  # convert bytes to univcode
                    item = item.astype(str)
                return item
            return item
        else:
            raise AttributeError("Item {} not found in {}".format(item, self.__class__.__name__))


class StaticImageSet(H5ArraySet):
    def __init__(self, filename, *data_keys, transforms=None, cache_raw=False, stats_source=None):
        """
        Dataset for h5 files.
        Args:
            filename:       filename of the hdf5 file
            *data_keys:     datasets to be extracted
            transforms:     transforms applied to each data point
            cache_raw:      whether to cache the raw (untransformed) datapoints
            stats_source:   statistic source to be used.
        """
        super().__init__(filename, *data_keys, transforms=transforms)
        self.cache_raw = cache_raw
        self.last_raw = None
        self.stats_source = stats_source if stats_source is not None else "all"

    @property
    def n_neurons(self):
        return len(self[0].responses)

    @property
    def neurons(self):
        return AttributeTransformer("neurons", self.data, self.transforms, data_group="responses")

    @property
    def info(self):
        return AttributeHandler("item_info", self.data)

    @property
    def img_shape(self):
        return (1,) + self[0].images.shape

    def transformed_mean(self, stats_source=None):
        if stats_source is None:
            stats_source = self.stats_source

        tmp = [np.atleast_1d(self.statistics[dk][stats_source]["mean"][()]) for dk in self.data_keys]
        return self.transform(self.data_point(*tmp))

    def __repr__(self):
        return super().__repr__() + (
            "\n\t[Stats source: {}]".format(self.stats_source) if self.stats_source is not None else ""
        )

    def __dir__(self):
        attrs = set(self.__dict__).union(set(dir(type(self))))
        return attrs.union(set(self.data.keys()))
