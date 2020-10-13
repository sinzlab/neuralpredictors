import json
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
from scipy.signal import convolve2d
from torch.utils.data import Dataset

from .exceptions import InconsistentDataException, DoesNotExistException
from .transforms import DataTransform, MovieTransform, StaticTransform, Invertible, Subsequence, Delay
from .utils import convert_static_h5_dataset_to_folder, zip_dir
from ..utils import recursively_load_dict_contents_from_group


class AttributeHandler:
    def __init__(self, name, h5_handle):
        """
        Can be used to turn a dataset within a hdf5 dataset into an attribute.

        Args:
            name:       name of the dataset in the hdf5 file
            h5_handle:  file handle for the hdf5 file
        """
        assert name in h5_handle, "{} must be in {}".format(name, h5_handle)
        self.name = name
        self.h5_handle = h5_handle

    def __getattr__(self, item):
        ret = self.h5_handle[self.name][item][()]
        if ret.dtype.char == "S":  # convert bytes to unicode
            ret = ret.astype(str)
        return ret

    def __getitem__(self, item):
        return getattr(self, item)

    def keys(self):
        return self.h5_handle[self.name].keys()

    def __dir__(self):
        attrs = set(super().__dir__())
        return attrs.union(set(self.h5_handle[self.name].keys()))


class AttributeTransformer(AttributeHandler):
    def __init__(self, name, h5_handle, transforms, data_group):
        """
        Allows for id_transform of transforms to be applied to the
        specified attribute. Otherwise behaves like an AttributeHandler

        Args:
            name:       see AttributeHandler
            h5_handle:  see AttributeHandler
            transforms: the set of transforms that's supposed to be applied
            data_group: the data_key of the dataset that this attribute represents
        """
        super().__init__(name, h5_handle)
        self.transforms = transforms
        self.data_group = data_group

    def __getattr__(self, item):
        ret = {self.data_group: super().__getattr__(item)}
        for tr in self.transforms:
            ret = tr.id_transform(ret)

        return ret[self.data_group]


class TransformDataset(Dataset):
    def __init__(self, transforms=None):
        """
        Abstract Class for Datasets. Classes that inherit from this class can implement the transform and invert function.

        Args:
            transforms: list of transforms applied to each data point
        """
        self.transforms = transforms or []

    def transform(self, x, exclude=None):
        for tr in self.transforms:
            if exclude is None or not isinstance(tr, exclude):
                x = tr(x)
        return x

    def invert(self, x, exclude=None):
        for tr in reversed(filter(lambda tr: not isinstance(tr, exclude), self.transforms)):
            if not isinstance(tr, Invertible):
                raise TypeError("Cannot invert", tr.__class__.__name__)
            else:
                x = tr.inv(x)
        return x

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __repr__(self):
        return (
            "{} m={}:\n\t({})".format(self.__class__.__name__, len(self), ", ".join(self.data_groups))
            + "\n\t[Transforms: "
            + "->".join([repr(tr) for tr in self.transforms])
            + "]"
        )


class H5SequenceSet(TransformDataset):
    def __init__(self, filename, *data_groups, output_rename=None, transforms=None):
        super().__init__(transforms=transforms)

        if output_rename is None:
            output_rename = {}

        # a flag that can be changed to turn renaming on/off
        self.rename_output = True

        self.output_rename = output_rename

        self._fid = h5py.File(filename, "r")
        self.data = self._fid
        self.data_loaded = False

        m = None
        for key in data_groups:
            assert key in self.data, "Could not find {} in file".format(key)
            l = len(self.data[key])
            if m is not None and l != m:
                raise ValueError("groups have different length")
            m = l
        self._len = m

        # Specify which types of transforms are accepted
        self._transform_set = DataTransform

        self.data_groups = data_groups
        self.transforms = transforms or []

        self.data_point = namedtuple("DataPoint", data_groups)
        self.output_point = namedtuple("OutputPoint", [output_rename.get(k, k) for k in data_groups])

    def load_content(self):
        self.data = recursively_load_dict_contents_from_group(self._fid)
        self.data_loaded = True

    def unload_content(self):
        self.data = self._fid
        self.data_loaded = False

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        x = self.data_point(
            *(np.array(self.data[g][item if self.data_loaded else str(item)]) for g in self.data_groups)
        )
        for tr in self.transforms:
            assert isinstance(tr, self._transform_set)
            x = tr(x)

        # convert to output point
        if self.rename_output:
            x = self.output_point(*x)
        return x

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

    def __repr__(self):
        names = [
            "{} -> {}".format(k, self.output_rename[k]) if k in self.output_rename else k for k in self.data_groups
        ]
        s = "{} m={}:\n\t({})".format(self.__class__.__name__, len(self), ", ".join(names))
        if self.transforms is not None:
            s += "\n\t[Transforms: " + "->".join([repr(tr) for tr in self.transforms]) + "]"
        return s


class MovieSet(H5SequenceSet):
    def __init__(self, filename, *data_groups, output_rename=None, transforms=None, stats_source=None):
        super().__init__(filename, *data_groups, output_rename=output_rename, transforms=transforms)
        self.shuffle_dims = {}
        self.stats_source = stats_source if stats_source is not None else "all"

        # set to accept only MovieTransform
        self._transform_set = MovieTransform

    @property
    def neurons(self):
        return AttributeTransformer("neurons", self.data, self.transforms, data_group="responses")

    @property
    def n_neurons(self):
        return len(self.neurons.unit_ids)

    @property
    def input_shape(self):
        return (1,) + getattr(self[0], self.output_rename.get("inputs", "inputs")).shape

    def transformed_mean(self, stats_source=None):
        if stats_source is None:
            stats_source = self.stats_source

        tmp = [np.atleast_1d(self.statistics[g][stats_source]["mean"][()]) for g in self.data_groups]
        return self.transform(self.data_point(*tmp), exclude=(Subsequence, Delay))

    def rf_base(self, stats_source="all"):
        N, c, t, w, h = self.img_shape
        t = min(t, 150)
        mean = lambda dk: self.statistics[dk][stats_source]["mean"][()]
        d = dict(
            inputs=np.ones((1, c, t, w, h)) * np.array(mean("inputs")),
            eye_position=np.ones((1, t, 1)) * mean("eye_position")[None, None, :],
            behavior=np.ones((1, t, 1)) * mean("behavior")[None, None, :],
            responses=np.ones((1, t, 1)) * mean("responses")[None, None, :],
        )
        return self.transform(self.data_point(*[d[dk] for dk in self.data_group]), exclude=Subsequence)

    def rf_noise_stim(self, m, t, stats_source="all"):
        """
        Generates a Gaussian white noise stimulus filtered with a 3x3 Gaussian filter
        for the computation of receptive fields. The mean and variance of the Gaussian
        noise are set to the mean and variance of the stimulus ensemble.

        The behvavior, eye movement statistics, and responses are set to their respective means.
        Args:
            m: number of noise samples
            t: length in time

        Returns: tuple of input, behavior, eye, and response

        """
        N, c, _, w, h = self.img_shape
        stat = lambda dk, what: self.statistics[dk][stats_source][what][()]
        mu, s = stat("inputs", "mean"), stat("inputs", "std")
        h_filt = np.float64([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
        noise_input = (
            np.stack([convolve2d(np.random.randn(w, h), h_filt, mode="same") for _ in range(m * t * c)]).reshape(
                (m, c, t, w, h)
            )
            * s
            + mu
        )

        mean_beh = np.ones((m, t, 1)) * stat("behavior", "mean")[None, None, :]
        mean_eye = np.ones((m, t, 1)) * stat("eye_position", "mean")[None, None, :]
        mean_resp = np.ones((m, t, 1)) * stat("responses", "mean")[None, None, :]

        d = dict(
            inputs=noise_input.astype(np.float32),
            eye_position=mean_eye.astype(np.float32),
            behavior=mean_beh.astype(np.float32),
            responses=mean_resp.astype(np.float32),
        )

        return self.transform(self.data_point(*[d[dk] for dk in self.data_groups.values()]), exclude=Subsequence)


default_datapoint = namedtuple("DefaultDataPoint", ["images", "responses"])


class StaticSet(TransformDataset):
    def __init__(self, *data_keys, transforms=None):
        """
        Abstract class for static datasets. Defines data_keys and a corresponding datapoint.
        """
        super().__init__(transforms=transforms)

        self.data_keys = data_keys
        if set(data_keys) == {"images", "responses"}:
            # this version IS serializable in pickle
            self.data_point = default_datapoint
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


class DirectoryAttributeHandler:
    def __init__(self, path, links=None):
        """
        Class that can be used to represent a subdirectory of a FileTree as a property in a FileTree dataset.
        Caches already loaded data items.

        Args:
            path: path to the subdiretory (pathlib.Path object)
        """
        self.links = links or {}
        self.path = path

    def __getattr__(self, item):
        temp_path = self.resolve_item_path(item)
        if temp_path.exists() and temp_path.is_dir():
            val = DirectoryAttributeHandler(temp_path, links=self.links)
        else:
            val = np.load(self.path / "{}.npy".format(item))
        return val

    def resolve_item_path(self, item):
        if item in self.links:
            item = self.links[item]
        return self.path / item

    def __getitem__(self, item):
        return getattr(self, item)

    def keys(self):
        return [e.stem for e in self.path.glob("*")]

    def __dir__(self):
        attrs = set(super().__dir__())
        return attrs.union(set(self.keys())).union(set(self.links.keys()))


class DirectoryAttributeTransformer(DirectoryAttributeHandler):
    def __init__(self, path, transforms, data_group, links=None):
        """
        Class that can be used to represent a subdirectory of a FileTree as a property in a FileTree dataset.
        Like DirectoryAttributeHandler but allows for id_transform of transforms to be applied to the
        specified attribute.

        Args:
            path: path to the subdiretory (pathlib.Path object)
        """

        super().__init__(path, links=links)
        self.transforms = transforms
        self.data_group = data_group

    def __getattr__(self, item):
        ret = {self.data_group: super().__getattr__(item)}
        for tr in self.transforms:
            ret = tr.id_transform(ret)
        return ret[self.data_group]


class FileTreeDataset(StaticSet):
    def __init__(self, dirname, *data_keys, transforms=None):
        """
        Dataset stored as a file tree. The tree needs to have the subdirs data, meta, meta/neurons, meta/statistics,
        and meta/trials. Please refer to convert_static_h5_dataset_to_folder in neuralpredictors.data.utils
        how to export an hdf5 file into that structure.


        Here is an example. Data directories with too many entries have trials as .npy files
        named 0.npy, 1.npy, ...
        The meta/trials subdirectory must have single .npy files with arrays that provide additional trial based
        meta data.

        static22564-2-13-preproc0
        ├── data
        │   ├── behavior [5955 entries exceeds filelimit, not opening dir]
        │   ├── images [5955 entries exceeds filelimit, not opening dir]
        │   ├── pupil_center [5955 entries exceeds filelimit, not opening dir]
        │   └── responses [5955 entries exceeds filelimit, not opening dir]
        └── meta
            ├── neurons
            │   ├── animal_ids.npy
            │   ├── area.npy
            │   ├── layer.npy
            │   ├── scan_idx.npy
            │   ├── sessions.npy
            │   └── unit_ids.npy
            ├── statistics
            │   ├── behavior
            │   │   ├── all
            │   │   │   ├── max.npy
            │   │   │   ├── mean.npy
            │   │   │   ├── median.npy
            │   │   │   ├── min.npy
            │   │   │   └── std.npy
            │   │   └── stimulus_frame
            │   │       ├── max.npy
            │   │       ├── mean.npy
            │   │       ├── median.npy
            │   │       ├── min.npy
            │   │       └── std.npy
            │   ├── images
            │   │   ├── all
            │   │   │   ├── max.npy
            │   │   │   ├── mean.npy
            │   │   │   ├── median.npy
            │   │   │   ├── min.npy
            │   │   │   └── std.npy
            │   │   └── stimulus_frame
            │   │       ├── max.npy
            │   │       ├── mean.npy
            │   │       ├── median.npy
            │   │       ├── min.npy
            │   │       └── std.npy
            │   ├── pupil_center
            │   │   ├── all
            │   │   │   ├── max.npy
            │   │   │   ├── mean.npy
            │   │   │   ├── median.npy
            │   │   │   ├── min.npy
            │   │   │   └── std.npy
            │   │   └── stimulus_frame
            │   │       ├── max.npy
            │   │       ├── mean.npy
            │   │       ├── median.npy
            │   │       ├── min.npy
            │   │       └── std.npy
            │   └── responses
            │       ├── all
            │       │   ├── max.npy
            │       │   ├── mean.npy
            │       │   ├── median.npy
            │       │   ├── min.npy
            │       │   └── std.npy
            │       └── stimulus_frame
            │           ├── max.npy
            │           ├── mean.npy
            │           ├── median.npy
            │           ├── min.npy
            │           └── std.npy
            └── trials [12 entries exceeds filelimit, not opening dir]

        Args:
            dirname:     root directory name
            *data_keys:  data items to be extraced (must be subdirectories of root/data)
            transforms:  transforms to be applied to the data (see TransformDataset)
        """
        super().__init__(*data_keys, transforms=transforms)

        number_of_files = []

        if dirname.endswith(".zip"):
            if not Path(dirname[:-4]).exists():
                self.unzip(dirname, Path(dirname).absolute().parent)
            else:
                print("{} exists already. Not unpacking {}".format(dirname[:-4], dirname))
            dirname = dirname[:-4]

        self.basepath = Path(dirname).absolute()
        self._config_file = self.basepath / "config.json"

        if not self._config_file.exists():
            self._save_config(self._default_config)

        for data_key in data_keys:
            datapath = self.resolve_data_path(data_key)
            number_of_files.append(len(list(datapath.glob("*"))))

        if not np.all(np.diff(number_of_files) == 0):
            raise InconsistentDataException("Number of data points is not equal")
        else:
            self._len = number_of_files[0]

        self._cache = {data_key: {} for data_key in data_keys}

    _default_config = {"links": {}}

    def resolve_data_path(self, data_key):
        if self.link_exists(data_key):
            data_key = self.config["links"][data_key]
        datapath = self.basepath / "data" / data_key

        if not datapath.exists():
            raise DoesNotExistException("Data path {} does not exist".format(datapath))
        return datapath

    def link_exists(self, link):
        return "links" in self.config and link in self.config["links"]

    @property
    def config(self):
        with open(self._config_file) as fid:
            return json.load(fid)

    def _save_config(self, cfg):
        with open(self._config_file, "w") as fid:
            return json.dump(cfg, fid)

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        # load data from cache or disk
        ret = []
        for data_key in self.data_keys:
            if item in self._cache[data_key]:
                ret.append(self._cache[data_key][item])
            else:
                datapath = self.resolve_data_path(data_key)
                val = np.load(datapath / "{}.npy".format(item))
                self._cache[data_key][item] = val
                ret.append(val)

        # create data point and transform
        x = self.data_point(*ret)
        for tr in self.transforms:
            assert isinstance(tr, StaticTransform)
            x = tr(x)
        return x

    def add_log_entry(self, msg):
        timestamp = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
        with open(self.basepath / "change.log", "a+") as fid:
            fid.write("{}: {}\n".format(timestamp, msg))

    @staticmethod
    def match_order(target, permuted, not_exist_ok=False):
        """
        Matches the order or rows in permuted to by returning an index array such that.

        Args:
            not_exist_ok: if the element does not exist, don't return an index

        Returns: index array `idx` such that `target == permuted[idx, :]`
        """

        order, target_idx = [], []
        unmatched_counter = 0
        for i, row in enumerate(target):
            idx = np.sum(permuted - row, axis=1) == 0
            if not not_exist_ok:
                assert idx.sum() == 1
            if idx.sum() == 1:
                order.append(np.where(idx)[0][0])
                target_idx.append(i)
            else:
                unmatched_counter += 1
        if not_exist_ok:
            print("Encountered {} unmatched elements".format(unmatched_counter))
        return np.array(target_idx, dtype=int), np.array(order, dtype=int)

    def add_neuron_meta(self, name, animal_id, session, scan_idx, unit_id, values, fill_missing=None):
        """
        Add new meta information about neurons.

        Args:
            name:       name of the new meta information
            animal_id:  array with animal_ids per first dimension of values
            session:    array with session per first dimension of values
            scan_idx:   array with scan_idx per first dimension of values
            unit_id:    array with unit_id per first dimension of values
            values:     new meta information. First dimension must refer to neurons.
            fill_missing:   fill the values of the new attribute with NaN if not provided
        """
        if not len(animal_id) == len(session) == len(scan_idx) == len(unit_id) == len(values):
            raise InconsistentDataException("number of trials and identifiers not consistent")

        target = np.c_[(self.neurons.animal_ids, self.neurons.sessions, self.neurons.scan_idx, self.neurons.unit_ids)]
        permuted = np.c_[(animal_id, session, scan_idx, unit_id)]
        vals = np.ones((len(target),) + values.shape[1:], dtype=values.dtype) * (
            np.nan if fill_missing is None else fill_missing
        )
        tidx, idx = self.match_order(target, permuted, not_exist_ok=fill_missing is not None)

        assert np.sum(target[tidx] - permuted[idx, ...]) == 0, "Something went wrong in sorting"

        vals[tidx, ...] = values[idx, ...]
        np.save(self.basepath / "meta/neurons/{}.npy".format(name), vals)
        self.add_log_entry("Added new neuron meta attribute {} to meta/neurons".format(name))

    @staticmethod
    def initialize_from(filename, outpath=None, overwrite=False):
        """
        Convenience function. See `convert_static_h5_dataset_to_folder` in `.utils`
        """
        convert_static_h5_dataset_to_folder(filename, outpath=outpath, overwrite=overwrite)

    @property
    def change_log(self):
        if (self.basepath / "change.log").exists():
            with open(self.basepath / "change.log", "r") as fid:
                print("".join(fid.readlines()))

    def zip(self, filename=None):
        """
        Zips current dataset.

        Args:
            filename:  Filename for the zip. Directory name + zip by default.
        """

        if filename is None:
            filename = str(self.basepath) + ".zip"
        zip_dir(filename, self.basepath)

    def unzip(self, filename, path):
        print("Unzipping {} into {}".format(filename, path))
        with ZipFile(filename, "r") as zip_obj:
            zip_obj.extractall(path)

    def add_link(self, attr, new_name):
        """
        Add a new dataset that links to an existing dataset.

        For instance `targets` that links to `responses`

        Args:
            attr:       existing attribute such as `responses`
            new_name:   name of the new attribute reference.
        """
        if not (self.basepath / "data/{}".format(attr)).exists():
            raise DoesNotExistException("Link target does not exist")

        if (self.basepath / "data/{}".format(new_name)).exists():
            raise FileExistsError("Link target already exists")

        config = self.config
        if not "links" in config:
            config["links"] = {}
        config["links"][new_name] = attr
        self._save_config(config)

    @property
    def n_neurons(self):
        return len(self[0].responses)

    @property
    def neurons(self):
        return DirectoryAttributeTransformer(
            self.basepath / "meta/neurons",
            self.transforms,
            data_group="responses" if "responses" in self.data_keys else "targets",
        )

    @property
    def trial_info(self):
        return DirectoryAttributeHandler(self.basepath / "meta/trials")

    @property
    def statistics(self):
        return DirectoryAttributeHandler(self.basepath / "meta/statistics", self.config["links"])

    @property
    def img_shape(self):
        return (1,) + self[0].images.shape

    def __repr__(self):
        return "{} {} (n={} items)\n\t{}".format(
            self.__class__.__name__, self.basepath, self._len, ", ".join(self.data_keys)
        )
