import json
import logging
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from torch.utils.data import Dataset

from ...utils import no_transforms
from ..exceptions import DoesNotExistException, InconsistentDataException
from ..transforms import DataTransform, Invertible
from ..utils import zip_dir

logger = logging.getLogger(__name__)


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
        if item in self.h5_handle[self.name]:
            ret = self.h5_handle[self.name][item][()]
            if ret.dtype.char == "S":  # convert bytes to unicode
                ret = ret.astype(str)
            return ret
        else:
            raise AttributeError("Attribute {} not found".format(item))

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
            if hasattr(tr, "id_transform"):
                ret = tr.id_transform(ret)

        return ret[self.data_group]


default_image_datapoint = namedtuple("DefaultDataPoint", ["images", "responses"])


class TransformDataset(Dataset):
    def __init__(self, *data_keys, transforms=None):
        """
        Abstract Class for Datasets with transformations, providing `transform` and `invert` functions
        to apply data transformation on the elements.
        Args:
            transforms: list of transforms to be applied to each data point
        """
        self.transforms = transforms or []

        self.data_keys = data_keys
        if set(data_keys) == {"images", "responses"}:
            # this version IS serializable in pickle
            self.data_point = default_image_datapoint
        else:
            # this version is NOT - you cannot use this with a dataloader with num_workers > 1
            self.data_point = namedtuple("DataPoint", data_keys)

    def transform(self, x, exclude=None):
        """
        Apply transform on a data element from the dataset
        Args:
            x (tuple): a data element from the dataset
            exclude (Transform, optional): Type of data transformer to be excluded from transform list. Defaults to None.
        Returns:
            tuple: transformed data element
        """

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
            "{} m={}:\n\t({})".format(self.__class__.__name__, len(self), ", ".join(self.data_keys))
            + "\n\t[Transforms: "
            + "->".join([repr(tr) for tr in self.transforms])
            + "]"
        )


class DirectoryAttributeHandler:
    def __init__(self, path, links=None):
        """
        Class that can be used to represent a subdirectory of a FileTree as a property in a FileTree dataset.
        Args:
            path (pathlib.Path object): path to the subdiretory
            links (dict, optional): rename mapping for entries within the `path`. Defaults to None, in which case
                no name mapping is performed when fetching the attribute.
        """
        self.links = links or {}
        self.path = path

    def __getattr__(self, item):
        item_path = self.resolve_item_path(item)

        if item_path.exists() and item_path.is_dir():
            val = DirectoryAttributeHandler(item_path, links=self.links)
        else:
            data_path = item_path.with_suffix(".npy")
            if data_path.exists() and data_path.is_file():
                val = np.load(data_path)
            else:
                raise AttributeError("Attribute {} not found".format(item))
        return val

    def resolve_item_path(self, item):
        """
        Formulates a path to the `item`, taken relative to the
        `path` attribute of this object. If the `item` has an entry in
        `links` dictionary, then the name is mapped into that instead.

        Args:
            item (str): Name of item to obtain.

        Returns:
            pathlib.Path object: Formulated full path to the target item
        """
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
            path (pathlib.Path object): path to the subdiretory
            transforms (list): A list of DataTransform objects, whose `id_transform` will be applied to the loaded property sequentially
            data_group (str): Name of data_group that the transforms should be applied as
            links (dict, optional): rename mapping for entries within the `path`. Defaults to None, in which case
                no name mapping is performed when fetching the attribute.
        """

        super().__init__(path, links=links)
        self.transforms = transforms
        self.data_group = data_group

    def __getattr__(self, item):
        ret = {self.data_group: super().__getattr__(item)}
        for tr in self.transforms:
            ret = tr.id_transform(ret)
        return ret[self.data_group]


class FileTreeDatasetBase(TransformDataset):
    _default_config = {"links": {}}
    # specify list of transform types that are acceptable
    _transform_types = (DataTransform,)

    def __init__(self, dirname, *data_keys, transforms=None, use_cache=True, output_rename=None, output_dict=False):
        """
        Dataset stored as a file tree. The tree needs to have the subdirs data, meta, meta/neurons, meta/statistics,
        and meta/trials. Please refer to convert_static_h5_dataset_to_folder in neuralpredictors.data.utils for an
        example on how to export an hdf5 file into folder structure compatible with this dataset.


        Here is an example. Data directories with too many entries have trials as .npy files
        named 0.npy, 1.npy, ...
        The meta/trials subdirectory must have single .npy files with arrays that provide additional trial based
        meta data.
        static22564-2-13-preproc0
        ├── data
        │   ├── behavior [directory with 5955 entries]
        │   ├── images [directory with 5955 entries]
        │   ├── pupil_center [5955 entries]
        │   └── responses [5955 entries]
        └── meta
            ├── neurons
            │   ├── animal_ids.npy
            │   ├── area.npy
            │   ├── layer.npy
            │   ├── scan_idx.npy
            │   ├── sessions.npy
            │   └── unit_ids.npy
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
            └── trials [12 entries]

        Args:
            dirname:     root directory name
            *data_keys:  data items to be extraced (must be subdirectories of root/data)
            transforms:  transforms to be applied to the data (see TransformDataset)
        """
        super().__init__(*data_keys, transforms=transforms)

        number_of_files = []

        self.output_dict = output_dict
        self.use_cache = use_cache

        if output_rename is None:
            output_rename = {}

        # a flag that can be changed to turn renaming on/off
        self.rename_output = bool(output_rename)
        self._output_rename = output_rename
        renamed_keys = [output_rename.get(k, k) for k in data_keys]
        self._output_point = namedtuple("OutputPoint", renamed_keys) if output_rename else self.data_point

        # if dirname is a zip file, auto expand into the container folder
        if dirname.endswith(".zip"):
            if not Path(dirname[:-4]).exists():
                self.unzip(dirname, Path(dirname).absolute().parent)
            else:
                logger.info(f"{dirname[:-4]} exists already. Not unpacking {dirname}")
            dirname = dirname[:-4]

        self.dirname = dirname
        self.basepath = Path(dirname).absolute()
        self._config_file = self.basepath / "config.json"

        # if no config file, create one based on default config
        if not self._config_file.exists():
            self._save_config(self._default_config)

        # verify that valid data path exists for each data_key and count number of files
        for data_key in data_keys:
            if data_key not in self.trial_info.keys():
                datapath = self.resolve_data_path(data_key)
                number_of_files.append(len(list(datapath.glob("*"))))
            else:
                number_of_files.append(len(self.trial_info[data_key]))

        # verify that all data_keys directories contain identical number of files (= number of data points)
        if not np.all(np.diff(number_of_files) == 0):
            raise InconsistentDataException("Number of data points is not equal")
        # keep the number of files as the number of data points
        self._len = number_of_files[0]

        self._cache = {data_key: {} for data_key in data_keys}

    def resolve_data_path(self, data_key):
        """
        Given a data_key, resolves the folder within self.basepath/data directory. If relevant "links"
        entry exists in the config, the name mapping is performed. Finally, the resultant path is checked
        for validness (is it a directory that exists) and raises an exception if not found. Otherwise, the
        resultant path object is returned.

        Args:
            data_key (str): data_key to find corresponding subdirectory under `self.basepath/data`

        Raises:
            DoesNotExistException: If the target path is not a valid directory, this exception is raised.

        Returns:
            pathlib.Path object: Valid directory path to the target data_group
        """
        if self.link_exists(data_key):
            data_key = self.config["links"][data_key]
        datapath = self.basepath / "data" / data_key

        if not datapath.exists() or not datapath.is_dir():
            raise DoesNotExistException("Data path {} is not a valid directory".format(datapath))
        return datapath

    @staticmethod
    def unzip(filename, path):
        """
        Unzips the target file with `filename` into the specified `path`

        Args:
            filename (str): Path to the zip file
            path (str): Path to expand the zip content into
        """
        logger.info(f"Unzipping {filename} into {path}")
        with ZipFile(filename, "r") as zip_obj:
            zip_obj.extractall(path)

    def link_exists(self, link):
        """
        Checks if an entry for `link` exists in the "links" config

        Args:
            link (str): data_group name to check for an entry in "links"

        Returns:
            bool: True if a relevant entry is found in "links" config
        """
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
            if self.use_cache and item in self._cache[data_key]:
                ret.append(self._cache[data_key][item])
            else:
                if data_key in self.trial_info.keys():
                    val = self.trial_info[data_key][item : item + 1]
                else:
                    datapath = self.resolve_data_path(data_key)
                    val = np.load(datapath / "{}.npy".format(item))
                if self.use_cache:
                    self._cache[data_key][item] = val
                ret.append(val)

        # create data point and transform
        x = self.data_point(*ret)

        for tr in self.transforms:
            # ensure only specified types of transforms are used
            assert isinstance(tr, self._transform_types)
            x = tr(x)

        # apply output rename if necessary
        if self.rename_output:
            x = self._output_point(*x)

        if self.output_dict:
            x = x._asdict()

        return x

    def add_log_entry(self, msg):
        """
        Add a new log entry `msg` into the "change.log" file. The message will be timestamped

        Args:
            msg (str): Message to be logged
        """
        timestamp = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
        with open(self.basepath / "change.log", "a+") as fid:
            fid.write("{}: {}\n".format(timestamp, msg))

    @property
    def change_log(self):
        """
        Convenience property to print the content of change.log file if it exists
        """
        if (self.basepath / "change.log").exists():
            with open(self.basepath / "change.log", "r") as fid:
                logger.info("".join(fid.readlines()))

    def zip(self, filename=None):
        """
        Zips current dataset.

        Args:
            filename:  Filename for the zip. Directory name + zip by default.
        """

        if filename is None:
            filename = str(self.basepath) + ".zip"
        zip_dir(filename, self.basepath)

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
            logger.warning(f"Encountered {unmatched_counter} unmatched elements")
        return np.array(target_idx, dtype=int), np.array(order, dtype=int)

    def add_neuron_meta(self, name, animal_id, session, scan_idx, unit_id, values, fill_missing=None):
        """
        Add new meta information about neurons.
        Args:
            name:       name of the new meta information
            animal_id:  array with animal_ids matching the size of the first dimension of values
            session:    array with session matching the size of the first dimension of values
            scan_idx:   array with scan_idx matching the size of the first dimension of values
            unit_id:    array with unit_id matching the size of the first dimension of values
            values:     new meta information. First dimension must correspond to neurons.
            fill_missing: fill the values of the new attribute with the specified value. Defaults to None,
                in which case missing values are filled with NaN
        """

        with no_transforms(self):

            if not self.n_neurons == len(values):
                raise InconsistentDataException(
                    f"Number of values ({len(values)}) and neurons in the datasets ({self.n_neurons}) is not consistent."
                )

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

    def __repr__(self):
        return "{} {} (n={} items)\n\t{}".format(
            self.__class__.__name__, self.basepath, self._len, ", ".join(self.data_keys)
        )
