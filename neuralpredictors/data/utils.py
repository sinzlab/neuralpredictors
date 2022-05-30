import logging
from collections import Mapping
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import h5py
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def zip_dir(zip_name: str, source_dir):
    """
    Zips all files in `source_dir` into a zip file named `zip_name`.
    """
    src_path = Path(source_dir).absolute().resolve(strict=True)
    with ZipFile(zip_name, "w", ZIP_DEFLATED) as zf:
        for file in tqdm(src_path.rglob("*")):
            zf.write(file, file.relative_to(src_path.parent))


def _savenpy(path, val, overwrite):
    if not np.isscalar(val) and len(val.shape) > 0 and val[0].dtype.char == "S":  # convert bytes to unicode
        val = val.astype(str)
    if not path.exists() or overwrite:
        logger.info("Saving", path)
        np.save(path, val)
    else:
        logger.warn("Not overwriting", path)


def convert_movies_h5_dataset_to_folder(
    filename,
    outpath=None,
    overwrite=False,
    data_keys=("inputs", "responses", "behavior", "eye_position"),
):
    """
    Converts an HDF5 dataset used for mouse movie data into a directory structure
    that can be used by the FileTreeDataset.

    Args:
        filename:       filename of the hdf5 file
        outpath:        location of the FileTreeDataset (default .)
        overwrite:      overwrite existing files
    """
    if not isinstance(data_keys, Mapping):
        data_keys = {k: k for k in data_keys}

    h5file = Path(filename)
    outpath = Path(outpath) or h5file.with_suffix("")  # drop extension to form target output

    with h5py.File(filename, "r") as fid:
        for data_key, mapped_name in data_keys.items():
            subpath = outpath / "data/{}".format(mapped_name)
            subpath.mkdir(exist_ok=True, parents=True)
            for key in tqdm(
                fid[data_key].keys(),
                desc="Saving {}".format(data_key),
            ):
                outfile = subpath / "{}.npy".format(key)
                _savenpy(outfile, fid[data_key][key][()], overwrite)

        # map data for trials
        trial_info_keys = [
            "durations",
            "condition_hashes",
            "movie_names",
            "tiers",
            "trial_idx",
            "types",
        ]

        # save tiers
        subpath = outpath / "meta/trials"
        subpath.mkdir(exist_ok=True, parents=True)
        for trial_info_key in trial_info_keys:
            _savenpy(
                subpath / "{}.npy".format(trial_info_key),
                fid[trial_info_key][()],
                overwrite,
            )

        # save nested fields:
        nested_keys = {"neurons": "neurons"}
        for meta_type, target in nested_keys.items():
            for meta_key in fid[meta_type].keys():
                subpath = outpath / "meta/{}".format(target)
                subpath.mkdir(exist_ok=True, parents=True)

                _savenpy(
                    subpath / "{}.npy".format(meta_key),
                    fid[meta_type][meta_key][()],
                    overwrite,
                )

        # save statistics
        def statistics_func(name, node):
            name = name.replace(".", "_").lower()
            if isinstance(node, h5py.Dataset):
                data_file = outpath / "meta/statistics/{}.npy".format(name)
                data_file.parent.mkdir(exist_ok=True, parents=True)
                _savenpy(
                    data_file,
                    node[()],
                    overwrite,
                )

        logger.info("Saving statistics")
        fid["statistics"].visititems(statistics_func)


def convert_static_h5_dataset_to_folder(filename, outpath=None, overwrite=False, ignore_all_behaviors=False):
    """
    Converts a h5 dataset used for mouse data into a directory structure that can be used by the FileTreeDataset.

    Args:
        filename:       filename of the hdf5 file
        outpath:        location of the FileTreeDataset (default .)
        overwrite:      overwrite existing files

    """
    h5file = Path(filename)
    outpath = outpath or (h5file.parent / h5file.stem)

    with h5py.File(filename) as fid:
        attributes = (
            ["images", "responses", "behavior", "pupil_center"] if not ignore_all_behaviors else ["images", "responses"]
        )
        for data_key in attributes:
            subpath = outpath / "data/{}".format(data_key)
            subpath.mkdir(exist_ok=True, parents=True)
            for i, value in tqdm(
                enumerate(fid[data_key]),
                total=fid[data_key].shape[0],
                desc="Saving {}".format(data_key),
            ):
                outfile = subpath / "{}.npy".format(i)
                if not outfile.exists() or overwrite:
                    np.save(outfile, value)

        # save tiers
        for data_key in ["tiers"]:
            subpath = outpath / "meta/trials"
            subpath.mkdir(exist_ok=True, parents=True)
            _savenpy(subpath / "{}.npy".format(data_key), fid[data_key][...], overwrite)

        # save meta info
        for meta_type, target in zip(["item_info", "neurons"], ["trials", "neurons"]):
            for meta_key in fid[meta_type].keys():
                subpath = outpath / "meta/{}".format(target)
                subpath.mkdir(exist_ok=True, parents=True)

                _savenpy(
                    subpath / "{}.npy".format(meta_key),
                    fid[meta_type][meta_key][...],
                    overwrite,
                )

        # save statistics
        def statistics_func(name, node):
            name = name.replace(".", "_").lower()
            if not isinstance(node, h5py.Dataset):
                subpath = outpath / "meta/statistics" / name
                subpath.mkdir(exist_ok=True, parents=True)
            else:
                _savenpy(
                    outpath / "meta/statistics" / "{}.npy".format(name),
                    node[...],
                    overwrite,
                )

        fid["statistics"].visititems(statistics_func)


def load_dict_from_hdf5(filename):
    """
    Given a `filename` of a HDF5 file, opens the file and
    load the entire content as a (nested) dictionary.

    Args:
        filename - name of HDF5 file

    Returns:
        (nested) dictionary corresponding to the content of the HDF5 file.
    """
    with h5py.File(filename, "r") as h5file:
        return recursively_load_dict_contents_from_group(h5file)


def recursively_load_dict_contents_from_group(h5file, path="/"):
    """
    Given a `h5file` h5py object, loads the object at `path`
    as nested dictionary.

    Args:
        h5file - h5py object
        path - Path within the h5py file to load the content of recursively.

    Returns:
        (nested) dictionary corresponding to the content of the HDF5 file at the path.
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            dtype = item.dtype
            v = item[()]
            if dtype.char == "S":  # convert bytes to univcode
                v = v.astype(str)
            ans[key] = v
        elif isinstance(item, h5py.Group):
            if item.attrs.get("_iterable", False):
                ans[key] = [item[str(i)][()] for i in range(len(item))]
            else:
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + "/")
    return ans
