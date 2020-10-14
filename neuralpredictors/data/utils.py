from pathlib import Path
import h5py as h5
import numpy as np
from tqdm import tqdm

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def zip_dir(zip_name: str, source_dir):
    """
    Zips all files in `source_dir` into a zip file names `zip_name`.
    """
    src_path = Path(source_dir).absolute().resolve(strict=True)
    with ZipFile(zip_name, "w", ZIP_DEFLATED) as zf:
        for file in tqdm(src_path.rglob("*")):
            zf.write(file, file.relative_to(src_path.parent))


def _savenpy(path, val, overwrite):

    if not np.isscalar(val) and val[0].dtype.char == "S":  # convert bytes to univcode
        val = val.astype(str)
    if not path.exists() or overwrite:
        print("Saving", path)
        np.save(path, val)
    else:
        print("Not overwriting", path)


def convert_static_h5_dataset_to_folder(filename, outpath=None, overwrite=False):
    """
    Converts a h5 dataset used for mouse data into a directory structure that can be used by the FileTreeDataset.

    Args:
        filename:       filename of the hdf5 file
        outpath:        location of the FileTreeDataset (default .)
        overwrite:      overwrite existing files

    """
    h5file = Path(filename)
    outpath = outpath or (h5file.parent / h5file.stem)

    with h5.File(filename) as fid:
        for data_key in ["images", "responses", "behavior", "pupil_center"]:
            subpath = outpath / "data/{}".format(data_key)
            subpath.mkdir(exist_ok=True, parents=True)
            for i, value in tqdm(
                enumerate(fid[data_key]), total=fid[data_key].shape[0], desc="Saving {}".format(data_key)
            ):
                outfile = subpath / "{}.npy".format(i)
                if not outfile.exists() or overwrite:
                    np.save(outfile, value)

        # save tiers
        for data_key in ["tiers"]:
            subpath = outpath / "meta/trials"
            subpath.mkdir(exist_ok=True, parents=True)
            _savenpy(subpath / "{}.npy".format(data_key), fid[data_key].value, overwrite)

        # save meta info
        for meta_type, target in zip(["item_info", "neurons"], ["trials", "neurons"]):
            for meta_key in fid[meta_type].keys():
                subpath = outpath / "meta/{}".format(target)
                subpath.mkdir(exist_ok=True, parents=True)

                _savenpy(subpath / "{}.npy".format(meta_key), fid[meta_type][meta_key].value, overwrite)

        # save statistics
        def statistics_func(name, node):
            name = name.replace(".", "_").lower()
            if not isinstance(node, h5.Dataset):
                subpath = outpath / "meta/statistics" / name
                subpath.mkdir(exist_ok=True, parents=True)
            else:
                _savenpy(outpath / "meta/statistics" / "{}.npy".format(name), node.value, overwrite)

        fid["statistics"].visititems(statistics_func)
