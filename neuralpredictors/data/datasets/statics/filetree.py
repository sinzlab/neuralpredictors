import logging
from zipfile import ZipFile

from ...exceptions import DoesNotExistException
from ...transforms import StaticTransform
from ...utils import convert_static_h5_dataset_to_folder, zip_dir
from ..base import FileTreeDatasetBase

logger = logging.getLogger(__name__)


class FileTreeDataset(FileTreeDatasetBase):
    _transform_types = (StaticTransform,)

    @staticmethod
    def initialize_from(filename, outpath=None, overwrite=False):
        """
        Convenience function. See `convert_static_h5_dataset_to_folder` in `.utils`
        """
        convert_static_h5_dataset_to_folder(filename, outpath=outpath, overwrite=overwrite)

    @property
    def img_shape(self):
        return (1,) + self[0].images.shape

    @property
    def n_neurons(self):
        target_group = "responses" if "responses" in self.data_keys else "targets"
        val = self[0]
        if hasattr(val, target_group):
            val = getattr(val, target_group)
        else:
            val = val[target_group]
        return len(val)

    def change_log(self):
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

    def unzip(self, filename, path):
        logger.info(f"Unzipping {filename} into {path}")
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
