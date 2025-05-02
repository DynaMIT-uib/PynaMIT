"""IO Class.

This module provides the IO class for handling input and output
operations. It includes methods for saving and loading datasets in
NetCDF format.
"""

import os
import xarray as xr


class IO:
    """Class for handling input and output operations.

    This class provides methods to save and load datasets in NetCDF
    format.
    """

    def __init__(self, dataset_filename_prefix):
        """Initialize the IO class.

        Parameters
        ----------
        dataset_filename_prefix : str, optional
            Prefix for the dataset filenames. If None, no prefix is
            used.
        """
        self.dataset_filename_prefix = dataset_filename_prefix

    def update_dataset_filename_prefix(self, dataset_filename_prefix):
        """Update the prefix for the dataset filenames.

        Parameters
        ----------
        dataset_filename_prefix : str
            Prefix for the dataset filenames.
        """
        self.dataset_filename_prefix = dataset_filename_prefix

    def save_dataset(self, dataset, name):
        """Save a dataset to NetCDF file.

        Parameters
        ----------
        dataset : xarray.Dataset or xarray.DataArray
            The dataset to save.
        name : str
            Name to use in the filename.
        """
        if self.dataset_filename_prefix is None:
            raise ValueError("dataset_filename_prefix is None. Cannot save dataset.")

        filename = self.dataset_filename_prefix + "_" + name + ".ncdf"

        try:
            dataset.to_netcdf(filename + ".tmp")
            os.rename(filename + ".tmp", filename)

        except Exception as e:
            if os.path.exists(filename + ".tmp"):
                os.remove(filename + ".tmp")
            raise e

    def load_dataset(self, name):
        """Load a dataset from NetCDF file.

        Parameters
        ----------
        name : str
            Name of the dataset.

        Returns
        -------
        xarray.Dataset or None
            Loaded dataset, or None if the file does not exist.
        """
        if self.dataset_filename_prefix is not None:
            filename = self.dataset_filename_prefix + "_" + name + ".ncdf"

            if os.path.exists(filename):
                return xr.load_dataset(filename)
            else:
                return None
