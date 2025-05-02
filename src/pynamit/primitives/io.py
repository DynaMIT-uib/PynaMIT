"""IO Class.

This module provides the IO class for handling input and output
operations. It includes methods for saving and loading data in
NetCDF format using xarray.
"""

import os
import xarray as xr


class IO:
    """Class for handling input and output operations.

    This class provides methods to save and load data in NetCDF
    format using xarray.
    """

    def __init__(self, filename_prefix):
        """Initialize the IO class.

        Parameters
        ----------
        filename_prefix : str, optional
            Prefix for the filenames. If None, files are not loaded and
            cannot be saved.
        """
        self.filename_prefix = filename_prefix

    def update_filename_prefix(self, filename_prefix):
        """Update the prefix for the filenames.

        Parameters
        ----------
        filename_prefix : str
            Prefix for the filenames.
        """
        self.filename_prefix = filename_prefix

    def save_dataset(self, dataset, name, print_info=False):
        """Save a Dataset to NetCDF file.

        Parameters
        ----------
        dataset : xarray.Dataset
            The Dataset to save.
        name : str
            Name to use in the filename.
        """
        if self.filename_prefix is None:
            raise ValueError("filename_prefix is None. Cannot save Dataset.")

        filename = self.filename_prefix + "_" + name + ".ncdf"

        try:
            dataset.to_netcdf(filename + ".tmp")
            os.rename(filename + ".tmp", filename)

        except Exception as e:
            if os.path.exists(filename + ".tmp"):
                os.remove(filename + ".tmp")
            raise e

        if print_info:
            print(f"Saved Dataset to {filename}", flush=True)

    def load_dataset(self, name, print_info=False):
        """Load a Dataset from NetCDF file.

        Parameters
        ----------
        name : str
            Name of the Dataset.

        Returns
        -------
        xarray.Dataset or None
            Loaded Dataset, or None if the file does not exist.
        """
        if self.filename_prefix is not None:
            filename = self.filename_prefix + "_" + name + ".ncdf"

            if os.path.exists(filename):
                if print_info:
                    print(f"Loading Dataset from {filename}", flush=True)
                return xr.load_dataset(filename)

        return None

    def load_dataarray(self, name, print_info=False):
        """Load a DataArray from NetCDF file.

        Parameters
        ----------
        name : str
            Name of the DataArray.

        Returns
        -------
        xarray.DataArray or None
            Loaded DataArray, or None if the file does not exist.
        """
        if self.filename_prefix is not None:
            filename = self.filename_prefix + "_" + name + ".ncdf"

            if os.path.exists(filename):
                if print_info:
                    print(f"Loading DataArray from {filename}", flush=True)
                return xr.load_dataarray(filename)

        return None

    def save_dataarray(self, dataarray, name, print_info=False):
        """Save a DataArray to NetCDF file.

        Parameters
        ----------
        dataarray : xarray.DataArray
            The DataArray to save.
        name : str
            Name to use in the filename.
        """
        if self.filename_prefix is None:
            raise ValueError("filename_prefix is None. Cannot save DataArray.")

        filename = self.filename_prefix + "_" + name + ".ncdf"

        try:
            dataarray.to_netcdf(filename + ".tmp")
            os.rename(filename + ".tmp", filename)

        except Exception as e:
            if os.path.exists(filename + ".tmp"):
                os.remove(filename + ".tmp")
            raise e

        if print_info:
            print(f"Saved DataArray to {filename}", flush=True)
