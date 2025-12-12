"""Timeseries Class.

This module contains the Timeseries class, which is responsible for
handling input and output operations in the simulation. It manages
the reading and writing of datasets, including time series data,
and provides methods for setting input data and selecting data for
the simulation.
"""

import numpy as np
import pandas as pd
import xarray as xr

FLOAT_ERROR_MARGIN = 1e-6  # Safety margin for floating point errors


class Timeseries:
    """Timeseries Class.

    Class for handling input and output operations in the simulation.
    This class manages the reading and writing of datasets, including
    time series data, and provides methods for setting input data and
    selecting data for the simulation.
    """

    def __init__(self, storage_bases, vars):
        """Initialize the TimeSeries class.

        Parameters
        ----------
        storage_bases : dict
            Dictionary of basis objects for storage.
        vars : dict
            Dictionary defining the variable structure.
        """
        self.storage_bases = storage_bases

        # Initialize variables and timeseries storage
        self.vars = vars

        self.datasets = {}

        self.basis_multiindices = {}
        for key in self.vars.keys():
            if all(self.vars[key][var] == "scalar" for var in self.vars[key]):
                self.basis_multiindices[key] = pd.MultiIndex.from_arrays(
                    self.storage_bases[key].index_arrays, names=self.storage_bases[key].index_names
                )
            elif all(self.vars[key][var] == "tangential" for var in self.vars[key]):
                self.basis_multiindices[key] = pd.MultiIndex.from_arrays(
                    [
                        np.tile(self.storage_bases[key].index_arrays[i], 2)
                        for i in range(len(self.storage_bases[key].index_arrays))
                    ],
                    names=self.storage_bases[key].index_names,
                )
            else:
                raise ValueError(
                    "Mixed scalar and tangential input (unsupported), or invalid input type"
                )

    def load_all(self, io):
        """Load all timeseries from NetCDF files."""
        for key in self.vars.keys():
            self.load(key, io)

    def load(self, key, io):
        """Load a timeseries from NetCDF file.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to load.
        """
        dataset = io.load_dataset(key)

        if dataset is not None:
            basis_multiindex = pd.MultiIndex.from_arrays(
                [
                    dataset[self.storage_bases[key].index_names[i]].values
                    for i in range(len(self.storage_bases[key].index_names))
                ],
                names=self.storage_bases[key].index_names,
            )
            coords = xr.Coordinates.from_pandas_multiindex(basis_multiindex, dim="i").merge(
                {"time": dataset.time.values}
            )
            self.datasets[key] = dataset.drop_vars(
                self.storage_bases[key].index_names
            ).assign_coords(coords)

    def add_entry(self, key, data, time):
        """Add entry to the timeseries.

        Creates a new timeseries if one does not exist, otherwise
        concatenates the new data along the time dimension.

        Parameters
        ----------
        key : {'jr', 'conductance', 'u', 'state', 'steady_state'}
            The key identifying the type of data.
        data : dict
            Dictionary of variables to set.
        time : float
            The time point for the data.
        """
        data_vars = {}
        for var in data:
            data_vars[self.storage_bases[key].kind + "_" + var] = (
                ["time", "i"],
                data[var].reshape((1, -1)),
            )

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=xr.Coordinates.from_pandas_multiindex(
                self.basis_multiindices[key], dim="i"
            ).merge({"time": [time]}),
        )

        if key not in self.datasets.keys():
            self.datasets[key] = dataset.sortby("time")
        else:
            self.datasets[key] = xr.concat(
                [self.datasets[key].drop_sel(time=dataset.time, errors="ignore"), dataset],
                dim="time",
            ).sortby("time")



    def get_entry(self, key, time, interpolation=False):
        """Select time series data corresponding to the specified time.

        Parameters
        ----------
        key : str
            Key for the time series.
        time : float
            Current time for which to select data.
        interpolation : bool, optional
            Whether to use linear interpolation.

        Returns
        -------
        dict or None
            Dictionary containing the latest data for the specified
            key, or None if no data is available.
        """
        if np.any(self.datasets[key].time.values <= time + FLOAT_ERROR_MARGIN):
            current_data = {}

            # Select latest data before the current time.
            dataset_before = self.datasets[key].sel(
                time=[time + FLOAT_ERROR_MARGIN], method="ffill"
            )

            for var in self.vars[key]:
                current_data[var] = dataset_before[
                    self.storage_bases[key].kind + "_" + var
                ].values.flatten()

            # If requested, add linear interpolation correction.
            if interpolation and np.any(
                self.datasets[key].time.values > time + FLOAT_ERROR_MARGIN
            ):
                dataset_after = self.datasets[key].sel(
                    time=[time + FLOAT_ERROR_MARGIN], method="bfill"
                )
                for var in self.vars[key]:
                    current_data[var] += (
                        (time - dataset_before.time.item())
                        / (dataset_after.time.item() - dataset_before.time.item())
                        * (
                            dataset_after[
                                self.storage_bases[key].kind + "_" + var
                            ].values.flatten()
                            - dataset_before[
                                self.storage_bases[key].kind + "_" + var
                            ].values.flatten()
                        )
                    )

            return current_data
        else:
            # No data available for the specified time.
            return None

    def save(self, key, io):
        """Save a timeseries to NetCDF file.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to save.
        """
        io.save_dataset(self.datasets[key].reset_index("i"), key)
