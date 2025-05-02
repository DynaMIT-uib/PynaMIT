"""Timeseries Class.

This module contains the Timeseries class, which is responsible for
handling input and output operations in the simulation. It manages
the reading and writing of datasets, including time series data,
and provides methods for setting input data and selecting data for
the simulation.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.grid import Grid
from pynamit.primitives.field_expansion import FieldExpansion

FLOAT_ERROR_MARGIN = 1e-6  # Safety margin for floating point errors


class Timeseries:
    """Timeseries Class.

    Class for handling input and output operations in the simulation.
    This class manages the reading and writing of datasets, including
    time series data, and provides methods for setting input data and
    selecting data for the simulation.
    """

    def __init__(self, bases, state_grid, cs_basis, vars, vector_storage):
        """Initialize the Timeseries class.

        Parameters
        ----------
        bases : dict
            Dictionary of basis objects.
        state_grid : Grid
            Grid object representing the state grid.
        cs_basis : object
            Object representing the coordinate system basis.
        dataset_filename_prefix : str
            Prefix for the dataset filenames.
        """
        self.bases = bases
        self.state_grid = state_grid
        self.cs_basis = cs_basis

        # Initialize variables and timeseries storage
        self.vars = vars
        self.vector_storage = vector_storage

        self.dataset_filename_prefix = None

        self.datasets = {}
        self.previous_data = {}

        self.basis_multiindices = {}
        for key in self.vars.keys():
            if self.vector_storage[key]:
                basis_index_arrays = self.bases[key].index_arrays
                basis_index_names = self.bases[key].index_names
            else:
                basis_index_arrays = [self.state_grid.theta, self.state_grid.phi]
                basis_index_names = ["theta", "phi"]

            if all(self.vars[key][var] == "scalar" for var in self.vars[key]):
                self.basis_multiindices[key] = pd.MultiIndex.from_arrays(
                    basis_index_arrays, names=basis_index_names
                )
            elif all(self.vars[key][var] == "tangential" for var in self.vars[key]):
                self.basis_multiindices[key] = pd.MultiIndex.from_arrays(
                    [np.tile(basis_index_arrays[i], 2) for i in range(len(basis_index_arrays))],
                    names=basis_index_names,
                )
            else:
                raise ValueError(
                    "Mixed scalar and tangential input (unsupported), or invalid input type"
                )

    def add_coeffs(self, key, data, time):
        """Set the variables for the simulation.

        Parameters
        ----------
        data : dict
            Dictionary of variables to set.
        vars : list
            List of variable names.
        """
        processed_data = {}

        time = np.atleast_1d(time)

        for time_index in range(time.size):
            processed_data = {}

            for var in self.vars[key]:
                coeff_array = np.array(
                    [np.atleast_2d(data[var][component])[time_index] for component in range(len(data[var]))]
                )
                if len(data[var]) == 1:
                    coeffs = coeff_array[0]
                else:
                    coeffs = coeff_array

                processed_data[self.bases[key].short_name + "_" + var] = (
                    ["time", "i"],
                    coeffs.reshape((1, -1)),
                )

            dataset = xr.Dataset(
                data_vars=processed_data,
                coords=xr.Coordinates.from_pandas_multiindex(
                    self.basis_multiindices[key], dim="i"
                ).merge({"time": [time[time_index]]}),
            )

            self.add_dataset(dataset, key)

    def add_input(
        self,
        key,
        input_data,
        time,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set input data for the simulation.

        Parameters
        ----------
        key : str
            The type of input data ('jr', 'conductance', or 'u').
        input_data : dict
            Dictionary containing the input data arrays.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the input data.
        weights : array-like, optional
            Weights for the input data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for pseudo-inverse.

        Raises
        ------
        ValueError
            If neither (lat, lon) nor (theta, phi) coordinates are
            provided.
        """
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)

        if not hasattr(self, "input_basis_evaluators"):
            self.input_basis_evaluators = {}

        if not (
            key in self.input_basis_evaluators.keys()
            and np.allclose(
                input_grid.theta,
                self.input_basis_evaluators[key].grid.theta,
                rtol=0.0,
                atol=FLOAT_ERROR_MARGIN,
            )
            and np.allclose(
                input_grid.phi,
                self.input_basis_evaluators[key].grid.phi,
                rtol=0.0,
                atol=FLOAT_ERROR_MARGIN,
            )
        ):
            self.input_basis_evaluators[key] = BasisEvaluator(
                self.bases[key],
                input_grid,
                weights=weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
            )

        time = np.atleast_1d(time)

        for time_index in range(time.size):
            processed_data = {}

            for var in self.vars[key]:
                grid_value_array = np.array(
                    [
                        np.atleast_2d(input_data[var][component])[time_index]
                        for component in range(len(input_data[var]))
                    ]
                )
                if self.vector_storage[key]:
                    if len(input_data[var]) == 1:
                        grid_values = grid_value_array[0]
                    else:
                        grid_values = grid_value_array
                    vector = FieldExpansion(
                        self.bases[key],
                        basis_evaluator=self.input_basis_evaluators[key],
                        grid_values=grid_values,
                        field_type=self.vars[key][var],
                    )

                    processed_data[self.bases[key].short_name + "_" + var] = (
                        ["time", "i"],
                        vector.coeffs.reshape((1, -1)),
                    )

                else:
                    # Interpolate to state_grid
                    if self.vars[key][var] == "scalar":
                        interpolated_data = self.cs_basis.interpolate_scalar(
                            grid_value_array[0],
                            input_grid.theta,
                            input_grid.phi,
                            self.state_grid.theta,
                            self.state_grid.phi,
                        )
                    elif self.vars[key][var] == "tangential":
                        interpolated_east, interpolated_north, _ = (
                            self.cs_basis.interpolate_vector_components(
                                grid_value_array[1],
                                -grid_value_array[0],
                                np.zeros_like(grid_value_array[0]),
                                input_grid.theta,
                                input_grid.phi,
                                self.state_grid.theta,
                                self.state_grid.phi,
                            )
                        )
                        interpolated_data = np.hstack(
                            (-interpolated_north, interpolated_east)
                        )  # convert to theta, phi

                    processed_data["GRID_" + var] = (
                        ["time", "i"],
                        interpolated_data.reshape((1, -1)),
                    )

            dataset = xr.Dataset(
                data_vars=processed_data,
                coords=xr.Coordinates.from_pandas_multiindex(
                    self.basis_multiindices[key], dim="i"
                ).merge({"time": [time[time_index]]}),
            )

            self.add_dataset(dataset, key)

        self.save(key)

    def add_dataset(self, dataset, key):
        """Add a dataset to the timeseries.

        Creates a new timeseries if one does not exist, otherwise
        concatenates the new data along the time dimension.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset containing the timeseries data.
        key : str
            The key identifying the type of data ('state', 'jr',
            'conductance', or 'u').
        """
        if key not in self.datasets.keys():
            self.datasets[key] = dataset.sortby("time")
        else:
            self.datasets[key] = xr.concat(
                [self.datasets[key].drop_sel(time=dataset.time, errors="ignore"), dataset],
                dim="time",
            ).sortby("time")

    def get_updated_data(self, key, current_time, interpolation=False):
        """Select time series data corresponding to the specified time.

        Parameters
        ----------
        key : str
            Key for the time series.
        current_time : float
            Current time for which to select data.
        interpolation : bool, optional
            Whether to use linear interpolation.

        Returns
        -------
        dict or None
            Dictionary containing the latest data for the specified
            key, or None if no new data is available.
        """
        if np.any(self.datasets[key].time.values <= current_time + FLOAT_ERROR_MARGIN):
            if self.vector_storage[key]:
                short_name = self.bases[key].short_name
            else:
                short_name = "GRID"

            current_data = {}

            # Select latest data before the current time.
            dataset_before = self.datasets[key].sel(
                time=[current_time + FLOAT_ERROR_MARGIN], method="ffill"
            )

            for var in self.vars[key]:
                current_data[var] = dataset_before[short_name + "_" + var].values.flatten()

            # If requested, add linear interpolation correction.
            if interpolation and np.any(
                self.datasets[key].time.values > current_time + FLOAT_ERROR_MARGIN
            ):
                dataset_after = self.datasets[key].sel(
                    time=[current_time + FLOAT_ERROR_MARGIN], method="bfill"
                )
                for var in self.vars[key]:
                    current_data[var] += (
                        (current_time - dataset_before.time.item())
                        / (dataset_after.time.item() - dataset_before.time.item())
                        * (
                            dataset_after[short_name + "_" + var].values.flatten()
                            - dataset_before[short_name + "_" + var].values.flatten()
                        )
                    )

            # Check if the data has changed since the last time.
            if not all([var in self.previous_data.keys() for var in self.vars[key]]) or (
                not all(
                    [
                        np.allclose(
                            current_data[var],
                            self.previous_data[var],
                            rtol=FLOAT_ERROR_MARGIN,
                            atol=0.0,
                        )
                        for var in self.vars[key]
                    ]
                )
            ):
                # Update the previous data with the current data.
                for var in self.vars[key]:
                    self.previous_data[var] = current_data[var]

                return current_data

        # No new data available.
        return None

    def save(self, key):
        """Save a timeseries to NetCDF file.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to save.
        """
        self.save_dataset(self.datasets[key].reset_index("i"), key)

    def load(self, key):
        """Load a timeseries from NetCDF file.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to load.
        """
        if self.dataset_filename_prefix is not None:
            dataset = self.load_dataset(key)

            if dataset is not None:
                if self.vector_storage[key]:
                    basis_index_names = self.bases[key].index_names
                else:
                    basis_index_names = ["theta", "phi"]

                basis_multiindex = pd.MultiIndex.from_arrays(
                    [
                        dataset[basis_index_names[i]].values
                        for i in range(len(basis_index_names))
                    ],
                    names=basis_index_names,
                )
                coords = xr.Coordinates.from_pandas_multiindex(
                    basis_multiindex, dim="i"
                ).merge({"time": dataset.time.values})
                self.datasets[key] = dataset.drop_vars(basis_index_names).assign_coords(
                    coords
                )

    def save_dataset(self, dataset, name):
        """Save a dataset to NetCDF file.

        Parameters
        ----------
        dataset : xarray.Dataset or xarray.DataArray
            The dataset to save.
        name : str
            Name to use in the filename.
        """
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
        filename = self.dataset_filename_prefix + "_" + name + ".ncdf"

        if os.path.exists(filename):
            return xr.load_dataset(filename)
        else:
            return None

    def set_dataset_filename_prefix(self, dataset_filename_prefix):
        """Set the prefix for the dataset filenames.

        Parameters
        ----------
        dataset_filename_prefix : str
            Prefix for the dataset filenames.
        """
        self.dataset_filename_prefix = dataset_filename_prefix
