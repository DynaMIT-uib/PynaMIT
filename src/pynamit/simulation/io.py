"""IO Class.

This module contains the IO class, which is responsible for handling
input and output operations in the simulation. It manages the reading
and writing of datasets, including time series data, and provides
methods for setting input data and selecting data for the simulation.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.grid import Grid
from pynamit.primitives.field_expansion import FieldExpansion

FLOAT_ERROR_MARGIN = 1e-6  # Safety margin for floating point errors


class IO:
    """IO Class.

    Class for handling input and output operations in the simulation.
    This class manages the reading and writing of datasets, including
    time series data, and provides methods for setting input data and
    selecting data for the simulation.
    """

    def __init__(self, bases, state_grid, cs_basis, vars, vector_storage, basis_multiindices):
        """Initialize the IO class.

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
        self.basis_multiindices = basis_multiindices

        self.dataset_filename_prefix = None

        self.timeseries = {}

    def set_input(
        self,
        key,
        input_data,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        current_time=None,
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

        if time is None:
            if any(
                [
                    input_data[var][component].shape[0] > 1
                    for var in input_data.keys()
                    for component in range(len(input_data[var]))
                ]
            ):
                raise ValueError(
                    "Time must be specified if the input data is given for multiple time values."
                )
            time = current_time

        time = np.atleast_1d(time)

        for time_index in range(time.size):
            processed_data = {}

            for var in self.vars[key]:
                if self.vector_storage[key]:
                    grid_value_array = np.array(
                        [
                            input_data[var][component][time_index]
                            for component in range(len(input_data[var]))
                        ]
                    )
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
                            input_data[var][0][time_index],
                            input_grid.theta,
                            input_grid.phi,
                            self.state_grid.theta,
                            self.state_grid.phi,
                        )
                    elif self.vars[key][var] == "tangential":
                        interpolated_east, interpolated_north, _ = (
                            self.cs_basis.interpolate_vector_components(
                                input_data[var][1][time_index],
                                -input_data[var][0][time_index],
                                np.zeros_like(input_data[var][1][time_index]),
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

            self.add_to_timeseries(dataset, key)

        self.save_timeseries(key)

    def add_to_timeseries(self, dataset, key):
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
        if key not in self.timeseries.keys():
            self.timeseries[key] = dataset.sortby("time")
        else:
            self.timeseries[key] = xr.concat(
                [self.timeseries[key].drop_sel(time=dataset.time, errors="ignore"), dataset],
                dim="time",
            ).sortby("time")

    def select_timeseries_data(self, key, current_time, interpolation=False):
        """Select time series data corresponding to the latest time.

        Parameters
        ----------
        key : str
            Key for the time series.
        interpolation : bool, optional
            Whether to use linear interpolation.

        Returns
        -------
        bool
            Whether the input data was selected.
        """
        if np.any(self.timeseries[key].time.values <= current_time + FLOAT_ERROR_MARGIN):
            if self.vector_storage[key]:
                short_name = self.bases[key].short_name
            else:
                short_name = "GRID"

            current_data = {}

            # Select latest data before the current time.
            dataset_before = self.timeseries[key].sel(
                time=[current_time + FLOAT_ERROR_MARGIN], method="ffill"
            )

            for var in self.vars[key]:
                current_data[var] = dataset_before[short_name + "_" + var].values.flatten()

            # If requested, add linear interpolation correction.
            if (
                interpolation
                and (key != "state")
                and np.any(self.timeseries[key].time.values > current_time + FLOAT_ERROR_MARGIN)
            ):
                dataset_after = self.timeseries[key].sel(
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

            return current_data

        else:
            # No data is available from before the current time.
            return None

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
        """Load dataset from file.

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

    def save_timeseries(self, key):
        """Save a timeseries to NetCDF file.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to save.
        """
        self.save_dataset(self.timeseries[key].reset_index("i"), key)

    def load_timeseries(self):
        """Load all time series that exist on file."""
        if self.dataset_filename_prefix is not None:
            for key in self.vars.keys():
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
                    self.timeseries[key] = dataset.drop_vars(basis_index_names).assign_coords(
                        coords
                    )

    def set_dataset_filename_prefix(self, dataset_filename_prefix):
        """Set the prefix for the dataset filenames.

        Parameters
        ----------
        dataset_filename_prefix : str
            Prefix for the dataset filenames.
        """
        self.dataset_filename_prefix = dataset_filename_prefix
