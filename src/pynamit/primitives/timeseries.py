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
        """
        self.bases = bases
        self.state_grid = state_grid
        self.cs_basis = cs_basis

        # Initialize variables and timeseries storage
        self.vars = vars
        self.vector_storage = vector_storage

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
            data_vars[var] = (["time", "i"], data[var].reshape((1, -1)))

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
            and input_grid.theta.shape == self.input_basis_evaluators[key].grid.theta.shape
            and input_grid.phi.shape == self.input_basis_evaluators[key].grid.phi.shape
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

        for time_index in range(time.size):
            processed_data = {}

            for var in self.vars[key]:
                if self.vector_storage[key]:
                    vector = FieldExpansion(
                        self.bases[key],
                        basis_evaluator=self.input_basis_evaluators[key],
                        grid_values=input_data[var][time_index],
                        field_type=self.vars[key][var],
                    )

                    processed_data[self.bases[key].kind + "_" + var] = vector.coeffs

                else:
                    # Interpolate to state_grid
                    if self.vars[key][var] == "scalar":
                        interpolated_data = self.cs_basis.interpolate_scalar(
                            input_data[var][time_index],
                            input_grid.theta,
                            input_grid.phi,
                            self.state_grid.theta,
                            self.state_grid.phi,
                        )
                    elif self.vars[key][var] == "tangential":
                        interpolated_east, interpolated_north, _ = (
                            self.cs_basis.interpolate_vector_components(
                                input_data[var][time_index, 1],
                                -input_data[var][time_index, 0],
                                np.zeros_like(input_data[var][time_index, 0]),
                                input_grid.theta,
                                input_grid.phi,
                                self.state_grid.theta,
                                self.state_grid.phi,
                            )
                        )
                        interpolated_data = np.hstack(
                            (-interpolated_north, interpolated_east)
                        )  # convert to theta, phi

                    processed_data[self.bases[key].kind + "_" + var] = interpolated_data

            self.add_entry(key, processed_data, time[time_index])

    def get_entry_if_changed(self, key, time, interpolation=False):
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
            key, or None if no new data is available.
        """
        if np.any(self.datasets[key].time.values <= time + FLOAT_ERROR_MARGIN):
            current_data = {}

            # Select latest data before the current time.
            dataset_before = self.datasets[key].sel(
                time=[time + FLOAT_ERROR_MARGIN], method="ffill"
            )

            for var in self.vars[key]:
                current_data[var] = dataset_before[
                    self.bases[key].kind + "_" + var
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
                            dataset_after[self.bases[key].kind + "_" + var].values.flatten()
                            - dataset_before[self.bases[key].kind + "_" + var].values.flatten()
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

    def save(self, key, io):
        """Save a timeseries to NetCDF file.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to save.
        """
        io.save_dataset(self.datasets[key].reset_index("i"), key)

    def load(self, key, io):
        """Load a timeseries from NetCDF file.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to load.
        """
        dataset = io.load_dataset(key)

        if dataset is not None:
            if self.vector_storage[key]:
                basis_index_names = self.bases[key].index_names
            else:
                basis_index_names = ["theta", "phi"]

            basis_multiindex = pd.MultiIndex.from_arrays(
                [dataset[basis_index_names[i]].values for i in range(len(basis_index_names))],
                names=basis_index_names,
            )
            coords = xr.Coordinates.from_pandas_multiindex(basis_multiindex, dim="i").merge(
                {"time": dataset.time.values}
            )
            self.datasets[key] = dataset.drop_vars(basis_index_names).assign_coords(coords)
