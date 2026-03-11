"""Input Manager Module.

This module contains the InputManager class, which manages the interpolation
and state tracking of input data for the simulation.
"""

import numpy as np

from pynamit.primitives.grid import Grid
from pynamit.primitives.projection_pipeline import (
    normalize_projection_input_batch,
    project_batch_to_coefficients,
)
from pynamit.primitives.timeseries import Timeseries

FLOAT_ERROR_MARGIN = 1e-6


class InputManager:
    """Class for managing simulation inputs.

    Handles the interpolation of input data to the simulation grid/basis
    and tracks changes in input data to optimize updates.
    """

    def __init__(
        self,
        timeseries: Timeseries,
        simulation_basis,
        variables_dict,
        *,
        enable_fast_path: bool = True,
    ):
        """Initialize the InputManager.

        Parameters
        ----------
        timeseries : TimeSeries
            The storage object for the time series data.
        simulation_basis : Basis
            The basis defining the simulation grid (e.g. Cubed Sphere).
        variables_dict : dict
            Dictionary defining the variable stucture (e.g. scalar/tangential).
        """
        self.timeseries = timeseries
        self.simulation_basis = simulation_basis
        self.variables = variables_dict
        self.enable_fast_path = bool(enable_fast_path)

    def _project_variable_batch(
        self,
        *,
        raw_values,
        vector_type,
        input_grid,
        n_times,
        projection_basis,
        target_basis,
        target_grid,
        target_mean_free,
        sqrt_weights,
        reg_lambda,
        pinv_rtol,
    ):
        """Project one variable batch to coefficient columns."""
        value_batch = normalize_projection_input_batch(
            raw_values, vector_type=vector_type, n_points=input_grid.size, n_times=n_times
        )
        return project_batch_to_coefficients(
            value_batch,
            input_grid=input_grid,
            vector_type=vector_type,
            projection_basis=projection_basis,
            target_basis=target_basis,
            target_grid=target_grid,
            target_mean_free=target_mean_free,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
            enable_fast_path=self.enable_fast_path,
        )

    def interpolate_and_add_entry(
        self,
        key,
        input_data,
        time,
        projection_basis,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Interpolate data and add it to the timeseries.

        Parameters
        ----------
        key : str
            The type of data ('jr', 'conductance', or 'u').
        input_data : dict
            Dictionary containing the input data arrays.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the input data.
        projection_basis : Basis
            The basis used to project/interpret the input data.
        sqrt_weights : array-like, optional
            sqrt_weights for the input data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for pseudo-inverse.
        """
        time = np.atleast_1d(np.asarray(time))
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)
        storage_spec = self.timeseries.get_storage_spec(key)
        target_basis = storage_spec
        target_mean_free = bool(storage_spec.mean_free)
        target_grid = self.simulation_basis.grid

        batched_coeffs = {}
        for var in self.variables[key]:
            vector_type = self.variables[key][var]
            batched_coeffs[var] = self._project_variable_batch(
                raw_values=input_data[var],
                vector_type=vector_type,
                input_grid=input_grid,
                n_times=time.size,
                projection_basis=projection_basis,
                target_basis=target_basis,
                target_grid=target_grid,
                target_mean_free=target_mean_free,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
            )

        for time_index in range(time.size):
            interpolated_data = {
                var: batched_coeffs[var][:, time_index] for var in self.variables[key]
            }
            self.timeseries.add_entry(key, interpolated_data, time[time_index])

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
            key, or None if no new data is available.
        """
        return self.timeseries.get_entry(key, time, interpolation=interpolation)

    def get_entry_with_derivative(self, key, time, interpolation=False):
        """Get data and derivative from timeseries."""
        return self.timeseries.get_entry_with_derivative(key, time, interpolation=interpolation)

    @property
    def input_keys(self):
        """Return the keys of the available input datasets."""
        return self.timeseries.datasets.keys()
