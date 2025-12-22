"""Input Manager Module.

This module contains the InputManager class, which manages the interpolation
and state tracking of input data for the simulation.
"""

import numpy as np
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.grid import Grid
from pynamit.primitives.timeseries import Timeseries

FLOAT_ERROR_MARGIN = 1e-6


class InputManager:
    """Class for managing simulation inputs.

    Handles the interpolation of input data to the simulation grid/basis
    and tracks changes in input data to optimize updates.
    """

    def __init__(self, timeseries: Timeseries, grid_basis, vars_dict):
        """Initialize the InputManager.

        Parameters
        ----------
        timeseries : TimeSeries
            The storage object for the time series data.
        grid_basis : Basis
            The basis defining the simulation grid (e.g. Cubed Sphere).
        vars_dict : dict
            Dictionary defining the variable stucture (e.g. scalar/tangential).
        """
        self.timeseries = timeseries
        self.grid_basis = grid_basis
        self.vars = vars_dict

        # Evaluators
        self.storage_basis_evaluators = {}
        self.input_basis_evaluators = {}

        # Initialize storage evaluators
        # Note: We access storage_bases from the timeseries object
        for key in self.timeseries.storage_bases.keys():
            self.storage_basis_evaluators[key] = BasisEvaluator(
                self.timeseries.storage_bases[key], grid_basis.grid
            )

    def interpolate_and_add_entry(
        self,
        key,
        input_data,
        time,
        interpolation_basis,
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
        sqrt_weights : array-like, optional
            sqrt_weights for the input data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for pseudo-inverse.
        """
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)

        for time_index in range(time.size):
            interpolated_data = {}

            for var in self.vars[key]:
                # Use the grid from the storage evaluator as the target grid
                target_grid = self.storage_basis_evaluators[key].grid
                target_basis = self.timeseries.storage_bases[key]

                def get_storage_evaluator():
                    return self.storage_basis_evaluators[key]

                def get_input_evaluator():
                    if not (
                        key in self.input_basis_evaluators.keys()
                        and input_grid == self.input_basis_evaluators[key].grid
                    ):
                        self.input_basis_evaluators[key] = BasisEvaluator(
                            interpolation_basis,
                            input_grid,
                            sqrt_weights=sqrt_weights,
                            reg_lambda=reg_lambda,
                            pinv_rtol=pinv_rtol,
                        )
                    return self.input_basis_evaluators[key]

                coeffs = interpolation_basis.project_to_basis(
                    input_data[var][time_index],
                    input_grid,
                    vector_type=self.vars[key][var],
                    target_grid=target_grid,
                    target_basis=target_basis,
                    on_storage_grid=get_storage_evaluator,
                    on_input_grid=get_input_evaluator,
                )

                interpolated_data[var] = coeffs

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

    @property
    def input_keys(self):
        """Return the keys of the available input datasets."""
        return self.timeseries.datasets.keys()

    def get_storage_basis(self, key):
        """Return the storage basis for a given key."""
        return self.timeseries.storage_bases.get(key)
