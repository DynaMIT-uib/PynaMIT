"""Input Manager Module.

This module contains the InputManager class, which manages the interpolation
and state tracking of input data for the simulation.
"""

import numpy as np

from pynamit.primitives.grid import Grid
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

    @staticmethod
    def _extract_tangential_components(raw_values, n_points):
        """Extract two tangential components from common container layouts.

        Supports:
        - tuple/list: (u_theta, u_phi)
        - ndarray shaped (2, N) or (N, 2)
        - ndarray shaped (2, N_theta, N_phi) or (N_theta, N_phi, 2)
        """
        if isinstance(raw_values, (tuple, list)) and len(raw_values) == 2:
            comp0, comp1 = raw_values
        else:
            arr = np.asarray(raw_values)
            if arr.ndim == 2 and arr.shape[0] == 2:
                comp0, comp1 = arr[0], arr[1]
            elif arr.ndim == 2 and arr.shape[1] == 2:
                comp0, comp1 = arr[:, 0], arr[:, 1]
            elif arr.ndim == 3 and arr.shape[0] == 2:
                comp0, comp1 = arr[0], arr[1]
            elif arr.ndim == 3 and arr.shape[-1] == 2:
                comp0, comp1 = arr[..., 0], arr[..., 1]
            else:
                return None

        comp0 = np.asarray(comp0)
        comp1 = np.asarray(comp1)
        if comp0.size != n_points or comp1.size != n_points:
            return None
        return comp0.reshape(-1), comp1.reshape(-1)

    @staticmethod
    def _extract_fast_weight_points(sqrt_weights, n_points, is_vector):
        """Extract point-wise weights for fast path from scalar/vector layouts."""
        arr = np.asarray(sqrt_weights)

        if not is_vector:
            if arr.ndim == 1:
                return arr
            return arr.reshape(-1)

        # Common vector-weight layouts: duplicated component axis.
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == n_points:
            if np.allclose(arr[0], arr[1]):
                return arr[0]
            return None
        if arr.ndim == 3 and arr.shape[0] == 2 and arr.shape[1] * arr.shape[2] == n_points:
            if np.allclose(arr[0], arr[1]):
                return arr[0].reshape(-1)
            return None
        if arr.ndim == 1 and arr.size == 2 * n_points:
            a0, a1 = arr[:n_points], arr[n_points:]
            if np.allclose(a0, a1):
                return a0
            return None

        # Already point-wise (same as scalar case).
        if arr.size == n_points:
            return arr.reshape(-1)
        return None

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
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)
        storage_spec = self.timeseries.get_storage_spec(key)
        target_basis = storage_spec
        target_mean_free = bool(storage_spec.mean_free)

        for time_index in range(time.size):
            interpolated_data = {}

            for var in self.variables[key]:

                target_grid = self.simulation_basis.grid
                raw_values = input_data[var][time_index]

                # Check for Fast Path (Regular Grid + SHBasis)
                # --------------------------------------------
                # If specific 1D coordinate arrays are provided (separable grid) 
                # and the data size matches the tensor product, we can simple reshape 
                # and use the fast transform.
                use_fast_path = False
                
                # Check regularity based on lat/lon or theta/phi from the Grid object
                # (which contains valid flattened arrays)
                if self.enable_fast_path and hasattr(target_basis, 'grid_to_basis_fast'):
                    # Heuristic: Check if grid corresponds to tensor product of unique values
                    # We use limited precision for uniqueness to handle float noise
                    u_lat = np.unique(np.round(input_grid.lat, 6))
                    u_lon = np.unique(np.round(input_grid.lon, 6))
                    
                    if u_lat.size * u_lon.size == input_grid.size:
                        # Regular Grid Detected!
                        # Now we need to ensure the data ordering matches (N_lat, N_lon) row-major
                        # If Grid was created via meshgrid(ij).flatten(), it matches.
                        # We assume standard ordering (lat changes slowly, lon changes fast)
                        # We should also verify sorting if we want to be 100% sure, but for now 
                        # we assume standard input generation.
                        
                        use_fast_path = True
                        N_theta_in = u_lat.size
                        N_phi_in = u_lon.size
                        
                        # Sorting unique values to pass expected 1D axes
                        # Note: SHBasis expects theta (colatitude) and phi (longitude)
                        # We need theta corresponding to the rows. 
                        # Usually lat is 90..-90 or -90..90.
                        # If grid follows "lat varies slowly", we take the lat sequence.
                        # But input_grid.lat is flattened.
                        # We need to construct the 1D arrays that generated this.
                        
                        # Extract 1D arrays from the flattened grid structure
                        # Reshape to check axes
                        try:
                            lat_2d = input_grid.lat.reshape(N_theta_in, N_phi_in)
                            lon_2d = input_grid.lon.reshape(N_theta_in, N_phi_in)
                            
                            # Verify separability structure
                            # Lat should be constant along columns (axis 1)
                            # Lon should be constant along rows (axis 0)
                            if (np.allclose(lat_2d[:, 0:1], lat_2d) and 
                                np.allclose(lon_2d[0:1, :], lon_2d)):
                                
                                # Extract 1D axes
                                lat_1d = lat_2d[:, 0]
                                lon_1d = lon_2d[0, :]
                                
                                theta_1d = np.deg2rad(90 - lat_1d)
                                phi_1d = np.deg2rad(lon_1d)
                            else:
                                use_fast_path = False
                        except ValueError:
                            # Reshape failed imply wrong dimensions for this assumption
                            use_fast_path = False

                if use_fast_path:
                    # FAST PATH: Reshape and Transform directly
                    
                    vector_type = self.variables[key][var]
                    is_vector = (vector_type == "tangential")

                    # Handle Weights (Separable Extraction)
                    # -------------------------------------
                    weights_1d = None
                    if sqrt_weights is not None:
                         # sqrt_weights may be point-wise scalar weights (N),
                         # or duplicated vector weights (2, N) / (2*N) for tangential fields.
                         # We reduce to point-wise weights, then require zonal separability
                         # for the fast per-m solver.
                         try:
                             n_points = N_theta_in * N_phi_in
                             w_points = self._extract_fast_weight_points(
                                 sqrt_weights, n_points, is_vector
                             )
                             if w_points is None:
                                 use_fast_path = False
                                 raise ValueError("Unsupported vector weight layout for fast path")

                             # Allow directly provided theta-only weights as a convenience.
                             if w_points.size == N_theta_in:
                                 weights_1d = w_points
                                 W_2d = None
                             else:
                                 W_2d = w_points.reshape(N_theta_in, N_phi_in)
                             if W_2d is not None:
                                 if np.allclose(W_2d[:, 0:1], W_2d):
                                      weights_1d = W_2d[:, 0]
                                 else:
                                      # Weights not purely zonal?
                                      # If they vary in phi, we can't efficiently use the 1D stacked solver per m
                                      # (coupling m modes).
                                      # Fallback to slow path if complex weights are present.
                                      use_fast_path = False
                         except ValueError:
                             use_fast_path = False

                    if use_fast_path:
                        if is_vector:
                            # Accept tuple/list pairs and ndarray packings such as
                            # (2, N) produced by Dynamics.set_u after time slicing.
                            n_points = N_theta_in * N_phi_in
                            comps = self._extract_tangential_components(raw_values, n_points)
                            if comps is None:
                                use_fast_path = False
                            else:
                                u_th = comps[0].reshape(N_theta_in, N_phi_in)
                                u_ph = comps[1].reshape(N_theta_in, N_phi_in)
                                data_in = (u_th, u_ph)

                        else:
                            # Scalar
                            data_in = raw_values.reshape(N_theta_in, N_phi_in)

                # Finalizing Fast Path Decision
                # If any check inside the block failed, use_fast_path is now False
                
                if use_fast_path:
                    # FAST PATH Execution
                     coeffs = target_basis.grid_to_basis_fast(
                          data_in, 
                          theta_1d, 
                          phi_1d, 
                          weights=weights_1d, 
                          reg_lambda=reg_lambda, 
                          vector_type=vector_type
                     )
                else:
                    # SLOW PATH Execution (Fallback)
                    coeffs = projection_basis.project_to_basis(
                        raw_values,
                        input_grid,
                        vector_type=self.variables[key][var],
                        target_grid=target_grid,
                        target_basis=target_basis,
                        mean_free=target_mean_free,
                        weights=sqrt_weights,
                        reg_lambda=reg_lambda,
                        pinv_rtol=pinv_rtol,
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

    def get_entry_with_derivative(self, key, time, interpolation=False):
        """Get data and derivative from timeseries."""
        return self.timeseries.get_entry_with_derivative(key, time, interpolation=interpolation)

    @property
    def input_keys(self):
        """Return the keys of the available input datasets."""
        return self.timeseries.datasets.keys()
