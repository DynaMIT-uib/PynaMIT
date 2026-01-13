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

    def __init__(self, timeseries: Timeseries, simulation_basis, variables_dict):
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

        for time_index in range(time.size):
            interpolated_data = {}

            for var in self.variables[key]:

                target_grid = self.simulation_basis.grid
                target_basis = self.timeseries.storage_bases[key]
                raw_values = input_data[var][time_index]

                # Check for Fast Path (Regular Grid + SHBasis)
                # --------------------------------------------
                # If specific 1D coordinate arrays are provided (separable grid) 
                # and the data size matches the tensor product, we can simple reshape 
                # and use the fast transform.
                use_fast_path = False
                
                # Check regularity based on lat/lon or theta/phi from the Grid object
                # (which contains valid flattened arrays)
                if hasattr(target_basis, 'grid_to_basis_fast'):
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
                         # sqrt_weights is flattened (N_theta * N_phi)
                         # We assume it's separable: W(theta, phi) = W_th(theta) * W_ph(phi)
                         # PynaMIT typically uses sqrt(sin(theta)) weights which are pure theta.
                         # Check separability:
                         try:
                             W_2d = sqrt_weights.reshape(N_theta_in, N_phi_in)
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
                            # For vectors, we need input_data to be a tuple (u_theta, u_phi)
                            # But input_data[var] is just one component?
                            # Usually 'u' key has sub-vars 'u_theta', 'u_phi'.
                            # Wait, 'interpolate_and_add_entry' is called per KEY (e.g. 'u').
                            # It loops over 'var' in variables_dict['u'].
                            # 'variables_dict' maps key -> {var_name: vector_type}.
                            # If vector_type is tangential, does 'var' hold both components?
                            # Standard PynaMIT usage: variables['u'] = {'u': 'tangential'}?
                            # Or variables['u'] = {'u_theta': 'scalar', 'u_phi': 'scalar'}?
                            
                            # Let's check mage_forcing_2.py.
                            # variables={'u': {'u_theta': 'scalar', 'u_phi': 'scalar'}} usually implies treated as scalars?
                            # No, usually vector quantities are bundled or handled distinctly.
                            # If InputManager iterates variables, it treats them individually.
                            # IF they are individual scalars in the dict, vector_type is 'scalar'.
                            # IF vector_type is 'tangential', input_data[var] must be a tuple/object.
                            
                            # However, in PynaMIT vector fields are often stored as coefficients of Pol/Tor.
                            # If the input is raw grid data, we need 2 components to fit Pol/Tor.
                            # If 'var' refers to a composite field, raw_values should be (u_th, u_ph).
                            # If raw_values is a single array, it can't be tangent vector project.
                            
                            # Logic:
                            # If vector_type == 'tangential':
                            #    Assumes raw_values is a tuple (data_th, data_ph)
                            if isinstance(raw_values, (tuple, list)) and len(raw_values) == 2:
                                u_th = raw_values[0].reshape(N_theta_in, N_phi_in)
                                u_ph = raw_values[1].reshape(N_theta_in, N_phi_in)
                                data_in = (u_th, u_ph)
                            else:
                                # Data doesn't match vector expectation -> Fallback
                                use_fast_path = False

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

    @property
    def input_keys(self):
        """Return the keys of the available input datasets."""
        return self.timeseries.datasets.keys()

    def get_storage_basis(self, key):
        """Return the storage basis for a given key."""
        return self.timeseries.storage_bases.get(key)
