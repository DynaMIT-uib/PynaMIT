import numpy as np
import xarray as xr
import pandas as pd
from pynamit.simulation.mainfield import Mainfield
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.cubed_sphere.cubed_sphere import csp
from pynamit.cubed_sphere.cubed_sphere import CSProjection
from pynamit.primitives.vector import Vector
import os
from pynamit.primitives.grid import Grid
from pynamit.simulation.state import State
from pynamit.various.constants import RE
import scipy.sparse as sp

FLOAT_ERROR_MARGIN = 1e-6 # safety margin for floating point errors

class Dynamics(object):
    """ 2D ionosphere. """

    def __init__(self,
                 dataset_filename_prefix = 'simulation',
                 Nmax = 20,
                 Mmax = 20,
                 Ncs = 30,
                 RI = RE + 110.e3,
                 mainfield_kind = 'dipole',
                 mainfield_epoch = 2020,
                 mainfield_B0 = None,
                 FAC_integration_steps = np.logspace(np.log10(RE + 110.e3), np.log10(4 * RE), 11),
                 ignore_PFAC = False,
                 connect_hemispheres = False,
                 latitude_boundary = 50,
                 ih_constraint_scaling = 1e-5,
                 PFAC_matrix = None,
                 vector_jr = True,
                 vector_conductance = True,
                 vector_u = True,
                 t0 = '2020-01-01 00:00:00'):
        """

        Parameters
        ----------
        sh: sha.SHBasis object
            Spherical harmonic basis object.
        csp: cubedsphere.CSProjection object
            Cubed sphere projection object.
        RI: float, optional, default = RE + 110.e3
            Radius of the ionosphere in m.
        mainfield_kind: string, {'dipole', 'radial', 'igrf'}, default = 'dipole'
            Set to the main field model you want. For 'dipole' and
            'igrf', you can specify epoch via `B0_parameters`.
        FAC_integration_steps: array-like
            Use this to specify the radii used in the integral to calculate
            the poloidal field of FACs
        t0: str, optional
            string representaion of UT of t = 0. This is optional, and not used in the 
            simulation. It can be used to retrieve local time if the simulations are 
            performed in lon/lat
        """

        self.dataset_filename_prefix = dataset_filename_prefix

        # Store setting arguments in xarray dataset
        settings = xr.Dataset(attrs = {
            'Nmax':                   Nmax,
            'Mmax':                   Mmax,
            'Ncs':                    Ncs,
            'RI':                     RI,
            'latitude_boundary':      latitude_boundary,
            'ignore_PFAC':            int(ignore_PFAC),
            'connect_hemispheres':    int(connect_hemispheres),
            'FAC_integration_steps':  FAC_integration_steps,
            'ih_constraint_scaling':  ih_constraint_scaling,
            'mainfield_kind':         mainfield_kind,
            'mainfield_epoch':        mainfield_epoch,
            'mainfield_B0':           0 if mainfield_B0 is None else mainfield_B0,
            'vector_jr':              int(vector_jr),
            'vector_conductance':     int(vector_conductance),
            'vector_u':               int(vector_u),
            't0':                     t0,
        })

        # Overwrite settings with any settings existing on file
        settings_on_file = (self.dataset_filename_prefix is not None) and os.path.exists(self.dataset_filename_prefix + '_settings.ncdf')
        if settings_on_file:
            settings = xr.load_dataset(self.dataset_filename_prefix + '_settings.ncdf')

        # Load PFAC matrix if it exists on file
        PFAC_matrix_on_file = (self.dataset_filename_prefix is not None) and os.path.exists(self.dataset_filename_prefix + '_PFAC_matrix.ncdf')
        if PFAC_matrix_on_file:
            PFAC_matrix = xr.load_dataarray(self.dataset_filename_prefix + '_PFAC_matrix.ncdf')

        self.RI = settings.RI

        self.mainfield = Mainfield(kind = settings.mainfield_kind,
                                   epoch = settings.mainfield_epoch,
                                   hI = (settings.RI - RE) * 1e-3,
                                   B0 = None if settings.mainfield_B0 == 0 else settings.mainfield_B0)

        self.csp = CSProjection(settings.Ncs)
        self.state_grid = Grid(theta = self.csp.arr_theta, phi = self.csp.arr_phi)

        self.bases = {
            'state':       SHBasis(settings.Nmax, settings.Mmax),
            'jr':          SHBasis(settings.Nmax, settings.Mmax),
            'conductance': SHBasis(settings.Nmax, settings.Mmax, Nmin = 0),
            'u':           SHBasis(settings.Nmax, settings.Mmax),
        }

        self.pinv_rtols = {
            'state':       1e-15,
            'jr':          1e-15,
            'conductance': 1e-15,
            'u':           1e-15,
        }

        self.vector_storage = {
            'state':       True,
            'jr':          bool(settings.vector_jr),
            'conductance': bool(settings.vector_conductance),
            'u':           bool(settings.vector_u),
        }

        self.vars = {
            'state':       {'m_ind': 'scalar', 'm_imp': 'scalar', 'Phi': 'scalar', 'W': 'scalar'},
            'jr':          {'jr': 'scalar'},
            'conductance': {'etaP': 'scalar', 'etaH': 'scalar'},
            'u':           {'u': 'tangential'},
        }

        self.basis_multiindices = {}
        for key in self.vars.keys():
            if self.vector_storage[key]:
                basis_index_arrays = self.bases[key].index_arrays
                basis_index_names = self.bases[key].index_names
            else:
                basis_index_arrays = [self.state_grid.theta, self.state_grid.phi]
                basis_index_names = ['theta', 'phi']

            if all(self.vars[key][var] == 'scalar' for var in self.vars[key]):
                self.basis_multiindices[key] = pd.MultiIndex.from_arrays(basis_index_arrays, names = basis_index_names)
            elif all(self.vars[key][var] == 'tangential' for var in self.vars[key]):
                self.basis_multiindices[key] = pd.MultiIndex.from_arrays([np.tile(basis_index_arrays[i], 2) for i in range(len(basis_index_arrays))], names = basis_index_names)
            else:
                raise ValueError('Mixed scalar and tangential input (unsupported), or unknown input type')

        # Initialize the state of the ionosphere
        self.state = State(self.bases,
                           self.pinv_rtols,
                           self.mainfield,
                           self.state_grid,
                           settings,
                           PFAC_matrix = PFAC_matrix)

        self.timeseries = {}
        self.load_timeseries()

        if 'state' in self.timeseries.keys():
            self.current_time = np.max(self.timeseries['state'].time.values) # latest time in state time series
            self.select_timeseries_data('state')
        else:
            self.current_time = np.float64(0)
            self.state.set_coeffs(m_ind = np.zeros(self.bases['state'].index_length))

        if self.dataset_filename_prefix is None:
            self.dataset_filename_prefix = 'simulation'

        if not settings_on_file:
            self.save_dataset(settings, 'settings')
            print('Saved settings to {}_settings.ncdf'.format(self.dataset_filename_prefix))

        if not PFAC_matrix_on_file:
            self.save_dataset(self.state.m_imp_to_B_pol, 'PFAC_matrix')
            print('Saved PFAC matrix to {}_PFAC_matrix.ncdf'.format(self.dataset_filename_prefix))


    def evolve_to_time(self, t, dt = np.float64(5e-4), sampling_step_interval = 200, saving_sample_interval = 10, quiet = False, interpolation = False):
        """
        Evolve to the given time `t`. Will overwrite the values
        corresponding to the start time, to account for any changes in
        jr, conductance or neutral wind since the end of the previous call
        to `evolve_to_time`.

        """

        # Will be set to True when the corresponding time series is different from the one saved on disk
        self.save_jr          = False
        self.save_conductance = False
        self.save_u           = False
 
        count = 0
        while True:
            timeseries_keys = list(self.timeseries.keys())
            if 'state' in timeseries_keys:
                timeseries_keys.remove('state')
            if timeseries_keys is not None:
                for key in timeseries_keys:
                    self.select_timeseries_data(key, interpolation = interpolation)

            self.state.impose_constraints()
            self.state.update_Phi_and_W()

            if count % sampling_step_interval == 0:
                current_state_dataset = xr.Dataset(
                    data_vars = {
                        self.bases['state'].short_name + '_m_imp': (['time', 'i'], self.state.m_imp.coeffs.reshape((1, -1))),
                        self.bases['state'].short_name + '_m_ind': (['time', 'i'], self.state.m_ind.coeffs.reshape((1, -1))),
                        self.bases['state'].short_name + '_Phi':   (['time', 'i'], self.state.Phi.coeffs.reshape((1, -1))),
                        self.bases['state'].short_name + '_W':     (['time', 'i'], self.state.W.coeffs.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(self.basis_multiindices['state'], dim = 'i').merge({'time': [self.current_time]})
                )

                self.add_to_timeseries(current_state_dataset, 'state')

                if (count % (sampling_step_interval * saving_sample_interval) == 0):
                    self.save_timeseries('state')

                    if quiet:
                        pass
                    else:
                        print('Saved output at t = {:.2f} s'.format(self.current_time), end = '\r')

            next_time = self.current_time + dt

            if next_time > t + FLOAT_ERROR_MARGIN:
                break

            self.state.evolve_Br(dt)
            self.current_time = next_time

            count += 1


    def set_FAC(self, FAC, lat = None, lon = None, theta = None, phi = None, time = None, weights = None):
        """
        Set the field-aligned current at the given coordinate points.
        """

        FAC_b_evaluator = FieldEvaluator(self.mainfield, Grid(lat = lat, lon = lon, theta = theta, phi = phi), self.RI)

        self.set_jr(FAC * FAC_b_evaluator.br, lat = lat, lon = lon, theta = theta, phi = phi, time = time, weights = weights)


    def set_jr(self, jr, lat = None, lon = None, theta = None, phi = None, time = None, weights = None):
        """
        Specify radial current at ``self.state_grid.theta``,
        ``self.state_grid.phi``.

            Parameters
            ----------
            jr: array
                The radial current, in A/m^2, at
                ``self.state_grid.theta`` and ``self.state_grid.phi``, at
                ``RI``. The values in the array have to match the
                corresponding coordinates.

        """

        input_data = {
            'jr': [np.atleast_2d(jr)],
        }

        self.set_input('jr', input_data, lat = lat, lon = lon, theta = theta, phi = phi, time = time, weights = weights)


    def set_conductance(self, Hall, Pedersen, lat = None, lon = None, theta = None, phi = None, time = None, weights = None):
        """
        Specify Hall and Pedersen conductance at
        ``self.state_grid.theta``, ``self.state_grid.phi``.

        """

        Hall = np.atleast_2d(Hall)
        Pedersen = np.atleast_2d(Pedersen)

        input_data = {
            'etaP': [np.empty_like(Pedersen)],
            'etaH': [np.empty_like(Hall)],
        }

        # Convert to resistivity
        for i in range(max(input_data['etaP'][0].shape[0], 1)):
            input_data['etaP'][0][i] = Pedersen[i] / (Hall[i]**2 + Pedersen[i]**2)

        for i in range(max(input_data['etaH'][0].shape[0], 1)):
            input_data['etaH'][0][i] = Hall[i] / (Hall[i]**2 + Pedersen[i]**2)

        self.set_input('conductance', input_data, lat = lat, lon = lon, theta = theta, phi = phi, time = time, weights = weights)


    def set_u(self, u_theta, u_phi, lat = None, lon = None, theta = None, phi = None, time = None, weights = None):
        """ set neutral wind theta and phi components 
            For now, they *have* to be given on grid
        """

        input_data = {
            'u': [np.atleast_2d(u_theta), np.atleast_2d(u_phi)],
        }

        self.set_input('u', input_data, lat = lat, lon = lon, theta = theta, phi = phi, time = time, weights = weights)


    def set_input(self, key, input_data, lat = None, lon = None, theta = None, phi = None, time = None, weights = None):
        """ Set input. """

        input_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if not hasattr(self.state, 'input_basis_evaluators'):
            self.input_basis_evaluators = {}

        if not (key in self.input_basis_evaluators.keys() and np.allclose(input_grid.theta, self.input_basis_evaluators[key].grid.theta, rtol = 0.0, atol = FLOAT_ERROR_MARGIN) and np.allclose(input_grid.phi, self.input_basis_evaluators[key].grid.phi, rtol = 0.0, atol = FLOAT_ERROR_MARGIN)):
            self.input_basis_evaluators[key] = BasisEvaluator(self.bases[key], input_grid, self.pinv_rtols[key], weights = weights)

        if time is None:
            if any([input_data[var][component].shape[0] > 1 for var in input_data.keys() for component in range(len(input_data[var]))]):
                raise ValueError('Time must be specified if the input data is given for multiple time values.')
            time = self.current_time

        time = np.atleast_1d(time)

        for time_index in range(time.size):
            processed_data = {}

            for var in self.vars[key]:
                if self.vector_storage[key]:
                    vector = Vector(self.bases[key], basis_evaluator = self.input_basis_evaluators[key], grid_values = np.hstack([input_data[var][component][time_index] for component in range(len(input_data[var]))]), type = self.vars[key][var])

                    processed_data[self.bases[key].short_name + '_' + var] = (['time', 'i'], vector.coeffs.reshape((1, -1)))

                else:
                    # Interpolate to state_grid
                    if self.vars[key][var] == 'scalar':
                        interpolated_data = csp.interpolate_scalar(input_data[var][0][time_index], input_grid.theta, input_grid.phi, self.state_grid.theta, self.state_grid.phi)
                    elif self.vars[key][var] == 'tangential':
                        interpolated_east, interpolated_north, _ = csp.interpolate_vector_components(input_data[var][1][time_index], -input_data[var][0][time_index], np.zeros_like(input_data[var][1][time_index]), input_grid.theta, input_grid.phi, self.state_grid.theta, self.state_grid.phi)
                        interpolated_data = np.hstack((-interpolated_north, interpolated_east)) # convert to theta, phi

                    processed_data['GRID_' + var] = (['time', 'i'], interpolated_data.reshape((1, -1)))

            dataset = xr.Dataset(
                data_vars = processed_data,
                coords = xr.Coordinates.from_pandas_multiindex(self.basis_multiindices[key], dim = 'i').merge({'time': [time[time_index]]})
            )

            self.add_to_timeseries(dataset, key)

        self.save_timeseries(key)


    def add_to_timeseries(self, dataset, key):
        """ Add a dataset to the time series. """

        if key not in self.timeseries.keys():
            self.timeseries[key] = dataset.sortby('time')
        else:
            self.timeseries[key] = xr.concat([self.timeseries[key].drop_sel(time = dataset.time, errors = 'ignore'), dataset], dim = 'time').sortby('time')


    def select_timeseries_data(self, key, interpolation = False):
        """ Select time series data corresponding to the latest time. """

        if np.any(self.timeseries[key].time.values <= self.current_time + FLOAT_ERROR_MARGIN):
            if self.vector_storage[key]:
                short_name = self.bases[key].short_name
            else:
                short_name = 'GRID'

            current_data = {}

            # Select latest data before the current time
            dataset_before = self.timeseries[key].sel(time = [self.current_time + FLOAT_ERROR_MARGIN], method = 'ffill')

            for var in self.vars[key]:
                current_data[var] = dataset_before[short_name + '_' + var].values.flatten()

            # If requested, add linear interpolation correction
            if interpolation and (key != 'state') and np.any(self.timeseries[key].time.values > self.current_time + FLOAT_ERROR_MARGIN):
                dataset_after = self.timeseries[key].sel(time = [self.current_time + FLOAT_ERROR_MARGIN], method = 'bfill')
                for var in self.vars[key]:
                    current_data[var] +=  (self.current_time - dataset_before.time.item()) / (dataset_after.time.item() - dataset_before.time.item()) * (dataset_after[short_name + '_' + var].values.flatten() - dataset_before[short_name + '_' + var].values.flatten())

        else:
            # No data available before the current time
            return

        if not hasattr(self, 'previous_data'):
            self.previous_data = {}

        # Update state if is the first data selection or if the data has changed since the last selection
        if (not all([var in self.previous_data.keys() for var in self.vars[key]]) or (not all([np.allclose(current_data[var], self.previous_data[var], rtol = FLOAT_ERROR_MARGIN, atol = 0.0) for var in self.vars[key]]))):
            if key == 'state':
                self.state.set_coeffs(m_ind = current_data['m_ind'])
                self.state.set_coeffs(m_imp = current_data['m_imp'])
                self.state.set_coeffs(Phi   = current_data['Phi'])
                self.state.set_coeffs(W     = current_data['W'])

            if key == 'jr':
                if self.vector_storage[key]:
                    jr = Vector(basis = self.bases[key], coeffs = current_data['jr'], type = self.vars[key]['jr'])
                else:
                    jr = current_data['jr']

                self.state.set_jr(jr, self.vector_storage[key])

            elif key == 'conductance':
                if self.vector_storage[key]:
                    etaP = Vector(basis = self.bases[key], coeffs = current_data['etaP'], type = self.vars[key]['etaP'])
                    etaH = Vector(basis = self.bases[key], coeffs = current_data['etaH'], type = self.vars[key]['etaH'])
                else:
                    etaP = current_data['etaP']
                    etaH = current_data['etaH']

                self.state.set_conductance(etaP, etaH, self.vector_storage[key])

            elif key == 'u':
                if self.vector_storage[key]:
                    u = Vector(basis = self.bases[key], coeffs = current_data['u'], type = self.vars[key]['u'])
                else:
                    u = current_data['u']

                self.state.set_u(u, self.vector_storage[key])

            for var in self.vars[key]:
                self.previous_data[var] = current_data[var]


    def save_dataset(self, dataset, name):
        """ Save dataset to file. """

        filename = self.dataset_filename_prefix + '_' + name + '.ncdf'

        try:
            dataset.to_netcdf(filename + '.tmp')
            os.rename(filename + '.tmp', filename)

        except Exception as e:
            if os.path.exists(filename + '.tmp'):
                os.remove(filename + '.tmp')
            raise e


    def load_dataset(self, name):
        """ Load dataset from file. """

        filename = self.dataset_filename_prefix + '_' + name + '.ncdf'

        if os.path.exists(filename):
            return xr.load_dataset(filename)
        else:
            return None


    def save_timeseries(self, key):
        """ Save time series to file. """

        self.save_dataset(self.timeseries[key].reset_index('i'), key)


    def load_timeseries(self):
        """ Load all time series that exist on file. """

        if (self.dataset_filename_prefix is not None):

            for key in self.vars.keys():
                dataset = self.load_dataset(key)

                if dataset is not None:
                    if self.vector_storage[key]:
                        basis_index_names = self.bases[key].index_names
                    else:
                        basis_index_names = ['theta', 'phi']

                    basis_multiindex = pd.MultiIndex.from_arrays([dataset[basis_index_names[i]].values for i in range(len(basis_index_names))], names = basis_index_names)
                    coords = xr.Coordinates.from_pandas_multiindex(basis_multiindex, dim = 'i').merge({'time': dataset.time.values})
                    self.timeseries[key] = dataset.drop_vars(basis_index_names).assign_coords(coords)


    def calculate_fd_curl_matrix(self, stencil_size = 1, interpolation_points = 4):
        """ Calculate matrix that returns the radial curl, using finite differences 
            when operated on a column vector of (theta, phi) vector components. 
        """

        Dxi, Deta = self.csp.get_Diff(self.csp.N, coordinate = 'both', Ns = stencil_size, Ni = interpolation_points, order = 1)
        sqrtg = np.sqrt(self.csp.detg)
        g11_scaled = sp.diags(self.csp.g[:, 0, 0] / sqrtg)
        g12_scaled = sp.diags(self.csp.g[:, 0, 1] / sqrtg)
        g22_scaled = sp.diags(self.csp.g[:, 1, 1] / sqrtg)

        # matrix that operates on column vector of u1, u2 and produces radial curl
        D_curlr_u1u2 = sp.hstack(((Dxi.dot(g12_scaled) - Deta.dot(g11_scaled)),
                                  (Dxi.dot(g22_scaled) - Deta.dot(g12_scaled))))

        # matrix that transforms theta, phi to u1, u2:
        Ps_dense = self.csp.get_Ps(self.csp.arr_xi, self.csp.arr_eta, block = self.csp.arr_block) # N x 3 x 3
        # extract relevant elements, rearrange so that the matrix operates on (theta, phi) and not (east, north), 
        # and insert in sparse diagonal matrices. Also include the normalization from the Q matrix in Yin et al.:
        rr, rrcosl = self.RI, self.RI * np.cos(np.deg2rad(self.state_grid.lat)) # normalization factors
        Ps00 = sp.diags(-Ps_dense[:, 0, 1] / rr    ) 
        Ps01 = sp.diags( Ps_dense[:, 0, 0] / rrcosl) 
        Ps10 = sp.diags(-Ps_dense[:, 1, 1] / rr    ) 
        Ps11 = sp.diags( Ps_dense[:, 1, 0] / rrcosl)
        # stack:
        Ps = sp.vstack((sp.hstack((Ps00, Ps01)), sp.hstack((Ps10, Ps11))))

        return D_curlr_u1u2.dot(Ps)


    @property
    def fd_curl_matrix(self):
        """ Finite difference curl matrix
        """

        if not hasattr(self, '_fd_curl_matrix'):
            self._fd_curl_matrix = self.calculate_fd_curl_matrix()

        return(self._fd_curl_matrix)


    @property
    def sh_curl_matrix(self):
        """
        Calculate matrix that returns the radial curl when operating on a
        column vector of (theta, phi) vector components, using spherical
        harmonic analysis.

        """

        if not hasattr(self, '_sh_curl_matrix'):
            # Matrix that gets divergence free SH coefficients from vector of (theta, phi)-components on grid via Helmholtz decomposition
            G_df = self.state.basis_evaluator.G_helmholtz_inv[self.state.basis.index_length:, :]

            # Multiply with Laplacian and evaluation matrix to get the curl matrix
            self._sh_curl_matrix = self.state.basis_evaluator.G.dot(self.state.basis.laplacian().reshape((-1, 1)) * G_df)

        return(self._sh_curl_matrix)


    def steady_state_m_ind(self):
        """ Calculate coefficients for induced field in steady state 
            
            Parameters:
            -----------
            m_imp: array, optional
                the coefficient vector for the imposed magnetic field. If None, the
                vector for the current state will be used

            Returns:
            --------
            m_ind_ss: array
                array of coefficients for the induced magnetic field in steady state

        """

        helmholtz_E_jr = self.state.jr_to_helmholtz_E.dot(self.state.jr_on_grid)
        helmholtz_E_cu = self.state.c_to_helmholtz_E.dot(self.state.cu * self.state.ih_constraint_scaling)
        helmholtz_E_noind = helmholtz_E_jr + helmholtz_E_cu + self.state.helmholtz_E_uxB

        G_m_ind_to_helmholtz_E = self.state.G_m_ind_to_helmholtz_E_direct + self.state.G_m_ind_to_helmholtz_E_constraint

        m_ind = -np.linalg.pinv(G_m_ind_to_helmholtz_E[self.state.basis.index_length:, :]).dot(helmholtz_E_noind[self.state.basis.index_length:])

        return(m_ind)