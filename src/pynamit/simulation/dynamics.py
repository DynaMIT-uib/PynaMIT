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
                 result_filename_prefix = 'tmp',
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
                 zero_jr_at_dip_equator = False,
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

        self.result_filename_prefix = result_filename_prefix

        # Store setting arguments in xarray dataset
        settings = xr.Dataset(attrs = {
            'Nmax':                   Nmax,
            'Mmax':                   Mmax,
            'Ncs':                    Ncs,
            'RI':                     RI,
            'latitude_boundary':      latitude_boundary,
            'ignore_PFAC':            int(ignore_PFAC),
            'connect_hemispheres':    int(connect_hemispheres),
            'zero_jr_at_dip_equator': int(zero_jr_at_dip_equator),
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
        settings_on_file = (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_settings.ncdf')
        if settings_on_file:
            settings = xr.load_dataset(self.result_filename_prefix + '_settings.ncdf')

        # Load PFAC matrix if it exists on file
        PFAC_matrix_on_file = (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_PFAC_matrix.ncdf')
        if PFAC_matrix_on_file:
            PFAC_matrix = xr.load_dataarray(self.result_filename_prefix + '_PFAC_matrix.ncdf')

        self.RI = settings.RI

        self.mainfield = Mainfield(kind = settings.mainfield_kind,
                                   epoch = settings.mainfield_epoch,
                                   hI = (settings.RI - RE) * 1e-3,
                                   B0 = None if settings.mainfield_B0 == 0 else settings.mainfield_B0)

        self.vector_input = {
            'jr':          bool(settings.vector_jr),
            'conductance': bool(settings.vector_conductance),
            'u':           bool(settings.vector_u),
        }

        self.csp = CSProjection(settings.Ncs)
        self.state_grid = Grid(theta = self.csp.arr_theta, phi = self.csp.arr_phi)

        self.state_basis       = SHBasis(settings.Nmax, settings.Mmax)
        self.input_bases = {
            'jr':          SHBasis(settings.Nmax, settings.Mmax),
            'conductance': SHBasis(settings.Nmax, settings.Mmax, Nmin = 0),
            'u':           SHBasis(settings.Nmax, settings.Mmax),
        }

        self.state_basis_evaluator = BasisEvaluator(self.state_basis, self.state_grid)
        self.input_basis_evaluators = dict([(key, BasisEvaluator(self.input_bases[key], self.state_grid)) for key in self.input_bases.keys()])

        # Initialize the state of the ionosphere
        self.state = State(self.state_basis,
                           self.input_bases['jr'],
                           self.input_bases['conductance'],
                           self.input_bases['u'],
                           self.mainfield,
                           self.state_grid,
                           settings,
                           PFAC_matrix = PFAC_matrix)

        self.input_timeseries = {}
        self.load_timeseries()

        if hasattr(self, 'state_timeseries'):
            if not self.state_timeseries.coords['i'].equals(pd.MultiIndex.from_arrays(self.state_basis.indices, names = self.state_basis.index_names)):
                raise ValueError('The index of the state time series does not match the index of the state basis')
            self.latest_time = np.max(self.state_timeseries.time.values)
            self.state.set_coeffs(m_ind = self.state_timeseries[self.state_basis.short_name + '_m_ind'].sel(time = self.latest_time).values)
            self.state.set_coeffs(m_imp = self.state_timeseries[self.state_basis.short_name + '_m_imp'].sel(time = self.latest_time).values)
            self.state.set_coeffs(Phi   = self.state_timeseries[self.state_basis.short_name + '_Phi'].sel(time = self.latest_time).values)
            self.state.set_coeffs(W     = self.state_timeseries[self.state_basis.short_name + '_W'].sel(time = self.latest_time).values)
        else:
            self.latest_time = np.float64(0)
            self.state.set_coeffs(m_ind = np.zeros(self.state_basis.index_length))

        if self.result_filename_prefix is None:
            self.result_filename_prefix = 'tmp'

        # Save settings if they do not exist on file
        if not settings_on_file:
            settings.to_netcdf(self.result_filename_prefix + '_settings.ncdf')
            print('Saved settings to {}_settings.ncdf'.format(self.result_filename_prefix))

        # Save PFAC matrix if it does not exist on file
        if not PFAC_matrix_on_file:
            self.state.m_imp_to_B_pol.to_netcdf(self.result_filename_prefix + '_PFAC_matrix.ncdf')
            print('Saved PFAC matrix to {}_PFAC_matrix.ncdf'.format(self.result_filename_prefix))

        self.last_input = {}


    def evolve_to_time(self, t, dt = np.float64(5e-4), sampling_step_interval = 200, saving_sample_interval = 10, quiet = False):
        """
        Evolve to the given time `t`. Will overwrite the values
        corresponding to the start time, to account for any changes in
        jr, conductance or neutral wind since the end of the last call
        to `evolve_to_time`.

        """

        # Will be set to True when the corresponding time series is different from the one saved on disk
        self.save_jr         = False
        self.save_conductance = False
        self.save_u           = False

        index = pd.MultiIndex.from_arrays(self.state_basis.indices, names = self.state_basis.index_names)
 
        count = 0
        while True:
            for key in self.input_timeseries.keys():
                self.update_input(key)

            self.state.impose_constraints()
            self.state.update_Phi_and_W()

            if count % sampling_step_interval == 0:
                # Add current state to state time series
                current_state = xr.Dataset(
                    data_vars = {
                        self.state_basis.short_name + '_m_imp': (['time', 'i'], self.state.m_imp.coeffs.reshape((1, -1))),
                        self.state_basis.short_name + '_m_ind': (['time', 'i'], self.state.m_ind.coeffs.reshape((1, -1))),
                        self.state_basis.short_name + '_Phi':   (['time', 'i'], self.state.Phi.coeffs.reshape((1, -1))),
                        self.state_basis.short_name + '_W':     (['time', 'i'], self.state.W.coeffs.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(index, dim = 'i').merge({'time': [self.latest_time]})
                )

                if not hasattr(self, 'state_timeseries'):
                    self.state_timeseries = current_state
                else:
                    self.state_timeseries = xr.concat([self.state_timeseries.drop_sel(time = self.latest_time, errors = 'ignore'), current_state], dim = 'time')

                # Save output if requested
                if (count % (sampling_step_interval * saving_sample_interval) == 0):
                    self.state_timeseries.reset_index('i').to_netcdf(self.result_filename_prefix + '_state.ncdf')
                    if quiet:
                        pass
                    else:
                        print('Saved output at t = {:.2f} s'.format(self.latest_time), end = '\r')

            next_time = self.latest_time + dt

            if next_time > t + FLOAT_ERROR_MARGIN:
                break

            self.state.evolve_Br(dt)
            self.latest_time = next_time

            count += 1


    def set_FAC(self, FAC, lat = None, lon = None, theta = None, phi = None, time = None):
        """
        Set the field-aligned current at the given coordinate points.
        """

        FAC_b_evaluator = FieldEvaluator(self.mainfield, Grid(lat = lat, lon = lon, theta = theta, phi = phi), self.RI)

        self.set_jr(FAC * FAC_b_evaluator.br, lat = lat, lon = lon, theta = theta, phi = phi, time = time)


    def set_jr(self, jr, lat = None, lon = None, theta = None, phi = None, time = None):
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

        key = 'jr'

        current_input = {}
        current_input['jr'] = {'values': np.atleast_2d(jr)}

        input_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if time is None:
            if any([current_input[var][component].shape[0] > 1 for var in current_input.keys() for component in current_input[var].keys()]):
                raise ValueError('Time has to be specified if input is given for multiple times')
            time = self.latest_time

        time = np.atleast_1d(time)

        if self.vector_input[key]:
            indices = self.input_bases[key].indices
            index_names = self.input_bases[key].index_names
        else:
            indices = [self.state_grid.theta, self.state_grid.phi]
            index_names = ['theta', 'phi']

        index = pd.MultiIndex.from_arrays(indices, names = index_names)

        for i in range(time.size):
            # Interpolate to state_grid
            jr_int = csp.interpolate_scalar(current_input['jr']['values'][i], input_grid.theta, input_grid.phi, self.state_grid.theta, self.state_grid.phi)

            # Extract the radial component of the jr and set the corresponding basis coefficients
            if self.vector_input[key]:
                # Represent as expansion in spherical harmonics
                jr_vector = Vector(self.input_bases[key], basis_evaluator = self.input_basis_evaluators[key], grid_values = jr_int)

                current_jr = xr.Dataset(
                    data_vars = {
                        self.input_bases[key].short_name + '_jr': (['time', 'i'], jr_vector.coeffs.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(index, dim = 'i').merge({'time': [time[i]]})
                )
            else:
                # Represent as values on state_grid
                current_jr = xr.Dataset(
                    data_vars = {
                        'GRID_jr': (['time', 'i'], jr_int.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(index, dim = 'i').merge({'time': [time[i]]})
                )

            # Add to the jr time series
            if key not in self.input_timeseries.keys():
                self.input_timeseries[key] = current_jr
            else:
                self.input_timeseries[key] = xr.concat([self.input_timeseries[key].drop_sel(time = time[i], errors = 'ignore'), current_jr], dim = 'time')

        # Save the jr time series
        self.input_timeseries[key].reset_index('i').to_netcdf(self.result_filename_prefix + '_' + key + '.ncdf')


    def set_conductance(self, Hall, Pedersen, lat = None, lon = None, theta = None, phi = None, time = None):
        """
        Specify Hall and Pedersen conductance at
        ``self.state_grid.theta``, ``self.state_grid.phi``.

        """

        key = 'conductance'

        current_input = {}

        Hall = np.atleast_2d(Hall)
        Pedersen = np.atleast_2d(Pedersen)

        current_input['etaP'] = {'values': np.empty_like(Pedersen)}
        for i in range(max(current_input['etaP']['values'].shape[0], 1)):
            current_input['etaP']['values'][i] = Pedersen[i] / (Hall[i]**2 + Pedersen[i]**2)

        current_input['etaH'] = {'values': np.empty_like(Hall)}
        for i in range(max(current_input['etaH']['values'].shape[0], 1)):
            current_input['etaH']['values'][i] = Hall[i] / (Hall[i]**2 + Pedersen[i]**2)

        input_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if time is None:
            if any([current_input[var][component].shape[0] > 1 for var in current_input.keys() for component in current_input[var].keys()]):
                raise ValueError('Time has to be specified if input is given for multiple times')
            time = self.latest_time

        time = np.atleast_1d(time)

        if self.vector_input[key]:
            indices = self.input_bases[key].indices
            index_names = self.input_bases[key].index_names
        else:
            indices = [self.state_grid.theta, self.state_grid.phi]
            index_names = ['theta', 'phi']

        index = pd.MultiIndex.from_arrays(indices, names = index_names)

        for i in range(time.size):
            # Interpolate to state_grid
            etaP_int = csp.interpolate_scalar(current_input['etaP']['values'][i], input_grid.theta, input_grid.phi, self.state_grid.theta, self.state_grid.phi)
            etaH_int = csp.interpolate_scalar(current_input['etaH']['values'][i], input_grid.theta, input_grid.phi, self.state_grid.theta, self.state_grid.phi)

            if self.vector_input[key]:
                # Represent as expansion in spherical harmonics
                etaP_vector = Vector(self.input_bases[key], basis_evaluator = self.input_basis_evaluators[key], grid_values = etaP_int)
                etaH_vector = Vector(self.input_bases[key], basis_evaluator = self.input_basis_evaluators[key], grid_values = etaH_int)

                current_conductance = xr.Dataset(
                    data_vars = {
                        self.input_bases[key].short_name + '_etaP': (['time', 'i'], etaP_vector.coeffs.reshape((1, -1))),
                        self.input_bases[key].short_name + '_etaH': (['time', 'i'], etaH_vector.coeffs.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(index, dim = 'i').merge({'time': [time[i]]})
                )
            else:
                # Represent as values on state_grid
                current_conductance = xr.Dataset(
                    data_vars = {
                        'GRID_etaP': (['time', 'i'], etaP_int.reshape((1, -1))),
                        'GRID_etaH': (['time', 'i'], etaH_int.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(index, dim = 'i').merge({'time': [time[i]]})
                )

            # Add to the conductance time series
            if key not in self.input_timeseries.keys():
                self.input_timeseries[key] = current_conductance
            else:
                self.input_timeseries[key] = xr.concat([self.input_timeseries[key].drop_sel(time = time[i], errors = 'ignore'), current_conductance], dim = 'time')

        # Save the conductance time series
        self.input_timeseries[key].reset_index('i').to_netcdf(self.result_filename_prefix + '_' + key + '.ncdf')


    def set_u(self, u, lat = None, lon = None, theta = None, phi = None, time = None):
        """ set neutral wind theta and phi components 
            For now, they *have* to be given on grid
        """

        key = 'u'

        current_input = {}
        current_input['u'] = {'theta': np.atleast_2d(u[0]), 'phi': np.atleast_2d(u[1])}

        input_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if time is None:
            if any([current_input[var][component].shape[0] > 1 for var in current_input.keys() for component in current_input[var].keys()]):
                raise ValueError('Time has to be specified if input is given for multiple times')
            time = self.latest_time

        time = np.atleast_1d(time)

        if self.vector_input[key]:
            indices = self.input_bases[key].indices
            index_names = self.input_bases[key].index_names
        else:
            indices = [self.state_grid.theta, self.state_grid.phi]
            index_names = ['theta', 'phi']

        index = pd.MultiIndex.from_arrays([np.tile(indices[i], 2) for i in range(len(indices))], names = index_names)

        for i in range(time.size):
            # Interpolate to state_grid
            u_int_east, u_int_north, _ = csp.interpolate_vector_components(current_input['u']['phi'], -current_input['u']['theta'][i], np.zeros_like(current_input['u']['phi'][i]), input_grid.theta, input_grid.phi, self.state_grid.theta, self.state_grid.phi)
            u_int = np.hstack((-u_int_north, u_int_east)) # convert to theta, phi

            if self.vector_input[key]:
                # Represent as expansion in spherical harmonics
                u_vector = Vector(self.input_bases[key], basis_evaluator = self.input_basis_evaluators[key], grid_values = u_int, helmholtz = True)

                current_u = xr.Dataset(
                    data_vars = {
                        self.input_bases[key].short_name + '_u': (['time', 'i'], u_vector.coeffs.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(index, dim = 'i').merge({'time': [time[i]]})
                )
            else:
                # Represent as values on state_grid
                current_u = xr.Dataset(
                    data_vars = {
                        'GRID_u': (['time', 'i'], u_int.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(index, dim = 'i').merge({'time': [time[i]]})
                )

            # Add to the neutral wind time series
            if key not in self.input_timeseries.keys():
                self.input_timeseries[key] = current_u
            else:
                self.input_timeseries[key] = xr.concat([self.input_timeseries[key].drop_sel(time = time[i], errors = 'ignore'), current_u], dim = 'time')

        # Save the neutral wind timeseries
        self.input_timeseries[key].reset_index('i').to_netcdf(self.result_filename_prefix + '_' + key + '.ncdf')


    def update_input(self, key):
        """ Update input """

        if key == 'jr':
            vars = {'jr': 'scalar'}
        elif key == 'conductance':
            vars = {'etaP': 'scalar', 'etaH': 'scalar'}
        elif key == 'u':
            vars = {'u': 'tangential'}
        else:
            raise ValueError('Unknown input key')

        dataset = self.input_timeseries[key].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad')
        last_input_exists = all([var in self.last_input.keys() for var in vars])

        current_input = {}
        if self.vector_input[key]:
            for var in vars:
                current_input[var] = Vector(basis = self.input_bases[key], basis_evaluator = self.input_basis_evaluators[key], coeffs = dataset[self.input_bases[key].short_name + '_' + var].values, helmholtz = (vars[var] == 'tangential'))
            if last_input_exists:
                close_to_last = all([np.allclose(current_input[var].coeffs, self.last_input[var].coeffs) for var in vars])
        else:
            for var in vars:
                current_input[var] = dataset['GRID_' + var].values
            if last_input_exists:
                close_to_last = all([np.allclose(current_input[var], self.last_input[var]) for var in vars])

        # Check if input has changed since last update
        if (not last_input_exists) or (not close_to_last):
            if key == 'jr':
                self.state.set_jr(current_input['jr'], self.vector_input[key])
            elif key == 'conductance':
                self.state.set_conductance(current_input['etaP'], current_input['etaH'], self.vector_input[key])
            elif key == 'u':
                self.state.set_u(current_input['u'], self.vector_input[key])
            for var in vars:
                self.last_input[var] = current_input[var]


    def load_timeseries(self):
        """ Load time series from file """

        if (self.result_filename_prefix is not None):
            # Load state time series if it exists on file
            if os.path.exists(self.result_filename_prefix + '_state.ncdf'):
                state_timeseries = xr.load_timeseries(self.result_filename_prefix + '_state.ncdf')

                state_basis_index = pd.MultiIndex.from_arrays([state_timeseries[self.state_basis.index_names[i]].values for i in range(len(self.state_basis.index_names))], names = self.state_basis.index_names)
                state_coords = xr.Coordinates.from_pandas_multiindex(state_basis_index, dim = 'i').merge({'time': state_timeseries.time.values})
                self.state_timeseries = state_timeseries.drop_vars(['m', 'n']).assign_coords(state_coords)

            # Load input time series if they exist on file
            for key in self.input_bases.keys():
                if os.path.exists(self.result_filename_prefix + '_' + key + '.ncdf'):
                    self.input_timeseries[key] = xr.load_timeseries(self.result_filename_prefix + '_' + key + '.ncdf')

                    if self.vector_input[key]:
                        basis_labels = self.input_bases[key].index_names
                    else:
                        basis_labels = ['theta', 'phi']

                    basis_index = pd.MultiIndex.from_arrays([self.input_timeseries[key][basis_labels[i]].values for i in range(len(basis_labels))], names = basis_labels)
                    coords = xr.Coordinates.from_pandas_multiindex(basis_index, dim = 'i').merge({'time': self.input_timeseries[key].time.values})
                    self.input_timeseries[key] = self.input_timeseries[key].drop_vars(basis_labels).assign_coords(coords)

    @property
    def fd_curl_matrix(self, stencil_size = 1, interpolation_points = 4):
        """ Calculate matrix that returns the radial curl, using finite differences 
            when operated on a column vector of (theta, phi) vector components. 
            The function also returns the pseudo-inverse of the matrix. 
        """

        if not hasattr(self, '_fd_curl_matrix'):
            
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

            # combine:
            self._fd_curl_matrix = D_curlr_u1u2.dot(Ps)

        return(self._fd_curl_matrix)


    def steady_state_m_ind(self, m_imp = None):
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
        
        GVJ = self.state.G_m_ind_to_JS
        GTJ = self.state.G_m_imp_to_JS

        br, bt, bp = self.state.b_evaluator.br, self.state.b_evaluator.btheta, self.state.b_evaluator.bphi
        eP, eH = self.state.etaP_on_grid, self.state.etaH_on_grid
        C00 = sp.diags(eP * (bp**2 + br**2))
        C01 = sp.diags(eP * (-bt * bp) + eH * br)
        C10 = sp.diags(eP * (-bt * bp) - eH * br)
        C11 = sp.diags(eP * (bt**2 + br**2))
        C = sp.vstack((sp.hstack((C00, C01)), sp.hstack((C10, C11))))

        uxb = np.hstack((self.state.uxB_theta, self.state.uxB_phi))

        GcCGVJ = self.fd_curl_matrix.dot(C).dot(GVJ)
        GcCGTJ = self.fd_curl_matrix.dot(C).dot(GTJ)

        if m_imp is None:
            m_imp = self.state.m_imp.coeffs

        m_ind_ss = np.linalg.pinv(GcCGVJ, rcond = 0).dot(self.fd_curl_matrix.dot(uxb) - GcCGTJ.dot(m_imp))

        return(m_ind_ss)
