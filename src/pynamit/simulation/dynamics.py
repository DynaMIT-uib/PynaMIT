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

        self.basis_evaluators = dict([(key, BasisEvaluator(self.bases[key], self.state_grid)) for key in self.bases.keys()])

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

        self.indices = {}
        for key in self.vars.keys():
            if self.vector_storage[key]:
                indices = self.bases[key].indices
                index_names = self.bases[key].index_names
            else:
                indices = [self.state_grid.theta, self.state_grid.phi]
                index_names = ['theta', 'phi']

            if all(self.vars[key][var] == 'scalar' for var in self.vars[key]):
                self.indices[key] = pd.MultiIndex.from_arrays(indices, names = index_names)
            elif all(self.vars[key][var] == 'tangential' for var in self.vars[key]):
                self.indices[key] = pd.MultiIndex.from_arrays([np.tile(indices[i], 2) for i in range(len(indices))], names = index_names)
            else:
                raise ValueError('Mixed scalar and tangential input (unsupported), or unknown input type')

        # Initialize the state of the ionosphere
        self.state = State(self.bases['state'],
                           self.bases['jr'],
                           self.bases['conductance'],
                           self.bases['u'],
                           self.mainfield,
                           self.state_grid,
                           settings,
                           PFAC_matrix = PFAC_matrix)

        self.timeseries = {}
        self.load_timeseries()

        if 'state' in self.timeseries.keys():
            self.latest_time = np.max(self.timeseries['state'].time.values)
            self.select_timeseries_data('state')
        else:
            self.latest_time = np.float64(0)
            self.state.set_coeffs(m_ind = np.zeros(self.bases['state'].index_length))

        if self.dataset_filename_prefix is None:
            self.dataset_filename_prefix = 'simulation'

        if not settings_on_file:
            self.save_dataset(settings, 'settings')
            print('Saved settings to {}_settings.ncdf'.format(self.dataset_filename_prefix))

        if not PFAC_matrix_on_file:
            self.save_dataset(self.state.m_imp_to_B_pol, 'PFAC_matrix')
            print('Saved PFAC matrix to {}_PFAC_matrix.ncdf'.format(self.dataset_filename_prefix))


    def evolve_to_time(self, t, dt = np.float64(5e-4), sampling_step_interval = 200, saving_sample_interval = 10, quiet = False):
        """
        Evolve to the given time `t`. Will overwrite the values
        corresponding to the start time, to account for any changes in
        jr, conductance or neutral wind since the end of the last call
        to `evolve_to_time`.

        """

        # Will be set to True when the corresponding time series is different from the one saved on disk
        self.save_jr          = False
        self.save_conductance = False
        self.save_u           = False
 
        count = 0
        while True:
            keys = list(self.timeseries.keys())
            if 'state' in keys:
                keys.remove('state')
            if keys is not None:
                for key in keys:
                    self.select_timeseries_data(key)

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
                    coords = xr.Coordinates.from_pandas_multiindex(self.indices['state'], dim = 'i').merge({'time': [self.latest_time]})
                )

                self.add_to_timeseries(current_state_dataset, 'state')

                # Save output if requested
                if (count % (sampling_step_interval * saving_sample_interval) == 0):
                    self.save_timeseries('state')

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


    def set_FAC(self, FAC, lat = None, lon = None, theta = None, phi = None, times = None):
        """
        Set the field-aligned current at the given coordinate points.
        """

        FAC_b_evaluator = FieldEvaluator(self.mainfield, Grid(lat = lat, lon = lon, theta = theta, phi = phi), self.RI)

        self.set_jr(FAC * FAC_b_evaluator.br, lat = lat, lon = lon, theta = theta, phi = phi, times = times)


    def set_jr(self, jr, lat = None, lon = None, theta = None, phi = None, times = None):
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

        input_values = {
            'jr': {'values': np.atleast_2d(jr)},
        }

        self.set_input('jr', input_values, lat = lat, lon = lon, theta = theta, phi = phi, times = times)


    def set_conductance(self, Hall, Pedersen, lat = None, lon = None, theta = None, phi = None, times = None):
        """
        Specify Hall and Pedersen conductance at
        ``self.state_grid.theta``, ``self.state_grid.phi``.

        """

        input_values = {}

        Hall = np.atleast_2d(Hall)
        Pedersen = np.atleast_2d(Pedersen)

        input_values = {
            'etaP': {'values': np.empty_like(Pedersen)},
            'etaH': {'values': np.empty_like(Hall)},
        }

        # Convert to resistivity
        for i in range(max(input_values['etaP']['values'].shape[0], 1)):
            input_values['etaP']['values'][i] = Pedersen[i] / (Hall[i]**2 + Pedersen[i]**2)

        for i in range(max(input_values['etaH']['values'].shape[0], 1)):
            input_values['etaH']['values'][i] = Hall[i] / (Hall[i]**2 + Pedersen[i]**2)

        self.set_input('conductance', input_values, lat = lat, lon = lon, theta = theta, phi = phi, times = times)


    def set_u(self, u, lat = None, lon = None, theta = None, phi = None, times = None):
        """ set neutral wind theta and phi components 
            For now, they *have* to be given on grid
        """

        input_values = {
            'u': {'theta': np.atleast_2d(u[0]), 'phi': np.atleast_2d(u[1])},
        }

        self.set_input('u', input_values, lat = lat, lon = lon, theta = theta, phi = phi, times = times)


    def set_input(self, key, input_values, lat = None, lon = None, theta = None, phi = None, times = None):
        """ Set input. """

        input_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if times is None:
            if any([input_values[var][component].shape[0] > 1 for var in input_values.keys() for component in input_values[var].keys()]):
                raise ValueError('Times have to be specified if input is given for multiple times')
            times = self.latest_time

        times = np.atleast_1d(times)

        for time in range(times.size):
            current_data = {}
            for var in self.vars[key]:
                # Interpolate to state_grid
                if self.vars[key][var] == 'scalar':
                    interpolated = csp.interpolate_scalar(input_values[var]['values'][time], input_grid.theta, input_grid.phi, self.state_grid.theta, self.state_grid.phi)
                elif self.vars[key][var] == 'tangential':
                    interpolated_east, interplated_north, _ = csp.interpolate_vector_components(input_values[var]['phi'], -input_values[var]['theta'][time], np.zeros_like(input_values[var]['phi'][time]), input_grid.theta, input_grid.phi, self.state_grid.theta, self.state_grid.phi)
                    interpolated = np.hstack((-interplated_north, interpolated_east)) # convert to theta, phi

                if self.vector_storage[key]:
                    vector = Vector(self.bases[key], basis_evaluator = self.basis_evaluators[key], grid_values = interpolated, type = self.vars[key][var])
                    current_data[self.bases[key].short_name + '_' + var] = (['time', 'i'], vector.coeffs.reshape((1, -1)))
                else:
                    current_data['GRID_' + var] = (['time', 'i'], interpolated.reshape((1, -1)))

            current_dataset = xr.Dataset(
                data_vars = current_data,
                coords = xr.Coordinates.from_pandas_multiindex(self.indices[key], dim = 'i').merge({'time': [times[time]]})
            )

            self.add_to_timeseries(current_dataset, key)

        self.save_timeseries(key)


    def add_to_timeseries(self, dataset, key):
        """ Add a dataset to the time series. """

        if key not in self.timeseries.keys():
            self.timeseries[key] = dataset
        else:
            self.timeseries[key] = xr.concat([self.timeseries[key].drop_sel(time = dataset.time, errors = 'ignore'), dataset], dim = 'time')


    def select_timeseries_data(self, key):
        """ Select time series data corresponding to the latest time. """

        dataset = self.timeseries[key].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad')

        if not hasattr(self, 'last_data'):
            self.last_data = {}

        last_data_exists = all([var in self.last_data.keys() for var in self.vars[key]])

        current_data = {}

        if self.vector_storage[key]:
            for var in self.vars[key]:
                current_data[var] = Vector(basis = self.bases[key], basis_evaluator = self.basis_evaluators[key], coeffs = dataset[self.bases[key].short_name + '_' + var].values, type = self.vars[key][var])
            if last_data_exists:
                close_to_last_data = all([np.allclose(current_data[var].coeffs, self.last_data[var].coeffs) for var in self.vars[key]])
        else:
            for var in self.vars[key]:
                current_data[var] = dataset['GRID_' + var].values
            if last_data_exists:
                close_to_last_data = all([np.allclose(current_data[var], self.last_data[var]) for var in self.vars[key]])

        if (not last_data_exists) or (not close_to_last_data):
            if key == 'state':
                self.state.set_coeffs(m_ind = current_data['m_ind'].coeffs)
                self.state.set_coeffs(m_imp = current_data['m_imp'].coeffs)
                self.state.set_coeffs(Phi   = current_data['Phi'].coeffs)
                self.state.set_coeffs(W     = current_data['W'].coeffs)
            if key == 'jr':
                self.state.set_jr(current_data['jr'], self.vector_storage[key])
            elif key == 'conductance':
                self.state.set_conductance(current_data['etaP'], current_data['etaH'], self.vector_storage[key])
            elif key == 'u':
                self.state.set_u(current_data['u'], self.vector_storage[key])
            for var in self.vars[key]:
                self.last_data[var] = current_data[var]


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
                        basis_labels = self.bases[key].index_names
                    else:
                        basis_labels = ['theta', 'phi']

                    basis_index = pd.MultiIndex.from_arrays([dataset[basis_labels[i]].values for i in range(len(basis_labels))], names = basis_labels)
                    coords = xr.Coordinates.from_pandas_multiindex(basis_index, dim = 'i').merge({'time': dataset.time.values})
                    self.timeseries[key] = dataset.drop_vars(basis_labels).assign_coords(coords)


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
            self._fd_curl_matrix = self.calcualte_fd_curl_matrix()

        return(self._fd_curl_matrix)


    def calculate_sh_curl_matrix(self, helmholtz = True):
        """ Calculate matrix that returns the radial curl, using spherical harmonic analysis 
            when operated on a column vector of (theta, phi) vector components. 
        """
            
        # matrix that gets SH coefficients from vector of (theta, phi)-components on grid:
        if helmholtz: # estimate coefficients for both DF and CF parts:
            G_grid_to_coeffs = self.state.basis_evaluator.G_helmholtz_inv
            Nc = self.state.basis.index_length
            G_grid_to_coeffs =  sp.hstack((sp.eye(Nc) * 0, sp.eye(Nc))) * G_grid_to_coeffs # select only DF coefficients
        else:
            G_grid_to_coeffs = self.state.basis_evaluator.G_rxgrad_inv

        coeffs_to_curl = sp.diags(self.state.basis.laplacian())

        # combine and return:
        return(self.state.basis_evaluator.G.dot(coeffs_to_curl.dot(G_grid_to_coeffs)))


    @property
    def sh_curl_matrix(self, helmoltz = True):
        """ Calculate matrix that returns the radial curl, using spherical harmonic analysis 
            when operated on a column vector of (theta, phi) vector components. 
        """

        if not hasattr(self, '_sh_curl_matrix'):
            self._sh_curl_matrix = self.calculate_sh_curl_matrix()

        return(self._sh_curl_matrix)


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

        GcCGVJ = self.sh_curl_matrix.dot(C).dot(GVJ)
        GcCGTJ = self.sh_curl_matrix.dot(C).dot(GTJ)

        if m_imp is None:
            m_imp = self.state.m_imp.coeffs

        m_ind_ss = np.linalg.pinv(GcCGVJ, rcond = 0).dot(self.sh_curl_matrix.dot(uxb) - GcCGTJ.dot(m_imp))

        return(m_ind_ss)
