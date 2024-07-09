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
from pynamit.simulation.ionosphere_state import State
from pynamit.various.constants import RE
import scipy.sparse as sp

class I2D(object):
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
                 vector_FAC = True,
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
            'vector_FAC':             int(vector_FAC),
            'vector_conductance':     int(vector_conductance),
            'vector_u':               int(vector_u),
            't0':                     t0,
        })

        self.state_history_exists       = False
        self.FAC_history_exists         = False
        self.conductance_history_exists = False
        self.u_history_exists           = False

        # Will be set to True when the corresponding history is different from the one saved on disk
        self.save_state_history       = False
        self.save_FAC_history         = False
        self.save_conductance_history = False
        self.save_u_history           = False

        self.latest_time = np.float64(0)

        # Overwrite settings with any settings existing on file
        settings_on_file = (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_settings.ncdf')
        if settings_on_file:
            settings = xr.load_dataset(self.result_filename_prefix + '_settings.ncdf')

        # Load PFAC matrix if it exists on file
        PFAC_matrix_on_file = (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_PFAC_matrix.ncdf')
        if PFAC_matrix_on_file:
            PFAC_matrix = xr.load_dataarray(self.result_filename_prefix + '_PFAC_matrix.ncdf')

        self.RI = settings.RI

        mainfield = Mainfield(kind = settings.mainfield_kind,
                              epoch = settings.mainfield_epoch,
                              hI = (settings.RI - RE) * 1e-3,
                              B0 = None if settings.mainfield_B0 == 0 else settings.mainfield_B0)

        self.vector_FAC         = bool(settings.vector_FAC)
        self.vector_conductance = bool(settings.vector_conductance)
        self.vector_u           = bool(settings.vector_u)

        self.csp = CSProjection(settings.Ncs)
        self.num_grid = Grid(theta = self.csp.arr_theta, phi = self.csp.arr_phi)

        self.basis             = SHBasis(settings.Nmax, settings.Mmax)
        self.conductance_basis = SHBasis(settings.Nmax, settings.Mmax, Nmin = 0)
        self.u_basis           = SHBasis(settings.Nmax, settings.Mmax, Nmin = 0)

        self.basis_evaluator             = BasisEvaluator(self.basis,             self.num_grid)
        self.conductance_basis_evaluator = BasisEvaluator(self.conductance_basis, self.num_grid)
        self.u_basis_evaluator           = BasisEvaluator(self.u_basis,           self.num_grid)

        self.b_evaluator = FieldEvaluator(mainfield, self.num_grid, RI)

        # Initialize the state of the ionosphere
        self.state = State(self.basis,
                           self.conductance_basis,
                           self.u_basis,
                           mainfield,
                           self.num_grid, 
                           settings,
                           PFAC_matrix = PFAC_matrix)

        self.load_histories()

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


    def evolve_to_time(self, t, dt = np.float64(5e-4), history_update_interval = 200, history_save_interval = 10, quiet = False):
        """ Evolve to given time

        """

        # Skip the first update if the state has been evolved before (previous endpoint has been accounted for)
        skip_first_update = (self.latest_time > np.float64(0))
 
        count = 0
        while True:
            if not (count == 0 and skip_first_update):
                self.update_FAC()
                self.update_conductance()
                self.update_u()

                self.state.impose_constraints()
                self.state.update_Phi_and_W()

                # Update state history and save if appropriate
                if count % history_update_interval == 0:
                    self.update_state_history()
                    if (count % (history_update_interval * history_save_interval) == 0):
                        self.save_histories()
                        if quiet:
                            pass
                        else:
                            print('Saved output at t = {:.2f} s'.format(self.latest_time), end = '\r')

            next_time = self.latest_time + dt

            if next_time > t + dt / 2: # with safety margin to avoid floating point errors
                break

            self.state.evolve_Br(dt)
            self.latest_time = next_time

            count += 1


    def set_FAC(self, FAC, lat = None, lon = None, theta = None, phi = None, time = None):
        """
        Specify field-aligned current at ``self.num_grid.theta``,
        ``self.num_grid.phi``.

            Parameters
            ----------
            FAC: array
                The field-aligned current, in A/m^2, at
                ``self.num_grid.theta`` and ``self.num_grid.phi``, at
                ``RI``. The values in the array have to match the
                corresponding coordinates.

        """

        self.FAC = np.atleast_2d(FAC)

        self.FAC_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if time is None:
            if self.FAC.shape[0] > 1:
                raise ValueError('Time has to be specified if FAC is given for multiple times')
            time = self.latest_time

        self.FAC_time = np.atleast_1d(time)

        self.next_FAC = 0


    def set_conductance(self, Hall, Pedersen, lat = None, lon = None, theta = None, phi = None, time = None):
        """
        Specify Hall and Pedersen conductance at
        ``self.num_grid.theta``, ``self.num_grid.phi``.

        """

        self.Hall = np.atleast_2d(Hall)
        self.Pedersen = np.atleast_2d(Pedersen)

        self.conductance_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if time is None:
            if self.Hall.shape[0] > 1 or self.Pedersen.shape[0] > 1:
                raise ValueError('Time has to be specified if conductance is given for multiple times')
            time = self.latest_time

        self.conductance_time = np.atleast_1d(time)

        self.next_conductance = 0


    def set_u(self, u, lat = None, lon = None, theta = None, phi = None, time = None):
        """ set neutral wind theta and phi components 
            For now, they *have* to be given on grid
        """

        self.u_theta = np.atleast_2d(u[0])
        self.u_phi = np.atleast_2d(u[1])

        self.u_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if time is None:
            if self.u_theta.shape[0] > 1 or self.u_phi.shape[0] > 1:
                raise ValueError('Time has to be specified if u is given for multiple times')
            time = self.latest_time

        self.u_time = np.atleast_1d(time)

        self.next_u = 0


    def update_FAC(self):
        """ Update FAC """

        if hasattr(self, 'next_FAC') and self.next_FAC < self.FAC_time.size and self.latest_time >= self.FAC_time[self.next_FAC]:
            # Represent as values on num_grid
            Jpar_int = csp.interpolate_scalar(self.FAC[self.next_FAC], self.FAC_grid.theta, self.FAC_grid.phi, self.num_grid.theta, self.num_grid.phi)

            # Extract the radial component of the FAC and set the corresponding basis coefficients
            if self.vector_FAC:
                Jr = Vector(self.basis, basis_evaluator = self.basis_evaluator, grid_values = Jpar_int * self.b_evaluator.br)
            else:
                Jr = Jpar_int * self.b_evaluator.br

            self.state.set_FAC(Jr, self.vector_FAC)
            self.update_FAC_history()

            self.next_FAC += 1


    def update_conductance(self):
        """ Update conductance """

        if hasattr(self, 'next_conductance') and self.next_conductance < self.conductance_time.size and self.latest_time >= self.conductance_time[self.next_conductance]:
            # Check if Pedersen and Hall conductances are positive
            if np.any(self.Hall[self.next_conductance] < 0) or np.any(self.Pedersen[self.next_conductance] < 0):
                raise ValueError('Conductances have to be positive')

            # Transform to resistivities
            etaP = self.Pedersen[self.next_conductance] / (self.Hall[self.next_conductance]**2 + self.Pedersen[self.next_conductance]**2)
            etaH = self.Hall[self.next_conductance]     / (self.Hall[self.next_conductance]**2 + self.Pedersen[self.next_conductance]**2)

            # Represent as values on num_grid
            etaP_int = csp.interpolate_scalar(etaP, self.conductance_grid.theta, self.conductance_grid.phi, self.num_grid.theta, self.num_grid.phi)
            etaH_int = csp.interpolate_scalar(etaH, self.conductance_grid.theta, self.conductance_grid.phi, self.num_grid.theta, self.num_grid.phi)

            if self.vector_conductance:
                # Represent as expansion in spherical harmonics
                etaP = Vector(self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, grid_values = etaP_int)
                etaH = Vector(self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, grid_values = etaH_int)
            else:
                etaP = etaP_int
                etaH = etaH_int

            self.state.set_conductance(etaP, etaH, self.vector_conductance)
            self.update_conductance_history()

            self.next_conductance += 1


    def update_u(self):
        """ Update neutral wind """

        if hasattr(self, 'next_u') and self.next_u < self.u_time.size and self.latest_time >= self.u_time[self.next_u]:
            # Represent as values on num_grid
            u_int = csp.interpolate_vector_components(self.u_phi[self.next_u], -self.u_theta[self.next_u], np.zeros_like(self.u_phi[self.next_u]), self.u_grid.theta, self.u_grid.phi, self.num_grid.theta, self.num_grid.phi)
            u_int_theta, u_int_phi = -u_int[1], u_int[0]

            if self.vector_u:
                # Represent as expansion in spherical harmonics
                u = Vector(self.u_basis, basis_evaluator = self.u_basis_evaluator, grid_values = (u_int_theta, u_int_phi), helmholtz = True)
            else:
                u = (u_int_theta, u_int_phi)

            self.state.set_u(u, self.vector_u)
            self.update_u_history()

            self.next_u += 1


    def update_state_history(self):
        """ Add current state to state history """

        if not self.state_history_exists:
            self.state_history_times = np.array([self.latest_time])
            self.m_imp_history       = np.array(self.state.m_imp.coeffs, dtype = np.float64).reshape((1, -1))
            self.m_ind_history       = np.array(self.state.m_ind.coeffs, dtype = np.float64).reshape((1, -1))
            self.Phi_history         = np.array(self.state.Phi.coeffs,   dtype = np.float64).reshape((1, -1))
            self.W_history           = np.array(self.state.W.coeffs,     dtype = np.float64).reshape((1, -1))

            self.state_history_exists = True
        else:
            self.state_history_times = np.append(self.state_history_times, self.latest_time)
            self.m_imp_history       = np.vstack((self.m_imp_history, self.state.m_imp.coeffs))
            self.m_ind_history       = np.vstack((self.m_ind_history, self.state.m_ind.coeffs))
            self.Phi_history         = np.vstack((self.Phi_history,   self.state.Phi.coeffs))
            self.W_history           = np.vstack((self.W_history,     self.state.W.coeffs))

        self.save_state_history = True


    def update_FAC_history(self):
        """ Add current FAC to FAC history """

        if not self.FAC_history_exists:
            self.FAC_history_times = np.array([self.latest_time])
            self.Jr_history        = np.array(self.state.Jr.coeffs, dtype = np.float64).reshape((1, -1))

            self.FAC_history_exists = True
        else:
            self.FAC_history_times = np.append(self.FAC_history_times, self.latest_time)
            self.Jr_history        = np.vstack((self.Jr_history, self.state.Jr.coeffs))

        self.save_FAC_history = True


    def update_conductance_history(self):
        """ Add current conductance to conductance history """

        if not self.conductance_history_exists:
            self.conductance_history_times = np.array([self.latest_time])
            self.etaP_history              = np.array(self.state.etaP.coeffs, dtype = np.float64).reshape((1, -1))
            self.etaH_history              = np.array(self.state.etaH.coeffs, dtype = np.float64).reshape((1, -1))

            self.conductance_history_exists = True
        else:
            self.conductance_history_times = np.append(self.conductance_history_times, self.latest_time)
            self.etaP_history              = np.vstack((self.etaP_history, self.state.etaP.coeffs))
            self.etaH_history              = np.vstack((self.etaH_history, self.state.etaH.coeffs))

        self.save_conductance_history = True


    def update_u_history(self):
        """ Add current neutral wind to neutral wind history """

        if not self.u_history_exists:
            self.u_history_times = np.array([self.latest_time])
            self.u_cf_history    = np.array(self.state.u.coeffs[0], dtype = np.float64).reshape((1, -1))
            self.u_df_history    = np.array(self.state.u.coeffs[1], dtype = np.float64).reshape((1, -1))

            self.u_history_exists = True
        else:
            self.u_history_times = np.append(self.u_history_times, self.latest_time)
            self.u_cf_history    = np.vstack((self.u_cf_history, self.state.u.coeffs[0]))
            self.u_df_history    = np.vstack((self.u_df_history, self.state.u.coeffs[1]))
        
        self.save_u_history = True


    def save_histories(self):
        """ Store the histories """

        if self.save_state_history:
            
            state_dataset = xr.Dataset(
                data_vars =
                {
                    'SH_m_imp': (['time', 'i'], self.m_imp_history),
                    'SH_m_ind': (['time', 'i'], self.m_ind_history),
                    'SH_Phi':   (['time', 'i'], self.Phi_history),
                    'SH_W':     (['time', 'i'], self.W_history),
                },
                coords = xr.Coordinates.from_pandas_multiindex(pd.MultiIndex.from_arrays([self.basis.n, self.basis.m], names = ['n', 'm']), dim = 'i').merge({'time': self.state_history_times})
            )
            state_dataset.reset_index('i').to_netcdf(self.result_filename_prefix + '_state.ncdf')

            self.save_state_history = False

        if self.save_FAC_history:
            FAC_dataset = xr.Dataset(
                data_vars = {
                    'SH_Jr': (['time', 'i'], self.Jr_history),
                },
                coords = xr.Coordinates.from_pandas_multiindex(pd.MultiIndex.from_arrays([self.basis.n, self.basis.m], names = ['n', 'm']), dim = 'i').merge({'time': self.FAC_history_times})
            )
            
            FAC_dataset.reset_index('i').to_netcdf(self.result_filename_prefix + '_FAC.ncdf')

            self.save_FAC_history = False

        if self.save_conductance_history:
            conductance_dataset = xr.Dataset(
                data_vars = {
                    'SH_etaP': (['time', 'i'], self.etaP_history),
                    'SH_etaH': (['time', 'i'], self.etaH_history),
                },
                coords = xr.Coordinates.from_pandas_multiindex(pd.MultiIndex.from_arrays([self.conductance_basis.n, self.conductance_basis.m], names = ['n', 'm']), dim = 'i').merge({'time': self.conductance_history_times})
            )
            conductance_dataset.reset_index('i').to_netcdf(self.result_filename_prefix + '_conductance.ncdf')

            self.save_conductance_history = False

        if self.save_u_history:
            u_dataset = xr.Dataset(
                data_vars = {
                    'SH_u_cf': (['time', 'i'], self.u_cf_history),
                    'SH_u_df': (['time', 'i'], self.u_df_history),
                },
                coords = xr.Coordinates.from_pandas_multiindex(pd.MultiIndex.from_arrays([self.u_basis.n, self.u_basis.m], names = ['n', 'm']), dim = 'i').merge({'time': self.u_history_times})
            )
            u_dataset.reset_index('i').to_netcdf(self.result_filename_prefix + '_u.ncdf')

            self.save_u_history = False


    def load_histories(self):
        """ Load histories from file """

        # Load state history if it exists on file
        if (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_state.ncdf'):
            state_dataset = xr.load_dataset(self.result_filename_prefix + '_state.ncdf')
            self.m_imp_history       = state_dataset['SH_m_imp'].values
            self.m_ind_history       = state_dataset['SH_m_ind'].values
            self.Phi_history         = state_dataset['SH_Phi'].values
            self.W_history           = state_dataset['SH_W'].values
            self.state_history_times = state_dataset.time.values

            self.state_history_exists = True

            self.latest_time = self.state_history_times[-1]
            self.state.set_coeffs(m_ind = self.m_ind_history[-1])
            self.state.set_coeffs(m_imp = self.m_imp_history[-1])
            self.state.set_coeffs(Phi   = self.Phi_history[-1])
            self.state.set_coeffs(W     = self.W_history[-1])

        # Load FAC history if it exists on file
        if (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_FAC.ncdf'):
            FAC_dataset = xr.load_dataset(self.result_filename_prefix + '_FAC.ncdf')

            self.Jr_history        = FAC_dataset['SH_Jr'].values
            self.FAC_history_times = FAC_dataset.time.values

            self.FAC_history_exists = True

            Jr = Vector(basis = self.basis, basis_evaluator = self.basis_evaluator, coeffs = self.Jr_history[-1])
            self.state.set_FAC(Jr, vector_FAC = True)

        # Load conductance history if it exists on file
        if (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_conductance.ncdf'):
            conductance_dataset = xr.load_dataset(self.result_filename_prefix + '_conductance.ncdf')

            self.etaP_history              = conductance_dataset['SH_etaP'].values
            self.etaH_history              = conductance_dataset['SH_etaH'].values
            self.conductance_history_times = conductance_dataset.time.values

            self.conductance_history_exists = True

            etaP = Vector(basis = self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, coeffs = self.etaP_history[-1])
            etaH = Vector(basis = self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, coeffs = self.etaH_history[-1])
            self.state.set_conductance(etaP, etaH, vector_conductance = True)

        # Load neutral wind history if it exists on file
        if (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_u.ncdf'):
            u_dataset = xr.load_dataset(self.result_filename_prefix + '_u.ncdf')

            self.u_cf_history    = u_dataset['SH_u_cf'].values
            self.u_df_history    = u_dataset['SH_u_df'].values
            self.u_history_times = u_dataset.time.values

            self.u_history_exists = True

            u = Vector(basis = self.u_basis, basis_evaluator = self.u_basis_evaluator, coeffs = np.hstack((self.u_cf_history[-1], self.u_df_history[-1])), helmholtz = True)
            self.state.set_u(u, vector_u = True)


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
            rr, rrcosl = self.RI, self.RI * np.cos(np.deg2rad(self.num_grid.lat)) # normalization factors
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
