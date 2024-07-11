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

        self.vector_jr          = bool(settings.vector_jr)
        self.vector_conductance = bool(settings.vector_conductance)
        self.vector_u           = bool(settings.vector_u)

        self.csp = CSProjection(settings.Ncs)
        self.num_grid = Grid(theta = self.csp.arr_theta, phi = self.csp.arr_phi)

        self.state_basis       = SHBasis(settings.Nmax, settings.Mmax)
        self.jr_basis          = SHBasis(settings.Nmax, settings.Mmax)
        self.conductance_basis = SHBasis(settings.Nmax, settings.Mmax, Nmin = 0)
        self.u_basis           = SHBasis(settings.Nmax, settings.Mmax)

        self.state_basis_evaluator       = BasisEvaluator(self.state_basis,       self.num_grid)
        self.jr_basis_evaluator          = BasisEvaluator(self.jr_basis,          self.num_grid)
        self.conductance_basis_evaluator = BasisEvaluator(self.conductance_basis, self.num_grid)
        self.u_basis_evaluator           = BasisEvaluator(self.u_basis,           self.num_grid)

        # Initialize the state of the ionosphere
        self.state = State(self.state_basis,
                           self.jr_basis,
                           self.conductance_basis,
                           self.u_basis,
                           self.mainfield,
                           self.num_grid, 
                           settings,
                           PFAC_matrix = PFAC_matrix)

        self.load_timeseries()

        if hasattr(self, 'state_timeseries'):
            if not self.state_timeseries.coords['i'].equals(self.state_basis.index):
                raise ValueError('The index of the state time series does not match the index of the state basis')
            self.latest_time = np.max(self.state_timeseries.time.values)
            self.state.set_coeffs(m_ind = self.state_timeseries[self.state_basis.short_name + '_m_ind'].sel(time = self.latest_time).values)
            self.state.set_coeffs(m_imp = self.state_timeseries[self.state_basis.short_name + '_m_imp'].sel(time = self.latest_time).values)
            self.state.set_coeffs(Phi   = self.state_timeseries[self.state_basis.short_name + '_Phi'].sel(time = self.latest_time).values)
            self.state.set_coeffs(W     = self.state_timeseries[self.state_basis.short_name + '_W'].sel(time = self.latest_time).values)
        else:
            self.latest_time = np.float64(0)
            self.state.set_coeffs(m_ind = np.zeros(len(self.state_basis.index)))

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
 
        count = 0
        while True:
            self.update_jr()
            self.update_conductance()
            self.update_u()

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
                    coords = xr.Coordinates.from_pandas_multiindex(self.state_basis.index, dim = 'i').merge({'time': [self.latest_time]})
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
        Specify radial current at ``self.num_grid.theta``,
        ``self.num_grid.phi``.

            Parameters
            ----------
            jr: array
                The radial current, in A/m^2, at
                ``self.num_grid.theta`` and ``self.num_grid.phi``, at
                ``RI``. The values in the array have to match the
                corresponding coordinates.

        """

        jr = np.atleast_2d(jr)

        jr_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if time is None:
            if jr.shape[0] > 1:
                raise ValueError('Time has to be specified if jr is given for multiple times')
            time = self.latest_time

        time = np.atleast_1d(time)

        for i in range(time.size):
            # Interpolate to num_grid
            jr_int = csp.interpolate_scalar(jr[i], jr_grid.theta, jr_grid.phi, self.num_grid.theta, self.num_grid.phi)
            
            # Extract the radial component of the jr and set the corresponding basis coefficients
            if self.vector_jr:
                # Represent as expansion in spherical harmonics
                jr = Vector(self.jr_basis, basis_evaluator = self.jr_basis_evaluator, grid_values = jr_int)

                current_jr = xr.Dataset(
                    data_vars = {
                        self.jr_basis.short_name + '_jr': (['time', 'i'], jr.coeffs.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(self.jr_basis.index, dim = 'i').merge({'time': [time[i]]})
                )
            else:
                # Represent as values on num_grid
                current_jr = xr.Dataset(
                    data_vars = {
                        'GRID_jr': (['time', 'i'], jr_int.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(pd.MultiIndex.from_arrays([self.num_grid.theta, self.num_grid.phi], names = ['theta', 'phi']), dim = 'i').merge({'time': [time[i]]})
                )

            # Add to the jr time series
            if not hasattr(self, 'jr_timeseries'):
                self.jr_timeseries = current_jr
            else:
                self.jr_timeseries = xr.concat([self.jr_timeseries.drop_sel(time = time[i], errors = 'ignore'), current_jr], dim = 'time')

        # Save the jr time series
        self.jr_timeseries.reset_index('i').to_netcdf(self.result_filename_prefix + '_jr.ncdf')


    def set_conductance(self, Hall, Pedersen, lat = None, lon = None, theta = None, phi = None, time = None):
        """
        Specify Hall and Pedersen conductance at
        ``self.num_grid.theta``, ``self.num_grid.phi``.

        """

        Hall = np.atleast_2d(Hall)
        Pedersen = np.atleast_2d(Pedersen)

        conductance_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if time is None:
            if Hall.shape[0] > 1 or Pedersen.shape[0] > 1:
                raise ValueError('Time has to be specified if conductance is given for multiple times')
            time = self.latest_time

        time = np.atleast_1d(time)

        for i in range(time.size):
            # Transform to resistivities
            etaP = Pedersen[i] / (Hall[i]**2 + Pedersen[i]**2)
            etaH = Hall[i]     / (Hall[i]**2 + Pedersen[i]**2)

            # Interpolate to num_grid
            etaP_int = csp.interpolate_scalar(etaP, conductance_grid.theta, conductance_grid.phi, self.num_grid.theta, self.num_grid.phi)
            etaH_int = csp.interpolate_scalar(etaH, conductance_grid.theta, conductance_grid.phi, self.num_grid.theta, self.num_grid.phi)

            if self.vector_conductance:
                # Represent as expansion in spherical harmonics
                etaP = Vector(self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, grid_values = etaP_int)
                etaH = Vector(self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, grid_values = etaH_int)

                current_conductance = xr.Dataset(
                    data_vars = {
                        self.conductance_basis.short_name + '_etaP': (['time', 'i'], etaP.coeffs.reshape((1, -1))),
                        self.conductance_basis.short_name + '_etaH': (['time', 'i'], etaH.coeffs.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(self.conductance_basis.index, dim = 'i').merge({'time': [time[i]]})
                )
            else:
                # Represent as values on num_grid
                etaP = etaP_int
                etaH = etaH_int

                current_conductance = xr.Dataset(
                    data_vars = {
                        'GRID_etaP': (['time', 'i'], etaP.reshape((1, -1))),
                        'GRID_etaH': (['time', 'i'], etaH.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(pd.MultiIndex.from_arrays([self.num_grid.theta, self.num_grid.phi], names = ['theta', 'phi']), dim = 'i').merge({'time': [time[i]]})
                )

            # Add to the conductance time series
            if not hasattr(self, 'conductance_timeseries'):
                self.conductance_timeseries = current_conductance
            else:
                self.conductance_timeseries = xr.concat([self.conductance_timeseries.drop_sel(time = time[i], errors = 'ignore'), current_conductance], dim = 'time')

        # Save the conductance time series
        self.conductance_timeseries.reset_index('i').to_netcdf(self.result_filename_prefix + '_conductance.ncdf')


    def set_u(self, u, lat = None, lon = None, theta = None, phi = None, time = None):
        """ set neutral wind theta and phi components 
            For now, they *have* to be given on grid
        """

        u_theta = np.atleast_2d(u[0])
        u_phi = np.atleast_2d(u[1])

        u_grid = Grid(lat = lat, lon = lon, theta = theta, phi = phi)

        if time is None:
            if u_theta.shape[0] > 1 or u_phi.shape[0] > 1:
                raise ValueError('Time has to be specified if u is given for multiple times')
            time = self.latest_time

        time = np.atleast_1d(time)

        for i in range(time.size):
            # Interpolate to num_grid
            u_int = csp.interpolate_vector_components(u_phi[i], -u_theta[i], np.zeros_like(u_phi[i]), u_grid.theta, u_grid.phi, self.num_grid.theta, self.num_grid.phi)
            u_int_theta, u_int_phi = -u_int[1], u_int[0]

            if self.vector_u:
                # Represent as expansion in spherical harmonics
                u = Vector(self.u_basis, basis_evaluator = self.u_basis_evaluator, grid_values = (u_int_theta, u_int_phi), helmholtz = True)

                current_u = xr.Dataset(
                    data_vars = {
                        self.u_basis.short_name + '_u_cf': (['time', 'i'], u.coeffs[0].reshape((1, -1))),
                        self.u_basis.short_name + '_u_df': (['time', 'i'], u.coeffs[1].reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(self.u_basis.index, dim = 'i').merge({'time': [time[i]]})
                )
            else:
                # Represent as values on num_grid
                current_u = xr.Dataset(
                    data_vars = {
                        'GRID_u_theta': (['time', 'i'], u_int_theta.reshape((1, -1))),
                        'GRID_u_phi':   (['time', 'i'], u_int_phi.reshape((1, -1))),
                    },
                    coords = xr.Coordinates.from_pandas_multiindex(pd.MultiIndex.from_arrays([self.num_grid.theta, self.num_grid.phi], names = ['theta', 'phi']), dim = 'i').merge({'time': [time[i]]})
                )

            # Add to the neutral wind time series
            if not hasattr(self, 'u_timeseries'):
                self.u_timeseries = current_u
            else:
                self.u_timeseries = xr.concat([self.u_timeseries.drop_sel(time = time[i], errors = 'ignore'), current_u], dim = 'time')

        # Save the neutral wind timeseries
        self.u_timeseries.reset_index('i').to_netcdf(self.result_filename_prefix + '_u.ncdf')


    def update_jr(self):
        """ Update jr """

        if hasattr(self, 'jr_timeseries'):
            # Use xarray sel with padding to get the jr values at the current time
            if self.vector_jr:
                self.current_jr = Vector(basis = self.jr_basis, basis_evaluator = self.jr_basis_evaluator, coeffs = self.jr_timeseries[self.jr_basis.short_name + '_jr'].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad').values)
            else:
                self.current_jr = self.jr_timeseries['GRID_jr'].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad').values

            # Check if current jr is different from the one used in the last call to update_jr
            if not hasattr(self, 'last_jr') or (self.vector_jr and not np.allclose(self.current_jr.coeffs, self.last_jr.coeffs)) or (not self.vector_jr and not np.allclose(self.current_jr, self.last_jr)):
                self.state.set_jr(self.current_jr, self.vector_jr)
                self.last_jr = self.current_jr


    def update_conductance(self):
        """ Update conductance """

        if hasattr(self, 'conductance_timeseries'):
            # Use xarray sel with padding to get the conductance values at the current time
            if self.vector_conductance:
                self.current_etaP = Vector(basis = self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, coeffs = self.conductance_timeseries[self.conductance_basis.short_name + '_etaP'].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad').values)
                self.current_etaH = Vector(basis = self.conductance_basis, basis_evaluator = self.conductance_basis_evaluator, coeffs = self.conductance_timeseries[self.conductance_basis.short_name + '_etaH'].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad').values)
            else:
                self.current_etaP = self.conductance_timeseries['GRID_etaP'].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad').values
                self.current_etaH = self.conductance_timeseries['GRID_etaH'].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad').values

            # Check if current etaP or etaH are different from the ones used in the last call to update_conductance
            if not (hasattr(self, 'last_etaP') and hasattr(self, 'last_etaH')) or (self.vector_conductance and (not np.allclose(self.current_etaP.coeffs, self.last_etaP.coeffs) or not np.allclose(self.current_etaH.coeffs, self.last_etaH.coeffs))) or (not self.vector_conductance and (not np.allclose(self.current_etaP, self.last_etaP) or not np.allclose(self.current_etaH, self.last_etaH))):
                self.state.set_conductance(self.current_etaP, self.current_etaH, self.vector_conductance)
                self.last_etaP = self.current_etaP
                self.last_etaH = self.current_etaH


    def update_u(self):
        """ Update neutral wind """

        if hasattr(self, 'u_timeseries'):
            # Use xarray sel with padding to get the neutral wind values at the current time
            if self.vector_u:
                self.current_u = Vector(basis = self.u_basis, basis_evaluator = self.u_basis_evaluator, coeffs = np.hstack((self.u_timeseries[self.u_basis.short_name + '_u_cf'].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad').values, self.u_timeseries[self.u_basis.short_name + '_u_df'].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad').values)), helmholtz = True)
            else:
                self.current_u = (self.u_timeseries['GRID_u_theta'].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad').values, self.u_timeseries['GRID_u_phi'].sel(time = self.latest_time + FLOAT_ERROR_MARGIN, method = 'pad').values)

            # Check if current u is different from the one used in the last call to update_u
            if not hasattr(self, 'last_u') or (self.vector_u and not np.allclose(self.current_u.coeffs, self.last_u.coeffs)) or (not self.vector_u and not np.allclose(self.current_u[0], self.last_u[0]) and not np.allclose(self.current_u[1], self.last_u[1])):
                self.state.set_u(self.current_u, self.vector_u)
                self.last_u = self.current_u


    def load_timeseries(self):
        """ Load time series from file """

        # Load state time series if it exists on file
        if (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_state.ncdf'):
            state_timeseries = xr.load_timeseries(self.result_filename_prefix + '_state.ncdf')

            state_basis_index = pd.MultiIndex.from_arrays([state_timeseries[self.state_basis.index_labels[i]].values for i in range(len(self.state_basis.index_labels))], names = self.state_basis.index_labels)
            state_coords = xr.Coordinates.from_pandas_multiindex(state_basis_index, dim = 'i').merge({'time': state_timeseries.time.values})
            self.state_timeseries = state_timeseries.drop_vars(['m', 'n']).assign_coords(state_coords)

        # Load jr time series if it exists on file
        if (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_jr.ncdf'):
            jr_timeseries = xr.load_timeseries(self.result_filename_prefix + '_jr.ncdf')

            if self.vector_jr:
                jr_basis_labels = self.jr_basis.index_labels
            else:
                jr_basis_labels = ['theta', 'phi']

            jr_basis_index = pd.MultiIndex.from_arrays([jr_timeseries[jr_basis_labels[i]].values for i in range(len(jr_basis_labels))], names = jr_basis_labels)
            jr_coords = xr.Coordinates.from_pandas_multiindex(jr_basis_index, dim = 'i').merge({'time': jr_timeseries.time.values})
            self.jr_timeseries = jr_timeseries.drop_vars(jr_basis_labels).assign_coords(jr_coords)

        # Load conductance time series if it exists on file
        if (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_conductance.ncdf'):
            conductance_timeseries = xr.load_timeseries(self.result_filename_prefix + '_conductance.ncdf')

            if self.vector_conductance:
                conductance_basis_labels = self.conductance_basis.index_labels
            else:
                conductance_basis_labels = ['theta', 'phi']

            conductance_basis_index = pd.MultiIndex.from_arrays([conductance_timeseries[conductance_basis_labels[i]].values for i in range(len(conductance_basis_labels))], names = conductance_basis_labels)
            conductance_coords = xr.Coordinates.from_pandas_multiindex(conductance_basis_index, dim = 'i').merge({'time': conductance_timeseries.time.values})
            self.conductance_timeseries = conductance_timeseries.drop_vars(conductance_basis_labels).assign_coords(conductance_coords)

        # Load neutral wind time series if it exists on file
        if (self.result_filename_prefix is not None) and os.path.exists(self.result_filename_prefix + '_u.ncdf'):
            u_timeseries = xr.load_timeseries(self.result_filename_prefix + '_u.ncdf')

            if self.vector_u:
                u_basis_labels = self.u_basis.index_labels
            else:
                u_basis_labels = ['theta', 'phi']

            u_basis_index = pd.MultiIndex.from_arrays([u_timeseries[u_basis_labels[i]].values for i in range(len(u_basis_labels))], names = u_basis_labels)
            u_coords = xr.Coordinates.from_pandas_multiindex(u_basis_index, dim = 'i').merge({'time': u_timeseries.time.values})
            self.u_timeseries = u_timeseries.drop_vars(u_basis_labels).assign_coords(u_coords)


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
