""" pynamit visualization class """
import os
import warnings
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import apexpy
from dipole import Dipole
from polplot import Polarplot 
import datetime
from pynamit.primitives.grid import Grid
from pynamit.primitives.vector import Vector
from pynamit.cubed_sphere.cubed_sphere import CSProjection
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.simulation.mainfield import Mainfield
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.math.constants import RE, mu0

class PynamEye(object):
    def __init__(self, filename_prefix, t = 0, Nlat = 60, Nlon = 100, NCS_plot = 10, mlatlim = 50):
        """
        Parameters
        ----------
        filename_prefix: string
            filename prefix for the simulation save files that will be visualized
        Nlat: int, optional
            number of grid points between -90 and 90 degrees latitude evaluated for 
            visualization. Default is 60.
        Nlon: int, optional
            number of grid points between -180 and 180 degrees longitude evaluated for 
            visualziation. Default is 100.
        """

        keys = ['settings', 'conductance', 'state', 'u']
        filename_suffix = dict(zip(keys, ['_settings', '_conductance', '_state', '_u']))

        # load the file with simulation settings:
        self.datasets = {}
        for key in keys: # load each file
            fn = filename_prefix + filename_suffix[key] + '.ncdf'

            if os.path.isfile(fn):
                self.datasets[key] = xr.load_dataset(fn )
            else:
                raise ValueError('{}.ncdf does not exist'.format(fn))

        self.mlatlim = mlatlim
        settings = self.datasets['settings'] # shorthand
        self.RI = settings.RI

        # Define mainfield:
        self.mainfield = Mainfield(kind = settings.mainfield_kind,
                                   epoch = settings.mainfield_epoch,
                                   hI = (settings.RI - RE) * 1e-3)


        # Set up cubed sphere grid for vector plotting
        self.vector_csp = CSProjection(NCS_plot)
        k, i, j = self.vector_csp.get_gridpoints(NCS_plot)
        arr_xi  = self.vector_csp.xi( i[:, :-1, :-1] + .5, NCS_plot).flatten() # crop to skip duplicate points
        arr_eta = self.vector_csp.eta(j[:, :-1, :-1] + .5, NCS_plot).flatten()
        _, arr_theta, arr_phi = self.vector_csp.cube2spherical(arr_xi, arr_eta, k[:, :-1, :-1].flatten(), deg = True)
        self.global_vector_grid = Grid(theta = arr_theta, lon = arr_phi)


        # Define t0 and set up dipole objct
        self.t0 = datetime.datetime.strptime(settings.t0, "%Y-%m-%d %H:%M:%S")
        self.dp = Dipole(self.t0.year)

        self.basis             = SHBasis(settings.Nmax, settings.Mmax)

        cNmax = int(self.datasets['conductance'].n.max())
        cMmax = int(self.datasets['conductance'].m.max())
        self.conductance_basis = SHBasis(cNmax, cMmax, Nmin = 0)

        # Basis evaluator for wind
        self.u_basis_evaluator = BasisEvaluator(self.basis, self.global_vector_grid)

        # Set up global grid and basis evaluators:
        self.evaluator = {}
        self.conductance_evaluator = {}
        lat, lon = np.linspace(-89.9, 89.9, Nlat), np.linspace(-180, 180, Nlon)
        self.lat, self.lon = np.meshgrid(lat, lon)
        self.global_grid = Grid(lat = self.lat, lon = self.lon)
        self.evaluator['global'] = BasisEvaluator(self.basis, self.global_grid)
        self.conductance_evaluator['global'] = BasisEvaluator(self.conductance_basis, self.global_grid)
        self.evaluator['global_vector'] = BasisEvaluator(self.basis, self.global_vector_grid)
        self.conductance_evaluator['global_vector'] = BasisEvaluator(self.conductance_basis, self.global_vector_grid)

        # Set up polar grids and basis evaluators:
        self.mlat, self.mlon = np.meshgrid(np.linspace(mlatlim, 89.9, Nlat // 2), np.linspace(-180, 180, Nlon))
        if settings.mainfield_kind.lower() == 'igrf': # define a grid, then mask depending on mlatmin
            self.apx = apexpy.Apex(self.t0.year, refh = (settings.RI - RE) * 1e-3)
            self.lat_n, self.lon_n, _ = self.apx.apex2geo( self.mlat, self.mlon, (settings.RI - RE) * 1e-3)
            self.lat_s, self.lon_s, _ = self.apx.apex2geo(-self.mlat, self.mlon, (settings.RI - RE) * 1e-3)
            self.polar_grid_n = Grid(lat = self.lat_n, lon = self.lon_n)
            self.polar_grid_s = Grid(lat = self.lat_s, lon = self.lon_s)
            self.evaluator['north'] = BasisEvaluator(self.basis, self.polar_grid_n)
            self.evaluator['south'] = BasisEvaluator(self.basis, self.polar_grid_s)
            self.conductance_evaluator['north'] = BasisEvaluator(self.conductance_basis, self.polar_grid_n)
            self.conductance_evaluator['south'] = BasisEvaluator(self.conductance_basis, self.polar_grid_s)
        else: # assume simulations are done in magnetic coordinates:
            self.polar_grid = Grid(lat = self.mlat, lon = self.mlon)
            self.evaluator['north'] = BasisEvaluator(self.basis, self.polar_grid)
            self.evaluator['south'] = self.evaluator['north']
            self.conductance_evaluator['north'] = BasisEvaluator(self.conductance_basis, self.polar_grid)
            self.conductance_evaluator['south'] = self.conductance_evaluator['north']

        self.B_parameters_calculated = False

        # conversion factors for electromagnetic quantities:
        n = self.basis.n
        self.m_ind_to_Br  = -n
        self.laplacian    = -n * (n + 1) / self.RI**2
        self.m_imp_to_jr  =  self.laplacian * self.RI / mu0
        self.W_to_dBr_dt  = -self.laplacian * self.RI
        self.m_ind_to_Jeq =  self.RI / mu0 * (2 * n + 1) / (n + 1)

        self._define_defaults()
        self.set_time(t)

    def derive_E_from_B(self):
        """ Re-calculate E coefficients from B coefficients. If B coefficients
        are not manipulated, this should have no meaningful effect. Calling this function
        can be expensive with high resoultions due to matrix inversion
        """

        print('does not work. Rewrite')
        if not self.B_parameters_calculated:

            PFAC = self.datasets['settings'].PFAC_matrix
            nn = int(np.sqrt(PFAC.size))
            self.m_imp_to_B_pol = PFAC.reshape((nn, nn))

            # reproduce numerical grid used in the simulation
            self.csp = CSProjection(self.datasets['settings'].Ncs)
            self.state_grid = Grid(theta = self.csp.arr_theta, phi = self.csp.arr_phi)

            self.evaluator['num'] = BasisEvaluator(self.basis, self.state_grid)
            self.conductance_evaluator['num'] = BasisEvaluator(self.conductance_basis, self.state_grid)

            # evaluate elelctric field on that grid
            self.b_evaluator = FieldEvaluator(self.mainfield, self.state_grid, self.RI)
            self.bP_00 =  self.b_evaluator.bphi**2 + self.b_evaluator.br**2
            self.bP_01 = -self.b_evaluator.btheta * self.b_evaluator.bphi
            self.bP_10 = -self.b_evaluator.btheta * self.b_evaluator.bphi
            self.bP_11 =  self.b_evaluator.btheta**2 + self.b_evaluator.br**2

            self.bH_01 =  self.b_evaluator.br
            self.bH_10 = -self.b_evaluator.br

            self.G_B_pol_to_JS = -self.evaluator['num'].G_rxgrad * self.basis.delta_internal_external / mu0
            self.G_B_tor_to_JS = -self.evaluator['num'].G_grad / mu0
            self.G_m_ind_to_JS = self.G_B_pol_to_JS
            self.G_m_imp_to_JS = self.G_B_tor_to_JS + self.G_B_pol_to_JS.dot(self.m_imp_to_B_pol)


            self.B_parameters_calculated = True

        # calculate electric field values on state_grid:
        Js_ind, Je_ind = np.split(self.G_m_ind_to_JS.dot(self.m_ind), 2, axis = 0)
        Js_imp, Je_imp = np.split(self.G_m_imp_to_JS.dot(self.m_imp), 2, axis = 0)

        Jth, Jph = Js_ind + Js_imp, Je_ind + Je_imp

        etaP_on_grid =  self.conductance_evaluator['num'].basis_to_grid(self.m_etaP)
        #etaH_on_grid =  self.conductance_evaluator['num'].basis_to_grid(self.m_etaH)

        Eth = etaP_on_grid * (self.bP_00 * Jth + self.bP_01 * Jph) + self.etaH_on_grid * (self.bH_01 * Jph)
        Eph = etaP_on_grid * (self.bP_10 * Jth + self.bP_11 * Jph) + self.etaH_on_grid * (self.bH_10 * Jth)

        self.u_coeffs = np.array([self.m_u_cf, self.m_u_df])
        self.u = Vector(self.basis, basis_evaluator = self.evaluator['num'], coeffs = self.u_coeffs, type = 'tangential')
        self.u_theta_on_grid, self.u_phi_on_grid = np.split(self.u.to_grid(basis_evaluator = self.evaluator['num']), 2)

        uxB_theta =  self.u_phi_on_grid   * self.b_evaluator.Br
        uxB_phi   = -self.u_theta_on_grid * self.b_evaluator.Br

        Eth -= uxB_theta
        Eph -= uxB_phi

        self.m_Phi, self.m_W = np.split(self.evaluator['num'].grid_to_basis(np.array([Eth, Eph]), helmholtz = True), 2)
        self.m_Phi = self.m_Phi * self.RI
        self.m_W   = self.m_W * self.RI




    def _define_defaults(self):
        """ Define default settings for various plots """
        self.wind_defaults         = {'color':'black',       'scale':1e3}
        self.conductance_defaults  = {'cmap':plt.cm.viridis, 'levels':np.linspace(0, 20, 22), 'extend':'max'}
        self.Br_defaults           = {'cmap':plt.cm.bwr,     'levels':np.linspace(-100, 100, 22) * 1e-9, 'extend':'both'}
        self.eqJ_defaults          = {'colors':'black',      'levels':np.r_[-210:220:20] * 1e3}
        self.jr_defaults           = {'cmap':plt.cm.bwr,     'levels':np.linspace(-.95, .95, 22) * 1e-6, 'extend':'both'}
        self.Phi_defaults          = {'colors':'black',      'levels':np.r_[-211.5:220:3] * 1e3}
        self.W_defaults            = {'colors':'orange',     'levels':self.Phi_defaults['levels']}


    def set_time(self, t):
        """ set time for PynamEye object in seconds """

        self.t = t
        self.time = self.t0 + datetime.timedelta(seconds = t)

        # 
        for ds in ['state', 'u', 'conductance']:
            if not np.any(np.isclose(self.t - np.atleast_1d(self.datasets[ds].time.values), 0)):
                new_time = sorted(list(self.datasets[ds].time.values) + [self.t])
                self.datasets[ds] = self.datasets[ds].reindex(time=new_time).ffill(dim = 'time')

        self.m_ind  = self.datasets['state'      ].SH_m_ind.sel(time = self.t, method='nearest').values
        self.m_imp  = self.datasets['state'      ].SH_m_imp.sel(time = self.t, method='nearest').values
        self.m_W    = self.datasets['state'      ].SH_W    .sel(time = self.t, method='nearest').values * self.RI
        self.m_Phi  = self.datasets['state'      ].SH_Phi  .sel(time = self.t, method='nearest').values * self.RI
        self.m_etaP = self.datasets['conductance'].SH_etaP .sel(time = self.t, method='nearest').values
        self.m_etaH = self.datasets['conductance'].SH_etaH .sel(time = self.t, method='nearest').values
        self.m_u    = np.vstack(np.split(self.datasets['u'          ].SH_u    .sel(time = self.t, method='nearest').values, 2)).T
        self.m_u_df, self.m_u_cf = np.split(self.m_u.flatten(), 2)

        if np.any(np.isnan(self.m_ind)):
            print('induced magnetic field coefficients at t = {:.2f} s are nans'.format(t))

        return(self)


    def get_global_projection(self):
        noon_longitude = self.dp.mlt2mlon(12, self.time) 

        if self.datasets['settings'].mainfield_kind == 'igrf': # convert to geographic
            _, noon_longitude, _ = self.apx.apex2geo(0, noon_longitude, 0)

        return( ccrs.PlateCarree(central_longitude = noon_longitude) )


    def jazz_global_plot(self, ax, draw_labels = True):
        """ add coastlines, coordinates, ... """
        ax.coastlines(zorder = 2, color = 'grey')
        gridlines = ax.gridlines(draw_labels = draw_labels)
        gridlines.right_labels = False
        gridlines.top_labels = False

        ll = np.linspace(-180, 180, 200)
        dip_lat = 90 - self.mainfield.dip_equator(ll)

        lbn = 90 - self.mainfield.dip_equator(ll, theta = 90 - self.datasets['settings'].latitude_boundary)
        lbs = 90 - self.mainfield.dip_equator(ll, theta = 90 + self.datasets['settings'].latitude_boundary)



        ax.plot(ll, dip_lat, color = 'blue', linestyle = '--', linewidth = 1  , transform = ccrs.PlateCarree())
        ax.plot(ll, lbn    , color = 'blue', linestyle = '--', linewidth = 0.5, transform = ccrs.PlateCarree())
        ax.plot(ll, lbs    , color = 'blue', linestyle = '--', linewidth = 0.5, transform = ccrs.PlateCarree())

    def _plot_contour(self, values, ax, region = 'global', **kwargs):
        """ plot contour """
        if region in ['south', 'north']:
            assert isinstance(ax, Polarplot)
            mlt = self.dp.mlon2mlt(self.mlon, self.time) # magnetic local time
            xx, yy = ax._latlt2xy(self.mlat, mlt)      
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
                return ax.ax.contour(xx, yy, values.reshape(self.mlat.shape), **kwargs)
        elif region == 'global':
            assert ax.projection.equals(self.get_global_projection())
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
                return ax.contour(self.lon, self.lat, values.reshape(self.lon.shape), transform = ccrs.PlateCarree(), **kwargs) 
        else:
            raise ValueError('region must be either global, north, or south')


    def _plot_filled_contour(self, values, ax, region = 'global', **kwargs):
        """ plot filled contour """

        if region in ['south', 'north']:
            assert isinstance(ax, Polarplot)
            mlt = self.dp.mlon2mlt(self.mlon, self.time) # magnetic local time
            xx, yy = ax._latlt2xy(self.mlat, mlt)      
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
                return ax.ax.contourf(xx, yy, values.reshape(self.mlat.shape), **kwargs)
        elif region == 'global':
            assert ax.projection.equals(self.get_global_projection())
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
                return ax.contourf(self.lon, self.lat, values.reshape(self.lon.shape), transform = ccrs.PlateCarree(), **kwargs) 
        else:
            raise ValueError('region must be either global, north, or south')


    def _quiver(self, east, north, ax, region = 'global', **kwargs):
        """ quiver plot """

        if region in ['south', 'north']:
            print('vector plot on polar grid not yet implemented')
        elif region == 'global':
            assert ax.projection == self.get_global_projection()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
                lon, lat = self.global_vector_grid.lon, self.global_vector_grid.lat
                return ax.quiver(lon, lat, east, north, transform = ccrs.PlateCarree(), **kwargs) 
        else:
            raise ValueError('region must be either global, north, or south')


    def plot_conductance(self, ax, hp = 'h', region = 'global', **kwargs):
        """ plot conductance

        Parameters
        ----------
        t: float
            simulation time in seconds
        ax: matplotlib.axes or Polarplot
            where to plot - must be either polplot object or axis with PlateCarree project
        hp: string, optional
            'h' for Hall, 'p' for Pedersen
        region: string, optional
            string, either 'global', 'north', or 'south'
        kwargs: dict, optional
            keyword arguments passed to contourf
        """

        # populate kwargs with default values if not specificed in function call:
        for key in self.conductance_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.conductance_defaults[key]

        etaP_on_grid =  self.conductance_evaluator[region].basis_to_grid(self.m_etaP)
        etaH_on_grid =  self.conductance_evaluator[region].basis_to_grid(self.m_etaP)

        if 'h':
            Sigma = etaH_on_grid / (etaP_on_grid ** 2 + etaH_on_grid ** 2)
        elif 'p':
            Sigma = etaP_on_grid / (etaP_on_grid ** 2 + etaH_on_grid ** 2)
        else:
            raise ValueError('hp must be h or p')

        return self._plot_filled_contour(Sigma, ax, region, **kwargs)


    def plot_wind(self, ax, region = 'global', **kwargs):
        """ plot wind vector field

        Parameters
        ----------
        t: float
            simulation time in seconds
        ax: matplotlib.axes or Polarplot
            where to plot - must be either polplot object or axis with PlateCarree project
        region: string, optional
            string, !!! only 'global' is implemented for vector plots
        kwargs: dict, optional
            keyword arguments passed to contourf
        """

        # populate kwargs with default values if not specificed in function call:
        for key in self.wind_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.wind_defaults[key]

        utheta, uphi = self.u_basis_evaluator.basis_to_grid(self.m_u, helmholtz = True).T

        return self._quiver(uphi, -utheta, ax, region, **kwargs)




    def plot_Br(self, ax, region = 'global', **kwargs):
        """ plot Br

        Parameters
        ----------
        t: float
            simulation time in seconds
        ax: matplotlib.axes or Polarplot
            where to plot - must be either polplot object or axis with PlateCarree project
        region: string, optional
            string, either 'global', 'north', or 'south'
        kwargs: dict, optional
            keyword arguments passed to contourf
        """

        # populate kwargs with default values if not specificed in function call:
        for key in self.Br_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.Br_defaults[key]

        Br = self.evaluator[region].basis_to_grid(self.m_ind * self.m_ind_to_Br)

        return self._plot_filled_contour(Br, ax, region, **kwargs)


    def plot_equivalent_current(self, ax, region = 'global', **kwargs):
        """ plot equivalent current

        Parameters
        ----------
        t: float
            simulation time in seconds
        ax: matplotlib.axes or Polarplot
            where to plot - must be either polplot object or axis with PlateCarree project
        region: string, optional
            string, either 'global', 'north', or 'south'
        kwargs: dict, optional
            keyword arguments passed to contourf
        """

        # populate kwargs with default values if not specificed in function call:
        for key in self.eqJ_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.eqJ_defaults[key]

        Jeq = self.evaluator[region].basis_to_grid(self.m_ind * self.m_ind_to_Jeq)

        return self._plot_contour(Jeq, ax, region, **kwargs)


    def plot_jr(self, ax, region = 'global', **kwargs):
        """ plot jr 

        Parameters
        ----------
        t: float
            simulation time in seconds
        ax: matplotlib.axes or Polarplot
            where to plot - must be either polplot object or axis with PlateCarree project
        region: string, optional
            string, either 'global', 'north', or 'south'
        kwargs: dict, optional
            keyword arguments passed to contourf
        """

        # populate kwargs with default values if not specificed in function call:
        for key in self.jr_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.jr_defaults[key]

        jr = self.evaluator[region].basis_to_grid(self.m_imp * self.m_imp_to_jr)

        return self._plot_filled_contour(jr, ax, region, **kwargs)


    def plot_electric_potential(self, ax, region = 'global', from_B = False, **kwargs):
        """ plot electric_potential

        Parameters
        ----------
        t: float
            simulation time in seconds
        ax: matplotlib.axes or Polarplot
            where to plot - must be either polplot object or axis with PlateCarree project
        region: string, optional
            string, either 'global', 'north', or 'south'
        kwargs: dict, optional
            keyword arguments passed to contourf
        """

        # populate kwargs with default values if not specificed in function call:
        for key in self.Phi_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.Phi_defaults[key]


        Phi = self.evaluator[region].basis_to_grid(self.m_Phi)


        return self._plot_contour(Phi, ax, region, **kwargs)


    def plot_electric_field_stream_function(self, ax, region = 'global', **kwargs):
        """ plot electric field stream function (the inductive part)

        Parameters
        ----------
        t: float
            simulation time in seconds
        ax: matplotlib.axes or Polarplot
            where to plot - must be either polplot object or axis with PlateCarree project
        region: string, optional
            string, either 'global', 'north', or 'south'
        kwargs: dict, optional
            keyword arguments passed to contourf
        """

        # populate kwargs with default values if not specificed in function call:
        for key in self.W_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.W_defaults[key]

        W = self.evaluator[region].basis_to_grid(self.m_W)

        return self._plot_contour(W, ax, region, **kwargs)



    def make_multipanel_output_figure(self, label = None):

        if label is None:
            label = ''

        fig = plt.figure(figsize = (14, 14))
    
        gax1 = fig.add_subplot(333, projection = self.get_global_projection())
        gax2 = fig.add_subplot(336, projection = self.get_global_projection())
        gax3 = fig.add_subplot(339, projection = self.get_global_projection())

        paxn1 = Polarplot(fig.add_subplot(331))
        paxn2 = Polarplot(fig.add_subplot(334))
        paxn3 = Polarplot(fig.add_subplot(337))
        paxs1 = Polarplot(fig.add_subplot(332))
        paxs2 = Polarplot(fig.add_subplot(335))
        paxs3 = Polarplot(fig.add_subplot(338))

        for ax in [gax1, gax2, gax3]: 
            self.jazz_global_plot(ax)

        self.plot_Br(                            gax1, region = 'global')
        self.plot_equivalent_current(            gax1, region = 'global')
        self.plot_jr(                            gax2, region = 'global')
        self.plot_electric_potential(            gax3, region = 'global')
        self.plot_electric_field_stream_function(gax3, region = 'global')

        self.plot_Br(                            paxn1, region = 'north')
        self.plot_equivalent_current(            paxn1, region = 'north')
        self.plot_jr(                            paxn2, region = 'north')
        self.plot_electric_potential(            paxn3, region = 'north')
        self.plot_electric_field_stream_function(paxn3, region = 'north')

        self.plot_Br(                            paxs1, region = 'south')
        self.plot_equivalent_current(            paxs1, region = 'south')
        self.plot_jr(                            paxs2, region = 'south')
        self.plot_electric_potential(            paxs3, region = 'south')
        self.plot_electric_field_stream_function(paxs3, region = 'south')

        gax1.set_title(label)

        plt.tight_layout()

        return(fig)






if __name__ == '__main__':
    fn = "/".join(os.path.abspath(__file__).split("/")[:-1]) + '/../../../scripts/hdtest'
    a = PynamEye(fn).set_time(14.92)

    a.make_multipanel_output_figure()

    plt.show()




