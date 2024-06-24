""" pynamit visualization class """
import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import apexpy
from dipole import Dipole
from polplot import Polarplot 
import datetime
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.simulation.mainfield import Mainfield
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.various.constants import RE, mu0

class PynamEye(object):
    def __init__(self, filename_prefix, t = 0, Nlat = 60, Nlon = 100, mlatlim = 50):
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

        keys = ['settings', 'FAC', 'conductance', 'state', 'u']
        filename_suffix = dict(zip(keys, ['', '_FAC', '_conductance', '_state', '_u']))

        # load the file with simulation settings:
        self.datasets = {}
        for key in keys: # load each file
            fn = filename_prefix + filename_suffix[key] + '.ncdf'

            if os.path.isfile(fn):
                self.datasets[key] = xr.load_dataset(fn)
            else:
                raise ValueError('{}.ncdf does not exist'.format(filename_prefix))

        self.mlatlim = mlatlim
        settings = self.datasets['settings'] # shorthand
        self.RI = settings.RI

        # Define mainfield:
        B0_params = {'epoch':settings.mainfield_epoch, 'hI':(settings.RI - RE) * 1e-3}
        self.mainfield = Mainfield(kind = settings.mainfield_kind, **B0_params)

        # Define t0 and set up dipole objct
        self.t0 = datetime.datetime.strptime(settings.t0, "%Y-%m-%d %H:%M:%S")
        self.dp = Dipole(self.t0.year)

        self.basis             = SHBasis(settings.Nmax, settings.Mmax)
        self.conductance_basis = SHBasis(settings.Nmax, settings.Mmax, Nmin = 0)

        # Set up global grid and basis evaluators:
        self.evaluator = {}
        self.conductance_evaluator = {}
        lat, lon = np.linspace(-89.9, 89.9, Nlat), np.linspace(-180, 180, Nlon)
        self.lat, self.lon = np.meshgrid(lat, lon)
        self.global_grid = Grid(self.lat, self.lon)
        self.evaluator['global'] = BasisEvaluator(self.basis, self.global_grid)
        self.conductance_evaluator['global'] = BasisEvaluator(self.conductance_basis, self.global_grid)

        # Set up polar grids and basis evaluators:
        self.mlat, self.mlon = np.meshgrid(np.linspace(mlatlim, 89.9, Nlat // 2), np.linspace(-180, 180, Nlon))
        if settings.mainfield_kind.lower() == 'igrf': # define a grid, then mask depending on mlatmin
            self.apx = apexpy.Apex(self.t0.year, refh = (settings.RI - RE) * 1e-3)
            self.lat_n, self.lon_n, _ = self.apx.apex2geo( self.mlat, self.mlon, (settings.RI - RE) * 1e-3)
            self.lat_s, self.lon_s, _ = self.apx.apex2geo(-self.mlat, self.mlon, (settings.RI - RE) * 1e-3)
            self.polar_grid_n = Grid(self.lat_n, self.lon_n)
            self.polar_grid_s = Grid(self.lat_s, self.lon_s)
            self.evaluator['north'] = BasisEvaluator(self.basis, self.polar_grid_n)
            self.evaluator['south'] = BasisEvaluator(self.basis, self.polar_grid_s)
            self.conductance_evaluator['north'] = BasisEvaluator(self.conductance_basis, self.polar_grid_n)
            self.conductance_evaluator['south'] = BasisEvaluator(self.conductance_basis, self.polar_grid_s)
        else: # assume simulations are done in magnetic coordinates:
            self.polar_grid = Grid(self.mlat, self.mlon)
            self.evaluator['north'] = BasisEvaluator(self.basis, self.polar_grid)
            self.evaluator['south'] = self.evaluator['north']
            self.conductance_evaluator['north'] = BasisEvaluator(self.conductance_basis, self.polar_grid)
            self.conductance_evaluator['south'] = self.conductance_evaluator['north']

        # conversion factors for electromagnetic quantities:
        n, RI = settings.n.values, settings.RI
        self.m_ind_to_Br  = -n
        self.laplacian    = -n * (n + 1) / RI**2
        self.m_imp_to_Jr  =  self.laplacian * RI / mu0
        self.EW_to_dBr_dt = -self.laplacian * RI
        self.m_ind_to_Jeq =  RI / mu0 * (2 * n + 1) / (n + 1)

        self._define_defaults()
        self.set_time(t)

    def _define_defaults(self):
        """ Define default settings for various plots """
        self.Br_defaults  = {'cmap':plt.cm.bwr, 'levels':np.linspace(-100, 100, 22) * 1e-9, 'extend':'both'}
        self.eqJ_defaults = {'colors':'black', 'levels':np.r_[-210:220:20] * 1e3}
        self.FAC_defaults = {'cmap':plt.cm.bwr, 'levels':np.linspace(-.95, .95, 22)/6 * 1e-6, 'extend':'both'}
        self.Phi_defaults = {'colors':'black', 'levels':np.r_[-212.5:220:5] * 1e3}
        self.EW_defaults  = {'colors':'orange', 'levels':self.Phi_defaults['levels']}


    def set_time(self, t):
        """ set time for PynamEye object in seconds """

        self.t = t
        self.time = self.t0 + datetime.timedelta(seconds = t)

        # find coefficients
        self.m_ind  = self.datasets['state'      ].SH_m_ind_coeffs.interp(time = self.t, assume_sorted = True, method = 'linear').values
        self.m_imp  = self.datasets['state'      ].SH_m_imp_coeffs.interp(time = self.t, assume_sorted = True, method = 'linear').values
        self.m_W    = self.datasets['state'      ].SH_W_coeffs    .interp(time = self.t, assume_sorted = True, method = 'linear').values * self.RI
        self.m_Phi  = self.datasets['state'      ].SH_Phi_coeffs  .interp(time = self.t, assume_sorted = True, method = 'linear').values * self.RI
        self.m_u_df = self.datasets['u'          ].SH_u_df_coeffs .interp(time = self.t, assume_sorted = True, method = 'linear').values
        self.m_u_cf = self.datasets['u'          ].SH_u_cf_coeffs .interp(time = self.t, assume_sorted = True, method = 'linear').values 
        self.m_etaP = self.datasets['conductance'].SH_etaP_coeffs .interp(time = self.t, assume_sorted = True, method = 'linear').values
        self.m_etaH = self.datasets['conductance'].SH_etaH_coeffs .interp(time = self.t, assume_sorted = True, method = 'linear').values

        if np.any(np.isnan(self.m_ind)):
            raise ValueError('induced magnetic field coefficients at t = {:.2f} s are nans'.format(t))

        return(self)


    def get_global_projection(self):
        noon_longitude = self.dp.mlt2mlon(12, self.time) 

        if self.datasets['settings'].mainfield_kind == 'igrf': # convert to geographic
            _, noon_longitude, _ = self.apx.apex2geo(0, noon_longitude, 0)

        return( ccrs.PlateCarree(central_longitude = noon_longitude) )


    def jazz_global_plot(self, ax):
        """ add coastlines, coordinates, ... """
        ax.coastlines(zorder = 2, color = 'grey')
        gridlines = ax.gridlines(draw_labels=True)
        gridlines.right_labels = False
        gridlines.top_labels = False

        ll = np.linspace(-180, 180, 200)
        dip_lat = 90 - self.mainfield.dip_equator(ll)

        lbn = 90 - self.mainfield.dip_equator(ll, theta = 90 - a.datasets['settings'].latitude_boundary)
        lbs = 90 - self.mainfield.dip_equator(ll, theta = 90 + a.datasets['settings'].latitude_boundary)

        ax.plot(ll, dip_lat, color = 'blue', linestyle = '--', linewidth = 1)
        ax.plot(ll, lbn, color = 'blue', linestyle = '--', linewidth = 0.5)
        ax.plot(ll, lbs, color = 'blue', linestyle = '--', linewidth = 0.5)

    def _plot_contour(self, values, ax, region = 'global', **kwargs):
        """ plot contour """
        if region in ['south', 'north']:
            assert isinstance(ax, Polarplot)
            mlt = self.dp.mlon2mlt(self.mlon, self.time) # magnetic local time
            xx, yy = ax._latlt2xy(self.mlat, mlt)      
            return ax.ax.contour(xx, yy, values.reshape(self.mlat.shape), **kwargs)
        elif region == 'global':
            assert ax.projection == self.get_global_projection()
            return ax.contour(self.lon, self.lat, values.reshape(self.lon.shape), transform = ccrs.PlateCarree(), **kwargs) 
        else:
            raise ValueError('region must be either global, north, or south')


    def _plot_filled_contour(self, values, ax, region = 'global', **kwargs):
        """ plot filled contour """

        if region in ['south', 'north']:
            assert isinstance(ax, Polarplot)
            mlt = self.dp.mlon2mlt(self.mlon, self.time) # magnetic local time
            xx, yy = ax._latlt2xy(self.mlat, mlt)      
            return ax.ax.contourf(xx, yy, values.reshape(self.mlat.shape), **kwargs)
        elif region == 'global':
            assert ax.projection == self.get_global_projection()
            return ax.contourf(self.lon, self.lat, values.reshape(self.lon.shape), transform = ccrs.PlateCarree(), **kwargs) 
        else:
            raise ValueError('region must be either global, north, or south')


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
        for key in self.FAC_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.FAC_defaults[key]

        jr = self.evaluator[region].basis_to_grid(self.m_imp * self.m_imp_to_Jr)

        return self._plot_filled_contour(jr, ax, region, **kwargs)


    def plot_electric_potential(self, ax, region = 'global', **kwargs):
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
        for key in self.EW_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.EW_defaults[key]

        W = self.evaluator[region].basis_to_grid(self.m_W)

        return self._plot_contour(W, ax, region, **kwargs)



    def make_multipanel_output_figure(self):
        fig = plt.figure(figsize = (14, 14))
    
        gax1 = fig.add_subplot(333, projection = a.get_global_projection())
        gax2 = fig.add_subplot(336, projection = a.get_global_projection())
        gax3 = fig.add_subplot(339, projection = a.get_global_projection())

        paxn1 = Polarplot(fig.add_subplot(331))
        paxn2 = Polarplot(fig.add_subplot(334))
        paxn3 = Polarplot(fig.add_subplot(337))
        paxs1 = Polarplot(fig.add_subplot(332))
        paxs2 = Polarplot(fig.add_subplot(335))
        paxs3 = Polarplot(fig.add_subplot(338))

        for ax in [gax1, gax2, gax3]: a.jazz_global_plot(ax)

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

        plt.tight_layout()

        return(fig)






if __name__ == '__main__':
    fn = "/".join(os.path.abspath(__file__).split("/")[:-1]) + '/../../../scripts/tst'
    a = PynamEye(fn).set_time(1.92)

    a.make_multipanel_output_figure()

    plt.show()




