"""PynamEye module.

This module contains the PynamEye class for visualizing simulation
results.
"""

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
from pynamit.primitives.field import Field
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.math.constants import RE, mu0


class PynamEye(object):
    """Class for visualizing simulation results.

    Attributes
    ----------
    datasets : dict
        Dictionary holding simulation datasets loaded from file(s).
    mainfield : Mainfield
        An instance of the Mainfield class representing the magnetic
        field model in use.
    global_grid : Grid
        Global grid used for evaluations.
    grids : dict
        Dictionary of Grid instances for different regions.
    conductance_grids : dict
        Dictionary of Grid instances for conductance
        evaluations across regions.
    ...additional attributes as needed...
    """

    def __init__(
        self, filename_prefix, t=0, Nlat=60, Nlon=100, NCS_plot=10, mlatlim=50, steady_state=True
    ):
        """Initialize the PynamEye object.

        Parameters
        ----------
        filename_prefix : str
            Filename prefix for the simulation save files that will be
            visualized.
        t : int, optional
            Simulation time in seconds.
        Nlat : int, optional
            Number of grid points between -90 and 90 degrees latitude
            for visualization.
        Nlon : int, optional
            Number of grid points between -180 and 180 degrees longitude
            for visualization.
        NCS_plot : int, optional
            Number of grid points for the cubed sphere plot.
        mlatlim : int, optional
            Magnetic latitude limit.
        steady_state : bool, optional
            Whether to use steady state data.
        """
        keys = ["settings", "conductance", "state", "u"]
        filename_suffix = dict(zip(keys, ["_settings", "_conductance", "_state", "_u"]))

        # Load all datasets specified in keys.
        self.datasets = {}
        for key in keys:
            fn = filename_prefix + filename_suffix[key] + ".ncdf"

            if os.path.isfile(fn):
                self.datasets[key] = xr.load_dataset(fn)
            else:
                raise ValueError("{} does not exist".format(fn))

        if steady_state:
            try:
                self.datasets["steady_state"] = xr.load_dataset(
                    filename_prefix + "_steady_state.ncdf"
                )
            except FileNotFoundError:
                print("Could not find {}.".format(filename_prefix + "_steady_state.ncdf"))

        self.T_to_Ve = xr.load_dataarray(filename_prefix + "_PFAC_matrix.ncdf").values

        self.mlatlim = mlatlim
        settings = self.datasets["settings"]
        self.RI = settings.RI

        # Define mainfield.
        self.mainfield = Mainfield(
            kind=settings.mainfield_kind,
            epoch=settings.mainfield_epoch,
            hI=(settings.RI - RE) * 1e-3,
        )

        # Set up cubed sphere grid for vector plotting.
        self.vector_cs_basis = CSBasis(NCS_plot)
        k, i, j = self.vector_cs_basis.get_gridpoints(NCS_plot)
        # Crop to skip duplicate points.
        arr_xi = self.vector_cs_basis.xi(i[:, :-1, :-1] + 0.5, NCS_plot).flatten()
        arr_eta = self.vector_cs_basis.eta(j[:, :-1, :-1] + 0.5, NCS_plot).flatten()
        _, arr_theta, arr_phi = self.vector_cs_basis.cube2spherical(
            arr_xi, arr_eta, k[:, :-1, :-1].flatten(), deg=True
        )
        self.global_vector_grid = Grid(theta=arr_theta, lon=arr_phi)

        # Define t0 and set up dipole object.
        self.t0 = datetime.datetime.strptime(settings.t0, "%Y-%m-%d %H:%M:%S")
        self.dp = Dipole(self.t0.year)

        self.basis = SHBasis(settings.Nmax, settings.Mmax)

        cNmax = int(self.datasets["conductance"].n.max())
        cMmax = int(self.datasets["conductance"].m.max())
        self.conductance_basis = SHBasis(cNmax, cMmax, Nmin=0)

        # Set up grids.
        self.grids = {}
        lat, lon = np.linspace(-89.9, 89.9, Nlat), np.linspace(-180, 180, Nlon)
        self.lat, self.lon = np.meshgrid(lat, lon)
        self.grids["global"] = Grid(lat=self.lat, lon=self.lon)
        self.grids["global_vector"] = self.global_vector_grid

        # Set up polar grids.
        self.mlat, self.mlon = np.meshgrid(
            np.linspace(mlatlim, 89.9, Nlat // 2), np.linspace(-180, 180, Nlon)
        )
        if settings.mainfield_kind.lower() == "igrf":
            # Define a grid, then mask depending on mlatmin.
            self.apx = apexpy.Apex(self.t0.year, refh=(settings.RI - RE) * 1e-3)
            self.lat_n, self.lon_n, _ = self.apx.apex2geo(
                self.mlat, self.mlon, (settings.RI - RE) * 1e-3
            )
            self.lat_s, self.lon_s, _ = self.apx.apex2geo(
                -self.mlat, self.mlon, (settings.RI - RE) * 1e-3
            )
            self.grids["north"] = Grid(lat=self.lat_n, lon=self.lon_n)
            self.grids["south"] = Grid(lat=self.lat_s, lon=self.lon_s)
        else:
            # Assume simulations are done in magnetic coordinates.
            grid_polar = Grid(lat=self.mlat, lon=self.mlon)
            self.grids["north"] = grid_polar
            self.grids["south"] = grid_polar

        self.B_parameters_calculated = False

        # Prepare conversion factors for electromagnetic quantities.
        self.m_ind_to_Br = -(self.RI**2) * self.basis.laplacian(self.RI)
        self.m_imp_to_jr = self.RI / mu0 * self.basis.laplacian(self.RI)
        self.W_to_dBr_dt = 1 / self.RI
        self.m_ind_to_Jeq = -self.RI / mu0 * self.basis.coeffs_to_delta_V

        # Calculate matrices to calculate current.
        self.G_B_pol_to_JS = {}
        self.G_B_tor_to_JS = {}
        self.G_m_ind_to_JS = {}
        self.G_m_imp_to_JS = {}
        for region in ["global", "north", "south"]:
            grid = self.grids[region]
            G_rxgrad = self.basis.get_evaluation_matrix(grid, derivative="rxgrad")
            G_grad = self.basis.get_evaluation_matrix(grid, derivative="grad")
            
            self.G_B_pol_to_JS[region] = (
                -G_rxgrad * self.basis.coeffs_to_delta_V / mu0
            )
            self.G_B_tor_to_JS[region] = -G_grad / mu0
            self.G_m_ind_to_JS[region] = self.G_B_pol_to_JS[region]
            self.G_m_imp_to_JS[region] = self.G_B_tor_to_JS[region] + np.tensordot(
                self.G_B_pol_to_JS[region], self.T_to_Ve, 1
            )

        self._define_defaults()
        self.set_time(t)

    def derive_E_from_B(self):
        """Derive E coefficients from B coefficients.

        If B coefficients are not manipulated, this should have no
        meaningful effect. Calling this function can be expensive with
        high resolutions due to matrix inversion.
        """
        print("does not work. Rewrite")
        if not self.B_parameters_calculated:
            PFAC = self.datasets["settings"].PFAC_matrix
            nn = int(np.sqrt(PFAC.size))
            self.T_to_Ve = PFAC.reshape((nn, nn))

            # Reproduce numerical grid used in the simulation.
            self.cs_basis = CSBasis(self.datasets["settings"].Ncs)
            self.state_grid = Grid(theta=self.cs_basis.arr_theta, phi=self.cs_basis.arr_phi) # Removed BasisEvaluator. Use grids for evaluation.
            self.grids["num"] = self.state_grid
            self.conductance_grids["num"] = self.state_grid

            # Evaluate elelctric field on that grid.
            self.b_field = self.mainfield.discretize(self.state_grid, self.RI)

            self.mag = self.b_field.magnitude
            self.br = self.b_field.vec.r / self.mag
            self.btheta = self.b_field.vec.theta / self.mag
            self.bphi = self.b_field.vec.phi / self.mag

            self.bP_00 = self.bphi**2 + self.br**2
            self.bP_01 = -self.btheta * self.bphi
            self.bP_10 = -self.btheta * self.bphi
            self.bP_11 = self.btheta**2 + self.br**2

            self.bH_01 = self.br
            self.bH_10 = -self.br

            self.num_grid = self.state_grid
            G_rxgrad = self.basis.get_evaluation_matrix(self.num_grid, derivative="rxgrad")
            G_grad = self.basis.get_evaluation_matrix(self.num_grid, derivative="grad")

            self.G_B_pol_to_JS = (
                -G_rxgrad * self.basis.coeffs_to_delta_V / mu0
            )
            self.G_B_tor_to_JS = -G_grad / mu0
            self.G_m_ind_to_JS = self.G_B_pol_to_JS
            self.G_m_imp_to_JS = self.G_B_tor_to_JS + self.G_B_pol_to_JS.dot(self.T_to_Ve)

            self.B_parameters_calculated = True

        # Calculate electric field values on state_grid.
        Js_ind, Je_ind = np.split(self.G_m_ind_to_JS.dot(self.m_ind), 2, axis=0)
        Js_imp, Je_imp = np.split(self.G_m_imp_to_JS.dot(self.m_imp), 2, axis=0)

        Jth, Jph = Js_ind + Js_imp, Je_ind + Je_imp

        etaP_on_grid = self.conductance_basis.evaluate(self.m_etaP, self.num_grid)

        Eth = etaP_on_grid * (self.bP_00 * Jth + self.bP_01 * Jph) + self.etaH_on_grid * (
            self.bH_01 * Jph
        )
        Eph = etaP_on_grid * (self.bP_10 * Jth + self.bP_11 * Jph) + self.etaH_on_grid * (
            self.bH_10 * Jth
        )

        self.u_coeffs = np.array([self.m_u_cf, self.m_u_df])
        self.u = Field.from_coefficients(self.basis, coeffs=self.u_coeffs, field_type="tangential")
        self.u_theta_on_grid, self.u_phi_on_grid = np.split(
            self.u.to_grid_values(self.num_grid), 2
        )

        uxB_theta = self.u_phi_on_grid * self.b_evaluator.Br
        uxB_phi = -self.u_theta_on_grid * self.b_evaluator.Br

        Eth -= uxB_theta
        Eph -= uxB_phi

        self.m_Phi, self.m_W = np.split(
            self.basis.grid_to_basis(np.array([Eth, Eph]), self.num_grid, helmholtz=True), 2
        )
        self.m_Phi = self.m_Phi * self.RI
        self.m_W = self.m_W * self.RI

    def _define_defaults(self):
        """Define default settings for various plots."""
        self.wind_defaults = {"color": "black", "scale": 1e3}
        self.conductance_defaults = {
            "cmap": plt.cm.viridis,
            "levels": np.linspace(0, 20, 22),
            "extend": "max",
        }
        self.joule_defaults = {
            "cmap": plt.cm.bwr,
            "levels": np.linspace(-10, 10, 22) * 1e-3,
            "extend": "both",
        }
        self.Br_defaults = {
            "cmap": plt.cm.bwr,
            "levels": np.linspace(-100, 100, 22) * 1e-9,
            "extend": "both",
        }
        self.eqJ_defaults = {"colors": "black", "levels": np.r_[-610:620:20] * 1e3}
        self.jr_defaults = {
            "cmap": plt.cm.bwr,
            "levels": np.linspace(-0.95, 0.95, 22) * 1e-6,
            "extend": "both",
        }
        self.Phi_defaults = {"colors": "black", "levels": np.r_[-211.5:220:3] * 1e3}
        self.W_defaults = {"colors": "orange", "levels": self.Phi_defaults["levels"]}

    def set_time(self, t, steady_state=False):
        """Set time for PynamEye object in seconds.

        Parameters
        ----------
        t : int
            Simulation time in seconds.
        steady_state : bool, optional
            Whether to use steady state data.
        """
        self.t = t
        self.time = self.t0 + datetime.timedelta(seconds=t)

        #
        for ds in ["state", "u", "conductance"]:
            if not np.any(np.isclose(self.t - np.atleast_1d(self.datasets[ds].time.values), 0)):
                new_time = sorted(list(self.datasets[ds].time.values) + [self.t])
                self.datasets[ds] = self.datasets[ds].reindex(time=new_time).ffill(dim="time")

        if steady_state:  # use the steady-state version of state dataset
            print("using steady state dataset")
            state_ds = self.datasets["steady_state"]
        else:
            state_ds = self.datasets["state"]

        self.m_ind = state_ds.SH_m_ind.sel(time=self.t, method="nearest").values
        self.m_imp = state_ds.SH_m_imp.sel(time=self.t, method="nearest").values
        self.m_W = state_ds.SH_W.sel(time=self.t, method="nearest").values * self.RI
        self.m_Phi = state_ds.SH_Phi.sel(time=self.t, method="nearest").values * self.RI

        self.m_etaP = (
            self.datasets["conductance"].SH_etaP.sel(time=self.t, method="nearest").values
        )
        self.m_etaH = (
            self.datasets["conductance"].SH_etaH.sel(time=self.t, method="nearest").values
        )
        self.m_u = np.vstack(
            np.split(self.datasets["u"].SH_u.sel(time=self.t, method="nearest").values, 2)
        )
        self.m_u_df, self.m_u_cf = np.split(self.m_u.flatten(), 2)

        if np.any(np.isnan(self.m_ind)):
            print(f"induced magnetic field coefficients at t = {(t,):.2f} s are nans")

        return self

    def get_global_projection(self):
        """Get the global projection for plotting.

        Returns
        -------
        ccrs.PlateCarree
            The global projection for plotting.
        """
        noon_longitude = self.dp.mlt2mlon(12, self.time)

        if self.datasets["settings"].mainfield_kind == "igrf":
            # Convert to geographic coordinates.
            _, noon_longitude, _ = self.apx.apex2geo(0, noon_longitude, 0)

        return ccrs.PlateCarree(central_longitude=noon_longitude)

    def jazz_global_plot(self, ax, draw_labels=True, draw_coastlines=True):
        """Add coastlines and coordinates to the global plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to plot on.
        draw_labels : bool, optional
            Whether to draw labels.
        draw_coastlines : bool, optional
            Whether to draw coastlines.
        """
        if draw_coastlines:
            ax.coastlines(zorder=2, color="grey")

        gridlines = ax.gridlines(draw_labels=draw_labels)
        gridlines.right_labels = False
        gridlines.top_labels = False

        ll = np.linspace(-180, 180, 200)
        dip_lat = 90 - self.mainfield.dip_equator(ll)

        lbn = 90 - self.mainfield.dip_equator(
            ll, theta=90 - self.datasets["settings"].latitude_boundary
        )
        lbs = 90 - self.mainfield.dip_equator(
            ll, theta=90 + self.datasets["settings"].latitude_boundary
        )

        ax.plot(
            ll, dip_lat, color="blue", linestyle="--", linewidth=1, transform=ccrs.PlateCarree()
        )
        ax.plot(ll, lbn, color="blue", linestyle="--", linewidth=0.5, transform=ccrs.PlateCarree())
        ax.plot(ll, lbs, color="blue", linestyle="--", linewidth=0.5, transform=ccrs.PlateCarree())

    def _plot_contour(self, values, ax, region="global", **kwargs):
        """Plot contour.

        Parameters
        ----------
        values : array-like
            The values to plot.
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        **kwargs
            Additional keyword arguments passed to contour.
        """
        if region in ["south", "north"]:
            assert isinstance(ax, Polarplot)
            mlt = self.dp.mlon2mlt(self.mlon, self.time)  # Magnetic local time
            xx, yy = ax._latlt2xy(self.mlat, mlt)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="No contour levels were found within the data range."
                )
                return ax.ax.contour(xx, yy, values.reshape(self.mlat.shape), **kwargs)
        elif region == "global":
            assert ax.projection.equals(self.get_global_projection())
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="No contour levels were found within the data range."
                )
                return ax.contour(
                    self.lon,
                    self.lat,
                    values.reshape(self.lon.shape),
                    transform=ccrs.PlateCarree(),
                    **kwargs,
                )
        else:
            raise ValueError("region must be either global, north, or south")

    def _plot_filled_contour(self, values, ax, region="global", **kwargs):
        """Plot filled contour.

        Parameters
        ----------
        values : array-like
            The values to plot.
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        **kwargs
            Additional keyword arguments passed to contourf.
        """
        if region in ["south", "north"]:
            assert isinstance(ax, Polarplot)
            mlt = self.dp.mlon2mlt(self.mlon, self.time)  # Magnetic local time
            xx, yy = ax._latlt2xy(self.mlat, mlt)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="No contour levels were found within the data range."
                )
                return ax.ax.contourf(xx, yy, values.reshape(self.mlat.shape), **kwargs)
        elif region == "global":
            assert ax.projection.equals(self.get_global_projection())
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="No contour levels were found within the data range."
                )
                return ax.contourf(
                    self.lon,
                    self.lat,
                    values.reshape(self.lon.shape),
                    transform=ccrs.PlateCarree(),
                    **kwargs,
                )
        else:
            raise ValueError("region must be either global, north, or south")

    def _quiver(self, east, north, ax, region="global", **kwargs):
        """Quiver plot.

        Parameters
        ----------
        east : array-like
            The eastward component of the vector field.
        north : array-like
            The northward component of the vector field.
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        **kwargs
            Additional keyword arguments passed to quiver.
        """
        if region in ["south", "north"]:
            print("vector plot on polar grid not yet implemented")
        elif region == "global":
            assert ax.projection == self.get_global_projection()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="No contour levels were found within the data range."
                )
                lon, lat = (self.global_vector_grid.lon, self.global_vector_grid.lat)
                return ax.quiver(lon, lat, east, north, transform=ccrs.PlateCarree(), **kwargs)
        else:
            raise ValueError("region must be either global, north, or south")

    def plot_joule(self, ax, region="global", **kwargs):
        """Plot Joule heating.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        **kwargs
            Additional keyword arguments passed to contourf.
        """
        # Populate kwargs with default values if not specificed in
        # function call.
        for key in self.conductance_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.joule_defaults[key]

        # Calculate electric field.
        e_coeffs = Field.from_coefficients(
            self.basis, coeffs=np.array([self.m_Phi, self.m_W]), field_type="tangential"
        )
        field = e_coeffs.to_grid_values(self.grids[region]) / self.RI
        print("todo: is the scaling as expected?")

        # Calculate current.
        JS_imp = self.G_m_imp_to_JS[region].dot(self.m_imp)
        JS_ind = self.G_m_ind_to_JS[region].dot(self.m_ind)
        JS = np.split(JS_imp + JS_ind, 2)

        # Calculate Joule heating.
        Q = JS[0] * E[0] + JS[1] * E[1]
        self._Q = Q
        self._E = E
        self._JS = JS

        # Plot.
        return self._plot_filled_contour(Q, ax, region, **kwargs)

    def plot_conductance(self, ax, hp="h", region="global", **kwargs):
        """Plot conductance.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        hp : str, optional
            'h' for Hall, 'p' for Pedersen.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        **kwargs
            Additional keyword arguments passed to contourf.
        """
        # Populate kwargs with default values if not specificed in
        # function call.
        for key in self.conductance_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.conductance_defaults[key]

        grid = self.grids[region]
        etaP_on_grid = self.conductance_basis.evaluate(self.m_etaP, grid)
        etaH_on_grid = self.conductance_basis.evaluate(self.m_etaH, grid)

        if hp == "h":
            Sigma = etaH_on_grid / (etaP_on_grid**2 + etaH_on_grid**2)
        elif hp == "p":
            Sigma = etaP_on_grid / (etaP_on_grid**2 + etaH_on_grid**2)
        else:
            raise ValueError("hp must be h or p")

        return self._plot_filled_contour(Sigma, ax, region, **kwargs)

    def plot_wind(self, ax, region="global", **kwargs):
        """Plot wind vector field.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        **kwargs
            Additional keyword arguments passed to quiver.
        """
        # Populate kwargs with default values if not specificed in
        # function call.
        for key in self.wind_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.wind_defaults[key]

        utheta, uphi = self.basis.evaluate(self.m_u, self.grids[region], vector_type="tangential")

        return self._quiver(uphi, -utheta, ax, region, **kwargs)

    def plot_Br(self, ax, region="global", **kwargs):
        """Plot Br.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        **kwargs
            Additional keyword arguments passed to contourf.
        """
        # Populate kwargs with default values if not specificed in
        # function call.
        for key in self.Br_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.Br_defaults[key]

        Br = self.basis.evaluate(self.m_ind * self.m_ind_to_Br, self.grids[region])

        return self._plot_filled_contour(Br, ax, region, **kwargs)

    def plot_equivalent_current(self, ax, region="global", **kwargs):
        """Plot equivalent current.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        **kwargs
            Additional keyword arguments passed to contour.
        """
        # Populate kwargs with default values if not specificed in
        # function call.
        for key in self.eqJ_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.eqJ_defaults[key]

        Jeq = self.basis.evaluate(self.m_ind * self.m_ind_to_Jeq, self.grids[region])

        return self._plot_contour(Jeq, ax, region, **kwargs)

    def plot_jr(self, ax, region="global", **kwargs):
        """Plot jr.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        **kwargs
            Additional keyword arguments passed to contourf.
        """
        # Populate kwargs with default values if not specificed in
        # function call.
        for key in self.jr_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.jr_defaults[key]

        jr = self.basis.evaluate(self.m_imp * self.m_imp_to_jr, self.grids[region])

        return self._plot_filled_contour(jr, ax, region, **kwargs)

    def plot_electric_potential(self, ax, region="global", from_B=False, **kwargs):
        """Plot electric potential.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        from_B : bool, optional
            Whether to derive from B coefficients.
        **kwargs
            Additional keyword arguments passed to contour.
        """
        # Populate kwargs with default values if not specificed in
        # function call.
        for key in self.Phi_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.Phi_defaults[key]

        Phi = self.basis.evaluate(self.m_Phi, self.grids[region])

        return self._plot_contour(Phi, ax, region, **kwargs)

    def plot_electric_field_stream_function(self, ax, region="global", **kwargs):
        """Plot electric field stream function (the inductive part).

        Parameters
        ----------
        ax : matplotlib.axes.Axes or Polarplot
            The axis to plot on.
        region : str, optional
            The region to plot ('global', 'north', or 'south').
        **kwargs
            Additional keyword arguments passed to contour.
        """
        # Populate kwargs with default values if not specificed in
        # function call.
        for key in self.W_defaults:
            if key not in kwargs.keys():
                kwargs[key] = self.W_defaults[key]

        W = self.basis.evaluate(self.m_W, self.grids[region])

        return self._plot_contour(W, ax, region, **kwargs)

    def make_multipanel_output_figure(self, label=None):
        """Create a multipanel output figure.

        Parameters
        ----------
        label : str, optional
            Label for the figure.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if label is None:
            label = ""

        fig = plt.figure(figsize=(14, 14))

        gax1 = fig.add_subplot(333, projection=self.get_global_projection())
        gax2 = fig.add_subplot(336, projection=self.get_global_projection())
        gax3 = fig.add_subplot(339, projection=self.get_global_projection())

        paxn1 = Polarplot(fig.add_subplot(331))
        paxn2 = Polarplot(fig.add_subplot(334))
        paxn3 = Polarplot(fig.add_subplot(337))
        paxs1 = Polarplot(fig.add_subplot(332))
        paxs2 = Polarplot(fig.add_subplot(335))
        paxs3 = Polarplot(fig.add_subplot(338))

        for ax in [gax1, gax2, gax3]:
            self.jazz_global_plot(ax)

        self.plot_Br(gax1, region="global")
        self.plot_equivalent_current(gax1, region="global")
        self.plot_jr(gax2, region="global")
        self.plot_electric_potential(gax3, region="global")
        self.plot_electric_field_stream_function(gax3, region="global")

        self.plot_Br(paxn1, region="north")
        self.plot_equivalent_current(paxn1, region="north")
        self.plot_jr(paxn2, region="north")
        self.plot_electric_potential(paxn3, region="north")
        self.plot_electric_field_stream_function(paxn3, region="north")

        self.plot_Br(paxs1, region="south")
        self.plot_equivalent_current(paxs1, region="south")
        self.plot_jr(paxs2, region="south")
        self.plot_electric_potential(paxs3, region="south")
        self.plot_electric_field_stream_function(paxs3, region="south")

        gax1.set_title(label)

        plt.tight_layout()

        return fig


if __name__ == "__main__":
    fn = (
        "/".join(os.path.abspath(__file__).split("/")[:-1]) + "/../../../scripts/simulation/hdtest"
    )
    a = PynamEye(fn).set_time(14.92)

    a.make_multipanel_output_figure()

    plt.show()
