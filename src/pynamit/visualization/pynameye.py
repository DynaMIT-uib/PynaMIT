"""PynamEye module.

This module contains the PynamEye class for visualizing simulation
results.
"""

import logging
import warnings
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import apexpy
from dipole import Dipole
from polplot import Polarplot
import datetime
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.primitives.grid import Grid
from pynamit.primitives.field import Field
from pynamit.math.constants import RE
from pynamit.postprocess.grid_evaluation import decode_conductance_entry_to_grids
from pynamit.simulation.data import SimulationData

logger = logging.getLogger(__name__)


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
        self.simulation_data = SimulationData.from_prefix(filename_prefix)
        self.datasets = self.simulation_data.datasets
        self.settings = self.simulation_data.settings
        if steady_state and not self.simulation_data.has_dataset("steady_state"):
            warnings.warn(
                f"Could not find {filename_prefix + '_steady_state.ncdf'}.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.T_to_Ve = self.simulation_data.pfac_matrix

        self.mlatlim = mlatlim
        settings = self.settings
        self.RI = settings.RI

        # Define mainfield.
        self.mainfield = self.simulation_data.mainfield

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

        self.state_spec = self.simulation_data.solution_spec
        input_spec = self.simulation_data.get_storage_spec("u")
        conductance_spec = self.simulation_data.get_storage_spec("conductance")
        self.input_basis = input_spec.basis
        self.input_mean_free = bool(input_spec.mean_free)
        self.conductance_basis = conductance_spec.basis

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

        # Prepare explicit postprocessing operators on each plot grid.
        self.operator_bundles = {}
        for region in ["global", "north", "south"]:
            bundle = self.simulation_data.get_poloidal_results_operators(
                basis=self.state_spec,
                grid=self.grids[region],
            )
            self.operator_bundles[region] = bundle

        self._define_defaults()
        self.set_time(t)

    def _get_settings_scalar(self, name, default):
        """Return a scalar settings value from the loaded settings dataset."""
        if hasattr(self.settings, name):
            return float(getattr(self.settings, name))
        return float(default)

    def _get_conductance_entry_at_time(self):
        """Return the conductance representation stored at the current time."""
        entry = self.simulation_data.get_input_entry("conductance", self.t, interpolation=False)
        if entry is None:
            raise KeyError(
                f"No conductance entry is available at t={float(self.t):.2f}s."
            )
        return entry

    def _get_eta_on_grid(self, grid):
        """Decode the stored conductance representation to resistivity on ``grid``."""
        _, _, etaP, etaH = decode_conductance_entry_to_grids(
            self._get_conductance_entry_at_time(),
            self.conductance_basis,
            grid,
            target_shape=np.asarray(grid.lat).shape,
            sigma_floor=self._get_settings_scalar("conductance_interpolation_floor", 1e-3),
        )
        return etaP, etaH

    def derive_E_from_B(self):
        """Derive E coefficients from B coefficients.

        This path is not maintained in the current visualization stack.
        Use stored ``SH_Phi``/``SH_W`` outputs or runtime package operators
        instead of reconstructing electric coefficients inside ``PynamEye``.
        """
        raise NotImplementedError(
            "PynamEye.derive_E_from_B() is not maintained. Use stored SH_Phi/SH_W "
            "outputs or the package operator paths instead."
        )

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

        state_entry = self.simulation_data.get_output_entry(
            "steady_state" if steady_state else "state",
            self.t,
            interpolation=False,
        )
        if state_entry is None:
            raise KeyError(
                f"No {'steady-state ' if steady_state else ''}state entry is available "
                f"at t={float(self.t):.2f}s."
            )
        u_entry = self.simulation_data.get_input_entry("u", self.t, interpolation=False)
        if u_entry is None:
            raise KeyError(f"No wind entry is available at t={float(self.t):.2f}s.")

        self.m_ind = np.asarray(state_entry["m_ind"])
        self.m_imp = np.asarray(state_entry["m_imp"])
        self.m_W = np.asarray(state_entry["W"]) * self.RI
        self.m_Phi = np.asarray(state_entry["Phi"]) * self.RI
        self.m_u = np.asarray(u_entry["u"]).reshape(2, -1)
        self.m_u_df, self.m_u_cf = np.split(self.m_u.flatten(), 2)

        if np.any(np.isnan(self.m_ind)):
            warnings.warn(
                f"Induced magnetic field coefficients at t={float(t):.2f}s contain NaNs.",
                RuntimeWarning,
                stacklevel=2,
            )

        return self

    def get_global_projection(self):
        """Get the global projection for plotting.

        Returns
        -------
        ccrs.PlateCarree
            The global projection for plotting.
        """
        noon_longitude = self.dp.mlt2mlon(12, self.time)

        if self.settings.mainfield_kind == "igrf":
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
            ll, theta=90 - self.settings.latitude_boundary
        )
        lbs = 90 - self.mainfield.dip_equator(
            ll, theta=90 + self.settings.latitude_boundary
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
            warnings.warn(
                "Vector plotting on polar grids is not implemented; returning None.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
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
            self.state_spec,
            coeffs=np.array([self.m_Phi, self.m_W]),
            field_type="tangential",
        )
        E = np.asarray(e_coeffs.evaluate_on_grid(self.grids[region])) / float(self.RI)

        # Calculate current.
        bundle = self.operator_bundles[region]
        JS_imp = bundle.evaluate_js_from_m_imp(self.m_imp)
        JS_ind = bundle.evaluate_js_from_m_ind(self.m_ind)
        JS = JS_imp + JS_ind

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
        etaP_on_grid, etaH_on_grid = self._get_eta_on_grid(grid)

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

        utheta, uphi = self.input_basis.evaluate(
            self.m_u,
            self.grids[region],
            vector_type="tangential",
            mean_free=self.input_mean_free,
        )

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

        Br = self.operator_bundles[region].evaluate_br(self.m_ind)

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

        Jeq = self.operator_bundles[region].evaluate_jeq(self.m_ind)

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

        jr = self.operator_bundles[region].evaluate_jr(self.m_imp)

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

        if getattr(self.state_spec, "kind", "") == "SH":
            Phi = self.state_spec.evaluate(
                self.m_Phi,
                self.grids[region],
                mean_free=getattr(self.state_spec, "mean_free", None),
            )
        else:
            Phi = self.state_spec.evaluate(self.m_Phi, self.grids[region])

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

        if getattr(self.state_spec, "kind", "") == "SH":
            W = self.state_spec.evaluate(
                self.m_W,
                self.grids[region],
                mean_free=getattr(self.state_spec, "mean_free", None),
            )
        else:
            W = self.state_spec.evaluate(self.m_W, self.grids[region])

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
