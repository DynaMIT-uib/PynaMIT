"""Figure assembly and plot methods for ``SimulationViewer``."""

import logging
import numpy as np
import matplotlib.pyplot as plt
from polplot import Polarplot
from pynamit.primitives.field import Field
from pynamit.visualization.map_plotting import (
    decorate_global_axes,
    make_global_projection,
    plot_region_contour,
    plot_region_filled_contour,
    plot_region_quiver,
)
from pynamit.visualization.simulation_viewer_state import _SimulationViewerState

logger = logging.getLogger(__name__)


class SimulationViewer(_SimulationViewerState):
    """High-level saved-run viewer and figure builder."""

    def __init__(
        self, run_directory, t=0, Nlat=60, Nlon=100, NCS_plot=10, mlatlim=50, steady_state=True
    ):
        super().__init__(
            run_directory=run_directory,
            t=t,
            Nlat=Nlat,
            Nlon=Nlon,
            NCS_plot=NCS_plot,
            mlatlim=mlatlim,
            steady_state=steady_state,
        )
        self._define_defaults()

    def derive_E_from_B(self):
        """Derive E coefficients from B coefficients.

        This path is not maintained in the current visualization stack.
        Use stored ``SH_Phi``/``SH_W`` outputs or runtime package operators
        instead of reconstructing electric coefficients inside ``SimulationViewer``.
        """
        raise NotImplementedError(
            "SimulationViewer.derive_E_from_B() is not maintained. Use stored SH_Phi/SH_W "
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

    def _get_global_projection(self):
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

        return make_global_projection(noon_longitude)

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
        return plot_region_filled_contour(
            ax,
            Q,
            region=region,
            global_lon=self.lon,
            global_lat=self.lat,
            polar_lat=self.mlat,
            polar_lon=self.mlon,
            dipole=self.dp,
            time=self.time,
            projection=self._get_global_projection(),
            **kwargs,
        )

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

        return plot_region_filled_contour(
            ax,
            Sigma,
            region=region,
            global_lon=self.lon,
            global_lat=self.lat,
            polar_lat=self.mlat,
            polar_lon=self.mlon,
            dipole=self.dp,
            time=self.time,
            projection=self._get_global_projection(),
            **kwargs,
        )

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

        return plot_region_quiver(
            ax,
            uphi,
            -utheta,
            region=region,
            global_lon=self.global_vector_grid.lon,
            global_lat=self.global_vector_grid.lat,
            projection=self._get_global_projection(),
            **kwargs,
        )

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

        return plot_region_filled_contour(
            ax,
            Br,
            region=region,
            global_lon=self.lon,
            global_lat=self.lat,
            polar_lat=self.mlat,
            polar_lon=self.mlon,
            dipole=self.dp,
            time=self.time,
            projection=self._get_global_projection(),
            **kwargs,
        )

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

        return plot_region_contour(
            ax,
            Jeq,
            region=region,
            global_lon=self.lon,
            global_lat=self.lat,
            polar_lat=self.mlat,
            polar_lon=self.mlon,
            dipole=self.dp,
            time=self.time,
            projection=self._get_global_projection(),
            **kwargs,
        )

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

        return plot_region_filled_contour(
            ax,
            jr,
            region=region,
            global_lon=self.lon,
            global_lat=self.lat,
            polar_lat=self.mlat,
            polar_lon=self.mlon,
            dipole=self.dp,
            time=self.time,
            projection=self._get_global_projection(),
            **kwargs,
        )

    def plot_electric_potential(self, ax, region="global", **kwargs):
        """Plot electric potential.

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

        return plot_region_contour(
            ax,
            Phi,
            region=region,
            global_lon=self.lon,
            global_lat=self.lat,
            polar_lat=self.mlat,
            polar_lon=self.mlon,
            dipole=self.dp,
            time=self.time,
            projection=self._get_global_projection(),
            **kwargs,
        )

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

        return plot_region_contour(
            ax,
            W,
            region=region,
            global_lon=self.lon,
            global_lat=self.lat,
            polar_lat=self.mlat,
            polar_lon=self.mlon,
            dipole=self.dp,
            time=self.time,
            projection=self._get_global_projection(),
            **kwargs,
        )

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

        gax1 = fig.add_subplot(333, projection=self._get_global_projection())
        gax2 = fig.add_subplot(336, projection=self._get_global_projection())
        gax3 = fig.add_subplot(339, projection=self._get_global_projection())

        paxn1 = Polarplot(fig.add_subplot(331))
        paxn2 = Polarplot(fig.add_subplot(334))
        paxn3 = Polarplot(fig.add_subplot(337))
        paxs1 = Polarplot(fig.add_subplot(332))
        paxs2 = Polarplot(fig.add_subplot(335))
        paxs3 = Polarplot(fig.add_subplot(338))

        for ax in [gax1, gax2, gax3]:
            decorate_global_axes(
                ax,
                mainfield=self.mainfield,
                latitude_boundary=self.settings.latitude_boundary,
            )

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
