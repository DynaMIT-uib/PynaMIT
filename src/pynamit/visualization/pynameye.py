"""Legacy PynamEye visualization frontend.

``PynamEye`` remains available for compatibility with older scripts.
New interactive and publication workflows should prefer
:mod:`pynamit.visualization.figure_builder` and the Panel app.
"""

import datetime

import apexpy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from dipole import Dipole
from polplot import Polarplot

from pynamit.fields import FieldCoefficients, FieldSpace
from pynamit.geomagnetism import MagneticFieldEvaluation
from pynamit.math.constants import RE
from pynamit.simulation.config import setting_value
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    electric_field_on_grid,
    joule_heating_from_current,
    pedersen_geometry_tensor,
    resistance_tensor_on_grid,
)
from pynamit.sphere import CSBasis, Grid
from pynamit.sphere.spherical_transform import SphericalTransform
from pynamit.visualization.field_maps import (
    evaluate_conductance_coefficients,
    evaluate_JS_from_maps,
    evaluate_wind_coefficients,
)
from pynamit.visualization.grid_evaluation import transform_for_basis
from pynamit.visualization.map_coordinates import MapCoordinateContext
from pynamit.visualization.plot_helpers import style_global_axis as style_cartopy_global_axis
from pynamit.visualization.plot_helpers import suppress_empty_contour_warnings
from pynamit.visualization.saved_run import SavedRunView
from pynamit.visualization.state_fields import (
    evaluate_Br_coefficients,
    evaluate_equivalent_current_coefficients,
    evaluate_jr_coefficients,
    evaluate_Phi_coefficients,
    evaluate_W_coefficients,
)


class PynamEye:
    """Compatibility frontend for visualizing simulation results.

    Attributes
    ----------
    datasets : dict
        Dictionary holding simulation datasets loaded from file(s).
    main_field : MainField
        An instance of the MainField class representing the magnetic
        field model in use.
    global_grid : Grid
        Global grid used for evaluations.
    transforms : dict
        Spherical transforms for different regions.
    resistance_transforms : dict
        Spherical transforms for stored resistance across regions.
    ...additional attributes as needed...
    """

    def __init__(
        self, run_directory, t=0, Nlat=60, Nlon=100, NCS_plot=10, mlatlim=50, steady_state=True
    ):
        """Initialize the PynamEye object.

        Parameters
        ----------
        run_directory : str
            Directory for the simulation save files to visualize.
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
        optional_datasets = ["Br", "u", "Q_eff", "E_source"]
        if steady_state:
            optional_datasets.append("steady_state")
        self.run_view = SavedRunView.from_directory(
            run_directory,
            required_datasets=("resistance", "state"),
            optional_datasets=optional_datasets,
            build_geometry=True,
        )
        if steady_state and "steady_state" not in self.run_view.datasets:
            print(f"Could not find steady_state dataset at {run_directory!r}.")
        self.datasets = self.run_view.datasets
        self.pfac_coupling_matrix = (
            None
            if self.run_view.geometry.main_field.kind == "radial"
            or not self.run_view.config.enable_pfac_coupling
            else self.run_view.geometry.pfac_coupling_matrix
        )

        self.mlatlim = mlatlim
        self.settings = self.run_view.datasets["settings"]
        self.config = self.run_view.config
        self.RI = float(self.config.RI)
        self.main_field = self.run_view.main_field

        # Set up cubed sphere grid for vector plotting.
        self.vector_cs_basis = CSBasis(NCS_plot)
        k, i, j = self.vector_cs_basis.get_gridpoints(NCS_plot)
        # Crop to skip duplicate points.
        arr_xi = self.vector_cs_basis.xi(i[:, :-1, :-1] + 0.5, NCS_plot).reshape(-1)
        arr_eta = self.vector_cs_basis.eta(j[:, :-1, :-1] + 0.5, NCS_plot).reshape(-1)
        _, arr_theta, arr_phi = self.vector_cs_basis.cube2spherical(
            arr_xi, arr_eta, k[:, :-1, :-1].reshape(-1), deg=True
        )
        self.global_vector_grid = Grid(theta=arr_theta, lon=arr_phi)

        # Define t0 and set up the configured dipole object.
        self.t0 = datetime.datetime.fromisoformat(self.config.t0)
        self.main_field_epoch = float(self._config_value("main_field_epoch", self.t0.year))
        self.dp = Dipole(self.main_field_epoch)

        self.schema = self.run_view.schema
        self.cs_basis = self.schema.cs_basis
        self.sh_basis = self.schema.sh_basis
        self.mean_free_sh_basis = self.schema.mean_free_sh_basis
        self.basis = self.schema.horizontal_basis
        self.solid_harmonics = self.schema.solid_harmonics
        self.input_field_spaces = self.schema.input_field_spaces
        self.output_field_spaces = self.schema.output_field_spaces

        self.resistance_field_space = self.input_field_spaces["resistance"]
        self.scalar_field_space = self.output_field_spaces["state"]["m_imp"]
        self.tangential_field_space = FieldSpace.from_representation(
            self.basis, field_type="tangential", mean_free=self.scalar_field_space.mean_free
        )
        self.geometry = self.run_view.geometry

        # Set up global grid and spherical transforms.
        self.transforms = {}
        self.resistance_transforms = {}
        self.solid_harmonic_transforms = {}
        lat, lon = np.linspace(-89.9, 89.9, Nlat), np.linspace(-180, 180, Nlon)
        self.lat, self.lon = np.meshgrid(lat, lon)
        self.global_grid = Grid(lat=self.lat, lon=self.lon)
        self._add_transforms("global", self.global_grid)
        self._add_transforms("global_vector", self.global_vector_grid)

        # Set up polar grids and spherical transforms.
        self.mlat, self.mlon = np.meshgrid(
            np.linspace(mlatlim, 89.9, Nlat // 2), np.linspace(-180, 180, Nlon)
        )
        if self.config.main_field_kind.lower() == "igrf":
            # Define a grid, then mask depending on mlatmin.
            self.apx = apexpy.Apex(self.main_field_epoch, refh=(self.RI - RE) * 1e-3)
            self.lat_n, self.lon_n, _ = self.apx.apex2geo(
                self.mlat, self.mlon, (self.RI - RE) * 1e-3
            )
            self.lat_s, self.lon_s, _ = self.apx.apex2geo(
                -self.mlat, self.mlon, (self.RI - RE) * 1e-3
            )
            self.polar_grid_n = Grid(lat=self.lat_n, lon=self.lon_n)
            self.polar_grid_s = Grid(lat=self.lat_s, lon=self.lon_s)
            self._add_transforms("north", self.polar_grid_n)
            self._add_transforms("south", self.polar_grid_s)
        else:
            # Assume simulations are done in magnetic coordinates.
            self.polar_grid = Grid(lat=self.mlat, lon=self.mlon)
            self._add_transforms("north", self.polar_grid)
            self.transforms["south"] = self.transforms["north"]
            self.resistance_transforms["south"] = self.resistance_transforms["north"]
            self.solid_harmonic_transforms["south"] = self.solid_harmonic_transforms["north"]

        self._e_from_b_cache_ready = False
        self._pedersen_geometry_cache = {}

        # Keep operator handles for electromagnetic quantities.
        self.m_ind_to_Br_operator = self.geometry.m_ind_to_Br_operator
        self.m_imp_to_jr_operator = self.geometry.m_imp_to_jr_operator
        # Cache maps needed by Joule heating and E-from-B derivation.
        self.sheet_current_maps = {}

        self._define_defaults()
        self.set_time(t, steady_state=steady_state)

    def _add_transforms(self, region, grid):
        """Add region transforms."""
        self.transforms[region] = SphericalTransform(self.basis, grid)
        self.resistance_transforms[region] = SphericalTransform(
            self.resistance_field_space.representation, grid
        )
        self.solid_harmonic_transforms[region] = self.geometry.solid_harmonic_transform_for(
            self.transforms[region]
        )

    def _sheet_current_maps_for(self, region):
        """Return lazy sheet-current maps for one region."""
        if region not in self.sheet_current_maps:
            transform = self.transforms[region]
            solid_transform = self.solid_harmonic_transforms[region]
            self.sheet_current_maps[region] = {
                "m_ind_to_JS": self.geometry.m_ind_to_gridded_JS(
                    transform, solid_transform=solid_transform
                ),
                "m_imp_to_JS": self.geometry.m_imp_to_gridded_JS(
                    transform, solid_transform=solid_transform
                ),
                "Br_to_JS": self.geometry.Br_to_gridded_JS(
                    transform, solid_transform=solid_transform
                ),
            }
        return self.sheet_current_maps[region]

    @staticmethod
    def _data_var_name(field_space, var):
        """Return the persisted dataset variable name."""
        return f"{field_space.kind}_{var}"

    def _select_values(self, dataset, field_space, var):
        """Select one coefficient row from a saved dataset."""
        name = self._data_var_name(field_space, var)
        if name not in dataset:
            raise KeyError(
                f"Dataset is missing {name!r}; available variables are "
                f"{sorted(dataset.data_vars)}."
            )
        return dataset[name].sel(time=self.t, method="nearest").values.reshape(-1)

    def derive_E_from_B(self):
        """Derive E coefficients from B coefficients.

        If B coefficients are not manipulated, this should have no
        meaningful effect. Calling this function can be expensive with
        high resolutions due to matrix inversion.
        """
        if self.m_u is None:
            raise RuntimeError("No saved 'u' dataset is available for E derivation.")
        if not self._e_from_b_cache_ready:
            # Reuse the exact numerical geometry from the saved run.
            self.state_grid = self.geometry.model_grid
            self.transforms["num"] = self.geometry.horizontal_transform
            self.resistance_transforms["num"] = SphericalTransform(
                self.resistance_field_space.representation, self.state_grid
            )
            self.solid_harmonic_transforms["num"] = self.geometry.solid_harmonic_transform
            self._num_pedersen_geometry = self.geometry.pedersen_geometry_tensor
            self._num_hall_geometry = self.geometry.hall_geometry_tensor
            self._num_wind_to_E = self.geometry.wind_motional_E_tensor

            self._e_from_b_cache_ready = True

        # Calculate electric field values on state_grid.
        current_maps = self._sheet_current_maps_for("num")
        JS = evaluate_JS_from_maps(
            self.m_imp,
            self.m_ind,
            m_imp_to_JS=current_maps["m_imp_to_JS"],
            m_ind_to_JS=current_maps["m_ind_to_JS"],
            Br=self.m_Br,
            Br_to_JS=current_maps["Br_to_JS"],
        )

        closure_values = evaluate_conductance_coefficients(
            self.resistance_transforms["num"], self.m_etaP, self.m_etaH
        )
        etaP_on_grid = closure_values["etaP"]
        etaH_on_grid = closure_values["etaH"]

        self.u_coeffs = self.u.array
        wind_transform = transform_for_basis(
            self.u.field_space.representation, self.transforms["num"]
        )
        wind = evaluate_wind_coefficients(wind_transform, self.u)
        self.u_theta_on_grid = wind["u_theta"]
        self.u_phi_on_grid = wind["u_phi"]

        resistance = resistance_tensor_on_grid(
            etaP_on_grid, etaH_on_grid, self._num_pedersen_geometry, self._num_hall_geometry
        )
        Eth, Eph = electric_field_on_grid(
            JS,
            resistance,
            wind=np.array([self.u_theta_on_grid, self.u_phi_on_grid]),
            wind_to_E=self._num_wind_to_E,
        )

        E_coeffs = self.transforms["num"].analyze_helmholtz(np.array([Eth, Eph]))
        self.Phi_coeffs = self.basis.get_helmholtz_curl_free_potential_operator().matvec(E_coeffs)
        self.W_coeffs = self.basis.get_helmholtz_divergence_free_potential_operator().matvec(
            E_coeffs
        )
        self.m_Phi = self.Phi_coeffs * self.RI
        self.m_W = self.W_coeffs * self.RI

    def _define_defaults(self):
        """Define default settings for various plots."""
        self.wind_defaults = {"color": "black", "scale": 1e3}
        self.conductance_defaults = {
            "cmap": plt.cm.viridis,
            "levels": np.linspace(0, 20, 22),
            "extend": "max",
        }
        self.joule_defaults = {
            "cmap": plt.cm.inferno,
            "levels": np.linspace(0, 10, 21) * 1e-3,
            "extend": "max",
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

    @staticmethod
    def _fill_plot_defaults(kwargs, defaults):
        """Fill missing plotting keyword arguments in-place."""
        for key, value in defaults.items():
            kwargs.setdefault(key, value)

    def _ensure_dataset_covers_time(self, key):
        """Forward-fill a saved dataset to include ``self.t``."""
        dataset = self.datasets.get(key)
        if dataset is None or "time" not in dataset.coords:
            return
        stored_times = np.atleast_1d(dataset.time.values)
        if np.any(np.isclose(self.t - stored_times, 0)):
            return
        new_time = sorted(list(stored_times) + [self.t])
        self.datasets[key] = dataset.reindex(time=new_time).ffill(dim="time")

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

        for key in ["state", "steady_state", "Br", "u", "Q_eff", "E_source", "resistance"]:
            self._ensure_dataset_covers_time(key)

        if steady_state and "steady_state" in self.datasets:
            print("using steady state dataset")
            state_ds = self.datasets["steady_state"]
        else:
            state_ds = self.datasets["state"]

        state_field_spaces = self.output_field_spaces["state"]
        self.m_ind = self._select_values(state_ds, state_field_spaces["m_ind"], "m_ind")
        self.m_imp = self._select_values(state_ds, state_field_spaces["m_imp"], "m_imp")
        self.W_coeffs = self._select_values(state_ds, state_field_spaces["W"], "W")
        self.Phi_coeffs = self._select_values(state_ds, state_field_spaces["Phi"], "Phi")
        self.m_W = self.W_coeffs * self.RI
        self.m_Phi = self.Phi_coeffs * self.RI

        self.m_etaP = self._select_values(
            self.datasets["resistance"], self.input_field_spaces["resistance"], "etaP"
        )
        self.m_etaH = self._select_values(
            self.datasets["resistance"], self.input_field_spaces["resistance"], "etaH"
        )
        if "Br" in self.datasets:
            self.m_Br = self._select_values(
                self.datasets["Br"], self.input_field_spaces["Br"], "Br"
            )
        else:
            self.m_Br = None
        if "u" in self.datasets:
            self.u = FieldCoefficients(
                self.input_field_spaces["u"],
                self._select_values(self.datasets["u"], self.input_field_spaces["u"], "u"),
            )
            self.m_u = self.u.array
            self.m_u_cf, self.m_u_df = np.split(self.m_u.reshape(-1), 2)
        else:
            self.u = None
            self.m_u = None
            self.m_u_df = None
            self.m_u_cf = None

        if np.any(np.isnan(self.m_ind)):
            print(f"induced magnetic field coefficients at t = {t:.2f} s are nans")

        return self

    def get_magnetic_coordinate_context(self):
        """Return the magnetic local-time context."""
        if str(self._config_value("main_field_kind", "dipole")).lower() == "kaiju_dipole":
            return MapCoordinateContext.from_noon_longitude(
                0.0,
                longitude_kind="magnetic",
                local_time_kind="magnetic",
                label="MLT",
                reference_time=self.time,
            )
        return MapCoordinateContext.magnetic(self.time, self.dp)

    def _config_value(self, name, default=None):
        """Return a config value, falling back to raw settings."""
        config = getattr(self, "config", None)
        if config is not None:
            return getattr(config, name, default)
        return setting_value(self.settings, name, default)

    def get_global_coordinate_context(self):
        """Return the coordinate context for global map plots."""
        if str(self._config_value("main_field_kind", "dipole")).lower() == "igrf":
            return MapCoordinateContext.magnetic(self.time, self.dp, apex=self.apx)
        return self.get_magnetic_coordinate_context()

    def get_global_projection(self):
        """Get the global projection for plotting.

        Returns
        -------
        ccrs.PlateCarree
            The global projection for plotting.
        """
        return self.get_global_coordinate_context().projection()

    def style_global_axis(
        self, ax, draw_labels=True, draw_coastlines=True, local_time_labels=False
    ):
        """Style a global map axis and add magnetic reference curves.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to plot on.
        draw_labels : bool, optional
            Whether to draw labels.
        draw_coastlines : bool, optional
            Whether to draw coastlines.
        local_time_labels : bool, optional
            Whether to label longitudes using the global context.
        """
        style_cartopy_global_axis(
            ax,
            coordinate_context=(
                self.get_global_coordinate_context() if local_time_labels else None
            ),
            draw_labels=draw_labels,
            draw_coastlines=draw_coastlines,
            coastline_color="grey",
        )

        ll = np.linspace(-180, 180, 200)
        dip_lat = 90 - self.main_field.magnetic_colatitude_at_longitude(ll)

        interhemispheric_coupling_latitude = self._config_value(
            "interhemispheric_coupling_latitude", 50
        )
        lbn = 90 - self.main_field.magnetic_colatitude_at_longitude(
            ll, magnetic_colatitude=90 - interhemispheric_coupling_latitude
        )
        lbs = 90 - self.main_field.magnetic_colatitude_at_longitude(
            ll, magnetic_colatitude=90 + interhemispheric_coupling_latitude
        )

        ax.plot(
            ll, dip_lat, color="blue", linestyle="--", linewidth=1, transform=ccrs.PlateCarree()
        )
        ax.plot(ll, lbn, color="blue", linestyle="--", linewidth=0.5, transform=ccrs.PlateCarree())
        ax.plot(ll, lbs, color="blue", linestyle="--", linewidth=0.5, transform=ccrs.PlateCarree())

    @staticmethod
    def _validate_region(region):
        """Validate and normalize a named plot region."""
        if region not in {"global", "north", "south"}:
            raise ValueError("region must be either global, north, or south")
        return region

    def _plot_scalar_values(self, values, ax, region, method, **kwargs):
        """Plot scalar values on a visualization grid."""
        region = self._validate_region(region)
        values = np.asarray(values)
        if region in {"south", "north"}:
            if not isinstance(ax, Polarplot):
                raise TypeError("Polar regions require a polplot.Polarplot axis.")
            mlt = self.get_magnetic_coordinate_context().longitude_to_local_time(self.mlon)
            x_coord, y_coord = ax._latlt2xy(self.mlat, mlt)
            plotter = getattr(ax.ax, method)
            plot_args = (x_coord, y_coord, values.reshape(self.mlat.shape))
        else:
            axis_projection = getattr(ax, "projection", None)
            if axis_projection is None or not axis_projection.equals(self.get_global_projection()):
                raise ValueError("Global plots require the PynamEye global projection.")
            plotter = getattr(ax, method)
            plot_args = (self.lon, self.lat, values.reshape(self.lon.shape))
            kwargs = {"transform": ccrs.PlateCarree(), **kwargs}

        with suppress_empty_contour_warnings():
            return plotter(*plot_args, **kwargs)

    def _plot_contour(self, values, ax, region="global", **kwargs):
        """Plot scalar contours.

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
        return self._plot_scalar_values(values, ax, region, "contour", **kwargs)

    def _plot_filled_contour(self, values, ax, region="global", **kwargs):
        """Plot filled scalar contours.

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
        return self._plot_scalar_values(values, ax, region, "contourf", **kwargs)

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
        region = self._validate_region(region)
        if region in {"south", "north"}:
            print("vector plot on polar grid not yet implemented")
            return None
        axis_projection = getattr(ax, "projection", None)
        if axis_projection is None or not axis_projection.equals(self.get_global_projection()):
            raise ValueError("Global plots require the PynamEye global projection.")
        with suppress_empty_contour_warnings():
            lon, lat = (self.global_vector_grid.lon, self.global_vector_grid.lat)
            return ax.quiver(lon, lat, east, north, transform=ccrs.PlateCarree(), **kwargs)

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
        self._fill_plot_defaults(kwargs, self.joule_defaults)

        current_maps = self._sheet_current_maps_for(region)
        JS = evaluate_JS_from_maps(
            self.m_imp,
            self.m_ind,
            m_imp_to_JS=current_maps["m_imp_to_JS"],
            m_ind_to_JS=current_maps["m_ind_to_JS"],
            Br=self.m_Br,
            Br_to_JS=current_maps["Br_to_JS"],
        )
        closure_values = evaluate_conductance_coefficients(
            self.resistance_transforms[region], self.m_etaP, self.m_etaH
        )
        if region not in self._pedersen_geometry_cache:
            field = MagneticFieldEvaluation(self.main_field, self.transforms[region].grid, self.RI)
            self._pedersen_geometry_cache[region] = pedersen_geometry_tensor(
                field.unit_btheta, field.unit_bphi, field.unit_br
            )
        Q = joule_heating_from_current(
            JS, closure_values["etaP"], self._pedersen_geometry_cache[region]
        )
        self._Q = Q
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
        self._fill_plot_defaults(kwargs, self.conductance_defaults)

        conductance = evaluate_conductance_coefficients(
            self.resistance_transforms[region], self.m_etaP, self.m_etaH
        )

        if hp == "h":
            Sigma = conductance["SigmaH"]
        elif hp == "p":
            Sigma = conductance["SigmaP"]
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
        self._fill_plot_defaults(kwargs, self.wind_defaults)

        if self.m_u is None:
            raise RuntimeError("No saved 'u' dataset is available for wind plotting.")
        wind_transform = transform_for_basis(
            self.u.field_space.representation, self.transforms["global_vector"]
        )
        wind = evaluate_wind_coefficients(wind_transform, self.u, include_magnitude=False)

        return self._quiver(wind["u_east"], wind["u_north"], ax, region, **kwargs)

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
        self._fill_plot_defaults(kwargs, self.Br_defaults)

        Br = evaluate_Br_coefficients(self.geometry, self.m_ind, self.transforms[region])

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
        self._fill_plot_defaults(kwargs, self.eqJ_defaults)

        Jeq = evaluate_equivalent_current_coefficients(
            self.geometry, self.m_ind, self.transforms[region]
        )

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
        self._fill_plot_defaults(kwargs, self.jr_defaults)

        jr = evaluate_jr_coefficients(self.geometry, self.m_imp, self.transforms[region])

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
        self._fill_plot_defaults(kwargs, self.Phi_defaults)

        if from_B:
            self.derive_E_from_B()
        Phi = evaluate_Phi_coefficients(self.geometry, self.Phi_coeffs, self.transforms[region])

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
        self._fill_plot_defaults(kwargs, self.W_defaults)

        W = evaluate_W_coefficients(self.geometry, self.W_coeffs, self.transforms[region])

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
            self.style_global_axis(ax)

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
