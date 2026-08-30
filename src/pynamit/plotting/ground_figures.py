"""Ground magnetic curve-map and time-series figure renderers."""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kompe import SphericalGrid, SphericalTransform
from kompe.constants import EARTH_RADIUS_M
from matplotlib.lines import Line2D

from pynamit.coordinates import local_time_hours_to_longitude
from pynamit.magnetometers import (
    download_and_load_iaga2002_station_data,
    load_iaga2002_magnetometer_data,
    normalize_station_metadata,
    shift_station_datetime_index,
)
from pynamit.plotting.map_coordinates import MapCoordinateContext
from pynamit.plotting.map_curves import (
    build_even_global_sites,
    curve_site_group_zorders,
    draw_curve_scale_inset,
    draw_timeseries_curve_map,
    geographic_local_time_mask,
    local_time_window_extent,
    reference_aligned_curve_centers,
    split_wrapped_curve,
    wrap_longitudes,
)
from pynamit.plotting.plot_data import _coerce_figure_settings, get_plot_data
from pynamit.results.time_series import (
    compute_centered_difference_matrix_at_times,
    compute_centered_difference_series_at_times,
    get_time_index_median_cadence_seconds,
    resample_matrix_to_times,
    resample_series_to_times,
    vector_magnitude_preserve_shape,
)

_GROUND_FIELD_CACHE = {}
_STATION_FILE_CACHE = {}


class GroundFigureRenderer:
    """Render ground magnetic figures."""

    def __init__(self, settings, plot_data=None):
        self.settings = _coerce_figure_settings(settings)
        self.plot_data = get_plot_data(self.settings) if plot_data is None else plot_data
        self._station_table_cache = None
        self._station_data_directory_cache = None

    @property
    def _time_index(self):
        """Saved simulation time index."""
        return self.plot_data.time_index

    def render_curve_map(self):
        """Render a ground magnetic time-curve map."""
        target_times = self._ground_plot_times()
        source_times = self._time_index + pd.to_timedelta(
            float(self.settings.simulation_time_offset_seconds), unit="s"
        )
        dbdt_cadence_seconds = get_time_index_median_cadence_seconds(source_times)
        normalized_time = np.linspace(0.0, 1.0, len(target_times))
        lon, lat, station_labels, measured_values = self._curve_sites_and_measurements(
            target_times, dbdt_cadence_seconds=dbdt_cadence_seconds
        )
        if lon.size == 0:
            raise ValueError(
                "No ground curve sites remain after station-data, latitude, "
                "and local-time filtering."
            )

        br_dynamic, bh_dynamic, br_equilibrium, bh_equilibrium = self._ground_field_matrices(
            lat, lon
        )
        layers = self._curve_layers(
            br_dynamic,
            bh_dynamic,
            br_equilibrium,
            bh_equilibrium,
            source_times,
            target_times,
            measured_values,
            dbdt_cadence_seconds=dbdt_cadence_seconds,
        )
        if not layers:
            raise ValueError("Enable at least one model series for the ground curve map.")

        curve_width_deg = 10.0 * float(self.settings.curve_time_width_scale)
        curve_height_deg = 4.0
        low_latitude_scale = float(self.settings.low_latitude_scale)
        low_latitude_cutoff = float(self.settings.min_abs_dip_latitude)
        low_latitude_values = self._site_magnetic_latitude(lat, lon, target_times[0])
        site_curve_scale = np.where(
            np.abs(low_latitude_values) < low_latitude_cutoff, low_latitude_scale, 1.0
        )
        value_scale, display_scale = (
            (0.5 * float(self.settings.curve_scale_value), float(self.settings.curve_scale_value))
            if self.settings.curve_scale_mode == "manual"
            else self._ground_value_scale(layers, fallback=self.settings.curve_scale_value)
        )
        signal_label = self._ground_signal_label(self.settings.ground_quantity)
        unit = "nT/s" if self.settings.ground_quantity == "dbdt" else "nT"
        reference_line = self._ground_reference_line(target_times, display_scale)
        curve_center_lon, curve_center_lat = reference_aligned_curve_centers(
            lon,
            lat,
            normalized_time,
            layers,
            curve_width_deg=curve_width_deg,
            curve_height_deg=curve_height_deg,
            value_scale=value_scale,
            site_curve_scale=site_curve_scale,
            reference_line=reference_line,
        )

        fig = plt.figure(figsize=(13, 7), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, hspace=0.02, wspace=0.02)
        central_longitude = float(
            np.asarray(local_time_hours_to_longitude(12.0, target_times[0])).reshape(-1)[0]
        )
        display_projection = ccrs.PlateCarree(central_longitude=central_longitude)
        data_projection = ccrs.PlateCarree()
        axis = fig.add_subplot(111, projection=display_projection)
        selection_extent = local_time_window_extent(
            lat_window=(self.settings.geo_lat_min, self.settings.geo_lat_max),
            local_time_window=(self.settings.local_time_min, self.settings.local_time_max),
            reference_time=target_times[0],
            central_longitude=central_longitude,
        )
        map_extent = selection_extent if self.settings.zoom_window else None
        if map_extent is None:
            axis.set_global()
        else:
            axis.set_extent(map_extent, crs=data_projection)
        self._style_ground_map_axis(axis, data_projection, target_times[0])
        show_low_latitude_guides = (
            self.settings.show_dip_equator_curve or self.settings.show_low_latitude_curve
        )
        low_latitude_legend_handles = []
        if show_low_latitude_guides:
            low_latitude_legend_handles = self._draw_low_latitude_curves(
                axis, data_projection, central_longitude
            )
        conductance_legend_handles = self._draw_conductance_overlays(
            axis, data_projection, target_times[0]
        )
        if selection_extent is not None and not self.settings.zoom_window:
            self._draw_zoom_window_lines(
                axis, data_projection, selection_extent, central_longitude
            )

        if station_labels and self.settings.show_station_labels:
            self._draw_curve_labels(
                axis,
                station_labels,
                curve_center_lon,
                curve_center_lat,
                layers,
                curve_width_deg=curve_width_deg,
                curve_height_deg=curve_height_deg,
                site_curve_scale=site_curve_scale,
                value_scale=value_scale,
                scale_display_value=display_scale,
                central_longitude=central_longitude,
                data_projection=data_projection,
            )

        draw_timeseries_curve_map(
            axis,
            site_lon=lon,
            site_lat=lat,
            normalized_time=normalized_time,
            layers=layers,
            curve_width_deg=curve_width_deg,
            curve_height_deg=curve_height_deg,
            value_scale=value_scale,
            central_longitude=central_longitude,
            site_curve_scale=site_curve_scale,
            reference_line=reference_line,
            curve_center_lon=curve_center_lon,
            curve_center_lat=curve_center_lat,
            extra_legend_handles=conductance_legend_handles + low_latitude_legend_handles,
            legend_kwargs={"loc": "lower right", "framealpha": 0.92, "fontsize": 9},
            reference_color="#0072B2",
            reference_linewidth=1.5,
            reference_linestyle=(0, (1, 1)),
        )
        draw_curve_scale_inset(
            axis,
            curve_width_deg=curve_width_deg,
            curve_height_deg=curve_height_deg,
            value_scale=value_scale,
            scale_display_value=display_scale,
            scale_annotation=self._scale_label(display_scale, unit),
            duration_annotation=self._duration_label(target_times),
            map_extent=map_extent,
            low_lat_scale_annotation=(
                rf"$|\lambda_\mathrm{{m}}| < {low_latitude_cutoff:g}^\circ$ x "
                f"{low_latitude_scale:g}"
                if (low_latitude_cutoff > 0.0 and abs(low_latitude_scale - 1.0) > 1e-12)
                else ""
            ),
        )
        title = (
            f"Ground {signal_label} Curve Map: {self.settings.ground_component}; "
            f"{target_times[0].strftime('%H:%M:%S')} to "
            f"{target_times[-1].strftime('%H:%M:%S')}; "
            f"scale {display_scale:g} {unit}"
        )
        fig.suptitle(title, fontsize=15)
        return fig

    def render_timeseries(self):
        """Render selected-station ground magnetic time series."""
        stations, _ = self._station_table()
        station_code = str(self.settings.ground_station).upper()
        rows = stations[stations["IAGA"] == station_code]
        if rows.empty:
            raise ValueError(f"Unknown station {station_code!r}.")
        station = rows.iloc[0]
        br_dynamic, bh_dynamic, br_equilibrium, bh_equilibrium = self._ground_field_matrices(
            [station["GEOLAT"]], [station["GEOLON"]]
        )
        source_times = self._time_index + pd.to_timedelta(
            float(self.settings.simulation_time_offset_seconds), unit="s"
        )
        dbdt_cadence_seconds = get_time_index_median_cadence_seconds(source_times)
        target_times = self._ground_plot_times()
        measured = self._station_measured_dataframe(station_code, target_times)

        components = ["North", "East", "Down"]
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
        x_start = target_times[0]
        x_end = target_times[-1]
        for axis, component in zip(axes, components, strict=True):
            if measured is not None and self.settings.include_station_data:
                values = measured.loc[x_start:x_end, component]
                if self.settings.ground_quantity == "dbdt":
                    values = pd.Series(
                        compute_centered_difference_series_at_times(
                            measured.index,
                            measured[component].to_numpy(dtype=float),
                            values.index,
                            half_window_points=self.settings.dbdt_window_points,
                            cadence_seconds=dbdt_cadence_seconds,
                        ),
                        index=values.index,
                    )
                axis.plot(values.index, values, color="black", label="Measured")
            for br_values, bh_values, label, color, linestyle, enabled in [
                (br_dynamic, bh_dynamic, "Dynamic", "#D55E00", "-", self.settings.show_dynamic),
                (
                    br_equilibrium,
                    bh_equilibrium,
                    "Equilibrium",
                    "#009E73",
                    "-",
                    self.settings.show_equilibrium,
                ),
            ]:
                if not enabled:
                    continue
                if br_values is None or bh_values is None:
                    raise ValueError(
                        "This simulation has no equilibrium output. Disable Equilibrium "
                        "ground curves, or rerun with save_equilibria=True."
                    )
                values = self._ground_matrix_at_times(
                    component,
                    br_values,
                    bh_values,
                    source_times,
                    target_times,
                    quantity=self.settings.ground_quantity,
                    dbdt_cadence_seconds=dbdt_cadence_seconds,
                )[0]
                axis.plot(target_times, values, color=color, linestyle=linestyle, label=label)
            self._draw_reference_line_on_axis(axis, target_times)
            axis.set_xlim(x_start, x_end)
            axis.set_ylabel("nT/s" if self.settings.ground_quantity == "dbdt" else "nT")
            axis.set_title(component)
            axis.grid(True, linestyle="--", alpha=0.5)
            axis.legend(loc="best")
        axes[-1].set_xlabel(f"Time on {target_times[0].strftime('%Y-%m-%d')}")
        fig.suptitle(
            f"Ground {self._ground_signal_label(self.settings.ground_quantity)} at {station_code}",
            fontsize=14,
        )
        return fig

    def _target_times(self):
        """Return the selected target time interval."""
        start, end = [int(value) for value in self.settings.time_range]
        start = max(0, min(start, self.plot_data.n_time - 1))
        end = max(start, min(end, self.plot_data.n_time - 1))
        if end == start:
            end = min(
                self.plot_data.n_time - 1, start + min(60, max(self.plot_data.n_time - 1, 1))
            )
        return self._time_index[start : end + 1]

    def _ground_plot_times(self):
        """Return ground-plot times with 1 s station resolution."""
        selected = self._target_times()
        if not self.settings.include_station_data or len(selected) < 2:
            return selected
        start = pd.Timestamp(selected[0]).ceil("1s")
        end = pd.Timestamp(selected[-1]).floor("1s")
        if end < start:
            return selected
        return pd.date_range(start, end, freq="1s")

    def _ground_field_matrices(self, site_lat, site_lon):
        """Return ground field matrices for sites."""
        lat_arr = np.asarray(site_lat, dtype=float).reshape(-1)
        lon_arr = np.asarray(site_lon, dtype=float).reshape(-1)
        if lat_arr.size != lon_arr.size:
            raise ValueError("site_lat and site_lon must have the same length.")

        key = (
            id(self.plot_data),
            str(self.plot_data.results.artifact_store.directory),
            tuple(np.round(lat_arr, 8).tolist()),
            tuple(np.round(lon_arr, 8).tolist()),
        )
        cached = _GROUND_FIELD_CACHE.get(key)
        if cached is not None:
            return cached

        geometry = self.plot_data.geometry
        grid = SphericalGrid(lat=lat_arr, lon=lon_arr)
        ri = float(self.plot_data.results.config.RI)
        solid_harmonics = geometry.solid_harmonics
        solid_basis = solid_harmonics.basis
        transform = SphericalTransform(solid_basis, grid)
        ve_to_ground = solid_harmonics.regular_reference_shift_factors(ri, EARTH_RADIUS_M)
        induced_Br_to_br_ground = ve_to_ground * transform.scalar_synthesis_matrix
        induced_Br_to_bh_ground = ve_to_ground / solid_basis.n * transform.surface_gradient_matrix

        induced_Br = self.plot_data.dataset_values("dynamic", "induced_Br").T
        equilibrium_dataset = self.plot_data.results.datasets.get("equilibrium")
        if equilibrium_dataset is None:
            br_equilibrium = None
            bh_equilibrium = None
        else:
            equilibrium_induced_Br = self.plot_data.dataset_values("equilibrium", "induced_Br").T
            br_equilibrium = induced_Br_to_br_ground.dot(equilibrium_induced_Br)
            bh_equilibrium = induced_Br_to_bh_ground.dot(equilibrium_induced_Br)
        cached = (
            induced_Br_to_br_ground.dot(induced_Br),
            induced_Br_to_bh_ground.dot(induced_Br),
            br_equilibrium,
            bh_equilibrium,
        )
        _GROUND_FIELD_CACHE[key] = cached
        if len(_GROUND_FIELD_CACHE) > 64:
            _GROUND_FIELD_CACHE.pop(next(iter(_GROUND_FIELD_CACHE)))
        return cached

    def _curve_sites_and_measurements(self, target_times, *, dbdt_cadence_seconds=None):
        """Return curve-map sites and measured values."""
        station_labels = []
        measured_values = None
        if self.settings.include_station_data:
            station_sites = self._ground_curve_station_sites(target_times)
            station_lon = station_sites["GEOLON"].to_numpy(dtype=float)
            station_lat = station_sites["GEOLAT"].to_numpy(dtype=float)
            candidate_labels = station_sites["IAGA"].astype(str).to_list()
            lon_values = []
            lat_values = []
            measured_rows = []
            for station_code, site_lon, site_lat in zip(
                candidate_labels, station_lon, station_lat, strict=True
            ):
                measured = self._load_local_station_dataframe(station_code, target_times)
                if measured is None:
                    continue
                station_values = self._station_values_at_times(
                    measured, target_times, dbdt_cadence_seconds=dbdt_cadence_seconds
                )
                if station_values.size == len(target_times) and np.isfinite(station_values[0]):
                    station_labels.append(station_code)
                    lon_values.append(site_lon)
                    lat_values.append(site_lat)
                    measured_rows.append(station_values)
            lon = np.asarray(lon_values, dtype=float)
            lat = np.asarray(lat_values, dtype=float)
            measured_values = np.vstack(measured_rows) if measured_rows else None
            return lon, lat, station_labels, measured_values

        lon, lat = build_even_global_sites(
            min_lat=-75.0,
            max_lat=75.0,
            lat_count=self.settings.ground_model_lat_count,
            equatorial_count=self.settings.ground_model_lt_count,
            min_sites_per_row=1,
            reference_time=target_times[0],
            visually_even=self.settings.uniform_ground_longitude_count,
        )
        site_mask = geographic_local_time_mask(
            lat,
            lon,
            lat_window=(self.settings.geo_lat_min, self.settings.geo_lat_max),
            local_time_window=(self.settings.local_time_min, self.settings.local_time_max),
            reference_time=target_times[0],
        )
        return lon[site_mask], lat[site_mask], station_labels, measured_values

    def _curve_layers(
        self,
        br_dynamic,
        bh_dynamic,
        br_equilibrium,
        bh_equilibrium,
        source_times,
        target_times,
        measured_values,
        *,
        dbdt_cadence_seconds=None,
    ):
        """Return curve-map layer dictionaries."""
        layers = []
        if measured_values is not None and np.any(np.isfinite(measured_values)):
            layers.append(
                {
                    "series_key": "measured",
                    "label": "Measured",
                    "values": measured_values,
                    "color": "black",
                    "linewidth": 1.0,
                    "alpha": 0.82,
                    "zorder": 8,
                }
            )
        if self.settings.show_dynamic:
            layers.append(
                {
                    "series_key": "dynamic",
                    "label": "Dynamic",
                    "values": self._ground_matrix_at_times(
                        self.settings.ground_component,
                        br_dynamic,
                        bh_dynamic,
                        source_times,
                        target_times,
                        quantity=self.settings.ground_quantity,
                        dbdt_cadence_seconds=dbdt_cadence_seconds,
                    ),
                    "color": "#D55E00",
                    "linewidth": 1.0,
                    "linestyle": "-",
                    "zorder": 7,
                }
            )
        if self.settings.show_equilibrium:
            if br_equilibrium is None or bh_equilibrium is None:
                raise ValueError(
                    "This simulation has no equilibrium output. Disable Equilibrium "
                    "ground curves, or rerun with save_equilibria=True."
                )
            layers.append(
                {
                    "series_key": "equilibrium",
                    "label": "Equilibrium",
                    "values": self._ground_matrix_at_times(
                        self.settings.ground_component,
                        br_equilibrium,
                        bh_equilibrium,
                        source_times,
                        target_times,
                        quantity=self.settings.ground_quantity,
                        dbdt_cadence_seconds=dbdt_cadence_seconds,
                    ),
                    "color": "#009E73",
                    "linewidth": 1.0,
                    "linestyle": "-",
                    "zorder": 6,
                }
            )
        return layers

    def _station_table(self):
        """Return normalized station metadata and source path."""
        if self._station_table_cache is not None:
            return self._station_table_cache

        simulation_directory = Path(self.settings.simulation_directory).expanduser()
        repo_root = Path(__file__).resolve().parents[3]
        candidates = []
        if self.settings.station_data_directory:
            candidates.append(
                Path(self.settings.station_data_directory).expanduser() / "stations_full_list.csv"
            )
        candidates.extend(
            [
                simulation_directory / "mag_data" / "stations_full_list.csv",
                simulation_directory / "data" / "mag_data" / "stations_full_list.csv",
                Path("mag_data/stations_full_list.csv"),
                Path("notebooks/mag_data/stations_full_list.csv"),
                repo_root / "notebooks" / "mag_data" / "stations_full_list.csv",
            ]
        )
        for candidate in candidates:
            try:
                table = normalize_station_metadata(pd.read_csv(candidate))
                self._station_table_cache = (table, str(candidate))
                return self._station_table_cache
            except FileNotFoundError:
                continue
        raise ValueError(
            "Could not find stations_full_list.csv. Set station_data_directory in "
            "pynamit_plot_defaults.json or place station data in mag_data/."
        )

    def _station_data_directory(self):
        """Return directory containing station data files."""
        if self._station_data_directory_cache is not None:
            return self._station_data_directory_cache
        _, stations_path = self._station_table()
        self._station_data_directory_cache = Path(stations_path).expanduser().parent
        return self._station_data_directory_cache

    def _load_local_station_dataframe(self, station_code, target_times):
        """Load local IAGA2002 station data for a curve map."""
        data_dir = self._station_data_directory()
        source_start = pd.Timestamp(target_times[0]) - pd.to_timedelta(
            float(self.settings.data_time_offset_seconds), unit="s"
        )
        filename = (
            data_dir / f"{str(station_code).lower()}{source_start.strftime('%Y%m%d')}vsec.sec"
        )
        if not filename.exists():
            return None
        measured = self._load_cached_station_file(filename, station_code)
        if measured is None:
            return None
        return self._station_xyz_dataframe(
            measured, station_code, data_time_offset_seconds=self.settings.data_time_offset_seconds
        )

    def _station_measured_dataframe(self, station_code, target_index):
        """Download/load station data for one selected station."""
        try:
            _, stations_path = self._station_table()
        except ValueError:
            return None
        data_dir = str(pd.io.common.stringify_path(stations_path)).rsplit("/", 1)[0]
        measured = download_and_load_iaga2002_station_data(
            station_code, target_index[0], data_dir, logger=None
        )
        if measured is None:
            return None
        return self._station_xyz_dataframe(
            measured, station_code, data_time_offset_seconds=self.settings.data_time_offset_seconds
        )

    @staticmethod
    def _load_cached_station_file(filename, station_code):
        path = Path(filename)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        cache_key = (
            str(path.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
            str(station_code).upper(),
        )
        if cache_key not in _STATION_FILE_CACHE:
            _STATION_FILE_CACHE[cache_key] = load_iaga2002_magnetometer_data(
                path, station_code, logger=None
            )
            if len(_STATION_FILE_CACHE) > 512:
                _STATION_FILE_CACHE.pop(next(iter(_STATION_FILE_CACHE)))
        return _STATION_FILE_CACHE[cache_key]

    @staticmethod
    def _station_xyz_dataframe(measured, station_code, *, data_time_offset_seconds=0.0):
        station_code = str(station_code).upper()
        measured_index = shift_station_datetime_index(
            measured.index, data_time_offset_seconds=data_time_offset_seconds
        )
        return pd.DataFrame(
            {
                "North": measured[f"{station_code}X"].to_numpy(dtype=float),
                "East": measured[f"{station_code}Y"].to_numpy(dtype=float),
                "Down": measured[f"{station_code}Z"].to_numpy(dtype=float),
            },
            index=measured_index,
        )

    def _station_values_at_times(self, measured, target_times, *, dbdt_cadence_seconds=None):
        """Return measured station values sampled at target times."""
        component = self._ground_component_base(self.settings.ground_component)
        components = ["North", "East", "Down"] if component == "Magnitude" else [component]

        sampled = []
        for key in components:
            if self.settings.ground_quantity == "dbdt":
                sampled.append(
                    compute_centered_difference_series_at_times(
                        measured.index,
                        measured[key].to_numpy(dtype=float),
                        target_times,
                        half_window_points=self.settings.dbdt_window_points,
                        cadence_seconds=dbdt_cadence_seconds,
                    )
                )
            else:
                sampled.append(
                    resample_series_to_times(
                        measured.index, measured[key].to_numpy(dtype=float), target_times
                    )
                )
        values = (
            vector_magnitude_preserve_shape(np.vstack(sampled))
            if component == "Magnitude"
            else np.asarray(sampled[0], dtype=float)
        )
        if self._ground_component_uses_abs(self.settings.ground_component):
            return np.abs(values)
        return values

    def _ground_curve_station_sites(self, target_times):
        """Return filtered station metadata."""
        stations, _ = self._station_table()
        mask = geographic_local_time_mask(
            stations["GEOLAT"].to_numpy(dtype=float),
            stations["GEOLON"].to_numpy(dtype=float),
            lat_window=(self.settings.geo_lat_min, self.settings.geo_lat_max),
            local_time_window=(self.settings.local_time_min, self.settings.local_time_max),
            reference_time=target_times[0],
        )
        return stations.loc[mask].reset_index(drop=True)

    def _site_magnetic_latitude(self, lat, lon, event_time):
        """Return magnetic latitude used for low-latitude selection."""
        lat_arr = np.asarray(lat, dtype=float)
        lon_arr = np.asarray(lon, dtype=float)
        main_field = self.plot_data.geometry.main_field
        if main_field.kind in {"igrf", "kaiju_dipole"}:
            mlat = main_field.magnetic_latitude(
                self.plot_data.results.config.RI, 90.0 - lat_arr, lon_arr
            )
            return np.asarray(mlat, dtype=float)
        if main_field.kind == "dipole":
            mlat, _ = main_field.geo_to_model_coordinates(lat_arr, lon_arr, event_time=event_time)
            return np.asarray(mlat, dtype=float)
        raise ValueError(f"Unsupported main_field kind for magnetic latitude: {main_field.kind!r}")

    @staticmethod
    def _ground_component_base(component):
        """Strip absolute-value wrapper from a component name."""
        component = str(component)
        if component.startswith("Abs") and component[3:] in {"North", "East", "Down"}:
            return component[3:]
        return component

    @staticmethod
    def _ground_component_uses_abs(component):
        """Return whether a component is absolute valued."""
        component = str(component)
        return component.startswith("Abs") and component[3:] in {"North", "East", "Down"}

    def _ground_component_matrix(self, component, br_values, bh_values):
        """Return component matrix in nT."""
        base = self._ground_component_base(component)
        if base == "North":
            values = -np.asarray(bh_values[0], dtype=float) * 1e9
        elif base == "East":
            values = np.asarray(bh_values[1], dtype=float) * 1e9
        elif base == "Down":
            values = -np.asarray(br_values, dtype=float) * 1e9
        elif base == "Magnitude":
            values = vector_magnitude_preserve_shape(
                [
                    self._ground_component_matrix("North", br_values, bh_values),
                    self._ground_component_matrix("East", br_values, bh_values),
                    self._ground_component_matrix("Down", br_values, bh_values),
                ]
            )
        else:
            raise ValueError(f"Unsupported ground component: {component!r}")
        return np.abs(values) if self._ground_component_uses_abs(component) else values

    def _ground_matrix_at_times(
        self,
        component,
        br_values,
        bh_values,
        source_times,
        target_times,
        *,
        quantity="b",
        dbdt_cadence_seconds=None,
    ):
        """Return a model ground component sampled at target times."""
        source_index = pd.DatetimeIndex(source_times)
        target_index = pd.DatetimeIndex(target_times)
        if str(quantity) != "dbdt":
            return resample_matrix_to_times(
                source_index,
                self._ground_component_matrix(component, br_values, bh_values),
                target_index,
            )
        base = self._ground_component_base(component)
        cadence = get_time_index_median_cadence_seconds(source_index)
        if base == "Magnitude":
            return vector_magnitude_preserve_shape(
                [
                    self._ground_matrix_at_times(
                        sub_component,
                        br_values,
                        bh_values,
                        source_index,
                        target_index,
                        quantity="dbdt",
                        dbdt_cadence_seconds=dbdt_cadence_seconds,
                    )
                    for sub_component in ("North", "East", "Down")
                ]
            )
        values = compute_centered_difference_matrix_at_times(
            source_index,
            self._ground_component_matrix(base, br_values, bh_values),
            target_index,
            half_window_points=self.settings.dbdt_window_points,
            cadence_seconds=(cadence if dbdt_cadence_seconds is None else dbdt_cadence_seconds),
        )
        return np.abs(values) if self._ground_component_uses_abs(component) else values

    @staticmethod
    def _ground_value_scale(layers, *, fallback=10.0):
        """Choose a readable automatic curve value scale."""
        finite = []
        for layer in layers:
            values = np.asarray(layer["values"], dtype=float)
            valid = np.abs(values[np.isfinite(values)])
            if valid.size:
                finite.append(valid)
        if not finite:
            return 0.5 * float(fallback), float(fallback)
        display = float(np.nanpercentile(np.concatenate(finite), 95.0))
        if not np.isfinite(display) or display <= 0.0:
            display = float(fallback)
        display = float(np.ceil(2.0 * display))
        return 0.5 * display, display

    @staticmethod
    def _duration_label(time_index):
        """Return a compact duration label."""
        if len(time_index) < 2:
            return "0 s"
        total_seconds = int(round((time_index[-1] - time_index[0]).total_seconds()))
        minutes, seconds = divmod(max(total_seconds, 0), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            parts = [f"{hours} h"]
            if minutes:
                parts.append(f"{minutes} m")
            if seconds and not minutes:
                parts.append(f"{seconds} s")
            return " ".join(parts)
        if minutes:
            return f"{minutes} m" if seconds == 0 else f"{minutes} m {seconds} s"
        return f"{seconds} s"

    @staticmethod
    def _scale_label(value, unit):
        """Return curve scale label."""
        value = float(value)
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value))} {unit}"
        if abs(value) < 1.0:
            return f"{value:.2g} {unit}"
        return f"{value:.1f} {unit}"

    @staticmethod
    def _ground_signal_label(quantity):
        """Return a reader-facing ground signal label."""
        return "dB/dt" if str(quantity) == "dbdt" else "B"

    def _ground_reference_line(self, target_times, display_scale):
        """Return the reference line payload for curve maps."""
        if not self.settings.show_reference_line or len(target_times) < 2:
            return None
        reference_time = pd.Timestamp(
            f"{target_times[0].date()} {self.settings.reference_time_of_day_utc}"
        )
        total_seconds = (target_times[-1] - target_times[0]).total_seconds()
        if total_seconds <= 0.0:
            return None
        position = (reference_time - target_times[0]).total_seconds() / total_seconds
        if position < 0.0 or position > 1.0:
            return None
        return {
            "position": position,
            "time": reference_time,
            "label": reference_time.strftime("%H:%M:%S UTC"),
            "color": "#0072B2",
            "linewidth": 1.5,
            "linestyle": (0, (1, 1)),
            "value_span": (2.0 / 3.0) * float(display_scale),
        }

    def _draw_reference_line_on_axis(self, axis, target_times):
        if not self.settings.show_reference_line:
            return
        ref_time = pd.Timestamp(
            f"{target_times[0].date()} {self.settings.reference_time_of_day_utc}"
        )
        if target_times[0] <= ref_time <= target_times[-1]:
            axis.axvline(ref_time, color="#0072B2", linestyle=(0, (1, 1)), zorder=20)

    @staticmethod
    def _style_ground_map_axis(axis, data_projection, reference_time):
        axis.coastlines(color="0.5", linewidth=0.8, zorder=2)
        gridlines = axis.gridlines(
            crs=data_projection,
            draw_labels=True,
            linewidth=0.8,
            color="0.8",
            linestyle="--",
            zorder=1,
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xlabel_style = {"size": 10, "color": "0.25"}
        gridlines.ylabel_style = {"size": 10, "color": "0.25"}
        gridlines.xpadding = 8
        gridlines.ypadding = 8
        MapCoordinateContext.geographic(reference_time).apply_grid_labels(gridlines)

    def _draw_conductance_overlays(self, axis, data_projection, target_time):
        handles = []
        if not (
            self.settings.show_pedersen_conductance_overlay
            or self.settings.show_hall_conductance_overlay
        ):
            return handles

        fields = self.plot_data.input_plot_data_at_time(target_time)

        overlay_specs = [
            (
                self.settings.show_pedersen_conductance_overlay,
                "SigmaP",
                "Pedersen conductance",
                "#6A3D9A",
                "-",
            ),
            (
                self.settings.show_hall_conductance_overlay,
                "SigmaH",
                "Hall conductance",
                "#1F78B4",
                (0, (4, 2)),
            ),
        ]
        base_levels = np.arange(5.0, 45.0, 5.0)
        for enabled, field_key, label, color, linestyle in overlay_specs:
            if not enabled or field_key not in fields:
                continue
            values = np.asarray(fields[field_key], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            levels = base_levels[
                (base_levels > np.nanmin(finite)) & (base_levels < np.nanmax(finite))
            ]
            if levels.size == 0:
                median = float(np.nanmedian(finite))
                if not np.isfinite(median) or median <= 0.0:
                    continue
                levels = np.array([median], dtype=float)
            axis.contour(
                self.plot_data.lon,
                self.plot_data.lat,
                values,
                levels=levels,
                colors=color,
                linewidths=0.65,
                linestyles=linestyle,
                alpha=0.72,
                transform=data_projection,
                zorder=2.35,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=0.9,
                    linestyle=linestyle,
                    alpha=0.82,
                    label=label,
                )
            )
        return handles

    def _draw_low_latitude_curves(self, axis, data_projection, central_longitude):
        main_field = self.plot_data.geometry.main_field
        boundary = float(self.settings.min_abs_dip_latitude)
        dip_equator_style = {
            "color": "#0072B2",
            "linestyle": (0, (5, 3)),
            "linewidth": 1.0,
            "alpha": 0.9,
            "label": "_nolegend_",
        }
        low_latitude_style = {
            "color": "0.18",
            "linestyle": (0, (6, 3)),
            "linewidth": 1.0,
            "alpha": 0.9,
            "label": "_nolegend_",
        }
        traces = []
        if self.settings.show_dip_equator_curve:
            traces.append((0.0, dip_equator_style))
        if self.settings.show_low_latitude_curve and boundary > 0.0:
            for magnetic_latitude in (boundary, -boundary):
                traces.append((magnetic_latitude, low_latitude_style))
        for magnetic_latitude, style in traces:
            lat, lon = main_field.magnetic_latitude_trace_to_geographic(magnetic_latitude)
            finite = np.isfinite(lon) & np.isfinite(lat)
            if not np.any(finite):
                continue
            for lon_segment, lat_segment in split_wrapped_curve(
                lon[finite], lat[finite], central_longitude=central_longitude
            ):
                if lon_segment.size < 2:
                    continue
                axis.plot(lon_segment, lat_segment, transform=data_projection, zorder=2.6, **style)

        handles = []
        if self.settings.show_dip_equator_curve:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#0072B2",
                    linestyle=(0, (5, 3)),
                    linewidth=1.0,
                    alpha=0.9,
                    label=r"$\lambda_\mathrm{m} = 0^\circ$",
                )
            )
        if self.settings.show_low_latitude_curve and boundary > 0.0:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="0.18",
                    linestyle=(0, (6, 3)),
                    linewidth=1.0,
                    alpha=0.9,
                    label=rf"$|\lambda_\mathrm{{m}}| = {boundary:g}^\circ$",
                )
            )
        return handles

    @staticmethod
    def _draw_zoom_window_lines(axis, data_projection, extent, central_longitude):
        extent_arr = np.asarray(extent, dtype=float).reshape(-1)
        if extent_arr.size < 4 or not np.all(np.isfinite(extent_arr[:4])):
            return
        lon_min, lon_max, lat_min, lat_max = [float(value) for value in extent_arr[:4]]
        if lon_max <= lon_min or lat_max <= lat_min:
            return
        horizontal_lon = np.linspace(lon_min, lon_max, 160)
        vertical_lat = np.linspace(lat_min, lat_max, 80)
        sides = [
            (horizontal_lon, np.full(horizontal_lon.shape, lat_min, dtype=float)),
            (np.full(vertical_lat.shape, lon_max, dtype=float), vertical_lat),
            (horizontal_lon[::-1], np.full(horizontal_lon.shape, lat_max, dtype=float)),
            (np.full(vertical_lat.shape, lon_min, dtype=float), vertical_lat[::-1]),
        ]
        style = {
            "color": "red",
            "linestyle": "--",
            "linewidth": 1.2,
            "alpha": 0.9,
            "transform": data_projection,
            "zorder": 2.8,
        }
        for lon_values, lat_values in sides:
            for lon_segment, lat_segment in split_wrapped_curve(
                lon_values, lat_values, central_longitude=central_longitude
            ):
                axis.plot(lon_segment, lat_segment, **style)

    @staticmethod
    def _curve_label_positions(
        lon,
        lat,
        layers,
        *,
        curve_width_deg,
        curve_height_deg,
        site_curve_scale,
        value_scale,
        scale_display_value,
        central_longitude,
        station_labels=None,
    ):
        if not layers:
            return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        reference_values = None
        for layer in layers:
            series_key = str(layer.get("series_key", "")).lower()
            if series_key == "measured":
                reference_values = np.asarray(layer["values"], dtype=float)
                break
        if reference_values is None:
            reference_values = np.asarray(layers[0]["values"], dtype=float)
        lon_arr = np.asarray(lon, dtype=float).reshape(-1)
        lat_arr = np.asarray(lat, dtype=float).reshape(-1)
        scale_arr = np.asarray(site_curve_scale, dtype=float).reshape(-1)
        value_scale = float(value_scale)
        vertical_axis_height = float(curve_height_deg) * (float(scale_display_value) / value_scale)
        label_offset = 0.2 * vertical_axis_height
        time_origin_offset = -0.5 * float(curve_width_deg)
        label_lon = wrap_longitudes(
            lon_arr + time_origin_offset, central_longitude=central_longitude
        )
        first_point_lat = lat_arr.copy()
        for site_index in range(lon_arr.size):
            values = reference_values[site_index]
            finite_indices = np.flatnonzero(np.isfinite(values))
            if finite_indices.size:
                first_value = values[finite_indices[0]]
                first_point_lat[site_index] = lat_arr[site_index] + float(
                    curve_height_deg
                ) * scale_arr[site_index] * (float(first_value) / value_scale)
        label_lat = first_point_lat + label_offset

        plot_lon = wrap_longitudes(label_lon, central_longitude=central_longitude)
        order = np.argsort(plot_lon, kind="mergesort")
        accepted = []
        lower_label_names = {"FCC", "NEW", "BSL", "ASP"}
        if station_labels is None:
            station_labels = [""] * lon_arr.size
        else:
            station_labels = list(station_labels)[: lon_arr.size]
            station_labels.extend([""] * (lon_arr.size - len(station_labels)))
        for site_index in order:
            lon_value = plot_lon[site_index]
            lat_value = label_lat[site_index]
            station_name = str(station_labels[site_index])
            collides = any(
                abs(lon_value - other_lon) < 5.0 and abs(lat_value - other_lat) < 1.25
                for other_lon, other_lat in accepted
            )
            if collides or lat_value > 88.0 or station_name.upper() in lower_label_names:
                label_lat[site_index] = first_point_lat[site_index] - label_offset
            accepted.append((lon_value, label_lat[site_index]))
        return label_lon, label_lat

    def _draw_curve_labels(
        self,
        axis,
        station_labels,
        lon,
        lat,
        layers,
        *,
        curve_width_deg,
        curve_height_deg,
        site_curve_scale,
        value_scale,
        scale_display_value,
        central_longitude,
        data_projection,
    ):
        label_lon, label_lat = self._curve_label_positions(
            lon,
            lat,
            layers,
            curve_width_deg=curve_width_deg,
            curve_height_deg=curve_height_deg,
            site_curve_scale=site_curve_scale,
            value_scale=value_scale,
            scale_display_value=scale_display_value,
            central_longitude=central_longitude,
            station_labels=station_labels,
        )
        label_zorders = curve_site_group_zorders(lon, central_longitude=central_longitude) + 0.04
        for station_code, station_lon, station_lat, zorder in zip(
            station_labels, label_lon, label_lat, label_zorders, strict=True
        ):
            axis.text(
                station_lon,
                station_lat,
                station_code,
                transform=data_projection,
                ha="center",
                va="center",
                fontsize=7.5,
                color="0.15",
                zorder=zorder,
            )


__all__ = ["GroundFigureRenderer"]
