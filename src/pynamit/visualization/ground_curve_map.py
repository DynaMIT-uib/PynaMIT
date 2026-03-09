"""Internal notebook-support rendering for embedded ground-curve maps."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cartopy.crs as ccrs
import numpy as np
import pandas as pd

from pynamit.math.constants import RE
from pynamit.postprocess.ground_response import build_ground_magnetic_response_operators
from pynamit.postprocess.ground_station import DEFAULT_BASELINE_WINDOW, TimeWindow, compute_baseline_offset
from pynamit.primitives.grid import Grid
from pynamit.visualization.map_plotting import build_even_global_curve_sites, draw_timeseries_curve_map


@dataclass
class GroundCurveMapPayload:
    anchor_lon: np.ndarray
    anchor_lat: np.ndarray
    layers: list[dict[str, Any]]
    normalized_time: np.ndarray
    value_scale: float
    curve_width_deg: float
    curve_height_deg: float
    show_anchor_points: bool
    title: str
    n_sites: int
    station_labels: Optional[list[str]]
    show_station_labels: bool
    is_difference: bool
    is_normalized_difference: bool
    scale_display_value: float
    scale_annotation: str
    duration_annotation: str
    empty_message: Optional[str] = None


class GroundCurveMapBuilder:
    def __init__(
        self,
        *,
        simulation_start_time: datetime.datetime,
        sim_datetime_index: pd.DatetimeIndex,
        stations: pd.DataFrame,
        state_spec: Any,
        ionosphere_radius: float,
        station_loader: Callable[[str], Optional[pd.DataFrame]],
        curve_num_samples: int = 61,
        ground_radius: float = RE,
        baseline_window: TimeWindow = DEFAULT_BASELINE_WINDOW,
    ) -> None:
        self.simulation_start_time = simulation_start_time
        self.sim_datetime_index = pd.DatetimeIndex(pd.to_datetime(sim_datetime_index))
        self.stations = stations.reset_index(drop=True).copy()
        self.state_spec = state_spec
        self.ionosphere_radius = float(ionosphere_radius)
        self.ground_radius = float(ground_radius)
        self.station_loader = station_loader
        self.curve_num_samples = int(max(curve_num_samples, 2))
        self.baseline_window = baseline_window
        self._station_data_cache: dict[str, Optional[pd.DataFrame]] = {}
        self._even_cache: Optional[dict[str, Any]] = None

    def get_local_station_data_cached(self, station_code: str) -> Optional[pd.DataFrame]:
        station_code = str(station_code).upper()
        if station_code not in self._station_data_cache:
            self._station_data_cache[station_code] = self.station_loader(station_code)
        return self._station_data_cache[station_code]

    @staticmethod
    def get_ground_component_matrix(component: str, br_values: Any, bh_values: Any) -> np.ndarray:
        component = str(component)
        if component == "North":
            return -np.asarray(bh_values[0], dtype=float) * 1e9
        if component == "East":
            return np.asarray(bh_values[1], dtype=float) * 1e9
        if component == "Down":
            return -np.asarray(br_values, dtype=float) * 1e9
        raise ValueError(f"Unsupported ground component: {component}")

    @staticmethod
    def resample_series_to_times(index: Any, values: Any, target_times: Any) -> np.ndarray:
        values_arr = np.asarray(values, dtype=float).reshape(-1)
        time_index = pd.DatetimeIndex(pd.to_datetime(index))
        target_index = pd.DatetimeIndex(pd.to_datetime(target_times))
        finite_mask = np.isfinite(values_arr)
        if not np.any(finite_mask):
            return np.full(target_index.shape, np.nan, dtype=float)

        x_ns = time_index.view("int64")[finite_mask]
        y = values_arr[finite_mask]
        order = np.argsort(x_ns)
        x_ns = x_ns[order]
        y = y[order]
        x_unique, unique_idx = np.unique(x_ns, return_index=True)
        y_unique = y[unique_idx]
        target_ns = target_index.view("int64")
        left = float(x_unique[0])
        right = float(x_unique[-1])
        out = np.interp(target_ns, x_unique, y_unique, left=np.nan, right=np.nan)
        out[(target_ns < left) | (target_ns > right)] = np.nan
        return out.astype(float)

    @classmethod
    def resample_matrix_to_times(cls, index: Any, values: Any, target_times: Any) -> np.ndarray:
        values_arr = np.asarray(values, dtype=float)
        return np.vstack([cls.resample_series_to_times(index, row, target_times) for row in values_arr])

    def get_curve_map_target_times(self, start_idx: int, end_idx: int) -> pd.DatetimeIndex:
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        if end_idx <= start_idx and len(self.sim_datetime_index) > 1:
            end_idx = min(start_idx + 1, len(self.sim_datetime_index) - 1)
        start_time = self.sim_datetime_index[max(0, min(start_idx, len(self.sim_datetime_index) - 1))]
        end_time = self.sim_datetime_index[max(0, min(end_idx, len(self.sim_datetime_index) - 1))]
        if end_time <= start_time:
            end_time = start_time + pd.to_timedelta(1, unit="s")
        return pd.date_range(start_time, end_time, periods=self.curve_num_samples)

    @staticmethod
    def _format_duration_label(target_times: pd.DatetimeIndex) -> str:
        if len(target_times) < 2:
            return "0 s"
        total_seconds = int(round((target_times[-1] - target_times[0]).total_seconds()))
        minutes, seconds = divmod(max(total_seconds, 0), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            parts = [f"{hours} h"]
            if minutes > 0:
                parts.append(f"{minutes} m")
            if seconds > 0 and minutes == 0:
                parts.append(f"{seconds} s")
            return " ".join(parts)
        if minutes > 0:
            return f"{minutes} m" if seconds == 0 else f"{minutes} m {seconds} s"
        return f"{seconds} s"

    @staticmethod
    def _round_up_scale(value: float, *, normalized: bool) -> float:
        value = float(max(value, np.finfo(float).tiny))
        if normalized:
            if value <= 1.0:
                step = 0.1
            elif value <= 2.0:
                step = 0.2
            elif value <= 5.0:
                step = 0.5
            else:
                step = 1.0
        else:
            if value <= 10.0:
                step = 1.0
            elif value <= 25.0:
                step = 5.0
            else:
                step = 10.0
        return step * float(np.ceil(value / step))

    @staticmethod
    def _format_scale_label(value: float, *, scale_unit_label: str, normalized: bool) -> str:
        if normalized:
            if abs(value - round(value)) < 1e-9:
                return f"{int(round(value))} x {scale_unit_label}"
            return f"{value:.1f} x {scale_unit_label}"
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value))} {scale_unit_label}"
        return f"{value:.1f} {scale_unit_label}"

    def get_ground_curve_even_cache(
        self,
        *,
        m_ind: np.ndarray,
        m_ind_steady: np.ndarray,
    ) -> dict[str, Any]:
        if self._even_cache is not None:
            return self._even_cache

        even_lon, even_lat = build_even_global_curve_sites(
            min_lat=-75.0,
            max_lat=75.0,
            lat_step=10.0,
            equatorial_spacing_deg=18.0,
            min_sites_per_row=6,
        )
        even_grid = Grid(lat=even_lat, lon=even_lon)
        even_response = build_ground_magnetic_response_operators(
            state_spec=self.state_spec,
            ground_grid=even_grid,
            ionosphere_radius=self.ionosphere_radius,
            ground_radius=self.ground_radius,
        )

        Br_even_inductive = np.column_stack(
            [even_response.evaluate_radial(m_ind[:, i]) for i in range(m_ind.shape[1])]
        )
        Bh_even_inductive = np.stack(
            [even_response.evaluate_horizontal(m_ind[:, i]) for i in range(m_ind.shape[1])],
            axis=-1,
        )
        Br_even_steady = np.column_stack(
            [even_response.evaluate_radial(m_ind_steady[:, i]) for i in range(m_ind_steady.shape[1])]
        )
        Bh_even_steady = np.stack(
            [even_response.evaluate_horizontal(m_ind_steady[:, i]) for i in range(m_ind_steady.shape[1])],
            axis=-1,
        )

        self._even_cache = {
            "lon": even_lon,
            "lat": even_lat,
            "Br_inductive": Br_even_inductive,
            "Bh_inductive": Bh_even_inductive,
            "Br_steady": Br_even_steady,
            "Bh_steady": Bh_even_steady,
        }
        return self._even_cache

    def build_payload(
        self,
        *,
        component: str,
        include_data: bool,
        plot_differences: bool,
        normalize_differences: bool,
        normalization_metric: str,
        show_station_labels: bool,
        sim_offset_seconds: float,
        start_idx: int,
        end_idx: int,
        Br_inductive: np.ndarray,
        Bh_inductive: np.ndarray,
        Br_steady: np.ndarray,
        Bh_steady: np.ndarray,
        m_ind: Optional[np.ndarray] = None,
        m_ind_steady: Optional[np.ndarray] = None,
    ) -> GroundCurveMapPayload:
        target_times = self.get_curve_map_target_times(start_idx, end_idx)
        target_norm = np.linspace(0.0, 1.0, len(target_times))
        time_label = (
            f"{target_times[0].strftime('%H:%M:%S')}–{target_times[-1].strftime('%H:%M:%S')}"
            f" ({target_times[0].strftime('%Y-%m-%d')})"
        )
        duration_annotation = self._format_duration_label(target_times)

        def site_strength(values: Any) -> float:
            arr = np.asarray(values, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return np.nan
            if normalization_metric == "mean_abs":
                return float(np.mean(np.abs(finite)))
            return float(np.sqrt(np.mean(finite**2)))

        def apply_site_normalization(values: np.ndarray, strengths: np.ndarray) -> np.ndarray:
            values_arr = np.asarray(values, dtype=float)
            strengths_arr = np.asarray(strengths, dtype=float).reshape(-1)
            normalized = values_arr.copy()
            valid = np.isfinite(strengths_arr) & (strengths_arr > 0.0)
            if np.any(valid):
                normalized[valid] = normalized[valid] / strengths_arr[valid, None]
            return normalized

        metric_label = "RMS" if normalization_metric == "rms" else "mean|.|"
        scale_unit_label = "nT"

        if include_data:
            sim_shifted_index = self.sim_datetime_index + pd.to_timedelta(sim_offset_seconds, unit="s")
            station_inductive = self.get_ground_component_matrix(component, Br_inductive, Bh_inductive)
            station_steady = self.get_ground_component_matrix(component, Br_steady, Bh_steady)
            measured_rows: list[np.ndarray] = []
            inductive_rows: list[np.ndarray] = []
            steady_rows: list[np.ndarray] = []
            lon_rows: list[float] = []
            lat_rows: list[float] = []
            station_labels: list[str] = []

            component_suffix = {"North": "X", "East": "Y", "Down": "Z"}[component]
            sim_date = self.simulation_start_time.date()
            baseline_start = datetime.datetime.combine(sim_date, self.baseline_window[0])
            baseline_end = datetime.datetime.combine(sim_date, self.baseline_window[1])
            for station_idx, station in self.stations.iterrows():
                station_code = str(station["IAGA"]).upper()
                mag_df = self.get_local_station_data_cached(station_code)
                if mag_df is None:
                    continue
                column = f"{station_code}{component_suffix}"
                if column not in mag_df.columns:
                    continue

                sim_inductive_series = pd.Series(station_inductive[station_idx], index=sim_shifted_index)
                sim_steady_series = pd.Series(station_steady[station_idx], index=sim_shifted_index)
                baseline_diff = compute_baseline_offset(
                    mag_df[column],
                    sim_inductive_series,
                    sim_steady_series,
                    baseline_start=baseline_start,
                    baseline_end=baseline_end,
                )

                measured = self.resample_series_to_times(
                    mag_df.index,
                    mag_df[column].values - baseline_diff,
                    target_times,
                )
                if np.all(~np.isfinite(measured)):
                    continue

                inductive = self.resample_series_to_times(
                    sim_shifted_index,
                    station_inductive[station_idx],
                    target_times,
                )
                steady = self.resample_series_to_times(
                    sim_shifted_index,
                    station_steady[station_idx],
                    target_times,
                )
                if np.all(~np.isfinite(inductive)) and np.all(~np.isfinite(steady)):
                    continue

                lon_rows.append(float(station["GEOLON"]))
                lat_rows.append(float(station["GEOLAT"]))
                station_labels.append(station_code)
                measured_rows.append(measured)
                inductive_rows.append(inductive)
                steady_rows.append(steady)

            if not lon_rows:
                return GroundCurveMapPayload(
                    anchor_lon=np.array([], dtype=float),
                    anchor_lat=np.array([], dtype=float),
                    layers=[],
                    normalized_time=target_norm,
                    value_scale=1.0,
                    curve_width_deg=8.0,
                    curve_height_deg=2.6,
                    show_anchor_points=True,
                    title=f"Ground Curve Map: {component}",
                    n_sites=0,
                    station_labels=None,
                    show_station_labels=False,
                    is_difference=plot_differences,
                    is_normalized_difference=normalize_differences,
                    scale_annotation="±1.0 nT",
                    duration_annotation=duration_annotation,
                    empty_message="No locally available station data overlap the selected time range.",
                )

            measured_arr = np.vstack(measured_rows)
            inductive_arr = np.vstack(inductive_rows)
            steady_arr = np.vstack(steady_rows)
            if plot_differences:
                diff_inductive = inductive_arr - measured_arr
                diff_steady = steady_arr - measured_arr
                if normalize_differences:
                    strengths = np.array([site_strength(row) for row in measured_arr], dtype=float)
                    diff_inductive = apply_site_normalization(diff_inductive, strengths)
                    diff_steady = apply_site_normalization(diff_steady, strengths)
                    scale_unit_label = metric_label
                layers = [
                    {
                        "label": "Inductive - Measured",
                        "values": diff_inductive,
                        "color": "crimson",
                        "linewidth": 0.9,
                        "alpha": 0.9,
                        "zorder": 3,
                    },
                    {
                        "label": "Magnetostatic - Measured",
                        "values": diff_steady,
                        "color": "royalblue",
                        "linewidth": 0.9,
                        "alpha": 0.9,
                        "zorder": 2,
                    },
                ]
            else:
                layers = [
                    {
                        "label": "Measured",
                        "values": measured_arr,
                        "color": "black",
                        "linestyle": "None",
                        "marker": "o",
                        "markersize": 1.4,
                        "alpha": 0.8,
                        "zorder": 4,
                    },
                    {
                        "label": "Inductive",
                        "values": inductive_arr,
                        "color": "red",
                        "linewidth": 0.9,
                        "alpha": 0.9,
                        "zorder": 3,
                    },
                    {
                        "label": "Magnetostatic",
                        "values": steady_arr,
                        "color": "royalblue",
                        "linewidth": 0.9,
                        "alpha": 0.9,
                        "zorder": 2,
                    },
                ]
            anchor_lon = np.asarray(lon_rows, dtype=float)
            anchor_lat = np.asarray(lat_rows, dtype=float)
            offset_label = f", sim offset={sim_offset_seconds:+.1f} s" if abs(sim_offset_seconds) > 0 else ""
            mode_label = "differences" if plot_differences else "with station data"
            norm_label = f", normalized by local {metric_label}" if normalize_differences else ""
            title = f"Ground Curve Map: {component} {mode_label}{norm_label} ({time_label}{offset_label})"
            curve_width_deg = 8.0
            curve_height_deg = 2.6
            show_anchor_points = True
        else:
            if m_ind is None or m_ind_steady is None:
                raise ValueError("m_ind and m_ind_steady are required for no-data curve maps.")
            even_cache = self.get_ground_curve_even_cache(
                m_ind=m_ind,
                m_ind_steady=m_ind_steady,
            )
            anchor_lon = np.asarray(even_cache["lon"], dtype=float)
            anchor_lat = np.asarray(even_cache["lat"], dtype=float)
            inductive = self.resample_matrix_to_times(
                self.sim_datetime_index,
                self.get_ground_component_matrix(component, even_cache["Br_inductive"], even_cache["Bh_inductive"]),
                target_times,
            )
            steady = self.resample_matrix_to_times(
                self.sim_datetime_index,
                self.get_ground_component_matrix(component, even_cache["Br_steady"], even_cache["Bh_steady"]),
                target_times,
            )
            if plot_differences:
                diff_values = inductive - steady
                if normalize_differences:
                    strengths = np.array(
                        [np.nanmean([site_strength(inductive[idx]), site_strength(steady[idx])]) for idx in range(inductive.shape[0])],
                        dtype=float,
                    )
                    diff_values = apply_site_normalization(diff_values, strengths)
                    scale_unit_label = metric_label
                layers = [
                    {
                        "label": "Inductive - Magnetostatic",
                        "values": diff_values,
                        "color": "darkgreen",
                        "linewidth": 0.95,
                        "alpha": 0.95,
                        "zorder": 3,
                    }
                ]
            else:
                layers = [
                    {
                        "label": "Inductive",
                        "values": inductive,
                        "color": "red",
                        "linewidth": 0.9,
                        "alpha": 0.9,
                        "zorder": 3,
                    },
                    {
                        "label": "Magnetostatic",
                        "values": steady,
                        "color": "royalblue",
                        "linewidth": 0.9,
                        "alpha": 0.9,
                        "zorder": 2,
                    },
                ]
            mode_label = "difference" if plot_differences else "comparison"
            norm_label = f", normalized by local {metric_label}" if normalize_differences else ""
            title = f"Ground Curve Map: {component} {mode_label}{norm_label} ({time_label})"
            curve_width_deg = 10.0
            curve_height_deg = 3.0
            show_anchor_points = False
            station_labels = None

        finite_arrays = []
        for layer in layers:
            layer_values = np.asarray(layer["values"], dtype=float)
            finite_layer = np.abs(layer_values[np.isfinite(layer_values)])
            if finite_layer.size > 0:
                finite_arrays.append(finite_layer)
        finite_values = np.concatenate(finite_arrays) if finite_arrays else np.array([], dtype=float)
        if finite_values.size == 0:
            value_scale = 1.0
        else:
            value_scale = float(np.nanpercentile(finite_values, 95.0))
            if not np.isfinite(value_scale) or value_scale <= 0.0:
                value_scale = float(np.nanmax(finite_values)) if finite_values.size > 0 else 1.0
            value_scale = max(value_scale, 1.0)

        full_display_scale = self._round_up_scale(2.0 * value_scale, normalized=normalize_differences)
        value_scale = max(0.5 * full_display_scale, np.finfo(float).tiny)
        scale_annotation = self._format_scale_label(
            full_display_scale,
            scale_unit_label=scale_unit_label,
            normalized=normalize_differences,
        )

        return GroundCurveMapPayload(
            anchor_lon=anchor_lon,
            anchor_lat=anchor_lat,
            layers=layers,
            normalized_time=target_norm,
            value_scale=value_scale,
            curve_width_deg=curve_width_deg,
            curve_height_deg=curve_height_deg,
            show_anchor_points=show_anchor_points,
            title=title,
            n_sites=int(anchor_lon.size),
            station_labels=station_labels,
            show_station_labels=bool(include_data and show_station_labels),
            is_difference=plot_differences,
            is_normalized_difference=normalize_differences,
            scale_display_value=full_display_scale,
            scale_annotation=scale_annotation,
            duration_annotation=duration_annotation,
        )


def _add_station_labels(ax: Any, payload: GroundCurveMapPayload) -> list[Any]:
    labels = payload.station_labels
    if labels is None or not payload.show_station_labels:
        return []

    def wrap_lon(lon: float, center: float = 0.0) -> float:
        return ((float(lon) - center + 180.0) % 360.0) - 180.0 + center

    lon_sites = np.asarray([wrap_lon(lon) for lon in payload.anchor_lon], dtype=float)
    lat_sites = np.asarray(payload.anchor_lat, dtype=float)
    half_width = 0.5 * float(payload.curve_width_deg)
    half_height = max(1.2, 0.85 * float(payload.curve_height_deg))
    curve_pad_x = 1.2
    curve_pad_y = 1.0
    curve_boxes = [
        (
            lon - half_width - curve_pad_x,
            lon + half_width + curve_pad_x,
            lat - half_height - curve_pad_y,
            lat + half_height + curve_pad_y,
        )
        for lon, lat in zip(lon_sites, lat_sites)
    ]
    artists = []
    map_xmin, map_xmax = -176.0, 176.0
    map_ymin, map_ymax = -82.0, 82.0
    right_bias = 0.05

    def label_box(x: float, y: float, ha: str, label_width: float, label_height: float) -> tuple[float, float, float, float]:
        if ha == "left":
            xmin, xmax = x, x + label_width
        else:
            xmin, xmax = x - label_width, x
        ymin, ymax = y - 0.5 * label_height, y + 0.5 * label_height
        return (xmin, xmax, ymin, ymax)

    def horizontal_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    def vertical_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        if a[3] < b[2]:
            return b[2] - a[3]
        if b[3] < a[2]:
            return a[2] - b[3]
        return 0.0

    def overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        return horizontal_overlap(a, b) * max(0.0, min(a[3], b[3]) - max(a[2], b[2]))

    def boundary_penalty(box: tuple[float, float, float, float]) -> float:
        penalty = 0.0
        if box[0] < map_xmin:
            penalty += (map_xmin - box[0]) * 30.0
        if box[1] > map_xmax:
            penalty += (box[1] - map_xmax) * 30.0
        if box[2] < map_ymin:
            penalty += (map_ymin - box[2]) * 30.0
        if box[3] > map_ymax:
            penalty += (box[3] - map_ymax) * 30.0
        return penalty

    def curve_penalty(box: tuple[float, float, float, float], self_index: int) -> float:
        penalty = 0.0
        for idx, other in enumerate(curve_boxes):
            if idx == self_index:
                continue
            hoverlap = horizontal_overlap(box, other)
            if hoverlap <= 0.0:
                continue
            vgap = vertical_gap(box, other)
            penalty += hoverlap / (vgap + 0.35)
            penalty += 8.0 * overlap_area(box, other)
        return penalty

    def label_pair_penalty(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        hoverlap = horizontal_overlap(a, b)
        if hoverlap <= 0.0:
            return 0.0
        vgap = vertical_gap(a, b)
        return hoverlap / (vgap + 0.35) + 4.0 * overlap_area(a, b)

    stations: list[dict[str, Any]] = []
    for idx, (lon, lat, label) in enumerate(zip(lon_sites, lat_sites, labels)):
        label_str = str(label)
        label_width = max(4.8, 1.05 * len(label_str) + 1.8)
        label_height = 1.7
        side_dx = half_width + 1.5
        right_x, right_y, right_ha, right_va = lon + side_dx, lat, "left", "center"
        left_x, left_y, left_ha, left_va = lon - side_dx, lat, "right", "center"
        right_box = label_box(right_x, right_y, right_ha, label_width, label_height)
        left_box = label_box(left_x, left_y, left_ha, label_width, label_height)
        stations.append(
            {
                "label": label_str,
                "right": {"xy": (right_x, right_y), "ha": right_ha, "va": right_va, "box": right_box},
                "left": {"xy": (left_x, left_y), "ha": left_ha, "va": left_va, "box": left_box},
                "fixed_right": boundary_penalty(right_box) + curve_penalty(right_box, idx),
                "fixed_left": boundary_penalty(left_box) + curve_penalty(left_box, idx),
            }
        )

    order = list(np.argsort(lon_sites))
    choices = ["right"] * len(stations)

    def local_score(i: int, side: str) -> float:
        station = stations[i]
        box = station[side]["box"]
        score = float(station[f"fixed_{side}"])
        if side == "left":
            score += right_bias
        for j, other_side in enumerate(choices):
            if j == i:
                continue
            other_box = stations[j][other_side]["box"]
            score += label_pair_penalty(box, other_box)
        return score

    for _ in range(6):
        changed = False
        for i in order:
            right_score = local_score(i, "right")
            left_score = local_score(i, "left")
            new_choice = "left" if left_score < right_score else "right"
            if new_choice != choices[i]:
                choices[i] = new_choice
                changed = True
        if not changed:
            break

    for station, side in zip(stations, choices):
        chosen = station[side]
        x, y = chosen["xy"]
        artists.append(
            ax.text(
                x,
                y,
                station["label"],
                transform=ccrs.PlateCarree(),
                fontsize=6.5,
                ha=chosen["ha"],
                va=chosen["va"],
                color="0.15",
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.12},
                zorder=5,
            )
        )
    return artists


def _add_scale_inset(ax: Any, payload: GroundCurveMapPayload) -> list[Any]:
    x0, y0 = 0.048, 0.048

    # Match the scale bars to the actual embedded curve size in map coordinates.
    curve_width_ax = float(payload.curve_width_deg) / 360.0
    trace_height_ax = 2.0 * float(payload.curve_height_deg) / 180.0
    curve_width_ax = max(curve_width_ax, 0.02)
    trace_height_ax = max(trace_height_ax, 0.025)
    full_trace_scale = 2.0 * max(float(payload.value_scale), np.finfo(float).tiny)
    scale_ratio = float(payload.scale_display_value) / full_trace_scale
    # The horizontal bar represents the trace centerline. The vertical bar
    # therefore shows the full peak-to-peak displayed span about that centerline.
    bar_height_ax = trace_height_ax * max(scale_ratio, 1e-6)

    left_margin_ax = 0.026
    right_margin_ax = 0.006
    bottom_margin_ax = 0.012
    top_margin_ax = 0.012
    inset_width_ax = left_margin_ax + curve_width_ax + right_margin_ax
    inset_height_ax = bottom_margin_ax + bar_height_ax + top_margin_ax

    scale_ax = ax.inset_axes(
        [x0, y0, inset_width_ax, inset_height_ax],
        transform=ax.transAxes,
        zorder=11,
    )
    scale_ax.set_facecolor("none")
    scale_ax.set_xlim(0.0, 1.0)
    scale_ax.set_ylim(0.0, 1.0)
    scale_ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        bottom=True,
        left=True,
        top=False,
        right=False,
        direction="out",
        length=4.0,
        width=1.0,
        colors="0.2",
        pad=1.0,
    )
    scale_ax.set_xticks([])
    scale_ax.set_yticks([])
    scale_ax.patch.set_alpha(0.0)

    x_origin = left_margin_ax / inset_width_ax
    x_end = (left_margin_ax + curve_width_ax) / inset_width_ax
    y_bottom = bottom_margin_ax / inset_height_ax
    y_top = (bottom_margin_ax + bar_height_ax) / inset_height_ax
    y_center = 0.5 * (y_bottom + y_top)

    for spine_name, spine in scale_ax.spines.items():
        spine.set_visible(spine_name in {"left", "bottom"})
        spine.set_color("0.2")
        spine.set_linewidth(1.2)

    scale_ax.spines["left"].set_position(("axes", x_origin))
    scale_ax.spines["left"].set_bounds(y_bottom, y_top)
    scale_ax.spines["bottom"].set_position(("axes", y_center))
    scale_ax.spines["bottom"].set_bounds(x_origin, x_end)

    scale_ax.set_xticks([x_origin, x_end])
    scale_ax.set_yticks([y_bottom, y_top])

    duration_text = scale_ax.text(
        0.5 * (x_origin + x_end),
        y_bottom - 0.10,
        payload.duration_annotation,
        fontsize=7.8,
        ha="center",
        va="top",
        color="0.15",
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.15},
    )

    amplitude_text = scale_ax.text(
        x_origin - 0.16,
        y_center,
        payload.scale_annotation,
        fontsize=7.8,
        ha="center",
        va="center",
        rotation=90,
        rotation_mode="anchor",
        color="0.15",
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.15},
    )

    return [scale_ax, duration_text, amplitude_text]


def draw_ground_curve_map_figure(fig_handle: Any, payload: GroundCurveMapPayload) -> list[Any]:
    artists: list[Any] = []
    fig_handle.clear()
    fig_handle.set_size_inches(14, 7)
    fig_handle.set_constrained_layout(True)
    ax = fig_handle.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_global()
    ax.coastlines(color="0.7", linewidth=0.7, zorder=1)
    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        color="gray",
        alpha=0.4,
        linestyle="--",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False

    if payload.empty_message is not None:
        ax.text(
            0.5,
            0.5,
            payload.empty_message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            color="red",
        )
        fig_handle.suptitle(payload.title, fontsize=15)
        return artists

    result = draw_timeseries_curve_map(
        ax,
        site_lon=payload.anchor_lon,
        site_lat=payload.anchor_lat,
        normalized_time=payload.normalized_time,
        layers=payload.layers,
        curve_width_deg=payload.curve_width_deg,
        curve_height_deg=payload.curve_height_deg,
        value_scale=payload.value_scale,
        central_longitude=0.0,
        show_anchor_points=payload.show_anchor_points,
        anchor_point_kwargs={"marker": "x", "s": 12, "color": "dimgray", "linewidths": 0.5},
        legend_kwargs={"loc": "upper right", "fontsize": 9},
    )
    artists.extend(result["artists"])
    artists.extend(_add_station_labels(ax, payload))
    artists.extend(_add_scale_inset(ax, payload))
    if result.get("legend") is not None:
        artists.append(result["legend"])

    fig_handle.suptitle(payload.title, fontsize=15)
    return artists
