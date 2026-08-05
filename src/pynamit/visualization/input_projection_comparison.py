"""Compare prepared inputs with their projected reconstruction."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import cartopy.crs as ccrs
import h5py
import matplotlib.pyplot as plt
import numpy as np
from kompe import SphericalGrid
from kompe.spherical_transform import SphericalTransform, grid_sqrt_area_weights

from pynamit.simulation.electrodynamics.ionospheric_closure import CONDUCTANCE_REFERENCE_S
from pynamit.visualization.input_projection import evaluate_projected_input
from pynamit.visualization.map_coordinates import MapCoordinateContext
from pynamit.visualization.plot_helpers import (
    build_percentile_color_scale,
    style_global_input_axis,
)
from pynamit.visualization.saved_run import SavedRunView

DEFAULT_PROJECTION_COMPARISON_FIELDS = ("etaP", "etaH", "SigmaP", "SigmaH", "jr", "Br")

_FIELD_DETAILS = {
    "etaP": {
        "label": r"$\eta_P$ [$\Omega$]",
        "units": "ohm",
        "series": "conductance",
        "component": "etaP",
        "grid": "ionosphere",
        "positive": True,
    },
    "etaH": {
        "label": r"$\eta_H$ [$\Omega$]",
        "units": "ohm",
        "series": "conductance",
        "component": "etaH",
        "grid": "ionosphere",
        "positive": True,
    },
    "SigmaP": {
        "label": r"$\Sigma_P$ [S]",
        "units": "S",
        "series": "conductance",
        "component": "SigmaP",
        "grid": "ionosphere",
        "positive": True,
    },
    "SigmaH": {
        "label": r"$\Sigma_H$ [S]",
        "units": "S",
        "series": "conductance",
        "component": "SigmaH",
        "grid": "ionosphere",
        "positive": True,
    },
    "jr": {
        "label": r"$j_r$ [A m$^{-2}$]",
        "units": "A m-2",
        "series": "boundary_jr",
        "component": "boundary_jr",
        "grid": "ionosphere",
        "positive": False,
    },
    "Br": {
        "label": r"$\Delta B_r$ [T]",
        "units": "T",
        "series": "boundary_Br",
        "component": "boundary_Br",
        "grid": "magnetosphere",
        "positive": False,
    },
}


def _parse_timestamp(value) -> dt.datetime:
    """Parse one prepared-forcing timestamp."""
    if isinstance(value, bytes):
        value = value.decode("ascii")
    timestamp = dt.datetime.fromisoformat(str(value))
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return timestamp


def _forcing_times(h5_file) -> tuple[list[dt.datetime], np.ndarray]:
    """Return nominal timestamps and relative simulation seconds."""
    timestamps = [_parse_timestamp(value) for value in h5_file["time"][:]]
    if not timestamps:
        raise ValueError("Prepared forcing contains no timestamps.")
    seconds = np.asarray(
        [(timestamp - timestamps[0]).total_seconds() for timestamp in timestamps], dtype=float
    )
    return timestamps, seconds


def _comparison_steps(requested_steps, n_steps: int) -> tuple[int, ...]:
    """Normalize steps or choose the first, middle, and final steps."""
    if n_steps <= 0:
        raise ValueError("Projection comparison requires at least one forcing step.")
    if requested_steps is None:
        return tuple(dict.fromkeys((0, n_steps // 2, n_steps - 1)))
    normalized = []
    for requested in requested_steps:
        if isinstance(requested, (bool, np.bool_)) or not isinstance(requested, (int, np.integer)):
            raise ValueError("Projection-comparison steps must be integer indices.")
        step = int(requested)
        if step < 0:
            step += n_steps
        if not 0 <= step < n_steps:
            raise ValueError(
                f"Projection-comparison step {requested} is outside a {n_steps}-step forcing."
            )
        if step not in normalized:
            normalized.append(step)
    if not normalized:
        raise ValueError("Projection comparison requires at least one selected step.")
    return tuple(normalized)


def _validate_fields(fields) -> tuple[str, ...]:
    """Return a unique, validated field sequence."""
    fields = tuple(fields)
    unknown = sorted(set(fields) - set(_FIELD_DETAILS))
    if unknown:
        raise ValueError(
            f"Unsupported projection-comparison fields {unknown}; "
            f"choose from {sorted(_FIELD_DETAILS)}."
        )
    if not fields:
        raise ValueError("Projection comparison requires at least one field.")
    if len(set(fields)) != len(fields):
        raise ValueError("Projection-comparison fields must not contain duplicates.")
    return fields


def _comparison_grids(h5_file):
    """Return prepared GEO grids and their fitting weights."""
    ionosphere_lat = np.asarray(h5_file["ionosphere_lat"][:], dtype=float)
    ionosphere_lon = np.asarray(h5_file["ionosphere_lon"][:], dtype=float)
    boundary_lat = np.asarray(h5_file["boundary_lat"][:], dtype=float)
    boundary_lon = np.asarray(h5_file["boundary_lon"][:], dtype=float)
    boundary_weights = np.asarray(h5_file["boundary_solid_angle"][:], dtype=float)

    ionosphere_grid = SphericalGrid(lat=ionosphere_lat, lon=ionosphere_lon)
    boundary_grid = SphericalGrid(
        lat=boundary_lat, lon=boundary_lon, area_weights=boundary_weights
    )
    return {
        "ionosphere": {
            "grid": ionosphere_grid,
            "latitude": ionosphere_lat,
            "longitude": ionosphere_lon,
            "shape": ionosphere_lat.shape,
            "weights": np.asarray(grid_sqrt_area_weights(ionosphere_grid), dtype=float) ** 2,
        },
        "magnetosphere": {
            "grid": boundary_grid,
            "latitude": boundary_lat,
            "longitude": boundary_lon,
            "shape": boundary_lat.shape,
            "weights": np.asarray(grid_sqrt_area_weights(boundary_grid), dtype=float) ** 2,
        },
    }


def _raw_field(h5_file, field: str, step: int) -> np.ndarray:
    """Read one prepared field in the units used by PynaMIT."""
    if field == "Br":
        return np.asarray(h5_file["delta_Br"][step], dtype=float) * 1e-9
    if field == "jr":
        return np.asarray(h5_file["jr"][step], dtype=float) * 1e-6

    sigma_p = np.asarray(h5_file["SP"][step], dtype=float)
    sigma_h = np.asarray(h5_file["SH"][step], dtype=float)
    if field == "SigmaP":
        return sigma_p
    if field == "SigmaH":
        return sigma_h
    denominator = sigma_p**2 + sigma_h**2
    if np.any(denominator <= np.finfo(float).tiny):
        raise ValueError("Prepared conductances cannot both be zero.")
    if field == "etaP":
        return sigma_p / denominator
    if field == "etaH":
        return sigma_h / denominator
    raise AssertionError(f"Unhandled projection-comparison field {field!r}.")


def _collect_comparison_data(h5_file, input_series, steps, fields, time_seconds, grids):
    """Evaluate all selected source and reconstructed fields once."""
    evaluators = {}
    evaluations = {}
    comparison = {}

    for step in steps:
        for field in fields:
            details = _FIELD_DETAILS[field]
            grid_details = grids[details["grid"]]
            series_key = details["series"]
            if series_key not in input_series.datasets:
                raise ValueError(
                    f"Projected package has no {series_key!r} input required for {field!r}."
                )
            field_space = input_series.get_field_space(series_key)
            evaluator_key = (field_space.representation.signature, details["grid"])
            if evaluator_key not in evaluators:
                evaluators[evaluator_key] = SphericalTransform(
                    field_space.representation, grid_details["grid"]
                )
            evaluation_key = (step, series_key, details["grid"])
            if evaluation_key not in evaluations:
                evaluations[evaluation_key] = evaluate_projected_input(
                    input_series,
                    series_key,
                    float(time_seconds[step]),
                    transform=evaluators[evaluator_key],
                )
            projected = np.asarray(
                evaluations[evaluation_key][details["component"]], dtype=float
            ).reshape(grid_details["shape"])
            raw = _raw_field(h5_file, field, step)
            comparison[(step, field)] = {
                "input": raw,
                "projected": projected,
                "residual": projected - raw,
            }
    return comparison


def _weighted_percentile(values, weights, percentile: float) -> float:
    """Return an area-weighted percentile of finite flattened values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.broadcast_to(np.asarray(weights, dtype=float).reshape(-1), values.shape)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[finite]
    weights = weights[finite]
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order])
    target = float(percentile) / 100.0 * cumulative[-1]
    return float(values[np.searchsorted(cumulative, target, side="left")])


def _error_metrics(raw, projected, weights) -> dict[str, float | int]:
    """Return fit-relevant area-weighted reconstruction metrics."""
    raw = np.asarray(raw, dtype=float).reshape(-1)
    projected = np.asarray(projected, dtype=float).reshape(-1)
    weights = np.broadcast_to(np.asarray(weights, dtype=float).reshape(-1), raw.shape)
    finite = np.isfinite(raw) & np.isfinite(projected) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        raise ValueError("Projection comparison contains no finite weighted samples.")
    raw = raw[finite]
    projected = projected[finite]
    weights = weights[finite]
    weights = weights / np.sum(weights)
    residual = projected - raw
    rms_error = float(np.sqrt(np.sum(weights * residual**2)))
    rms_input = float(np.sqrt(np.sum(weights * raw**2)))
    return {
        "weighted_rms_error": rms_error,
        "weighted_relative_rms_error": (
            rms_error / rms_input if rms_input > np.finfo(float).tiny else float("nan")
        ),
        "weighted_bias": float(np.sum(weights * residual)),
        "weighted_p95_absolute_error": _weighted_percentile(np.abs(residual), weights, 95.0),
        "max_absolute_error": float(np.max(np.abs(residual))),
        "minimum_input": float(np.min(raw)),
        "minimum_projected": float(np.min(projected)),
        "maximum_input": float(np.max(raw)),
        "maximum_projected": float(np.max(projected)),
        "negative_projected_count": int(np.count_nonzero(projected < 0.0)),
        "negative_projected_area_fraction": float(np.sum(weights[projected < 0.0])),
    }


def _physical_bounds(h5_file):
    """Return bounds implied by the global conductance floor."""
    sigma_p_min = float(h5_file.attrs["pedersen_conductance_floor_S"])
    sigma_h_min = float(h5_file.attrs["hall_conductance_floor_S"])
    return {
        "etaP": (
            0.0,
            1.0 / (2.0 * sigma_h_min)
            if sigma_p_min <= sigma_h_min
            else sigma_p_min / (sigma_p_min**2 + sigma_h_min**2),
        ),
        "etaH": (
            0.0,
            1.0 / (2.0 * sigma_p_min)
            if sigma_h_min <= sigma_p_min
            else sigma_h_min / (sigma_p_min**2 + sigma_h_min**2),
        ),
        "SigmaP": (sigma_p_min, None),
        "SigmaH": (sigma_h_min, None),
    }


def _bound_metrics(projected, weights, bounds) -> dict[str, float | int | None]:
    """Measure reconstructed area outside physical bounds."""
    lower, upper = bounds
    projected = np.asarray(projected, dtype=float).reshape(-1)
    weights = np.broadcast_to(np.asarray(weights, dtype=float).reshape(-1), projected.shape)
    finite = np.isfinite(projected) & np.isfinite(weights) & (weights > 0.0)
    projected = projected[finite]
    weights = weights[finite]
    weights = weights / np.sum(weights)
    below = projected < lower
    above = np.zeros(projected.shape, dtype=bool) if upper is None else projected > upper
    return {
        "expected_minimum": float(lower),
        "expected_maximum": None if upper is None else float(upper),
        "projected_below_minimum_count": int(np.count_nonzero(below)),
        "projected_below_minimum_area_fraction": float(np.sum(weights[below])),
        "projected_above_maximum_count": int(np.count_nonzero(above)),
        "projected_above_maximum_area_fraction": float(np.sum(weights[above])),
    }


def _comparison_report(
    *,
    forcing_path,
    projected_directory,
    steps,
    fields,
    timestamps,
    time_seconds,
    grids,
    comparison,
    physical_bounds,
):
    """Build serializable per-step and aggregate diagnostics."""
    per_step = {}
    aggregate = {}
    for field in fields:
        details = _FIELD_DETAILS[field]
        weights = grids[details["grid"]]["weights"]
        raw_values = []
        projected_values = []
        repeated_weights = []
        for step in steps:
            values = comparison[(step, field)]
            raw_values.append(values["input"].reshape(-1))
            projected_values.append(values["projected"].reshape(-1))
            repeated_weights.append(weights)
            step_metrics = _error_metrics(values["input"], values["projected"], weights)
            if field in physical_bounds:
                step_metrics.update(
                    _bound_metrics(values["projected"], weights, physical_bounds[field])
                )
            per_step.setdefault(str(step), {})[field] = step_metrics
        aggregate_metrics = _error_metrics(
            np.concatenate(raw_values),
            np.concatenate(projected_values),
            np.concatenate(repeated_weights),
        )
        if field in physical_bounds:
            aggregate_metrics.update(
                _bound_metrics(
                    np.concatenate(projected_values),
                    np.concatenate(repeated_weights),
                    physical_bounds[field],
                )
            )
        aggregate[field] = {"units": details["units"], **aggregate_metrics}

    return {
        "schema_version": 2,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prepared_forcing": str(Path(forcing_path).expanduser().resolve()),
        "projected_directory": str(Path(projected_directory).expanduser().resolve()),
        "residual_definition": "projected_minus_prepared",
        "metric_weighting": "projection_surface_area",
        "conductance_representation": {
            "fitted": ["log_conductance_magnitude", "log_hall_to_pedersen_ratio"],
            "derived": ["SigmaP", "SigmaH", "etaP", "etaH"],
            "reference_conductance_S": CONDUCTANCE_REFERENCE_S,
        },
        "selected_steps": [
            {
                "index": int(step),
                "timestamp": timestamps[step].isoformat(),
                "time_seconds": float(time_seconds[step]),
            }
            for step in steps
        ],
        "fields": list(fields),
        "per_step": per_step,
        "aggregate": aggregate,
    }


def plot_scalar_map_on_ax(
    ax, longitude, latitude, values, *, norm, cmap, coordinate_context, left_labels, bottom_labels
):
    """Draw one global field with the standard GEO presentation."""
    mappable = ax.pcolormesh(
        longitude,
        latitude,
        np.ma.masked_invalid(values),
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
        zorder=1,
    )
    style_global_input_axis(
        ax,
        coordinate_context=coordinate_context,
        draw_labels=True,
        left_labels=left_labels,
        bottom_labels=bottom_labels,
    )
    return mappable


def _field_scales(comparison, steps, field, *, vmin_percentile, vmax_percentile):
    """Return shared physical and residual scales for one field."""
    details = _FIELD_DETAILS[field]
    physical_values = [
        comparison[(step, field)][source] for step in steps for source in ("input", "projected")
    ]
    scale_values = (
        [np.maximum(values, 0.0) for values in physical_values]
        if details["positive"]
        else physical_values
    )
    physical_scale = build_percentile_color_scale(
        scale_values,
        strictly_positive=details["positive"],
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
        label=field,
    )
    residual_scale = build_percentile_color_scale(
        [comparison[(step, field)]["residual"] for step in steps],
        strictly_positive=False,
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
        label=f"{field} residual",
    )
    return physical_scale, residual_scale


def _render_comparison_figure(
    *,
    steps,
    fields,
    timestamps,
    time_seconds,
    grids,
    comparison,
    report,
    vmin_percentile,
    vmax_percentile,
):
    """Render prepared, projected, and residual rows."""
    rows_per_field = 3
    n_rows = rows_per_field * len(fields)
    n_columns = len(steps)
    figure = plt.figure(
        figsize=(max(9.0, 3.4 * n_columns), max(7.0, 2.25 * n_rows)), layout="constrained"
    )
    axes = np.empty((n_rows, n_columns), dtype=object)
    contexts = [MapCoordinateContext.geographic(timestamps[step]) for step in steps]
    grid_spec = figure.add_gridspec(n_rows, n_columns)
    row_labels = ("Prepared", "Projected", "Residual")

    for field_index, field in enumerate(fields):
        details = _FIELD_DETAILS[field]
        grid_details = grids[details["grid"]]
        physical_scale, residual_scale = _field_scales(
            comparison,
            steps,
            field,
            vmin_percentile=vmin_percentile,
            vmax_percentile=vmax_percentile,
        )
        physical_mappable = None
        residual_mappable = None
        for column, step in enumerate(steps):
            context = contexts[column]
            for row_offset, source in enumerate(("input", "projected", "residual")):
                row = rows_per_field * field_index + row_offset
                axis = figure.add_subplot(grid_spec[row, column], projection=context.projection())
                axes[row, column] = axis
                scale = residual_scale if source == "residual" else physical_scale
                mappable = plot_scalar_map_on_ax(
                    axis,
                    grid_details["longitude"],
                    grid_details["latitude"],
                    comparison[(step, field)][source],
                    norm=scale["norm"],
                    cmap=scale["cmap"],
                    coordinate_context=context,
                    left_labels=column == 0,
                    bottom_labels=row == n_rows - 1,
                )
                if source == "residual":
                    residual_mappable = mappable
                    relative_error = report["per_step"][str(step)][field][
                        "weighted_relative_rms_error"
                    ]
                    axis.set_title(f"relative RMS {relative_error:.2%}", fontsize=8)
                else:
                    physical_mappable = mappable
                if column == 0:
                    axis.text(
                        -0.18,
                        0.5,
                        f"{details['label']}\n{row_labels[row_offset]}",
                        transform=axis.transAxes,
                        ha="right",
                        va="center",
                        fontsize=8,
                    )
                if row == 0:
                    axis.text(
                        0.5,
                        1.14,
                        f"{timestamps[step]:%Y-%m-%d %H:%M:%S}\nt = {time_seconds[step]:g} s",
                        transform=axis.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        physical_rows = (
            axes[rows_per_field * field_index : rows_per_field * field_index + 2, :]
            .reshape(-1)
            .tolist()
        )
        residual_rows = (
            axes[rows_per_field * field_index + 2 : rows_per_field * field_index + 3, :]
            .reshape(-1)
            .tolist()
        )
        figure.colorbar(
            physical_mappable, ax=physical_rows, label=details["label"], shrink=0.82, pad=0.01
        )
        figure.colorbar(
            residual_mappable,
            ax=residual_rows,
            label=f"Projected − prepared [{details['units']}]",
            shrink=0.82,
            pad=0.01,
        )

    figure.suptitle("Prepared MAGE forcing and projected reconstruction", fontsize=14)
    return figure


def plot_input_projection_comparison(
    forcing_path,
    projection_directory,
    *,
    steps=None,
    fields=DEFAULT_PROJECTION_COMPARISON_FIELDS,
    figure_path=None,
    metrics_path=None,
    operator_cache_directory=None,
    vmin_percentile=0.2,
    vmax_percentile=99.8,
):
    """Compare prepared MAGE inputs with a projected input package.

    ``steps=None`` selects the first, middle, and final
    available forcing steps. The returned report contains area-weighted
    errors for both the fitted resistance and derived physical
    conductance fields.
    """
    forcing_path = Path(forcing_path).expanduser()
    projection_directory = Path(projection_directory).expanduser()
    fields = _validate_fields(fields)
    run_view = SavedRunView.from_directory(
        projection_directory, operator_cache_directory=operator_cache_directory
    )
    input_series = run_view.load_input_series()
    with h5py.File(forcing_path, "r") as h5_file:
        timestamps, time_seconds = _forcing_times(h5_file)
        required_series = {_FIELD_DETAILS[field]["series"] for field in fields}
        missing_series = sorted(required_series - set(input_series.datasets))
        if missing_series:
            raise ValueError(f"Projected package is missing required inputs {missing_series}.")
        final_projected_time = min(
            float(np.max(input_series.datasets[key].time.values)) for key in required_series
        )
        available_steps = int(np.count_nonzero(time_seconds <= final_projected_time + 1e-9))
        steps = _comparison_steps(steps, available_steps)
        grids = _comparison_grids(h5_file)
        comparison = _collect_comparison_data(
            h5_file, input_series, steps, fields, time_seconds, grids
        )
        report = _comparison_report(
            forcing_path=forcing_path,
            projected_directory=projection_directory,
            steps=steps,
            fields=fields,
            timestamps=timestamps,
            time_seconds=time_seconds,
            grids=grids,
            comparison=comparison,
            physical_bounds=_physical_bounds(h5_file),
        )
        figure = _render_comparison_figure(
            steps=steps,
            fields=fields,
            timestamps=timestamps,
            time_seconds=time_seconds,
            grids=grids,
            comparison=comparison,
            report=report,
            vmin_percentile=vmin_percentile,
            vmax_percentile=vmax_percentile,
        )

    if metrics_path is not None:
        metrics_path = Path(metrics_path).expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(_json_compatible(report), indent=2, allow_nan=False) + "\n"
        )
        print(f"Projection metrics written to {metrics_path}", flush=True)
    if figure_path is None:
        plt.show()
    else:
        figure_path = Path(figure_path).expanduser()
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=180)
        plt.close(figure)
        print(f"Projection comparison written to {figure_path}", flush=True)
    return report


def _json_compatible(value):
    """Replace non-finite diagnostics with JSON null values."""
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_input_projection_diagnostics(
    forcing_path,
    projection_directory,
    *,
    output_directory=None,
    timesteps=None,
    fields=DEFAULT_PROJECTION_COMPARISON_FIELDS,
    operator_cache_directory=None,
):
    """Write the comparison figure and JSON metrics."""
    projection_directory = Path(projection_directory).expanduser()
    diagnostics_directory = (
        projection_directory / "diagnostics"
        if output_directory is None
        else Path(output_directory).expanduser()
    )
    diagnostics_directory.mkdir(parents=True, exist_ok=True)
    figure_path = diagnostics_directory / "input_projection_comparison.png"
    metrics_path = diagnostics_directory / "input_projection_metrics.json"
    report = plot_input_projection_comparison(
        forcing_path,
        projection_directory,
        steps=timesteps,
        fields=fields,
        figure_path=figure_path,
        metrics_path=metrics_path,
        operator_cache_directory=operator_cache_directory,
    )
    return {"figure": figure_path, "metrics": metrics_path, "report": report}


__all__ = [
    "DEFAULT_PROJECTION_COMPARISON_FIELDS",
    "plot_input_projection_comparison",
    "plot_scalar_map_on_ax",
    "write_input_projection_diagnostics",
]
