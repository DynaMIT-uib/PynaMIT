"""Compare raw driver inputs with their projected reconstruction."""

import datetime as dt

import cartopy.crs as ccrs
import h5py as h5
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from pynamit.sphere import Grid
from pynamit.sphere.spherical_transform import SphericalTransform
from pynamit.visualization.input_projection import evaluate_projected_input
from pynamit.visualization.map_coordinates import MapCoordinateContext
from pynamit.visualization.plot_helpers import (
    build_percentile_color_scale,
    style_global_input_axis,
)
from pynamit.visualization.saved_run import SavedRunView

_INPUT_TIMESERIES_KEY = {
    "Br": "Br",
    "jr": "jr",
    "SH": "resistance",
    "SP": "resistance",
    "u_mag": "u",
    "u_theta": "u",
    "u_phi": "u",
}

_PROJECTED_COMPONENT_KEY = {
    "Br": "Br",
    "jr": "jr",
    "SH": "SigmaH",
    "SP": "SigmaP",
    "u_mag": "u_mag",
    "u_theta": "u_theta",
    "u_phi": "u_phi",
}

_INPUT_DATA_DETAILS = {
    "Br": {
        "label": r"$\Delta B_r$ [T]",
        "strictly_positive": False,
        "grid_type": "magnetosphere",
        "h5_key_primary": "delta_Br",
    },
    "jr": {
        "label": r"$j_r$ [A/m$^2$]",
        "strictly_positive": False,
        "grid_type": "ionosphere",
        "h5_key_primary": "jr",
    },
    "SH": {
        "label": r"$\Sigma_H$ [S]",
        "strictly_positive": True,
        "grid_type": "ionosphere",
        "h5_key_primary": "SH",
    },
    "SP": {
        "label": r"$\Sigma_P$ [S]",
        "strictly_positive": True,
        "grid_type": "ionosphere",
        "h5_key_primary": "SP",
    },
    "u_mag": {
        "label": r"$|u|$ [m/s]",
        "strictly_positive": True,
        "grid_type": "ionosphere",
        "h5_key_primary": "u_p_theta",
        "h5_key_secondary": "u_p_phi",
    },
    "u_theta": {
        "label": r"$u_\theta$ (South) [m/s]",
        "strictly_positive": False,
        "grid_type": "ionosphere",
        "h5_key_primary": "u_p_theta",
        "h5_key_secondary": "u_p_phi",
    },
    "u_phi": {
        "label": r"$u_\phi$ (East) [m/s]",
        "strictly_positive": False,
        "grid_type": "ionosphere",
        "h5_key_primary": "u_p_theta",
        "h5_key_secondary": "u_p_phi",
    },
}


def _read_raw_input_field(h5file, data_type, details, timestep, shape):
    if data_type == "Br":
        return h5file[details["h5_key_primary"]][timestep, :, :] * 1e-9
    if data_type == "jr":
        return h5file[details["h5_key_primary"]][timestep, :, :] * 1e-6
    if data_type in {"SH", "SP"}:
        return h5file[details["h5_key_primary"]][timestep, :, :]
    if data_type in {"u_mag", "u_theta", "u_phi"}:
        u_theta_h5 = h5file[details["h5_key_primary"]][timestep, :, :]
        u_phi_h5 = h5file[details["h5_key_secondary"]][timestep, :, :]
        if data_type == "u_mag":
            return np.hypot(u_theta_h5, u_phi_h5)
        if data_type == "u_theta":
            return u_theta_h5
        return u_phi_h5
    return np.full(shape, np.nan)


def plot_scalar_map_on_ax(
    ax,
    lon_coords_2d,
    lat_coords_2d,
    data_2d_arr,
    title="",
    cmap="viridis",
    norm=None,
    coordinate_context=None,
    left_labels=True,
    bottom_labels=True,
):
    """Plot a scalar map with specified coordinates and data."""
    if norm is None:
        raise ValueError("Norm object must be provided to plot_scalar_map_on_ax.")
    data_to_plot_masked = np.ma.masked_invalid(data_2d_arr)
    im = ax.pcolormesh(
        lon_coords_2d,
        lat_coords_2d,
        data_to_plot_masked,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
        zorder=1,
    )
    style_global_input_axis(
        ax,
        coordinate_context=coordinate_context,
        draw_labels=coordinate_context is not None,
        left_labels=left_labels,
        bottom_labels=bottom_labels,
    )
    ax.set_title(title, fontsize=9)
    return im


def _validate_comparison_request(h5file, timesteps, data_types, positive_scale_type):
    """Validate requested times, fields, and color-scale policy."""
    if not timesteps:
        print("Warning: No timesteps specified for plotting. Exiting.")
        return False
    num_steps = h5file["time"].shape[0]
    invalid_timesteps = [step for step in timesteps if not 0 <= step < num_steps]
    if invalid_timesteps:
        raise ValueError(
            f"Invalid timesteps provided: {invalid_timesteps}. "
            f"All timesteps must be within the range [0, {num_steps - 1}]."
        )
    if not data_types:
        print("Warning: No data types specified for plotting. Exiting.")
        return False
    unknown_data_types = sorted(set(data_types) - set(_INPUT_DATA_DETAILS))
    if unknown_data_types:
        raise ValueError(
            f"Unsupported data types for projection comparison: {unknown_data_types}. "
            f"Supported values are {sorted(_INPUT_DATA_DETAILS)}."
        )
    if positive_scale_type not in {"linear", "log"}:
        raise ValueError("strictly_positive_scale_type must be 'linear' or 'log'.")
    return True


def _comparison_grids(h5file):
    """Return the prepared fixed-GEO comparison grids."""
    ionosphere_lat = h5file["ionosphere_lat"][:]
    ionosphere_lon = h5file["ionosphere_lon"][:]
    magnetosphere_lat = h5file["boundary_lat"][:]
    magnetosphere_lon = h5file["boundary_lon"][:]
    return {
        "ionosphere": (ionosphere_lon, ionosphere_lat),
        "magnetosphere": (magnetosphere_lon, magnetosphere_lat),
    }


def _target_grid(grids, grid_type):
    """Return longitude, latitude, and shape for one input-grid type."""
    longitude, latitude = grids[grid_type]
    return longitude, latitude, latitude.shape


def _collect_comparison_data(h5file, input_series, timesteps, input_dt, data_types, grids):
    """Evaluate raw and reconstructed input fields for all panels."""
    scale_data = {key: [] for key in data_types}
    cached_data = {}
    evaluators = {}
    for timestep in timesteps:
        time_value = timestep * input_dt
        for data_type in data_types:
            details = _INPUT_DATA_DETAILS[data_type]
            series_key = _INPUT_TIMESERIES_KEY[data_type]
            target_lon, target_lat, target_shape = _target_grid(grids, details["grid_type"])
            raw = _read_raw_input_field(
                h5file, data_type, details, timestep, target_shape
            )

            projected = np.full(target_shape, np.nan)
            if series_key in input_series.datasets:
                field_space = input_series.get_field_space(series_key)
                evaluator_key = (series_key, details["grid_type"])
                if evaluator_key not in evaluators:
                    evaluators[evaluator_key] = SphericalTransform(
                        field_space.representation, Grid(lat=target_lat, lon=target_lon)
                    )
                try:
                    projected_components = evaluate_projected_input(
                        input_series, series_key, time_value, transform=evaluators[evaluator_key]
                    )
                except ValueError:
                    projected_components = {}
                projected = projected_components.get(
                    _PROJECTED_COMPONENT_KEY[data_type], projected
                ).reshape(target_shape)

            scale_data[data_type].extend((raw.reshape(-1), projected.reshape(-1)))
            cached_data[(timestep, data_type)] = {"input": raw, "projected": projected}
    return cached_data, scale_data


def _comparison_color_scales(
    scale_data, data_types, *, vmin_percentile, vmax_percentile, strictly_positive_scale_type
):
    """Build and report one shared raw/projected scale per field."""
    scales = {}
    for data_type in data_types:
        details = _INPUT_DATA_DETAILS[data_type]
        strictly_positive = details["strictly_positive"]
        scale_type = strictly_positive_scale_type if strictly_positive else "linear"
        scales[data_type] = build_percentile_color_scale(
            scale_data[data_type],
            strictly_positive=strictly_positive,
            vmin_percentile=vmin_percentile,
            vmax_percentile=vmax_percentile,
            scale_type=scale_type,
            label=data_type,
        )
        scale = scales[data_type]
        print(
            f"  Global scale for '{data_type}' ({scale_type}, "
            f"strictly_positive={strictly_positive}): "
            f"vmin={scale['vmin']:.3e}, vmax={scale['vmax']:.3e}, "
            f"cmap='{scale['cmap']}'"
        )
    return scales


def _comparison_figure_size(n_fields, n_times, time_fraction, label_fraction):
    """Return a bounded figure size for the requested panel grid."""
    n_rows = 2 * n_fields
    map_cell_width = 2.2
    map_cell_height = 1.8
    estimated_width = n_times * map_cell_width / (1.0 - label_fraction)
    estimated_height = n_rows * map_cell_height / (1.0 - time_fraction)
    width = min(max(8.0, estimated_width), 30.0)
    minimum_height = max(n_rows * 0.75 + 1.5, 5.0)
    maximum_height = n_rows * (map_cell_height + 0.3) + 2.0
    height = min(max(minimum_height, estimated_height), maximum_height)
    print(f"User fractions: TimeRow={time_fraction:.2f}, CbarLabelCol={label_fraction:.2f}")
    print(
        f'Target map cell: {map_cell_width}"x{map_cell_height}". '
        f'"Est. Total Fig: {estimated_width:.1f}"x{estimated_height:.1f}". '
        f'Final Fig: {width:.1f}"x{height:.1f}"'
    )
    return width, height


def _create_comparison_layout(
    data_types, timesteps, input_dt, coordinate_contexts, time_fraction, label_fraction
):
    """Create the projection-comparison axes and label column."""
    n_fields = len(data_types)
    n_rows = 2 * n_fields
    n_columns = len(timesteps)
    figure_size = _comparison_figure_size(n_fields, n_columns, time_fraction, label_fraction)
    fig = plt.figure(figsize=figure_size, layout="constrained")
    subfigures = fig.subfigures(
        2,
        2,
        height_ratios=[time_fraction, 1.0 - time_fraction],
        width_ratios=[label_fraction, 1.0 - label_fraction],
        hspace=0.01,
        wspace=0.01,
    )
    empty_subfigure = subfigures[0, 0]
    time_subfigure = subfigures[0, 1]
    label_subfigure = subfigures[1, 0]
    plot_subfigure = subfigures[1, 1]
    empty_subfigure.patch.set_alpha(0.0)
    for axis in empty_subfigure.get_axes():
        axis.remove()

    time_axes = np.atleast_1d(time_subfigure.subplots(1, n_columns, sharey=True))
    for axis, timestep in zip(time_axes, timesteps, strict=True):
        axis.text(0.5, 0.5, f"{timestep * input_dt}s", ha="center", va="center", fontsize=9)
        axis.axis("off")

    map_grid = plot_subfigure.add_gridspec(n_rows, n_columns, hspace=0.05, wspace=0.03)
    map_axes = np.empty((n_rows, n_columns), dtype=object)
    for row_index in range(n_rows):
        for column_index, context in enumerate(coordinate_contexts):
            map_axes[row_index, column_index] = plot_subfigure.add_subplot(
                map_grid[row_index, column_index], projection=context.projection()
            )
    label_grid = gridspec.GridSpec(
        n_fields,
        3,
        figure=label_subfigure,
        width_ratios=[0.30, 0.45, 0.25],
        hspace=0.15,
        wspace=0.05,
    )
    return fig, label_subfigure, label_grid, map_axes


def _add_field_labels(label_subfigure, label_grid, field_index, field_label):
    """Add field, colorbar, and row-label axes."""
    field_axis = label_subfigure.add_subplot(label_grid[field_index, 0])
    field_axis.text(0.5, 0.5, field_label, ha="center", va="center", rotation=90, fontsize=9)
    field_axis.axis("off")
    colorbar_axis = label_subfigure.add_subplot(label_grid[field_index, 1])
    row_labels = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=label_grid[field_index, 2], height_ratios=[0.5, 0.5], hspace=0.01
    )
    for row_index, label in enumerate(("Input", "Projected")):
        axis = label_subfigure.add_subplot(row_labels[row_index, 0])
        axis.text(0.5, 0.5, label, ha="center", va="center", rotation=90, fontsize=8)
        axis.axis("off")
    return colorbar_axis


def _draw_comparison_panels(
    fig,
    label_subfigure,
    label_grid,
    map_axes,
    data_types,
    timesteps,
    grids,
    cached_data,
    color_scales,
    coordinate_contexts,
):
    """Draw all raw/projected map pairs and their colorbars."""
    n_rows = 2 * len(data_types)
    for field_index, data_type in enumerate(data_types):
        details = _INPUT_DATA_DETAILS[data_type]
        scale = color_scales[data_type]
        colorbar_axis = _add_field_labels(
            label_subfigure, label_grid, field_index, details["label"]
        )
        longitude, latitude, _ = _target_grid(grids, details["grid_type"])
        first_mappable = None
        for time_index, timestep in enumerate(timesteps):
            coordinate_context = coordinate_contexts[time_index]
            cache_key = (timestep, data_type)
            try:
                fields = cached_data[cache_key]
            except KeyError as exc:
                raise RuntimeError(
                    f"Projection-comparison data for {cache_key} was not prepared."
                ) from exc

            for row_offset, source in enumerate(("input", "projected")):
                row_index = 2 * field_index + row_offset
                axis = map_axes[row_index, time_index]
                axis.clear()
                values = fields[source]
                mappable = plot_scalar_map_on_ax(
                    axis,
                    longitude,
                    latitude,
                    values,
                    title="",
                    cmap=scale["cmap"],
                    norm=scale["norm"],
                    coordinate_context=coordinate_context,
                    left_labels=time_index == 0,
                    bottom_labels=row_index == n_rows - 1,
                )
                if first_mappable is None and not np.all(np.isnan(values)):
                    first_mappable = mappable

        if first_mappable is not None:
            colorbar = fig.colorbar(first_mappable, cax=colorbar_axis, orientation="vertical")
            colorbar.ax.tick_params(labelsize=7)
        else:
            colorbar_axis.text(
                0.5,
                0.5,
                "No Valid Data\nfor Colorbar",
                ha="center",
                va="center",
                fontsize=7,
                wrap=True,
            )
            colorbar_axis.axis("off")


def _render_input_projection_comparison(
    h5file,
    projected_run_directory,
    timesteps_to_plot,
    input_dt,
    data_types_to_plot,
    output_filename=None,
    vmin_percentile=0.2,
    vmax_percentile=99.8,
    strictly_positive_scale_type="linear",
    time_row_h_frac_user=0.015,
    cbars_labels_col_w_frac_user=0.11,
):
    """Compare raw and projected inputs on their fixed GEO grid."""
    if not _validate_comparison_request(
        h5file, timesteps_to_plot, data_types_to_plot, strictly_positive_scale_type
    ):
        return

    run_view = SavedRunView.from_directory(projected_run_directory)
    input_series = run_view.load_input_series()
    grids = _comparison_grids(h5file)
    coordinate_contexts = []
    for timestep in timesteps_to_plot:
        timestamp = h5file["time"][timestep]
        if isinstance(timestamp, bytes):
            timestamp = timestamp.decode("ascii")
        event_time = dt.datetime.fromisoformat(str(timestamp))
        coordinate_contexts.append(MapCoordinateContext.geographic(event_time))

    print("Collecting raw and projected input fields for global color scales...")
    cached_plot_data, scale_data = _collect_comparison_data(
        h5file, input_series, timesteps_to_plot, input_dt, data_types_to_plot, grids
    )

    print("Calculating projection-comparison color scales from percentiles...")
    global_plot_scales = _comparison_color_scales(
        scale_data,
        data_types_to_plot,
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
        strictly_positive_scale_type=strictly_positive_scale_type,
    )

    fig, label_subfigure, label_grid, map_axes = _create_comparison_layout(
        data_types_to_plot,
        timesteps_to_plot,
        input_dt,
        coordinate_contexts,
        time_row_h_frac_user,
        cbars_labels_col_w_frac_user,
    )
    _draw_comparison_panels(
        fig,
        label_subfigure,
        label_grid,
        map_axes,
        data_types_to_plot,
        timesteps_to_plot,
        grids,
        cached_plot_data,
        global_plot_scales,
        coordinate_contexts,
    )

    if output_filename:
        fig.savefig(output_filename, dpi=200)
        print(f"Figure saved to {output_filename}")
    else:
        plt.show()


def plot_input_projection_comparison(
    h5_filepath,
    projected_run_directory,
    timesteps_to_plot,
    input_dt,
    data_types_to_plot,
    output_filename=None,
    vmin_percentile=0.2,
    vmax_percentile=99.8,
    strictly_positive_scale_type="linear",
    time_row_h_frac_user=0.015,
    cbars_labels_col_w_frac_user=0.11,
):
    """Compare raw and projected inputs on their fixed GEO grid."""
    try:
        h5file = h5.File(h5_filepath, "r")
    except Exception as exc:
        raise ValueError(f"Failed to open HDF5 file '{h5_filepath}': {exc}") from exc

    with h5file:
        return _render_input_projection_comparison(
            h5file,
            projected_run_directory,
            timesteps_to_plot,
            input_dt,
            data_types_to_plot,
            output_filename=output_filename,
            vmin_percentile=vmin_percentile,
            vmax_percentile=vmax_percentile,
            strictly_positive_scale_type=strictly_positive_scale_type,
            time_row_h_frac_user=time_row_h_frac_user,
            cbars_labels_col_w_frac_user=cbars_labels_col_w_frac_user,
        )


__all__ = ["plot_input_projection_comparison", "plot_scalar_map_on_ax"]
