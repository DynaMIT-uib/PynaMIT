"""Compare raw driver inputs with their projected reconstruction."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py as h5
import cartopy.crs as ccrs

from pynamit.sphere import Grid
from pynamit.sphere.spherical_transform import SphericalTransform
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.coordinates import (
    local_time_longitude_to_geographic as _local_time_longitude_to_geographic,
)
from pynamit.visualization.input_projection import evaluate_projected_input
from pynamit.visualization.plot_helpers import (
    build_percentile_color_scale,
    style_global_input_axis,
)
from pynamit.visualization.saved_run import SavedRunView


_INPUT_TIMESERIES_KEY = {
    "Br": "Br",
    "jr": "jr",
    "SH": "conductance",
    "SP": "conductance",
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
        "h5_key_primary": "Bu",
    },
    "jr": {
        "label": r"$j_r$ [A/m$^2$]",
        "strictly_positive": False,
        "grid_type": "ionosphere",
        "h5_key_primary": "FAC",
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
        "h5_key_primary": "We",
        "h5_key_secondary": "Wn",
    },
    "u_theta": {
        "label": r"$u_\theta$ (South) [m/s]",
        "strictly_positive": False,
        "grid_type": "ionosphere",
        "h5_key_primary": "We",
        "h5_key_secondary": "Wn",
    },
    "u_phi": {
        "label": r"$u_\phi$ (East) [m/s]",
        "strictly_positive": False,
        "grid_type": "ionosphere",
        "h5_key_primary": "We",
        "h5_key_secondary": "Wn",
    },
}


def _read_raw_input_field(h5file, data_type, details, timestep, ionosphere_br_2d, shape):
    if data_type == "Br":
        return h5file[details["h5_key_primary"]][timestep, :, :] * 1e-9
    if data_type == "jr":
        return h5file[details["h5_key_primary"]][timestep, :, :] * 1e-6 * ionosphere_br_2d
    if data_type in {"SH", "SP"}:
        return h5file[details["h5_key_primary"]][timestep, :, :]
    if data_type in {"u_mag", "u_theta", "u_phi"}:
        u_e_h5 = h5file[details["h5_key_primary"]][timestep, :, :]
        u_n_h5 = h5file[details["h5_key_secondary"]][timestep, :, :]
        if data_type == "u_mag":
            return np.sqrt(u_n_h5**2 + u_e_h5**2)
        if data_type == "u_theta":
            return -u_n_h5
        return u_e_h5
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


def plot_input_projection_comparison(
    h5_filepath,
    projected_run_directory,
    timesteps_to_plot,
    input_dt,
    data_types_to_plot,
    noon_longitude=0,
    output_filename=None,
    vmin_percentile=0.2,
    vmax_percentile=99.8,
    strictly_positive_scale_type="linear",
    time_row_h_frac_user=0.015,
    cbars_labels_col_w_frac_user=0.11,
    magnetosphere_local_noon_longitude=None,
    coordinate_context=None,
):
    """Compare raw HDF5 inputs with projected reconstruction."""
    try:
        h5file = h5.File(h5_filepath, "r")
    except Exception as e:
        raise ValueError(f"Failed to open HDF5 file '{h5_filepath}': {e}")

    num_h5_steps = h5file["time"].shape[0]

    if not timesteps_to_plot:
        h5file.close()
        print("Warning: No timesteps specified for plotting. Exiting.")
        return
    invalid_timesteps = [t for t in timesteps_to_plot if not (0 <= t < num_h5_steps)]
    if invalid_timesteps:
        h5file.close()
        raise ValueError(
            f"Invalid timesteps provided: {invalid_timesteps}. "
            f"All timesteps must be within the range [0, {num_h5_steps - 1}]."
        )
    if not data_types_to_plot:
        h5file.close()
        print("Warning: No data types specified for plotting. Exiting.")
        return
    unknown_data_types = sorted(set(data_types_to_plot) - set(_INPUT_DATA_DETAILS))
    if unknown_data_types:
        h5file.close()
        raise ValueError(
            f"Unsupported data types for projection comparison: {unknown_data_types}. "
            f"Supported values are {sorted(_INPUT_DATA_DETAILS)}."
        )
    if strictly_positive_scale_type not in ["linear", "log"]:
        h5file.close()
        raise ValueError("strictly_positive_scale_type must be 'linear' or 'log'.")

    try:
        run_view = SavedRunView.from_directory(projected_run_directory)
    except ValueError:
        h5file.close()
        raise

    config = run_view.config
    ri_value = float(config.RI)
    mainfield = run_view.mainfield
    input_timeseries = run_view.load_input_timeseries()

    ionosphere_lat, ionosphere_lon = h5file["glat"][:], h5file["glon"][:]
    magnetosphere_lat, magnetosphere_lon = h5file["Blat"][:], h5file["Blon"][:]
    if coordinate_context is not None:
        noon_longitude = coordinate_context.noon_longitude
    if magnetosphere_local_noon_longitude is not None:
        if coordinate_context is None:
            magnetosphere_lon = _local_time_longitude_to_geographic(
                magnetosphere_lon,
                noon_longitude=noon_longitude,
                local_noon_longitude=magnetosphere_local_noon_longitude,
            )
        else:
            magnetosphere_lon = coordinate_context.local_time_longitude_to_coordinate(
                magnetosphere_lon, local_noon_longitude=magnetosphere_local_noon_longitude
            )
    ionosphere_grid = Grid(lat=ionosphere_lat, lon=ionosphere_lon)
    ionosphere_b_evaluator = FieldEvaluator(mainfield, ionosphere_grid, ri_value)
    ionosphere_br_2d = ionosphere_b_evaluator.br.reshape(ionosphere_lat.shape)

    print("Collecting raw and projected input fields for global color scales...")
    all_data_for_scaling = {
        dt_str: {"input": [], "projected": []} for dt_str in data_types_to_plot
    }
    cached_plot_data = {}
    plot_evaluators = {}

    for timestep in timesteps_to_plot:
        time_val = timestep * input_dt
        for data_type_str in data_types_to_plot:
            details = _INPUT_DATA_DETAILS[data_type_str]
            pynamit_ts_key = _INPUT_TIMESERIES_KEY[data_type_str]
            is_magnetosphere_grid = details["grid_type"] == "magnetosphere"
            target_lon, target_lat, target_shape = (
                (magnetosphere_lon, magnetosphere_lat, magnetosphere_lat.shape)
                if is_magnetosphere_grid
                else (ionosphere_lon, ionosphere_lat, ionosphere_lat.shape)
            )
            calculated_input_data_2d = _read_raw_input_field(
                h5file, data_type_str, details, timestep, ionosphere_br_2d, target_shape
            )
            all_data_for_scaling[data_type_str]["input"].append(
                calculated_input_data_2d.reshape(-1)
            )

            calculated_projected_data_2d = np.full(target_shape, np.nan)
            if pynamit_ts_key in input_timeseries.datasets:
                field_space = input_timeseries.get_field_space(pynamit_ts_key)
                evaluator_key = (pynamit_ts_key, details["grid_type"])
                if evaluator_key not in plot_evaluators:
                    plot_evaluators[evaluator_key] = SphericalTransform(
                        field_space.representation, Grid(lat=target_lat, lon=target_lon)
                    )
                current_plot_evaluator = plot_evaluators[evaluator_key]
                try:
                    projected_input = evaluate_projected_input(
                        input_timeseries,
                        pynamit_ts_key,
                        time_val,
                        transform=current_plot_evaluator,
                    )
                except ValueError:
                    projected_input = {}
                calculated_projected_data_2d = projected_input.get(
                    _PROJECTED_COMPONENT_KEY[data_type_str], calculated_projected_data_2d
                ).reshape(target_shape)
            all_data_for_scaling[data_type_str]["projected"].append(
                calculated_projected_data_2d.reshape(-1)
            )
            cached_plot_data[(timestep, data_type_str)] = {
                "input": calculated_input_data_2d,
                "projected": calculated_projected_data_2d,
            }

    print("Calculating projection-comparison color scales from percentiles...")
    global_plot_scales = {}
    for data_type_str in data_types_to_plot:
        details = _INPUT_DATA_DETAILS[data_type_str]
        is_strictly_positive = details["strictly_positive"]
        current_scale_type = strictly_positive_scale_type if is_strictly_positive else "linear"
        scale_arrays = (
            all_data_for_scaling[data_type_str]["input"]
            + all_data_for_scaling[data_type_str]["projected"]
        )
        try:
            global_plot_scales[data_type_str] = build_percentile_color_scale(
                scale_arrays,
                strictly_positive=is_strictly_positive,
                vmin_percentile=vmin_percentile,
                vmax_percentile=vmax_percentile,
                scale_type=current_scale_type,
                label=data_type_str,
            )
        except ValueError:
            h5file.close()
            raise
        current_scale = global_plot_scales[data_type_str]
        print(
            f"  Global scale for '{data_type_str}' ({current_scale_type}, "
            f"strictly_positive={is_strictly_positive}): "
            f"vmin={current_scale['vmin']:.3e}, "
            f"vmax={current_scale['vmax']:.3e}, cmap='{current_scale['cmap']}'"
        )

    num_dt = len(data_types_to_plot)
    num_plot_rows_maps = num_dt * 2
    num_plot_cols_maps = len(timesteps_to_plot)

    ref_map_cell_w = 2.2
    ref_map_cell_h = 1.8
    target_plots_area_w = num_plot_cols_maps * ref_map_cell_w
    target_plots_area_h = num_plot_rows_maps * ref_map_cell_h

    plots_row_h_frac_user = 1.0 - time_row_h_frac_user
    plots_col_w_frac_user = 1.0 - cbars_labels_col_w_frac_user

    est_total_fig_width = target_plots_area_w / plots_col_w_frac_user
    est_total_fig_height = target_plots_area_h / plots_row_h_frac_user

    fig_width = min(max(8.0, est_total_fig_width), 30.0)
    min_practical_fig_height = num_dt * 2 * 0.75 + 1.5
    fig_height_lower_bound = max(min_practical_fig_height, 5.0)
    fig_height_upper_bound = num_plot_rows_maps * (ref_map_cell_h + 0.3) + 2.0
    fig_height = min(max(fig_height_lower_bound, est_total_fig_height), fig_height_upper_bound)

    print(
        f"User fractions: TimeRow={time_row_h_frac_user:.2f}, "
        f"CbarLabelCol={cbars_labels_col_w_frac_user:.2f}"
    )
    print(
        f'Target map cell: {ref_map_cell_w}"x{ref_map_cell_h}". '
        f'"Est. Total Fig: {est_total_fig_width:.1f}"x{est_total_fig_height:.1f}". '
        f'Final Fig: {fig_width:.1f}"x{fig_height:.1f}"'
    )

    fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")

    sfigs_grid = fig.subfigures(
        2,
        2,
        height_ratios=[time_row_h_frac_user, plots_row_h_frac_user],
        width_ratios=[cbars_labels_col_w_frac_user, plots_col_w_frac_user],
        hspace=0.01,
        wspace=0.01,
    )
    sfig_TL_empty = sfigs_grid[0, 0]
    sfig_TR_times = sfigs_grid[0, 1]
    sfig_BL_cbars_and_labels = sfigs_grid[1, 0]
    sfig_BR_plots = sfigs_grid[1, 1]

    sfig_TL_empty.patch.set_alpha(0.0)
    [ax.remove() for ax in sfig_TL_empty.get_axes()]

    if num_plot_cols_maps > 0:
        time_label_axes_list = sfig_TR_times.subplots(1, num_plot_cols_maps, sharey=True)
        time_label_axes = (
            [time_label_axes_list] if num_plot_cols_maps == 1 else time_label_axes_list
        )
        for ts_idx, timestep in enumerate(timesteps_to_plot):
            time_label_axes[ts_idx].text(
                0.5, 0.5, f"{timestep * input_dt}s", ha="center", va="center", fontsize=9
            )
            time_label_axes[ts_idx].axis("off")

    map_axes_flat = sfig_BR_plots.subplots(
        num_plot_rows_maps,
        num_plot_cols_maps,
        sharex=True,
        sharey=True,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=noon_longitude)},
    )
    if num_plot_rows_maps == 1 and num_plot_cols_maps == 1:
        map_axes = np.array([[map_axes_flat]])
    elif num_plot_rows_maps == 1:
        map_axes = map_axes_flat[np.newaxis, :]
    elif num_plot_cols_maps == 1:
        map_axes = map_axes_flat[:, np.newaxis]
    else:
        map_axes = map_axes_flat

    coordinate_labels_enabled = coordinate_context is not None

    gs_main_left_panel = gridspec.GridSpec(
        num_dt,
        3,
        figure=sfig_BL_cbars_and_labels,
        width_ratios=[0.30, 0.45, 0.25],
        hspace=0.15,
        wspace=0.05,
    )

    for dt_idx, data_type_str in enumerate(data_types_to_plot):
        details = _INPUT_DATA_DETAILS[data_type_str]
        current_global_scale = global_plot_scales[data_type_str]
        cmap_use, norm_use = current_global_scale["cmap"], current_global_scale["norm"]

        row_idx_input_map = dt_idx * 2
        row_idx_projected_map = row_idx_input_map + 1
        current_mappable_this_dt = None

        ax_dt_label = sfig_BL_cbars_and_labels.add_subplot(gs_main_left_panel[dt_idx, 0])
        ax_dt_label.text(
            0.5, 0.5, details["label"], ha="center", va="center", rotation=90, fontsize=9
        )
        ax_dt_label.axis("off")

        ax_cbar_placeholder = sfig_BL_cbars_and_labels.add_subplot(gs_main_left_panel[dt_idx, 1])

        gs_input_projected_text_block = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_main_left_panel[dt_idx, 2], height_ratios=[0.5, 0.5], hspace=0.01
        )

        ax_input_text = sfig_BL_cbars_and_labels.add_subplot(gs_input_projected_text_block[0, 0])
        ax_input_text.text(0.5, 0.5, "Input", ha="center", va="center", rotation=90, fontsize=8)
        ax_input_text.axis("off")

        ax_projected_text = sfig_BL_cbars_and_labels.add_subplot(
            gs_input_projected_text_block[1, 0]
        )
        ax_projected_text.text(
            0.5, 0.5, "Projected", ha="center", va="center", rotation=90, fontsize=8
        )
        ax_projected_text.axis("off")

        for ts_idx, timestep in enumerate(timesteps_to_plot):
            ax_input, ax_projected = (
                map_axes[row_idx_input_map, ts_idx],
                map_axes[row_idx_projected_map, ts_idx],
            )
            ax_input.clear()
            ax_projected.clear()
            is_magnetosphere_grid_plot = details["grid_type"] == "magnetosphere"
            current_lon_plot, current_lat_plot = (
                (magnetosphere_lon, magnetosphere_lat)
                if is_magnetosphere_grid_plot
                else (ionosphere_lon, ionosphere_lat)
            )
            cache_key = (timestep, data_type_str)
            try:
                cached_data = cached_plot_data[cache_key]
                retrieved_input_data, retrieved_projected_data = (
                    cached_data["input"],
                    cached_data["projected"],
                )
            except KeyError:
                h5file.close()
                raise RuntimeError(f"Projection-comparison data for {cache_key} was not prepared.")
            im_input = plot_scalar_map_on_ax(
                ax_input,
                current_lon_plot,
                current_lat_plot,
                retrieved_input_data,
                title="",
                cmap=cmap_use,
                norm=norm_use,
                coordinate_context=coordinate_context if coordinate_labels_enabled else None,
                left_labels=coordinate_labels_enabled and ts_idx == 0,
                bottom_labels=(
                    coordinate_labels_enabled and row_idx_input_map == num_plot_rows_maps - 1
                ),
            )
            if current_mappable_this_dt is None and not np.all(np.isnan(retrieved_input_data)):
                current_mappable_this_dt = im_input
            im_projected = plot_scalar_map_on_ax(
                ax_projected,
                current_lon_plot,
                current_lat_plot,
                retrieved_projected_data,
                title="",
                cmap=cmap_use,
                norm=norm_use,
                coordinate_context=coordinate_context if coordinate_labels_enabled else None,
                left_labels=coordinate_labels_enabled and ts_idx == 0,
                bottom_labels=(
                    coordinate_labels_enabled and row_idx_projected_map == num_plot_rows_maps - 1
                ),
            )
            if current_mappable_this_dt is None and not np.all(np.isnan(retrieved_projected_data)):
                current_mappable_this_dt = im_projected

        if current_mappable_this_dt:
            cb = fig.colorbar(
                current_mappable_this_dt, cax=ax_cbar_placeholder, orientation="vertical"
            )
            cb.ax.tick_params(labelsize=7)
        else:
            ax_cbar_placeholder.text(
                0.5,
                0.5,
                "No Valid Data\nfor Colorbar",
                ha="center",
                va="center",
                fontsize=7,
                wrap=True,
            )
            ax_cbar_placeholder.axis("off")

    if output_filename:
        plt.savefig(output_filename, dpi=200)
        print(f"Figure saved to {output_filename}")
    else:
        plt.show()
    h5file.close()


__all__ = ["plot_input_projection_comparison", "plot_scalar_map_on_ax"]
