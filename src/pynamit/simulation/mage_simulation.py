import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import h5py as h5
import cartopy.crs as ccrs

# Assuming pynamit imports are correct
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.primitives.io import IO
from pynamit.primitives.timeseries import Timeseries
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.simulation.mainfield import Mainfield
from pynamit.math.constants import RE


# --- MODIFIED Helper function for evaluating scalar coefficients ---
def _evaluate_scalar_coeffs_to_grid(  # Renamed and signature changed
    coeffs, storage_basis, plot_evaluator, target_shape
):
    if coeffs is None:  # Coefficients might not exist for a given time
        return np.full(target_shape, np.nan)
    field_exp = FieldExpansion(storage_basis, coeffs=coeffs, field_type="scalar")
    return field_exp.to_grid(plot_evaluator).reshape(target_shape)


# --- NEW Helper function for evaluating tangential coefficients ---
def _evaluate_tangential_coeffs_to_grid_components(  # New function
    coeffs, storage_basis, plot_evaluator, target_shape
):
    if coeffs is None:  # Coefficients might not exist
        return np.full(target_shape, np.nan), np.full(target_shape, np.nan)
    # Ensure coeffs are (2, N_coeffs) for tangential field
    # pynamit Timeseries.get_entry usually returns them in the correct shape for FieldExpansion
    field_exp = FieldExpansion(
        storage_basis, coeffs=coeffs.reshape((2, -1)), field_type="tangential"
    )
    field_grid_components = field_exp.to_grid(plot_evaluator)  # Returns (theta, phi) components
    field_t_2d = field_grid_components[0].reshape(target_shape)
    field_p_2d = field_grid_components[1].reshape(target_shape)
    return field_t_2d, field_p_2d


def plot_scalar_map_on_ax(
    ax, lon_coords_2d, lat_coords_2d, data_2d_arr, title="", cmap="viridis", norm=None
):
    if norm is None:
        raise ValueError("Norm object must be provided to plot_scalar_map_on_ax.")

    ax.coastlines(color="grey", zorder=3, linewidth=0.5)
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
    ax.set_title(title, fontsize=9)
    return im


def plot_input_vs_interpolated(
    h5_filepath,
    interpolated_filename_prefix,
    timesteps_to_plot,
    input_dt,
    data_types_to_plot,
    noon_longitude=0,
    output_filename=None,
    vmin_percentile=0.2,
    vmax_percentile=99.8,
    positive_definite_zeromin=True,
    non_bwr_scale_type="linear",
):
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
            f"All timesteps must be within the range [0, {num_h5_steps - 1})."
        )
    if not data_types_to_plot:
        h5file.close()
        print("Warning: No data types specified for plotting. Exiting.")
        return

    io = IO(interpolated_filename_prefix)
    settings = io.load_dataset("settings", print_info=False)
    if settings is None:
        h5file.close()
        raise ValueError("Settings dataset not found in the HDF5 file.")

    ri_value = float(settings.RI)
    mainfield = Mainfield(
        kind=str(settings.mainfield_kind),
        epoch=int(settings.mainfield_epoch),
        hI=(ri_value - RE) * 1e-3,
        B0=None if float(settings.mainfield_B0) == 0 else float(settings.mainfield_B0),
    )

    cs_basis = CSBasis(int(settings.Ncs))
    sh_basis = SHBasis(int(settings.Nmax), int(settings.Mmax), Nmin=0)
    sh_basis_zero_removed = SHBasis(int(settings.Nmax), int(settings.Mmax))

    input_vars_pynamit = {
        "jr": {"jr": "scalar"},
        "Br": {"Br": "scalar"},
        "conductance": {"etaP": "scalar", "etaH": "scalar"},
        "u": {"u": "tangential"},
    }
    input_storage_bases = {
        "jr": sh_basis_zero_removed,
        "Br": sh_basis_zero_removed,
        "conductance": sh_basis,
        "u": sh_basis_zero_removed,
    }
    pynamit_timeseries_key_map = {  # Maps plot data type to timeseries entry key
        "Br": "Br",
        "jr": "jr",
        "SH": "conductance",
        "SP": "conductance",
        "u_mag": "u",
        "u_theta": "u",
        "u_phi": "u",
    }
    data_type_details = {
        "Br": {
            "label": r"$\Delta B_r$ [T]",
            "cmap": "bwr",
            "grid_type": "magnetosphere",
            "h5_key_primary": "Bu",
            "scale_type": "linear",
        },
        "jr": {
            "label": r"$j_r$ [A/m$^2$]",
            "cmap": "bwr",
            "grid_type": "ionosphere",
            "h5_key_primary": "FAC",
            "scale_type": "linear",
        },
        "SH": {
            "label": r"$\Sigma_H$ [S]",
            "cmap": "viridis",
            "grid_type": "ionosphere",
            "h5_key_primary": "SH",
            "scale_type": non_bwr_scale_type,
        },
        "SP": {
            "label": r"$\Sigma_P$ [S]",
            "cmap": "viridis",
            "grid_type": "ionosphere",
            "h5_key_primary": "SP",
            "scale_type": non_bwr_scale_type,
        },
        "u_mag": {
            "label": r"$|u|$ [m/s]",
            "cmap": "viridis",
            "grid_type": "ionosphere",
            "h5_key_primary": "We",
            "h5_key_secondary": "Wn",
            "scale_type": non_bwr_scale_type,
        },
        "u_theta": {
            "label": r"$u_\theta$ (South) [m/s]",
            "cmap": "bwr",
            "grid_type": "ionosphere",
            "h5_key_primary": "We",
            "h5_key_secondary": "Wn",
            "scale_type": "linear",
        },
        "u_phi": {
            "label": r"$u_\phi$ (East) [m/s]",
            "cmap": "bwr",
            "grid_type": "ionosphere",
            "h5_key_primary": "We",
            "h5_key_secondary": "Wn",
            "scale_type": "linear",
        },
    }

    input_timeseries = Timeseries(cs_basis, input_storage_bases, input_vars_pynamit)
    input_timeseries.load_all(io)

    ionosphere_lat, ionosphere_lon = h5file["glat"][:], h5file["glon"][:]
    magnetosphere_lat, magnetosphere_lon = h5file["Blat"][:], h5file["Blon"][:]
    ionosphere_grid = Grid(lat=ionosphere_lat, lon=ionosphere_lon)
    ionosphere_b_evaluator = FieldEvaluator(mainfield, ionosphere_grid, ri_value)
    ionosphere_br_2d = ionosphere_b_evaluator.br.reshape(ionosphere_lat.shape)

    print("Starting Pass 1: Collecting and Caching data for global vmin/vmax...")
    all_data_for_scaling = {
        dt_str: {"input": [], "interpolated": []} for dt_str in data_types_to_plot
    }
    cached_plot_data = {}
    plot_evaluators = {}

    for timestep in timesteps_to_plot:
        time_val = timestep * input_dt
        for data_type_str in data_types_to_plot:
            details = data_type_details[data_type_str]
            pynamit_ts_key = pynamit_timeseries_key_map[data_type_str]

            is_magnetosphere_grid = details["grid_type"] == "magnetosphere"
            current_lon_coords_pass1, current_lat_coords_pass1, target_shape_pass1 = (
                (magnetosphere_lon, magnetosphere_lat, magnetosphere_lat.shape)
                if is_magnetosphere_grid
                else (ionosphere_lon, ionosphere_lat, ionosphere_lat.shape)
            )

            # --- Calculate input_data_2d (from HDF5) ---
            calculated_input_data_2d = np.full(target_shape_pass1, np.nan)
            if data_type_str == "Br":
                calculated_input_data_2d = h5file[details["h5_key_primary"]][timestep, :, :] * 1e-9
            elif data_type_str == "jr":
                calculated_input_data_2d = (
                    h5file[details["h5_key_primary"]][timestep, :, :] * 1e-6
                ) * ionosphere_br_2d
            elif data_type_str in ["SH", "SP"]:
                calculated_input_data_2d = h5file[details["h5_key_primary"]][timestep, :, :]
            elif data_type_str in ["u_mag", "u_theta", "u_phi"]:
                u_e_h5, u_n_h5 = (
                    h5file[details["h5_key_primary"]][timestep, :, :],
                    h5file[details["h5_key_secondary"]][timestep, :, :],
                )
                if data_type_str == "u_mag":
                    calculated_input_data_2d = np.sqrt(u_n_h5**2 + u_e_h5**2)
                elif data_type_str == "u_theta":
                    calculated_input_data_2d = -u_n_h5
                elif data_type_str == "u_phi":
                    calculated_input_data_2d = u_e_h5
            all_data_for_scaling[data_type_str]["input"].append(calculated_input_data_2d.ravel())

            # --- Calculate interpolated_data_2d (from PynaMIT timeseries) ---
            calculated_interpolated_data_2d = np.full(target_shape_pass1, np.nan)
            timeseries_entry = input_timeseries.get_entry(
                pynamit_ts_key, time_val, interpolation=False
            )

            if timeseries_entry:
                storage_basis = input_timeseries.storage_bases[pynamit_ts_key]
                if pynamit_ts_key not in plot_evaluators:
                    plot_evaluators[pynamit_ts_key] = BasisEvaluator(
                        storage_basis,
                        Grid(lat=current_lat_coords_pass1, lon=current_lon_coords_pass1),
                    )
                current_plot_evaluator = plot_evaluators[pynamit_ts_key]

                if pynamit_ts_key == "Br":
                    br_coeffs = timeseries_entry.get("Br")
                    calculated_interpolated_data_2d = _evaluate_scalar_coeffs_to_grid(
                        br_coeffs, storage_basis, current_plot_evaluator, target_shape_pass1
                    )
                elif pynamit_ts_key == "jr":
                    jr_coeffs = timeseries_entry.get("jr")
                    calculated_interpolated_data_2d = _evaluate_scalar_coeffs_to_grid(
                        jr_coeffs, storage_basis, current_plot_evaluator, target_shape_pass1
                    )
                elif pynamit_ts_key == "conductance":
                    etaP_coeffs = timeseries_entry.get("etaP")
                    etaH_coeffs = timeseries_entry.get("etaH")

                    etaP_f = _evaluate_scalar_coeffs_to_grid(
                        etaP_coeffs, storage_basis, current_plot_evaluator, target_shape_pass1
                    )
                    etaH_f = _evaluate_scalar_coeffs_to_grid(
                        etaH_coeffs, storage_basis, current_plot_evaluator, target_shape_pass1
                    )

                    den = etaP_f**2 + etaH_f**2
                    sH_f, sP_f = (
                        np.full_like(etaH_f, np.nan),
                        np.full_like(etaP_f, np.nan),
                    )  # Initialize with NaN
                    valid = den > 1e-12  # Avoid division by zero or tiny numbers
                    if np.any(valid):  # Check if there are any valid points before division
                        sH_f[valid] = etaH_f[valid] / den[valid]
                        sP_f[valid] = etaP_f[valid] / den[valid]

                    if data_type_str == "SH":
                        calculated_interpolated_data_2d = sH_f
                    elif data_type_str == "SP":
                        calculated_interpolated_data_2d = sP_f

                elif pynamit_ts_key == "u":
                    u_coeffs = timeseries_entry.get("u")  # These are tangential coeffs
                    u_t_2d, u_p_2d = _evaluate_tangential_coeffs_to_grid_components(
                        u_coeffs, storage_basis, current_plot_evaluator, target_shape_pass1
                    )
                    if data_type_str == "u_mag":
                        calculated_interpolated_data_2d = np.sqrt(u_t_2d**2 + u_p_2d**2)
                    elif data_type_str == "u_theta":
                        calculated_interpolated_data_2d = u_t_2d
                    elif data_type_str == "u_phi":
                        calculated_interpolated_data_2d = u_p_2d

            all_data_for_scaling[data_type_str]["interpolated"].append(
                calculated_interpolated_data_2d.ravel()
            )
            cache_key = (timestep, data_type_str)
            cached_plot_data[cache_key] = {
                "input": calculated_input_data_2d,
                "interpolated": calculated_interpolated_data_2d,
            }

    print("Calculating global vmin/vmax from percentiles...")
    global_plot_scales = {}
    # --- Global scale calculation ---
    for data_type_str in data_types_to_plot:
        details = data_type_details[data_type_str]
        cmap_global = details["cmap"]
        current_scale_type = details.get("scale_type", "linear")

        cat_input_all = all_data_for_scaling[data_type_str]["input"]
        cat_interp_all = all_data_for_scaling[data_type_str]["interpolated"]

        flat_input = (
            np.concatenate(cat_input_all)
            if cat_input_all and any(a.size > 0 for a in cat_input_all)
            else np.array([])
        )
        flat_interp = (
            np.concatenate(cat_interp_all)
            if cat_interp_all and any(a.size > 0 for a in cat_interp_all)
            else np.array([])
        )

        combined_flat_list = [arr for arr in (flat_input, flat_interp) if arr.size > 0]
        if not combined_flat_list:
            h5file.close()
            raise ValueError(
                f"No data arrays found for '{data_type_str}' to calculate scales after Pass 1."
            )

        combined_flat = np.concatenate(combined_flat_list)
        temp_valid_data = combined_flat[~np.isnan(combined_flat)]

        if temp_valid_data.size == 0:
            h5file.close()
            raise ValueError(
                f"No valid (non-NaN) data found for '{data_type_str}' to calculate scales."
            )

        if current_scale_type == "log":
            valid_data_for_percentile = temp_valid_data[temp_valid_data > 0]
            if valid_data_for_percentile.size == 0:
                h5file.close()
                raise ValueError(
                    f"No positive valid data found for '{data_type_str}', as required for log scale."
                )
        else:
            valid_data_for_percentile = temp_valid_data

        if cmap_global == "bwr":
            abs_max_s = np.percentile(np.abs(valid_data_for_percentile), vmax_percentile)
            vmin, vmax = -abs_max_s, abs_max_s
        else:
            vmin = np.percentile(valid_data_for_percentile, vmin_percentile)
            vmax = np.percentile(valid_data_for_percentile, vmax_percentile)

            if positive_definite_zeromin:
                if vmin < -1e-9:
                    h5file.close()
                    raise ValueError(
                        f"Data type '{data_type_str}' is configured with positive_definite_zeromin=True, "
                        f"but calculated vmin based on percentiles is {vmin:.3e} (negative)."
                    )
                vmin = 0.0

        if vmin == vmax:
            # For LogNorm, this will be caught by LogNorm itself later if vmin <= 0 or vmin >= vmax
            if not (current_scale_type == "log"):
                print(
                    f"Warning: vmin ({vmin:.3e}) and vmax ({vmax:.3e}) are identical for '{data_type_str}'. Plot may show a single color."
                )

        norm_for_plot = None
        if current_scale_type == "log" and cmap_global != "bwr":
            if vmin <= 0:
                h5file.close()
                raise ValueError(
                    f"Log scale for '{data_type_str}' requires vmin > 0, but got vmin={vmin:.3e}."
                )
            try:
                norm_for_plot = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
            except ValueError as e:  # Catches vmin >= vmax from LogNorm
                h5file.close()
                raise ValueError(
                    f"Error creating LogNorm for '{data_type_str}' with vmin={vmin:.3e}, vmax={vmax:.3e}: {e}"
                )
        else:
            norm_for_plot = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        global_plot_scales[data_type_str] = {
            "vmin": vmin,
            "vmax": vmax,
            "cmap": cmap_global,
            "norm": norm_for_plot,
            "scale_type": current_scale_type,
        }
        print(
            f"  Global scale for '{data_type_str}' ({current_scale_type}): vmin={vmin:.3e}, vmax={vmax:.3e}"
        )

    # --- Figure Creation ---
    num_dt = len(data_types_to_plot)
    num_plot_rows = num_dt * 2
    num_plot_cols = len(timesteps_to_plot)

    time_row_h_frac = 0.04
    plots_row_h_frac = 1.0 - time_row_h_frac
    cbars_labels_col_w_frac = 0.20
    plots_col_w_frac = 1.0 - cbars_labels_col_w_frac

    base_plot_w, base_plot_h = 2.0, 1.5
    fig_width_est = (num_plot_cols * base_plot_w) / plots_col_w_frac
    fig_height_est = (num_plot_rows * base_plot_h) / plots_row_h_frac

    fig_width = min(max(8, fig_width_est), 25)
    fig_height = min(
        max(4 + time_row_h_frac * fig_height_est, fig_height_est),
        num_plot_rows * (base_plot_h + 0.15) + 1.0,
    )

    print(f"Creating figure. Target Size: ({fig_width:.1f},{fig_height:.1f})")
    fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
    sfigs_grid = fig.subfigures(
        2,
        2,
        height_ratios=[time_row_h_frac, plots_row_h_frac],
        width_ratios=[cbars_labels_col_w_frac, plots_col_w_frac],
        hspace=0.01,
        wspace=0.01,
    )
    sfigs_grid[0, 0].patch.set_alpha(0.0)
    [ax.remove() for ax in sfigs_grid[0, 0].get_axes()]

    if num_plot_cols > 0:
        time_label_axes_list = sfigs_grid[0, 1].subplots(1, num_plot_cols, sharey=True)
        time_label_axes = [time_label_axes_list] if num_plot_cols == 1 else time_label_axes_list
        for ts_idx, timestep in enumerate(timesteps_to_plot):
            time_label_axes[ts_idx].text(
                0.5, 0.5, f"{timestep * input_dt}s", ha="center", va="center", fontsize=9
            )
            time_label_axes[ts_idx].axis("off")

    map_axes_flat = sfigs_grid[1, 1].subplots(
        num_plot_rows,
        num_plot_cols,
        sharex=True,
        sharey=True,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=noon_longitude)},
    )
    if num_plot_rows == 1 and num_plot_cols == 1:
        map_axes = np.array([[map_axes_flat]])
    elif num_plot_rows == 1:
        map_axes = map_axes_flat[np.newaxis, :]
    elif num_plot_cols == 1:
        map_axes = map_axes_flat[:, np.newaxis]
    else:
        map_axes = map_axes_flat

    gs_bottom_left = gridspec.GridSpec(
        num_plot_rows,
        2,
        figure=sfigs_grid[1, 0],
        width_ratios=[0.8, 0.2],
        wspace=0.05,
        hspace=0.05,
    )

    for dt_idx, data_type_str in enumerate(data_types_to_plot):
        details = data_type_details[data_type_str]
        current_global_scale = global_plot_scales[data_type_str]
        cmap_use = current_global_scale["cmap"]
        norm_use = current_global_scale["norm"]

        row_idx_input = dt_idx * 2
        row_idx_fitted = row_idx_input + 1
        current_mappable_this_dt = None

        gs_dt_cbar_block = gridspec.GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=gs_bottom_left[row_idx_input : row_idx_input + 2, 0],
            width_ratios=[0.4, 0.6],
            wspace=0.1,
        )
        ax_dt_label = sfigs_grid[1, 0].add_subplot(gs_dt_cbar_block[0])
        ax_dt_label.text(
            0.5, 0.5, details["label"], ha="center", va="center", rotation=90, fontsize=9
        )
        ax_dt_label.axis("off")

        ax_input_text_label = sfigs_grid[1, 0].add_subplot(gs_bottom_left[row_idx_input, 1])
        ax_input_text_label.text(
            0.5, 0.5, "Input", ha="center", va="center", rotation=90, fontsize=8
        )
        ax_input_text_label.axis("off")
        ax_fitted_text_label = sfigs_grid[1, 0].add_subplot(gs_bottom_left[row_idx_fitted, 1])
        ax_fitted_text_label.text(
            0.5, 0.5, "Fitted", ha="center", va="center", rotation=90, fontsize=8
        )
        ax_fitted_text_label.axis("off")

        for ts_idx, timestep in enumerate(timesteps_to_plot):
            ax_input = map_axes[row_idx_input, ts_idx]
            ax_fitted = map_axes[row_idx_fitted, ts_idx]
            ax_input.clear()
            ax_fitted.clear()

            is_magnetosphere_grid_plot = details["grid_type"] == "magnetosphere"
            current_lon_plot, current_lat_plot = (
                (magnetosphere_lon, magnetosphere_lat)
                if is_magnetosphere_grid_plot
                else (ionosphere_lon, ionosphere_lat)
            )

            cache_key = (timestep, data_type_str)
            try:
                cached_data = cached_plot_data[cache_key]
                retrieved_input_data = cached_data["input"]
                retrieved_interpolated_data = cached_data["interpolated"]
            except KeyError:
                h5file.close()
                raise RuntimeError(
                    f"Critical: Data for {cache_key} not found in cache. This indicates an internal logic error."
                )

            im_input = plot_scalar_map_on_ax(
                ax_input,
                current_lon_plot,
                current_lat_plot,
                retrieved_input_data,
                title="",
                cmap=cmap_use,
                norm=norm_use,
            )
            if current_mappable_this_dt is None and not np.all(np.isnan(retrieved_input_data)):
                current_mappable_this_dt = im_input

            im_fitted = plot_scalar_map_on_ax(
                ax_fitted,
                current_lon_plot,
                current_lat_plot,
                retrieved_interpolated_data,
                title="",
                cmap=cmap_use,
                norm=norm_use,
            )
            if current_mappable_this_dt is None and not np.all(
                np.isnan(retrieved_interpolated_data)
            ):
                current_mappable_this_dt = im_fitted

        ax_cbar = sfigs_grid[1, 0].add_subplot(gs_dt_cbar_block[1])
        if current_mappable_this_dt:
            cb = fig.colorbar(current_mappable_this_dt, cax=ax_cbar, orientation="vertical")
            cb.ax.tick_params(labelsize=7)
        else:
            ax_cbar.text(
                0.5,
                0.5,
                "No Valid Data\nfor Colorbar",
                ha="center",
                va="center",
                fontsize=7,
                wrap=True,
            )
            ax_cbar.axis("off")

    if output_filename:
        plt.savefig(output_filename, dpi=200)
        print(f"Figure saved to {output_filename}")
    else:
        plt.show()
    h5file.close()
