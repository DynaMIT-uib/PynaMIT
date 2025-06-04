import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
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


# --- Helper functions for calculating interpolated data ---
# ... (These functions remain the same as your last working version)
def _calculate_interpolated_scalar_field(
    timeseries_entry, data_key, storage_basis, plot_evaluator, target_shape
):
    coeffs = timeseries_entry.get(data_key)
    field_exp = FieldExpansion(storage_basis, coeffs=coeffs, field_type="scalar")
    return field_exp.to_grid(plot_evaluator).reshape(target_shape)

def _calculate_interpolated_conductance(
    timeseries_entry, component, storage_basis, plot_evaluator, target_shape
):
    etaP_coeffs, etaH_coeffs = timeseries_entry.get("etaP"), timeseries_entry.get("etaH")

    etaP, etaH = (
        FieldExpansion(storage_basis, coeffs=etaP_coeffs, field_type="scalar"),
        FieldExpansion(storage_basis, coeffs=etaH_coeffs, field_type="scalar"),
    )
    etaP_f, etaH_f = (
        etaP.to_grid(plot_evaluator).reshape(target_shape),
        etaH.to_grid(plot_evaluator).reshape(target_shape),
    )
    den = etaP_f**2 + etaH_f**2
    sH_f, sP_f = np.zeros_like(etaH_f), np.zeros_like(etaP_f)
    valid = den > 1e-12
    sH_f[valid] = etaH_f[valid] / den[valid]
    sP_f[valid] = etaP_f[valid] / den[valid]
    return sH_f if component == "SH" else sP_f

def _calculate_interpolated_u_field(
    timeseries_entry, component, storage_basis, plot_evaluator, target_shape
):
    u_coeffs = timeseries_entry.get("u")
    u = FieldExpansion(
        storage_basis, coeffs=u_coeffs.reshape((2, -1)), field_type="tangential"
    )
    u_grid = u.to_grid(plot_evaluator)
    u_t_2d, u_p_2d = u_grid[0].reshape(target_shape), u_grid[1].reshape(target_shape)
    if component == "u_mag":
        return np.sqrt(u_t_2d**2 + u_p_2d**2)
    elif component == "u_theta":
        return u_t_2d
    elif component == "u_phi":
        return u_p_2d

def plot_scalar_map_on_ax(
    ax,
    lon_coords_2d,
    lat_coords_2d,
    data_2d_arr,
    title="",
    cmap="viridis",
    vmin=None,
    vmax=None,
    use_pcolormesh=False,
    norm=None,
):
    ax.coastlines(color="grey", zorder=3, linewidth=0.5)
    data_to_plot_masked = np.ma.masked_invalid(data_2d_arr)

    if norm is None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        # Ensure that if a norm object is passed, its vmin/vmax are updated
        # This is important if the same norm object is reused.
        norm.vmin = vmin
        norm.vmax = vmax


    im = None
    if use_pcolormesh:
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
    else:
        num_levels = 16
        plot_vmin, plot_vmax = norm.vmin, norm.vmax
        if isinstance(norm, mcolors.LogNorm):
            # Ensure levels are finite and > 0 for LogNorm
            if plot_vmin > 0 and plot_vmax > 0 and plot_vmax > plot_vmin:
                log_levels = np.geomspace(plot_vmin, plot_vmax, num_levels)
                levels = log_levels[np.isfinite(log_levels)]
                # Ensure at least two unique levels for contourf after rounding
                if len(np.unique(np.round(levels, decimals=15))) < 2:
                    levels = np.array([plot_vmin, plot_vmax])
            else: # Fallback to linear if log conditions not met
                levels = np.array([vmin, vmax]) # Use input vmin/vmax for fallback
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else: # Linear norm
            if abs(plot_vmax - plot_vmin) < 1e-12: # Avoid issues with identical vmin/vmax
                levels = np.array([plot_vmin, plot_vmax])
            else:
                levels = np.linspace(plot_vmin, plot_vmax, num_levels)

        if len(levels) < 2: # Ensure contourf has at least two levels
             levels = np.array([plot_vmin, plot_vmax])

        im = ax.contourf(
            lon_coords_2d,
            lat_coords_2d,
            data_to_plot_masked,
            levels=levels,
            norm=norm,
            cmap=cmap,
            extend="both",
            transform=ccrs.PlateCarree(),
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
    io = IO(interpolated_filename_prefix)

    settings = io.load_dataset("settings", print_info=False)
    if settings is None:
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

    pynamit_timeseries_key_map = {
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

    print("Starting Pass 1: Collecting and Caching data for global vmin/vmax...") # Modified print
    all_data_for_scaling = {
        dt_str: {"input": [], "interpolated": []} for dt_str in data_types_to_plot
    }
    cached_plot_data = {} # NEW: Initialize cache for 2D plot data
    plot_evaluators = {} # Evaluators needed for Pass 1 interpolation

    for timestep in timesteps_to_plot:
        if (timestep < 0) or (timestep >= num_h5_steps):
            # Skip invalid timesteps but allow others to proceed for scaling
            print(f"Warning: Invalid timestep {timestep} skipped for data collection.")
            continue 

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

            # --- Calculate input_data_2d ---
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

            # --- Calculate interpolated_data_2d ---
            calculated_interpolated_data_2d = np.full(target_shape_pass1, np.nan)
            timeseries_entry = input_timeseries.get_entry(
                pynamit_ts_key, time_val, interpolation=False
            )
            if timeseries_entry:
                storage_basis = input_timeseries.storage_bases[pynamit_ts_key]
                if data_type_str not in plot_evaluators:
                    plot_evaluators[data_type_str] = BasisEvaluator(
                        storage_basis, Grid(lat=current_lat_coords_pass1, lon=current_lon_coords_pass1)
                    )
                current_plot_evaluator = plot_evaluators[data_type_str]

                if pynamit_ts_key == "Br":
                    calculated_interpolated_data_2d = _calculate_interpolated_scalar_field(
                        timeseries_entry, "Br", storage_basis, current_plot_evaluator, target_shape_pass1
                    )
                elif pynamit_ts_key == "jr":
                    calculated_interpolated_data_2d = _calculate_interpolated_scalar_field(
                        timeseries_entry, "jr", storage_basis, current_plot_evaluator, target_shape_pass1
                    )
                elif pynamit_ts_key == "conductance":
                    calculated_interpolated_data_2d = _calculate_interpolated_conductance(
                        timeseries_entry, data_type_str, storage_basis, current_plot_evaluator, target_shape_pass1
                    )
                elif pynamit_ts_key == "u":
                    calculated_interpolated_data_2d = _calculate_interpolated_u_field(
                        timeseries_entry, data_type_str, storage_basis, current_plot_evaluator, target_shape_pass1
                    )
            
            all_data_for_scaling[data_type_str]["interpolated"].append(calculated_interpolated_data_2d.ravel())

            # NEW: Cache the calculated 2D arrays
            cache_key = (timestep, data_type_str)
            cached_plot_data[cache_key] = {
                'input': calculated_input_data_2d,
                'interpolated': calculated_interpolated_data_2d
            }

    print("Calculating global vmin/vmax from percentiles (minimal adjustment if identical)...")
    global_plot_scales = {}
    # ... (Global scale calculation logic - unchanged from previous correct version) ...
    for data_type_str in data_types_to_plot:
        details = data_type_details[data_type_str]
        cmap_global = details["cmap"]
        current_scale_type = details.get("scale_type", "linear")
        cat_input_all = all_data_for_scaling[data_type_str]["input"]
        cat_interp_all = all_data_for_scaling[data_type_str]["interpolated"]
        
        # Ensure lists are not empty before concatenation
        flat_input = np.concatenate(cat_input_all) if cat_input_all and any(a.size > 0 for a in cat_input_all) else np.array([])
        flat_interp = np.concatenate(cat_interp_all) if cat_interp_all and any(a.size > 0 for a in cat_interp_all) else np.array([])
        
        combined_flat_list = []
        if flat_input.size > 0:
            combined_flat_list.append(flat_input)
        if flat_interp.size > 0:
            combined_flat_list.append(flat_interp)

        if not combined_flat_list: # If both are empty, no data to scale
             vmin, vmax = 0, 1 # Default values, or handle as error
             print(f"Warning: No data found for '{data_type_str}' to calculate scales. Using default range [0,1].")
        else:
            combined_flat = np.concatenate(combined_flat_list)
            temp_valid_data = combined_flat[~np.isnan(combined_flat)]

            if temp_valid_data.size == 0:
                vmin, vmax = 0, 1 # Default values if all are NaN
                print(f"Warning: All data for '{data_type_str}' is NaN. Using default range [0,1].")
            else:
                if current_scale_type == "log":
                    valid_data_for_percentile = temp_valid_data[temp_valid_data > 0]
                    if valid_data_for_percentile.size == 0: # All non-NaN data is <=0
                        print(f"Warning: No positive data for log scale in '{data_type_str}'. Using linear scale over actual range.")
                        current_scale_type = "linear" # Fallback to linear
                        valid_data_for_percentile = temp_valid_data # Recalculate percentiles on all valid data
                else:
                    valid_data_for_percentile = temp_valid_data

                if valid_data_for_percentile.size == 0: # Still no data (e.g. log scale fallback didn't find data)
                    vmin, vmax = 0, 1
                    print(f"Warning: No valid data for percentile calculation for '{data_type_str}'. Using default range [0,1].")
                elif cmap_global == "bwr":
                    abs_max_s = np.percentile(np.abs(valid_data_for_percentile), vmax_percentile)
                    vmin = -abs_max_s
                    vmax = abs_max_s
                else:
                    vmin = np.percentile(valid_data_for_percentile, vmin_percentile)
                    vmax = np.percentile(valid_data_for_percentile, vmax_percentile)
                    if current_scale_type == "linear" and positive_definite_zeromin:
                        if vmin >= 0: # Only set to 0 if vmin is already non-negative
                            vmin = 0.0
        
        # Ensure vmin and vmax are not identical for plotting
        if abs(vmax - vmin) < 1e-12:
            print(f"  Adjusting identical vmin/vmax for '{data_type_str}': {vmin:.3e}, {vmax:.3e}")
            if vmin == 0:
                vmax = 1.0 # default if vmin=vmax=0
            else: # Add a small range
                vmin -= abs(vmin * 0.01) if vmin != 0 else 0.01
                vmax += abs(vmax * 0.01) if vmax != 0 else 0.01
            if abs(vmax - vmin) < 1e-12: # if still identical (e.g. both were 0 and became +/-0.01)
                vmin = vmin - 0.5
                vmax = vmax + 0.5


        norm_for_plot = None
        if current_scale_type == "log" and cmap_global != "bwr":
            if vmin <= 0 or vmax <= vmin : # Check again after potential adjustments
                print(f"Warning: Invalid log scale for '{data_type_str}': vmin={vmin:.2e}, vmax={vmax:.2e}. Both must be positive and vmax > vmin. Switching to linear.")
                norm_for_plot = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                current_scale_type = "linear" # Update scale type if changed
            else:
                norm_for_plot = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        
        # If norm_for_plot is still None (i.e., linear or bwr), create it now
        if norm_for_plot is None:
             norm_for_plot = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)


        global_plot_scales[data_type_str] = {
            "vmin": vmin,
            "vmax": vmax,
            "cmap": cmap_global,
            "norm": norm_for_plot, # Use the potentially adjusted norm
            "scale_type": current_scale_type, # Use the potentially adjusted scale_type
        }
        print(
            f"  Global scale for '{data_type_str}' ({current_scale_type}): vmin={vmin:.3e}, vmax={vmax:.3e}"
        )


    # --- Figure Creation using 2x2 Subfigures ---
    num_dt = len(data_types_to_plot)
    num_plot_rows = num_dt * 2
    num_plot_cols = len(timesteps_to_plot)

    # Define relative sizes for subfigures - make top row and left col narrow
    time_row_h_frac = 0.04  # Small height for the top row (time labels)
    plots_row_h_frac = 1.0 - time_row_h_frac

    # For the two columns of the main figure grid:
    # Col 0 (BL): Colorbars & Labels | Col 1 (BR): Plots
    cbars_labels_col_w_frac = 0.20  # Adjusted for all labels + cbars
    plots_col_w_frac = 1.0 - cbars_labels_col_w_frac

    # Estimate overall figure size
    base_plot_w, base_plot_h = 2.0, 1.5
    fig_width = (num_plot_cols * base_plot_w) / plots_col_w_frac
    fig_height = (num_plot_rows * base_plot_h) / plots_row_h_frac

    fig_width = min(max(8, fig_width), 25)
    fig_height = min(
        max(4 + time_row_h_frac * fig_height, fig_height),
        num_plot_rows * (base_plot_h + 0.15) + 1.0,
    )

    print(f"Creating figure with 2x2 subfigures. Target Size: ({fig_width:.1f},{fig_height:.1f})")
    fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")

    sfigs_grid = fig.subfigures(
        2,
        2,
        height_ratios=[time_row_h_frac, plots_row_h_frac],
        width_ratios=[cbars_labels_col_w_frac, plots_col_w_frac],
        hspace=0.01,
        wspace=0.01,
    )

    sfig_TL_empty = sfigs_grid[0, 0]
    sfig_TR_times = sfigs_grid[0, 1]
    sfig_BL_cbars_and_labels = sfigs_grid[1, 0]
    sfig_BR_plots = sfigs_grid[1, 1]

    sfig_TL_empty.patch.set_alpha(0.0)
    for ax_ in sfig_TL_empty.get_axes():
        ax_.remove()

    # --- Time Labels in sfig_TR_times (Top-Right) ---
    if num_plot_cols > 0:
        time_label_axes_flat = sfig_TR_times.subplots(1, num_plot_cols, sharey=True)
        time_label_axes = [time_label_axes_flat] if num_plot_cols == 1 else time_label_axes_flat
        for ts_idx, timestep in enumerate(timesteps_to_plot):
            time_val = timestep * input_dt
            time_label_axes[ts_idx].text(
                0.5, 0.5, f"{time_val}s", ha="center", va="center", fontsize=9
            )
            time_label_axes[ts_idx].axis("off")

    # --- Main Map Plots in sfig_BR_plots (Bottom-Right) ---
    map_axes_flat = sfig_BR_plots.subplots(
        num_plot_rows,
        num_plot_cols,
        sharex=True,
        sharey=True,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=noon_longitude)},
    )
    if num_plot_rows == 0 or num_plot_cols == 0:
        if h5file:
            h5file.close()
            print("No data types or timesteps to plot. Exiting.")
            return
    if num_plot_rows == 1 and num_plot_cols == 1:
        map_axes = np.array([[map_axes_flat]])
    elif num_plot_rows == 1:
        map_axes = map_axes_flat[np.newaxis, :]
    elif num_plot_cols == 1:
        map_axes = map_axes_flat[:, np.newaxis]
    else:
        map_axes = map_axes_flat

    # --- Bottom-Left Panel (sfig_BL_cbars_and_labels) ---
    gs_bottom_left = gridspec.GridSpec(
        num_plot_rows,
        2,
        figure=sfig_BL_cbars_and_labels,
        width_ratios=[0.8, 0.2],
        wspace=0.05,
        hspace=0.05,
    )

    mappables_for_cbars = [None] * num_dt

    for dt_idx, data_type_str in enumerate(data_types_to_plot):
        details = data_type_details[data_type_str]
        current_global_scale = global_plot_scales[data_type_str]
        vmin_use, vmax_use, cmap_use = (
            current_global_scale["vmin"],
            current_global_scale["vmax"],
            current_global_scale["cmap"],
        )
        norm_use = current_global_scale["norm"] # This is now a norm object

        row_idx_input = dt_idx * 2
        row_idx_fitted = row_idx_input + 1
        current_mappable_this_dt = None

        gs_dt_cbar_block = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_bottom_left[row_idx_input : row_idx_input + 2, 0],
            width_ratios=[0.4, 0.6], wspace=0.1,
        )

        ax_dt_label = sfig_BL_cbars_and_labels.add_subplot(gs_dt_cbar_block[0])
        ax_dt_label.text(0.5, 0.5, details["label"], ha="center", va="center", rotation=90, fontsize=9)
        ax_dt_label.axis("off")

        ax_input_text_label = sfig_BL_cbars_and_labels.add_subplot(gs_bottom_left[row_idx_input, 1])
        ax_input_text_label.text(0.5, 0.5, "Input", ha="center", va="center", rotation=90, fontsize=8)
        ax_input_text_label.axis("off")

        ax_fitted_text_label = sfig_BL_cbars_and_labels.add_subplot(gs_bottom_left[row_idx_fitted, 1])
        ax_fitted_text_label.text(0.5, 0.5, "Fitted", ha="center", va="center", rotation=90, fontsize=8)
        ax_fitted_text_label.axis("off")

        for ts_idx, timestep in enumerate(timesteps_to_plot):
            # time_val = timestep * input_dt # Already available if needed for titles, etc.
            ax_input = map_axes[row_idx_input, ts_idx]
            ax_fitted = map_axes[row_idx_fitted, ts_idx]
            ax_input.clear()
            ax_fitted.clear()

            plot_title = "" 

            is_magnetosphere_grid_plot = details["grid_type"] == "magnetosphere"
            current_lon_plot, current_lat_plot, target_shape_plot = (
                (magnetosphere_lon, magnetosphere_lat, magnetosphere_lat.shape)
                if is_magnetosphere_grid_plot
                else (ionosphere_lon, ionosphere_lat, ionosphere_lat.shape)
            )
            
            # NEW: Retrieve data from cache
            cache_key = (timestep, data_type_str)
            if cache_key in cached_plot_data:
                retrieved_input_data = cached_plot_data[cache_key]['input']
                retrieved_interpolated_data = cached_plot_data[cache_key]['interpolated']
            else:
                # Fallback if a timestep was invalid and skipped in Pass 1, or other error
                retrieved_input_data = np.full(target_shape_plot, np.nan)
                retrieved_interpolated_data = np.full(target_shape_plot, np.nan)
                if 0 <= timestep < num_h5_steps: # Only warn if timestep was expected to be valid
                    print(f"Warning: Data for {cache_key} not found in cache. Plotting NaNs.")
            
            # This handles the "Data OOB" case as well, since invalid timesteps won't be in cache
            # and will plot NaNs, or if you want explicit text:
            if not (0 <= timestep < num_h5_steps):
                 ax_input.text(0.5, 0.5, "Data OOB", ha="center", va="center", transform=ax_input.transAxes)
                 ax_input.set_xticks([]); ax_input.set_yticks([])
                 ax_fitted.text(0.5, 0.5, "Data OOB", ha="center", va="center", transform=ax_fitted.transAxes)
                 ax_fitted.set_xticks([]); ax_fitted.set_yticks([])
                 # continue # Data already set to NaN, plot_scalar_map will handle it. Or skip plotting.


            im_input = plot_scalar_map_on_ax(
                ax_input, current_lon_plot, current_lat_plot, retrieved_input_data,
                plot_title, cmap_use, vmin_use, vmax_use,
                use_pcolormesh=True, norm=norm_use # Pass the norm object
            )
            if current_mappable_this_dt is None and not np.all(np.isnan(retrieved_input_data)):
                current_mappable_this_dt = im_input

            im_fitted = plot_scalar_map_on_ax(
                ax_fitted, current_lon_plot, current_lat_plot, retrieved_interpolated_data,
                "", cmap_use, vmin_use, vmax_use,
                use_pcolormesh=True, norm=norm_use # Pass the norm object
            )
            if current_mappable_this_dt is None and not np.all(np.isnan(retrieved_interpolated_data)):
                current_mappable_this_dt = im_fitted

        mappables_for_cbars[dt_idx] = current_mappable_this_dt

        if current_mappable_this_dt:
            ax_cbar = sfig_BL_cbars_and_labels.add_subplot(gs_dt_cbar_block[1])
            cb = fig.colorbar(current_mappable_this_dt, cax=ax_cbar, orientation="vertical")
            cb.ax.tick_params(labelsize=7)
        else:
            ax_cbar = sfig_BL_cbars_and_labels.add_subplot(gs_dt_cbar_block[1])
            ax_cbar.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=7)
            ax_cbar.axis("off")

    if output_filename:
        plt.savefig(output_filename, dpi=200)
        print(f"Figure saved to {output_filename}")
    else:
        plt.show()
    h5file.close()