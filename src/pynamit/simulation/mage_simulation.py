import numpy as np
import matplotlib

# matplotlib.use('Agg') # Uncomment if running script-only and suspect backend issues
import matplotlib.pyplot as plt
import h5py as h5
import cartopy.crs as ccrs
import pynamit
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.primitives.field_expansion import FieldExpansion
import os
import traceback  # For printing full tracebacks

from pynamit.primitives.io import IO
from pynamit.primitives.timeseries import Timeseries
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.simulation.mainfield import Mainfield
from pynamit.math.constants import RE


# --- Helper plotting function (from your last code block) ---
def plot_scalar_map_on_ax(
    ax,
    lon_coords_2d,
    lat_coords_2d,
    data_2d_arr,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None,
    use_pcolormesh=False,
):
    ax.coastlines(color="grey", zorder=3, linewidth=0.5)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    plot_transform = ccrs.PlateCarree()

    data_min_val, data_max_val, num_nans = (
        np.nanmin(data_2d_arr),
        np.nanmax(data_2d_arr),
        np.isnan(data_2d_arr).sum(),
    )
    print(
        f"    data_2d stats: min={data_min_val:.4g}, max={data_max_val:.4g}, NaNs={num_nans}/{data_2d_arr.size}"
    )

    current_vmin, current_vmax = vmin, vmax
    valid_data = data_2d_arr[~np.isnan(data_2d_arr)]

    if valid_data.size == 0:
        print(f"    Plotting Warning for '{title}': All data is NaN. Setting dummy vmin/vmax.")
        current_vmin, current_vmax = 0.0, 1.0
    else:
        if vmin is None or vmax is None:
            print(f"    vmin/vmax not provided, calculating automatically for '{title}'.")
            if cmap == "bwr":
                abs_max_val = np.percentile(np.abs(valid_data), 99.5)
                auto_vmin = -abs_max_val if abs_max_val > 1e-9 else -0.1
                auto_vmax = abs_max_val if abs_max_val > 1e-9 else 0.1
            else:
                auto_vmin = np.percentile(valid_data, 0.5)
                auto_vmax = np.percentile(valid_data, 99.5)
            if vmin is None:
                current_vmin = auto_vmin
            if vmax is None:
                current_vmax = auto_vmax
            print(f"    Auto-calculated: auto_vmin={auto_vmin:.4g}, auto_vmax={auto_vmax:.4g}")
        if not isinstance(current_vmin, (int, float, np.number)) or not isinstance(
            current_vmax, (int, float, np.number)
        ):
            print(
                f"    CRITICAL WARNING: vmin or vmax is not a number! vmin={current_vmin} (type {type(current_vmin)}), vmax={current_vmax} (type {type(current_vmax)})"
            )
            try:
                current_vmin = float(current_vmin)
                current_vmax = float(current_vmax)
            except:
                print("    Could not convert vmin/vmax to float, using dummy values.")
                current_vmin, current_vmax = 0.0, 1.0
        if current_vmin >= current_vmax:
            print(
                f"    Plotting Warning for '{title}': vmin ({current_vmin:.4g}) >= vmax ({current_vmax:.4g}). Adjusting."
            )
            original_calc_vmin = current_vmin
            original_calc_vmax = current_vmax
            if np.isclose(current_vmin, current_vmax):
                delta = 0.1 * abs(current_vmin) if not np.isclose(current_vmin, 0) else 0.1
                current_vmin -= delta
                current_vmax += delta
            else:
                if valid_data.size > 0:
                    current_vmin = np.min(valid_data)
                    current_vmax = np.max(valid_data)
                    if np.isclose(current_vmin, current_vmax):
                        delta = 0.1 * abs(current_vmin) if not np.isclose(current_vmin, 0) else 0.1
                        current_vmin -= delta
                        current_vmax += delta
                else:
                    current_vmin, current_vmax = 0.0, 1.0
            print(
                f"    Adjusted: vmin={current_vmin:.4g}, vmax={current_vmax:.4g} (from original calc vmin={original_calc_vmin:.4g}, vmax={original_calc_vmax:.4g})"
            )
    print(f"    Final pre-plot vmin={current_vmin:.4g}, vmax={current_vmax:.4g}, cmap={cmap}")

    im = None
    try:
        if current_vmin < current_vmax and not (np.isnan(current_vmin) or np.isnan(current_vmax)):
            data_to_plot_masked = np.ma.masked_invalid(data_2d_arr)
            if use_pcolormesh:
                print(f"    Using pcolormesh for '{title}' with transform={plot_transform}")
                im = ax.pcolormesh(
                    lon_coords_2d,
                    lat_coords_2d,
                    data_to_plot_masked,
                    cmap=cmap,
                    vmin=current_vmin,
                    vmax=current_vmax,
                    transform=plot_transform,
                    shading="auto",
                    zorder=1,
                )
            else:
                print(f"    Using contourf for '{title}' with transform={plot_transform}")
                if num_nans > 0:
                    print(
                        f"    NOTE: Data for '{title}' contains {num_nans} NaNs. Masking applied for contourf."
                    )
                print(lat_coords_2d.shape, lon_coords_2d.shape, data_to_plot_masked.shape)
                im = ax.contourf(
                    lon_coords_2d,
                    lat_coords_2d,
                    data_to_plot_masked,
                    transform=plot_transform,
                    cmap=cmap,
                    vmin=current_vmin,
                    vmax=current_vmax,
                    levels=15,
                    extend="both",
                    zorder=1,
                )
            print(f"    Plotting call successful for '{title}'.")
        else:
            print(f"    Skipping plot for '{title}' due to invalid vmin/vmax range.")
            ax.text(
                0.5,
                0.5,
                "Data Range Issue",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="orange",
                fontsize=8,
            )
    except Exception as e:
        print(f"    EXCEPTION during plotting for '{title}': {e}")
        traceback.print_exc()
        ax.text(
            0.5,
            0.5,
            "Plotting Error",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="red",
            fontsize=8,
        )
    ax.set_title(title, fontsize=10)
    return im


# --- Main plotting function ---
def plot_input_vs_interpolated(
    h5_filepath,
    interpolated_filename_prefix,
    times_to_plot,
    data_types_to_plot,
    dt_inputs,
    noon_longitude=0,
    output_filename=None,
):
    try:
        h5file = h5.File(h5_filepath, "r")
    except Exception as e:
        print(f"CRITICAL ERROR: Opening HDF5 {h5_filepath}: {e}")
        return

    bu_data_shape = h5file["Bu"][0, :, :].shape
    ionospheric_data_shape = h5file["FAC"][0, :, :].shape
    num_h5_steps = h5file["Bu"].shape[0]

    io = IO(interpolated_filename_prefix)
    settings = io.load_dataset("settings", print_info=False)
    if settings is None:
        print(f"CRITICAL ERROR: No 'settings' from '{interpolated_filename_prefix}'.")
        h5file.close()
        return

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
    input_vars = {
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
    input_timeseries = Timeseries(cs_basis, input_storage_bases, input_vars)
    input_timeseries.load_all(io)
    if not input_timeseries.datasets:
        print("WARNING: Pynamit input_timeseries empty.")

    # --- HDF5 Grid Coordinate Setup ---
    # These are assumed to be 2D arrays of DEGREES from HDF5
    ionosphere_lat = h5file["glat"][:]
    ionosphere_lon = h5file["glon"][:]
    magnetosphere_lat = h5file["Blat"][:]
    magnetosphere_lon = h5file["Blon"][:]

    print(
        f"Magnetosphere HDF5 raw coord (Blat,Blon) values (ASSUMED DEGREES): LAT {magnetosphere_lat.min():.2f}/{magnetosphere_lat.max():.2f}, LON {magnetosphere_lon.min():.2f}/{magnetosphere_lon.max():.2f}"
    )
    print(
        f"Ionosphere HDF5 raw coord (glat,glon) values (ASSUMED DEGREES): LAT {ionosphere_lat.min():.2f}/{ionosphere_lat.max():.2f}, LON {ionosphere_lon.min():.2f}/{ionosphere_lon.max():.2f}"
    )

    # ... (FAC_b_evaluator, figure setup as in your last full code) ...
    ionosphere_grid = Grid(lat=ionosphere_lat, lon=ionosphere_lon)
    ionosphere_b_evaluator = FieldEvaluator(mainfield, ionosphere_grid, ri_value)
    ionosphere_br_2d = ionosphere_b_evaluator.br.reshape(ionospheric_data_shape)

    num_rows = len(times_to_plot)
    num_cols = len(data_types_to_plot) * 2
    fig_width = min(max(10, num_cols * 4.0), 40)
    fig_height = min(max(7, num_rows * 3.5), 35)

    print(f"Creating figure: {num_rows}x{num_cols}, size:({fig_width},{fig_height})")

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(fig_width, fig_height),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=noon_longitude)},
        squeeze=False,
    )

    for row_idx, time_val_secs in enumerate(times_to_plot):
        print(
            f"\n--- Processing plot row {row_idx + 1}/{num_rows} for time_val_secs = {time_val_secs}s ---"
        )
        target_step_idx_float = time_val_secs / dt_inputs
        step_idx = int(round(target_step_idx_float))

        if step_idx < 0 or step_idx >= num_h5_steps:
            print(f"Warning: HDF5 index {step_idx} out of bounds. Skipping row.")
            for col_idx_to_hide in range(num_cols):
                axes[row_idx, col_idx_to_hide].set_visible(False)
            continue

        actual_h5_sim_time = step_idx * dt_inputs

        if not np.isclose(actual_h5_sim_time, time_val_secs, atol=dt_inputs / 1.9):
            print(
                f"Note: Requested {time_val_secs}s. Using HDF5 input from {actual_h5_sim_time}s (index {step_idx})."
            )
        if num_cols > 0:
            axes[row_idx, 0].set_ylabel(
                f"{time_val_secs}s", fontsize=10, labelpad=35, rotation=0, ha="right", va="center"
            )

        for data_type_idx, data_type_str in enumerate(data_types_to_plot):
            print(
                f"  -- Processing data type '{data_type_str}' (column pair {data_type_idx * 2}, {data_type_idx * 2 + 1}) --"
            )
            col_idx_input = data_type_idx * 2
            col_idx_interpolated = col_idx_input + 1
            ax_input = axes[row_idx, col_idx_input]
            ax_interpolated = axes[row_idx, col_idx_interpolated]
            input_data_2d, interpolated_data_2d = None, None
            current_lon, current_lat, input_data_2d.shape = None, None, None
            data_label, cmap = data_type_str, "viridis"

            if data_type_str == "Br":
                input_data_2d = h5file["Bu"][:][step_idx, :, :] * 1e-9
                current_lon, current_lat = magnetosphere_lon, magnetosphere_lat
                data_label, cmap = r"$\Delta B_r$ [T]", "bwr"
            elif data_type_str == "jr":
                FAC_input_2d = h5file["FAC"][:][step_idx, :, :] * 1e-6
                input_data_2d = FAC_input_2d * ionosphere_br_2d
                current_lon, current_lat = ionosphere_lon, ionosphere_lat
                data_label, cmap = r"$j_r$ [A/m$^2$]", "bwr"
            # ... (SH, SP, u logic as before) ...
            elif data_type_str == "SH":
                input_data_2d = h5file["SH"][:][step_idx, :, :]
                current_lon, current_lat = ionosphere_lon, ionosphere_lat
                data_label = r"$\Sigma_H$ [S]"
            elif data_type_str == "SP":
                input_data_2d = h5file["SP"][:][step_idx, :, :]
                current_lon, current_lat = ionosphere_lon, ionosphere_lat
                data_label = r"$\Sigma_P$ [S]"
            elif data_type_str in ["u_mag", "u_theta", "u_phi"]:
                u_east_input = h5file["We"][:][step_idx, :, :]
                u_north_input = h5file["Wn"][:][step_idx, :, :]
                _u_theta_in_2d, _u_phi_in_2d = -u_north_input, u_east_input
                input_data_2d.shape = ionospheric_data_shape
                if data_type_str == "u_mag":
                    input_data_2d = np.sqrt(_u_theta_in_2d**2 + _u_phi_in_2d**2)
                elif data_type_str == "u_theta":
                    input_data_2d = _u_theta_in_2d
                elif data_type_str == "u_phi":
                    input_data_2d = _u_phi_in_2d
                if data_type_str == "u_mag":
                    data_label = r"$|u|$ [m/s]"
                elif data_type_str == "u_theta":
                    data_label, cmap = r"$u_\theta$ (South) [m/s]", "bwr"
                elif data_type_str == "u_phi":
                    data_label, cmap = r"$u_\phi$ (East) [m/s]", "bwr"

            else:
                print(f"ERROR: Unknown input data_type: {data_type_str}.")
                ax_input.set_visible(False)
                ax_interpolated.set_visible(False)
                continue
            print(
                f"Input data for '{data_type_str}' loaded, shape: {input_data_2d.shape if input_data_2d is not None else 'None'}. Target plot data shape: {input_data_2d.shape}"
            )

            timeseries_key_map = {
                "Br": "Br",
                "jr": "jr",
                "SH": "conductance",
                "SP": "conductance",
                "u_mag": "u",
                "u_theta": "u",
                "u_phi": "u",
            }
            timeseries_key = timeseries_key_map.get(data_type_str)
            if timeseries_key:
                print(
                    f"Attempting to get fitted data for '{timeseries_key}' at sim_time {time_val_secs}s"
                )
                timeseries_entry = input_timeseries.get_entry_if_changed(
                    timeseries_key, time_val_secs, interpolation=True
                )
                if timeseries_entry:
                    print(
                        f"Timeseries entry found for '{timeseries_key}'. Keys: {list(timeseries_entry.keys())}"
                    )

                    storage_basis = input_timeseries.storage_bases[timeseries_key]
                    target_plot_grid = Grid(lat=current_lat, lon=current_lon)
                    plot_evaluator = BasisEvaluator(storage_basis, target_plot_grid)

                    if timeseries_key == "conductance":
                        if "etaP" in timeseries_entry and "etaH" in timeseries_entry:
                            etaP_coeffs = timeseries_entry["etaP"]
                            etaH_coeffs = timeseries_entry["etaH"]
                            field_exp_etaP = FieldExpansion(
                                storage_basis, coeffs=etaP_coeffs, field_type="scalar"
                            )
                            field_exp_etaH = FieldExpansion(
                                storage_basis, coeffs=etaH_coeffs, field_type="scalar"
                            )
                            etaP_fitted_2d = field_exp_etaP.to_grid(plot_evaluator).reshape(
                                input_data_2d.shape
                            )
                            etaH_fitted_2d = field_exp_etaH.to_grid(plot_evaluator).reshape(
                                input_data_2d.shape
                            )
                            denominator = etaP_fitted_2d**2 + etaH_fitted_2d**2
                            sigma_H_fitted_2d = np.zeros_like(etaH_fitted_2d)
                            sigma_P_fitted_2d = np.zeros_like(etaP_fitted_2d)
                            valid_den = denominator > 1e-12
                            sigma_H_fitted_2d[valid_den] = (
                                etaH_fitted_2d[valid_den] / denominator[valid_den]
                            )
                            sigma_P_fitted_2d[valid_den] = (
                                etaP_fitted_2d[valid_den] / denominator[valid_den]
                            )
                            if data_type_str == "SH":
                                interpolated_data_2d = sigma_H_fitted_2d
                            elif data_type_str == "SP":
                                interpolated_data_2d = sigma_P_fitted_2d
                        else:
                            print(
                                f"Fitted Warning: etaP/etaH coeffs not found for '{timeseries_key}' at {time_val_secs}s."
                            )

                    elif timeseries_key == "u":
                        if "u" in timeseries_entry:
                            u_coeffs_helmholtz = timeseries_entry["u"]
                            field_exp_u = FieldExpansion(
                                storage_basis, coeffs=u_coeffs_helmholtz, field_type="tangential"
                            )
                            u_theta_flat, u_phi_flat = field_exp_u.to_grid(plot_evaluator)
                            _u_theta_fit_2d = u_theta_flat.reshape(input_data_2d.shape)
                            _u_phi_fit_2d = u_phi_flat.reshape(input_data_2d.shape)
                            if data_type_str == "u_mag":
                                interpolated_data_2d = np.sqrt(
                                    _u_theta_fit_2d**2 + _u_phi_fit_2d**2
                                )
                            elif data_type_str == "u_theta":
                                interpolated_data_2d = _u_theta_fit_2d
                            elif data_type_str == "u_phi":
                                interpolated_data_2d = _u_phi_fit_2d
                        else:
                            print(
                                f"Fitted Warning: u coeffs not found for '{timeseries_key}' at {time_val_secs}s."
                            )

                    else:
                        if data_type_str in timeseries_entry:
                            field_coeffs = timeseries_entry[data_type_str]
                            field_exp = FieldExpansion(
                                storage_basis, coeffs=field_coeffs, field_type="scalar"
                            )
                            interpolated_data_flat = field_exp.to_grid(plot_evaluator)
                            interpolated_data_2d = interpolated_data_flat.reshape(
                                input_data_2d.shape
                            )
                        else:
                            print(
                                f"Fitted Warning: {data_type_str} coeffs not found for '{timeseries_key}' at {time_val_secs}s."
                            )

                    if interpolated_data_2d is not None:
                        print(
                            f"Fitted data for '{data_type_str}' processed, shape: {interpolated_data_2d.shape}"
                        )
                    else:
                        print(f"Fitted data for '{data_type_str}' is None after processing.")

                else:
                    print(
                        f"Fitted Warning: No timeseries entry for '{timeseries_key}' at {time_val_secs}s."
                    )

            # --- Plotting call ---
            vmin_shared, vmax_shared = None, None
            if (
                input_data_2d is not None and interpolated_data_2d is not None
            ):  # ... (vmin/vmax calc) ...
                d1_flat_valid = input_data_2d.astype(float).ravel()
                d1_flat_valid = d1_flat_valid[~np.isnan(d1_flat_valid)]
                d2_flat_valid = interpolated_data_2d.astype(float).ravel()
                d2_flat_valid = d2_flat_valid[~np.isnan(d2_flat_valid)]
                combined_valid_data = []
                if d1_flat_valid.size > 0:
                    combined_valid_data.append(d1_flat_valid)
                if d2_flat_valid.size > 0:
                    combined_valid_data.append(d2_flat_valid)
                if combined_valid_data:
                    combined_valid_data = np.concatenate(combined_valid_data)
                    if combined_valid_data.size > 0:
                        if cmap == "bwr":
                            abs_max_s = np.percentile(np.abs(combined_valid_data), 99.8)
                            vmin_shared, vmax_shared = (
                                -abs_max_s if abs_max_s > 1e-9 else -0.1,
                                abs_max_s if abs_max_s > 1e-9 else 0.1,
                            )
                        else:
                            vmin_shared = np.percentile(combined_valid_data, 0.2)
                            vmax_shared = np.percentile(combined_valid_data, 99.8)
                        print(
                            f"Shared scale for '{data_label}': vmin={vmin_shared:.4g}, vmax={vmax_shared:.4g}"
                        )
                    else:
                        print(f"Warning: No valid data for '{data_label}' to set shared scale.")
                else:
                    print(
                        f"Warning: Both input/fitted data for '{data_label}' are all NaN or empty."
                    )
            plot_title_input = f"Input {data_label}" if row_idx == 0 else "Input"
            plot_title_fitted = f"Fitted {data_label}" if row_idx == 0 else "Fitted"

            # Determine plotting method based on data type and NaNs
            use_pcolor_for_input = (data_type_str == "Br") or (
                data_type_str == "jr" and np.isnan(input_data_2d).any()
            )
            use_pcolor_for_fitted = data_type_str == "Br"  # Fitted jr should be smooth

            # if input_data_2d is not None:
            plot_scalar_map_on_ax(
                ax_input,
                current_lon,
                current_lat,
                input_data_2d,
                plot_title_input,
                cmap,
                vmin_shared,
                vmax_shared,
                use_pcolormesh=True,
            )
            # else: print(f"Input data for '{data_label}' is None."); ax_input.set_visible(False)

            # if interpolated_data_2d is not None:
            plot_scalar_map_on_ax(
                ax_interpolated,
                current_lon,
                current_lat,
                interpolated_data_2d,
                plot_title_fitted,
                cmap,
                vmin_shared,
                vmax_shared,
                use_pcolormesh=True,
            )
            # else: print(f"Fitted data for '{data_label}' is None."); ax_interpolated.set_visible(False)

    # --- 5. Finalize Figure ---
    # ... (Identical to your last full code version) ...
    print(f"\n--- Finalizing Figure ---")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.05, hspace=0.4, wspace=0.15)
    if num_rows > 0 and num_cols > 0:
        fig.suptitle(
            f"Input vs. Fitted Data (bwr plots use shared vmin/vmax per pair, 'bwr' centered; other cmaps use 0.2-99.8 percentile)",
            fontsize=12,
        )
    if output_filename:
        print(f"Attempting to save figure to: {output_filename}")
        try:
            plt.savefig(output_filename, dpi=200, bbox_inches="tight")
            print(f"SUCCESS: Figure saved to {output_filename}")
        except Exception as e_save:
            print(f"CRITICAL ERROR during plt.savefig: {e_save}")
            traceback.print_exc()
    else:
        print("Output filename not provided. Attempting plt.show().")
        try:
            plt.show()
            print("plt.show() executed.")
        except Exception as e_show:
            print(f"ERROR during plt.show(): {e_show}")
            traceback.print_exc()
    h5file.close()
    print("--- plot_input_vs_interpolated finished ---")
