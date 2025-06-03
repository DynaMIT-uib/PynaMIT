import numpy as np

import matplotlib.pyplot as plt
import h5py as h5
import cartopy.crs as ccrs
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
        raise ValueError(
            f"CRITICAL ERROR: Invalid vmin/vmax for '{title}': vmin={current_vmin}, vmax={current_vmax}. Cannot plot."
        )

    ax.set_title(title, fontsize=10)
    return im


# --- Main plotting function ---
def plot_input_vs_interpolated(
    h5_filepath,
    interpolated_filename_prefix,
    timesteps_to_plot,
    input_dt,
    data_types_to_plot,
    noon_longitude=0,
    output_filename=None,
):
    try:
        h5file = h5.File(h5_filepath, "r")
    except Exception as e:
        print(f"CRITICAL ERROR: Opening HDF5 {h5_filepath}: {e}")
        return

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

    timeseries_key_map = {
        "Br": "Br",
        "jr": "jr",
        "SH": "conductance",
        "SP": "conductance",
        "u_mag": "u",
        "u_theta": "u",
        "u_phi": "u",
    }

    input_timeseries = Timeseries(cs_basis, input_storage_bases, input_vars)
    input_timeseries.load_all(io)

    # H5 coordinates are assumed to be 2D arrays of lat/lon in degrees.
    ionosphere_lat = h5file["glat"][:]
    ionosphere_lon = h5file["glon"][:]
    magnetosphere_lat = h5file["Blat"][:]
    magnetosphere_lon = h5file["Blon"][:]

    ionosphere_grid = Grid(lat=ionosphere_lat, lon=ionosphere_lon)
    ionosphere_b_evaluator = FieldEvaluator(mainfield, ionosphere_grid, ri_value)
    ionosphere_br_2d = ionosphere_b_evaluator.br.reshape(h5file["FAC"][0, :, :].shape)

    num_rows = len(timesteps_to_plot)
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

    for row_idx, timestep in enumerate(timesteps_to_plot):
        if timestep < 0 or timestep >= num_h5_steps:
            raise ValueError(
                f"Invalid timestep {timestep} for HDF5 data with {num_h5_steps} steps."
            )

        datetime = h5file["time"][timestep]
        time = timestep * input_dt

        print(f"  -- Datetime: {datetime} (timestep {timestep + 1}/{num_rows}, time {time}s) --")

        axes[row_idx, 0].set_ylabel(
            f"{time}s", fontsize=10, labelpad=35, rotation=0, ha="right", va="center"
        )

        for data_type_idx, data_type_str in enumerate(data_types_to_plot):
            print(
                f"  -- Processing data type '{data_type_str}' (column pair {data_type_idx * 2}, {data_type_idx * 2 + 1}) --"
            )
            col_idx_input = data_type_idx * 2
            col_idx_interpolated = col_idx_input + 1

            ax_input = axes[row_idx, col_idx_input]
            ax_interpolated = axes[row_idx, col_idx_interpolated]

            if data_type_str == "Br":
                input_data_2d = h5file["Bu"][:][row_idx, :, :] * 1e-9
                current_lon, current_lat = magnetosphere_lon, magnetosphere_lat
                data_label = r"$\Delta B_r$ [T]"
                cmap = "bwr"

            elif data_type_str == "jr":
                FAC_input_2d = h5file["FAC"][:][row_idx, :, :] * 1e-6
                input_data_2d = FAC_input_2d * ionosphere_br_2d
                current_lon, current_lat = ionosphere_lon, ionosphere_lat
                data_label = r"$j_r$ [A/m$^2$]"
                cmap = "bwr"

            elif data_type_str == "SH":
                input_data_2d = h5file["SH"][:][row_idx, :, :]
                current_lon, current_lat = ionosphere_lon, ionosphere_lat
                data_label = r"$\Sigma_H$ [S]"
                cmap = "viridis"

            elif data_type_str == "SP":
                input_data_2d = h5file["SP"][:][row_idx, :, :]
                current_lon, current_lat = ionosphere_lon, ionosphere_lat
                data_label = r"$\Sigma_P$ [S]"
                cmap = "viridis"

            elif data_type_str in ["u_mag", "u_theta", "u_phi"]:
                u_east_input = h5file["We"][:][row_idx, :, :]
                u_north_input = h5file["Wn"][:][row_idx, :, :]

                if data_type_str == "u_mag":
                    input_data_2d = np.sqrt(u_north_input**2 + u_east_input**2)
                    data_label = r"$|u|$ [m/s]"
                    cmap = "viridis"
                elif data_type_str == "u_theta":
                    input_data_2d = -u_north_input
                    data_label = r"$u_\theta$ (South) [m/s]"
                    cmap = "bwr"
                elif data_type_str == "u_phi":
                    input_data_2d = u_east_input
                    data_label = r"$u_\phi$ (East) [m/s]"
                    cmap = "bwr"

            else:
                raise ValueError(f"Unsupported data type '{data_type_str}' for plotting.")

            timeseries_key = timeseries_key_map.get(data_type_str)
            timeseries_entry = input_timeseries.get_entry_if_changed(
                timeseries_key, time, interpolation=False
            )
            if timeseries_entry is None:
                raise ValueError(
                    f"Timeseries entry for '{timeseries_key}' at time {time}s not found in input timeseries."
                )

            if timeseries_entry:
                storage_basis = input_timeseries.storage_bases[timeseries_key]
                plot_evaluator = BasisEvaluator(
                    storage_basis, Grid(lat=current_lat, lon=current_lon)
                )

                if timeseries_key == "conductance":
                    etaP = FieldExpansion(
                        storage_basis, coeffs=timeseries_entry["etaP"], field_type="scalar"
                    )
                    etaH = FieldExpansion(
                        storage_basis, coeffs=timeseries_entry["etaH"], field_type="scalar"
                    )
                    etaP_fitted_2d = etaP.to_grid(plot_evaluator).reshape(input_data_2d.shape)
                    etaH_fitted_2d = etaH.to_grid(plot_evaluator).reshape(input_data_2d.shape)
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

                elif timeseries_key == "u":
                    u = FieldExpansion(
                        storage_basis, coeffs=timeseries_entry["u"], field_type="tangential"
                    )
                    u_theta_flat, u_phi_flat = u.to_grid(plot_evaluator)
                    if data_type_str == "u_mag":
                        interpolated_data_2d = np.sqrt(
                            u_theta_flat.reshape(input_data_2d.shape) ** 2
                            + u_phi_flat.reshape(input_data_2d.shape) ** 2
                        )
                    elif data_type_str == "u_theta":
                        interpolated_data_2d = u_theta_flat.reshape(input_data_2d.shape)
                    elif data_type_str == "u_phi":
                        interpolated_data_2d = u_phi_flat.reshape(input_data_2d.shape)

                else:
                    field_exp = FieldExpansion(
                        storage_basis, coeffs=timeseries_entry[data_type_str], field_type="scalar"
                    )
                    interpolated_data_flat = field_exp.to_grid(plot_evaluator)
                    interpolated_data_2d = interpolated_data_flat.reshape(input_data_2d.shape)

            if input_data_2d is None or interpolated_data_2d is None:
                raise ValueError(
                    f"Input or interpolated data for '{data_label}' is None at timestep {timestep}."
                )

            # Plotting call
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
                print(f"Warning: Both input/fitted data for '{data_label}' are all NaN or empty.")

            plot_title_input = f"Input {data_label}" if row_idx == 0 else "Input"
            plot_title_fitted = f"Fitted {data_label}" if row_idx == 0 else "Fitted"

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

    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.05, hspace=0.4, wspace=0.15)

    fig.suptitle(
        f"Input vs. Fitted Data (bwr plots use shared vmin/vmax per pair, 'bwr' centered; other cmaps use 0.2-99.8 percentile)",
        fontsize=12,
    )

    if output_filename:
        plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    else:
        plt.show()

    h5file.close()
