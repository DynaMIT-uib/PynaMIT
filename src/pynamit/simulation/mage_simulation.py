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
import traceback # For printing full tracebacks

from pynamit.primitives.io import IO
from pynamit.primitives.timeseries import Timeseries
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.simulation.mainfield import Mainfield
from pynamit.math.constants import RE

# --- Helper plotting function with extensive debugging ---
def plot_scalar_map_on_ax(ax, lon_coords_1d, lat_coords_1d, data_2d, title,
                          cmap='viridis', vmin=None, vmax=None, use_pcolormesh_for_debug=False):
    print(f"\n--- plot_scalar_map_on_ax for: {title} ---")
    print(f"    Initial vmin={vmin}, vmax={vmax}, use_pcolormesh_for_debug={use_pcolormesh_for_debug}")

    ax.coastlines(color='grey', zorder=3, linewidth=0.5)
    try:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=2)
    except TypeError:
        ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=2)

    print(f"    lon_coords_1d received: shape={lon_coords_1d.shape}, min={np.min(lon_coords_1d):.2f}, max={np.max(lon_coords_1d):.2f}, is_sorted={np.all(np.diff(lon_coords_1d) >= 0)}")
    print(f"    lat_coords_1d received: shape={lat_coords_1d.shape}, min={np.min(lat_coords_1d):.2f}, max={np.max(lat_coords_1d):.2f}, is_sorted={np.all(np.diff(lat_coords_1d) >= 0)}")

    lon_coords_1d_arr = np.asarray(lon_coords_1d)
    lat_coords_1d_arr = np.asarray(lat_coords_1d)
    data_2d_arr = np.asarray(data_2d)

    if not (np.all(np.diff(lon_coords_1d_arr) > 0) or np.all(np.diff(lon_coords_1d_arr) < 0)):
        print(f"    WARNING: lon_coords_1d for '{title}' is not strictly monotonic!")
    if not (np.all(np.diff(lat_coords_1d_arr) > 0) or np.all(np.diff(lat_coords_1d_arr) < 0)):
        print(f"    WARNING: lat_coords_1d for '{title}' is not strictly monotonic!")

    X_mesh, Y_mesh = np.meshgrid(lon_coords_1d_arr, lat_coords_1d_arr)
    print(f"    X_mesh shape: {X_mesh.shape}, Y_mesh shape: {Y_mesh.shape}")
    print(f"    data_2d shape: {data_2d_arr.shape}, dtype: {data_2d_arr.dtype}")

    if data_2d_arr.shape != Y_mesh.shape:
        print(f"    CRITICAL WARNING: data_2d shape {data_2d_arr.shape} MISMATCHES meshgrid shape {Y_mesh.shape} for '{title}'!")
        ax.text(0.5, 0.5, "Data/Grid Shape Mismatch", ha='center', va='center', transform=ax.transAxes, color='red', fontsize=8)
        ax.set_title(title, fontsize=9)
        return None

    data_min_val, data_max_val, num_nans = np.nanmin(data_2d_arr), np.nanmax(data_2d_arr), np.isnan(data_2d_arr).sum()
    print(f"    data_2d stats: min={data_min_val:.4g}, max={data_max_val:.4g}, NaNs={num_nans}/{data_2d_arr.size}")

    current_vmin, current_vmax = vmin, vmax
    valid_data = data_2d_arr[~np.isnan(data_2d_arr)]

    if valid_data.size == 0:
        print(f"    Plotting Warning for '{title}': All data is NaN. Setting dummy vmin/vmax.")
        current_vmin, current_vmax = 0.0, 1.0
    else:
        if vmin is None or vmax is None:
            print(f"    vmin/vmax not provided, calculating automatically for '{title}'.")
            if cmap == 'bwr':
                abs_max_val = np.percentile(np.abs(valid_data), 99.5)
                auto_vmin = -abs_max_val if abs_max_val > 1e-9 else -0.1
                auto_vmax = abs_max_val if abs_max_val > 1e-9 else 0.1
            else:
                auto_vmin = np.percentile(valid_data, 0.5)
                auto_vmax = np.percentile(valid_data, 99.5)
            if vmin is None: current_vmin = auto_vmin
            if vmax is None: current_vmax = auto_vmax
            print(f"    Auto-calculated: auto_vmin={auto_vmin:.4g}, auto_vmax={auto_vmax:.4g}")

        if not isinstance(current_vmin, (int, float, np.number)) or \
           not isinstance(current_vmax, (int, float, np.number)):
            print(f"    CRITICAL WARNING: vmin or vmax is not a number! vmin={current_vmin} (type {type(current_vmin)}), vmax={current_vmax} (type {type(current_vmax)})")
            try:
                current_vmin = float(current_vmin)
                current_vmax = float(current_vmax)
            except:
                print("    Could not convert vmin/vmax to float, using dummy values.")
                current_vmin, current_vmax = 0.0, 1.0

        if current_vmin >= current_vmax:
            print(f"    Plotting Warning for '{title}': vmin ({current_vmin:.4g}) >= vmax ({current_vmax:.4g}). Adjusting.")
            original_calc_vmin = current_vmin
            original_calc_vmax = current_vmax
            if np.isclose(current_vmin, current_vmax):
                delta = 0.1 * abs(current_vmin) if not np.isclose(current_vmin, 0) else 0.1
                current_vmin -= delta
                current_vmax += delta
            else: # vmin > vmax
                if valid_data.size > 0:
                    current_vmin = np.min(valid_data)
                    current_vmax = np.max(valid_data)
                    if np.isclose(current_vmin, current_vmax):
                        delta = 0.1 * abs(current_vmin) if not np.isclose(current_vmin, 0) else 0.1
                        current_vmin -= delta
                        current_vmax += delta
                else:
                    current_vmin, current_vmax = 0.0, 1.0
            print(f"    Adjusted: vmin={current_vmin:.4g}, vmax={current_vmax:.4g} (from original calc vmin={original_calc_vmin:.4g}, vmax={original_calc_vmax:.4g})")

    print(f"    Final pre-plot values for '{title}': vmin={current_vmin:.4g}, vmax={current_vmax:.4g}, cmap={cmap}")

    im = None
    try:
        if current_vmin < current_vmax and not (np.isnan(current_vmin) or np.isnan(current_vmax)):
            if use_pcolormesh_for_debug:
                print(f"    Using pcolormesh for '{title}'")
                im = ax.pcolormesh(X_mesh, Y_mesh, data_2d_arr,
                                 cmap=cmap, vmin=current_vmin, vmax=current_vmax,
                                 transform=ccrs.PlateCarree(), shading='auto', zorder=1)
            else:
                print(f"    Using contourf for '{title}'")
                im = ax.contourf(X_mesh, Y_mesh, data_2d_arr, transform=ccrs.PlateCarree(),
                                 cmap=cmap, vmin=current_vmin, vmax=current_vmax, levels=15, extend='both', zorder=1)
        else:
            print(f"    Skipping plot for '{title}' due to invalid/problematic vmin/vmax range: vmin={current_vmin:.4g}, vmax={current_vmax:.4g}")
            ax.text(0.5, 0.5, "Data Range Issue", ha='center', va='center', transform=ax.transAxes, color='orange', fontsize=8)

    except Exception as e:
        print(f"    EXCEPTION during plotting for '{title}': {e}")
        traceback.print_exc()
        ax.text(0.5, 0.5, "Plotting Error", ha='center', va='center', transform=ax.transAxes, color='red', fontsize=8)

    ax.set_title(title, fontsize=9)
    return im

# --- Revised get_1d_coords_and_shape_from_h5 ---
def get_1d_coords_and_shape_from_h5(lat_arr_h5, lon_arr_h5, expected_data_shape, arr_name=""):
    print(f"--- get_1d_coords_and_shape_from_h5 for {arr_name} ---")
    print(f"    Input HDF5 coord shapes: lat_arr_h5: {lat_arr_h5.shape}, lon_arr_h5: {lon_arr_h5.shape}")
    print(f"    Expected data_shape for this grid: {expected_data_shape}")

    if not (lat_arr_h5.ndim == lon_arr_h5.ndim):
         raise ValueError(f"Mismatch in HDF5 coordinate array dimensions for {arr_name}: lat {lat_arr_h5.ndim}D, lon {lon_arr_h5.ndim}D")

    if lat_arr_h5.ndim == 1: # Assumes HDF5 coordinates are already 1D axes
        lats_1d = lat_arr_h5
        lons_1d = lon_arr_h5
        if lats_1d.size != expected_data_shape[0] or lons_1d.size != expected_data_shape[1]:
            print(f"    CRITICAL WARNING for {arr_name}: 1D HDF5 coord array sizes (lats:{lats_1d.size}, lons:{lons_1d.size}) "
                  f"MISMATCH expected_data_shape {expected_data_shape}. This will cause plotting errors.")
        # Return them as is, assuming they are correct. Sorting might be desired if not already sorted.
        print(f"    Using 1D HDF5 coords directly. lons: {lons_1d.size}, lats: {lats_1d.size}.")
        return lons_1d, lats_1d, expected_data_shape # Return the data_shape it's for

    elif lat_arr_h5.ndim == 2: # Assumes HDF5 coordinates are 2D meshgrid-like
        if lat_arr_h5.shape != expected_data_shape or lon_arr_h5.shape != expected_data_shape:
            print(f"    CRITICAL WARNING for {arr_name}: 2D HDF5 coord array shapes (lat:{lat_arr_h5.shape}, lon:{lon_arr_h5.shape}) "
                  f"MISMATCH expected_data_shape {expected_data_shape}.")
            # This case is problematic; cannot reliably extract axes if coord array shape itself is wrong.
            # As a desperate fallback, try to use expected_data_shape to slice, but this is a guess.
            lats_1d = lat_arr_h5[:expected_data_shape[0], 0]
            lons_1d = lon_arr_h5[0, :expected_data_shape[1]]
        else:
            # Standard assumption: data is (Nlat, Nlon)
            # Latitudes vary along the first axis (rows), so take the first column.
            lats_1d = lat_arr_h5[:, 0]
            # Longitudes vary along the second axis (columns), so take the first row.
            lons_1d = lon_arr_h5[0, :]

        # Verify that the extracted 1D arrays have the correct lengths
        if lats_1d.size != expected_data_shape[0]:
            print(f"    WARNING for {arr_name}: Extracted lats_1d size ({lats_1d.size}) != expected_data_shape[0] ({expected_data_shape[0]}). May use unique values.")
            lats_1d = np.unique(lat_arr_h5.ravel()) # Fallback to all unique lats if slicing fails
            if lats_1d.size != expected_data_shape[0]:
                 print(f"    STILL MISMATCH for {arr_name} lats after unique: {lats_1d.size} vs {expected_data_shape[0]}")


        if lons_1d.size != expected_data_shape[1]:
            print(f"    WARNING for {arr_name}: Extracted lons_1d size ({lons_1d.size}) != expected_data_shape[1] ({expected_data_shape[1]}). May use unique values.")
            lons_1d = np.unique(lon_arr_h5.ravel()) # Fallback to all unique lons
            if lons_1d.size != expected_data_shape[1]:
                 print(f"    STILL MISMATCH for {arr_name} lons after unique: {lons_1d.size} vs {expected_data_shape[1]}")


        print(f"    Extracted from 2D: lons_1d ({lons_1d.size}), lats_1d ({lats_1d.size}). Target data_shape: {expected_data_shape}")
        # It's usually good practice to sort them for contourf/pcolormesh,
        # though meshgrid itself doesn't require sorted input.
        return np.sort(lons_1d), np.sort(lats_1d), expected_data_shape
    else:
        raise ValueError(f"Unsupported HDF5 lat/lon grid dimensions for {arr_name}: {lat_arr_h5.ndim}D")


# --- Main plotting function ---
def plot_input_vs_interpolated(
    h5_filepath,
    interpolated_filename_prefix,
    times_to_plot,
    data_types_to_plot,
    dt_inputs,
    noon_longitude=0,
    output_filename=None
):
    print(f"--- Starting plot_input_vs_interpolated ---")
    # ... (initial prints as before) ...

    try:
        h5file = h5.File(h5_filepath, "r")
    except Exception as e: # ... (error handling) ...
        print(f"    CRITICAL ERROR: Error opening HDF5 file {h5_filepath}: {e}")
        return

    try: # Determine shapes of data from HDF5
        bu_data_shape = h5file["Bu"][0,:,:].shape
        fac_data_shape = h5file["FAC"][0,:,:].shape
        sh_data_shape = h5file["SH"][0,:,:].shape # Assuming SH, SP, We, Wn use iono grid
        # If SH, SP, We, Wn can have different shapes than FAC, define them separately
        # For now, assume ionospheric quantities (FAC, SH, SP, We, Wn) share one grid structure from HDF5
        ionospheric_data_shape = fac_data_shape # Pick one as representative
        num_h5_steps = h5file["Bu"].shape[0] # Get total steps
    except KeyError as ke:
        print(f"    CRITICAL ERROR: HDF5 file missing essential dataset: {ke}. Cannot determine data shapes or step count.")
        h5file.close()
        return
    print(f"    HDF5 file contains {num_h5_steps} time steps.")
    print(f"    Magnetospheric data (Bu) shape from HDF5: {bu_data_shape}")
    print(f"    Ionospheric data (FAC, SH, etc.) shape from HDF5: {ionospheric_data_shape}")

    io = IO(interpolated_filename_prefix)
    settings = io.load_dataset("settings", print_info=True)

    mainfield = Mainfield(
        kind=settings.mainfield_kind,
        epoch=settings.mainfield_epoch,
        hI=(settings.RI - RE) * 1e-3,
        B0=None if settings.mainfield_B0 == 0 else settings.mainfield_B0,
    )

    cs_basis = CSBasis(settings.Ncs)

    sh_basis = SHBasis(settings.Nmax, settings.Mmax, Nmin=0)
    sh_basis_zero_removed = SHBasis(settings.Nmax, settings.Mmax)

    # Specify input format and load input data.
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

    raw_ionosphere_lat_h5 = h5file["glat"][:]
    raw_ionosphere_lon_h5 = h5file["glon"][:]
    raw_magnetosphere_lat_h5 = h5file["Blat"][:]
    raw_magnetosphere_lon_h5 = h5file["Blon"][:]

    # Get 1D plot coords based on the *actual data shapes*
    ionosphere_lon_1d_plot, ionosphere_lat_1d_plot, _ = \
        get_1d_coords_and_shape_from_h5(raw_ionosphere_lat_h5, raw_ionosphere_lon_h5, ionospheric_data_shape, "Ionosphere")
    magnetosphere_lon_1d_plot, magnetosphere_lat_1d_plot, _ = \
        get_1d_coords_and_shape_from_h5(raw_magnetosphere_lat_h5, raw_magnetosphere_lon_h5, bu_data_shape, "Magnetosphere")

    ionosphere_lat_flat = raw_ionosphere_lat_h5.flatten() # Used for creating pynamit.Grid for evaluators
    ionosphere_lon_flat = raw_ionosphere_lon_h5.flatten()
    magnetosphere_lat_flat = raw_magnetosphere_lat_h5.flatten()
    magnetosphere_lon_flat = raw_magnetosphere_lon_h5.flatten()

    ionosphere_input_grid_for_br = Grid(lat=ionosphere_lat_flat, lon=ionosphere_lon_flat)
    FAC_b_evaluator_for_input_jr = FieldEvaluator(
        mainfield,
        ionosphere_input_grid_for_br,
        settings.RI
    )
    br_on_iono_input_grid_flat = FAC_b_evaluator_for_input_jr.br
    # Reshape using the known ionospheric_data_shape
    br_on_iono_input_grid_2d = br_on_iono_input_grid_flat.reshape(ionospheric_data_shape)


    num_rows = len(times_to_plot)
    num_cols = len(data_types_to_plot) * 2
    # ... (figure setup as before, using squeeze=False for axes) ...
    fig_width = min(max(10, num_cols * 4.0), 40) 
    fig_height = min(max(7, num_rows * 3.5), 35) 
    
    print(f"    Creating figure: {num_rows} rows, {num_cols} columns. Size: ({fig_width}, {fig_height})")
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(fig_width, fig_height),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=noon_longitude)},
        squeeze=False 
    )

    for row_idx, time_val_secs in enumerate(times_to_plot):
        # ... (time loop and step_idx calculation as before) ...
        print(f"\n--- Processing plot row {row_idx+1}/{num_rows} for time_val_secs = {time_val_secs}s ---")
        target_step_idx_float = time_val_secs / dt_inputs
        step_idx = int(round(target_step_idx_float))

        if step_idx < 0 or step_idx >= num_h5_steps: # ... (skip logic) ...
            print(f"    Warning: Requested plot time {time_val_secs}s -> HDF5 index {step_idx} (out of bounds [0, {num_h5_steps-1}]). Skipping row.")
            for col_idx_to_hide in range(num_cols):
                if num_rows > 0 and num_cols > 0 : axes[row_idx, col_idx_to_hide].set_visible(False)
            continue
        
        actual_h5_sim_time = step_idx * dt_inputs # ... (print warning if not close) ...
        if not np.isclose(actual_h5_sim_time, time_val_secs, atol=dt_inputs/1.9):
            print(f"    Note: Requested plot time {time_val_secs}s. Using HDF5 input from sim_time {actual_h5_sim_time}s (HDF5 index {step_idx}).")

        if num_cols > 0: axes[row_idx, 0].set_ylabel(f"{time_val_secs}s", fontsize=10, labelpad=35, rotation=0, ha='right', va='center')
            
        for data_type_idx, data_type_str in enumerate(data_types_to_plot):
            # ... (column setup as before) ...
            print(f"  -- Processing data type '{data_type_str}' (column pair {data_type_idx*2}, {data_type_idx*2+1}) --")
            col_idx_input = data_type_idx * 2
            col_idx_interpolated = col_idx_input + 1
            ax_input = axes[row_idx, col_idx_input]
            ax_interpolated = axes[row_idx, col_idx_interpolated]

            input_data_2d, interpolated_data_2d = None, None
            # These will now be set based on data_type_str
            current_lon_1d_plot, current_lat_1d_plot, current_data_shape_for_plot = None, None, None
            # plotting_grid_flat_coords for BasisEvaluator target grid
            current_plotting_grid_flat_coords = None 
            data_label, cmap = data_type_str, 'viridis'

            if data_type_str == 'Br':
                input_data_2d = h5file["Bu"][:][step_idx, :, :] * 1e-9
                current_lon_1d_plot, current_lat_1d_plot = magnetosphere_lon_1d_plot, magnetosphere_lat_1d_plot
                current_data_shape_for_plot = bu_data_shape # Use the known data shape
                current_plotting_grid_flat_coords = (magnetosphere_lat_flat, magnetosphere_lon_flat)
                data_label, cmap = r'$\Delta B_r$ [T]', 'bwr'
            elif data_type_str == 'jr':
                FAC_input_2d = h5file["FAC"][:][step_idx, :, :] * 1e-6
                input_data_2d = FAC_input_2d * br_on_iono_input_grid_2d # br_on_iono_input_grid_2d already has ionospheric_data_shape
                current_lon_1d_plot, current_lat_1d_plot = ionosphere_lon_1d_plot, ionosphere_lat_1d_plot
                current_data_shape_for_plot = ionospheric_data_shape # Use known data shape
                current_plotting_grid_flat_coords = (ionosphere_lat_flat, ionosphere_lon_flat)
                data_label, cmap = r'$j_r$ [A/m$^2$]', 'bwr'
            elif data_type_str == 'SH':
                input_data_2d = h5file["SH"][:][step_idx, :, :]
                current_lon_1d_plot, current_lat_1d_plot = ionosphere_lon_1d_plot, ionosphere_lat_1d_plot
                current_data_shape_for_plot = ionospheric_data_shape # Assuming SH uses iono grid
                current_plotting_grid_flat_coords = (ionosphere_lat_flat, ionosphere_lon_flat)
                data_label = r'$\Sigma_H$ [S]'
            elif data_type_str == 'SP':
                input_data_2d = h5file["SP"][:][step_idx, :, :]
                current_lon_1d_plot, current_lat_1d_plot = ionosphere_lon_1d_plot, ionosphere_lat_1d_plot
                current_data_shape_for_plot = ionospheric_data_shape # Assuming SP uses iono grid
                current_plotting_grid_flat_coords = (ionosphere_lat_flat, ionosphere_lon_flat)
                data_label = r'$\Sigma_P$ [S]'
            elif data_type_str in ['u_mag', 'u_theta', 'u_phi']:
                u_east_input = h5file["We"][:][step_idx, :, :]
                u_north_input = h5file["Wn"][:][step_idx, :, :]
                _u_theta_in_2d, _u_phi_in_2d = -u_north_input, u_east_input
                current_lon_1d_plot, current_lat_1d_plot = ionosphere_lon_1d_plot, ionosphere_lat_1d_plot
                current_data_shape_for_plot = ionospheric_data_shape # Assuming We/Wn use iono grid
                current_plotting_grid_flat_coords = (ionosphere_lat_flat, ionosphere_lon_flat)
                if data_type_str == 'u_mag': input_data_2d = np.sqrt(_u_theta_in_2d**2 + _u_phi_in_2d**2)
                elif data_type_str == 'u_theta': input_data_2d = _u_theta_in_2d
                elif data_type_str == 'u_phi': input_data_2d = _u_phi_in_2d
                # Update data_label, cmap for u an d u_phi...
                if data_type_str == 'u_mag': data_label = r'$|u|$ [m/s]'
                elif data_type_str == 'u_theta': data_label, cmap = r'$u_\theta$ (South) [m/s]', 'bwr'
                elif data_type_str == 'u_phi': data_label, cmap = r'$u_\phi$ (East) [m/s]', 'bwr'
            else: # ... (skip logic) ...
                print(f"    ERROR: Unknown input data_type: {data_type_str}. Skipping this data type.")
                ax_input.set_visible(False); ax_interpolated.set_visible(False)
                continue
            
            print(f"    Input data for '{data_type_str}' loaded, HDF5 shape: {input_data_2d.shape if input_data_2d is not None else 'None'}. "
                  f"Target plot data shape for this grid: {current_data_shape_for_plot}")

            # --- 4b. Get Interpolated (Fitted) Data using FieldExpansion ---
            timeseries_key_map = { # ... (as before) ...
                'Br': 'Br', 'jr': 'jr', 'SH': 'conductance', 'SP': 'conductance',
                'u_mag': 'u', 'u_theta': 'u', 'u_phi': 'u'
            }
            timeseries_key = timeseries_key_map.get(data_type_str)
            if timeseries_key: # ... (logic for timeseries_entry as before) ...
                print(f"    Attempting to get fitted data for '{timeseries_key}' at sim_time {time_val_secs}s")
                timeseries_entry = input_timeseries.get_entry_if_changed(
                    timeseries_key, time_val_secs, interpolation=True
                )
                if timeseries_entry: # ... (process timeseries_entry as before) ...
                    print(f"    Timeseries entry found for '{timeseries_key}'. Keys: {list(timeseries_entry.keys())}")
                    storage_basis = input_timeseries.storage_bases[timeseries_key]
                    # Use the flat coordinates of the grid this data type belongs to
                    target_plot_grid = Grid(lat=current_plotting_grid_flat_coords[0], lon=current_plotting_grid_flat_coords[1])
                    plot_evaluator = BasisEvaluator(storage_basis, target_plot_grid)

                    if timeseries_key == 'conductance': # ... (as before) ...
                        if 'etaP' in timeseries_entry and 'etaH' in timeseries_entry:
                            etaP_coeffs = timeseries_entry['etaP']
                            etaH_coeffs = timeseries_entry['etaH']
                            field_exp_etaP = FieldExpansion(storage_basis, coeffs=etaP_coeffs, field_type='scalar')
                            field_exp_etaH = FieldExpansion(storage_basis, coeffs=etaH_coeffs, field_type='scalar')
                            etaP_fitted_flat = field_exp_etaP.to_grid(plot_evaluator)
                            etaH_fitted_flat = field_exp_etaH.to_grid(plot_evaluator)
                            # Reshape to the known data shape for this grid type
                            etaP_fitted_2d = etaP_fitted_flat.reshape(current_data_shape_for_plot)
                            etaH_fitted_2d = etaH_fitted_flat.reshape(current_data_shape_for_plot)
                            
                            denominator = etaP_fitted_2d**2 + etaH_fitted_2d**2 # ... (rest of calc) ...
                            sigma_H_fitted_2d = np.zeros_like(etaH_fitted_2d)
                            sigma_P_fitted_2d = np.zeros_like(etaP_fitted_2d)
                            valid_den = denominator > 1e-12
                            sigma_H_fitted_2d[valid_den] = etaH_fitted_2d[valid_den] / denominator[valid_den]
                            sigma_P_fitted_2d[valid_den] = etaP_fitted_2d[valid_den] / denominator[valid_den]
                            if data_type_str == 'SH': interpolated_data_2d = sigma_H_fitted_2d
                            elif data_type_str == 'SP': interpolated_data_2d = sigma_P_fitted_2d
                        else: print(f"    Fitted Warning: etaP/etaH coeffs not found in timeseries_entry for '{timeseries_key}' at {time_val_secs}s.")
                    
                    elif timeseries_key == 'u': # ... (as before) ...
                        if 'u' in timeseries_entry:
                            u_coeffs_helmholtz = timeseries_entry['u']
                            field_exp_u = FieldExpansion(storage_basis, coeffs=u_coeffs_helmholtz, field_type='tangential')
                            u_theta_flat, u_phi_flat = field_exp_u.to_grid(plot_evaluator)
                            _u_theta_fit_2d = u_theta_flat.reshape(current_data_shape_for_plot)
                            _u_phi_fit_2d = u_phi_flat.reshape(current_data_shape_for_plot)
                            if data_type_str == 'u_mag': interpolated_data_2d = np.sqrt(_u_theta_fit_2d**2 + _u_phi_fit_2d**2)
                            elif data_type_str == 'u_theta': interpolated_data_2d = _u_theta_fit_2d
                            elif data_type_str == 'u_phi': interpolated_data_2d = _u_phi_fit_2d
                        else: print(f"    Fitted Warning: u coeffs not found in timeseries_entry for '{timeseries_key}' at {time_val_secs}s.")

                    else: # Br or jr ... (as before) ...
                        if data_type_str in timeseries_entry:
                            field_coeffs = timeseries_entry[data_type_str]
                            field_exp = FieldExpansion(storage_basis, coeffs=field_coeffs, field_type='scalar')
                            interpolated_data_flat = field_exp.to_grid(plot_evaluator)
                            interpolated_data_2d = interpolated_data_flat.reshape(current_data_shape_for_plot)
                        else: print(f"    Fitted Warning: {data_type_str} coeffs not found in timeseries_entry for '{timeseries_key}' at {time_val_secs}s.")
                    
                    if interpolated_data_2d is not None:
                         print(f"    Fitted data for '{data_type_str}' processed, shape: {interpolated_data_2d.shape}")
                    else:
                         print(f"    Fitted data for '{data_type_str}' is None after processing.")
                else:
                    print(f"    Fitted Warning: No timeseries entry for '{timeseries_key}' at {time_val_secs}s. Check if dynamics.input_timeseries is populated for this time.")

            # --- 4c. Plotting ---
            # (Shared vmin/vmax logic as before)
            vmin_shared, vmax_shared = None, None 
            if input_data_2d is not None and interpolated_data_2d is not None:
                # ... (vmin/vmax calculation as before) ...
                d1_flat_valid = input_data_2d.astype(float).ravel()
                d1_flat_valid = d1_flat_valid[~np.isnan(d1_flat_valid)]
                d2_flat_valid = interpolated_data_2d.astype(float).ravel()
                d2_flat_valid = d2_flat_valid[~np.isnan(d2_flat_valid)]
                
                combined_valid_data = []
                if d1_flat_valid.size > 0: combined_valid_data.append(d1_flat_valid)
                if d2_flat_valid.size > 0: combined_valid_data.append(d2_flat_valid)

                if combined_valid_data:
                    combined_valid_data = np.concatenate(combined_valid_data)
                    if combined_valid_data.size > 0:
                        if cmap == 'bwr':
                            abs_max_s = np.percentile(np.abs(combined_valid_data), 99.8)
                            vmin_shared, vmax_shared = -abs_max_s if abs_max_s > 1e-9 else -0.1, abs_max_s if abs_max_s > 1e-9 else 0.1
                        else:
                            vmin_shared = np.percentile(combined_valid_data, 0.2)
                            vmax_shared = np.percentile(combined_valid_data, 99.8)
                        print(f"    Shared scale for '{data_label}': vmin={vmin_shared:.4g}, vmax={vmax_shared:.4g}")
                    else: print(f"    Warning: No valid data in combined input/fitted for '{data_label}' to set shared scale.")
                else: print(f"    Warning: Both input and fitted data for '{data_label}' are all NaN or empty.")

            plot_title_input = f"Input {data_label}" if row_idx == 0 else "Input"
            plot_title_fitted = f"Fitted {data_label}" if row_idx == 0 else "Fitted"

            if input_data_2d is not None:
                # Ensure shape of input_data_2d matches current_data_shape_for_plot
                if input_data_2d.shape != current_data_shape_for_plot:
                    print(f"    CRITICAL WARNING (Input Plot): input_data_2d shape {input_data_2d.shape} "
                          f"MISMATCHES current_data_shape_for_plot {current_data_shape_for_plot} for {plot_title_input}")
                    ax_input.text(0.5,0.5, "Input Data Shape Error", color='red', ha='center', va='center', transform=ax_input.transAxes)
                else:
                    plot_scalar_map_on_ax(ax_input, current_lon_1d_plot, current_lat_1d_plot, input_data_2d,
                                          plot_title_input, cmap, vmin_shared, vmax_shared, use_pcolormesh_for_debug=True) # DEBUG: pcolormesh for input
            else: # ... (hide plot) ...
                print(f"    Input data for '{data_label}' is None. Hiding input plot.")
                ax_input.set_visible(False)


            if interpolated_data_2d is not None:
                 if interpolated_data_2d.shape != current_data_shape_for_plot:
                    print(f"    CRITICAL WARNING (Fitted Plot): interpolated_data_2d shape {interpolated_data_2d.shape} "
                          f"MISMATCHES current_data_shape_for_plot {current_data_shape_for_plot} for {plot_title_fitted}")
                    ax_interpolated.text(0.5,0.5, "Fitted Data Shape Error", color='red', ha='center', va='center', transform=ax_interpolated.transAxes)
                 else:
                    plot_scalar_map_on_ax(ax_interpolated, current_lon_1d_plot, current_lat_1d_plot, interpolated_data_2d,
                                      plot_title_fitted, cmap, vmin_shared, vmax_shared, use_pcolormesh_for_debug=False) # contourf for fitted
            else: # ... (hide plot) ...
                print(f"    Interpolated (fitted) data for '{data_label}' is None. Hiding fitted plot.")
                ax_interpolated.set_visible(False)


    # --- 5. Finalize Figure ---
    # ... (finalize figure as before) ...
    print(f"\n--- Finalizing Figure ---")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.05, hspace=0.4, wspace=0.15)
    if num_rows > 0 and num_cols > 0 and any(dtp in data_types_to_plot for dtp in ['Br','jr','u_theta','u_phi']):
         fig.suptitle(f"Input vs. Fitted Data (bwr plots use shared vmin/vmax per pair, 'bwr' centered; other cmaps use 0.2-99.8 percentile)", fontsize=12)

    if output_filename: # ... (save logic) ...
        print(f"    Attempting to save figure to: {output_filename}")
        try:
            plt.savefig(output_filename, dpi=200, bbox_inches='tight') 
            print(f"    SUCCESS: Figure saved to {output_filename}")
        except Exception as e_save:
            print(f"    CRITICAL ERROR during plt.savefig: {e_save}")
            traceback.print_exc()
    else: # ... (show logic) ...
        print("    Output filename not provided. Attempting plt.show().")
        try:
            plt.show()
            print("    plt.show() executed.")
        except Exception as e_show:
            print(f"    ERROR during plt.show(): {e_show}")
            traceback.print_exc()
            
    h5file.close()
    print("--- plot_input_vs_interpolated finished ---")