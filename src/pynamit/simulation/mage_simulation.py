import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
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


# --- Helper for rounding color limits (BWR part significantly reworked again) ---
def get_rounded_nice_limits(vmin_orig, vmax_orig, cmap_global, sig_figs_for_step=1): # sig_figs_for_step is mainly for non-bwr now
    if vmin_orig > vmax_orig: vmin_orig, vmax_orig = vmax_orig, vmin_orig
    if np.isclose(vmin_orig, vmax_orig):
        # Adaptive delta based on magnitude, with a minimum for very small/zero values
        delta_abs = abs(vmin_orig)
        if delta_abs < 1e-10:  # If original value is extremely small or zero
            delta = 1e-9       # Use a small absolute delta
        elif delta_abs < 1e-2: # For small values, a slightly larger relative delta might be needed
            delta = delta_abs * 0.2 + 1e-9 # Mix of relative and absolute
        else:                  # For larger values, 10% relative is usually fine
            delta = delta_abs * 0.1
        
        vmin_orig -= delta
        vmax_orig += delta
        # Ensure they are different after adjustment
        if np.isclose(vmin_orig, vmax_orig): 
            final_kick = 10**(np.floor(np.log10(abs(vmax_orig))) -2) if not np.isclose(vmax_orig,0) else 1e-9
            vmin_orig -= final_kick
            vmax_orig += final_kick
    
    vmin_r, vmax_r = vmin_orig, vmax_orig

    if cmap_global == "bwr":
        abs_max_val = max(abs(vmin_orig), abs(vmax_orig))

        if abs_max_val < 1e-12: # If effectively zero
            return -1e-9, 1e-9

        # Manual "nice number" rounding for bwr, especially for small abs_max_val
        if abs_max_val > 0:
            exponent = np.floor(np.log10(abs_max_val))
            mantissa = abs_max_val / (10**exponent) # Should be between 1.0 and <10.0

            # Round mantissa UP to a "nice" value (e.g., 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 10)
            # This list can be adjusted for desired "roundness"
            nice_mantissas = [1.0, 1.2, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 9.0, 10.0]
            
            rounded_mantissa = mantissa # Default if not found below
            for r_m in nice_mantissas:
                if mantissa <= r_m:
                    rounded_mantissa = r_m
                    break
            # If mantissa was > largest nice_mantissa (e.g. 10), it would already be rounded_mantissa=10, exponent+1
            if rounded_mantissa >= 9.999 and mantissa < 9.999 : # e.g. mantissa was 9.1, rounded to 10
                 pass # This is fine, will become 1.0 * 10**(exponent+1)
            
            nice_abs_max = rounded_mantissa * (10**exponent)

            # Sanity check: if nice_abs_max is still much larger than abs_max_val (e.g. >5x for tiny numbers)
            # This can happen if abs_max_val is just above a very small nice_mantissa
            if abs_max_val < 1e-6 and nice_abs_max / abs_max_val > 5:
                 # Try rounding to 1 significant figure of abs_max_val itself more directly
                 power_of_10_to_round = 10**np.floor(np.log10(abs_max_val))
                 nice_abs_max = np.ceil(abs_max_val / power_of_10_to_round) * power_of_10_to_round


        else: # abs_max_val is 0
            nice_abs_max = 1e-9 # Default if input was truly 0
        
        vmin_r, vmax_r = -nice_abs_max, nice_abs_max
        # print(f"DEBUG BWR: abs_max_val={abs_max_val:.3e}, nice_abs_max={nice_abs_max:.3e}, vmin_r={vmin_r:.3e}, vmax_r={vmax_r:.3e}")

    else: # Not 'bwr' (existing logic, ensure sig_figs_for_step is positive)
        effective_sig_figs = max(1, sig_figs_for_step)
        data_range = vmax_orig - vmin_orig; locator = mticker.MaxNLocator(nbins=5, prune='both')
        try:
            ticks_in_range = locator.tick_values(vmin_orig, vmax_orig)
            if len(ticks_in_range) >= 2: step = ticks_in_range[1] - ticks_in_range[0]
            elif data_range > 1e-9: step = 10**(np.floor(np.log10(data_range)) - effective_sig_figs)
            else: step = 10**(np.floor(np.log10(abs(vmin_orig) if not np.isclose(vmin_orig,0) else 1.0)) - effective_sig_figs)
            
            if np.isclose(step, 0) or step < 1e-12: # Prevent extremely small or zero steps
                 min_sensible_step_abs = abs(vmin_orig * 1e-3) if not np.isclose(vmin_orig, 0) else 1e-9
                 min_sensible_step_range = abs(data_range * 1e-2) if not np.isclose(data_range, 0) else 1e-9
                 step = max(min_sensible_step_abs, min_sensible_step_range, 1e-9)

            vmin_r = np.floor(vmin_orig / step) * step; vmax_r = np.ceil(vmax_orig / step) * step
            if np.isclose(vmin_r, vmax_r): vmax_r += step;
            if np.isclose(vmin_r, vmax_r): vmin_r -=step 
        except Exception: # Fallback
            if data_range > 1e-9: decimals = -int(np.floor(np.log10(data_range))) + effective_sig_figs
            else: decimals = int(abs(np.floor(np.log10(abs(vmin_orig))))) + effective_sig_figs + 2 if not np.isclose(vmin_orig,0) else 9
            vmin_r = np.round(vmin_orig - data_range*0.05, decimals); vmax_r = np.round(vmax_orig + data_range*0.05, decimals)

    # Final safety check for all cmaps
    if np.isclose(vmin_r, vmax_r) or vmin_r > vmax_r:
        # print(f"Warning: Final safety check triggered for {cmap_global}. vmin_r={vmin_r:.2e}, vmax_r={vmax_r:.2e}. Originals: vmin={vmin_orig:.2e}, vmax={vmax_orig:.2e}")
        v_center = (vmin_orig + vmax_orig) / 2.0; original_half_range = abs(vmax_orig - vmin_orig) / 2.0
        
        # Determine padding adaptively based on the original range or value
        if original_half_range > 1e-7: padding = original_half_range * 0.1
        elif abs(v_center) > 1e-7: padding = abs(v_center) * 0.1
        else: padding = 1e-9 # Default small padding if range and center are tiny
        padding = max(padding, 1e-9) # Ensure padding is at least a very small number

        vmin_r = v_center - (original_half_range + padding); vmax_r = v_center + (original_half_range + padding)
        
        final_range = vmax_r - vmin_r
        if final_range > 1e-9: 
            # Try to get 2-3 significant figures for the limits
            decimals = -int(np.floor(np.log10(final_range))) + 1 
        else: 
            decimals = int(abs(np.floor(np.log10(abs(vmin_r))))) + 2 if not np.isclose(vmin_r,0) else 9
        
        vmin_r = np.round(vmin_r, max(0,decimals)); vmax_r = np.round(vmax_r, max(0,decimals)) # Ensure non-negative decimals
        
        if np.isclose(vmin_r, vmax_r): # Ultimate fallback if still too close
            smallest_step = 10**(-decimals if decimals < 10 else -9)
            vmin_r -= smallest_step
            vmax_r += smallest_step
            # print(f"Post-ultimate fallback: vmin_r={vmin_r:.2e}, vmax_r={vmax_r:.2e}")
            
    return vmin_r, vmax_r


# --- Helper plotting function (plot_scalar_map_on_ax - with refined gridline flags) ---
def plot_scalar_map_on_ax(
    ax, lon_coords_2d, lat_coords_2d, data_2d_arr, title="", cmap="viridis",
    vmin=None, vmax=None, use_pcolormesh=False,
    draw_top_labels=False, draw_right_labels=False # New flags for gridlines
):
    ax.coastlines(color="grey", zorder=3, linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = draw_top_labels
    gl.right_labels = draw_right_labels
    gl.left_labels = False 
    gl.bottom_labels = False 
    
    plot_transform = ccrs.PlateCarree() 
    current_vmin, current_vmax = vmin, vmax
    if vmin is None or vmax is None:
        valid_data = data_2d_arr[~np.isnan(data_2d_arr)]
        if valid_data.size == 0: auto_vmin, auto_vmax = 0.0, 1.0
        elif cmap == "bwr": abs_max_val = np.percentile(np.abs(valid_data), 99.5); auto_vmin, auto_vmax = (-abs_max_val if abs_max_val > 1e-9 else -0.1, abs_max_val if abs_max_val > 1e-9 else 0.1)
        else: auto_vmin, auto_vmax = np.percentile(valid_data, 0.5), np.percentile(valid_data, 99.5)
        if vmin is None: current_vmin = auto_vmin
        if vmax is None: current_vmax = auto_vmax
    if not (isinstance(current_vmin, (int, float, np.number)) and np.isfinite(current_vmin) and
            isinstance(current_vmax, (int, float, np.number)) and np.isfinite(current_vmax) and
            current_vmin < current_vmax):
        valid_data = data_2d_arr[~np.isnan(data_2d_arr)]
        if valid_data.size > 0 : current_vmin, current_vmax = np.min(valid_data), np.max(valid_data)
        else: current_vmin, current_vmax = 0.0, 1.0
        if np.isclose(current_vmin, current_vmax) or current_vmin > current_vmax:
            delta_abs = abs(current_vmin)
            if delta_abs < 1e-10: delta = 1e-9
            else: delta = delta_abs * 0.1
            current_vmin -= delta; current_vmax += delta
            if np.isclose(current_vmin, current_vmax): current_vmin -=1e-9; current_vmax +=1e-9
    data_to_plot_masked = np.ma.masked_invalid(data_2d_arr)
    im = None
    if use_pcolormesh:
        im = ax.pcolormesh(lon_coords_2d, lat_coords_2d, data_to_plot_masked, cmap=cmap, vmin=current_vmin, vmax=current_vmax, transform=plot_transform, shading="auto", zorder=1)
    else:
        im = ax.contourf(lon_coords_2d, lat_coords_2d, data_to_plot_masked, transform=plot_transform, cmap=cmap, vmin=current_vmin, vmax=current_vmax, levels=15, extend="both", zorder=1)
    if title: ax.set_title(title, fontsize=9)
    return im

# --- Helper functions for pynamit data calculation (_calculate_interpolated_... - unchanged) ---
def _calculate_interpolated_scalar_field(timeseries_entry, data_key, storage_basis, plot_evaluator, target_shape):
    coeffs = timeseries_entry.get(data_key);
    if coeffs is None: return np.full(target_shape, np.nan)
    try: field_exp = FieldExpansion(storage_basis, coeffs=coeffs, field_type="scalar"); return field_exp.to_grid(plot_evaluator).reshape(target_shape)
    except Exception: return np.full(target_shape, np.nan)
def _calculate_interpolated_conductance(timeseries_entry, component, storage_basis, plot_evaluator, target_shape):
    etaP_coeffs, etaH_coeffs = timeseries_entry.get("etaP"), timeseries_entry.get("etaH");
    if etaP_coeffs is None or etaH_coeffs is None: return np.full(target_shape, np.nan)
    try:
        etaP, etaH = FieldExpansion(storage_basis, coeffs=etaP_coeffs, field_type="scalar"), FieldExpansion(storage_basis, coeffs=etaH_coeffs, field_type="scalar")
        etaP_f, etaH_f = etaP.to_grid(plot_evaluator).reshape(target_shape), etaH.to_grid(plot_evaluator).reshape(target_shape)
        den = etaP_f**2 + etaH_f**2; sH_f, sP_f = np.zeros_like(etaH_f), np.zeros_like(etaP_f)
        valid = den > 1e-12; sH_f[valid] = etaH_f[valid] / den[valid]; sP_f[valid] = etaP_f[valid] / den[valid]
        return sH_f if component == "SH" else sP_f
    except Exception: return np.full(target_shape, np.nan)
def _calculate_interpolated_u_field(timeseries_entry, component, storage_basis, plot_evaluator, target_shape):
    u_coeffs = timeseries_entry.get("u");
    if u_coeffs is None: return np.full(target_shape, np.nan)
    try:
        u = FieldExpansion(storage_basis, coeffs=u_coeffs.reshape((2, -1)), field_type="tangential"); u_t, u_p = u.to_grid(plot_evaluator)
        u_t_2d, u_p_2d = u_t.reshape(target_shape), u_p.reshape(target_shape)
        if component == "u_mag": return np.sqrt(u_t_2d**2 + u_p_2d**2)
        elif component == "u_theta": return u_t_2d
        elif component == "u_phi": return u_p_2d
    except Exception: return np.full(target_shape, np.nan)

# --- Main plotting function ---
def plot_input_vs_interpolated(
    h5_filepath, interpolated_filename_prefix, timesteps_to_plot, input_dt, data_types_to_plot,
    noon_longitude=0, output_filename=None, sig_figs_for_rounding_step=1, 
    vmin_percentile=0.2, vmax_percentile=99.8
):
    if not timesteps_to_plot: print("Warning: timesteps_to_plot is empty."); return
    if not data_types_to_plot: print("Warning: data_types_to_plot is empty."); return
    try: h5file = h5.File(h5_filepath, "r")
    except Exception as e: print(f"CRITICAL ERROR: Opening HDF5 {h5_filepath}: {e}"); return
    num_h5_steps = h5file["Bu"].shape[0]
    io = IO(interpolated_filename_prefix); settings = io.load_dataset("settings", print_info=False)
    if settings is None: print(f"CRITICAL ERROR: No 'settings' from '{interpolated_filename_prefix}'."); h5file.close(); return
    ri_value = float(settings.RI)
    mainfield = Mainfield(kind=str(settings.mainfield_kind), epoch=int(settings.mainfield_epoch), hI=(ri_value - RE) * 1e-3, B0=None if float(settings.mainfield_B0) == 0 else float(settings.mainfield_B0))
    cs_basis = CSBasis(int(settings.Ncs)); sh_basis = SHBasis(int(settings.Nmax), int(settings.Mmax), Nmin=0); sh_basis_zero_removed = SHBasis(int(settings.Nmax), int(settings.Mmax))
    input_vars_pynamit = {"jr": {"jr": "scalar"}, "Br": {"Br": "scalar"}, "conductance": {"etaP": "scalar", "etaH": "scalar"}, "u": {"u": "tangential"}}
    input_storage_bases = {"jr": sh_basis_zero_removed, "Br": sh_basis_zero_removed, "conductance": sh_basis, "u": sh_basis_zero_removed}
    pynamit_timeseries_key_map = {"Br": "Br", "jr": "jr", "SH": "conductance", "SP": "conductance", "u_mag": "u", "u_theta": "u", "u_phi": "u"}
    data_type_details = {
        "Br": {"label": r"$\Delta B_r$ [T]", "cmap": "bwr", "grid_type": "magnetosphere", "h5_key_primary": "Bu"},
        "jr": {"label": r"$j_r$ [A/m$^2$]", "cmap": "bwr", "grid_type": "ionosphere", "h5_key_primary": "FAC"},
        "SH": {"label": r"$\Sigma_H$ [S]", "cmap": "viridis", "grid_type": "ionosphere", "h5_key_primary": "SH"},
        "SP": {"label": r"$\Sigma_P$ [S]", "cmap": "viridis", "grid_type": "ionosphere", "h5_key_primary": "SP"},
        "u_mag": {"label": r"$|u|$ [m/s]", "cmap": "viridis", "grid_type": "ionosphere", "h5_key_primary": "We", "h5_key_secondary": "Wn"},
        "u_theta": {"label": r"$u_\theta$ (South) [m/s]", "cmap": "bwr", "grid_type": "ionosphere", "h5_key_primary": "We", "h5_key_secondary": "Wn"},
        "u_phi": {"label": r"$u_\phi$ (East) [m/s]", "cmap": "bwr", "grid_type": "ionosphere", "h5_key_primary": "We", "h5_key_secondary": "Wn"},}
    input_timeseries = Timeseries(cs_basis, input_storage_bases, input_vars_pynamit); input_timeseries.load_all(io)
    ionosphere_lat, ionosphere_lon = h5file["glat"][:], h5file["glon"][:]; magnetosphere_lat, magnetosphere_lon = h5file["Blat"][:], h5file["Blon"][:]
    ionosphere_grid = Grid(lat=ionosphere_lat, lon=ionosphere_lon); ionosphere_b_evaluator = FieldEvaluator(mainfield, ionosphere_grid, ri_value)
    ionosphere_br_2d = ionosphere_b_evaluator.br.reshape(ionosphere_lat.shape)

    print("Starting Pass 1: Collecting data for global vmin/vmax...")
    all_data_for_scaling = {dt_str: {'input': [], 'interpolated': []} for dt_str in data_types_to_plot}
    for timestep in timesteps_to_plot:
        if not (0 <= timestep < num_h5_steps): continue
        time_val = timestep * input_dt
        for data_type_str in data_types_to_plot:
            details = data_type_details[data_type_str]; pynamit_ts_key = pynamit_timeseries_key_map[data_type_str]
            _, _, target_shape = (magnetosphere_lon, magnetosphere_lat, magnetosphere_lat.shape) if details["grid_type"] == "magnetosphere" else (ionosphere_lon, ionosphere_lat, ionosphere_lat.shape)
            input_data_2d = np.full(target_shape, np.nan); h5_key_pri = details["h5_key_primary"]
            if data_type_str=="Br": input_data_2d = h5file[h5_key_pri][timestep,:,:] * 1e-9
            elif data_type_str=="jr": input_data_2d = (h5file[h5_key_pri][timestep,:,:] * 1e-6) * ionosphere_br_2d
            elif data_type_str in ["SH","SP"]: input_data_2d = h5file[h5_key_pri][timestep,:,:]
            elif data_type_str in ["u_mag","u_theta","u_phi"]: u_e,u_n=h5file[h5_key_pri][timestep,:,:],h5file[details["h5_key_secondary"]][timestep,:,:];
            if data_type_str=="u_mag":input_data_2d=np.sqrt(u_n**2+u_e**2)
            elif data_type_str=="u_theta":input_data_2d=-u_n
            elif data_type_str=="u_phi":input_data_2d=u_e
            if input_data_2d is not None: all_data_for_scaling[data_type_str]['input'].append(input_data_2d.ravel())
            interpolated_data_2d = np.full(target_shape, np.nan)
            timeseries_entry = input_timeseries.get_entry(pynamit_ts_key, time_val, interpolation=False)
            if timeseries_entry:
                storage_basis = input_timeseries.storage_bases[pynamit_ts_key]
                current_lon_eval, current_lat_eval = (magnetosphere_lon, magnetosphere_lat) if details["grid_type"] == "magnetosphere" else (ionosphere_lon, ionosphere_lat)
                plot_evaluator = BasisEvaluator(storage_basis, Grid(lat=current_lat_eval, lon=current_lon_eval))
                if pynamit_ts_key=="Br": interpolated_data_2d=_calculate_interpolated_scalar_field(timeseries_entry,"Br",storage_basis,plot_evaluator,target_shape)
                elif pynamit_ts_key=="jr": interpolated_data_2d=_calculate_interpolated_scalar_field(timeseries_entry,"jr",storage_basis,plot_evaluator,target_shape)
                elif pynamit_ts_key=="conductance": interpolated_data_2d=_calculate_interpolated_conductance(timeseries_entry,data_type_str,storage_basis,plot_evaluator,target_shape)
                elif pynamit_ts_key=="u": interpolated_data_2d=_calculate_interpolated_u_field(timeseries_entry,data_type_str,storage_basis,plot_evaluator,target_shape)
            if interpolated_data_2d is not None: all_data_for_scaling[data_type_str]['interpolated'].append(interpolated_data_2d.ravel())

    print("Calculating and rounding global vmin/vmax using specified percentiles...")
    global_plot_scales = {}
    for data_type_str in data_types_to_plot:
        details = data_type_details[data_type_str]; cmap_global = details['cmap']
        cat_input_all = all_data_for_scaling[data_type_str]['input']; cat_interp_all = all_data_for_scaling[data_type_str]['interpolated']
        combined_flat = np.concatenate( (np.concatenate(cat_input_all) if cat_input_all else np.array([]), np.concatenate(cat_interp_all) if cat_interp_all else np.array([])) )
        valid_data = combined_flat[~np.isnan(combined_flat)]
        vmin_orig, vmax_orig = 0.0, 1.0
        if valid_data.size > 0:
            if cmap_global == "bwr": abs_max_s = np.percentile(np.abs(valid_data), vmax_percentile); vmin_orig, vmax_orig = (-abs_max_s if abs_max_s > 1e-12 else -1e-9, abs_max_s if abs_max_s > 1e-12 else 1e-9)
            else: vmin_orig, vmax_orig = np.percentile(valid_data, vmin_percentile), np.percentile(valid_data, vmax_percentile)
        else: print(f"    Warning: No valid data for '{data_type_str}'. Using default [0,1].")
        
        vmin_rounded, vmax_rounded = get_rounded_nice_limits(vmin_orig, vmax_orig, cmap_global, sig_figs_for_step=sig_figs_for_rounding_step)
        
        global_plot_scales[data_type_str] = {'vmin': vmin_rounded, 'vmax': vmax_rounded, 'cmap': cmap_global}
        print(f"  Global scale for '{data_type_str}': vmin={vmin_rounded:.3e}, vmax={vmax_rounded:.3e} (from Pctl: {vmin_orig:.3e}, {vmax_orig:.3e})")

    num_dt = len(data_types_to_plot); num_ts = len(timesteps_to_plot)
    fig_plot_rows = num_dt * 2
    n_label_cols = 2; n_data_cols = num_ts; n_cbar_cols = 1; total_gs_cols = n_label_cols + n_data_cols + n_cbar_cols
    label_main_w, label_sub_w, cbar_w = 0.08, 0.08, 0.10
    data_plot_w_rel = 1.0
    base_w_abs, base_h_abs = 2.8, 2.2
    fig_width_est = (label_main_w + label_sub_w) * base_w_abs * 1.5 + n_data_cols * base_w_abs + cbar_w * base_w_abs * 1.5 + 0.5
    fig_height_est = num_dt * base_h_abs * 1.05 + 1.5
    fig_width = min(max(12, fig_width_est), 45); fig_height = min(max(8, fig_height_est), num_dt * 6)
    print(f"Creating figure. Size: ({fig_width:.1f},{fig_height:.1f})")
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs_width_ratios = [label_main_w, label_sub_w] + [data_plot_w_rel]*n_data_cols + [cbar_w]
    gs = gridspec.GridSpec(fig_plot_rows, total_gs_cols, figure=fig,
                           width_ratios=gs_width_ratios, height_ratios=[1]*fig_plot_rows,
                           hspace=0.15, wspace=0.1,
                           left=0.05, right=0.95, bottom=0.1, top=0.92)
    gs_col_main_type_label = 0; gs_col_sub_type_label = 1; gs_col_data_start = 2

    for dt_idx, data_type_str in enumerate(data_types_to_plot):
        details = data_type_details[data_type_str]; pynamit_ts_key = pynamit_timeseries_key_map[data_type_str]
        current_global_scale = global_plot_scales[data_type_str]
        vmin_use, vmax_use, cmap_use = current_global_scale['vmin'], current_global_scale['vmax'], current_global_scale['cmap']
        row_offset_input = dt_idx * 2; row_offset_fitted = row_offset_input + 1
        im_for_colorbar = None
        ax_main_label = fig.add_subplot(gs[row_offset_input:row_offset_fitted+1, gs_col_main_type_label])
        ax_main_label.text(0.5, 0.5, details['label'], ha='center', va='center', rotation='vertical', fontsize=10, transform=ax_main_label.transAxes); ax_main_label.axis('off')
        ax_input_label = fig.add_subplot(gs[row_offset_input, gs_col_sub_type_label])
        ax_input_label.text(0.5, 0.5, 'Input', ha='center', va='center', rotation='vertical', fontsize=9, transform=ax_input_label.transAxes); ax_input_label.axis('off')
        ax_fitted_label = fig.add_subplot(gs[row_offset_fitted, gs_col_sub_type_label])
        ax_fitted_label.text(0.5, 0.5, 'Fitted', ha='center', va='center', rotation='vertical', fontsize=9, transform=ax_fitted_label.transAxes); ax_fitted_label.axis('off')

        for ts_idx, timestep in enumerate(timesteps_to_plot):
            gs_data_col_current = gs_col_data_start + ts_idx
            ax_input = fig.add_subplot(gs[row_offset_input, gs_data_col_current], projection=ccrs.PlateCarree(central_longitude=noon_longitude))
            ax_fitted = fig.add_subplot(gs[row_offset_fitted, gs_data_col_current], projection=ccrs.PlateCarree(central_longitude=noon_longitude), sharex=ax_input, sharey=ax_input)

            if not (0 <= timestep < num_h5_steps):
                ax_input.text(0.5,0.5,"Data OOB", ha='center', va='center', transform=ax_input.transAxes); ax_fitted.text(0.5,0.5,"Data OOB", ha='center', va='center', transform=ax_fitted.transAxes)
                continue
            
            time_val = timestep * input_dt
            current_lon, current_lat, target_shape = (magnetosphere_lon, magnetosphere_lat, magnetosphere_lat.shape) if details["grid_type"] == "magnetosphere" else (ionosphere_lon, ionosphere_lat, ionosphere_lat.shape)
            input_data_2d = np.full(target_shape, np.nan); h5_key_pri = details["h5_key_primary"]
            if data_type_str=="Br": input_data_2d = h5file[h5_key_pri][timestep,:,:] * 1e-9
            elif data_type_str=="jr": input_data_2d = (h5file[h5_key_pri][timestep,:,:] * 1e-6) * ionosphere_br_2d
            elif data_type_str in ["SH","SP"]: input_data_2d = h5file[h5_key_pri][timestep,:,:]
            elif data_type_str in ["u_mag","u_theta","u_phi"]: u_e,u_n=h5file[h5_key_pri][timestep,:,:],h5file[details["h5_key_secondary"]][timestep,:,:];
            if data_type_str=="u_mag":input_data_2d=np.sqrt(u_n**2+u_e**2)
            elif data_type_str=="u_theta":input_data_2d=-u_n
            elif data_type_str=="u_phi":input_data_2d=u_e
            interpolated_data_2d = np.full(target_shape, np.nan)
            timeseries_entry = input_timeseries.get_entry(pynamit_ts_key, time_val, interpolation=False)
            if timeseries_entry:
                storage_basis = input_timeseries.storage_bases[pynamit_ts_key]; plot_evaluator = BasisEvaluator(storage_basis, Grid(lat=current_lat, lon=current_lon))
                if pynamit_ts_key=="Br": interpolated_data_2d=_calculate_interpolated_scalar_field(timeseries_entry,"Br",storage_basis,plot_evaluator,target_shape)
                elif pynamit_ts_key=="jr": interpolated_data_2d=_calculate_interpolated_scalar_field(timeseries_entry,"jr",storage_basis,plot_evaluator,target_shape)
                elif pynamit_ts_key=="conductance": interpolated_data_2d=_calculate_interpolated_conductance(timeseries_entry,data_type_str,storage_basis,plot_evaluator,target_shape)
                elif pynamit_ts_key=="u": interpolated_data_2d=_calculate_interpolated_u_field(timeseries_entry,data_type_str,storage_basis,plot_evaluator,target_shape)

            draw_top_lon_degrees = (dt_idx == 0)
            draw_right_lat_degrees = (ts_idx == num_ts - 1)

            im_input = plot_scalar_map_on_ax(ax_input, current_lon, current_lat, input_data_2d, "", cmap_use, vmin_use, vmax_use, use_pcolormesh=True, 
                                             draw_top_labels=draw_top_lon_degrees, draw_right_labels=draw_right_lat_degrees)
            if im_for_colorbar is None: im_for_colorbar = im_input
            
            im_fitted = plot_scalar_map_on_ax(ax_fitted, current_lon, current_lat, interpolated_data_2d, "", cmap_use, vmin_use, vmax_use, use_pcolormesh=True, 
                                              draw_top_labels=False, 
                                              draw_right_labels=draw_right_lat_degrees)
            if im_for_colorbar is None: im_for_colorbar = im_fitted
            
            if dt_idx == num_dt - 1:
                ax_fitted.set_xlabel(f"{time_val}s", fontsize=8)
            else: 
                ax_fitted.tick_params(axis='x',which='both', bottom=False, top=False, labelbottom=False)
            ax_input.tick_params(axis='x',which='both', bottom=False, top=False, labelbottom=False)

        if im_for_colorbar:
            cax = fig.add_subplot(gs[row_offset_input:row_offset_fitted+1, -1])
            cb = fig.colorbar(im_for_colorbar, cax=cax, orientation='vertical')
            tick_locator = mticker.MaxNLocator(nbins=5); cb.set_ticks(tick_locator.tick_values(vmin_use, vmax_use))
            cb.ax.tick_params(labelsize=8)

    fig.supxlabel("Time [s]", fontsize=10, y=0.045)
    fig.suptitle(f"Input vs. Fitted Data (Color ranges from {vmin_percentile:.1f}-{vmax_percentile:.1f}th percentiles)", fontsize=12, y=0.97)

    if output_filename: plt.savefig(output_filename, dpi=200); print(f"Figure saved to {output_filename}")
    else: plt.show()
    h5file.close()