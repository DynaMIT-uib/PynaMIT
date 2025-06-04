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

# --- Helper for color limits (Ultra-Simplified) ---
def get_final_plot_limits(vmin_orig, vmax_orig): 
    if vmin_orig == vmax_orig:
        delta_abs = abs(vmin_orig)
        if np.isclose(delta_abs, 0.0, atol=1e-15): delta = 1e-12 
        else: delta = delta_abs * 1e-6; delta = max(delta, 1e-12) 
        vmin_r, vmax_r = vmin_orig - delta, vmax_orig + delta
        if np.isclose(vmin_r, vmax_r): 
            final_kick_mag = abs(vmax_r) if not np.isclose(vmax_r,0) else 1.0
            final_kick = 10**(np.floor(np.log10(final_kick_mag)) -8); final_kick = max(final_kick, 1e-13) 
            vmin_r -= final_kick; vmax_r += final_kick
        return vmin_r, vmax_r
    if vmin_orig > vmax_orig: return vmax_orig, vmin_orig
    return vmin_orig, vmax_orig

# --- Helper functions for calculating interpolated data ---
# ... (These functions remain the same as your last working version)
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
        if u_coeffs.ndim == 1: 
            if storage_basis.Ncoeffs * 2 == u_coeffs.size: u_coeffs = u_coeffs.reshape((2, storage_basis.Ncoeffs))
            else: return np.full(target_shape, np.nan)
        elif u_coeffs.shape[0] != 2: return np.full(target_shape, np.nan)
        u = FieldExpansion(storage_basis, coeffs=u_coeffs, field_type="tangential"); u_t, u_p = u.to_grid(plot_evaluator)
        u_t_2d, u_p_2d = u_t.reshape(target_shape), u_p.reshape(target_shape)
        if component == "u_mag": return np.sqrt(u_t_2d**2 + u_p_2d**2)
        elif component == "u_theta": return u_t_2d
        elif component == "u_phi": return u_p_2d
    except Exception: return np.full(target_shape, np.nan)


# --- Helper plotting function (plot_scalar_map_on_ax - Degree labels always off) ---
def plot_scalar_map_on_ax(
    ax, lon_coords_2d, lat_coords_2d, data_2d_arr, title="", cmap="viridis",
    vmin=None, vmax=None, use_pcolormesh=False,
    norm=None 
):
    ax.coastlines(color="grey", zorder=3, linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, 
                      linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    
    current_vmin, current_vmax = vmin, vmax 
    if not (isinstance(current_vmin, (float, int, np.number)) and isinstance(current_vmax, (float, int, np.number)) and current_vmin < current_vmax):
        current_vmin, current_vmax = get_final_plot_limits(current_vmin if current_vmin is not None else 0.0, current_vmax if current_vmax is not None else 1.0)
    
    data_to_plot_masked = np.ma.masked_invalid(data_2d_arr)
    
    if norm is None: norm = mcolors.Normalize(vmin=current_vmin, vmax=current_vmax)
    elif isinstance(norm, mcolors.LogNorm): 
        safe_vmin_log = max(current_vmin, 1e-9) if current_vmin is not None and current_vmin > 0 else 1e-9 
        safe_vmax_log = current_vmax if current_vmax is not None and current_vmax > safe_vmin_log else safe_vmin_log * 10
        if not (np.isclose(norm.vmin, safe_vmin_log) and np.isclose(norm.vmax, safe_vmax_log)): norm = mcolors.LogNorm(vmin=safe_vmin_log, vmax=safe_vmax_log, clip=True) 
    elif isinstance(norm, mcolors.Normalize): 
        if not (np.isclose(norm.vmin, current_vmin) and np.isclose(norm.vmax, current_vmax)):
            norm.vmin = current_vmin; norm.vmax = current_vmax
    
    im = None
    if use_pcolormesh:
        im = ax.pcolormesh(lon_coords_2d, lat_coords_2d, data_to_plot_masked, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto", zorder=1)
    else: 
        num_levels = 16; plot_vmin, plot_vmax = norm.vmin, norm.vmax
        if isinstance(norm, mcolors.LogNorm):
            if plot_vmin > 0 and plot_vmax > 0 and plot_vmax > plot_vmin :
                 log_levels = np.geomspace(plot_vmin, plot_vmax, num_levels)
                 levels = log_levels[np.isfinite(log_levels)] 
                 if len(np.unique(np.round(levels,decimals=15))) < 2 : levels = np.array([plot_vmin, plot_vmax])
            else: levels = np.array([current_vmin, current_vmax]); norm = mcolors.Normalize(vmin=current_vmin, vmax=current_vmax) 
        else: 
            if abs(plot_vmax - plot_vmin) < 1e-12 : levels = np.array([plot_vmin, plot_vmax]) 
            else: levels = np.linspace(plot_vmin, plot_vmax, num_levels)
        if len(levels)<2: levels = np.array([plot_vmin, plot_vmax]) 
        im = ax.contourf(lon_coords_2d, lat_coords_2d, data_to_plot_masked, levels=levels, norm=norm, cmap=cmap, extend="both", transform=ccrs.PlateCarree(), zorder=1) 
    
    # ax.set_title(title, fontsize=9) # Titles are handled in sfig_TR_times
    return im

# --- Main plotting function ---
def plot_input_vs_interpolated(
    h5_filepath, interpolated_filename_prefix, timesteps_to_plot, input_dt, data_types_to_plot,
    noon_longitude=0, output_filename=None, 
    vmin_percentile=0.2, vmax_percentile=99.8,
    positive_definite_zeromin=True, 
    non_bwr_scale_type='linear' 
):
    # ... (Initial checks and Pynamit setup - unchanged) ...
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
        "Br": {"label": r"$\Delta B_r$ [T]", "cmap": "bwr", "grid_type": "magnetosphere", "h5_key_primary": "Bu", "scale_type": "linear"},
        "jr": {"label": r"$j_r$ [A/m$^2$]", "cmap": "bwr", "grid_type": "ionosphere", "h5_key_primary": "FAC", "scale_type": "linear"},
        "SH": {"label": r"$\Sigma_H$ [S]", "cmap": "viridis", "grid_type": "ionosphere", "h5_key_primary": "SH", "scale_type": non_bwr_scale_type},
        "SP": {"label": r"$\Sigma_P$ [S]", "cmap": "viridis", "grid_type": "ionosphere", "h5_key_primary": "SP", "scale_type": non_bwr_scale_type},
        "u_mag": {"label": r"$|u|$ [m/s]", "cmap": "viridis", "grid_type": "ionosphere", "h5_key_primary": "We", "h5_key_secondary": "Wn", "scale_type": non_bwr_scale_type},
        "u_theta": {"label": r"$u_\theta$ (South) [m/s]", "cmap": "bwr", "grid_type": "ionosphere", "h5_key_primary": "We", "h5_key_secondary": "Wn", "scale_type": "linear"},
        "u_phi": {"label": r"$u_\phi$ (East) [m/s]", "cmap": "bwr", "grid_type": "ionosphere", "h5_key_primary": "We", "h5_key_secondary": "Wn", "scale_type": "linear"},}
    input_timeseries = Timeseries(cs_basis, input_storage_bases, input_vars_pynamit); input_timeseries.load_all(io)
    ionosphere_lat, ionosphere_lon = h5file["glat"][:], h5file["glon"][:]; magnetosphere_lat, magnetosphere_lon = h5file["Blat"][:], h5file["Blon"][:]
    ionosphere_grid = Grid(lat=ionosphere_lat, lon=ionosphere_lon); ionosphere_b_evaluator = FieldEvaluator(mainfield, ionosphere_grid, ri_value)
    ionosphere_br_2d = ionosphere_b_evaluator.br.reshape(ionosphere_lat.shape)

    print("Starting Pass 1: Collecting data for global vmin/vmax...") 
    all_data_for_scaling = {dt_str: {'input': [], 'interpolated': []} for dt_str in data_types_to_plot}
    # ... (Data collection loop - unchanged) ...
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

    print("Calculating global vmin/vmax from percentiles (minimal adjustment if identical)...")
    global_plot_scales = {}
    # ... (Global scale calculation logic - unchanged from previous correct version) ...
    for data_type_str in data_types_to_plot: 
        details = data_type_details[data_type_str]; cmap_global = details['cmap']
        current_scale_type = details.get('scale_type', 'linear') 
        cat_input_all = all_data_for_scaling[data_type_str]['input']; cat_interp_all = all_data_for_scaling[data_type_str]['interpolated']
        combined_flat = np.concatenate( (np.concatenate(cat_input_all) if cat_input_all else np.array([]), np.concatenate(cat_interp_all) if cat_interp_all else np.array([])) )
        temp_valid_data = combined_flat[~np.isnan(combined_flat)] 
        if current_scale_type == 'log': 
            valid_data_for_percentile = temp_valid_data[temp_valid_data > 0]
        else: valid_data_for_percentile = temp_valid_data
        vmin_pctl, vmax_pctl = 0.0, 1.0 
        if valid_data_for_percentile.size > 0:
            if cmap_global == "bwr": 
                abs_max_s = np.percentile(np.abs(valid_data_for_percentile), vmax_percentile)
                vmin_pctl = -abs_max_s ; vmax_pctl = abs_max_s
                if np.isclose(vmin_pctl, 0.0, atol=1e-12) and np.isclose(vmax_pctl, 0.0, atol=1e-12): vmin_pctl, vmax_pctl = -1e-9, 1e-9 
            else: 
                vmin_pctl = np.percentile(valid_data_for_percentile, vmin_percentile)
                vmax_pctl = np.percentile(valid_data_for_percentile, vmax_percentile)
                if current_scale_type == 'linear' and positive_definite_zeromin:
                    if vmin_pctl >= 0: vmin_pctl = 0.0 
                elif current_scale_type == 'log':
                    smallest_positive_val = 1e-9 
                    if np.any(valid_data_for_percentile > 0): smallest_positive_val = np.min(valid_data_for_percentile[valid_data_for_percentile > 0])
                    if vmin_pctl <= 0: vmin_pctl = max(smallest_positive_val * 0.1, 1e-9) 
                    if vmax_pctl <= vmin_pctl : vmax_pctl = vmin_pctl * 100 
        else: 
            print(f"    Warning: No valid data for percentiles of '{data_type_str}'. Using default [0,1].")
            if current_scale_type == 'log': vmin_pctl, vmax_pctl = 1e-3, 1e-1 
        vmin_final, vmax_final = get_final_plot_limits(vmin_pctl, vmax_pctl) 
        norm_for_plot = None
        if current_scale_type == 'log' and cmap_global != 'bwr':
            if vmin_final > 0 and vmax_final > vmin_final: 
                norm_for_plot = mcolors.LogNorm(vmin=vmin_final, vmax=vmax_final, clip=True)
            else: 
                print(f"    Warning: Cannot use log scale for {data_type_str} (vmin={vmin_final:.2e}, vmax={vmax_final:.2e}). Using linear.")
                current_scale_type = 'linear' 
                if positive_definite_zeromin and (temp_valid_data.size == 0 or np.percentile(temp_valid_data, vmin_percentile) >= 0) : vmin_pctl_linear = 0.0
                elif temp_valid_data.size > 0: vmin_pctl_linear = np.percentile(temp_valid_data, vmin_percentile)
                else: vmin_pctl_linear = 0.0
                if temp_valid_data.size > 0: vmax_pctl_linear = np.percentile(temp_valid_data, vmax_percentile)
                else: vmax_pctl_linear = 1.0
                vmin_final, vmax_final = get_final_plot_limits(vmin_pctl_linear, vmax_pctl_linear)
        global_plot_scales[data_type_str] = {'vmin': vmin_final, 'vmax': vmax_final, 'cmap': cmap_global, 'norm': norm_for_plot, 'scale_type': current_scale_type}
        print(f"  Global scale for '{data_type_str}' ({current_scale_type}): vmin={vmin_final:.3e}, vmax={vmax_final:.3e}")


    # --- Figure Creation using 2x2 Subfigures ---
    num_dt = len(data_types_to_plot)
    num_plot_rows = num_dt * 2 
    num_plot_cols = len(timesteps_to_plot)

    # Define relative sizes for subfigures - make top row and left col narrow
    time_row_h_frac = 0.04   # Small height for the top row (time labels)
    plots_row_h_frac = 1.0 - time_row_h_frac
    
    cbars_labels_col_w_frac = 0.20 # Width for Bottom-Left (cbars, data type labels, Input/Fitted)
    plots_col_w_frac = 1.0 - cbars_labels_col_w_frac  # Width for Bottom-Right (plots)

    # Estimate overall figure size
    base_plot_w, base_plot_h = 2.0, 1.5  # Smaller base size per plot
    fig_width = (num_plot_cols * base_plot_w) / plots_col_w_frac 
    fig_height = (num_plot_rows * base_plot_h) / plots_row_h_frac 
    
    fig_width = min(max(8, fig_width), 22) # Adjusted caps for potentially more compact layout
    fig_height = min(max(4 + time_row_h_frac*fig_height , fig_height), num_plot_rows * (base_plot_h + 0.1) + 1.0) 


    print(f"Creating figure with 2x2 subfigures. Target Size: ({fig_width:.1f},{fig_height:.1f})")
    fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
    
    sfigs_grid = fig.subfigures(2, 2, height_ratios=[time_row_h_frac, plots_row_h_frac], 
                                width_ratios=[cbars_labels_col_w_frac, plots_col_w_frac], 
                                hspace=0.01, wspace=0.01) # Minimal space between subfigures
    
    sfig_TL_empty = sfigs_grid[0, 0]
    sfig_TR_times = sfigs_grid[0, 1]
    sfig_BL_cbars_and_labels = sfigs_grid[1, 0]
    sfig_BR_plots = sfigs_grid[1, 1]

    sfig_TL_empty.patch.set_alpha(0.0) 
    for ax_ in sfig_TL_empty.get_axes(): ax_.remove() # Ensure it's truly empty

    # --- Time Labels in sfig_TR_times (Top-Right) ---
    if num_plot_cols > 0 :
        time_label_axes_flat = sfig_TR_times.subplots(1, num_plot_cols, sharey=True) # sharey for vertical alignment
        time_label_axes = [time_label_axes_flat] if num_plot_cols == 1 else time_label_axes_flat
        for ts_idx, timestep in enumerate(timesteps_to_plot):
            time_val = timestep * input_dt
            time_label_axes[ts_idx].text(0.5, 0.5, f"{time_val}s", ha='center', va='center', fontsize=8) # Centered
            time_label_axes[ts_idx].axis('off')

    # --- Main Map Plots in sfig_BR_plots (Bottom-Right) ---
    map_axes_flat = sfig_BR_plots.subplots(num_plot_rows, num_plot_cols, sharex=True, sharey=True,
                                     subplot_kw={'projection': ccrs.PlateCarree(central_longitude=noon_longitude)})
    if num_plot_rows == 0 or num_plot_cols == 0: 
        if h5file: h5file.close(); return
    if num_plot_rows == 1 and num_plot_cols == 1: map_axes = np.array([[map_axes_flat]])
    elif num_plot_rows == 1: map_axes = map_axes_flat[np.newaxis, :]
    elif num_plot_cols == 1: map_axes = map_axes_flat[:, np.newaxis]
    else: map_axes = map_axes_flat
    
    # --- Bottom-Left Panel (sfig_BL_cbars_and_labels) ---
    # GridSpec with num_plot_rows and 3 columns: DataTypeLabel | Colorbar | Input/Fitted Label
    gs_bottom_left = gridspec.GridSpec(num_plot_rows, 3, figure=sfig_BL_cbars_and_labels, 
                                       width_ratios=[0.4, 0.3, 0.3], # DataType | Cbar | Input/Fitted
                                       wspace=0.1, hspace=0.05) # Reduced hspace

    mappables_for_cbars = [None] * num_dt

    for dt_idx, data_type_str in enumerate(data_types_to_plot): 
        details = data_type_details[data_type_str]
        current_global_scale = global_plot_scales[data_type_str]
        vmin_use, vmax_use, cmap_use = current_global_scale['vmin'], current_global_scale['vmax'], current_global_scale['cmap']
        norm_use = current_global_scale['norm'] 
        
        row_idx_input = dt_idx * 2
        row_idx_fitted = row_idx_input + 1
        current_mappable_this_dt = None

        # Data Type Label for this block (spans two rows in gs_bottom_left, first column)
        ax_dt_label = sfig_BL_cbars_and_labels.add_subplot(gs_bottom_left[row_idx_input:row_idx_input+2, 0])
        ax_dt_label.text(0.5, 0.5, details['label'], ha='center', va='center', rotation=90, fontsize=9) # rotation 90
        ax_dt_label.axis('off')
        
        # Input/Fitted Labels (in the third column of gs_bottom_left, for each specific row)
        ax_input_text_label = sfig_BL_cbars_and_labels.add_subplot(gs_bottom_left[row_idx_input, 2])
        ax_input_text_label.text(0.5, 0.5, "Input", ha='center', va='center', rotation=90, fontsize=8) # rotation 90
        ax_input_text_label.axis('off')

        ax_fitted_text_label = sfig_BL_cbars_and_labels.add_subplot(gs_bottom_left[row_idx_fitted, 2])
        ax_fitted_text_label.text(0.5, 0.5, "Fitted", ha='center', va='center', rotation=90, fontsize=8) # rotation 90
        ax_fitted_text_label.axis('off')


        for ts_idx, timestep in enumerate(timesteps_to_plot):
            ax_input = map_axes[row_idx_input, ts_idx]
            ax_fitted = map_axes[row_idx_fitted, ts_idx]
            ax_input.clear(); ax_fitted.clear()

            if not (0 <= timestep < num_h5_steps): 
                ax_input.text(0.5,0.5,"Data OOB", ha='center', va='center', transform=ax_input.transAxes); ax_input.set_xticks([]); ax_input.set_yticks([])
                ax_fitted.text(0.5,0.5,"Data OOB", ha='center', va='center', transform=ax_fitted.transAxes); ax_fitted.set_xticks([]); ax_fitted.set_yticks([])
                continue
            
            plot_title = "" # Time labels are in sfig_TR_times

            current_lon, current_lat, target_shape = (magnetosphere_lon, magnetosphere_lat, magnetosphere_lat.shape) if details["grid_type"] == "magnetosphere" else (ionosphere_lon, ionosphere_lat, ionosphere_lat.shape)
            # ... (Data calculation - unchanged) ...
            input_data_2d = np.full(target_shape, np.nan); h5_key_pri = details["h5_key_primary"]
            if data_type_str=="Br": input_data_2d = h5file[h5_key_pri][timestep,:,:] * 1e-9
            elif data_type_str=="jr": input_data_2d = (h5file[h5_key_pri][timestep,:,:] * 1e-6) * ionosphere_br_2d
            elif data_type_str in ["SH","SP"]: input_data_2d = h5file[h5_key_pri][timestep,:,:]
            elif data_type_str in ["u_mag","u_theta","u_phi"]: u_e,u_n=h5file[h5_key_pri][timestep,:,:],h5file[details["h5_key_secondary"]][timestep,:,:];
            if data_type_str=="u_mag":input_data_2d=np.sqrt(u_n**2+u_e**2)
            elif data_type_str=="u_theta":input_data_2d=-u_n
            elif data_type_str=="u_phi":input_data_2d=u_e
            if data_type_str == "u_mag" and np.all(np.isnan(input_data_2d)): print(f"    DEBUG u_mag INPUT timestep {timestep}: ALL NaNs")
            interpolated_data_2d = np.full(target_shape, np.nan)
            timeseries_entry = input_timeseries.get_entry(pynamit_timeseries_key_map[data_type_str], time_val, interpolation=False)
            if timeseries_entry:
                storage_basis = input_timeseries.storage_bases[pynamit_timeseries_key_map[data_type_str]]; plot_evaluator = BasisEvaluator(storage_basis, Grid(lat=current_lat, lon=current_lon))
                if pynamit_timeseries_key_map[data_type_str]=="Br": interpolated_data_2d=_calculate_interpolated_scalar_field(timeseries_entry,"Br",storage_basis,plot_evaluator,target_shape)
                elif pynamit_timeseries_key_map[data_type_str]=="jr": interpolated_data_2d=_calculate_interpolated_scalar_field(timeseries_entry,"jr",storage_basis,plot_evaluator,target_shape)
                elif pynamit_timeseries_key_map[data_type_str]=="conductance": interpolated_data_2d=_calculate_interpolated_conductance(timeseries_entry,data_type_str,storage_basis,plot_evaluator,target_shape)
                elif pynamit_timeseries_key_map[data_type_str]=="u": interpolated_data_2d=_calculate_interpolated_u_field(timeseries_entry,data_type_str,storage_basis,plot_evaluator,target_shape)
            if data_type_str == "u_mag" and np.all(np.isnan(interpolated_data_2d)): print(f"    DEBUG u_mag INTERP timestep {timestep}: ALL NaNs")

            im_input = plot_scalar_map_on_ax(ax_input, current_lon, current_lat, input_data_2d, plot_title, cmap_use, vmin_use, vmax_use, use_pcolormesh=True, norm=norm_use)
            if current_mappable_this_dt is None and not np.all(np.isnan(input_data_2d)): current_mappable_this_dt = im_input
            
            im_fitted = plot_scalar_map_on_ax(ax_fitted, current_lon, current_lat, interpolated_data_2d, "", cmap_use, vmin_use, vmax_use, use_pcolormesh=True, norm=norm_use)
            if current_mappable_this_dt is None and not np.all(np.isnan(interpolated_data_2d)): current_mappable_this_dt = im_fitted
        
        mappables_for_cbars[dt_idx] = current_mappable_this_dt

    # Add Colorbars in the Bottom-Left subfigure (sfig_BL_cbars_and_labels)
    for i in range(num_dt): 
        mappable = mappables_for_cbars[i]
        
        row_start_bl = i * 2
        if mappable:
            ax_cbar = sfig_BL_cbars_and_labels.add_subplot(gs_bottom_left[row_start_bl:row_start_bl+2, 1]) 
            cb = fig.colorbar(mappable, cax=ax_cbar, orientation='vertical')
            cb.ax.tick_params(labelsize=7) 
        else: 
            ax_cbar = sfig_BL_cbars_and_labels.add_subplot(gs_bottom_left[row_start_bl:row_start_bl+2, 1])
            ax_cbar.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=7)
            ax_cbar.axis('off')
    
    if output_filename: plt.savefig(output_filename, dpi=200); print(f"Figure saved to {output_filename}")
    else: plt.show()
    h5file.close()