
import numpy as np
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid

def debug_n5_recovery():
    print("\n--- N=5 RECOVERY TEST ---")
    
    N = 5
    M = 5
    basis = SHBasis(Nmax=N, Mmax=M, Nmin=0)
    
    # Grid
    N_lat = 2 * N + 6
    N_lon = 2 * M + 6
    theta = np.linspace(0.1, 3.0, N_lat)
    phi = np.linspace(0, 2*np.pi, N_lon, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing='ij')
    grid = Grid(theta=np.rad2deg(tt).flatten(), phi=np.rad2deg(pp).flatten())
    
    # Weights for Legacy
    weights_th = np.sin(theta)
    weights_2d = np.repeat(weights_th[:, None], N_lon, axis=1)
    weights_flat = np.concatenate([weights_2d.flatten(), weights_2d.flatten()])
    
    # Generate Random Coeffs (Signal)
    np.random.seed(42)
    coeffs_in = np.random.randn(basis.index_length * 2) 
    # normalize to order 1
    coeffs_in /= np.max(np.abs(coeffs_in))
    
    print("Generating Field via Legacy `evaluate`...")
    # Reshape coeffs to (2, N_coeffs) for evaluate/legacy
    coeffs_matrix = coeffs_in.reshape(2, -1)
    
    # This uses get_gradient_matrix internally, so it represents the "Truth"
    u_vec_flat = basis.evaluate(coeffs_matrix, grid, vector_type='tangential')
    # u_vec_flat is (2*GridSize,) -> [u_th..., u_ph...]
    
    print(f"u_vec_flat shape: {u_vec_flat.shape}")
    print(f"Grid Size: {grid.size}")
    
    # Check if shape is (2, GridSize)?
    # If basis_to_grid used tensordot(G_h, coeffs, 2), result is (2, N_evals). 
    # Because G_h is (2, N_evals, 2, Coeffs). Coeffs is (2, Coeffs).
    # Result -> (2, N_evals).
    
    # Flatten if needed
    if u_vec_flat.ndim == 2 and u_vec_flat.shape[0] == 2:
        u_th_flat = u_vec_flat[0]
        u_ph_flat = u_vec_flat[1]
    else:
        # Fallback assumption
        u_th_flat = u_vec_flat[:grid.size]
        u_ph_flat = u_vec_flat[grid.size:]
        
    u_th = u_th_flat.reshape(N_lat, N_lon)
    u_ph = u_ph_flat.reshape(N_lat, N_lon)
    data_tuple = (u_th, u_ph)
    
    print("Running Fast SHT Inverse...")
    c_fast = basis.grid_to_basis_fast(data_tuple, theta, phi, weights=weights_th, reg_lambda=0, vector_type='tangential')
    
    # Analysis of Ratios per Mode
    print("\n--- DETAILED RATIO ANALYSIS ---")
    
    # Fast SHT layout is generally [Pol_Coeffs, Tor_Coeffs].
    # Pol Coeffs = [C_nm (all), S_nm (all)].
    # Let's verify layout by checking index_length
    
    # basis.cnm: Cosine indices (m>=0). basis.snm: Sine indices (m>=1).
    n_cnm = basis.cnm.n.size
    n_snm = basis.snm.n.size
    
    # Poloidal Block
    c_pol = c_fast[:basis.index_length]
    c_tor = c_fast[basis.index_length:]
    
    # Input Coeffs (reshaped to 2, -1 earlier)
    c_in_pol = coeffs_matrix[0]
    c_in_tor = coeffs_matrix[1]
    
    print(f"Num Indices: {basis.index_length} (Cnm: {n_cnm}, Snm: {n_snm})")
    
    def analyze_block(name, c_meas, c_true):
        print(f"\nBlock: {name}")
        print(f"{'Type':<5} | {'n':<2} | {'m':<2} | {'True':<10} | {'Meas':<10} | {'Ratio':<10}")
        print("-" * 55)
        
        # Cnm part
        for i in range(n_cnm):
            n = basis.cnm.n.flatten()[i]
            m = basis.cnm.m.flatten()[i]
            val_true = c_true[i]
            val_meas = c_meas[i]
            ratio = val_meas / val_true if abs(val_true) > 1e-6 else 0.0
            
            if abs(val_true) > 1e-6:
                 print(f"{'Cos':<5} | {n:<2} | {m:<2} | {val_true:>10.5f} | {val_meas:>10.5f} | {ratio:>10.5f}")
        
        # Snm part
        for i in range(n_snm):
            n = basis.snm.n.flatten()[i]
            m = basis.snm.m.flatten()[i]
            idx = n_cnm + i
            val_true = c_true[idx]
            val_meas = c_meas[idx]
            ratio = val_meas / val_true if abs(val_true) > 1e-6 else 0.0
            
            if abs(val_true) > 1e-6:
                 print(f"{'Sin':<5} | {n:<2} | {m:<2} | {val_true:>10.5f} | {val_meas:>10.5f} | {ratio:>10.5f}")

    analyze_block("Poloidal", c_pol, c_in_pol)
    analyze_block("Toroidal", c_tor, c_in_tor)

if __name__ == "__main__":
    debug_n5_recovery()
