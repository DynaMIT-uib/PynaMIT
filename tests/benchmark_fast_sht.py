
import time
import numpy as np
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid

def benchmark_fast_sht():
    print("\n--- FAST SHT BENCHMARK ---")
    print(f"{'N':<5} | {'Legacy (s)':<12} | {'Fast (s)':<12} | {'Speedup':<10} | {'Error':<10}")
    print("-" * 65)
    
    resolutions = [10, 30, 50]
    
    for N in resolutions:
        M = N
        basis = SHBasis(Nmax=N, Mmax=M, Nmin=0)
        
        # Grid
        N_lat = 2 * N + 10
        N_lon = 2 * M + 10
        theta = np.linspace(0.1, 3.0, N_lat)
        phi = np.linspace(0, 2*np.pi, N_lon, endpoint=False)
        tt, pp = np.meshgrid(theta, phi, indexing='ij')
        grid = Grid(theta=np.rad2deg(tt).flatten(), phi=np.rad2deg(pp).flatten())
        
        # Data
        np.random.seed(42)
        u_th = np.random.randn(N_lat, N_lon)
        u_ph = np.random.randn(N_lat, N_lon)
        data_tuple = (u_th, u_ph)
        data_flat = np.concatenate([u_th.flatten(), u_ph.flatten()])
        
        weights_th = np.sin(theta)
        weights_2d = np.repeat(weights_th[:, None], N_lon, axis=1)
        weights_flat = np.concatenate([weights_2d.flatten(), weights_2d.flatten()])
        
        # Determine Lambda=0 to test raw solver speed (no reg overhead difference)
        reg_lambda = 0.0
        
        # Time Legacy
        # Warmup
        if N < 40: # Legacy is too slow for large N to run multiple times in interactive session
            _ = basis.grid_to_basis(data_flat, grid, weights=weights_flat, reg_lambda=reg_lambda, helmholtz=True)
            t0 = time.time()
            c_slow = basis.grid_to_basis(data_flat, grid, weights=weights_flat, reg_lambda=reg_lambda, helmholtz=True)
            t_slow = time.time() - t0
        else:
            t_slow = 999.0 # Skip large N legacy
            
        # Time Fast
        # Warmup
        _ = basis.grid_to_basis_fast(data_tuple, theta, phi, weights=weights_th, reg_lambda=reg_lambda, vector_type='tangential')
        t0 = time.time()
        c_fast = basis.grid_to_basis_fast(data_tuple, theta, phi, weights=weights_th, reg_lambda=reg_lambda, vector_type='tangential')
        t_fast = time.time() - t0
        
        # Error (if Legacy ran)
        if N < 40:
            diff = c_fast - c_slow.flatten()
            err = np.linalg.norm(diff) / np.linalg.norm(c_slow)
            speedup = t_slow / t_fast
            print(f"{N:<5} | {t_slow:<12.5f} | {t_fast:<12.5f} | {speedup:<10.2f} | {err:<10.2e}")
        else:
            print(f"{N:<5} | {'Skip (>10s)':<12} | {t_fast:<12.5f} | {'inf':<10} | {'N/A':<10}")

if __name__ == "__main__":
    benchmark_fast_sht()
