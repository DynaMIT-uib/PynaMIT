
import numpy as np
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid

def debug_exact_recovery():
    print("\n--- EXACT RECOVERY TEST N=1 ---")
    
    basis = SHBasis(Nmax=1, Mmax=1, Nmin=0)
    
    # Create Standard Grid
    N_lat = 8
    N_lon = 8
    theta = np.linspace(0.1, 3.0, N_lat)
    phi = np.linspace(0, 2*np.pi, N_lon, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing='ij')
    grid = Grid(theta=np.rad2deg(tt).flatten(), phi=np.rad2deg(pp).flatten())
    
    # Weights for Legacy
    weights_th = np.sin(theta)
    weights_2d = np.repeat(weights_th[:, None], N_lon, axis=1)
    weights_flat = np.concatenate([weights_2d.flatten(), weights_2d.flatten()])
    
    # CASE 1: Pure Toroidal T11s (Sin)
    # T = sin(theta) sin(phi)
    # u_th = -1/sin dT/dphi = -1/sin (sin th cos phi) = -cos phi
    # u_ph = dT/dth = cos th sin phi
    print("\nInput: Pure T11s (Coeff=1.0)")
    
    sin_th = np.sin(tt)
    cos_th = np.cos(tt)
    sin_ph = np.sin(pp)
    cos_ph = np.cos(pp)
    
    u_th_true = -cos_ph # Matches "Standard" definition with -1 phase?
    # Wait, Standard u = Grad x (Tr).
    # u_th = 1/sin dT/dphi = cos phi.
    # Legacy u = r x Grad T.
    # u_th = -1/sin dT/dphi = -cos phi.
    # I will inject LEGACY Definition (-cos phi).
    
    u_ph_true = cos_th * sin_ph
    
    data_flat = np.concatenate([u_th_true.flatten(), u_ph_true.flatten()])
    data_tuple = (u_th_true, u_ph_true)
    
    print("running Legacy...")
    c_slow = basis.grid_to_basis(data_flat, grid, weights=weights_flat, reg_lambda=0, helmholtz=True)
    
    print("running Fast...")
    c_fast = basis.grid_to_basis_fast(data_tuple, theta, phi, weights=weights_th, reg_lambda=0, vector_type='tangential')
    
    print(f"c_slow shape: {c_slow.shape}")
    print(f"c_fast shape: {c_fast.shape}")
    
    # Flatten Legacy to match Fast
    c_slow_flat = c_slow.flatten()
    
    # Check Coeffs for T11s (Index 7?)
    # Layout Nmin=0: P00(0), P10(1), P11c(2), P11s(3). T00(4), T10(5), T11c(6), T11s(7).
    idx = 7
    print(f"Legacy T11s: {c_slow_flat[idx]:.5f}")
    print(f"Fast   T11s: {c_fast[idx]:.5f}")
    
    # CASE 2: Pure Poloidal P11s (Sin)
    # P = sin(theta) sin(phi)
    # u = -Grad P
    # u_th = -dP/dth = -cos th sin phi
    # u_ph = -1/sin dP/dphi = -cos phi
    print("\nInput: Pure P11s (Coeff=1.0)")
    
    u_th_true = -cos_th * sin_ph
    u_ph_true = -cos_ph
    
    data_flat = np.concatenate([u_th_true.flatten(), u_ph_true.flatten()])
    data_tuple = (u_th_true, u_ph_true)
    
    c_slow = basis.grid_to_basis(data_flat, grid, weights=weights_flat, reg_lambda=0, helmholtz=True)
    c_fast = basis.grid_to_basis_fast(data_tuple, theta, phi, weights=weights_th, reg_lambda=0, vector_type='tangential')
    
    idx = 3 # P11s
    print(f"Legacy P11s: {c_slow.flatten()[idx]:.5f}")
    print(f"Fast   P11s: {c_fast[idx]:.5f}")

if __name__ == "__main__":
    debug_exact_recovery()
