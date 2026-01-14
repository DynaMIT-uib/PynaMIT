import numpy as np
import pytest
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid

@pytest.mark.parametrize("N, M", [(5, 5), (10, 5), (15, 0)])
def test_fast_vs_slow_equivalence_scalar(N, M):
    import pynamit
    print(f"DEBUG: PynaMIT Path: {pynamit.__file__}")
    """
    Verify Fast vs Slow Equivalence for Scalar Fields.
    Expect Bit-Exact Identity for reg_lambda=0.
    """
    basis = SHBasis(Nmax=N, Mmax=M, Nmin=0)
    
    # Grid
    N_lat = 2 * N + 4
    N_lon = 2 * M + 4 if M > 0 else 4
    theta = np.linspace(0.1, 3.0, N_lat)
    phi = np.linspace(0, 2*np.pi, N_lon, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing='ij')
    grid = Grid(theta=np.rad2deg(tt).flatten(), phi=np.rad2deg(pp).flatten())
    
    # Random Data
    np.random.seed(42)
    data_2d = np.random.randn(N_lat, N_lon)
    data_flat = data_2d.flatten()
    
    weights_th = np.sin(theta)
    weights_2d = np.repeat(weights_th[:, None], N_lon, axis=1)
    weights_flat = weights_2d.flatten()
    
    # Solve Lambda=0
    c_slow = basis.grid_to_basis(data_flat, grid, weights=weights_flat, reg_lambda=0)
    c_fast = basis.grid_to_basis_fast(data_2d, theta, phi, weights=weights_th, reg_lambda=0, vector_type='scalar')
    
    diff = c_fast - c_slow.flatten()
    rel_err = np.linalg.norm(diff) / (np.linalg.norm(c_slow) + 1e-15)
    
    print(f"Scalar N={N} M={M} Rel: {rel_err}")
    
    # Scalar should be perfect
    assert rel_err < 1e-13, f"Scalar Identity Failed: {rel_err}"

@pytest.mark.parametrize("N, M", [(1, 1), (5, 5), (10, 5)])
def test_fast_vs_slow_equivalence_vector(N, M):
    """
    Verify Fast vs Slow Equivalence for Vector Fields.
    Expect Bit-Exact for Zonal (m=0), Approx 4% for Coupled (m>0).
    """
    basis = SHBasis(Nmax=N, Mmax=M, Nmin=0)
    
    # Grid
    N_lat = 2 * N + 6
    N_lon = 2 * M + 6 if M > 0 else 4
    theta = np.linspace(0.1, 3.0, N_lat)
    phi = np.linspace(0, 2*np.pi, N_lon, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing='ij')
    grid = Grid(theta=np.rad2deg(tt).flatten(), phi=np.rad2deg(pp).flatten())
    
    # Random Data
    np.random.seed(123)
    u_th = np.random.randn(N_lat, N_lon)
    u_ph = np.random.randn(N_lat, N_lon)
    data_tuple = (u_th, u_ph)
    data_flat = np.concatenate([u_th.flatten(), u_ph.flatten()])
    
    weights_th = np.sin(theta)
    weights_2d = np.repeat(weights_th[:, None], N_lon, axis=1)
    weights_flat = np.concatenate([weights_2d.flatten(), weights_2d.flatten()])
    
    # Solve Lambda=0
    c_slow = basis.grid_to_basis(data_flat, grid, weights=weights_flat, reg_lambda=0, helmholtz=True)
    c_fast = basis.grid_to_basis_fast(data_tuple, theta, phi, weights=weights_th, reg_lambda=0, vector_type='tangential')
    
    diff = c_fast - c_slow.flatten()
    norm_slow = np.linalg.norm(c_slow)
    rel_err = np.linalg.norm(diff) / (norm_slow + 1e-15)
    
    print(f"Vector N={N} M={M} Rel: {rel_err}")
    
    # Detailed Diagnostics if N=1 (where we expect close match)
    if N == 1 and rel_err > 0.05:
        print("\n--- Detailed Coefficients (N=1) ---")
        # Indices for N=1, M=1 are 0 (0,0), 1 (1,0), 2 (1,1).
        # Vector layout: [Pol_c, Pol_s, Tor_c, Tor_s] per m? No.
        # Layout is flat [Pol coeffs..., Tor coeffs...]
        # N=1 means indices 0,1,2,3?
        # Basis convention: 
        # m=0: (l=0?), l=1. (idx 0,1).
        # m=1: l=1. (idx 2).
        # Order: 
        # Pol: (0,0), (1,0), (1,1)c, (1,1)s.
        # Tor: same.
        # Total coeffs 8?
        print(f"Slow Norm: {norm_slow}, Fast Norm: {np.linalg.norm(c_fast)}")
        # Print first few ratios
        ratio = c_slow.flatten() / (c_fast + 1e-20)
        print("Ratios (Slow/Fast) first 10:")
        print(ratio[:10])
    
    # Exactness Requirement
    # After removing empirical scalings and unifying signs, the Fast Path and 
    # Slow Path (Gaunt) should match to precision (< 1e-10) for all modes.
    
    assert rel_err < 1e-13, f"Vector Identity Failed: {rel_err}"
