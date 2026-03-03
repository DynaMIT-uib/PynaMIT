
import numpy as np
import pytest
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid

def test_mw_exactness_scalar():
    """Verify Exact Recovery of Scalar Field using MW sampling."""
    print("\n--- MW SCALAR EXACTNESS ---")
    
    L = 10
    basis = SHBasis(Nmax=L, Mmax=L, mean_free=False)
    
    # Create MW Grid
    grid = Grid.create_mw_grid(L)
    print(f"Grid Size: {grid.size} (N_theta={L+1}, N_phi={2*L+1})")
    
    # Generate MW Weights
    mw_weights_th = SHBasis.get_mw_weights(L)
    print(f"Sum of MW Weights: {np.sum(mw_weights_th):.5f} (Expect ~2.0)")
    
    # Broadcast to full grid? 
    # grid_to_basis_fast takes `weights` as 1D theta weights (if separable).
    # Since MW is separable, we pass 1D weights.
    
    # Generate Band-Limited Signal
    np.random.seed(42)
    coeffs_true = np.random.randn(basis.index_length)
    # Zero out high coeffs? No, basis is truncated at Nmax=L.
    
    # Evaluate on Grid (Exact)
    # Using 'scalar' mode
    # grid_to_basis_fast expects (N_theta, N_phi) data 
    # But basis.evaluate returns flattened.
    u_flat = basis.evaluate(coeffs_true, grid, vector_type='scalar')
    
    # Project Back using Fast SHT + MW Weights
    # Reshape u_flat
    N_theta = L + 1
    N_phi = 2 * L + 1
    u_grid = u_flat.reshape(N_theta, N_phi)
    
    # Extract 1D axes from Grid
    theta_1d = np.deg2rad(grid.theta.reshape(N_theta, N_phi)[:, 0])
    phi_1d = np.deg2rad(grid.phi.reshape(N_theta, N_phi)[0, :])
    
    # Manual Quadrature Check
    # C_LM = sum_{theta, phi} u(theta, phi) * Y_LM(theta, phi) * w_theta * (2pi/N_phi)
    
    # We pick a specific mode to test. Say L=1, M=0.
    # coeffs_true[idx]
    
    # Reconstruct Y_LM on grid
    # We can use basis.evaluate with a single-mode coeff vector
    idx_test = basis.index_pairs.index((1, 0))
    c_test = np.zeros_like(coeffs_true)
    c_test[idx_test] = 1.0
    Y_test = basis.evaluate(c_test, grid, vector_type='scalar').reshape(N_theta, N_phi)
    
    # Quadrature
    # Integration over sphere: 
    # sum_{t} sum_{p} u[t, p] * Y_test[t, p] * w_theta[t] * (2pi/N_phi)
    
    dphi = 2 * np.pi / N_phi
    integral = np.sum(u_grid * Y_test * mw_weights_th[:, None]) * dphi
    
    print(f"Manual Quadrature (L=1, M=0): {integral:.5f}")
    print(f"Expected Coeff (L=1, M=0): {coeffs_true[idx_test]:.5f}")
    
    print("Running Fast SHT with MW Weights...")
    # NOTE: grid_to_basis_fast performs Weighted Least Squares: min || W * (Ax - b) ||^2.
    # This effectively weights the inner product by W^2.
    # MW Quadrature requires weighting by w_mw. So we must pass sqrt(w_mw).
    coeffs_rec = basis.grid_to_basis_fast(
        u_grid, 
        theta_1d, 
        phi_1d, 
        weights=np.sqrt(mw_weights_th), 
        reg_lambda=0.0, # Zero regularization for exact quadrature
        vector_type='scalar'
    )
    
    diff = coeffs_rec - coeffs_true
    err = np.linalg.norm(diff) / np.linalg.norm(coeffs_true)
    print(f"Rel Error: {err:.5e}")
    
    # Ideally should be < 1e-14
    assert err < 1e-14, f"MW Scalar failed: {err}"

def test_general_grid_exactness():
    """Verify Exact Recovery on an arbitrary Regular Grid using compute_exact_weights."""
    print("\n--- GENERAL GRID SCALAR EXACTNESS ---")
    
    L = 10
    basis = SHBasis(Nmax=L, Mmax=L, mean_free=False)
    
    # Create an Arbitrary Regular Grid (e.g. excluding poles or including them)
    # Ensure N_theta >= L+1
    N_theta = L + 5
    N_phi = 2 * L + 1
    
    # Linear spacing (exclude endpoints to avoid singularity 1/sin issues in weights if any?)
    # Actually, moment matching handles poles if 1/sin is not in the system matrix P.
    theta_1d = np.linspace(0.1, np.pi-0.1, N_theta) 
    phi_1d = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
    
    theta_grid, phi_grid = np.meshgrid(theta_1d, phi_1d, indexing='ij')
    
    # Compute Exact Weights for this theta grid
    weights_th = SHBasis.compute_exact_weights(theta_1d, N_theta)
    
    # Test
    np.random.seed(999)
    coeffs_true = np.random.randn(basis.index_length)
    
    # Evaluate using basis
    # We need to manually construct Grid object or use arrays?
    # SHBasis.evaluate usually takes Grid object.
    # We can mock a Grid or use raw arrays if supported.
    # Grid object requires specific internal structure.
    # Let's create a partial Grid.
    
    # Actually, we can just use `grid_to_basis_fast` with arrays directly if `u_grid` is passed.
    # But we need `u_grid`.
    # Let's use `basis.evaluate` which needs a Grid object.
    
    # Create a dummy Grid object
    # We will instantiate Grid directly
    lat = 90 - np.rad2deg(theta_grid)
    lon = np.rad2deg(phi_grid)
    grid = Grid(lat, lon)
    
    u_flat = basis.evaluate(coeffs_true, grid, vector_type='scalar')
    u_grid = u_flat.reshape(N_theta, N_phi)
    
    # Recover
    print("Running Fast SHT (General Grid) with Computed Weights...")
    coeffs_rec = basis.grid_to_basis_fast(
        u_grid, 
        theta_1d, 
        phi_1d, 
        weights=np.sqrt(weights_th), 
        reg_lambda=0.0,
        vector_type='scalar'
    )
    
    diff = coeffs_rec - coeffs_true
    err = np.linalg.norm(diff) / np.linalg.norm(coeffs_true)
    print(f"Rel Error: {err:.5e}")
    
    assert err < 1e-14, f"General Grid Scalar failed: {err}"

if __name__ == "__main__":
    test_mw_exactness_scalar()
    test_general_grid_exactness()
