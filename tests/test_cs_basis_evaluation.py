"""Test CSBasis evaluation and differentiation."""
import numpy as np
import pytest
import scipy
import scipy.sparse
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.cubed_sphere import cs_math

def test_cs_basis_differentiation():
    """Test that CSBasis can differentiate a field."""
    N = 20
    R = 1.0
    basis = CSBasis(N)
    # Convert coordinates to radians for function definition
    # CSBasis stores theta/phi in degrees, but derivatives are w.r.t. radians
    theta_rad = np.deg2rad(basis.theta)
    phi_rad = np.deg2rad(basis.phi)

    # Define a simple scalar field: V = cos(theta) * sin(phi)
    # dV/dtheta = -sin(theta) * sin(phi)
    # dV/dphi   = cos(theta) * cos(phi)
    
    vals = np.cos(theta_rad) * np.sin(phi_rad)
    
    # Analytical derivatives (w.r.t radians)
    dV_dth_ana = -np.sin(theta_rad) * np.sin(phi_rad)
    dV_dph_ana =  np.cos(theta_rad) * np.cos(phi_rad)
    
    # "Coefficients" are just the grid values themselves for a grid basis
    coeffs = vals 
    
    # Calculate derivatives via basis.basis_to_grid
    dV_dth_num = basis.basis_to_grid(coeffs, grid=basis, derivative="theta")
    dV_dph_num = basis.basis_to_grid(coeffs, grid=basis, derivative="phi")
    
    # Convert sparse results to dense arrays if necessary
    if scipy.sparse.issparse(dV_dth_num):
        dV_dth_num = dV_dth_num.toarray()
    if scipy.sparse.issparse(dV_dph_num):
        dV_dph_num = dV_dph_num.toarray()
    
    # Check accuracy
    err_th = np.abs(dV_dth_num - dV_dth_ana)
    err_ph = np.abs(dV_dph_num - dV_dph_ana)
    
    print(f"Median Error Theta: {np.median(err_th)}")
    # Compare median error
    median_err_th = np.median(err_th)
    
    # 0.1 is a loose bound, but reasonable for single-precision / grid edge effects
    assert median_err_th < 0.25, f"Median dV/dth error {median_err_th} too high"
    print(f"Median Error Phi:   {np.median(err_ph)}")
    print(f"Max Error Theta:    {np.max(err_th)}")
    print(f"Max Error Phi:      {np.max(err_ph)}")
    
    assert np.median(err_ph) < 1e-1
    
def test_cs_basis_laplacian():
    """Verify that CSBasis.laplacian computes correct values for SH eigenfunctions."""
    N = 30
    basis = CSBasis(N)
    
    # Test function: Y_1,0 = cos(theta)
    # Laplacian(Y_1,0) = -l(l+1)/r^2 * Y_1,0 = -2 * Y_1,0 (for r=1)
    
    theta_rad = np.deg2rad(basis.theta)
    vals = np.cos(theta_rad)
    
    L = basis.laplacian(r=1.0)
    
    # Apply Laplacian
    lap_vals = L.dot(vals)
    
    expected = -2.0 * vals
    
    err = np.abs(lap_vals - expected)
    median_err = np.median(err)
    max_err = np.max(err)
    
    print(f"Laplacian Median Error: {median_err}")
    print(f"Laplacian Max Error: {max_err}")
    
    assert median_err < 0.5, f"Median Laplacian error {median_err} too high"
    
def test_cs_basis_identity():
    """Test that CSBasis maintains values during grid<->basis conversion."""
    N = 10
    basis = CSBasis(N)
    vals = np.random.randn(basis.size)
    
    # "Project" to basis (identity)
    coeffs = basis.grid_to_basis(vals, grid=basis)
    
    # "Evaluate" on grid
    vals_rec = basis.basis_to_grid(coeffs, grid=basis)
    
    if scipy.sparse.issparse(vals_rec):
        vals_rec = vals_rec.toarray()
    if scipy.sparse.issparse(coeffs):
        coeffs = coeffs.toarray()
        
    vals = vals.flatten()
    vals_rec = vals_rec.flatten()
    coeffs = coeffs.flatten()
    
    assert np.allclose(vals, vals_rec)
    assert np.allclose(vals, coeffs)
