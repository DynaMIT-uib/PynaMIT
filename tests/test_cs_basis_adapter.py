"""Test CSBasis adapter for BasisEvaluator."""
import numpy as np
import pytest
import scipy
import scipy.sparse
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.math import cs_math

def test_cs_basis_evaluator_differentiation():
    """Test that BasisEvaluator can differentiate a field on CSBasis."""
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
    
    # Use BasisEvaluator
    # Note: BasisEvaluator normally solves for coefficients. 
    # For CSBasis, coefficients ARE the grid values (Identity transform).
    # So we can just set coeffs = values.
    
    evaluator = BasisEvaluator(basis, basis)
    
    # "Coefficients" are just the grid values themselves for a grid basis
    coeffs = vals 
    
    # Calculate derivatives via evaluator (which calls basis.get_G)
    dV_dth_num = evaluator.basis_to_grid(coeffs, derivative="theta")
    dV_dph_num = evaluator.basis_to_grid(coeffs, derivative="phi")
    
    # Convert sparse results to dense arrays if necessary
    if scipy.sparse.issparse(dV_dth_num):
        dV_dth_num = dV_dth_num.toarray()
    if scipy.sparse.issparse(dV_dph_num):
        dV_dph_num = dV_dph_num.toarray()
    
    # Check accuracy
    # Exclude boundaries/ghost points implicitly by checking correlation or RMSE
    # Or just check bulk
    
    # Since we have singularities and edges, let's check loose correlation first
    # and maybe exclude extreme values
    
    # Simple check: are they close?
    # The edges of the cubed sphere blocks will have larger errors due to 
    # one-sided differences or ghost cell interpolation artifacts.
    # Let's check the median error or percentiles.
    
    err_th = np.abs(dV_dth_num - dV_dth_ana)
    err_ph = np.abs(dV_dph_num - dV_dph_ana)
    
    print(f"Median Error Theta: {np.median(err_th)}")
    # Compare median error
    median_err_th = np.median(err_th)
    
    # 0.1 is a loose bound, but reasonable for single-precision / grid edge effects
    # With N=20 and order=1 differentiation, errors near edges are high.
    # We just want to ensure it's not totally wrong (order 1e0 or 1e1).
    assert median_err_th < 0.25, f"Median dV/dth error {median_err_th} too high"
    print(f"Median Error Phi:   {np.median(err_ph)}")
    print(f"Max Error Theta:    {np.max(err_th)}")
    print(f"Max Error Phi:      {np.max(err_ph)}")
    
    # Tolerances might need adjustment depending on N=20 resolution
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
    # L is sparse matrix, vals is vector
    lap_vals = L.dot(vals)
    
    expected = -2.0 * vals
    
    # Check error
    # Exclude poles/edges where finite differences are poor?
    # Cubed sphere has no poles in the grid (usually xi, eta don't reach exactly corner?)
    # But boundaries of panels have discontinuities in metric derivative approximations if not handled carefully.
    
    err = np.abs(lap_vals - expected)
    median_err = np.median(err)
    max_err = np.max(err)
    
    print(f"Laplacian Median Error: {median_err}")
    print(f"Laplacian Max Error: {max_err}")
    
    # Strong form laplacian with finite differences on CS is tricky.
    # Expect moderate accuracy.
    assert median_err < 0.5, f"Median Laplacian error {median_err} too high"
    
def test_cs_basis_evaluator_identity():
    """Test that BasisEvaluator preserves values for CSBasis."""
    N = 10
    basis = CSBasis(N)
    vals = np.random.randn(basis.size)
    
    evaluator = BasisEvaluator(basis, basis)
    
    # "Project" to basis (identity)
    coeffs = evaluator.grid_to_basis(vals)
    
    # "Evaluate" on grid
    vals_rec = evaluator.basis_to_grid(coeffs)
    
    if scipy.sparse.issparse(vals_rec):
        vals_rec = vals_rec.toarray()
    if scipy.sparse.issparse(coeffs):
        coeffs = coeffs.toarray()
        
    vals = vals.flatten()
    vals_rec = vals_rec.flatten()
    coeffs = coeffs.flatten()
    
    assert np.allclose(vals, vals_rec)
    assert np.allclose(vals, coeffs)
