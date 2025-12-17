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
    print(f"Median Error Phi:   {np.median(err_ph)}")
    print(f"Max Error Theta:    {np.max(err_th)}")
    print(f"Max Error Phi:      {np.max(err_ph)}")
    
    # Tolerances might need adjustment depending on N=20 resolution
    assert np.median(err_th) < 1e-1
    assert np.median(err_ph) < 1e-1
    
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
