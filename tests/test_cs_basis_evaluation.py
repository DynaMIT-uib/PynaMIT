"""Test CSBasis evaluation and differentiation."""

import numpy as np
import scipy.sparse

from pynamit.cubed_sphere.cs_basis import CSBasis


def _rel_rms(err: np.ndarray, ref: np.ndarray) -> float:
    denom = np.sqrt(np.mean(np.asarray(ref) ** 2))
    if denom <= 1e-14:
        return np.sqrt(np.mean(np.asarray(err) ** 2))
    return np.sqrt(np.mean(np.asarray(err) ** 2)) / denom


def test_cs_basis_differentiation():
    """CS derivatives should follow scaled phi semantics and be accurate."""
    basis = CSBasis(20)
    theta_rad = np.deg2rad(basis.theta)
    phi_rad = np.deg2rad(basis.phi)

    # f = sin(theta) cos(phi) (bounded scaled-phi derivative)
    vals = np.sin(theta_rad) * np.cos(phi_rad)

    # d/dtheta f
    dV_dth_expected = np.cos(theta_rad) * np.cos(phi_rad)
    # get_G(..., derivative="phi") is defined as (1/sin theta) * d/dphi.
    dV_dph_scaled_expected = -np.sin(phi_rad)

    dV_dth_num = np.asarray(basis.basis_to_grid(vals, grid=basis, derivative="theta"))
    dV_dph_num = np.asarray(basis.basis_to_grid(vals, grid=basis, derivative="phi"))

    rel_th = _rel_rms(dV_dth_num - dV_dth_expected, dV_dth_expected)
    rel_ph = _rel_rms(dV_dph_num - dV_dph_scaled_expected, dV_dph_scaled_expected)

    assert rel_th < 2e-3, f"d/dtheta relative RMS error too high: {rel_th:.3e}"
    assert rel_ph < 2e-3, f"scaled d/dphi relative RMS error too high: {rel_ph:.3e}"


def test_cs_basis_laplacian():
    """CS Laplacian should recover the Y_1,0 eigenvalue to high accuracy."""
    basis = CSBasis(30)
    theta_rad = np.deg2rad(basis.theta)
    vals = np.cos(theta_rad)

    lap_vals = np.asarray(basis.laplacian(r=1.0).dot(vals))
    expected = -2.0 * vals
    rel = _rel_rms(lap_vals - expected, expected)

    assert rel < 5e-3, f"Laplacian relative RMS error too high: {rel:.3e}"


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
