"""Unit tests for CSBasis Helmholtz coefficient semantics."""

from __future__ import annotations

import numpy as np

from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.primitives.grid import Grid
from pynamit.primitives.basis import get_repo_df_helmholtz_sign
from pynamit.spherical_harmonics.sh_basis import SHBasis


def _dense(arr: object) -> np.ndarray:
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr, dtype=float)


def test_cs_basis_extracts_helmholtz_channels_directly() -> None:
    """CSBasis should mirror SH-style channel extraction semantics."""
    rng = np.random.default_rng(0)
    basis = CSBasis(8)
    n = basis.index_length

    coeffs = rng.standard_normal((2, n))
    np.testing.assert_allclose(basis.get_poloidal_potential_coeffs(coeffs), coeffs[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(basis.get_toroidal_potential_coeffs(coeffs), coeffs[1], atol=0.0, rtol=0.0)

    flat = coeffs.reshape(2 * n)
    np.testing.assert_allclose(
        basis.get_poloidal_potential_coeffs(flat),
        coeffs[0],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        basis.get_toroidal_potential_coeffs(flat),
        coeffs[1],
        atol=0.0,
        rtol=0.0,
    )


def test_cs_basis_tangential_evaluate_uses_helmholtz_operator() -> None:
    """Tangential evaluation should apply the vector Helmholtz basis matrix."""
    rng = np.random.default_rng(1)
    basis = CSBasis(8)
    coeffs = rng.standard_normal((2, basis.index_length))

    eval_via_api = basis.evaluate(coeffs, basis.grid, vector_type="tangential")
    G_vec = basis.get_vector_basis_matrix(basis.grid)
    eval_via_matrix = np.tensordot(G_vec, coeffs, 2)

    np.testing.assert_allclose(eval_via_api, eval_via_matrix, rtol=1e-12, atol=1e-12)


def test_sh_df_channel_matches_active_surface_helmholtz_sign() -> None:
    """SH df basis should track the configured repo surface Helmholtz sign."""
    basis = SHBasis(6, 3)
    grid = Grid(lat=np.array([50.0, 65.0]), lon=np.array([10.0, 40.0]))

    g_th = _dense(basis.get_evaluation_matrix(grid, derivative="theta"))
    g_ph = _dense(basis.get_evaluation_matrix(grid, derivative="phi"))
    repo_df = _dense(basis.get_curl_matrix(grid))
    paper_df = np.array([-g_ph, g_th])

    np.testing.assert_allclose(
        repo_df,
        float(get_repo_df_helmholtz_sign()) * paper_df,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _dense(basis.get_vector_basis_matrix(grid))[:, :, 1, :],
        repo_df,
        rtol=1e-12,
        atol=1e-12,
    )


def test_cs_df_channel_matches_active_surface_helmholtz_sign() -> None:
    """CS df basis should track the configured repo surface Helmholtz sign."""
    basis = CSBasis(8)
    grid = basis.grid

    g_th = _dense(basis.get_evaluation_matrix(grid, derivative="theta"))
    g_ph = _dense(basis.get_evaluation_matrix(grid, derivative="phi"))
    repo_df = _dense(basis.get_vector_basis_matrix(grid))[:, :, 1, :]
    paper_df = np.array([-g_ph, g_th])

    np.testing.assert_allclose(
        repo_df,
        float(get_repo_df_helmholtz_sign()) * paper_df,
        rtol=1e-12,
        atol=1e-12,
    )
