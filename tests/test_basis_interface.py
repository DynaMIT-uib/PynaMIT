"""Tests for basis interface enforcement."""

import pytest
import numpy as np

from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.primitives.basis import Basis, EvaluableBasis, GridBasis, is_grid_basis
from pynamit.spherical_harmonics.sh_basis import SHBasis


def test_concrete_bases_implement_basis_interface():
    """Concrete basis classes satisfy the shared metadata interface."""
    sh_basis = SHBasis(3, 3)
    cs_basis = CSBasis(4)

    assert isinstance(sh_basis, Basis)
    assert isinstance(sh_basis, EvaluableBasis)
    assert isinstance(cs_basis, Basis)
    assert isinstance(cs_basis, GridBasis)
    assert isinstance(cs_basis, EvaluableBasis)
    assert is_grid_basis(cs_basis)
    assert sh_basis.kind == "SH"
    assert cs_basis.kind == "CS"
    assert cs_basis.index_length == cs_basis.arr_theta.size
    sh_basis.validate_metadata()
    cs_basis.validate_metadata()


def test_csbasis_evaluates_with_finite_difference_derivatives():
    """CSBasis exposes native finite-difference derivative matrices."""
    cs_basis = CSBasis(8)
    grid = type("GridLike", (), {"theta": cs_basis.arr_theta, "phi": cs_basis.arr_phi})()

    constant = np.ones(cs_basis.index_length)
    cos_theta = np.cos(np.deg2rad(cs_basis.arr_theta))
    expected_dtheta = -np.sin(np.deg2rad(cs_basis.arr_theta))

    G = cs_basis.get_G(grid)
    G_theta = cs_basis.get_G(grid, derivative="theta")

    np.testing.assert_allclose(G @ constant, constant)
    np.testing.assert_allclose(G_theta @ constant, 0.0, atol=1e-12)
    np.testing.assert_allclose(G_theta @ cos_theta, expected_dtheta, atol=1e-2)


def test_csbasis_derivatives_match_first_spherical_harmonics():
    """CS derivatives match first-degree sphere functions."""
    cs_basis = CSBasis(8)
    grid = type("GridLike", (), {"theta": cs_basis.arr_theta, "phi": cs_basis.arr_phi})()
    theta = np.deg2rad(cs_basis.arr_theta)
    phi = np.deg2rad(cs_basis.arr_phi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    fields = [
        (x, np.cos(theta) * np.cos(phi), -np.sin(phi), -2 * x),
        (y, np.cos(theta) * np.sin(phi), np.cos(phi), -2 * y),
        (z, -np.sin(theta), np.zeros_like(theta), -2 * z),
    ]

    G_theta = cs_basis.get_G(grid, derivative="theta")
    G_phi = cs_basis.get_G(grid, derivative="phi")
    laplacian = cs_basis.laplacian()

    for values, expected_theta, expected_phi, expected_laplacian in fields:
        np.testing.assert_allclose(G_theta @ values, expected_theta, atol=1e-2)
        np.testing.assert_allclose(G_phi @ values, expected_phi, atol=1e-2)
        np.testing.assert_allclose(laplacian @ values, expected_laplacian, atol=1.2e-1)


def test_csbasis_derivative_errors_decrease_with_resolution():
    """CS finite-difference errors decrease with finer CS grids."""

    def rms_errors(N):
        cs_basis = CSBasis(N)
        grid = type("GridLike", (), {"theta": cs_basis.arr_theta, "phi": cs_basis.arr_phi})()
        theta = np.deg2rad(cs_basis.arr_theta)
        phi = np.deg2rad(cs_basis.arr_phi)
        values = np.sin(theta) * np.cos(phi)

        theta_error = cs_basis.get_G(grid, derivative="theta") @ values
        theta_error -= np.cos(theta) * np.cos(phi)
        phi_error = cs_basis.get_G(grid, derivative="phi") @ values
        phi_error -= -np.sin(phi)
        laplacian_error = cs_basis.laplacian() @ values
        laplacian_error -= -2 * values

        return np.array(
            [
                np.sqrt(np.mean(theta_error**2)),
                np.sqrt(np.mean(phi_error**2)),
                np.sqrt(np.mean(laplacian_error**2)),
            ]
        )

    coarse = rms_errors(8)
    fine = rms_errors(12)

    assert np.all(fine < 0.75 * coarse)


def test_shbasis_mean_free_option_matches_legacy_nmin():
    """Mean-free SH spaces keep the old Nmin=1 behavior."""
    legacy = SHBasis(3, 2, Nmin=1)
    mean_free = SHBasis(3, 2, mean_free=True)
    extended = mean_free.get_extended_basis()

    assert mean_free.scalar_fields_are_mean_free_by_construction()
    assert mean_free.Nmin == legacy.Nmin == 1
    assert mean_free.index_length == legacy.index_length
    assert not extended.scalar_fields_are_mean_free_by_construction()
    assert extended.Nmin == 0
    assert extended.index_length > mean_free.index_length


def test_shbasis_rejects_inconsistent_mean_free_options():
    """Nmin and mean_free must describe the same scalar space."""
    with pytest.raises(ValueError, match="inconsistent scalar-space options"):
        SHBasis(3, 2, Nmin=0, mean_free=True)


def test_incomplete_basis_subclass_is_rejected():
    """Subclasses must declare the required metadata fields."""

    class IncompleteBasis(Basis):
        kind = "incomplete"

    with pytest.raises(TypeError):
        IncompleteBasis()


def test_evaluable_basis_subclass_must_implement_get_G():
    """Evaluable bases must define grid evaluation."""

    class IncompleteEvaluableBasis(EvaluableBasis):
        kind = "incomplete"
        index_names = ["i"]
        index_length = 1
        index_arrays = [[0]]
        minimum_phi_sampling = 1
        caching = False

    with pytest.raises(TypeError):
        IncompleteEvaluableBasis()
