"""Tests for selecting the horizontal calculation basis."""

import numpy as np
import pytest

from pynamit.default_run import run_pynamit
from pynamit.math.constants import RE
from pynamit.math.tensor_operations import tensor_pinv, weighted_tensor_pinv
from pynamit.simulation.dynamics import Dynamics


def test_default_horizontal_basis_is_sh(tmp_path):
    """Default horizontal basis is SH with radial continuation."""
    dynamics = Dynamics(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        artifact_storage="netcdf",
    )

    assert dynamics.settings.attrs["horizontal_basis_kind"] == "SH"
    assert dynamics.horizontal_basis is dynamics.solid_harmonics.basis
    geometry = dynamics.state.geometry
    assert geometry.horizontal_solid_projection_is_identity
    assert geometry._horizontal_to_solid_harmonic is None
    assert geometry._solid_harmonic_to_horizontal is None
    np.testing.assert_allclose(
        geometry.horizontal_potential_to_gridded_JS,
        geometry.solid_harmonic_poloidal_to_gridded_sheet_current,
    )
    np.testing.assert_allclose(
        geometry.horizontal_poloidal_to_gridded_sheet_current(
            geometry.spherical_transform
        ),
        geometry.solid_harmonic_poloidal_to_gridded_sheet_current,
    )
    assert geometry._horizontal_to_solid_harmonic is None


def test_horizontal_basis_kind_is_persisted(tmp_path):
    """Explicit horizontal basis choice keeps SH radial continuation."""
    dynamics = Dynamics(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        horizontal_basis_kind="cs",
        artifact_storage="netcdf",
    )

    assert dynamics.settings.attrs["horizontal_basis_kind"] == "CS"
    assert dynamics.horizontal_basis is dynamics.state.basis
    assert dynamics.solid_harmonics.basis is not dynamics.horizontal_basis
    assert dynamics.input_field_spaces["jr"].mean_free
    assert dynamics.input_field_spaces["Br"].mean_free
    assert dynamics.input_field_spaces["u"].mean_free
    assert not dynamics.input_field_spaces["conductance"].mean_free
    assert dynamics.output_field_spaces["state"].mean_free


def test_area_weighted_least_squares_option_is_persisted(tmp_path):
    """Area-weighted fits are a persisted global option."""
    dynamics = Dynamics(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        area_weighted_least_squares=True,
        artifact_storage="netcdf",
    )

    geometry = dynamics.state.geometry

    assert dynamics.settings.attrs["area_weighted_least_squares"] == 1
    assert dynamics.input_timeseries.area_weighted_least_squares
    assert dynamics.output_timeseries.area_weighted_least_squares
    assert geometry.area_weighted_least_squares
    np.testing.assert_allclose(
        geometry.grid_sqrt_weights(),
        np.sqrt(geometry.cs_basis.unit_area),
    )
    np.testing.assert_allclose(
        geometry.grid_sqrt_weights(vector=True),
        np.tile(np.sqrt(geometry.cs_basis.unit_area), (2, 1)),
    )


def test_cs_horizontal_basis_runs_with_cs_outputs(tmp_path):
    """CS horizontal basis routes the state through CS coefficients."""
    dynamics = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        use_wind=False,
        jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    state = dynamics.output_timeseries.datasets["state"]
    assert "CS_m_ind" in state
    assert "CS_m_imp" in state
    assert state["CS_m_ind"].shape[-1] == dynamics.state.basis.index_length
    assert dynamics.horizontal_basis is dynamics.state.basis
    assert dynamics.horizontal_spherical_transform is dynamics.state.geometry.spherical_transform


def test_cs_horizontal_basis_runs_with_pfac(tmp_path):
    """CS horizontal basis can use SH radial continuation for PFAC."""
    dynamics = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=False,
        use_wind=False,
        jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = dynamics.state.geometry
    pfac = geometry.T_to_Ve.values

    assert dynamics.horizontal_basis is dynamics.state.basis
    assert dynamics.solid_harmonics.basis is not dynamics.horizontal_basis
    assert pfac.shape == (
        dynamics.horizontal_basis.index_length,
        dynamics.horizontal_basis.index_length,
    )
    assert np.linalg.norm(pfac) > 0.0
    assert np.all(np.isfinite(pfac))
    assert np.all(np.isfinite(geometry.m_imp_to_gridded_JS))


def test_cs_horizontal_basis_supports_rm_solid_harmonics(tmp_path):
    """CS horizontal basis can use solid harmonics for RM terms."""
    dynamics = Dynamics(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * RE,
        ignore_PFAC=True,
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    geometry = dynamics.state.geometry

    assert geometry.m_ind_to_gridded_JS.shape == (
        2,
        geometry.grid.size,
        dynamics.horizontal_basis.index_length,
    )
    assert geometry.Br_to_gridded_JS.shape == geometry.m_ind_to_gridded_JS.shape
    assert np.all(np.isfinite(geometry.m_ind_to_gridded_JS))
    assert np.all(np.isfinite(geometry.Br_to_gridded_JS))


def test_cs_horizontal_basis_supports_connected_hemispheres(tmp_path):
    """CS horizontal basis can evaluate conjugate Helmholtz terms."""
    dynamics = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        connect_hemispheres=True,
        use_wind=False,
        jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = dynamics.state.geometry

    assert geometry.cp_spherical_transform.helmholtz_coeffs_to_gridded_vector.shape == (
        2,
        geometry.cp_grid.size,
        2,
        dynamics.horizontal_basis.index_length,
    )
    assert geometry.E_coeffs_to_E_apex_ll_diff.shape[-2:] == (
        2,
        dynamics.horizontal_basis.index_length,
    )
    assert np.all(
        np.isfinite(geometry.cp_spherical_transform.helmholtz_coeffs_to_gridded_vector)
    )
    assert np.all(np.isfinite(geometry.E_coeffs_to_E_apex_ll_diff))


def test_connected_E_apex_constraint_operator_is_lazy(tmp_path):
    """Connected E-apex constraint stays operator-backed."""
    dynamics = Dynamics(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        connect_hemispheres=True,
        artifact_storage="netcdf",
    )

    geometry = dynamics.state.geometry
    operator = geometry.E_coeffs_to_E_apex_ll_diff_operator
    assert operator is not None
    assert geometry._E_coeffs_to_E_apex_ll_diff is None

    rng = np.random.default_rng(20260612)
    coeffs = rng.standard_normal(operator.input_shape)

    actual = operator.matvec(coeffs).reshape(operator.output_shape)
    explicit = geometry.E_coeffs_to_E_apex_ll_diff
    expected = np.tensordot(explicit, coeffs, axes=([2, 3], [0, 1]))

    np.testing.assert_allclose(actual, expected)


def test_cs_horizontal_basis_combines_pfac_rm_and_connected_terms(tmp_path):
    """CS horizontal basis supports the combined radial/coupled path."""
    dynamics = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * RE,
        ignore_PFAC=False,
        connect_hemispheres=True,
        use_wind=False,
        jr_projection_basis="CS",
        Br_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = dynamics.state.geometry

    assert geometry.T_to_Ve.shape == (
        dynamics.horizontal_basis.index_length,
        dynamics.horizontal_basis.index_length,
    )
    assert geometry.Br_to_gridded_JS.shape == geometry.m_ind_to_gridded_JS.shape
    assert geometry.E_coeffs_to_E_apex_ll_diff.shape[-2:] == (
        2,
        dynamics.horizontal_basis.index_length,
    )
    assert np.linalg.norm(geometry.T_to_Ve.values) > 0.0
    assert np.all(np.isfinite(geometry.T_to_Ve.values))
    assert np.all(np.isfinite(geometry.Br_to_gridded_JS))


def test_cs_to_solid_harmonic_projection_matches_grid_least_squares(tmp_path):
    """CS-to-SH projection uses the standard grid LS projection."""
    dynamics = Dynamics(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    geometry = dynamics.state.geometry
    expected = tensor_pinv(
        geometry.solid_harmonic_transform.scalar_coeffs_to_grid,
        n_leading_flattened=1,
    ) @ geometry.spherical_transform.scalar_coeffs_to_grid

    np.testing.assert_allclose(geometry.horizontal_to_solid_harmonic, expected)

    rng = np.random.default_rng(20260520)
    radial_coeffs = rng.standard_normal(dynamics.solid_harmonics.basis.index_length)
    cs_coeffs = geometry.solid_harmonic_transform.scalar_coeffs_to_grid @ radial_coeffs

    np.testing.assert_allclose(
        geometry.horizontal_to_solid_harmonic @ cs_coeffs,
        radial_coeffs,
        atol=1e-10,
    )


def test_cs_to_solid_harmonic_supports_area_weighted_projection(tmp_path):
    """CS-to-SH projection can use CS cell-area weighting."""
    dynamics = Dynamics(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
        horizontal_basis_kind="CS",
        area_weighted_least_squares=True,
        artifact_storage="netcdf",
    )

    geometry = dynamics.state.geometry
    expected = weighted_tensor_pinv(
        geometry.solid_harmonic_transform.scalar_coeffs_to_grid,
        sqrt_weights=np.sqrt(geometry.cs_basis.unit_area),
        n_leading_flattened=1,
    ) @ geometry.spherical_transform.scalar_coeffs_to_grid

    np.testing.assert_allclose(geometry.horizontal_to_solid_harmonic, expected)


def test_invalid_horizontal_basis_kind_is_rejected(tmp_path):
    """Unknown horizontal basis names fail early."""
    with pytest.raises(ValueError, match="horizontal_basis_kind"):
        Dynamics(
            run_directory=str(tmp_path / "run"),
            Nmax=2,
            Mmax=1,
            Ncs=4,
            ignore_PFAC=True,
            horizontal_basis_kind="spectral",
            artifact_storage="netcdf",
        )
