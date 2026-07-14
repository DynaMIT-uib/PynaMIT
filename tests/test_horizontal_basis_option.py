"""Tests for selecting the horizontal calculation basis."""

import numpy as np
import pytest

from pynamit.simulation.workflows.standard import run_pynamit
from pynamit.math.constants import RE
from pynamit.math.tensor_operations import tensor_pinv, weighted_tensor_pinv
from pynamit.simulation.api import Simulation


def test_default_horizontal_basis_is_sh(tmp_path):
    """Default horizontal basis is SH with radial continuation."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )

    assert simulation.run_data.config.horizontal_basis_kind == "SH"
    assert simulation.geometry.horizontal_basis is simulation.geometry.solid_harmonics.basis
    geometry = simulation.geometry
    schema = simulation.run_data.schema
    assert geometry.horizontal_basis.mean_free
    assert not schema.sh_basis.mean_free
    assert schema.sh_basis.index_length == geometry.horizontal_basis.index_length + 1
    assert geometry.horizontal_solid_projection_is_identity
    assert "horizontal_to_solid_harmonic" not in geometry.__dict__
    assert "solid_harmonic_to_horizontal" not in geometry.__dict__


def test_horizontal_basis_kind_is_persisted(tmp_path):
    """Explicit horizontal basis choice keeps SH radial continuation."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        horizontal_basis_kind="cs",
        artifact_storage="netcdf",
    )

    assert simulation.run_data.config.horizontal_basis_kind == "CS"
    assert simulation.run_data.schema.horizontal_basis is simulation.geometry.horizontal_basis
    assert simulation.geometry.solid_harmonics.basis is not simulation.geometry.horizontal_basis
    assert simulation.run_data.schema.input_field_spaces["jr"].mean_free
    assert simulation.run_data.schema.input_field_spaces["Br"].mean_free
    assert simulation.run_data.schema.input_field_spaces["u"].mean_free
    assert not simulation.run_data.schema.input_field_spaces["resistance"].mean_free
    assert simulation.run_data.schema.output_field_spaces["state"].mean_free


def test_area_weighted_least_squares_option_is_persisted(tmp_path):
    """Area-weighted fits are a persisted global option."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        area_weighted_least_squares=True,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry

    assert simulation.run_data.config.area_weighted_least_squares
    assert geometry.area_weighted_least_squares
    np.testing.assert_allclose(
        geometry.model_grid_sqrt_weights(), np.sqrt(simulation.run_data.schema.cs_basis.unit_area)
    )
    np.testing.assert_allclose(
        geometry.model_grid_sqrt_weights(vector=True),
        np.tile(np.sqrt(simulation.run_data.schema.cs_basis.unit_area), (2, 1)),
    )


def test_cs_horizontal_basis_runs_with_cs_outputs(tmp_path):
    """CS horizontal basis routes the state through CS coefficients."""
    simulation = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        use_wind=False,
        jr_projection_basis="CS",
        resistance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    state = simulation.run_data.output_series.datasets["state"]
    assert "CS_m_ind" in state
    assert "CS_m_imp" in state
    assert state["CS_m_ind"].shape[-1] == simulation.geometry.horizontal_basis.index_length
    assert simulation.run_data.schema.horizontal_basis is simulation.geometry.horizontal_basis


def test_cs_horizontal_basis_runs_with_pfac(tmp_path):
    """CS horizontal basis can use SH radial continuation for PFAC."""
    simulation = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=True,
        use_wind=False,
        jr_projection_basis="CS",
        resistance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry
    pfac = geometry.pfac_coupling_matrix
    assert isinstance(pfac, np.ndarray)
    assert not pfac.flags.writeable

    assert simulation.run_data.schema.horizontal_basis is simulation.geometry.horizontal_basis
    assert simulation.geometry.solid_harmonics.basis is not simulation.geometry.horizontal_basis
    assert pfac.shape == (
        simulation.geometry.horizontal_basis.index_length,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert np.linalg.norm(pfac) > 0.0
    assert np.all(np.isfinite(pfac))
    assert np.all(np.isfinite(geometry.m_imp_to_gridded_JS()))


def test_cs_horizontal_basis_supports_rm_solid_harmonics(tmp_path):
    """CS horizontal basis can use solid harmonics for RM terms."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * RE,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry

    m_ind_to_JS = geometry.m_ind_to_gridded_JS()
    Br_to_JS = geometry.Br_to_gridded_JS()

    assert m_ind_to_JS.shape == (
        2,
        geometry.model_grid.size,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert Br_to_JS.shape == m_ind_to_JS.shape
    assert np.all(np.isfinite(m_ind_to_JS))
    assert np.all(np.isfinite(Br_to_JS))


def test_cs_horizontal_basis_supports_connected_hemispheres(tmp_path):
    """CS horizontal basis can evaluate conjugate Helmholtz terms."""
    simulation = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=True,
        use_wind=False,
        jr_projection_basis="CS",
        resistance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry

    assert geometry.conjugate_horizontal_transform.helmholtz_coeffs_to_gridded_vector.shape == (
        2,
        geometry.conjugate_grid.size,
        2,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert geometry.interhemispheric_electric_field_difference_matrix.shape[-2:] == (
        2,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert np.all(
        np.isfinite(geometry.conjugate_horizontal_transform.helmholtz_coeffs_to_gridded_vector)
    )
    assert np.all(np.isfinite(geometry.interhemispheric_electric_field_difference_matrix))


def test_connected_E_apex_constraint_operator_is_lazy(tmp_path):
    """Connected E-apex constraint stays operator-backed."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        enable_interhemispheric_coupling=True,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    operator = geometry.interhemispheric_electric_field_difference_operator
    assert operator is not None
    assert "interhemispheric_electric_field_difference_matrix" not in geometry.__dict__

    rng = np.random.default_rng(20260612)
    coeffs = rng.standard_normal(operator.input_shape)

    actual = operator.matvec(coeffs).reshape(operator.output_shape)
    explicit = geometry.interhemispheric_electric_field_difference_matrix
    expected = np.tensordot(explicit, coeffs, axes=([2, 3], [0, 1]))

    np.testing.assert_allclose(actual, expected)


def test_cs_horizontal_basis_combines_pfac_rm_and_connected_terms(tmp_path):
    """CS horizontal basis supports the combined radial/coupled path."""
    simulation = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        RM=4 * RE,
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        use_wind=False,
        jr_projection_basis="CS",
        Br_projection_basis="CS",
        resistance_projection_basis="CS",
        u_projection_basis="CS",
        run_directory=str(tmp_path / "run"),
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
        least_squares_solver="normal_pinv",
    )

    geometry = simulation.geometry

    assert geometry.pfac_coupling_matrix.shape == (
        simulation.geometry.horizontal_basis.index_length,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert geometry.Br_to_gridded_JS().shape == (geometry.m_ind_to_gridded_JS().shape)
    assert geometry.interhemispheric_electric_field_difference_matrix.shape[-2:] == (
        2,
        simulation.geometry.horizontal_basis.index_length,
    )
    assert np.linalg.norm(geometry.pfac_coupling_matrix) > 0.0
    assert np.all(np.isfinite(geometry.pfac_coupling_matrix))
    assert np.all(np.isfinite(geometry.Br_to_gridded_JS()))


def test_cs_to_solid_harmonic_projection_matches_grid_least_squares(tmp_path):
    """CS-to-SH projection uses the standard grid LS projection."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    expected = (
        tensor_pinv(geometry.solid_harmonic_transform.scalar_coeffs_to_grid, n_leading_flattened=1)
        @ geometry.horizontal_transform.scalar_coeffs_to_grid
    )

    np.testing.assert_allclose(geometry.horizontal_to_solid_harmonic, expected)

    rng = np.random.default_rng(20260520)
    radial_coeffs = rng.standard_normal(simulation.geometry.solid_harmonics.basis.index_length)
    cs_coeffs = geometry.solid_harmonic_transform.scalar_coeffs_to_grid @ radial_coeffs

    np.testing.assert_allclose(
        geometry.horizontal_to_solid_harmonic @ cs_coeffs, radial_coeffs, atol=1e-10
    )


def test_cs_to_solid_harmonic_supports_area_weighted_projection(tmp_path):
    """CS-to-SH projection can use CS cell-area weighting."""
    simulation = Simulation(
        run_directory=str(tmp_path / "run"),
        Nmax=2,
        Mmax=1,
        Ncs=8,
        enable_pfac_coupling=False,
        horizontal_basis_kind="CS",
        area_weighted_least_squares=True,
        artifact_storage="netcdf",
    )

    geometry = simulation.geometry
    expected = (
        weighted_tensor_pinv(
            geometry.solid_harmonic_transform.scalar_coeffs_to_grid,
            sqrt_weights=np.sqrt(simulation.run_data.schema.cs_basis.unit_area),
            n_leading_flattened=1,
        )
        @ geometry.horizontal_transform.scalar_coeffs_to_grid
    )

    np.testing.assert_allclose(geometry.horizontal_to_solid_harmonic, expected)


def test_invalid_horizontal_basis_kind_is_rejected(tmp_path):
    """Unknown horizontal basis names fail early."""
    with pytest.raises(ValueError, match="horizontal_basis_kind"):
        Simulation(
            run_directory=str(tmp_path / "run"),
            Nmax=2,
            Mmax=1,
            Ncs=4,
            enable_pfac_coupling=False,
            horizontal_basis_kind="spectral",
            artifact_storage="netcdf",
        )
