"""Tests for physical magnetic variables and private coordinate maps."""

import numpy as np
import pytest

from pynamit.simulation.api import Simulation


@pytest.mark.parametrize("horizontal_basis_kind", ["SH", "CS"])
def test_physical_magnetic_coordinates_roundtrip(tmp_path, horizontal_basis_kind):
    """Physical fields invert their private potential maps."""
    simulation = Simulation(
        run_directory=tmp_path / horizontal_basis_kind,
        Nmax=3,
        Mmax=2,
        Ncs=6,
        horizontal_basis_kind=horizontal_basis_kind,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
        backend="numpy",
    )
    geometry = simulation.geometry
    rng = np.random.default_rng(20260729)

    induced_poloidal_potential = rng.standard_normal(geometry.poloidal_basis.index_length)
    induced_Br = geometry.induced_poloidal_potential_to_Br_operator.matvec(
        induced_poloidal_potential
    )
    reconstructed_poloidal_potential = geometry.induced_Br_to_poloidal_potential_operator.matvec(
        induced_Br
    )
    np.testing.assert_allclose(
        reconstructed_poloidal_potential, induced_poloidal_potential, rtol=1e-14, atol=1e-14
    )

    toroidal_potential = simulation.response.project_surface_scalar_mean_free(
        rng.standard_normal(geometry.horizontal_basis.index_length)
    )
    boundary_jr = geometry.toroidal_potential_to_boundary_jr_operator.matvec(toroidal_potential)
    reconstructed_toroidal_potential = geometry.boundary_jr_to_toroidal_potential_operator.matvec(
        boundary_jr
    )
    np.testing.assert_allclose(
        reconstructed_toroidal_potential, toroidal_potential, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        geometry.toroidal_potential_to_boundary_jr_operator.matvec(
            reconstructed_toroidal_potential
        ),
        boundary_jr,
        rtol=1e-10,
        atol=1e-12,
    )


def test_output_schema_persists_only_physical_magnetic_variables(tmp_path):
    """Potential coordinates do not leak into persisted model output."""
    simulation = Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=4,
        enable_pfac_coupling=False,
        artifact_storage="netcdf",
    )

    assert simulation.run_data.schema.output_variables == {
        "dynamic": ("induced_Br", "boundary_jr", "Phi", "W"),
        "equilibrium": ("induced_Br", "boundary_jr", "Phi", "W"),
    }


def test_gap_response_has_physical_domain_and_codomain(tmp_path):
    """The cached gap map is explicitly current to radial field."""
    simulation = Simulation(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=4,
        enable_pfac_coupling=True,
        artifact_storage="netcdf",
        backend="numpy",
    )
    matrix = simulation.geometry.boundary_jr_to_gap_Br_matrix
    simulation.run_data.save_gap_Br_response_if_missing(matrix)

    stored = simulation.run_data.gap_Br_response
    assert stored is not None
    assert stored.attrs["input_quantity"] == "boundary_jr_at_RI"
    assert stored.attrs["output_quantity"] == "unshielded_gap_Br_at_RI"
    assert stored.shape == (
        simulation.geometry.poloidal_basis.index_length,
        simulation.geometry.horizontal_basis.index_length,
    )
