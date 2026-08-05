"""PynaMIT integration tests for Kompe spherical transforms."""

import numpy as np
from kompe import GlobalCSBasis, SHBasis, SphericalGrid, SphericalTransform

import pynamit
from pynamit.fields import FieldCoefficients, FieldSpace


def _regular_grid():
    lat = np.linspace(-70.0, 70.0, 11)
    lon = np.linspace(0.0, 330.0, 12)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return SphericalGrid(lat=lat_grid.reshape(-1), lon=lon_grid.reshape(-1))


def test_transform_accepts_pynamit_field_coefficients():
    """Kompe synthesis reads PynaMIT's canonical coefficient arrays."""
    basis = SHBasis(3, 2, mean_free=True)
    field_space = FieldSpace(basis, field_type="scalar")
    coefficients = np.zeros(basis.index_length)
    coefficients[1] = 1.0
    field = FieldCoefficients(field_space, coefficients)
    transform = SphericalTransform(basis, _regular_grid())

    np.testing.assert_allclose(
        transform.synthesize_scalar(field), transform.synthesize_scalar(field.array)
    )


def test_basis_evaluator_retains_collaborator_matrix_properties():
    """Retain the historical evaluator matrix spellings."""
    from pynamit.sphere import BasisEvaluator

    evaluator = BasisEvaluator(SHBasis(3, 2, mean_free=True), _regular_grid())

    assert issubclass(BasisEvaluator, SphericalTransform)
    assert pynamit.BasisEvaluator is BasisEvaluator
    assert evaluator.G is evaluator.scalar_synthesis_matrix
    assert evaluator.G_helmholtz is evaluator.helmholtz_synthesis_matrix


def test_field_space_applies_constraint_after_kompe_sample_analysis():
    """Keep PynaMIT constraints separate from Kompe sample analysis."""
    basis = GlobalCSBasis(4)
    field_space = FieldSpace(basis, field_type="scalar", mean_free=True)
    grid = basis.mesh.cell_centers
    values = np.linspace(0.0, 1.0, basis.index_length) + 3.0
    transform = SphericalTransform(basis, grid, grid_remap_basis=basis)

    analyzed = transform.analyze_scalar_samples(values, input_grid=grid, analysis_basis=basis)
    constrained = field_space.project_mean_free(analyzed[0])

    assert analyzed.shape == (1, basis.index_length)
    np.testing.assert_allclose(basis.scalar_mean(constrained), 0.0, atol=1e-12)
