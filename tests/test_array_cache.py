"""Tests for exact numerical fingerprints and persistent array reuse."""

import json

import numpy as np
import pytest
from kompe import SHBasis, SphericalGrid
from kompe.constants import EARTH_RADIUS_M
from kompe.math import array_fingerprint, content_fingerprint
from kompe.spherical_transform import SphericalTransform

import pynamit
from pynamit.storage import ArrayCache


def test_array_fingerprints_include_shape_dtype_and_exact_values():
    """Similar but non-identical arrays have distinct identities."""
    values = np.array([1.0, 2.0], dtype=np.float64)

    assert array_fingerprint(values) != array_fingerprint(values.astype(np.float32))
    assert array_fingerprint(values) != array_fingerprint(values.reshape(1, 2))
    assert content_fingerprint({"values": values}) != content_fingerprint(
        {"values": values + np.array([0.0, 1e-14])}
    )


def test_array_cache_reuses_only_an_exact_identity(tmp_path):
    """Only exact cache hits skip construction."""
    cache = ArrayCache(tmp_path / "cache")
    calls = []

    def build(value):
        calls.append(value)
        return np.full((2, 3), value, dtype=np.float64)

    first = cache.get_or_create("operators", {"resolution": 20}, lambda: build(1.0))
    second = cache.get_or_create("operators", {"resolution": 20}, lambda: build(2.0))
    third = cache.get_or_create("operators", {"resolution": 40}, lambda: build(3.0))

    np.testing.assert_array_equal(first, np.ones((2, 3)))
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(third, np.full((2, 3), 3.0))
    assert calls == [1.0, 3.0]
    assert first.flags.writeable is False


def test_array_cache_rejects_a_mismatched_manifest(tmp_path):
    """A damaged cache entry fails clearly instead of being trusted."""
    cache = ArrayCache(tmp_path / "cache")
    cache.get_or_create("operators", {"resolution": 20}, lambda: np.eye(2))
    manifest_path = next((tmp_path / "cache" / "operators").glob("*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shape"] = [3, 3]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest does not match"):
        cache.get_or_create("operators", {"resolution": 20}, lambda: np.eye(2))


def test_sh_evaluation_cache_uses_exact_grid_coordinates(tmp_path, monkeypatch):
    """Persisted SH evaluations use exact grid identities."""
    cache = ArrayCache(tmp_path / "cache")
    grid = SphericalGrid(lat=[10.0, 20.0], lon=[30.0, 40.0])
    first_basis = SHBasis(2, 2, mean_free=False, operator_cache=cache)
    expected = first_basis.scalar_evaluation_array(grid)

    second_basis = SHBasis(2, 2, mean_free=False, operator_cache=cache)
    monkeypatch.setattr(
        second_basis,
        "_evaluate_on_grid",
        lambda *_args, **_kwargs: pytest.fail("cached evaluation was rebuilt"),
    )
    observed = second_basis.scalar_evaluation_array(grid)
    np.testing.assert_array_equal(observed, expected)

    shifted_grid = SphericalGrid(lat=[10.0 + 1e-12, 20.0], lon=[30.0, 40.0])
    third_basis = SHBasis(2, 2, mean_free=False, operator_cache=cache)
    original_evaluate = third_basis._evaluate_on_grid
    rebuilds = []

    def track_rebuild(*args, **kwargs):
        rebuilds.append(True)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(third_basis, "_evaluate_on_grid", track_rebuild)
    third_basis.scalar_evaluation_array(shifted_grid)
    assert rebuilds == [True]


def test_transform_reuses_persisted_normal_pinv(tmp_path, monkeypatch):
    """A repeated regularized fit restores its expensive inverse."""
    cache = ArrayCache(tmp_path / "cache")
    grid = SphericalGrid(
        lat=np.repeat(np.linspace(-60.0, 60.0, 7), 12), lon=np.tile(np.arange(0.0, 360.0, 30.0), 7)
    )
    first_basis = SHBasis(3, 3, mean_free=True, operator_cache=cache)
    first_transform = SphericalTransform(first_basis, grid, reg_lambda=0.1)
    values = np.sin(np.deg2rad(grid.lat)) + 0.2 * np.cos(np.deg2rad(grid.lon))
    expected = first_transform.analyze_scalar(values)

    assert any((cache.directory / "least_squares_normal_pinv").glob("*.npy"))
    monkeypatch.setattr(
        np.linalg,
        "pinv",
        lambda *_args, **_kwargs: pytest.fail("persisted normal pseudo-inverse was rebuilt"),
    )
    monkeypatch.setattr(
        "kompe.spherical_transform._scalar_data_normal_matrix",
        lambda *_args, **_kwargs: pytest.fail("normal matrix was rebuilt on a cache hit"),
    )
    second_basis = SHBasis(3, 3, mean_free=True, operator_cache=cache)
    second_transform = SphericalTransform(second_basis, grid, reg_lambda=0.1)

    np.testing.assert_array_equal(second_transform.analyze_scalar(values), expected)


def test_transform_reuses_persisted_helmholtz_factor(tmp_path, monkeypatch):
    """A repeated transform restores its Cholesky factor."""
    cache = ArrayCache(tmp_path / "cache")
    grid = SphericalGrid(
        lat=np.repeat(np.linspace(-75.0, 75.0, 8), 16), lon=np.tile(np.arange(0.0, 360.0, 22.5), 8)
    )
    first_basis = SHBasis(3, 3, mean_free=True, operator_cache=cache)
    first_transform = SphericalTransform(first_basis, grid)
    expected = first_transform.helmholtz_analysis_operator.matvec(
        np.arange(2 * grid.size, dtype=float)
    )

    assert any((cache.directory / "least_squares_factor").glob("*.npy"))
    monkeypatch.setattr(
        "kompe.spherical_transform._helmholtz_normal_factor",
        lambda *_args, **_kwargs: pytest.fail("persisted Cholesky factor was rebuilt"),
    )
    second_basis = SHBasis(3, 3, mean_free=True, operator_cache=cache)
    second_transform = SphericalTransform(second_basis, grid)
    observed = second_transform.helmholtz_analysis_operator.matvec(
        np.arange(2 * grid.size, dtype=float)
    )

    np.testing.assert_array_equal(observed, expected)


def test_gap_Br_cache_excludes_transient_shell_evaluations(tmp_path, monkeypatch):
    """The gap-Br cache excludes one-use quadrature operators."""
    cache_directory = tmp_path / "operator-cache"
    simulation_kwargs = {
        "Nmax": 2,
        "Mmax": 2,
        "Ncs": 4,
        "RI": EARTH_RADIUS_M + 110e3,
        "RM": 2.0 * EARTH_RADIUS_M,
        "main_field_kind": "dipole",
        "fac_integration_radii": np.array([EARTH_RADIUS_M + 110e3, 1.5 * EARTH_RADIUS_M]),
        "enable_pfac_coupling": True,
        "artifact_storage": "netcdf",
        "operator_cache_directory": cache_directory,
        "backend": "numpy",
    }
    first = pynamit.Simulation(simulation_directory=tmp_path / "first", **simulation_kwargs)
    _ = first.geometry
    evaluation_directory = first.operator_cache.directory / "sh_evaluation"
    evaluations_before_gap_response = len(tuple(evaluation_directory.glob("*.npy")))
    expected = first.geometry.boundary_jr_to_gap_Br_matrix

    cache = first.operator_cache
    assert any((cache.directory / "gap_Br_response").glob("*.npy"))
    assert len(tuple(evaluation_directory.glob("*.npy"))) == evaluations_before_gap_response + 2

    second = pynamit.Simulation(simulation_directory=tmp_path / "second", **simulation_kwargs)
    monkeypatch.setattr(
        second.geometry,
        "_compute_boundary_jr_to_gap_Br_matrix",
        lambda: pytest.fail("persisted gap-Br response was rebuilt"),
    )
    np.testing.assert_array_equal(second.geometry.boundary_jr_to_gap_Br_matrix, expected)
