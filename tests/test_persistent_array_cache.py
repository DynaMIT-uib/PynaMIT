"""Persistent-cache integration with the magnetic boundary."""

import numpy as np
import pytest
from kompe.constants import EARTH_RADIUS_M

import pynamit
from pynamit.simulation.electrodynamics import magnetic_boundary


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
        magnetic_boundary,
        "boundary_jr_to_gap_Br_matrix",
        lambda *args, **kwargs: pytest.fail("persisted gap-Br response was rebuilt"),
    )
    np.testing.assert_array_equal(second.geometry.boundary_jr_to_gap_Br_matrix, expected)
