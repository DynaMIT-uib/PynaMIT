"""IGRF, PFAC, HC, and Q_eff wind-proxy test."""

import numpy as np
import pytest

from pynamit.simulation.workflows.standard import run_pynamit
from tests import magnetic_potential_coordinate_array


def test_2d_igrf_pfac_hc_wind_Q_eff():
    """Test wind driving represented through the Q_eff input path."""
    expected_coeff_norm = 1.0460771082860356e-08
    expected_coeff_max = 3.900932709710887e-09
    expected_coeff_min = -1.979185633797638e-09
    expected_n_coeffs = 228

    simulation = run_pynamit(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        main_field_kind="igrf",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        use_wind=True,
        use_Q_eff=True,
        equilibrium_initialization=False,
    )

    coeff_array = magnetic_potential_coordinate_array(simulation)

    actual_coeff_norm = np.linalg.norm(coeff_array)
    actual_coeff_max = np.max(coeff_array)
    actual_coeff_min = np.min(coeff_array)
    actual_n_coeffs = coeff_array.shape[0]

    print("actual_coeff_norm: ", actual_coeff_norm)
    print("actual_coeff_max: ", actual_coeff_max)
    print("actual_coeff_min: ", actual_coeff_min)
    print("actual_n_coeffs: ", actual_n_coeffs)

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-5)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-5)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-5)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-5)
