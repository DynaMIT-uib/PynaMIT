"""Dipole, PFAC, HC and exponential test."""

import numpy as np
import pytest

from pynamit.simulation.workflows.standard import run_pynamit


def test_2d_dipole_pfac_hc_exp():
    """Test 2D simulation with dipole, PFAC, HC and exponential."""
    # Arrange.
    expected_coeff_norm = 8.958879519810761e-09
    expected_coeff_max = 1.6925319055957777e-09
    expected_coeff_min = -3.785113821285535e-09
    expected_n_coeffs = 228

    # Act.
    simulation = run_pynamit(
        final_time=0.1,
        dt=0.1,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        main_field_kind="dipole",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        integrator="exponential",
        steady_state_initialization=False,
    )

    # Assert.
    coeff_array = np.hstack(
        (
            simulation.run_data.output_series.datasets["state"]["SH_m_ind"].values[-1],
            simulation.run_data.output_series.datasets["state"]["SH_m_imp"].values[-1],
        )
    )

    actual_coeff_norm = np.linalg.norm(coeff_array)
    actual_coeff_max = np.max(coeff_array)
    actual_coeff_min = np.min(coeff_array)
    actual_n_coeffs = coeff_array.shape[0]

    print("actual_coeff_norm: ", actual_coeff_norm)
    print("actual_coeff_max: ", actual_coeff_max)
    print("actual_coeff_min: ", actual_coeff_min)
    print("actual_n_coeffs: ", actual_n_coeffs)

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-10)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-10)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-10)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-10)
