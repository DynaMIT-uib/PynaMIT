"""Grid-based IGRF, PFAC, HC, and wind test."""

import numpy as np
import pytest

from pynamit.simulation.workflows.standard import run_pynamit


def test_2d_igrf_pfac_hc_wind_grid():
    """Test 2D grid-based simulation with IGRF, PFAC, HC, and wind."""
    # Arrange.
    expected_coeff_norm = 8.539937195217714e-09
    expected_coeff_max = 2.9383716319638394e-09
    expected_coeff_min = -3.302945299283126e-09
    expected_n_coeffs = 228

    # Act.
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
        jr_projection_basis="CS",
        conductance_projection_basis="CS",
        u_projection_basis="CS",
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
    resistance = simulation.run_data.input_series.datasets["conductance"]

    print("actual_coeff_norm: ", actual_coeff_norm)
    print("actual_coeff_max: ", actual_coeff_max)
    print("actual_coeff_min: ", actual_coeff_min)
    print("actual_n_coeffs: ", actual_n_coeffs)

    # pyHWM uses single precision, relax tolerances for wind tests.
    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-5)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-5)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-5)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-5)
    assert "CS_log_conductance_magnitude" in resistance
    assert "CS_log_hall_to_pedersen_ratio" in resistance
