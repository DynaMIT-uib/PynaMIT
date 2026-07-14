"""Multi-data DOP853 test module."""

import numpy as np
import pytest

from pynamit.simulation.workflows.standard import run_pynamit


def test_multi_data_dop853():
    """Test simulation with multiple data points and DOP853."""
    # Arrange.
    expected_coeff_norm = 2.5686566061400986e-08
    expected_coeff_max = 6.133350112801935e-09
    expected_coeff_min = -8.876382135048725e-09
    expected_n_coeffs = 228

    # Act.
    simulation = run_pynamit(
        final_time=15,
        dt=5,
        Nmax=10,
        Mmax=8,
        Ncs=20,
        main_field_kind="igrf",
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=True,
        interhemispheric_coupling_latitude=50,
        use_wind=True,
        steady_state_initialization=True,
        jr_projection_basis="SH",
        resistance_projection_basis="SH",
        u_projection_basis="SH",
        integrator="DOP853",
        multi_data=True,
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

    # pyHWM uses single precision, relax tolerances for wind tests.
    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-5)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-5)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-5)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-5)
