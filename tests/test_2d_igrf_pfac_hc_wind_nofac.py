"""IGRF, PFAC, HC, and wind test without radial-current driving."""

import numpy as np
import pytest

from pynamit.simulation.workflows.standard import run_pynamit


def test_2d_igrf_pfac_hc_wind_nofac():
    """Test 2D IGRF/PFAC/HC/wind simulation without jr input."""
    # Arrange.
    expected_coeff_norm = 3.481373639807845e-09
    expected_coeff_max = 1.2332703463139317e-09
    expected_coeff_min = -1.476139646570429e-09
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
        use_jr=False,
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

    assert "jr" not in simulation.run_data.input_series.datasets
    # pyHWM uses single precision, relax tolerances for wind tests.
    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-5)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-5)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-5)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-5)
