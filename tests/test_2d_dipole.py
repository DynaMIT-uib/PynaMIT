import os
import tempfile
import pytest

from pynamit.pynamit import run_pynamit
import cupy as np

def test_run_pynamit():
    # Arrange
    expected_coeff_norm = 2.030783453351264e-16
    expected_coeff_max = 1.0178191775378528e-17
    expected_coeff_min = -4.9443522504953425e-18
    expected_n_coeffs = 201

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Act
    coeffs = run_pynamit(totalsteps=200, dt=5e-4, Nmax=45, Mmax=3, Ncs=60, B0_type='dipole', fig_directory=temp_dir)

    # Assert
    coeff_array = np.array(coeffs)

    actual_coeff_norm = np.linalg.norm(coeff_array).item()
    actual_coeff_max = np.max(coeff_array).item()
    actual_coeff_min = np.min(coeff_array).item()
    actual_n_coeffs = len(coeffs)

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-10)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-10)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-10)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-10)