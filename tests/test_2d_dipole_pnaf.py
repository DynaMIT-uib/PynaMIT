import os
import tempfile
import pytest

from pynamit.pynamit import run_pynamit
import cupy as np

def test_2d_dipole_pnaf():
    # Arrange
    expected_coeff_norm = 3.1747131876906014e-18
    expected_coeff_max = 5.9194693291573005e-19
    expected_coeff_min = -3.0149027890855758e-19
    expected_n_coeffs = 21

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)  

    # Act
    coeffs = run_pynamit(totalsteps=20, dt=5e-4, Nmax=5, Mmax=3, Ncs=60, B0_type='dipole', fig_directory=temp_dir, ignore_PNAF=False)

    # Assert
    coeff_array = np.array(coeffs)

    actual_coeff_norm = np.linalg.norm(coeff_array)
    actual_coeff_max = np.max(coeff_array)
    actual_coeff_min = np.min(coeff_array)
    actual_n_coeffs = len(coeffs)

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-10)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-10)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-10)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-10)