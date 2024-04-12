import pytest
from pynamit import run_pynamit
import numpy as np
import tempfile
import os

def test_run_pynamit():
    # Arrange
    expected_coeff_norm = 2.0322837331048516e-16
    expected_coeff_max = 1.0302465681964904e-17
    expected_coeff_min = -5.040932467648457e-18
    expected_n_coeffs = 201

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Act
    coeffs = run_pynamit(totalsteps=200, dt=5e-4, Nmax=45, Mmax=3, Ncs=60, fig_directory=temp_dir)

    # Assert
    actual_coeff_norm = np.linalg.norm(coeffs)
    actual_coeff_max = np.max(coeffs)
    actual_coeff_min = np.min(coeffs)
    actuall_n_coeffs = len(coeffs)

    assert actual_coeff_norm == expected_coeff_norm
    assert actual_coeff_max == expected_coeff_max
    assert actual_coeff_min == expected_coeff_min
    assert actuall_n_coeffs == expected_n_coeffs