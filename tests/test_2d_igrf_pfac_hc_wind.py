import os
import tempfile
import pytest
import shutil

from pynamit.default_run import run_pynamit
import numpy as np

def test_2d_igrf_pfac_hc_wind():
    # Arrange
    expected_coeff_norm = 4.223430340032234e-10
    expected_coeff_max = 1.7222179909543226e-11
    expected_coeff_min = -2.339773548351343e-11
    expected_n_coeffs = 201

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    input_files = ['ulat.npy', 'ulon.npy', 'uphi.npy', 'utheta.npy'] # wind files
    for file in input_files:
        shutil.copyfile(os.path.join(input_dir, file), os.path.join(temp_dir, file))        

    # Act
    dynamics = run_pynamit(final_time = 0.1,
                           dt = 5e-4,
                           Nmax = 5,
                           Mmax = 3,
                           Ncs = 18,
                           mainfield_kind = 'dipole',
                           fig_directory = temp_dir,
                           ignore_PFAC = False,
                           connect_hemispheres = True,
                           latitude_boundary = 50,
                           wind = True)

    # Assert
    coeff_array = dynamics.timeseries['state']['SH_m_ind'].values

    actual_coeff_norm = np.linalg.norm(coeff_array)
    actual_coeff_max = np.max(coeff_array)
    actual_coeff_min = np.min(coeff_array)
    actual_n_coeffs = coeff_array.shape[0]

    print("actual_coeff_norm: ", actual_coeff_norm)
    print("actual_coeff_max: ", actual_coeff_max)
    print("actual_coeff_min: ", actual_coeff_min)
    print("actual_n_coeffs: ", actual_n_coeffs)

    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-8)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-8)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-8)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-8)