"""Gauss-Legendre spectral transform test module.

Tests the SPECTRAL_TRANSFORM_GL simulation mode which uses exact
Gauss-Legendre quadrature for SH transforms.
"""

import os
import tempfile
import pytest

from pynamit.simulation.runner import run_pynamit
import numpy as np


@pytest.mark.wind
def test_spectral_transform_gl():
    """Test simulation with Gauss-Legendre grid for exact SH transforms.

    Uses SPECTRAL_TRANSFORM_GL mode which employs Gauss-Legendre quadrature
    points for exact (to machine precision) spherical harmonic transforms.
    The solver operates in SH space while nonlinear products are computed
    on the GL grid.
    """
    # Expected values for GL mode with exact quadrature Helmholtz decomposition
    expected_coeff_norm = 3.611476307534931e-08
    expected_coeff_max = 3.188123429263445e-09
    expected_coeff_min = -1.6261933724953575e-08
    expected_n_coeffs = 228

    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit_gl")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Act.
    dynamics = run_pynamit(
        final_time=15,
        dt=5,
        Nmax=10,
        Mmax=8,
        Ncs=20,  # Not used for GL mode, but kept for compatibility
        mainfield_kind="igrf",
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        wind=True,
        steady_state_initialization=True,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
        integrator="DOP853",
        multi_data=True,
        simulation_mode="spectral_transform_gl",
    )

    # Assert.
    coeff_array = np.hstack(
        (
            dynamics.output_timeseries.datasets["state"]["SH_m_ind"].values[-1],
            dynamics.output_timeseries.datasets["state"]["SH_m_imp"].values[-1],
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
