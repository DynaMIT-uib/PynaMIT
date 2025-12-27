"""End-to-end simulation test using CSBasis."""

import os
import tempfile
import pytest
import numpy as np

from pynamit.simulation.runner import run_pynamit

def test_cs_basis_simulation_dop853(backend):
    """Test full simulation with CSBasis as solution basis using DOP853."""
    # Baseline with current Geometry scaling and PFAC integration
    expected_coeff_norm = 6.768092e-08 
    expected_coeff_max =  2.752912e-09
    expected_coeff_min = -1.857978e-09
    expected_n_coeffs = 1200

    # We use small integration time for speed.
    final_time = 0.003
    dt = 0.001
    
    temp_dir = os.path.join(tempfile.gettempdir(), "test_run_pynamit_cs")
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Act.
    # Run simulation with solution_basis_kind="CS"
    dynamics = run_pynamit(
        final_time=final_time,
        dt=dt,
        Nmax=8,  # Still used for spectral inputs/Mainfield
        Mmax=6,
        Ncs=10,   # Increased resolution to avoid aliasing singularity
        mainfield_kind="igrf",
        fig_directory=temp_dir,
        ignore_PFAC=False, 
        connect_hemispheres=True,
        latitude_boundary=50,
        wind=True,
        steady_state_initialization=False,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
        integrator="euler",
        multi_data=True,
        simulation_mode="cs_dominant",
        # Use CG solver (Matrix-Free) with High Regularization to ensure condition number is manageable
        least_squares_solver="cg",
        # High Regularization (1e-5) forces Kappa ~ 1e5, allowing CG to converge consistently across backends
        m_imp_regularization_lambda=1e-4,
    )



    # Assert.
    coeff_array = np.hstack(
        (
            dynamics.output_timeseries.datasets["state"]["CS_m_ind"].values[-1],
            dynamics.output_timeseries.datasets["state"]["CS_m_imp"].values[-1],
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
    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-6)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-6)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-6)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-6)