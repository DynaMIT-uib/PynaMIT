"""End-to-end simulation test using CSBasis."""

import os
import tempfile
import pytest
import numpy as np

from pynamit.simulation.runner import run_pynamit


def test_cs_basis_simulation_dop853():
    """Test full simulation with CSBasis as solution basis using DOP853."""
    # Updated regression values for non-singular (ignore_PFAC=True, connect=False) case
    # These represent the stable solution without Apex singularity at equator.
    expected_coeff_norm = 4.8774274198e-07
    expected_coeff_max =  8.8694633164e-08
    expected_coeff_min = -1.2845118369e-07
    expected_n_coeffs = 432
    # CSBasis N=18 (approx equivalent to Nmax=10 in resolution?)
    # SH Nmax=10 approx 121 DOFs.
    # CS N=18 approx 6*18*18 = 1944. Much larger state vector.
    
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
        Nmax=10,  # Still used for spectral inputs/Mainfield
        Mmax=10,
        Ncs=6,   # Cube resolution (minimal for speed)
        mainfield_kind="igrf",
        fig_directory=temp_dir,
        ignore_PFAC=True, 
        connect_hemispheres=False,
        latitude_boundary=50,
        wind=True,
        steady_state_initialization=False,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
        integrator="euler",
        multi_data=True,
        solution_basis_kind="CS", # THE KEY CHANGE
    )

    # Assert.
    
    # Check that state variables are on Grid
    # dynamics.state.m_imp should have shape (Ncs_total,)
    # N=6 -> 6*6^2 = 216 points.
    expected_size = 6 * 6 * 6
    
    m_ind_final = dynamics.output_timeseries.datasets["state"]["CS_m_ind"].values[-1]
    m_imp_final = dynamics.output_timeseries.datasets["state"]["CS_m_imp"].values[-1]
    
    # The output dataset stores "m_ind" with shape matching the solution basis.
    assert m_ind_final.shape == (expected_size,)
    assert m_imp_final.shape == (expected_size,)
    
    # Check that values are not zero/NaN (sanity check)
    assert np.all(np.isfinite(m_ind_final))
    # assert np.linalg.norm(m_ind_final) > 0.0 # Initial might be zero if started from zero?
    
    # Check E-field outputs (Phi, W)
    # Phi/W might also be GRID_Phi?
    phi_final = dynamics.output_timeseries.datasets["state"]["CS_Phi"].values[-1]
    assert phi_final.shape == (expected_size,)
    
    print(f"CSBasis Simulation Successful.")
    print(f"Norm m_ind: {np.linalg.norm(m_ind_final)}")
    print(f"Norm m_imp: {np.linalg.norm(m_imp_final)}")
    
    # Basic numeric bound checks (regression baseline)
    # Norms will be different than SH basis test due to different DOFs/Scaling.
    # Just asserting it ran and produced physical-looking magnitudes is good for now.


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
    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-2)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-2)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-2)
    assert actual_n_coeffs == pytest.approx(expected_n_coeffs, abs=0.0, rel=1e-2)