
import pytest
import numpy as np
from pynamit.simulation.runner import run_pynamit

def test_pure_spectral_execution():
    """Verify that pure_spectral=True runs without errors and matches regression baselines."""
    # Updated regression values for pure_spectral mode
    expected_coeff_norm = 2.7168892105e-08
    expected_coeff_max = 1.8360738910e-08
    expected_coeff_min = -7.0821962621e-09
    expected_n_coeffs = 70

    dynamics = run_pynamit(
        final_time=0.04,
        Nmax=5,
        Mmax=5,
        Ncs=6,
        simulation_mode="pure_spectral",
        steady_state_initialization=False,
    )
    
    state_ds = dynamics.output_timeseries.datasets["state"]
    
    # In pure_spectral mode, the state variables are prefixed with SH_
    m_ind_final = state_ds["SH_m_ind"].values[-1]
    m_imp_final = state_ds["SH_m_imp"].values[-1]
    coeff_array = np.hstack((m_ind_final, m_imp_final))
    
    actual_coeff_norm = np.linalg.norm(coeff_array)
    actual_coeff_max = np.max(coeff_array)
    actual_coeff_min = np.min(coeff_array)
    actual_n_coeffs = coeff_array.shape[0]

    # Assert.
    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-10)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-10)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-10)
    assert actual_n_coeffs == expected_n_coeffs
    
    print("Pure Spectral execution and numerical validation successful.")

if __name__ == "__main__":
    test_pure_spectral_execution()
