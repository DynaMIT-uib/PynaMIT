
import pytest
import numpy as np
from pynamit.simulation.runner import run_pynamit

def test_pure_spectral_execution():
    """Verify that pure_spectral=True runs without errors and matches regression baselines."""
    # Updated regression values for pure_spectral mode
    expected_coeff_norm = 1.0192787652e-07
    expected_coeff_max = 4.2444735773e-08
    expected_coeff_min = -4.0846742188e-08
    expected_n_coeffs = 70

    dynamics = run_pynamit(
        final_time=0.001,
        Nmax=5,
        Mmax=5,
        Ncs=6,
        simulation_mode="pure_spectral",
    )
    
    dynamics.evolve_to_time(t=0.002, dt=0.001)
    
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
    assert actual_coeff_norm == pytest.approx(expected_coeff_norm, abs=0.0, rel=1e-5)
    assert actual_coeff_max == pytest.approx(expected_coeff_max, abs=0.0, rel=1e-5)
    assert actual_coeff_min == pytest.approx(expected_coeff_min, abs=0.0, rel=1e-5)
    assert actual_n_coeffs == expected_n_coeffs
    
    print("Pure Spectral execution and numerical validation successful.")

if __name__ == "__main__":
    test_pure_spectral_execution()
