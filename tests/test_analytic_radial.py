
import pytest
import numpy as np
from pynamit.simulation.runner import run_pynamit

@pytest.mark.wind
def test_analytic_radial_execution(pynamit_approx):
    """
    Verify that pure_spectral=True with Radial Mainfield uses the Analytic Matrix
    and produces stable results.
    """
    # Expected values for Radial Field + Pure Spectral
    # Captured 2026-01-08: Matches Analytic VSH Logic
    expected_coeff_norm = 3.919307422446629e-08
    expected_coeff_max = 8.562985980286141e-09
    expected_coeff_min = -2.325852087017394e-08
    expected_n_coeffs = 70

    dynamics = run_pynamit(
        final_time=0.04,
        Nmax=5,
        Mmax=5,
        Ncs=6,
        simulation_mode="pure_spectral",
        steady_state_initialization=False,
        wind=True,
        mainfield_kind="radial", # Triggers Analytic Path
        mainfield_B0=30000e-9 
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

    print(f"Radial Analytic Norm: {actual_coeff_norm}")
    print(f"Radial Analytic Max: {actual_coeff_max}")
    print(f"Radial Analytic Min: {actual_coeff_min}")

    # Assert.
    assert actual_coeff_norm == pynamit_approx(expected_coeff_norm)
    assert actual_coeff_max == pynamit_approx(expected_coeff_max)
    assert actual_coeff_min == pynamit_approx(expected_coeff_min)
    assert actual_n_coeffs == expected_n_coeffs
    
if __name__ == "__main__":
    # Self-runner to print values
    class MockApprox:
        def __init__(self, v): pass
        def __eq__(self, o): return True
    test_analytic_radial_execution(lambda x: MockApprox(x))
