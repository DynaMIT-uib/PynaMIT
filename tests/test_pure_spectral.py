
import pytest
import numpy as np
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import MainfieldKind, SimulationMode

@pytest.mark.wind
def test_pure_spectral_execution(pynamit_approx):
    """Verify that pure_spectral=True runs without errors and matches regression baselines."""
    # Updated regression values for pure_spectral mode (Analytic Tensor Path)
    # Norm stabilized at 2.3585e-08 after Spin-2 Phase Fix (+1j) (2026-01-11)
    # Previous value (2.61e-08) reflected incorrect (-1j) Symmetric Off-Diagonal phase.
    expected_coeff_norm = 2.3585250922235536e-08
    expected_coeff_max = 7.704433571646292e-09
    expected_coeff_min = -1.736353795891065e-08
    expected_n_coeffs = 70

    dynamics = run_pynamit(
        final_time=0.04,
        Nmax=5,
        Mmax=5,
        Ncs=6,
        mainfield_kind=MainfieldKind.IGRF,
        ignore_PFAC=False,
        connect_hemispheres=True,
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        steady_state_initialization=False,
        wind=True,
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

    print("actual_coeff_norm: ", actual_coeff_norm)
    print("actual_coeff_max: ", actual_coeff_max)
    print("actual_coeff_min: ", actual_coeff_min)
    print("actual_n_coeffs: ", actual_n_coeffs)

    # Assert.
    assert actual_coeff_norm == pynamit_approx(expected_coeff_norm)
    assert actual_coeff_max == pynamit_approx(expected_coeff_max)
    assert actual_coeff_min == pynamit_approx(expected_coeff_min)
    assert actual_n_coeffs == expected_n_coeffs
    
    print("Pure Spectral execution and numerical validation successful.")

if __name__ == "__main__":
    from pytest import approx
    test_pure_spectral_execution(pynamit_approx=lambda x: approx(x, rel=1e-12))
