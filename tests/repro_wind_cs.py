
import pytest
import numpy as np
from pynamit.simulation.runner import run_pynamit

def test_wind_sensitivity_cs_dominant():
    """Verify if results are sensitive to wind in cs_dominant mode."""
    common_params = {
        "final_time": 0.003,
        "dt": 0.001,
        "Nmax": 10,
        "Mmax": 10,
        "Ncs": 6,
        "mainfield_kind": "igrf",
        "ignore_PFAC": True,
        "connect_hemispheres": False,
        "latitude_boundary": 50,
        "steady_state_initialization": False,
        "multi_data": True,
        "simulation_mode": "cs_dominant",
    }
    
    print("\nRunning with wind=False...")
    dynamics_no_wind = run_pynamit(wind=False, **common_params)
    m_ind_no_wind = dynamics_no_wind.output_timeseries.datasets["state"]["CS_m_ind"].values[-1]
    
    print("Running with wind=True...")
    dynamics_wind = run_pynamit(wind=True, **common_params)
    m_ind_wind = dynamics_wind.output_timeseries.datasets["state"]["CS_m_ind"].values[-1]
    
    diff = np.linalg.norm(m_ind_wind - m_ind_no_wind)
    print(f"Norm difference: {diff:.2e}")
    
    # In SH mode, wind usually makes a HUGE difference in m_ind evolution.
    # If the difference is EXACTLY 0, something is definitely wrong.
    assert diff > 0, "m_ind should change when wind is enabled!"

if __name__ == "__main__":
    test_wind_sensitivity_cs_dominant()

