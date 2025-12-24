
import pytest
import numpy as np
from pynamit.simulation.runner import run_pynamit

def test_pure_spectral_execution():
    """Verify that pure_spectral=True runs without errors."""
    try:
        dynamics = run_pynamit(
            final_time=0.001,
            Nmax=5,
            Mmax=5,
            Ncs=6,
            simulation_mode="pure_spectral",
        )
        
        dynamics.evolve_to_time(t=0.002, dt=0.001)
        print("Pure Spectral execution successful.")
    except Exception as e:
        pytest.fail(f"Pure Spectral path failed with: {e}")

if __name__ == "__main__":
    test_pure_spectral_execution()
