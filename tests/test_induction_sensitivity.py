from pynamit.simulation.settings import DynamicsMode, MainfieldKind, SimulationMode

def test_induction_sensitivity(tmp_path):
    """Test to verify sensitivity to toroidal forcing and radial-closure terms in isolation.
    
    This test runs with:
    1. Zero Magnetospheric Driver (jr = 0) -> Eliminates overwhelming constraint.
    2. Non-zero neutral wind -> Provides source for induction (-v x B).
    3. Full Induction Mode -> Physics terms active.
    
    We expect a non-zero toroidal potential (psi) driven purely by the wind-driven induction.
    """
    from pynamit.simulation.runner import run_pynamit
    import numpy as np
    
    # Run simulation via runner
    # use_jr=False -> Zero Driver
    # wind=True -> Built-in Wind (HWM14)
    dynamics = run_pynamit(
        run_directory=str(tmp_path / "induction_sensitivity"),
        final_time=10.0,
        dt=10.0,
        Nmax=5, Mmax=2,
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        mainfield_kind=MainfieldKind.IGRF,
        least_squares_solver="svd",
        wind=True,
        use_jr=False,
    )

    # 4. Check Response
    psi = dynamics.state.psi
    psi_norm = np.linalg.norm(psi)
    
    print(f"DEBUG: Sensitivity Test (Runner) |psi| = {psi_norm:.8e}")
    # Baseline updated 2026-02-23 after coupled steady-state projected Tikhonov solve refactor.
    expected_psi_norm = 3.02160609422578e-10
    import pytest
    assert psi_norm == pytest.approx(expected_psi_norm, rel=1e-8), \
        f"Induction response changed! Expected {expected_psi_norm}, got {psi_norm}"
