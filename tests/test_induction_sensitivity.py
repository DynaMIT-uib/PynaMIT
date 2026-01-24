
def test_induction_sensitivity():
    """Test to verify sensitivity to induction terms (K and D1) in isolation.
    
    This test runs with:
    1. Zero Magnetospheric Driver (jr = 0) -> Eliminates overwhelming constraint.
    2. Non-zero neutral wind -> Provides source for induction (v x B).
    3. Full Induction Mode -> Physics terms active.
    
    We expect a non-zero toroidal potential (psi) driven purely by the wind-driven induction.
    """
    from pynamit.simulation.runner import run_pynamit
    import numpy as np
    
    # Run simulation via runner
    # use_jr=False -> Zero Driver
    # wind=True -> Built-in Wind (HWM14)
    dynamics = run_pynamit(
        final_time=10.0,
        dt=10.0,
        Nmax=5, Mmax=2,
        simulation_mode="pure_spectral",
        dynamics_mode="full_induction",
        mainfield_kind="igrf",
        least_squares_solver="svd",
        induction_constraint_scaling=1e-12,
        wind=True,
        use_jr=False,
    )

    # 4. Check Response
    psi = dynamics.state.psi
    psi_norm = np.linalg.norm(psi)
    
    print(f"DEBUG: Sensitivity Test (Runner) |psi| = {psi_norm:.8e}")
    # Baseline observed from verified run based on built-in HWM14 wind and Zero Driver
    # Exact value captured: 0.000530838684883729
    expected_psi_norm = 5.30838684883729e-04
    import pytest
    assert psi_norm == pytest.approx(expected_psi_norm, rel=1e-10), \
        f"Induction response changed! Expected {expected_psi_norm}, got {psi_norm}"
