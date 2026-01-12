import logging
import pytest
import numpy as np
from pynamit.simulation.runner import run_pynamit
from pynamit.primitives.field import Field

def run_convergence_check(mainfield_kind, conductance_kind, N=12, tol=1e-6):
    """
    Run a convergence check for a specific configuration.
    
    Parameters
    ----------
    mainfield_kind : str
        "radial" or "dipole"
    conductance_kind : str
        "unit", "hall", "pedersen_smooth"
    N : int
        Spectral resolution
    tol : float
        Relative error tolerance
    """
    print(f"\nTesting {mainfield_kind} {conductance_kind} (N={N})...")
    
    sim = run_pynamit(
        Nmax=N, Mmax=N, Ncs=2*N+4,
        simulation_mode="spectral_transform_gl",
        mainfield_kind=mainfield_kind,
        final_time=0.01,
        steady_state_initialization=False,
        wind=True
    )
    
    geo = sim.state.geometry
    basis = geo.solution_basis
    
    # Custom Conductance Setup
    th_rad = np.deg2rad(geo.grid.theta)
    
    if conductance_kind == "unit":
        etaP = np.zeros(geo.grid.size)
        etaH = np.zeros(geo.grid.size) # Unit implies vacuum? No, unit conductance usually means sigma=I?
        # Analytic solver usually tests "Identity" which implies Integral( Y vector * Y vector ).
        # In PynaMIT, sigma is input as Fields.
        # Let's use etaP=1, etaH=0 for "Pedersen Identity".
        # Effectively Constant SigmaP=1.
        etaP_val = np.ones(geo.grid.size)
        etaH_val = np.zeros(geo.grid.size)
    elif conductance_kind == "hall":
        # Constant Hall
        etaP_val = np.zeros(geo.grid.size)
        etaH_val = np.ones(geo.grid.size)
    elif conductance_kind == "pedersen_smooth":
        # Smooth variable
        etaP_val = 1.0 + 0.5 * np.cos(th_rad)**2
        etaH_val = np.zeros(geo.grid.size)
    else:
        raise ValueError(f"Unknown conductance kind: {conductance_kind}")

    sim.state.etaP = Field.from_values(geo.grid, etaP_val, np.zeros(geo.grid.size), np.zeros(geo.grid.size))
    sim.state.etaH = Field.from_values(geo.grid, etaH_val, np.zeros(geo.grid.size), np.zeros(geo.grid.size))
    sim.state._invalidate_caches()

    sigma = sim.state.M_total_on_grid
    
    # Reference (Quadrature)
    eta_quad = geo._synthesize_to_gaunt(sigma)
    M_ref = basis.get_quadrature_interaction_matrix(eta_quad)
    norm_ref = np.linalg.norm(M_ref)
    
    # Analytic (General)
    M_gen = basis.get_analytic_interaction_matrix_from_real_grid(
        eta_quad[0, 0], eta_quad[1, 1], eta_quad[0, 1], eta_quad[1, 0]
    )
    norm_gen = np.linalg.norm(M_gen)
    
    diff = M_ref - M_gen
    rel_err = np.linalg.norm(diff) / norm_ref
    
    print(f"  Norm Ref: {norm_ref:.4e}")
    print(f"  Norm Gen: {norm_gen:.4e}")
    print(f"  Rel Error: {rel_err:.4e}")
    
    assert rel_err < tol, f"Convergence failed for {mainfield_kind} {conductance_kind}: {rel_err:.4e} > {tol}"
    return rel_err

def test_analytic_convergence_radial_hall():
    # Bit-exact expectation
    run_convergence_check("radial", "hall", N=12, tol=1e-10)

def test_analytic_convergence_dipole_pedersen():
    # High precision expectation (N=12 limited by spectral truncation ~3e-5)
    run_convergence_check("dipole", "pedersen_smooth", N=12, tol=1e-4)

if __name__ == "__main__":
    test_analytic_convergence_radial_hall()
    test_analytic_convergence_dipole_pedersen()
