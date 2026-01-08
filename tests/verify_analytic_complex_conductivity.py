
import numpy as np
import pytest
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.geometry import Geometry

def verify_complex_conductivity():
    print("Stress Testing Analytic Formula with Random Complex Conductivity...")
    
    # 1. Setup Environment
    # Nmax=5 enough to trigger non-trivial 3j interactions
    dynamics = run_pynamit(
        Nmax=5, Mmax=5, Ncs=6,
        simulation_mode="pure_spectral",
        mainfield_kind="radial",
        mainfield_B0=30000e-9,
        final_time=0.01,
        steady_state_initialization=False
    )
    geo = dynamics.state.geometry
    engine = geo.gaunt_engine
    
    L = engine.basis.index_length
    
    # 2. Generate Random Conductivity Coefficients
    # Populating all modes to ensure full spectral interaction (l_i, l_j, l_k)
    np.random.seed(42)
    coeffs_P = np.random.randn(L) * 1e-4
    coeffs_H = np.random.randn(L) * 1e-4
    
    # Ensure they are real-valued on grid (PynaMIT usually handles complex coeffs, 
    # but let's assume standard real input for conductivity physical validity)
    # The Gaunt engine handles complex coefficient arithmetic internally.
    
    # 3. Analytic Matrix
    print("Computing Analytic Matrix...")
    M_ana = engine.get_analytic_interaction_matrix(coeffs_P, coeffs_H)
    
    # 4. Quadrature Matrix (Ground Truth)
    print("Computing Quadrature Matrix...")
    
    # Synthesize random coeffs to grid
    G_eval = engine.basis.get_evaluation_matrix(engine.quad_grid)
    if hasattr(G_eval, "toarray"): G_eval = G_eval.toarray()
    
    sigma_P_grid = G_eval @ coeffs_P
    sigma_H_grid = G_eval @ coeffs_H
    
    # Construct tensor for Radial Field:
    # J = P*E + H*(er x E)
    # In Surface Basis (th, ph):
    # J_th = P*E_th - H*E_ph
    # J_ph = H*E_th + P*E_ph
    # Matrix: [[P, -H], [H, P]]
    
    Q = len(sigma_P_grid)
    sigma_quad = np.zeros((2, 2, Q))
    sigma_quad[0, 0, :] = sigma_P_grid
    sigma_quad[0, 1, :] = -sigma_H_grid
    sigma_quad[1, 0, :] = sigma_H_grid
    sigma_quad[1, 1, :] = sigma_P_grid
    
    M_quad = engine.get_vector_interaction_matrix(sigma_quad)
    
    # 5. Compare
    diff = M_ana - M_quad
    norm_diff = np.linalg.norm(diff)
    norm_ref = np.linalg.norm(M_quad)
    rel_error = norm_diff / norm_ref
    
    print(f"\n--- RESULTS ---")
    print(f"Matrix Shape: {M_quad.shape}")
    print(f"Reference Norm: {norm_ref:.6e}")
    print(f"Difference Norm: {norm_diff:.6e}")
    print(f"Relative Error: {rel_error:.6e}")
    
    # Check bounds
    if rel_error < 1e-14:
        print("PASS: Machine Precision Agreement Verified.")
    else:
        print("FAIL: Significant Discrepancy Found.")
        
        # Diagnostics
        print("\nDiagnostic - Block Errors:")
        d_PP = diff[:L, :L]; print(f"PP Rel: {np.linalg.norm(d_PP)/norm_ref:.4e}")
        d_PT = diff[:L, L:]; print(f"PT Rel: {np.linalg.norm(d_PT)/norm_ref:.4e}")
        d_TP = diff[L:, :L]; print(f"TP Rel: {np.linalg.norm(d_TP)/norm_ref:.4e}")
        d_TT = diff[L:, L:]; print(f"TT Rel: {np.linalg.norm(d_TT)/norm_ref:.4e}")

if __name__ == "__main__":
    verify_complex_conductivity()
