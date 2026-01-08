
import logging
import numpy as np
import matplotlib.pyplot as plt
from pynamit.simulation.runner import run_pynamit

def test_convergence():
    logging.basicConfig(level=logging.ERROR)
    
    N_values = [2, 3, 4] # Keep small for speed
    errors = []
    
    print("Testing Convergence of General Analytic Path...")
    print(f"{'N':<5} {'RefNorm':<12} {'GenNorm':<12} {'RelError':<12} {'GenDiagonal[0]'}")
    print("-" * 60)
    
    for N in N_values:
        # Ensure Ncs is even
        Ncs = N + 2 + (N % 2)
        
        # Run Baseline (Quadrature)
        sim_quad = run_pynamit(
            Nmax=N, Mmax=N, Ncs=Ncs,
            simulation_mode="spectral_transform_gl",
            mainfield_kind="dipole",
            final_time=0.01,
            steady_state_initialization=False,
            wind=True
        )
        
        geo = sim_quad.state.geometry
        engine = geo.gaunt_engine
        sigma = sim_quad.state.M_total_on_grid
        
        # Reference (Quadrature)
        M_ref = engine.get_vector_interaction_matrix(sigma)
        norm_ref = np.linalg.norm(M_ref)
        
        # Analytic (General)
        c_pp, c_mm, c_pm, c_mp = geo._get_spin_tensor_coeffs(sigma)
        M_gen = engine.get_general_analytic_interaction_matrix(c_pp, c_mm, c_pm, c_mp, input_is_complex=True)
        norm_gen = np.linalg.norm(M_gen)
        
        if N == N_values[0]:
            # Save N=2 matrices for diagnosis
            np.save("M_ref_failure.npy", M_ref)
            np.save("M_gen_failure.npy", M_gen)
            print("Saved M_ref_failure.npy and M_gen_failure.npy")
            
            # Print top-left 4x4 block for quick inspection
            print("\nDEBUG MATRIX DUMP (N=2, Top-Left 4x4 of Real):")
            print("Reference:")
            print(M_ref.real[:4, :4])
            print("Analytic:")
            print(M_gen.real[:4, :4])
            
            # Print Diagonal
            print("\nReference Diagonal (First 5):")
            print(np.diag(M_ref).real[:5])
            print("Analytic Diagonal (First 5):")
            print(np.diag(M_gen).real[:5])

        # Compare
        diff = M_gen - M_ref
        rel_err = np.linalg.norm(diff) / norm_ref
        
        diag0 = M_gen[0,0]
        
        errors.append(rel_err)
        print(f"{N:<5} {norm_ref:<12.4e} {norm_gen:<12.4e} {rel_err:<12.4f} {diag0:.2e}")
        
    print("\nConvergence Analysis:")
    if errors[-1] < errors[0]:
        print("Error is decreasing.")
    else:
        print("Error is NOT decreasing (likely structural mismatch).")

if __name__ == "__main__":
    test_convergence()
