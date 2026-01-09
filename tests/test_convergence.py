
import logging
import numpy as np
import matplotlib.pyplot as plt
from pynamit.simulation.runner import run_pynamit

def test_convergence():
    logging.basicConfig(level=logging.ERROR)
    
    N_values = [4, 8, 12] # Higher N stress test
    errors = []
    
    print("Testing Convergence of General Analytic Path...")
    print(f"{'N':<5} {'RefNorm':<12} {'GenNorm':<12} {'RelError':<12} {'GenDiagonal[0]'}")
    print("-" * 60)
    
    for N in N_values:
        # Ensure Ncs is even
        # Ensure Ncs is high enough for exact cubic integration
        # Order is roughly 3*N. GL integrates 2*Ncs - 1.
        # Need 2*Ncs - 1 >= 3*N
        # Ncs >= (3N+1)/2
        Ncs = 2 * N + 4 # Safe margin
        
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

        # Detailed Analysis
        diff = M_ref - M_gen
        rel_norm_error = np.linalg.norm(diff) / np.linalg.norm(M_ref)
        
        max_abs_err = np.max(np.abs(diff))
        mean_abs_err = np.mean(np.abs(diff))
        
        # Element-wise relative error (filter small values)
        mask = np.abs(M_ref) > 1e-4
        if np.any(mask):
            elem_rel_err = np.abs(diff[mask]) / np.abs(M_ref[mask])
            max_elem_rel_err = np.max(elem_rel_err)
            mean_elem_rel_err = np.mean(elem_rel_err)
        else:
            max_elem_rel_err = 0.0
            mean_elem_rel_err = 0.0
            
        print(f"{N:<5} {norm_ref:<12.4e} {norm_gen:<12.4e} {rel_norm_error:<12.4e} {M_gen.diagonal()[0].real:<14.2e}")
        print(f"      MaxAbs: {max_abs_err:.4e}  MeanAbs: {mean_abs_err:.4e}")
        print(f"      MaxRel: {max_elem_rel_err:.4e}  MeanRel: {mean_elem_rel_err:.4e}")
        
        # Check Ratio of Diagonals (Scaling factor check)
        diag_ref = M_ref.diagonal()
        diag_gen = M_gen.diagonal()
        ratios = diag_ref.real / diag_gen.real
        valid_r = np.abs(diag_gen) > 1e-4
        if np.any(valid_r):
             mean_ratio = np.mean(ratios[valid_r])
             std_ratio = np.std(ratios[valid_r])
             print(f"      DiagRatio Mean: {mean_ratio:.4f} +/- {std_ratio:.4f}")
        
        errors.append(rel_norm_error)
        
    print("\nConvergence Analysis:")
    if errors[-1] < errors[0]:
        print("Error is decreasing.")
    else:
        print("Error is NOT decreasing (likely structural mismatch).")

if __name__ == "__main__":
    test_convergence()
