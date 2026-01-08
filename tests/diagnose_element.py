
import logging
import numpy as np
from pynamit.simulation.runner import run_pynamit

def diagnose_element():
    print("Diagnosing Single Matrix Element (Ref vs Analytic)...")
    
    # 1. Run Baseline (Quad)
    # Use N=1 for absolute simplicity?
    # N=1: (00), (1-1), (10), (11).
    N = 1
    sim = run_pynamit(
        Nmax=N, Mmax=N, Ncs=4,
        simulation_mode="spectral_transform_gl",
        mainfield_kind="dipole",
        final_time=0.001,
        steady_state_initialization=False,
        wind=True
    )
    
    geo = sim.state.geometry
    engine = geo.gaunt_engine
    sigma = sim.state.M_total_on_grid
    
    # Get Spin Coeffs (Analytic Inputs)
    # Note: These are PynaMIT Coeffs.
    c_pp, c_mm, c_pm, c_mp = geo._get_spin_tensor_coeffs(sigma)
    
    # Identify Indices
    # Basis: Scalar indices (l,m).
    # map: 0->(0,0), 1->(1,-1), 2->(1,0), 3->(1,1).
    # Since we skip l=0, Vector Matrix indices:
    # Row 0 -> (1,-1). Row 1 -> (1,0). Row 2 -> (1,1).
    # Let's Pick Row 1 (1,0) (Poloidal) -> Row 1.
    # Ref Matrix M_vsh is (2L, 2L).
    # L_len for N=1 is 3 (1-1, 10, 11).
    # So M is (6, 6).
    
    M_ref = engine.get_vector_interaction_matrix(sigma)
    
    # Let's verify index mapping
    idx_map = {}
    offset = 0
    # Check basis used by engine
    # engine.basis is SHBasis
    for pair in engine.basis.cnm.index_pairs:
        if pair[0] == 0: continue # Skip l=0
        idx_map[offset] = pair
        offset += 1
    for pair in engine.basis.snm.index_pairs:
        if pair[0] == 0: continue
        idx_map[offset] = pair
        offset += 1
    
    # Let's look at Diagonal Element for (1,0) Poloidal
    # Find offset for (1,0)
    target_idx = -1
    for k, v in idx_map.items():
        if v == (1, 0):
            target_idx = k
            break
            
    if target_idx == -1:
        print("Could not find (1,0) index.")
        return

    print(f"Analyzing Element [{target_idx}, {target_idx}] (Poloidal l=1 m=0)...")
    val_ref = M_ref[target_idx, target_idx]
    
    # Compute Analytic Value manually or via engine
    # Use current engine implementation (120% version)
    M_gen = engine.get_general_analytic_interaction_matrix(c_pp, c_mm, c_pm, c_mp)
    val_gen = M_gen[target_idx, target_idx]
    
    print(f"Ref: {val_ref:.6e}")
    print(f"Gen: {val_gen:.6e}")
    
    if abs(val_gen) > 1e-15:
        ratio = val_ref / val_gen
        print(f"Ratio (Ref/Gen): {ratio:.6f}")
        print(f"Magnitude Ratio: {abs(ratio):.6f}")
        print(f"Phase Diff (deg): {np.angle(ratio, deg=True):.2f}")
    else:
        print("Gen is zero.")
        
    # Also check indices for (1,1)
    target_idx_11 = -1
    for k,v in idx_map.items(): 
        if v == (1,1): target_idx_11=k; break
    
    print(f"\nAnalyzing Element [{target_idx_11}, {target_idx_11}] (Poloidal l=1 m=1)...")
    val_ref_11 = M_ref[target_idx_11, target_idx_11]
    val_gen_11 = M_gen[target_idx_11, target_idx_11]
    
    print(f"Ref: {val_ref_11:.6e}")
    print(f"Gen: {val_gen_11:.6e}")

    if abs(val_gen_11) > 1e-15:
        ratio = val_ref_11 / val_gen_11
        print(f"Ratio (Ref/Gen): {ratio:.6f}")
        print(f"Magnitude Ratio: {abs(ratio):.6f}")
        print(f"Phase Diff (deg): {np.angle(ratio, deg=True):.2f}")


if __name__ == "__main__":
    diagnose_element()
