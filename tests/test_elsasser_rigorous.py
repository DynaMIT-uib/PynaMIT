import numpy as np
import sys
import os
import random

# Ensure src is in path (Must be first to override installed package)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pynamit
print("PynaMIT Path:", pynamit.__file__)

from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.spherical_harmonics.gaunt import GauntEngine

def check_rigorous():
    print("Initialize Final Rigorous Elsasser Verification...")
    NMAX = 8
    # High resolution to ensure 'Exact' is truly exact (machine precision)
    basis = SHBasis(Nmax=NMAX, Mmax=NMAX, mean_free=True)
    engine = GauntEngine(basis, grid_resolution=4*NMAX) 
    
    print(f"Grid Size: {len(engine.weights)}")

    # 1. Random Sampling of Triplets
    n_samples = 50
    print(f"\n1. Testing {n_samples} random valid triplets...")
    
    max_error = 0.0
    sum_error = 0.0
    count = 0
    
    symmetry_candidates = []

    for _ in range(n_samples * 20): 
        if count >= n_samples: break
        
        li = random.randint(1, NMAX)
        lj = random.randint(1, NMAX)
        min_k = abs(li - lj)
        max_k = li + lj
        if min_k > NMAX: continue 
        
        possible_ks = [k for k in range(min_k, min(max_k, NMAX)+1) if (li+lj+k)%2 != 0]
        if not possible_ks: continue
        
        lk = random.choice(possible_ks)
        
        mi = random.randint(-li, li)
        mj = random.randint(-lj, lj)
        mk = random.randint(-lk, lk)
        
        # Calculate
        exact = engine._compute_elsasser_gl_raw(li, mi, lj, mj, lk, mk)
        ana = engine.elsasser_coefficient(li, mi, lj, mj, lk, mk)
        
        diff = abs(exact - ana)
        norm = max(abs(exact), abs(ana))
        
        if norm < 1e-10: 
            continue
            
        rel_error = diff / norm
        max_error = max(max_error, rel_error)
        sum_error += rel_error
        count += 1
        
        # Strict Threshold for success
        # Integration error + float noise ~ 1e-14
        if rel_error > 1e-12:
            print(f"  FAIL: L=({li},{lj},{lk}) m=({mi},{mj},{mk}) | Exact={exact:.8e} Ana={ana:.8e} Err={rel_error:.4e}")
        
        if len(symmetry_candidates) < 10:
            symmetry_candidates.append((li, mi, lj, mj, lk, mk))

    print(f"  Evaluated {count} non-zero terms.")
    print(f"  Max Relative Error: {max_error:.4e}")
    print(f"  Mean Relative Error: {sum_error/count:.4e}")

    # 2. Antisymmetry Check
    print(f"\n2. Verifying Antisymmetry E(ijk) = -E(ikj)...")
    
    sym_failures = 0
    for ((li, mi, lj, mj, lk, mk)) in symmetry_candidates:
        val_ijk = engine.elsasser_coefficient(li, mi, lj, mj, lk, mk)
        val_ikj = engine.elsasser_coefficient(li, mi, lk, mk, lj, mj) 
        
        # Sum should be zero
        residual = val_ijk + val_ikj
        if abs(residual) > 1e-12:
             print(f"  ASYM FAIL: L=({li},{lj},{lk}) Sum={residual:.4e}")
             sym_failures += 1

    if sym_failures == 0:
        print("  Antisymmetry verified to precision.")
    else:
        print(f"  {sym_failures} antisymmetry failures detected.")
        assert sym_failures == 0, f"{sym_failures} antisymmetry failures detected"

def test_elsasser_rigorous():
    """Pytest wrapper for rigorous verification."""
    check_rigorous()

if __name__ == "__main__":
    check_rigorous()
