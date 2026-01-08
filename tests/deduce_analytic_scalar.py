import numpy as np
import sys
import os
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.math.gaunt import GauntEngine, get_real_decomposition
from pynamit.math.wigner import wigner_3j

# Ensure src in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def deduce_scalar_form():
    print("Deducing Analytic Form of S(li, lj, lk)...")
    
    basis = SHBasis(Nmax=8, Mmax=8)
    engine = GauntEngine(basis, grid_resolution=30)
    
    # Generate data points
    triplets = []
    # Only Odd Sum
    for li in range(1, 8):
        for lj in range(1, 8):
            for lk in range(abs(li-lj), li+lj+1):
                if (li+lj+lk)%2 != 0:
                    triplets.append((li, lj, lk))

    print(f"Analyzing {len(triplets)} triplets...")
    
    results = []
    
    for (li, lj, lk) in triplets:
        # Compute S numerically
        best_exact, best_ang = 0.0, 0.0
        # Search for good m
        limit = 3
        found = False
        for mi in range(-limit, limit+1):
            if abs(mi) > li: continue
            if found: break
            for mj in range(-limit, limit+1):
                if abs(mj) > lj: continue
                mk = -(mi+mj)
                if abs(mk) > lk: continue
                
                ang = 0.0j
                di = get_real_decomposition(li, mi)
                dj = get_real_decomposition(lj, mj)
                dk = get_real_decomposition(lk, mk)
                for ci,mui in di:
                    for cj,muj in dj:
                        for ck,muk in dk:
                            if mui+muj+muk==0:
                                ang += ci*cj*ck*wigner_3j(li,lj,lk,mui,muj,muk)
                                
                if abs(ang) > 1e-8:
                    ex = engine._compute_elsasser_gl_raw(li,mi,lj,mj,lk,mk)
                    if abs(ex) > 1e-6:
                        S_scalar = ex / ang
                        found = True
                        
                        S_prod = np.sqrt((2*li+1)*(2*lj+1)*(2*lk+1)/(4*np.pi))
                        # Check reduced
                        red = S_scalar / S_prod
                        
                        results.append({
                            'L': (li, lj, lk),
                            'S': S_scalar,
                            'Red': red
                        })
                        p = S_scalar
                        print(f"L=({li},{lj},{lk}) S={p:.4f} Red={red:.4f} AbsRed={abs(red):.4f}")
                        break
                        
    # Analyze dependence
    print("\nRaw Values Analysis:")
    for r in results:
        L = r['L']
        M = r['M']
        Ex = r['Ex']
        # Try normalizing by Ssqrt factors
        S_prod = np.sqrt((2*L[0]+1)*(2*L[1]+1)*(2*L[2]+1)/(4*np.pi))
        Ratio = Ex / S_prod
        
        print(f"L={L} M={M} Ex={Ex:.6f} Ex/S={Ratio:.6f}")


if __name__ == "__main__":
    deduce_scalar_form()
