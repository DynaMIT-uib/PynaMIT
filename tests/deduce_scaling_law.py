
import numpy as np
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid
from pynamit.spherical_harmonics.gaunt import GauntEngine
from pynamit.utils import tensor_pinv

def deduce_scaling():
    """
    Compute Matrix Norm Ratio (Ref/Gen) for pure L-shells.
    Directly constructing matrices to avoid Geometry dependency hell.
    """
    print("Deducing Analytic Scaling Law...")
    print(f"{'L':<4} {'RefNorm':<12} {'GenNorm':<12} {'Ratio':<12} {'1/Ratio':<12}")
    print("-" * 60)
    
    # Grid Res: need to resolve L. 2L+2 is safe.
    # L range 1..5
    for L in range(1, 6):
        N = L
        M = L
        basis = SHBasis(Nmax=N, Mmax=M)
        
        # Grid
        # Use simple Gauss-Legendre Grid
        N_grid = 2 * N + 4
        theta = np.linspace(0, 180, N_grid)
        phi = np.linspace(0, 360, N_grid*2)
        TH, PH = np.meshgrid(theta, phi)
        grid = Grid(theta=TH.flatten(), phi=PH.flatten())
        
        # 1. Reference Implementation (True Stiffness Matrix)
        # M_ref = Integral ( E_i . E_j ) dOmega
        # E_i are columns of G_vec.
        # M_ref = G_vec.H @ W @ G_vec
        
        import scipy.sparse
        
        def get_G_vec(basis, grid):
            # Evaluate derivatives using Matrix API
            Y_th = basis.get_evaluation_matrix(grid, derivative='theta')
            Y_ph = basis.get_evaluation_matrix(grid, derivative='phi')
            
            # If sparse, densify
            if scipy.sparse.issparse(Y_th): Y_th = Y_th.toarray()
            if scipy.sparse.issparse(Y_ph): Y_ph = Y_ph.toarray()
            
            # Sin(theta) for 1/sin factor
            sin_th = np.sin(np.deg2rad(grid.theta))
            # Avoid singularity approx
            sin_th[np.abs(sin_th)<1e-10] = 1e-10
            inv_sin = 1.0/sin_th
            
            # Poloidal
            GP_th = -Y_th
            GP_ph = -Y_ph * inv_sin[:, None]
            
            # Toroidal
            GT_th = Y_ph * inv_sin[:, None]
            GT_ph = -Y_th
            
            # Full Vector E field at each point has 2 components (Th, Ph)
            # We stack them vertically: [E_th_all; E_ph_all]
            # Weights should assume this structure.
            # Integral (E . E) = Integral (E_th^2 + E_ph^2).
            # So W matrix should look like Diag([w, w]).
            
            top = np.hstack([GP_th, GT_th])
            bot = np.hstack([GP_ph, GT_ph])
            G_vec = np.vstack([top, bot])
            return G_vec
            
        G_vec = get_G_vec(basis, grid)
        
        # 1. Reference Implementation (Identity / PynaMIT-like)
        # PynaMIT Simulation Ref effectively normalizes interactions.
        # We test against Identity to confirm Scaling Correction works.
        G_inv = np.linalg.pinv(G_vec)
        M_ref = G_inv @ G_vec
        
        # 2. Analytic Implementation
        # We call gaunt with Unity Sigma.
        # Sigma coeffs: L=0, m=0 coeff = sqrt(4pi)? 
        # PynaMIT Schmidt: Y_00 = 1.0 (since 4pi/(2*0+1) = 4pi, sqrt=sqrt(4pi)).
        # Wait, norm is 4pi. Y_00 = 1? 
        # Y_lm * Y_lm* = 1? No, Integral |Y|^2 = 4pi.
        # So Y_00 = 1.
        # Coeff for Identity field (Sigma=1 everywhere).
        # Sigma(r) = Sum c * Y.
        # If Y_00 = 1, then c_00 = 1.
        
        coeffs = np.zeros(basis.index_length * 4) # (PP, PT, ...) -> Just (PP, MM, PM, MP) -> (1, 0, 0, 1) diagonal?
        # Sigma tensor is 2x2. Diag=1.
        # PP=1, TT=1? No, sigma tensor has components.
        # Simpler: Isotropic Sigma = 1.
        # coeffs_pp = coeffs_tt = 1 (L=0).
        # coeffs_pm = coeffs_mp = 0.
        
        # Prepare coeffs array for Gaunt
        # Expected: [c_pp, c_mm, c_pm, c_mp]
        # Each is array of length L_sigma_basis.
        # L=0 basis has length 1.
        
        # We need a sigma basis (L=0). 
        # gaunt.py handles internally?
        # It expects coeffs arrays.
        
        # Pad coeffs to match SHBasis(Nmin=0) length
        # N=L.
        Nmin0_len = SHBasis(Nmax=L, Mmax=L, Nmin=0).index_length
        
        c_2_padded = np.zeros(Nmin0_len)
        c_2_padded[0] = 2.0 # (0,0) component
        
        c_0_padded = np.zeros(Nmin0_len)
        
        kwargs = {
            "coeffs_pp": c_2_padded,
            "coeffs_mm": c_2_padded,
            "coeffs_pm": c_0_padded,
            "coeffs_mp": c_0_padded
        }
        
        # Instantiate Engine
        engine = GauntEngine(basis)
        M_gen = engine.get_general_analytic_interaction_matrix(**kwargs)
        
        # 3. Compare Norms
        norm_ref = np.linalg.norm(M_ref)
        norm_gen = np.linalg.norm(M_gen)
        
        ratio = norm_gen / norm_ref if norm_ref > 0 else 0
        
        # Compare Diagonal 0,0
        # M is (4*N_min0_len) square?
        # Basis Nmax=L. Nmin=1. IndexLength = L*(L+2).
        # M_ref (Ncoeffs, Ncoeffs).
        # Element 0,0 corresponds to L=1, m=-1? Or L=1, m=0?
        # index 0 is (1,0).
        val_ref = M_ref[0,0]
        val_gen = M_gen[0,0]
        
        print(f"{L:<4} {norm_ref:<12.4e} {norm_gen:<12.4e} {ratio:<12.4f} 1/R:{1/ratio:<8.4f} ElRef:{val_ref.real:.2f} ElGen:{val_gen.real:.2f}")

if __name__ == "__main__":
    deduce_scaling()
