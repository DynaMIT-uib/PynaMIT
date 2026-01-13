import numpy as np
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid
import scipy.linalg

def debug_vector_n1():
    print("\n--- MATRIX DEBUGGER N=1 ---")
    
    # 1. Setup Minimal Basis
    N_sh = 1
    M_max = 1
    basis = SHBasis(Nmax=N_sh, Mmax=M_max, Nmin=0)
    # Coeffs: [P00, P10, P11c, P11s, T00, T10, T11c, T11s] (Total 8)

    # 2. Setup High-Res Grid (to ensure invertibility of G_helmholtz)
    # Use Gaussian quadrature points for accuracy
    # N=1 -> need at least 2 points in theta? Use 4 for safety.
    # M=1 -> need at least 3 points in phi? Use 4.
    theta = np.deg2rad([45.0, 90.0, 135.0, 150.0])
    phi = np.deg2rad([0.0, 90.0, 180.0, 270.0])
    tt, pp = np.meshgrid(theta, phi, indexing='ij')
    grid = Grid(theta=np.rad2deg(tt).flatten(), phi=np.rad2deg(pp).flatten())
    
    N_grid = grid.size
    print(f"Grid Size: {N_grid}")

    # 3. Compute Legacy Inverse Operator (Grid -> Coeffs)
    # The Legacy method calculates c = pinv(G_helmholtz) * u_grid
    # Let's inspect pinv(G_helmholtz).
    # G_helmholtz maps coeffs -> [u_th, u_ph]. Shape (2*N_grid, 2*N_coeffs) = (32, 16)?
    # Basis N=1, M=1 -> N_coeffs = 1 (l=0) + 3 (l=1) = 4 per component. Total 8.
    # So G is (32, 8).
    
    # Helper to get dense G_helmholtz
    G_grad = basis.get_gradient_matrix(grid) # (2, N_grid, N_coeffs_scalar)
    G_curl = basis.get_curl_matrix(grid)     # (2, N_grid, N_coeffs_scalar)
    
    # Check shape
    Nc_scalar = G_grad.shape[2]
    # N=1, M=1 scalar coeffs: 00, 10, 11c, 11s -> 4.
    
    # Construct Full Matrix G_vec mapping [C_pol, C_tor] -> [u_th, u_ph]
    # u = -Grad P + RxGrad T
    # u = [-Gth_P, -Gph_P] * Cp + [-Gph_T, Gth_T] * Ct
    # u_th = -Gth * Cp - Gph * Ct
    # u_ph = -Gph * Cp + Gth * Ct
    
    # Stack columns: [ -Grad | Curl ]
    # -Grad = [-Gth, -Gph] (stacked vertically in G_vec rows? No)
    # G_vec rows: [u_th_0, ..., u_th_N, u_ph_0, ... u_ph_N]
    
    G_th_scalar = G_grad[0] # (N_grid, 4)
    G_ph_scalar = G_grad[1] # (N_grid, 4)
    
    # Polodal Block (u_th part, u_ph part)
    # u_th_pol = -Gth * Cp
    # u_ph_pol = -Gph * Cp
    G_pol_block = np.concatenate([-G_th_scalar, -G_ph_scalar], axis=0) # (2*N_grid, 4)
    
    # Toroidal Block
    # u_th_tor = -Gph * Ct (Note: Curl_th = -1/sin dY/dphi = -G_ph)
    # u_ph_tor = +Gth * Ct (Note: Curl_ph = dY/dtheta = G_th)
    G_tor_block = np.concatenate([-G_ph_scalar, G_th_scalar], axis=0) # (2*N_grid, 4)
    
    G_full = np.concatenate([G_pol_block, G_tor_block], axis=1) # (2*N_grid, 8)
    
    # Compute P_inv (The Operator we are applying in Legacy)
    P_legacy = np.linalg.pinv(G_full) # (8, 2*N_grid)
    
    print("Legacy Pinv Shape:", P_legacy.shape)
    
    # 4. Compute Fast Operator Blocks
    # The fast solver does this per-m via block inversion.
    # For N=1, M=1:
    # m=0: P00, P10, T00, T10.
    # m=1: P11c, P11s, T11c, T11s.
    
    # Let's look at m=1 Coupling specifically.
    # Coeffs indices for m=1: 
    # l=1 is index 2 (00, 10, 11c...). 
    # cnm.m array:
    print("CnM m:", basis.cnm.m.flatten()) # 0, 0, 1, 0...
    print("SnM m:", basis.snm.m.flatten()) # 0, 0, 1, 0...
    
    # m=1 indices
    idx_c = np.where(basis.cnm.m.flatten() == 1)[0] # [2] (P11c)
    idx_s = np.where(basis.snm.m.flatten() == 1)[0] # [2] (P11s)
    # Note: Indices in coeffs vector. 
    # Pol coeffs: [P00, P10, P11c, P11s] -> 0, 1, 2, 3
    # Tor coeffs: [T00, T10, T11c, T11s] -> 4, 5, 6, 7
    # idx_c corresponds to 1c?
    
    # Let's pick a single test point to check the matrix equation
    # Theta=pi/2 (90), Phi=0.
    th_val = np.pi/2
    ph_val = 0.0
    
    # Re-evaluate G scalar at this point
    P11 = 1.0 # sin(th) at 90 is 1. Schmidt P11?
    # Basis checks...
    # Just use get_G at single point
    grid_pt = Grid(theta=np.array([90.0]), phi=np.array([0.0]))
    G_pt = basis.get_G(grid_pt, derivative=None) # P
    Gt_pt = basis.get_G(grid_pt, derivative='theta') # dP
    Gp_pt = basis.get_G(grid_pt, derivative='phi') # dY/dphi
    
    print("\n--- Point Check (90, 0) ---")
    print("G (P):", G_pt)
    print("G_th (dP):", Gt_pt)
    print("G_ph (dY/dphi/sin):", Gp_pt)
    
    # Focus on m=1 terms (Cols 2 and 3)
    # P11c (idx 2): P ~ sin(th) cos(phi). at (90,0) -> 1 * 1 = 1.
    # P11s (idx 3): P ~ sin(th) sin(phi). at (90,0) -> 1 * 0 = 0.
    
    # Derivatives
    # dP/dth 11c: cos(th) cos(phi) -> 0 * 1 = 0.
    # dP/dth 11s: cos(th) sin(phi) -> 0 * 0 = 0.
    
    # dY/dphi/sin:
    # 11c: -1 * sin(th) sin(phi) / sin(th) = -sin(phi) -> 0.
    # 11s:  1 * sin(th) cos(phi) / sin(th) =  cos(phi) -> 1.
    
    # Fast Code Matrix Construction for m=1
    # Block 1 [Pol_c, Tor_s] -> [u_th_c, u_ph_s]
    # In Fast Code:
    # A11 = -Gp (G_th scalar). At (90,0): G_th=0 -> A11=0.
    # A12 = -G_ang (G_ph scalar). At (90,0): G_ph (m=1, c? no s?)
    # G_ang uses idx_s_out (11s). Value is 1. So A12 = -1.
    # A_block1 = [[0, -1], [?, ?]]
    
    # Let's derive u from definitions at (90,0):
    # u = -Grad(P11c) + Curl(T11s)
    # P11c potential = sin(th)cos(phi).
    # Grad_th = cos(th)cos(phi)=0. Grad_ph = -sin(phi)=0.
    # -Grad P11c = [0, 0].
    
    # T11s potential = sin(th)sin(phi).
    # Curl_th = 1/sin d/dphi = cos(phi) = 1.
    # Curl_ph = -d/dth = -cos(th)sin(phi) = 0.
    # Curl T11s = [1, 0].
    
    # So for input Coeffs [P=1, T=1]: u = [0+1, 0+0] = [1, 0].
    # u_th = 1, u_ph = 0.
    
    # Fast Code Inputs:
    # u_th is cosine term?
    # FFT of [1, 0] at phi=0?
    
    # This manual trace is tricky.
    # Better to just run grid_to_basis_fast with identity input and see what coeffs it produces.
    
    print("\n--- Injection Test ---")
    # Feed u_th = T11s field (Pure Toroidal m=1 Sine).
    # u_theta = cos(phi) * 1 (from Curl T11s calculation above)
    # u_phi = 0
    
    # Define fields on grid
    u_th_in = np.cos(pp) * 1.0 # Matches m=1 cosine behavior?
    # Wait, T11s field is m=1 sine?
    # T11s = S_11 = sin(th)sin(phi).
    # u_tor = curl(r S_11).
    # u_th = 1/sin dS/dphi = cos(phi). (Cosine dependence in theta... no, const in theta at 90)
    # u_ph = -dS/dth = -cos(th)sin(phi).
    
    # At th=90: u_th = cos(phi), u_ph = 0.
    # This is a pure T11s signal.
    
    u_th_mesh = np.cos(pp) * (np.sin(tt)/np.sin(tt)) # Broadcast
    u_ph_mesh = np.zeros_like(pp)
    
    data_tuple = (u_th_mesh, u_ph_mesh)
    
    # Fast Solve
    weights = np.sin(theta)
    c_fast = basis.grid_to_basis_fast(data_tuple, theta, phi, weights=weights, reg_lambda=0, vector_type='tangential')
    
    # Expect: T11s = 1.0. All else 0.
    names = ["P00", "P10", "P11c", "P11s", "T00", "T10", "T11c", "T11s"]
    print("\nCoeffs for Pure T11s Input (u_th=cos(phi)):")
    for i, val in enumerate(c_fast):
        if abs(val) > 1e-5:
            print(f"{names[i]}: {val:.5f}")

if __name__ == "__main__":
    debug_vector_n1()
