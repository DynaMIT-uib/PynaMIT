
import numpy as np
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid

def debug_legacy_matrix():
    print("\n--- LEGACY MATRIX INSPECTION N=1 ---")
    
    basis = SHBasis(Nmax=1, Mmax=1, Nmin=0)
    
    # Define a single point at (90, 0) to check values
    grid = Grid(theta=np.array([90.0]), phi=np.array([0.0]))
    
    print("\nGetting Gradient Matrix (Scalars)...")
    G_grad = basis.get_gradient_matrix(grid)
    # Shape: (2, N_grid=1, N_coeffs=4)
    # Coeffs: [00, 10, 11c, 11s]
    
    print(f"G_grad shape: {G_grad.shape}")
    
    G_th = G_grad[0, 0, :]
    G_ph = G_grad[1, 0, :]
    
    names = ["P00", "P10", "P11c", "P11s"]
    print("\nG_theta (dP/dth):")
    for i, val in enumerate(G_th):
        print(f"{names[i]}: {val:.4f}")
        
    print("\nG_phi (1/sin dY/dphi):")
    for i, val in enumerate(G_ph):
        print(f"{names[i]}: {val:.4f}")
        
    print("\nGetting Vector Matrix (Pol/Tor)...")
    # This matrix stacks [-Grad, Curl]
    # G_vec = [-G_th_P -G_ph_P | -G_ph_T  G_th_T ]
    #         [-G_ph_P  G_th_P |  G_th_T  G_ph_T ??? No. ]
    
    G_vec_full = basis.get_vector_basis_matrix(grid)
    # G_vec_full shape: (2, 1, 2, 4) -> (Component, Grid, Vectors(Pol/Tor), Coeffs)
    # Flatten last two dimensions -> (2, 1, 8)
    G_vec_flat = G_vec_full.reshape(2, 1, 8)
    
    print(f"G_vec_flat shape: {G_vec_flat.shape}")
    
    # Print the row for u_theta (Row 0)
    # Coeffs: P00..P11s, T00..T11s
    row_th = G_vec_flat[0, 0, :]
    row_ph = G_vec_flat[1, 0, :]
    
    coeff_names = names + ["T" + n[1:] for n in names]
    
    print("\nMatrix Row u_theta:")
    for i, val in enumerate(row_th):
        if abs(val) > 1e-9:
            print(f"{coeff_names[i]}: {val:.4f}")
            
    print("\nMatrix Row u_phi:")
    for i, val in enumerate(row_ph):
        if abs(val) > 1e-9:
            print(f"{coeff_names[i]}: {val:.4f}")

if __name__ == "__main__":
    debug_legacy_matrix()
