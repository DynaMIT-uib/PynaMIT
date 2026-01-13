
import numpy as np
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid

def debug_vectors_full():
    print("\n--- LEGACY MATRIX INSPECTION (45, 45) ---")
    
    basis = SHBasis(Nmax=1, Mmax=1, Nmin=0)
    
    # Test Point: Theta=45, Phi=45
    th_deg = 45.0
    ph_deg = 45.0
    grid = Grid(theta=np.array([th_deg]), phi=np.array([ph_deg]))
    
    th = np.deg2rad(th_deg)
    ph = np.deg2rad(ph_deg)
    sin_th = np.sin(th)
    cos_th = np.cos(th) # 1/sqrt(2) approx 0.707
    sin_ph = np.sin(ph)
    cos_ph = np.cos(ph)
    
    print(f"Theta={th_deg}, Phi={ph_deg}")
    print(f"sin_th={sin_th:.4f}, cos_th={cos_th:.4f}")
    
    # Analytic Values for Schmidt Norm
    # P11c = \cos \phi \sin \theta
    # P11s = \sin \phi \sin \theta
    # T11c = \cos \phi \sin \theta
    # T11s = \sin \phi \sin \theta
    
    # Derivatives for Poloidal (-Grad P)
    # u_th = -dP/dth
    # u_ph = -1/sin dP/dphi
    
    # Analytic P11c:
    # dP/dth = cos phi cos theta
    # dP/dphi = -sin phi sin theta -> 1/sin dP/dphi = -sin phi
    # u_th_P11c = -cos phi cos theta
    # u_ph_P11c = +sin phi
    
    # Analytic T11c (Legacy Formula: r x Grad T)
    # u_th = -1/sin dT/dphi = -(-sin phi) = +sin phi
    # u_ph = +dT/dth = +cos phi cos theta
    
    # Analytic P11s:
    # dP/dth = sin phi cos theta
    # dP/dphi = cos phi sin theta -> 1/sin = cos phi
    # u_th_P11s = -sin phi cos theta
    # u_ph_P11s = -cos phi
    
    # Analytic T11s (Legacy: r x Grad T)
    # u_th = -1/sin dT/dphi = -(cos phi) = -cos phi
    # u_ph = dT/dth = sin phi cos theta
    
    G_vec_full = basis.get_vector_basis_matrix(grid)
    G_vec = G_vec_full.reshape(2, 1, 8)
    
    names = ["P00", "P10", "P11c", "P11s", "T00", "T10", "T11c", "T11s"]
    indices = {"P11c": 2, "P11s": 3, "T11c": 6, "T11s": 7}
    
    print("\n--- COMPARISON ---")
    
    # P11c
    u_th_meas = G_vec[0,0,indices["P11c"]]
    u_ph_meas = G_vec[1,0,indices["P11c"]]
    u_th_theory = -cos_ph * cos_th
    u_ph_theory = sin_ph
    print(f"\nP11c:")
    print(f"  u_th: Meas={u_th_meas:.4f}, Theory={u_th_theory:.4f}, Ratio={u_th_meas/u_th_theory:.4f}")
    print(f"  u_ph: Meas={u_ph_meas:.4f}, Theory={u_ph_theory:.4f}, Ratio={u_ph_meas/u_ph_theory:.4f}")

    # T11c
    u_th_meas = G_vec[0,0,indices["T11c"]]
    u_ph_meas = G_vec[1,0,indices["T11c"]]
    u_th_theory = sin_ph            # Derived from r x Grad T
    u_ph_theory = cos_ph * cos_th   # Derived from r x Grad T
    print(f"\nT11c (Legacy r x Grad T model):")
    print(f"  u_th: Meas={u_th_meas:.4f}, Theory={u_th_theory:.4f}, Ratio={u_th_meas/u_th_theory:.4f}")
    print(f"  u_ph: Meas={u_ph_meas:.4f}, Theory={u_ph_theory:.4f}, Ratio={u_ph_meas/u_ph_theory:.4f}")

    # P11s
    u_th_meas = G_vec[0,0,indices["P11s"]]
    u_ph_meas = G_vec[1,0,indices["P11s"]]
    u_th_theory = -sin_ph * cos_th
    u_ph_theory = -cos_ph
    print(f"\nP11s:")
    print(f"  u_th: Meas={u_th_meas:.4f}, Theory={u_th_theory:.4f}, Ratio={u_th_meas/u_th_theory:.4f}")
    print(f"  u_ph: Meas={u_ph_meas:.4f}, Theory={u_ph_theory:.4f}, Ratio={u_ph_meas/u_ph_theory:.4f}")

    # T11s
    u_th_meas = G_vec[0,0,indices["T11s"]]
    u_ph_meas = G_vec[1,0,indices["T11s"]]
    u_th_theory = -cos_ph             # Derived from r x Grad T
    u_ph_theory = sin_ph * cos_th       # Derived from r x Grad T
    print(f"\nT11s (Legacy r x Grad T model):")
    print(f"  u_th: Meas={u_th_meas:.4f}, Theory={u_th_theory:.4f}, Ratio={u_th_meas/u_th_theory:.4f}")
    print(f"  u_ph: Meas={u_ph_meas:.4f}, Theory={u_ph_theory:.4f}, Ratio={u_ph_meas/u_ph_theory:.4f}")

if __name__ == "__main__":
    debug_vectors_full()
