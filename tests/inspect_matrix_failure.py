import numpy as np
import matplotlib.pyplot as plt

def inspect():
    try:
        M_ref = np.load("M_ref_failure.npy")
        M_gen = np.load("M_gen_failure.npy")
    except FileNotFoundError:
        print("Matrices not found.")
        return

    print(f"Shape: Ref {M_ref.shape}, Gen {M_gen.shape}")
    print(f"Dtype: Ref {M_ref.dtype}, Gen {M_gen.dtype}")
    
    # Norms
    n_ref = np.linalg.norm(M_ref)
    n_gen = np.linalg.norm(M_gen)
    print(f"Norms: Ref {n_ref:.4f}, Gen {n_gen:.4f}")
    
    # Correlation
    # <A, B> = trace(A.H @ B) = sum(conj(A) * B)
    dot_prod = np.vdot(M_ref, M_gen).real
    cos_sim = dot_prod / (n_ref * n_gen)
    print(f"Cosine Similarity (Real part): {cos_sim:.4f}")
    
    # Difference Norm
    diff = M_gen - M_ref
    n_diff = np.linalg.norm(diff)
    print(f"Diff Norm: {n_diff:.4f} (Rel: {n_diff/n_ref:.4f})")
    
    # Diagonal Analysis
    d_ref = np.diag(M_ref)
    d_gen = np.diag(M_gen)
    
    # Print first few diagonal elements
    print("\n--- Diagonal Elements (First 5) ---")
    for i in range(min(5, len(d_ref))):
        print(f"idx {i}: Ref {d_ref[i]:.4f}, Gen {d_gen[i]:.4f} | RMatch: {d_gen[i].real/d_ref[i].real:.4f} IMatch: {d_gen[i].imag - d_ref[i].imag:.4e}")
        
    print(f"Diag Norms: Ref {np.linalg.norm(d_ref):.4f}, Gen {np.linalg.norm(d_gen):.4f}")
    
    # Off-Diagonal Analysis
    # Mask diagonal
    eye = np.eye(M_ref.shape[0], dtype=bool)
    off_ref = M_ref[~eye]
    off_gen = M_gen[~eye]
    
    n_off_ref = np.linalg.norm(off_ref)
    n_off_gen = np.linalg.norm(off_gen)
    print(f"\nOff-Diag Norms: Ref {n_off_ref:.4f}, Gen {n_off_gen:.4f}")
    
    # Check simple relations
    # Try Signs?
    dot_off = np.vdot(off_ref, off_gen).real
    cos_off = dot_off / (n_off_ref * n_off_gen) if n_off_ref*n_off_gen > 0 else 0
    print(f"Off-Diag Cos Sim: {cos_off:.4f}")
    
    # Check Transpose?
    # Gen vs Ref.T
    dot_T = np.vdot(M_ref.T, M_gen).real
    cos_T = dot_T / (n_ref * n_gen)
    print(f"Transpose Cos Sim: {cos_T:.4f}")
    
    # Check Conjugate?
    dot_C = np.vdot(M_ref.conj(), M_gen).real
    cos_C = dot_C / (n_ref * n_gen)
    print(f"Conjugate Cos Sim: {cos_C:.4f}")

    # Inspect a Block
    # Print top-left 4x4
    print("\n--- Ref Block (4x4) ---")
    print(np.array2string(M_ref[:4,:4], precision=3, suppress_small=True))
    print("\n--- Gen Block (4x4) ---")
    print(np.array2string(M_gen[:4,:4], precision=3, suppress_small=True))

if __name__ == "__main__":
    inspect()
