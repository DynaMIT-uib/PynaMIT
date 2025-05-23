import numpy as np
from scipy.linalg import expm


def krylov_expmv(matvec, v, t=1.0, tol=1e-8, m_max=50):
    """
    Computes u = exp(t*A) v using matrix-free Arnoldi and adaptive Krylov projection.

    Parameters:
    - matvec: function x ↦ A @ x (can include arbitrary operations)
    - v: initial vector
    - t: time multiplier (exp(t*A) @ v)
    - tol: convergence tolerance on update norm
    - m_max: max Krylov subspace size

    Returns:
    - u: approximation to exp(t*A) @ v
    """

    n = len(v)
    beta = np.linalg.norm(v)
    if beta == 0:
        return np.zeros_like(v)

    V = np.zeros((n, m_max + 1), dtype=v.dtype)
    H = np.zeros((m_max + 1, m_max), dtype=v.dtype)
    V[:, 0] = v / beta

    u_prev = np.zeros_like(v)

    for m in range(1, m_max + 1):
        # Arnoldi iteration: A @ V[:, m-1]
        w = matvec(V[:, m - 1])

        for j in range(m):
            H[j, m - 1] = np.dot(V[:, j].conj(), w)
            w -= H[j, m - 1] * V[:, j]

        H[m, m - 1] = np.linalg.norm(w)
        if H[m, m - 1] < 1e-14:
            # Breakdown — Krylov space is invariant
            break
        V[:, m] = w / H[m, m - 1]

        # Compute exp(t * H_m) @ e1
        Hm = H[:m, :m]
        Vm = V[:, :m]
        e1 = np.zeros(m)
        e1[0] = 1.0
        expHm_e1 = expm(t * Hm) @ e1
        u = beta * Vm @ expHm_e1

        # Check convergence
        err = np.linalg.norm(u - u_prev)
        if err < tol:
            print(f"Converged in {m} iterations with error {err:.2e}")
            exit()
            break
        u_prev = u

    return u
