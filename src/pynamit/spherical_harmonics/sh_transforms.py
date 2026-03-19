"""Fast Spherical Harmonic Transforms.

This module provides optimized transform methods for converting between
grid values and spherical harmonic coefficients on regular grids.
"""

from typing import Tuple, Union, Optional, TYPE_CHECKING
import numpy as np
from pynamit.primitives.basis import (
    get_repo_cf_helmholtz_sign,
    get_repo_df_helmholtz_sign,
)

if TYPE_CHECKING:
    from pynamit.spherical_harmonics.sh_basis import SHBasis


def grid_to_basis_fast(
    basis: "SHBasis",
    data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    theta: np.ndarray,
    phi: np.ndarray = None,
    weights: np.ndarray = None,
    reg_lambda: float = None,
    vector_type: str = "scalar",
) -> np.ndarray:
    """Fast Spherical Harmonic Transform for Regular Grids via Separation of Variables.

    This method is significantly faster (O(N^3)) than the generic Least Squares
    solver (O(N^4)) for data situated on a regular grid (separable in theta/phi).
    It performs an FFT in longitude (phi) followed by a per-m Regularized Least Squares
    fit in colatitude (theta).

    Parameters
    ----------
    basis : SHBasis
        The spherical harmonic basis.
    data : np.ndarray or Tuple[np.ndarray, np.ndarray]
        Input data.
        - If scalar: array of shape (N_theta, N_phi).
        - If tangential: tuple (u_theta, u_phi), each (N_theta, N_phi).
    theta : np.ndarray
        1D array of colatitudes in radians, shape (N_theta,).
    phi : np.ndarray, optional
        1D array of longitudes in radians, shape (N_phi,).
    weights : np.ndarray, optional
        1D array of weights for the theta dimension, shape (N_theta,).
    reg_lambda : float, optional
        Tikhonov regularization parameter.
    vector_type : str
        "scalar" or "tangential".

    Returns
    -------
    coeffs : np.ndarray
        Spectral coefficients vector.
    """
    from pynamit.primitives.grid import Grid

    N_theta = theta.size

    # Setup and Validation
    is_vector = vector_type == "tangential"

    if is_vector:
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            raise ValueError("For vector_type='tangential', data must be (u_theta, u_phi)")
        d_th_in = np.asarray(data[0], dtype=float)
        d_ph_in = np.asarray(data[1], dtype=float)
        if d_th_in.shape != d_ph_in.shape:
            raise ValueError(
                "Tangential fast-path inputs must share the same shape, got "
                f"{d_th_in.shape} and {d_ph_in.shape}."
            )
        if d_th_in.ndim == 2:
            d_th_in = d_th_in[np.newaxis, ...]
            d_ph_in = d_ph_in[np.newaxis, ...]
        elif d_th_in.ndim != 3:
            raise ValueError(
                "Tangential fast-path inputs must have shape (N_theta, N_phi) or "
                f"(N_time, N_theta, N_phi), got {d_th_in.shape}."
            )
        n_scenarios, N_theta, N_phi = d_th_in.shape
    else:
        d_in = np.asarray(data, dtype=float)
        if d_in.ndim == 2:
            d_in = d_in[np.newaxis, ...]
        elif d_in.ndim != 3:
            raise ValueError(
                "Scalar fast-path inputs must have shape (N_theta, N_phi) or "
                f"(N_time, N_theta, N_phi), got {d_in.shape}."
            )
        n_scenarios, N_theta, N_phi = d_in.shape

    if len(theta) != N_theta:
        raise ValueError(f"Theta shape {theta.shape} mismatch with data rows {N_theta}")

    # FFT in Longitude (index-space periodic ordering)
    if is_vector:
        fft_th = np.fft.fft(d_th_in, axis=2) / N_phi
        fft_ph = np.fft.fft(d_ph_in, axis=2) / N_phi
    else:
        fft_scalar = np.fft.fft(d_in, axis=2) / N_phi

    # Correct FFT phase when the longitude samples are uniformly spaced but
    # start at phi0 != 0 (e.g. [-180, 180) grids starting at 180 deg).
    # The FFT is taken over sample index j; if phi_j = phi0 + j * 2*pi/N (mod 2*pi),
    # then mode m picks up a factor exp(+i m phi0). We remove that so the per-m
    # targets match the real SH basis convention in actual longitude phi.
    if phi is not None:
        phi_arr = np.asarray(phi, dtype=float).reshape(-1)
        if phi_arr.size != N_phi:
            raise ValueError(f"Phi shape {phi_arr.shape} mismatch with data cols {N_phi}")
        if N_phi > 1:
            phi_unwrapped = np.unwrap(phi_arr)
            dphi = np.diff(phi_unwrapped)
            dphi_ref = 2 * np.pi / N_phi
            if np.allclose(dphi, dphi[0], rtol=0.0, atol=1e-10) and np.isclose(
                abs(dphi[0]), dphi_ref, rtol=0.0, atol=1e-10
            ):
                phase_sign = 1.0 if dphi[0] > 0 else -1.0
                m_idx = np.arange(N_phi, dtype=float)
                phase = np.exp(-1j * phase_sign * m_idx * phi_unwrapped[0])
                if is_vector:
                    fft_th = fft_th * phase[None, None, :]
                    fft_ph = fft_ph * phase[None, None, :]
                else:
                    fft_scalar = fft_scalar * phase[None, None, :]
            # If the grid is not the expected uniform periodic grid, we silently skip
            # phase correction and rely on the caller's fast-path regularity checks.

    # Pre-compute 1D Legendre Matrices
    theta_deg = np.rad2deg(theta)
    grid_1d = Grid(theta=theta_deg, phi=np.zeros(N_theta))

    G_0 = basis.get_evaluation_matrix(grid_1d, derivative=None)

    if is_vector:
        G_th = basis.get_evaluation_matrix(grid_1d, derivative="theta")
        G_ph = basis.get_evaluation_matrix(grid_1d, derivative="phi")

    # Regularization Setup
    reg_L = None
    if reg_lambda is not None and reg_lambda > 0:
        if is_vector:
            pen_pol = basis.n * (basis.n + 1) / (2 * basis.n + 1)
            pen_tor = (basis.n + 1) / 2
            reg_L = (pen_pol, pen_tor)
        else:
            reg_L = basis.n

    # Weights handling
    W_diag = None
    if weights is not None:
        W_diag = weights if weights.ndim == 1 else weights.flatten()

    # Indices
    offset_s = basis.cnm.n.size
    coeffs = np.zeros((basis.index_length * (2 if is_vector else 1), n_scenarios), dtype=float)

    # Solve per m
    limit_m = min(basis.Mmax, N_phi // 2)

    # Pre-compute Global Regularization Scaling
    scale_A_global = 1.0
    scale_L_global = 1.0

    if reg_lambda is not None and reg_lambda > 0 and reg_L is not None:
        all_norms_A = []
        all_norms_L = []

        for m in range(limit_m + 1):
            mask_c = basis.cnm.m.flatten() == m
            if not np.any(mask_c):
                continue
            idx_c_out = np.where(mask_c)[0]

            mask_s = basis.snm.m.flatten() == m
            idx_s_out = (np.where(mask_s)[0] + offset_s) if np.any(mask_s) else []

            # L-Penalty Norms
            if isinstance(reg_L, tuple):
                pen_pol, pen_tor = reg_L
                p_vals = pen_pol[idx_c_out % len(pen_pol)]
                all_norms_L.append(p_vals**2)
                if m > 0:
                    t_vals = pen_tor[idx_c_out % len(pen_pol)]
                    all_norms_L.append(t_vals**2)
                if np.any(mask_s):
                    p_vals_s = pen_pol[idx_s_out % len(pen_pol)]
                    all_norms_L.append(p_vals_s**2)
                    t_vals_s = pen_tor[idx_s_out % len(pen_pol)]
                    all_norms_L.append(t_vals_s**2)
            else:
                l_vals = reg_L[idx_c_out]
                all_norms_L.append(l_vals**2)
                if np.any(mask_s):
                    all_norms_L.append(reg_L[idx_s_out] ** 2)

            # A-Matrix Norms
            phi_factor = N_phi / 2.0 if m > 0 else float(N_phi)
            w_eff = W_diag if W_diag is not None else np.ones(N_theta)

            if not is_vector:
                G_sub = G_0[:, idx_c_out]
                Gw = G_sub * w_eff[:, None]
                n_A = np.sum(Gw**2, axis=0) * phi_factor
                all_norms_A.append(n_A)
                if np.any(mask_s):
                    all_norms_A.append(n_A)
            else:
                Gp = G_th[:, idx_c_out]
                Gang = np.zeros_like(Gp)
                if m > 0:
                    Gang = G_ph[:, idx_c_out]
                term = (Gp**2 + Gang**2) * (w_eff[:, None] ** 2)
                n_vec = np.sum(term, axis=0) * phi_factor
                all_norms_A.append(n_vec)
                all_norms_A.append(n_vec)
                if np.any(mask_s):
                    all_norms_A.append(n_vec)
                    all_norms_A.append(n_vec)

        if all_norms_A:
            flat_A = np.concatenate(all_norms_A)
            valid_A = flat_A[flat_A > 1e-14]
            scale_A_global = np.median(valid_A) if valid_A.size else 1.0

        if all_norms_L:
            flat_L = np.concatenate(all_norms_L)
            valid_L = flat_L[flat_L > 1e-14]
            scale_L_global = np.median(valid_L) if valid_L.size else 1.0

    for m in range(limit_m + 1):
        # Identify active L-indices for this m
        mask_c = basis.cnm.m.flatten() == m
        if not np.any(mask_c):
            continue
        idx_c_out = np.where(mask_c)[0]

        mask_s = basis.snm.m.flatten() == m
        has_sine = np.any(mask_s)
        idx_s_out = (np.where(mask_s)[0] + offset_s) if has_sine else []

        # SCALAR CASE
        if not is_vector:
            G_sub_c = G_0[:, idx_c_out]
            d_c, d_s = _get_fft_targets(fft_scalar, m)
            _solve_stacked(
                G_sub_c,
                d_c,
                coeffs,
                idx_c_out,
                W_diag,
                reg_lambda,
                reg_L,
                idx_c_out,
                scale_A_global,
                scale_L_global,
            )
            if has_sine:
                _solve_stacked(
                    G_sub_c,
                    d_s,
                    coeffs,
                    idx_s_out,
                    W_diag,
                    reg_lambda,
                    reg_L,
                    idx_c_out,
                    scale_A_global,
                    scale_L_global,
                )

        # VECTOR CASE
        else:
            if m == 0:
                # Decoupled Logic for Zonal Flow
                d_th = fft_th[:, :, 0].real.T
                d_ph = fft_ph[:, :, 0].real.T

                Gp = G_th[:, idx_c_out]

                # Poloidal Solve
                idx_pol = idx_c_out
                _solve_stacked(
                    -Gp,
                    d_th,
                    coeffs,
                    idx_pol,
                    W_diag,
                    reg_lambda,
                    reg_L,
                    (idx_pol, []),
                    scale_A_global,
                    scale_L_global,
                )

                # Toroidal Solve
                idx_tor = idx_c_out + basis.index_length
                _solve_stacked(
                    -Gp,
                    d_ph,
                    coeffs,
                    idx_tor,
                    W_diag,
                    reg_lambda,
                    reg_L,
                    ([], idx_tor),
                    scale_A_global,
                    scale_L_global,
                )
            else:
                # Coupled Logic (m > 0)
                t_th_c, t_th_s = _get_fft_targets(fft_th, m)
                t_ph_c, t_ph_s = _get_fft_targets(fft_ph, m)

                Gp = G_th[:, idx_c_out]
                G_ang = G_ph[:, idx_s_out]

                W_block = np.concatenate([W_diag, W_diag]) if W_diag is not None else None

                # Block 1
                A_11 = -Gp
                A_12 = G_ang
                A_21 = G_ang
                A_22 = -Gp

                A_block1 = np.block([[A_11, A_12], [A_21, A_22]])
                b1 = np.concatenate([t_th_c, t_ph_s], axis=0)

                idx_pol_c = idx_c_out
                idx_tor_s = idx_s_out + basis.index_length
                dest_indices_1 = np.concatenate([idx_pol_c, idx_tor_s])

                _solve_stacked(
                    A_block1,
                    b1,
                    coeffs,
                    dest_indices_1,
                    W_block,
                    reg_lambda,
                    reg_L,
                    (idx_c_out, idx_s_out),
                    scale_A_global,
                    scale_L_global,
                )

                # Block 2
                if has_sine:
                    b2 = np.concatenate([t_th_s, t_ph_c], axis=0)
                    A_11 = -Gp
                    A_12 = -G_ang
                    A_21 = -G_ang
                    A_22 = -Gp
                    A_block2 = np.block([[A_11, A_12], [A_21, A_22]])

                    idx_pol_s = idx_s_out
                    idx_tor_c = idx_c_out + basis.index_length
                    dest_indices_2 = np.concatenate([idx_pol_s, idx_tor_c])
                    _solve_stacked(
                        A_block2,
                        b2,
                        coeffs,
                        dest_indices_2,
                        W_block,
                        reg_lambda,
                        reg_L,
                        (idx_s_out, idx_c_out),
                        scale_A_global,
                        scale_L_global,
                    )
    if is_vector:
        # The fast tangential transform is derived for the canonical historical
        # SH basis ``[-grad, -rhat x grad]``. Convert the recovered Helmholtz
        # coefficients to the active repo convention when the global cf/df
        # signs are changed.
        coeffs[: basis.index_length] *= (-1.0 / float(get_repo_cf_helmholtz_sign()))
        coeffs[basis.index_length :] *= (-1.0 / float(get_repo_df_helmholtz_sign()))

    if n_scenarios == 1:
        return coeffs[:, 0]
    return coeffs


def _get_fft_targets(fft_data: np.ndarray, m: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract Real (Cosine) and Imag (Sine) targets from FFT.

    Parameters
    ----------
    fft_data : np.ndarray
        FFT data array.
    m : int
        Order index.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        Cosine and Sine targets.
    """
    if m == 0:
        return fft_data[:, :, 0].real.T, None
    else:
        return 2 * fft_data[:, :, m].real.T, -2 * fft_data[:, :, m].imag.T


def _solve_stacked(
    A: np.ndarray,
    b: np.ndarray,
    coeffs_out: np.ndarray,
    dest_idxs: np.ndarray,
    weights: Optional[np.ndarray],
    reg_lambda: Optional[float],
    reg_L,
    reg_idxs_source,
    scale_A_forced: Optional[float] = None,
    scale_L_forced: Optional[float] = None,
) -> None:
    """Solve weighted regularized system using Stacked Matrices.

    min || W(Ax - b) ||^2 + lambda || L x ||^2

    Parameters
    ----------
    A : np.ndarray
        Design matrix.
    b : np.ndarray
        Target vector or RHS block with shape ``(n_rows,)`` or ``(n_rows, n_rhs)``.
    coeffs_out : np.ndarray
        Output coefficient array (modified in place).
    dest_idxs : np.ndarray
        Destination indices for the solution.
    weights : np.ndarray, optional
        Weight array.
    reg_lambda : float, optional
        Regularization parameter.
    reg_L : np.ndarray or tuple
        Regularization penalty values.
    reg_idxs_source : np.ndarray or tuple
        Source indices for regularization.
    scale_A_forced : float, optional
        Forced scaling for A matrix.
    scale_L_forced : float, optional
        Forced scaling for L matrix.
    """
    if len(dest_idxs) == 0:
        return

    # Apply weights
    if weights is not None:
        A_w = A * weights[:, None]
        if np.asarray(b).ndim == 1:
            b_w = b * weights
        else:
            b_w = b * weights[:, None]
    else:
        A_w = A
        b_w = b

    # Add Regularization
    if reg_lambda is not None and reg_lambda > 0 and reg_L is not None:
        n_cols = A.shape[1]

        if isinstance(reg_L, tuple):
            pen_pol, pen_tor = reg_L
            idx_p, idx_t = reg_idxs_source
            idx_p = np.asarray(idx_p, dtype=int)
            idx_t = np.asarray(idx_t, dtype=int)

            L_vals = np.concatenate([pen_pol[idx_p % len(pen_pol)], pen_tor[idx_t % len(pen_tor)]])
        else:
            idx_source = reg_idxs_source
            L_vals = reg_L[idx_source]

        # Auto-Scaling Logic
        if scale_A_forced is not None and scale_L_forced is not None:
            scale_A = scale_A_forced
            scale_L = scale_L_forced
        else:
            norm_A = np.sum(A_w**2, axis=0)
            norm_L = L_vals**2

            valid_A = norm_A[norm_A > 1e-14]
            valid_L = norm_L[norm_L > 1e-14]

            scale_A = np.median(valid_A) if valid_A.size else 1.0
            scale_L = np.median(valid_L) if valid_L.size else 1.0

        factor = np.sqrt(scale_A / scale_L) if scale_L > 0 else 1.0
        effective_lam = np.sqrt(reg_lambda) * factor

        L_block = np.diag(effective_lam * L_vals)
        num_rhs = 1 if np.asarray(b_w).ndim == 1 else int(np.asarray(b_w).shape[1])
        zeros_rhs = np.zeros((n_cols, num_rhs), dtype=A_w.dtype)

        A_final = np.vstack([A_w, L_block])
        if num_rhs == 1:
            b_final = np.concatenate([np.asarray(b_w).reshape(-1), zeros_rhs.reshape(-1)])
        else:
            b_final = np.vstack([b_w, zeros_rhs])
    else:
        A_final = A_w
        b_final = b_w

    # Solve
    x, _, _, _ = np.linalg.lstsq(A_final, b_final, rcond=None)
    if coeffs_out.ndim == 2 and np.asarray(x).ndim == 1:
        x = np.asarray(x).reshape(-1, 1)
    coeffs_out[dest_idxs] = x


def compute_exact_weights(theta_1d: np.ndarray, L: int) -> np.ndarray:
    """Compute exact quadrature weights for a given 1D theta grid and bandlimit.

    Solves the moment equations: sum_t w_t P_l(theta_t) = 2 * delta_{l0}
    for l = 0 ... L-1.

    Parameters
    ----------
    theta_1d : np.ndarray
        Theta coordinates (colatitude) in radians.
    L : int
        Band limit (maximum degree P_l to match).

    Returns
    -------
    weights : np.ndarray
        Weights for each theta point.
    """
    from scipy.special import eval_legendre

    N_points = len(theta_1d)
    N_moments = N_points

    P_matrix = np.zeros((N_moments, N_points))
    for l in range(N_moments):
        P_matrix[l, :] = eval_legendre(l, np.cos(theta_1d))

    b = np.zeros(N_moments)
    b[0] = 2.0

    try:
        weights = np.linalg.solve(P_matrix, b)
    except np.linalg.LinAlgError:
        weights = np.ones(N_points) * 2.0 / N_points

    return weights


def get_mw_weights(L: int) -> np.ndarray:
    """Compute quadrature weights for McEwen-Wiaux (MW) sampling.

    Ref: McEwen & Wiaux (2011).

    Parameters
    ----------
    L : int
        Band limit. N_theta = L + 1.

    Returns
    -------
    weights : np.ndarray
        Weights for each theta_t (shape: L+1).
    """
    N = L + 1
    t = np.arange(N)
    theta_t = np.pi * (2 * t + 1) / (2 * N - 1)
    return compute_exact_weights(theta_t, N)
