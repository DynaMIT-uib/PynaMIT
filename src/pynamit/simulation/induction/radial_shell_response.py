"""Interfaces for explicit radial-shell toroidal response and trace models."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from pynamit.math.constants import mu0
from pynamit.primitives.basis import get_repo_cf_helmholtz_sign, is_sh_basis
from pynamit.simulation.induction.operator_utils import coerce_dense_operator_matrix
from pynamit.simulation.settings import RadialShellForcingMode
from pynamit.simulation.spatial.geometry_utils import to_dense
from pynamit.utils import asarray, to_numpy


def _get_factorized_toroidal_to_e_dense(state: Any, n: int) -> np.ndarray:
    """Return dense ``psi -> E`` using the explicit ``psi -> J_S -> E`` factorization."""
    psi_to_js_op = getattr(state, "toroidal_to_JS_coeffs", None)
    js_to_e_op = getattr(state, "JS_to_E_coeffs", None)
    if psi_to_js_op is not None and js_to_e_op is not None:
        psi_to_js = np.asarray(
            coerce_dense_operator_matrix(psi_to_js_op, n_component_rows=2, n_cols=n), dtype=float
        )
        js_to_e = np.asarray(
            coerce_dense_operator_matrix(js_to_e_op, n_component_rows=2, n_cols=2 * n),
            dtype=float,
        )
        return np.asarray(js_to_e @ psi_to_js, dtype=float)

    psi_to_e_op = getattr(state, "toroidal_to_E_coeffs", None)
    if psi_to_e_op is None:
        raise RuntimeError("State does not expose a toroidal shell-electric operator.")
    return np.asarray(
        coerce_dense_operator_matrix(psi_to_e_op, n_component_rows=2, n_cols=n), dtype=float
    )


def _get_toroidal_to_js_dense(state: Any, n: int) -> np.ndarray:
    """Return dense ``psi -> J_S`` in flattened vector-coefficient form."""
    psi_to_js_op = getattr(state, "toroidal_to_JS_coeffs", None)
    if psi_to_js_op is None:
        raise RuntimeError("State does not expose an explicit toroidal_to_JS_coeffs operator.")
    return np.asarray(
        coerce_dense_operator_matrix(psi_to_js_op, n_component_rows=2, n_cols=n), dtype=float
    )


def _get_js_to_e_dense(state: Any, n: int) -> np.ndarray:
    """Return dense ``J_S -> E`` in flattened vector-coefficient form."""
    js_to_e_op = getattr(state, "JS_to_E_coeffs", None)
    if js_to_e_op is None:
        raise RuntimeError("State does not expose an explicit JS_to_E_coeffs operator.")
    return np.asarray(
        coerce_dense_operator_matrix(js_to_e_op, n_component_rows=2, n_cols=2 * n),
        dtype=float,
    )


def _get_vector_divergence_from_js_coeff_dense(
    state: Any, n: int, coeff_basis: Any, js_n_rows: int
) -> np.ndarray:
    """Return dense ``div_Omega`` acting on flattened ``J_S`` coefficients.

    In pure spectral modes the live divergence operator is already exposed in
    coefficient space with shape ``(n, 2n)``. In transform-backed modes the
    live ``psi -> J_S`` map may already be grid-valued, in which case the
    matching divergence operator has shape ``(n, 2*n_grid)`` and should be
    used directly. If the live divergence is grid-valued but ``J_S`` is still
    coefficient-valued, we compose it with the active scalar basis evaluation
    map so the returned operator always acts on the same flattened ``J_S``
    representation as the live ``psi -> J_S`` operator.
    """
    solution_space = state.solution_space
    grid = getattr(state, "grid", None)
    if grid is None:
        geometry = getattr(state, "geometry", None)
        grid = getattr(geometry, "grid", None)
    if grid is None:
        poloidal = getattr(state, "poloidal_matrices", None)
        grid = getattr(poloidal, "grid", None)
    if grid is None:
        raise RuntimeError("Unable to resolve the live grid for the J_S divergence operator.")
    div_op = solution_space.get_vector_divergence_operator(grid)
    div_dense = np.asarray(coerce_dense_operator_matrix(div_op, n_cols=2 * n), dtype=float)
    g = np.asarray(to_dense(coeff_basis.get_evaluation_matrix(grid)), dtype=float)
    p = np.asarray(to_dense(coeff_basis.construct_scalar_projection_matrix(grid)), dtype=float)
    n_grid = int(g.shape[0])
    if div_dense.shape[1] == js_n_rows:
        return div_dense
    if div_dense.shape[1] == 2 * n and js_n_rows == 2 * n_grid:
        zeros = np.zeros_like(p)
        vec_proj = np.block([[p, zeros], [zeros, p]])
        return np.asarray(div_dense @ vec_proj, dtype=float)
    if div_dense.shape[1] == 2 * n_grid and js_n_rows == 2 * n:
        zeros = np.zeros_like(g)
        vec_eval = np.block([[g, zeros], [zeros, g]])
        return np.asarray(div_dense @ vec_eval, dtype=float)
    raise ValueError(
        "Vector divergence operator does not match the live J_S representation: "
        f"div={div_dense.shape}, js_rows={js_n_rows}, coeff_cols={2*n}, grid_cols={2*n_grid}."
    )


def _adapt_e_operator_to_js(state: Any, n: int, e_operator: np.ndarray) -> np.ndarray:
    """Compose a forcing-side ``E`` operator with the live ``J_S -> E`` map."""
    return np.asarray(np.asarray(e_operator, dtype=float) @ _get_js_to_e_dense(state, n), dtype=float)


def _adapt_source_from_js_to_live_representation(
    state: Any, n: int, coeff_basis: Any, source_from_js: np.ndarray, js_n_rows: int
) -> np.ndarray:
    """Adapt a ``J_S -> source`` operator to the live flattened ``J_S`` representation."""
    source_dense = np.asarray(source_from_js, dtype=float)
    if source_dense.shape[1] == js_n_rows:
        return source_dense

    solution_space = state.solution_space
    grid = getattr(state, "grid", None)
    if grid is None:
        geometry = getattr(state, "geometry", None)
        grid = getattr(geometry, "grid", None)
    if grid is None:
        poloidal = getattr(state, "poloidal_matrices", None)
        grid = getattr(poloidal, "grid", None)
    if grid is None:
        raise RuntimeError("Unable to resolve the live grid for the J_S source adaptation.")

    g = np.asarray(to_dense(coeff_basis.get_evaluation_matrix(grid)), dtype=float)
    p = np.asarray(to_dense(coeff_basis.construct_scalar_projection_matrix(grid)), dtype=float)
    n_grid = int(g.shape[0])
    coeff_js_cols = 2 * n
    grid_js_cols = 2 * n_grid
    if source_dense.shape[1] == coeff_js_cols and js_n_rows == grid_js_cols:
        zeros = np.zeros_like(p)
        vec_proj = np.block([[p, zeros], [zeros, p]])
        return np.asarray(source_dense @ vec_proj, dtype=float)
    if source_dense.shape[1] == grid_js_cols and js_n_rows == coeff_js_cols:
        zeros = np.zeros_like(g)
        vec_eval = np.block([[g, zeros], [zeros, g]])
        return np.asarray(source_dense @ vec_eval, dtype=float)
    raise ValueError(
        "J_S source operator does not match the live J_S representation: "
        f"source={source_dense.shape}, js_rows={js_n_rows}, coeff_cols={coeff_js_cols}, "
        f"grid_cols={grid_js_cols}."
    )


def _build_dense_pseudoinverse(a: np.ndarray, *, rtol: float) -> np.ndarray:
    """Return a robust dense pseudoinverse with an explicit relative cutoff."""
    a_np = np.asarray(a, dtype=float)
    if a_np.size == 0:
        return np.zeros((a_np.shape[1], a_np.shape[0]), dtype=float)
    if rtol <= 0.0:
        rtol = max(float(np.finfo(float).eps * max(a_np.shape)), 1e-15)
    return np.asarray(np.linalg.pinv(a_np, rcond=float(rtol)), dtype=float)


def _get_e_to_js_dense(state: Any, n: int) -> np.ndarray:
    """Return dense ``E -> J_S`` from the explicit shell constitutive operator.

    The runtime exposes the shell constitutive law as ``J_S -> E``. For the
    forcing-side radial-shell closure we need the inverse conductivity map

        ``E_known -> K_S,known``.

    This is built as the discrete pseudoinverse of the live ``J_S -> E``
    operator in the active coefficient space.
    """
    js_to_e_op = getattr(state, "JS_to_E_coeffs", None)
    if js_to_e_op is None:
        raise RuntimeError("State does not expose an explicit JS_to_E_coeffs operator.")
    js_to_e = np.asarray(
        coerce_dense_operator_matrix(js_to_e_op, n_component_rows=2, n_cols=2 * n),
        dtype=float,
    )
    rtol = float(max(getattr(state, "induction_null_svd_rtol", 0.0), 0.0))
    return _build_dense_pseudoinverse(js_to_e, rtol=rtol)


def _optional_dense_operator(
    op: Optional[np.ndarray], *, n_cols: int
) -> Optional[np.ndarray]:
    """Return ``None`` or a dense float array for an optional operator."""
    if op is None:
        return None
    return np.asarray(coerce_dense_operator_matrix(op, n_cols=n_cols), dtype=float)


def _assemble_dense_operator_from_column_solver(
    *,
    n_rows: int,
    n_cols: int,
    solve_column: Callable[[int], np.ndarray],
) -> np.ndarray:
    """Assemble a dense linear operator from basis-column solves."""
    dense = np.zeros((n_rows, n_cols), dtype=float)
    for i in range(n_cols):
        col = np.asarray(solve_column(i), dtype=float).reshape(-1)
        if col.shape != (n_rows,):
            raise ValueError(
                "Column solver returned wrong shape: "
                f"{col.shape}, expected ({n_rows},)."
            )
        dense[:, i] = col
    return dense


class ShellElectricTraceModel(ABC):
    """Abstract shell-electric trace model for radial-shell closure.

    A concrete model supplies the shell traces required by the radial balance

        ``E_coeffs -> (d_r U|_{R_I^+}, E_{r,I})``

    in the scalar coefficient space used by the live toroidal solve.
    """

    description: str = "shell-electric trace model"

    def bind_state(self, state: Any) -> None:
        """Bind a live ``State`` when the model needs runtime context."""
        return None

    def build_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping flattened ``E_coeffs`` to stacked traces.

        The returned operator has shape ``(2*N, 2*N)`` and maps

            ``[Phi, W] -> [d_r U|_{R_I^+}, E_{r,I}]``.
        """
        return None

    def compute_traces(
        self, toroidal_matrices: Any, E_coeffs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ``(d_r U|_{R_I^+}, E_{r,I})`` for one shell ``E`` state."""
        op = self.build_trace_operator(toroidal_matrices)
        if op is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement either build_trace_operator() "
                "or compute_traces()."
            )
        n = int(toroidal_matrices.basis.index_length)
        dense = coerce_dense_operator_matrix(op, n_cols=2 * n)
        traces = asarray(dense @ np.asarray(E_coeffs).reshape(-1)).reshape(2, n)
        return np.asarray(traces[0], dtype=float).reshape(-1), np.asarray(
            traces[1], dtype=float
        ).reshape(-1)


class PoloidalSideTraceModel(ABC):
    """Abstract immediate-above-shell side trace for the curl-free electric part.

    A concrete model supplies the one nonlocal side operator

        ``U|_{S_I} -> d_r U|_{R_I^+}``

    in the active scalar coefficient space used by the live toroidal solve.
    This is the smallest direct closure ingredient for

        ``q = d_r U|_{R_I^+} - E_{r,I}``.
    """

    description: str = "poloidal side-trace model"

    def bind_state(self, state: Any) -> None:
        """Bind a live ``State`` when the model needs runtime context."""
        return None

    def build_dudr_from_u_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping shell ``U`` coefficients to ``d_r U|_{R_I^+}``."""
        return None


class KnownSourceTraceModel(ABC):
    """Abstract model for the forcing-side radial-shell trace ``q = d_r U - E_r``.

    For the exact radial-shell forcing law with ideality adopted, the known
    upper-side shell-current source depends only on the single shell trace

        ``q_known = d_r U|_{R_I^+} - E_{r,I}``.

    A concrete model supplies this trace directly in the active scalar
    coefficient space:

        ``E_coeffs -> q_known``.
    """

    description: str = "known-source q-trace model"

    def bind_state(self, state: Any) -> None:
        """Bind a live ``State`` when the model needs runtime context."""
        return None

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping flattened ``E_coeffs`` to ``q_known`` coefficients."""
        return None

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping shell-current coefficients to ``q_known``."""
        return None


def _build_shell_helmholtz_channel_projector(n: int, *, channel: str) -> np.ndarray:
    """Return the coefficient-space projector onto one shell Helmholtz channel."""
    projector = np.zeros((2 * n, 2 * n), dtype=float)
    normalized = str(channel).strip().lower()
    if normalized == "cf":
        projector[:n, :n] = np.eye(n, dtype=float)
    elif normalized == "df":
        projector[n:, n:] = np.eye(n, dtype=float)
    else:
        raise ValueError(f"Invalid shell Helmholtz channel {channel!r}; expected 'cf' or 'df'.")
    return projector


@dataclass
class ColumnSolveKnownSourceTraceModel(KnownSourceTraceModel):
    """Assemble ``D_q`` explicitly by solving one shell basis vector at a time.

    This is the most direct numerical realization of the forcing-side
    Dirichlet-to-Neumann map described in the note: choose a shell basis,
    solve the adopted gap problem for each basis vector, extract

        ``q = d_r U - E_r``,

    and stack the returned columns into the dense operator

        ``D_q : E_shell -> q``.

    The class does not prescribe the gap physics itself. The caller supplies
    the per-column solver.
    """

    q_column_solver: Callable[[Any, np.ndarray], np.ndarray]
    q_from_js_column_solver: Optional[Callable[[Any, np.ndarray], np.ndarray]] = None
    description: str = "known-source q-trace assembled by column solves"
    _state: Any = field(default=None, init=False, repr=False, compare=False)
    _q_trace_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)
    _q_trace_from_js_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        self._state = state
        self._q_trace_cache = None
        self._q_trace_from_js_cache = None

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._q_trace_cache is not None:
            return self._q_trace_cache

        n = int(toroidal_matrices.basis.index_length)
        n_cols = 2 * n

        def solve_basis_column(i: int) -> np.ndarray:
            e_i = np.zeros(n_cols, dtype=float)
            e_i[i] = 1.0
            return self.q_column_solver(toroidal_matrices, e_i)

        self._q_trace_cache = _assemble_dense_operator_from_column_solver(
            n_rows=n,
            n_cols=n_cols,
            solve_column=solve_basis_column,
        )
        return self._q_trace_cache

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self.q_from_js_column_solver is None:
            return None
        if self._q_trace_from_js_cache is not None:
            return self._q_trace_from_js_cache

        n = int(toroidal_matrices.basis.index_length)
        n_cols = 2 * n

        def solve_basis_column(i: int) -> np.ndarray:
            js_i = np.zeros(n_cols, dtype=float)
            js_i[i] = 1.0
            return self.q_from_js_column_solver(toroidal_matrices, js_i)

        self._q_trace_from_js_cache = _assemble_dense_operator_from_column_solver(
            n_rows=n,
            n_cols=n_cols,
            solve_column=solve_basis_column,
        )
        return self._q_trace_from_js_cache


@dataclass(frozen=True)
class GapCoenergyBlocks:
    """Block data for the abstract condensed gap co-energy construction.

    The shell/gap quadratic form is written as

        [chi, x]^T [[K_cc, K_cx], [K_xc, K_xx]] [chi, x] / 2
        - [chi, x]^T [M_c, M_x] e

    where ``chi`` are shell connector coefficients, ``x`` are internal gap
    coefficients, and ``e`` are prescribed shell forcing coefficients.
    """

    K_chichi: np.ndarray
    K_chix: np.ndarray
    K_xx: np.ndarray
    M_chi: np.ndarray
    M_x: np.ndarray


@dataclass(frozen=True)
class RadialShellCondensedOperators:
    """Unified shell-level operator bundle for a radial-shell closure branch.

    The canonical shell representation is

        ``Lambda_gap(chi) = dt_jr^+``
        ``Gamma_known : input -> dt_jr,known^+``
        ``D_q : input -> q = R_I * dt_psi^+``

    with optional current-first and induced-response counterparts.
    """

    Lambda_gap: np.ndarray
    Gamma_known: Optional[np.ndarray] = None
    D_q: Optional[np.ndarray] = None
    Gamma_known_from_js: Optional[np.ndarray] = None
    D_q_from_js: Optional[np.ndarray] = None
    Gamma_induced_dtalpha: Optional[np.ndarray] = None
    D_q_induced_dtalpha: Optional[np.ndarray] = None


def build_gap_coenergy_blocks_from_condensed_operators(
    lambda_gap: np.ndarray, gamma_known: np.ndarray
) -> GapCoenergyBlocks:
    """Embed condensed shell operators as a zero-internal-variable block model.

    This realizes the shell-level quadratic form with no explicit internal gap
    coefficients ``x``. It is therefore a degenerate Schur-complement model:

        ``Lambda_gap = K_chichi``
        ``Gamma_known = M_chi``

    with ``K_chix = 0``, ``K_xx = 0``, and ``M_x = 0``.

    The construction is exact for the supplied shell operators, but it does
    not by itself provide an independent PFAC/RM derivation of those
    operators. It is best read as a shell-level co-energy realization of an
    already chosen forcing semantics.
    """

    lambda_np = np.asarray(lambda_gap, dtype=float)
    gamma_np = np.asarray(gamma_known, dtype=float)
    if lambda_np.ndim != 2 or gamma_np.ndim != 2:
        raise ValueError("Condensed gap operators must be rank-2 arrays.")
    if lambda_np.shape[0] != lambda_np.shape[1]:
        raise ValueError(
            f"Lambda_gap must be square, got shape {lambda_np.shape}."
        )
    if gamma_np.shape[0] != lambda_np.shape[0]:
        raise ValueError(
            "Gamma_known row dimension must match Lambda_gap. "
            f"Got {gamma_np.shape[0]} vs {lambda_np.shape[0]}."
        )
    n_shell = int(lambda_np.shape[0])
    n_forcing = int(gamma_np.shape[1])
    return GapCoenergyBlocks(
        K_chichi=lambda_np,
        K_chix=np.zeros((n_shell, 0), dtype=float),
        K_xx=np.zeros((0, 0), dtype=float),
        M_chi=gamma_np,
        M_x=np.zeros((0, n_forcing), dtype=float),
    )


def build_shell_pi_gap_coenergy_blocks_from_condensed_operators(
    lambda_gap: np.ndarray,
    gamma_known: np.ndarray,
    shell_pi_operator: np.ndarray,
    *,
    forcing_to_gap_operator: Optional[np.ndarray] = None,
) -> GapCoenergyBlocks:
    """Embed condensed shell operators using an explicit shell-``Pi`` gap state.

    This is a nondegenerate shell-gap realization of the same condensed shell
    closure. Let ``chi`` be the shell connector coefficients and ``x`` an
    internal gap variable identified with the runtime document-style shell
    correction ``Pi_shell``. The quadratic form is chosen so that stationarity
    enforces

        ``x = Pi_shell * chi + B_gap * e``

    while preserving the exact condensed shell operators

        ``Lambda_gap``
        ``Gamma_known``.

    With identity weighting on the internal gap state, one convenient choice is

        ``K_xx = I``
        ``K_xc = -Pi_shell``
        ``M_x = B_gap``

    and then

        ``K_cc = Lambda_gap + Pi_shell^T Pi_shell``
        ``M_c = Gamma_known - Pi_shell^T B_gap``.

    If ``forcing_to_gap_operator`` is omitted, the gap state is driven only
    through ``chi``. This is still a real internal-variable shell-gap model,
    but not yet an independent bulk PDE discretization.
    """

    lambda_np = np.asarray(lambda_gap, dtype=float)
    gamma_np = np.asarray(gamma_known, dtype=float)
    shell_pi_np = np.asarray(shell_pi_operator, dtype=float)

    if lambda_np.ndim != 2 or gamma_np.ndim != 2 or shell_pi_np.ndim != 2:
        raise ValueError("Shell-Pi gap operators must all be rank-2 arrays.")
    if lambda_np.shape[0] != lambda_np.shape[1]:
        raise ValueError(
            f"Lambda_gap must be square, got shape {lambda_np.shape}."
        )
    if gamma_np.shape[0] != lambda_np.shape[0]:
        raise ValueError(
            "Gamma_known row dimension must match Lambda_gap. "
            f"Got {gamma_np.shape[0]} vs {lambda_np.shape[0]}."
        )
    if shell_pi_np.shape != lambda_np.shape:
        raise ValueError(
            "shell_pi_operator must be square and act on the same shell space as Lambda_gap. "
            f"Got {shell_pi_np.shape} vs {lambda_np.shape}."
        )

    n_shell = int(lambda_np.shape[0])
    n_forcing = int(gamma_np.shape[1])
    if forcing_to_gap_operator is None:
        gap_forcing = np.zeros((n_shell, n_forcing), dtype=float)
    else:
        gap_forcing = np.asarray(forcing_to_gap_operator, dtype=float)
        if gap_forcing.shape != (n_shell, n_forcing):
            raise ValueError(
                "forcing_to_gap_operator must have shape "
                f"({n_shell}, {n_forcing}), got {gap_forcing.shape}."
            )

    k_xx = np.eye(n_shell, dtype=float)
    k_xc = -shell_pi_np
    k_cx = k_xc.T
    k_cc = np.asarray(lambda_np + (shell_pi_np.T @ shell_pi_np), dtype=float)
    m_x = np.asarray(gap_forcing, dtype=float)
    m_c = np.asarray(gamma_np - (shell_pi_np.T @ m_x), dtype=float)
    return GapCoenergyBlocks(
        K_chichi=k_cc,
        K_chix=k_cx,
        K_xx=k_xx,
        M_chi=m_c,
        M_x=m_x,
    )


def build_gap_coenergy_condensed_operators_from_blocks(
    blocks: GapCoenergyBlocks, *, rtol: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Lambda_gap, Gamma_known)`` from Schur-complement blocks."""

    K_cc = np.asarray(blocks.K_chichi, dtype=float)
    K_cx = np.asarray(blocks.K_chix, dtype=float)
    K_xx = np.asarray(blocks.K_xx, dtype=float)
    M_c = np.asarray(blocks.M_chi, dtype=float)
    M_x = np.asarray(blocks.M_x, dtype=float)

    if K_cc.ndim != 2 or K_cx.ndim != 2 or K_xx.ndim != 2 or M_c.ndim != 2 or M_x.ndim != 2:
        raise ValueError("Gap co-energy blocks must all be rank-2 arrays.")
    if K_cc.shape[0] != K_cc.shape[1]:
        raise ValueError(f"K_chichi must be square, got shape {K_cc.shape}.")
    if K_xx.shape[0] != K_xx.shape[1]:
        raise ValueError(f"K_xx must be square, got shape {K_xx.shape}.")
    if K_cx.shape[0] != K_cc.shape[0]:
        raise ValueError(
            "K_chix row dimension must match K_chichi. "
            f"Got {K_cx.shape[0]} vs {K_cc.shape[0]}."
        )
    if K_cx.shape[1] != K_xx.shape[0]:
        raise ValueError(
            "K_chix column dimension must match K_xx. "
            f"Got {K_cx.shape[1]} vs {K_xx.shape[0]}."
        )
    if M_c.shape[0] != K_cc.shape[0]:
        raise ValueError(
            "M_chi row dimension must match K_chichi. "
            f"Got {M_c.shape[0]} vs {K_cc.shape[0]}."
        )
    if M_x.shape[0] != K_xx.shape[0]:
        raise ValueError(
            "M_x row dimension must match K_xx. "
            f"Got {M_x.shape[0]} vs {K_xx.shape[0]}."
        )
    if M_c.shape[1] != M_x.shape[1]:
        raise ValueError(
            "M_chi and M_x must act on the same forcing space. "
            f"Got {M_c.shape[1]} vs {M_x.shape[1]}."
        )

    inv_Kxx = _build_dense_pseudoinverse(K_xx, rtol=rtol)
    lambda_gap = K_cc - (K_cx @ inv_Kxx @ K_cx.T)
    gamma_known = M_c - (K_cx @ inv_Kxx @ M_x)
    return np.asarray(lambda_gap, dtype=float), np.asarray(gamma_known, dtype=float)


@dataclass
class SchurComplementGapKnownSourceTraceModel(KnownSourceTraceModel):
    """Known-source ``q`` trace built from condensed gap co-energy blocks.

    This is the direct code counterpart of the abstract Schur-complement
    construction in the theory note. Once the block data are supplied, the
    condensed source operator is

        ``Gamma_known = M_chi - K_chix K_xx^{-1} M_x``,

    and the exact shell inversion gives

        ``D_q = R_I * jr_to_psi * Gamma_known``.

    This class does not derive the blocks from PFAC/RM by itself; it only
    turns an explicit block model into the forcing-side shell operator.
    """

    blocks_builder: Optional[Callable[[Any], GapCoenergyBlocks]] = None
    blocks: Optional[GapCoenergyBlocks] = None
    source_model: Optional[Any] = None
    schur_rtol: float = 0.0
    description: str = "known-source q-trace from Schur-complement gap co-energy blocks"
    _state: Any = field(default=None, init=False, repr=False, compare=False)
    _q_trace_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)
    _gamma_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)
    _lambda_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)

    def bind_state(self, state: Any) -> None:
        self._state = state
        self._q_trace_cache = None
        self._gamma_cache = None
        self._lambda_cache = None
        if self.source_model is not None and hasattr(self.source_model, "bind_state"):
            self.source_model.bind_state(state)

    def _resolve_blocks(self, toroidal_matrices: Any) -> GapCoenergyBlocks:
        if self.blocks is not None:
            return self.blocks
        if self.blocks_builder is not None:
            return self.blocks_builder(toroidal_matrices)
        raise RuntimeError(
            "SchurComplementGapKnownSourceTraceModel requires either static blocks "
            "or a blocks_builder(toroidal_matrices)."
        )

    def build_gamma_known_operator(self, toroidal_matrices: Any) -> np.ndarray:
        if self._gamma_cache is None:
            lambda_gap, gamma_known = build_gap_coenergy_condensed_operators_from_blocks(
                self._resolve_blocks(toroidal_matrices), rtol=float(self.schur_rtol)
            )
            self._lambda_cache = np.asarray(lambda_gap, dtype=float)
            self._gamma_cache = np.asarray(gamma_known, dtype=float)
        return self._gamma_cache

    def build_lambda_gap_operator(self, toroidal_matrices: Any) -> np.ndarray:
        if self._lambda_cache is None:
            self.build_gamma_known_operator(toroidal_matrices)
        return np.asarray(self._lambda_cache, dtype=float)

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._q_trace_cache is None:
            gamma_known = self.build_gamma_known_operator(toroidal_matrices)
            jr_to_psi = np.asarray(to_numpy(toroidal_matrices.jr_to_psi_coeff_operator), dtype=float)
            self._q_trace_cache = np.asarray(
                float(toroidal_matrices.RI) * (jr_to_psi @ gamma_known), dtype=float
            )
        return self._q_trace_cache

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self.source_model is not None and hasattr(self.source_model, "build_q_trace_from_js_operator"):
            return self.source_model.build_q_trace_from_js_operator(toroidal_matrices)
        return None


class ProjectedTangentialSecondTraceModel(ABC):
    """Abstract model for the omitted projected-tangential forcing traces.

    The reduced live tangential forcing keeps only the contribution from the
    radial curl scalar ``hat(r)·curl(E)``. The omitted complementary term in
    the exact projected Maxwell driver depends on the second radial shell
    traces

        ``q_r = d_r(d_r U - E_r)``
        ``p_r = d_r^2 V``.

    A concrete model supplies those traces in the active scalar coefficient
    space:

        ``E_coeffs -> [ q_r, p_r ]``.
    """

    description: str = "projected tangential second-trace model"

    def bind_state(self, state: Any) -> None:
        """Bind a live ``State`` when the model needs runtime context."""
        return None

    def build_second_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping flattened ``E_coeffs`` to stacked traces.

        The returned operator has shape ``(2*N, 2*N)`` and maps

            ``[Phi, W] -> [d_r(d_r U - E_r), d_r^2 V]``.
        """
        return None


def build_projected_tangential_omitted_rhs_operator(
    toroidal_matrices: Any, second_trace_model: ProjectedTangentialSecondTraceModel
) -> np.ndarray:
    """Build the omitted projected-tangential forcing block from second traces.

    If

        ``q_r = d_r(d_r U - E_r)``
        ``p_r = d_r^2 V``,

    then the omitted part of the exact projected Maxwell driver is

        ``f_om = B0S · grad_S(q_r) + B0S_perp · grad_S(p_r)``

    with ``B0S_perp = (-B0phi, B0theta)``. This routine assembles the
    corresponding coefficient-space operator from an explicit second-trace
    model.
    """

    op = second_trace_model.build_second_trace_operator(toroidal_matrices)
    if op is None:
        raise NotImplementedError(
            f"{second_trace_model.__class__.__name__} must implement "
            "build_second_trace_operator()."
        )

    n = int(toroidal_matrices.basis.index_length)
    dense_trace = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
    if dense_trace.shape != (2 * n, 2 * n):
        raise ValueError(
            "Projected tangential second-trace operator has wrong shape: "
            f"{dense_trace.shape}, expected ({2 * n}, {2 * n})."
        )

    q_r_op = dense_trace[:n]
    p_r_op = dense_trace[n:]
    P = np.asarray(to_dense(toroidal_matrices.projection_matrix), dtype=float)
    G_th = np.asarray(
        to_dense(toroidal_matrices.basis.get_evaluation_matrix(toroidal_matrices.grid, derivative="theta")),
        dtype=float,
    )
    G_ph = np.asarray(
        to_dense(toroidal_matrices.basis.get_evaluation_matrix(toroidal_matrices.grid, derivative="phi")),
        dtype=float,
    )
    _, B0th, B0ph, _ = toroidal_matrices._background_field_grid_components
    inv_R = 1.0 / float(toroidal_matrices.RI)
    omitted_grid = inv_R * (
        (B0th[:, None] * ((G_th @ q_r_op) + (G_ph @ p_r_op)))
        + (B0ph[:, None] * ((G_ph @ q_r_op) - (G_th @ p_r_op)))
    )
    return np.asarray(P @ omitted_grid, dtype=float)


def build_radial_shell_rhs_from_trace_operator(
    toroidal_matrices: Any, trace_model: ShellElectricTraceModel
) -> np.ndarray:
    """Build the exact radial-shell scalar from first shell electric traces.

    If a trace model supplies

        ``[Phi, W] -> [ d_r U|_{R_I^+}, E_{r,I} ]``,

    then the exact radial-shell scalar is

        ``-(Q_I - Delta_Omega E_r) / R_I^2``
        ``= Delta_S(E_r - d_r U)``

    because ``Q_I = Delta_Omega(d_r U)`` on the shell. This helper assembles
    the corresponding coefficient-space operator directly from those first
    shell traces.
    """

    op = trace_model.build_trace_operator(toroidal_matrices)
    if op is None:
        raise NotImplementedError(
            f"{trace_model.__class__.__name__} must implement build_trace_operator()."
        )

    n = int(toroidal_matrices.basis.index_length)
    dense_trace = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
    if dense_trace.shape != (2 * n, 2 * n):
        raise ValueError(
            "Shell electric trace operator has wrong shape: "
            f"{dense_trace.shape}, expected ({2 * n}, {2 * n})."
        )

    dudr_op = dense_trace[:n]
    er_op = dense_trace[n:]
    lap = np.asarray(
        coerce_dense_operator_matrix(
            toroidal_matrices.basis.get_laplacian_operator(r=toroidal_matrices.RI),
            n_cols=n,
        ),
        dtype=float,
    )
    return np.asarray(lap @ (er_op - dudr_op), dtype=float)


def build_known_source_operator_from_q_trace(
    toroidal_matrices: Any, q_trace_model: KnownSourceTraceModel
) -> np.ndarray:
    """Build the exact forcing-side shell-current source from the ``q`` trace.

    If a trace model supplies

        ``[Phi, W] -> q_known = d_r U|_{R_I^+} - E_{r,I}``,

    then ideality and the one-curl radial-shell identity give the exact
    upper-side shell-current source

        ``dt_jr,known^+ = -(1/mu0) * Delta_S(q_known)``.

    This helper assembles the corresponding coefficient-space operator
    directly.
    """

    op = q_trace_model.build_q_trace_operator(toroidal_matrices)
    if op is None:
        raise NotImplementedError(
            f"{q_trace_model.__class__.__name__} must implement build_q_trace_operator()."
        )

    n = int(toroidal_matrices.basis.index_length)
    dense_q = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
    if dense_q.shape != (n, 2 * n):
        raise ValueError(
            "Known-source q-trace operator has wrong shape: "
            f"{dense_q.shape}, expected ({n}, {2 * n})."
        )

    lap = np.asarray(
        coerce_dense_operator_matrix(
            toroidal_matrices.basis.get_laplacian_operator(r=toroidal_matrices.RI),
            n_cols=n,
        ),
        dtype=float,
    )
    return np.asarray((-(1.0 / float(mu0))) * (lap @ dense_q), dtype=float)


def build_radial_shell_rhs_from_q_trace_operator(
    toroidal_matrices: Any, q_trace_model: KnownSourceTraceModel
) -> np.ndarray:
    """Build the exact forcing-side radial-shell scalar from the ``q`` trace.

    This is just the source-level operator multiplied by ``mu0``:

        ``rhs_known = mu0 * dt_jr,known^+ = -Delta_S(q_known)``.
    """

    return np.asarray(
        float(mu0) * build_known_source_operator_from_q_trace(toroidal_matrices, q_trace_model),
        dtype=float,
    )


def build_q_trace_operator_from_known_source_model(
    toroidal_matrices: Any, known_source_model: "RadialShellResponseModel"
) -> np.ndarray:
    """Build the exact ``q = d_r U - E_r`` trace from a known-source model.

    Any forcing-side radial-shell model that already supplies the exact
    upper-side shell-current source

        ``E_coeffs -> dt_jr,known^+``

    also determines the mean-zero one-curl shell trace through

        ``dt_psi^+ = jr_to_psi * dt_jr^+``
        ``q = R_I * dt_psi^+``.

    This helper inverts the exact shell relation and returns the corresponding
    coefficient-space ``E_coeffs -> q`` operator.
    """

    n = int(toroidal_matrices.basis.index_length)
    known_source_op = known_source_model.build_known_source_operator(toroidal_matrices)
    if known_source_op is None:
        raise NotImplementedError(
            f"{known_source_model.__class__.__name__} must implement "
            "build_known_source_operator() to support the q-trace adapter."
        )

    dense_dtjr = np.asarray(
        coerce_dense_operator_matrix(known_source_op, n_cols=2 * n),
        dtype=float,
    )
    jr_to_psi = np.asarray(to_numpy(toroidal_matrices.jr_to_psi_coeff_operator), dtype=float)
    return np.asarray(float(toroidal_matrices.RI) * (jr_to_psi @ dense_dtjr), dtype=float)


def build_q_trace_operator_from_exterior_update_model(
    toroidal_matrices: Any, exterior_update_model: ExteriorToroidalUpdateModel
) -> np.ndarray:
    """Build the exact ``q = d_r U - E_r`` trace from an upper-side update model.

    For the one-curl connector relation,

        ``R_I * dt_psi^+ = Pi0(q)``,

    any concrete known-forcing exterior update model that supplies either
    ``dt_psi^+`` or ``dt_jr^+`` determines the mean-zero shell trace ``q``
    exactly. This helper converts that update into a coefficient-space
    ``E_coeffs -> q`` operator.
    """

    n = int(toroidal_matrices.basis.index_length)
    dtpsi_op = exterior_update_model.build_dtpsi_operator(toroidal_matrices)
    if dtpsi_op is not None:
        dense_dtpsi = np.asarray(coerce_dense_operator_matrix(dtpsi_op, n_cols=2 * n), dtype=float)
        return np.asarray(float(toroidal_matrices.RI) * dense_dtpsi, dtype=float)

    dtjr_op = exterior_update_model.build_dtjr_operator(toroidal_matrices)
    if dtjr_op is not None:
        dense_dtjr = np.asarray(coerce_dense_operator_matrix(dtjr_op, n_cols=2 * n), dtype=float)
        jr_to_psi = np.asarray(to_numpy(toroidal_matrices.jr_to_psi_coeff_operator), dtype=float)
        return np.asarray(float(toroidal_matrices.RI) * (jr_to_psi @ dense_dtjr), dtype=float)

    raise NotImplementedError(
        f"{exterior_update_model.__class__.__name__} must implement either "
        "build_dtpsi_operator() or build_dtjr_operator() to support the q-trace adapter."
    )


def _get_closure_basis(toroidal_matrices: Any) -> Any:
    """Return the scalar closure basis used for harmonic side continuation."""
    if getattr(toroidal_matrices, "_use_auxiliary_closure_basis", False):
        return toroidal_matrices._toroidal_closure_projector.closure_basis
    return toroidal_matrices.basis


def _get_state_to_closure_scalar_map(toroidal_matrices: Any) -> np.ndarray:
    """Return dense map from state scalar coefficients to closure-basis coefficients."""
    if getattr(toroidal_matrices, "_use_auxiliary_closure_basis", False):
        return np.asarray(toroidal_matrices._state_to_aux_scalar_map, dtype=float)
    n = int(toroidal_matrices.basis.index_length)
    return np.eye(n, dtype=float)


def _get_closure_to_state_scalar_map(toroidal_matrices: Any) -> np.ndarray:
    """Return dense map from closure-basis scalar coefficients back to state space."""
    if getattr(toroidal_matrices, "_use_auxiliary_closure_basis", False):
        return np.asarray(toroidal_matrices._aux_to_state_scalar_map, dtype=float)
    n = int(toroidal_matrices.basis.index_length)
    return np.eye(n, dtype=float)


def _build_harmonic_dudr_from_u_state_operator(
    toroidal_matrices: Any, *, outer_boundary_mode: str
) -> np.ndarray:
    """Return the harmonic side operator ``U|_S -> d_r U|_{R_I^+}`` in state space."""
    basis = _get_closure_basis(toroidal_matrices)
    if not is_sh_basis(basis):
        raise RuntimeError(
            "Harmonic side-trace operators require an SH closure basis "
            "to define the radial continuation cleanly."
        )

    n = np.asarray(basis.scalar_degrees(), dtype=float).reshape(-1)
    RI = float(toroidal_matrices.RI)
    mode = str(outer_boundary_mode).lower()

    if mode == "open":
        factors = n / RI
    elif mode in {"shielded", "closed"}:
        RM = getattr(toroidal_matrices, "RM", None)
        if RM in (None, 0):
            raise ValueError(
                "A shielded harmonic side trace requires a finite RM."
            )
        q = float(RM) / RI
        q_pow = np.power(q, 2.0 * n + 1.0)
        denom = 1.0 - q_pow
        factors = (n + ((n + 1.0) * q_pow)) / (RI * denom)
    else:
        raise ValueError(
            "Invalid outer_boundary_mode for harmonic side trace: "
            f"{outer_boundary_mode!r}."
        )

    s2c = _get_state_to_closure_scalar_map(toroidal_matrices)
    c2s = _get_closure_to_state_scalar_map(toroidal_matrices)
    return np.asarray(c2s @ np.diag(factors) @ s2c, dtype=float)


def _build_u_from_shell_e_operator(toroidal_matrices: Any) -> np.ndarray:
    """Return dense map from shell coefficients ``[Phi, W]`` to physical ``U``.

    The live shell-electric Helmholtz coefficients follow the repo-wide vector
    sign convention

        ``E_S = cf_sign * grad_S(Phi) + df_sign * (rhat x grad_S(W))``.

    The reduced-shell theory uses the physical curl-free scalar ``U`` defined
    by

        ``E_S = grad_S(U) - rhat x grad_S(V)``

    with ``V = W``. Therefore

        ``U = cf_sign * Phi``.

    With the current repo setting ``cf_sign = -1``, this is the same sign as
    the legacy electrostatic potential convention: ``U = -Phi``.
    """
    n = int(toroidal_matrices.basis.index_length)
    op = np.zeros((n, 2 * n), dtype=float)
    op[:, :n] = float(get_repo_cf_helmholtz_sign()) * np.eye(n, dtype=float)
    return op


def _build_ideal_er_from_shell_e_operator(toroidal_matrices: Any) -> np.ndarray:
    """Return dense map recovering ``E_r`` from shell ``E_S`` via ideality."""
    V_th = np.asarray(toroidal_matrices._vector_basis_component_maps[0], dtype=float)
    V_ph = np.asarray(toroidal_matrices._vector_basis_component_maps[1], dtype=float)
    B0r, B0th, B0ph, _ = toroidal_matrices._background_field_grid_components
    inv_B0r = np.asarray(to_numpy(toroidal_matrices.inverse_radial_field), dtype=float).reshape(-1)
    er_grid_op = -(inv_B0r[:, None]) * (
        (np.asarray(B0th, dtype=float)[:, None] * V_th)
        + (np.asarray(B0ph, dtype=float)[:, None] * V_ph)
    )
    P_state = np.asarray(to_numpy(toroidal_matrices.projection_matrix), dtype=float)
    return np.asarray(P_state @ er_grid_op, dtype=float)


def build_q_trace_operator_from_poloidal_side_trace(
    toroidal_matrices: Any, side_trace_model: PoloidalSideTraceModel
) -> np.ndarray:
    """Build the connector trace ``q = d_r U - E_r`` from a side operator.

    Here ``U`` is the physical curl-free shell scalar recovered through
    ``U = cf_sign * Phi``. The resulting ``q`` therefore follows the active
    repo curl-free Helmholtz sign automatically and remains aligned with the
    legacy electrostatic sign when ``cf_sign = -1``.
    """
    dudr_from_u = side_trace_model.build_dudr_from_u_operator(toroidal_matrices)
    if dudr_from_u is None:
        raise NotImplementedError(
            f"{side_trace_model.__class__.__name__} must implement "
            "build_dudr_from_u_operator()."
        )

    n = int(toroidal_matrices.basis.index_length)
    dense_dudr = np.asarray(coerce_dense_operator_matrix(dudr_from_u, n_cols=n), dtype=float)
    if dense_dudr.shape != (n, n):
        raise ValueError(
            "Poloidal side-trace operator has wrong shape: "
            f"{dense_dudr.shape}, expected ({n}, {n})."
        )

    u_from_e = _build_u_from_shell_e_operator(toroidal_matrices)
    er_from_e = _build_ideal_er_from_shell_e_operator(toroidal_matrices)
    return np.asarray(dense_dudr @ u_from_e - er_from_e, dtype=float)


class ExteriorToroidalUpdateModel(ABC):
    """Abstract model for the missing upper-side toroidal update scalar.

    A concrete model supplies one of the equivalent upper-shell quantities

        ``E_coeffs -> dt_jr^+``
        ``E_coeffs -> dt_psi^+``

    in the scalar coefficient space used by the live toroidal solve.
    """

    description: str = "upper-side toroidal update model"

    def bind_state(self, state: Any) -> None:
        """Bind a live ``State`` when the model needs runtime context."""
        return None

    def build_dtjr_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping flattened ``E_coeffs`` to ``dt_jr^+`` coefficients."""
        return None

    def build_dtpsi_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping flattened ``E_coeffs`` to ``dt_psi^+`` coefficients."""
        return None

    def build_dtjr_from_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping ``dt_alpha`` to ``dt_jr^+`` coefficients."""
        return None

    def build_dtpsi_from_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping ``dt_alpha`` to ``dt_psi^+`` coefficients."""
        return None


@dataclass(frozen=True)
class HarmonicPoloidalSideTraceModel(PoloidalSideTraceModel):
    """Harmonic immediate-above-shell side operator for the curl-free electric scalar.

    This is the minimal direct closure object

        ``Lambda_U : U|_{S_I} -> d_r U|_{R_I^+}``

    implemented with the same open/shielded harmonic continuation already used
    by the older explicit shell-trace model.
    """

    outer_boundary_mode: str = "open"
    description: str = "harmonic poloidal side operator U -> d_r U"

    def build_dudr_from_u_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return _build_harmonic_dudr_from_u_state_operator(
            toroidal_matrices, outer_boundary_mode=self.outer_boundary_mode
        )


@dataclass(frozen=True)
class HarmonicShellElectricTraceModel(ShellElectricTraceModel):
    """Harmonic SH shell-trace model for the radial-shell closure.

    Assumptions:
    - the shell tangential electric coefficients are the upper-side shell
      traces ``[Phi, W]`` in the repo Helmholtz representation,
    - the physical poloidal electric scalar is ``U = cf_sign * Phi``,
    - the curl-free electric part above the shell follows a harmonic radial
      continuation in the auxiliary SH closure basis,
    - only that poloidal electric part is continued to ``d_r U``; the
      toroidal shell electric channel ``W`` does not enter the ``d_r U`` block,
    - the shell-normal electric trace is recovered from the full shell
      tangential field by the ideal-gap relation ``E · B0 = 0``.

    This is an explicit upper-region model, but still a model: the radial
    continuation and the ideal-gap rule are chosen assumptions.
    """

    outer_boundary_mode: str = "open"
    description: str = (
        "harmonic SH shell-electric trace model with ideal-gap Er recovery"
    )

    def build_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        n_state = int(toroidal_matrices.basis.index_length)
        side_trace_model = HarmonicPoloidalSideTraceModel(
            outer_boundary_mode=self.outer_boundary_mode
        )
        dudr_from_u = np.asarray(
            side_trace_model.build_dudr_from_u_operator(toroidal_matrices), dtype=float
        )
        u_from_e = _build_u_from_shell_e_operator(toroidal_matrices)
        dudr_full = np.zeros((n_state, 2 * n_state), dtype=float)
        dudr_full[:, :] = dudr_from_u @ u_from_e
        er_full = _build_ideal_er_from_shell_e_operator(toroidal_matrices)

        return np.asarray(np.vstack([dudr_full, er_full]), dtype=float)

class RadialShellResponseModel(ABC):
    """Abstract radial-shell response model for toroidal full induction.

    A concrete model must supply an equivalent shell scalar response operator

        ``E_known -> -(Q_I - Delta_Omega E_r,I) / R_I^2``

    in the coefficient space used by the live toroidal solve.

    In the shell-source language used by the radial-shell note, this scalar is
    simply

        ``mu0 * dt_jr^+``.

    So every forcing-side radial-shell model may also be read as a known-source
    operator

        ``E_known -> dt_jr,known^+``.
    """

    description: str = "radial-shell response model"

    def bind_state(self, state: Any) -> None:
        """Bind a live ``State`` when the model needs runtime context."""
        return None

    def build_rhs_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping flattened ``E_coeffs`` to RHS coefficients.

        Returning ``None`` means the model only supports direct evaluation via
        :meth:`compute_rhs`.
        """
        return None

    def build_feedback_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping ``dt_alpha`` to induced radial-shell feedback.

        This represents the induced-response contribution in the closed radial
        shell equation

            ``mu0 * B0r * dt_alpha = rhs_driver + feedback(dt_alpha)``.

        Returning ``None`` means the model only supplies a direct forcing-side
        shell scalar response from known ``E_coeffs``.
        """
        return None

    def build_known_source_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping ``E_coeffs`` to ``dt_jr,known^+``.

        By default this is the source-level form of :meth:`build_rhs_operator`,
        using the exact radial-shell identity

            ``rhs = mu0 * dt_jr,known^+``.

        Concrete models may override this when they naturally assemble the
        shell-current source directly.
        """
        op = self.build_rhs_operator(toroidal_matrices)
        if op is None:
            return None
        n = int(toroidal_matrices.basis.index_length)
        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
        return np.asarray((1.0 / float(mu0)) * dense, dtype=float)

    def build_gamma_known_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return the condensed forcing-side shell source ``Gamma_known``.

        In the shell-source formulation of the radial-shell closure this is
        just the known-source operator

            ``Gamma_known : E_known -> dt_jr,known^+``.
        """
        return self.build_known_source_operator(toroidal_matrices)

    def build_lambda_gap_operator(self, toroidal_matrices: Any) -> np.ndarray:
        """Return the exact shell operator on ``chi = dt_psi^+``.

        Regardless of how the forcing-side source is built, the final shell
        inversion is the exact radial-shell relation

            ``-(R_I / mu0) * Delta_S(chi) = dt_jr^+``.

        This routine returns that shell operator in the active scalar
        coefficient space.
        """
        n = int(toroidal_matrices.basis.index_length)
        lap = np.asarray(
            coerce_dense_operator_matrix(
                toroidal_matrices.basis.get_laplacian_operator(r=toroidal_matrices.RI),
                n_cols=n,
            ),
            dtype=float,
        )
        return np.asarray((-(float(toroidal_matrices.RI) / float(mu0))) * lap, dtype=float)

    def build_gap_coenergy_blocks(
        self, toroidal_matrices: Any
    ) -> Optional[GapCoenergyBlocks]:
        """Return a shell-level co-energy realization of the current model.

        By default, any model that already supplies the condensed forcing-side
        operator ``Gamma_known`` can be represented as a zero-internal-variable
        block model using the exact shell operator on ``chi``. This is a
        unifying shell-level representation of the current semantics, not a
        claim that the model has been independently derived from explicit gap
        co-energy blocks.
        """
        gamma_known = self.build_gamma_known_operator(toroidal_matrices)
        if gamma_known is None:
            return None
        lambda_gap = self.build_lambda_gap_operator(toroidal_matrices)
        return build_gap_coenergy_blocks_from_condensed_operators(
            lambda_gap=lambda_gap,
            gamma_known=np.asarray(gamma_known, dtype=float),
        )

    def build_condensed_operators(
        self, toroidal_matrices: Any
    ) -> RadialShellCondensedOperators:
        """Return the unified shell-level operators exposed by this branch."""
        gamma_known = self.build_gamma_known_operator(toroidal_matrices)
        d_q = self.build_q_trace_operator(toroidal_matrices)
        gamma_known_from_js = self.build_known_source_from_js_operator(toroidal_matrices)
        d_q_from_js = self.build_q_trace_from_js_operator(toroidal_matrices)
        gamma_induced_dtalpha = self.build_induced_source_dtalpha_operator(toroidal_matrices)
        d_q_induced_dtalpha = self.build_induced_q_trace_from_dtalpha_operator(
            toroidal_matrices
        )
        return RadialShellCondensedOperators(
            Lambda_gap=np.asarray(self.build_lambda_gap_operator(toroidal_matrices), dtype=float),
            Gamma_known=None if gamma_known is None else np.asarray(gamma_known, dtype=float),
            D_q=None if d_q is None else np.asarray(d_q, dtype=float),
            Gamma_known_from_js=(
                None if gamma_known_from_js is None else np.asarray(gamma_known_from_js, dtype=float)
            ),
            D_q_from_js=None if d_q_from_js is None else np.asarray(d_q_from_js, dtype=float),
            Gamma_induced_dtalpha=(
                None
                if gamma_induced_dtalpha is None
                else np.asarray(gamma_induced_dtalpha, dtype=float)
            ),
            D_q_induced_dtalpha=(
                None if d_q_induced_dtalpha is None else np.asarray(d_q_induced_dtalpha, dtype=float)
            ),
        )

    def build_known_source_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping shell-current coefficients to ``dt_jr,known^+``.

        This optional current-first interface exposes the primitive shell source
        law when a forcing model is most naturally written as

            ``dtK_S,known -> dt_jr,known^+``.

        Models that only act on shell-electric coefficients may leave this as
        ``None``.
        """
        return None

    def build_rhs_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping shell-current coefficients to RHS coefficients."""
        op = self.build_known_source_from_js_operator(toroidal_matrices)
        if op is None:
            return None
        n = int(toroidal_matrices.basis.index_length)
        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
        return np.asarray(float(mu0) * dense, dtype=float)

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping ``E_coeffs`` to ``q = d_r U - E_r``.

        With ideality adopted, the forcing-side exact radial-shell trace is

            ``q = d_r U|_{R_I^+} - E_{r,I}``

        and the exact shell-current source satisfies

            ``dt_jr,known^+ = -(1/mu0) * Delta_S(q)``.

        So any concrete model that already supplies ``dt_jr,known^+`` also
        determines ``q`` uniquely in the mean-zero shell gauge. By default we
        therefore invert the exact shell relation and return

            ``q = R_I * jr_to_psi * dt_jr,known^+``.

        Concrete models may override this when they naturally assemble the
        one-curl trace directly.
        """
        op = self.build_known_source_operator(toroidal_matrices)
        if op is None:
            return None
        return build_q_trace_operator_from_known_source_model(toroidal_matrices, self)

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping shell-current coefficients to ``q = d_r U - E_r``."""
        op = self.build_known_source_from_js_operator(toroidal_matrices)
        if op is None:
            return None
        n = int(toroidal_matrices.basis.index_length)
        dense_source = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
        jr_to_psi = np.asarray(to_numpy(toroidal_matrices.jr_to_psi_coeff_operator), dtype=float)
        return np.asarray(float(toroidal_matrices.RI) * (jr_to_psi @ dense_source), dtype=float)

    def build_induced_source_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        """Return dense matrix mapping ``dt_alpha`` to ``dt_jr,induced^+``.

        By default this is the source-level form of
        :meth:`build_feedback_dtalpha_operator`, using

            ``feedback = mu0 * dt_jr,induced^+``.

        Concrete models may override this when they naturally assemble the
        upper-side shell-current source directly.
        """
        op = self.build_feedback_dtalpha_operator(toroidal_matrices)
        if op is None:
            return None
        n = int(toroidal_matrices.basis.index_length)
        dense = np.asarray(coerce_dense_operator_matrix(op, n_cols=n), dtype=float)
        return np.asarray((1.0 / float(mu0)) * dense, dtype=float)

    def build_induced_q_trace_from_dtalpha_operator(
        self, toroidal_matrices: Any
    ) -> Optional[np.ndarray]:
        """Return dense matrix mapping ``dt_alpha`` to induced ``q = d_r U - E_r``.

        Once the induced upper-side shell-current source is known, the exact
        one-curl shell inversion determines the mean-zero induced trace through

            ``q_induced = R_I * jr_to_psi * dt_jr,induced^+``.

        Concrete models may override this when they naturally assemble the
        induced q-trace directly.
        """
        op = self.build_induced_source_dtalpha_operator(toroidal_matrices)
        if op is None:
            return None
        n = int(toroidal_matrices.basis.index_length)
        dense_source = np.asarray(coerce_dense_operator_matrix(op, n_cols=n), dtype=float)
        jr_to_psi = np.asarray(to_numpy(toroidal_matrices.jr_to_psi_coeff_operator), dtype=float)
        return np.asarray(float(toroidal_matrices.RI) * (jr_to_psi @ dense_source), dtype=float)

    def compute_rhs(self, toroidal_matrices: Any, E_coeffs: np.ndarray) -> np.ndarray:
        """Evaluate the radial-shell RHS for one tangential shell ``E`` state."""
        op = self.build_rhs_operator(toroidal_matrices)
        if op is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement either build_rhs_operator() "
                "or compute_rhs()."
            )
        n = int(toroidal_matrices.basis.index_length)
        dense = coerce_dense_operator_matrix(op, n_cols=2 * n)
        return asarray(dense @ np.asarray(E_coeffs).reshape(-1)).reshape(-1)


@dataclass
class KnownSourceOperatorKnownSourceTraceModel(KnownSourceTraceModel):
    """Adapter exposing ``q = d_r U - E_r`` from an exact known-source model.

    If a forcing-side model already returns the exact upper-side shell-current
    source

        ``E_known -> dt_jr,known^+``,

    then the one-curl shell identity determines the mean-zero trace

        ``q = R_I * dt_psi^+ = R_I * jr_to_psi * dt_jr,known^+``.

    This adapter makes that exact inversion explicit, so shell-law and
    reduced-model forcing branches can both be represented in the same
    ``KnownSourceTraceModel`` interface.
    """

    known_source_model: Any
    description: str = "known-source q-trace from exact shell-current source"

    def bind_state(self, state: Any) -> None:
        if hasattr(self.known_source_model, "bind_state"):
            self.known_source_model.bind_state(state)

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return build_q_trace_operator_from_known_source_model(
            toroidal_matrices, self.known_source_model
        )

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if hasattr(self.known_source_model, "build_q_trace_from_js_operator"):
            return self.known_source_model.build_q_trace_from_js_operator(toroidal_matrices)
        return None


@dataclass
class PoloidalSideTraceKnownSourceTraceModel(KnownSourceTraceModel):
    """Direct forcing-side ``q`` trace from ``U|_S -> d_r U|_{R_I^+}``.

    This is the minimal direct closure proposed in the note:

        ``q = Lambda_U U - E_r``

    where ``Lambda_U`` is a side-trace operator on the curl-free shell electric
    scalar and ``E_r`` is recovered from shell ideality.
    """

    poloidal_side_trace_model: PoloidalSideTraceModel
    description: str = "known-source q-trace from direct poloidal side operator"
    _state: Any = field(default=None, init=False, repr=False, compare=False)

    def bind_state(self, state: Any) -> None:
        self._state = state
        if hasattr(self.poloidal_side_trace_model, "bind_state"):
            self.poloidal_side_trace_model.bind_state(state)

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return build_q_trace_operator_from_poloidal_side_trace(
            toroidal_matrices, self.poloidal_side_trace_model
        )

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._state is None:
            return None
        n = int(toroidal_matrices.basis.index_length)
        return _adapt_e_operator_to_js(
            self._state,
            n,
            np.asarray(self.build_q_trace_operator(toroidal_matrices), dtype=float),
        )


@dataclass
class HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel(KnownSourceTraceModel):
    """Column-assembled ``D_q^{iono}`` with homogeneous outer magnetic data.

    This is the narrow constructive target discussed in the note: keep the
    outer magnetic boundary homogeneous and assemble only the pure
    ionosphere-to-connector map

        ``D_q^{iono} : E_{S,I} -> q_I``.

    The present implementation keeps the physics baseline deliberately modest:
    each column solve is evaluated against the current harmonic side-trace
    closure for the chosen outer semantics. In other words, this is a real
    column-assembled operator with the correct interface, but not yet a full
    independent gap discretization.
    """

    outer_boundary_mode: str = "shielded"
    description: str = (
        "column-solve D_q^iono with homogeneous outer magnetic boundary"
    )
    _state: Any = field(default=None, init=False, repr=False, compare=False)
    _direct_q_model: Optional[PoloidalSideTraceKnownSourceTraceModel] = field(
        default=None, init=False, repr=False, compare=False
    )
    _delegate: Optional[ColumnSolveKnownSourceTraceModel] = field(
        default=None, init=False, repr=False, compare=False
    )
    _dense_q_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _dense_q_from_js_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        self._state = state
        self._dense_q_cache = None
        self._dense_q_from_js_cache = None

        direct_q_model = PoloidalSideTraceKnownSourceTraceModel(
            HarmonicPoloidalSideTraceModel(outer_boundary_mode=self.outer_boundary_mode)
        )
        direct_q_model.bind_state(state)
        self._direct_q_model = direct_q_model

        delegate = ColumnSolveKnownSourceTraceModel(
            q_column_solver=self._solve_e_column,
            q_from_js_column_solver=self._solve_js_column,
            description=self.description,
        )
        delegate.bind_state(state)
        self._delegate = delegate

    def _require_direct_model(self) -> PoloidalSideTraceKnownSourceTraceModel:
        if self._direct_q_model is None:
            raise RuntimeError(
                "HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel "
                "requires a bound State. Pass the model into Dynamics(...) or "
                "run_pynamit(...), so State can bind it."
            )
        return self._direct_q_model

    def _require_delegate(self) -> ColumnSolveKnownSourceTraceModel:
        if self._delegate is None:
            raise RuntimeError(
                "HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel "
                "requires a bound State. Pass the model into Dynamics(...) or "
                "run_pynamit(...), so State can bind it."
            )
        return self._delegate

    def _get_dense_q_operator(self, toroidal_matrices: Any) -> np.ndarray:
        if self._dense_q_cache is None:
            self._dense_q_cache = np.asarray(
                self._require_direct_model().build_q_trace_operator(toroidal_matrices),
                dtype=float,
            )
        return self._dense_q_cache

    def _get_dense_q_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._dense_q_from_js_cache is None:
            op = self._require_direct_model().build_q_trace_from_js_operator(toroidal_matrices)
            if op is None:
                return None
            self._dense_q_from_js_cache = np.asarray(op, dtype=float)
        return self._dense_q_from_js_cache

    def _solve_e_column(self, toroidal_matrices: Any, e: np.ndarray) -> np.ndarray:
        dense_q = self._get_dense_q_operator(toroidal_matrices)
        return np.asarray(dense_q @ np.asarray(e, dtype=float).reshape(-1), dtype=float)

    def _solve_js_column(self, toroidal_matrices: Any, js: np.ndarray) -> np.ndarray:
        dense_q_from_js = self._get_dense_q_from_js_operator(toroidal_matrices)
        if dense_q_from_js is None:
            raise RuntimeError(
                "No shell-current q-trace operator is available for the current bound state."
            )
        return np.asarray(dense_q_from_js @ np.asarray(js, dtype=float).reshape(-1), dtype=float)

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self._require_delegate().build_q_trace_operator(toroidal_matrices)

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self._require_delegate().build_q_trace_from_js_operator(toroidal_matrices)


@dataclass
class FilteredKnownSourceTraceModel(KnownSourceTraceModel):
    """Known-source q-trace wrapper with a precomposed shell-input filter.

    This is useful for experiments where the forcing-side trace is built only
    from one shell Helmholtz channel, e.g. a df/inductive particular part or a
    cf/curl-free homogeneous remainder. The filter can be applied either in the
    shell-electric coefficient space or in the shell-current coefficient space.
    """

    base_q_trace_model: KnownSourceTraceModel
    shell_channel: str
    input_space: str = "e"
    description: str = "filtered known-source q-trace model"
    _state: Any = field(default=None, init=False, repr=False, compare=False)

    def _build_input_filter(self, n: int) -> np.ndarray:
        return _build_shell_helmholtz_channel_projector(n, channel=self.shell_channel)

    def _normalized_input_space(self) -> str:
        normalized = str(self.input_space).strip().lower()
        if normalized not in {"e", "js"}:
            raise ValueError(
                f"Invalid filtered trace input_space {self.input_space!r}; expected 'e' or 'js'."
            )
        return normalized

    def bind_state(self, state: Any) -> None:
        self._state = state
        if hasattr(self.base_q_trace_model, "bind_state"):
            self.base_q_trace_model.bind_state(state)

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        n = int(toroidal_matrices.basis.index_length)
        filt = self._build_input_filter(n)
        input_space = self._normalized_input_space()

        if input_space == "e":
            base = self.base_q_trace_model.build_q_trace_operator(toroidal_matrices)
            if base is None:
                return None
            dense_base = np.asarray(coerce_dense_operator_matrix(base, n_cols=2 * n), dtype=float)
            return np.asarray(dense_base @ filt, dtype=float)

        if self._state is None:
            return None
        base_from_js = self.base_q_trace_model.build_q_trace_from_js_operator(toroidal_matrices)
        if base_from_js is None:
            return None
        dense_base_from_js = np.asarray(
            coerce_dense_operator_matrix(base_from_js, n_cols=2 * n), dtype=float
        )
        e_to_js = _get_e_to_js_dense(self._state, n)
        return np.asarray(dense_base_from_js @ filt @ e_to_js, dtype=float)

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        n = int(toroidal_matrices.basis.index_length)
        filt = self._build_input_filter(n)
        input_space = self._normalized_input_space()

        if input_space == "js":
            base_from_js = self.base_q_trace_model.build_q_trace_from_js_operator(toroidal_matrices)
            if base_from_js is None:
                return None
            dense_base_from_js = np.asarray(
                coerce_dense_operator_matrix(base_from_js, n_cols=2 * n), dtype=float
            )
            return np.asarray(dense_base_from_js @ filt, dtype=float)

        if self._state is None:
            return None
        dense_q = self.build_q_trace_operator(toroidal_matrices)
        if dense_q is None:
            return None
        return _adapt_e_operator_to_js(self._state, n, np.asarray(dense_q, dtype=float))


@dataclass
class AdditiveKnownSourceTraceModel(KnownSourceTraceModel):
    """Known-source q-trace formed by summing several explicit components."""

    q_trace_models: tuple[KnownSourceTraceModel, ...]
    description: str = "additive known-source q-trace model"
    _state: Any = field(default=None, init=False, repr=False, compare=False)

    def bind_state(self, state: Any) -> None:
        self._state = state
        for model in self.q_trace_models:
            if hasattr(model, "bind_state"):
                model.bind_state(state)

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        ops: list[np.ndarray] = []
        n = int(toroidal_matrices.basis.index_length)
        for model in self.q_trace_models:
            op = model.build_q_trace_operator(toroidal_matrices)
            if op is None:
                continue
            ops.append(np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float))
        if not ops:
            return None
        return np.asarray(sum(ops), dtype=float)

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        n = int(toroidal_matrices.basis.index_length)
        ops: list[np.ndarray] = []
        for model in self.q_trace_models:
            op = model.build_q_trace_from_js_operator(toroidal_matrices)
            if op is None:
                continue
            ops.append(np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float))
        if ops:
            return np.asarray(sum(ops), dtype=float)
        if self._state is None:
            return None
        dense_q = self.build_q_trace_operator(toroidal_matrices)
        if dense_q is None:
            return None
        return _adapt_e_operator_to_js(self._state, n, np.asarray(dense_q, dtype=float))


@dataclass
class InductivePlusHarmonicKnownSourceTraceModel(KnownSourceTraceModel):
    """Experimental ``q = q_part + q_hom`` split.

    The particular part is taken from the df/inductive shell channel using the
    existing nonlocal upper-side toroidal update model, while the homogeneous
    remainder is taken from the cf/curl-free shell channel using the direct
    harmonic side operator ``Lambda_U``.
    """

    outer_boundary_mode: str = "open"
    description: str = "experimental q_part + q_hom known-source trace model"
    _delegate: Optional[AdditiveKnownSourceTraceModel] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        particular = FilteredKnownSourceTraceModel(
            base_q_trace_model=ExteriorToroidalUpdateKnownSourceTraceModel(
                EquivalentNonlocalExteriorToroidalUpdateModel()
            ),
            shell_channel="df",
            description="df-filtered particular q-trace",
        )
        homogeneous = FilteredKnownSourceTraceModel(
            base_q_trace_model=PoloidalSideTraceKnownSourceTraceModel(
                HarmonicPoloidalSideTraceModel(outer_boundary_mode=self.outer_boundary_mode)
            ),
            shell_channel="cf",
            description="cf-filtered harmonic remainder q-trace",
        )
        delegate = AdditiveKnownSourceTraceModel(
            q_trace_models=(particular, homogeneous),
            description=self.description,
        )
        delegate.bind_state(state)
        self._delegate = delegate

    def _require_delegate(self) -> AdditiveKnownSourceTraceModel:
        if self._delegate is None:
            raise RuntimeError(
                "InductivePlusHarmonicKnownSourceTraceModel requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._delegate

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self._require_delegate().build_q_trace_operator(toroidal_matrices)

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self._require_delegate().build_q_trace_from_js_operator(toroidal_matrices)


@dataclass
class CurrentFirstParticularPlusHarmonicKnownSourceTraceModel(KnownSourceTraceModel):
    """Experimental ``q = q_part + q_hom`` split with a current-first particular part.

    This variant uses the thin-sheet shell-current source law as the
    particular branch, filtered in ``J_S`` space, and the direct harmonic
    side operator as the cf/homogeneous remainder. It is closer to the
    practical ``J_S -> q`` runtime normalization than the electric-first
    equivalent-update split above.
    """

    outer_boundary_mode: str = "open"
    description: str = "experimental current-first q_part + q_hom known-source trace model"
    _delegate: Optional[AdditiveKnownSourceTraceModel] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        current_driven_response = ShellCurrentDrivenKnownElectricRadialResponseModel(
            shell_current_source_model=ThinSheetCurrentContinuityKnownShellCurrentSourceModel()
        )
        particular = FilteredKnownSourceTraceModel(
            base_q_trace_model=KnownSourceOperatorKnownSourceTraceModel(
                known_source_model=current_driven_response
            ),
            shell_channel="df",
            input_space="js",
            description="df-filtered current-first particular q-trace",
        )
        homogeneous = FilteredKnownSourceTraceModel(
            base_q_trace_model=PoloidalSideTraceKnownSourceTraceModel(
                HarmonicPoloidalSideTraceModel(outer_boundary_mode=self.outer_boundary_mode)
            ),
            shell_channel="cf",
            input_space="js",
            description="cf-filtered harmonic remainder q-trace in JS space",
        )
        delegate = AdditiveKnownSourceTraceModel(
            q_trace_models=(particular, homogeneous),
            description=self.description,
        )
        delegate.bind_state(state)
        self._delegate = delegate

    def _require_delegate(self) -> AdditiveKnownSourceTraceModel:
        if self._delegate is None:
            raise RuntimeError(
                "CurrentFirstParticularPlusHarmonicKnownSourceTraceModel requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._delegate

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self._require_delegate().build_q_trace_operator(toroidal_matrices)

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self._require_delegate().build_q_trace_from_js_operator(toroidal_matrices)


@dataclass
class ShellElectricDifferenceKnownSourceTraceModel(KnownSourceTraceModel):
    """Adapter turning a first-trace model into the exact ``q = d_r U - E_r`` trace.

    This is the minimal forcing-side interface for the exact radial-shell
    source. It does not choose any new physics by itself; it simply exposes
    the single trace required by the one-curl forcing law from an existing
    ``ShellElectricTraceModel``.
    """

    shell_trace_model: ShellElectricTraceModel
    description: str = "known-source q-trace from shell electric first traces"

    def bind_state(self, state: Any) -> None:
        if hasattr(self.shell_trace_model, "bind_state"):
            self.shell_trace_model.bind_state(state)

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        op = self.shell_trace_model.build_trace_operator(toroidal_matrices)
        if op is None:
            return None

        n = int(toroidal_matrices.basis.index_length)
        dense_trace = np.asarray(coerce_dense_operator_matrix(op, n_cols=2 * n), dtype=float)
        if dense_trace.shape != (2 * n, 2 * n):
            raise ValueError(
                "Shell electric trace operator has wrong shape for q-trace adapter: "
                f"{dense_trace.shape}, expected ({2 * n}, {2 * n})."
            )
        dudr_op = dense_trace[:n]
        er_op = dense_trace[n:]
        return np.asarray(dudr_op - er_op, dtype=float)


@dataclass
class ExteriorToroidalUpdateKnownSourceTraceModel(KnownSourceTraceModel):
    """Adapter exposing ``q = d_r U - E_r`` from an upper-side update model.

    This is the non-harmonic q-trace route. For any concrete known-forcing
    upper-side update model,

        ``E_known -> dt_psi^+`` or ``E_known -> dt_jr^+``,

    the exact one-curl identity gives the mean-zero trace

        ``q = R_I * dt_psi^+``.

    So this adapter turns an ``ExteriorToroidalUpdateModel`` directly into the
    forcing-side q-trace required by the exact radial-shell source law.
    """

    exterior_update_model: ExteriorToroidalUpdateModel
    description: str = "known-source q-trace from upper-side toroidal update"

    def bind_state(self, state: Any) -> None:
        if hasattr(self.exterior_update_model, "bind_state"):
            self.exterior_update_model.bind_state(state)

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return build_q_trace_operator_from_exterior_update_model(
            toroidal_matrices, self.exterior_update_model
        )


@dataclass
class QTraceKnownSourceRadialResponseModel(RadialShellResponseModel):
    """Forcing-side radial-shell model built from the exact ``q = d_r U - E_r`` trace.

    This is the smallest exact forcing-side interface for the radial-shell law:

        ``E_known -> q_known -> dt_jr,known^+ -> rhs_known``.

    It becomes a fully first-principles forcing model once the supplied
    ``KnownSourceTraceModel`` is itself derived from the adopted gap physics.
    """

    q_trace_model: KnownSourceTraceModel
    description: str = "forcing-side radial-shell response from exact q-trace"

    def bind_state(self, state: Any) -> None:
        if hasattr(self.q_trace_model, "bind_state"):
            self.q_trace_model.bind_state(state)

    def build_known_source_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return build_known_source_operator_from_q_trace(toroidal_matrices, self.q_trace_model)

    def build_known_source_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        q_from_js = self.q_trace_model.build_q_trace_from_js_operator(toroidal_matrices)
        if q_from_js is None:
            return None

        n = int(toroidal_matrices.basis.index_length)
        dense_q = np.asarray(coerce_dense_operator_matrix(q_from_js, n_cols=2 * n), dtype=float)
        lap = np.asarray(
            coerce_dense_operator_matrix(
                toroidal_matrices.basis.get_laplacian_operator(r=toroidal_matrices.RI),
                n_cols=n,
            ),
            dtype=float,
        )
        return np.asarray((-(1.0 / float(mu0))) * (lap @ dense_q), dtype=float)

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self.q_trace_model.build_q_trace_operator(toroidal_matrices)

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self.q_trace_model.build_q_trace_from_js_operator(toroidal_matrices)

    def build_rhs_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return build_radial_shell_rhs_from_q_trace_operator(toroidal_matrices, self.q_trace_model)


@dataclass
class ExteriorToroidalScalarRadialResponseModel(RadialShellResponseModel):
    """Radial-shell response induced by an explicit upper-side toroidal update model.

    This is the sharp scalar formulation of the radial-shell closure. A
    concrete ``ExteriorToroidalUpdateModel`` provides either ``dt_jr^+`` or
    ``dt_psi^+``, and the shell RHS is then built from the exact identities

        ``rhs = mu0 * dt_jr^+``
        ``rhs = -R_I * Delta_S(dt_psi^+)``.
    """

    exterior_update_model: ExteriorToroidalUpdateModel
    description: str = "radial-shell response from explicit upper-side toroidal update model"

    def bind_state(self, state: Any) -> None:
        if hasattr(self.exterior_update_model, "bind_state"):
            self.exterior_update_model.bind_state(state)

    def build_rhs_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        n = int(toroidal_matrices.basis.index_length)

        dtjr_op = self.exterior_update_model.build_dtjr_operator(toroidal_matrices)
        if dtjr_op is not None:
            dense_dtjr = np.asarray(
                coerce_dense_operator_matrix(dtjr_op, n_cols=2 * n), dtype=float
            )
            return np.asarray(mu0 * dense_dtjr, dtype=float)

        dtpsi_op = self.exterior_update_model.build_dtpsi_operator(toroidal_matrices)
        if dtpsi_op is not None:
            dense_dtpsi = np.asarray(
                coerce_dense_operator_matrix(dtpsi_op, n_cols=2 * n), dtype=float
            )
            lap = np.asarray(
                coerce_dense_operator_matrix(
                    toroidal_matrices.basis.get_laplacian_operator(r=toroidal_matrices.RI),
                    n_cols=n,
                ),
                dtype=float,
            )
            return np.asarray((-(float(toroidal_matrices.RI))) * (lap @ dense_dtpsi), dtype=float)

        return None

    def build_q_trace_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return build_q_trace_operator_from_exterior_update_model(
            toroidal_matrices, self.exterior_update_model
        )

    def build_feedback_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        n = int(toroidal_matrices.basis.index_length)

        dtjr_op = self.exterior_update_model.build_dtjr_from_dtalpha_operator(toroidal_matrices)
        if dtjr_op is not None:
            dense_dtjr = np.asarray(coerce_dense_operator_matrix(dtjr_op, n_cols=n), dtype=float)
            return np.asarray(mu0 * dense_dtjr, dtype=float)

        dtpsi_op = self.exterior_update_model.build_dtpsi_from_dtalpha_operator(toroidal_matrices)
        if dtpsi_op is not None:
            dense_dtpsi = np.asarray(coerce_dense_operator_matrix(dtpsi_op, n_cols=n), dtype=float)
            lap = np.asarray(
                coerce_dense_operator_matrix(
                    toroidal_matrices.basis.get_laplacian_operator(r=toroidal_matrices.RI),
                    n_cols=n,
                ),
                dtype=float,
            )
            return np.asarray((-(float(toroidal_matrices.RI))) * (lap @ dense_dtpsi), dtype=float)

        return None


@dataclass
class EquivalentNonlocalExteriorToroidalUpdateModel(ExteriorToroidalUpdateModel):
    """Equivalent nonlocal upper-side toroidal update from the live tangential solve.

    This model does not derive the missing scalar from an explicit upper-region
    PDE. Instead, it extracts the equivalent upper-side updates produced by one
    benchmark tangential full-induction closure:

        ``E_known -> dt_alpha_tangential -> (dt_jr^+, dt_psi^+)``.

    It should be interpreted as an explicit equivalent scalar-update model for
    the existing runtime operator chain, not as a new first-principles
    derivation.
    """

    description: str = "equivalent nonlocal upper-side toroidal update from live tangential closure"
    benchmark_closure_mode: str = "tangential_full"
    _state: Any = field(default=None, init=False, repr=False, compare=False)
    _shadow_toroidal_matrices: Any = field(default=None, init=False, repr=False, compare=False)
    _dtjr_operator_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _dtpsi_operator_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        self._state = state
        self._shadow_toroidal_matrices = None
        self._dtjr_operator_cache = None
        self._dtpsi_operator_cache = None

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError(
                "EquivalentNonlocalExteriorToroidalUpdateModel requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._state

    def _get_shadow_toroidal_matrices(self, toroidal_matrices: Any) -> Any:
        if self._shadow_toroidal_matrices is not None:
            return self._shadow_toroidal_matrices

        closure_mode = str(self.benchmark_closure_mode).strip().lower()
        if closure_mode != "tangential_full":
            raise ValueError(
                "EquivalentNonlocalExteriorToroidalUpdateModel benchmark_closure_mode "
                "must be 'tangential_full', "
                f"got {self.benchmark_closure_mode!r}."
            )

        mats_cls = toroidal_matrices.__class__
        shadow = mats_cls(
            basis=toroidal_matrices.basis,
            grid=toroidal_matrices.grid,
            b_field=toroidal_matrices.b_field,
            RI=toroidal_matrices.RI,
            RM=toroidal_matrices.RM,
            closure_mode=closure_mode,
            radial_shell_response_model=None,
            closure_derivative_basis=toroidal_matrices.closure_derivative_basis,
            rhs_derivative_basis=toroidal_matrices.rhs_derivative_basis,
            radial_derivative_basis=toroidal_matrices.radial_derivative_basis,
            toroidal_solver=toroidal_matrices.toroidal_solver,
            toroidal_preconditioner=toroidal_matrices.toroidal_preconditioner,
            toroidal_tolerance=toroidal_matrices.toroidal_tolerance,
        )
        self._shadow_toroidal_matrices = shadow
        return shadow

    def _build_dtalpha_from_e_operator(self, toroidal_matrices: Any) -> np.ndarray:
        st = self._require_state()
        shadow = self._get_shadow_toroidal_matrices(toroidal_matrices)
        constraint_system = st.dt_alpha_constraint_system
        dtalpha_from_rhs = np.asarray(
            shadow.build_dtalpha_from_toroidal_rhs_matrix(
                constraint_operator=constraint_system.hard_operator,
                weighting=st.toroidal_weighting,
                regularization_lambda=st.toroidal_regularization_lambda,
                penalty_operator=constraint_system.soft_operator,
                penalty_scaling=float(constraint_system.soft_scaling),
                hinv_rtol=0.0,
            ),
            dtype=float,
        )
        tangential_rhs_from_e = np.asarray(shadow.toroidal_rhs_from_E_operator, dtype=float)
        return np.asarray(dtalpha_from_rhs @ tangential_rhs_from_e, dtype=float)

    def build_dtjr_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._dtjr_operator_cache is None:
            shadow = self._get_shadow_toroidal_matrices(toroidal_matrices)
            dtalpha_from_e = self._build_dtalpha_from_e_operator(toroidal_matrices)
            alpha_to_jr = np.asarray(to_numpy(shadow.alpha_to_jr_coeff_operator), dtype=float)
            self._dtjr_operator_cache = np.asarray(alpha_to_jr @ dtalpha_from_e, dtype=float)
        return self._dtjr_operator_cache

    def build_dtpsi_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._dtpsi_operator_cache is None:
            shadow = self._get_shadow_toroidal_matrices(toroidal_matrices)
            dtalpha_from_e = self._build_dtalpha_from_e_operator(toroidal_matrices)
            alpha_to_psi = np.asarray(to_numpy(shadow.alpha_to_psi_coeff_operator), dtype=float)
            self._dtpsi_operator_cache = np.asarray(alpha_to_psi @ dtalpha_from_e, dtype=float)
        return self._dtpsi_operator_cache


@dataclass
class RMToroidalBoundaryUpdateModel(ExteriorToroidalUpdateModel):
    """Explicit upper-side toroidal update from shell FAC and ``R_M`` closure.

    This model uses the direct shell toroidal identity together with the
    explicit ``R_M`` boundary closure already assembled by the code:

        ``dt_alpha -> dt_psi_open(R_I)``
        ``dt_alpha -> dt_psi_boundary(R_M^-) -> dt_psi_boundary(R_I)``

    The open contribution is always the local shell relation
    ``dt_alpha -> dt_jr -> dt_psi``. For ``rm_boundary_mode='closed'``, an
    additional internal continuation of the boundary toroidal scalar generated
    by the divergent closure current at ``R_M`` is added.
    """

    rm_boundary_mode: str = "closed"
    description: str = "explicit toroidal update from shell FAC and RM closure"
    _state: Any = field(default=None, init=False, repr=False, compare=False)

    def bind_state(self, state: Any) -> None:
        self._state = state

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError(
                "RMToroidalBoundaryUpdateModel requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._state

    def build_dtjr_from_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return np.asarray(to_numpy(toroidal_matrices.alpha_to_jr_coeff_operator), dtype=float)

    def build_dtpsi_from_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        mode = str(self.rm_boundary_mode).lower()
        if mode not in {"open", "closed"}:
            raise ValueError(
                f"Invalid rm_boundary_mode {self.rm_boundary_mode!r}; expected 'open' or 'closed'."
            )

        base = np.asarray(to_numpy(toroidal_matrices.alpha_to_psi_coeff_operator), dtype=float)
        if mode == "open" or getattr(toroidal_matrices, "RM", None) in (None, 0):
            return base

        st = self._require_state()
        if getattr(toroidal_matrices, "_use_auxiliary_closure_basis", False):
            closure_basis = toroidal_matrices._toroidal_closure_projector.closure_basis
            c2s = np.asarray(toroidal_matrices._aux_to_state_scalar_map, dtype=float)
            alpha_to_boundary_psi_rm = np.asarray(
                st.poloidal_matrices.toroidal_rm_closure_operators.alpha_to_boundary_psi_rm_coeff,
                dtype=float,
            )
        else:
            closure_basis = toroidal_matrices.basis
            c2s = np.eye(int(toroidal_matrices.basis.index_length), dtype=float)
            alpha_to_boundary_psi_rm = np.asarray(
                st.toroidal_rm_boundary_operators.alpha_to_boundary_psi_rm, dtype=float
            )

        shift_rm_to_ri = np.asarray(
            to_dense(
                closure_basis.get_radial_shift_operator(
                    float(toroidal_matrices.RM), float(toroidal_matrices.RI), kind="internal"
                )
            ),
            dtype=float,
        )
        boundary_contribution = c2s @ (shift_rm_to_ri @ alpha_to_boundary_psi_rm)
        return np.asarray(base + boundary_contribution, dtype=float)


@dataclass
class CurrentContinuityExteriorToroidalUpdateModel(ExteriorToroidalUpdateModel):
    """Upper-side toroidal update from thin-shell current continuity.

    This is the minimal connector law implied by:

    - the strong FAC shell source ``dt_jr = B0r * dt_alpha``,
    - the explicit shell current response ``dt_psi -> dtK_S``,
    - and instantaneous thin-shell current continuity with no stored surface charge,

        ``dt_jr^+ = dt_jr - div_S(dtK_S)``.

    In coefficient form, with ``div_Omega`` acting on the active vector basis,
    this becomes

        ``dt_jr^+ = alpha_to_jr @ dt_alpha - (1/RI) * div_Omega(dtK_S)``.

    The shell-current increment ``dtK_S`` is taken from the explicit
    pre-resistivity operator ``dt_psi -> dtJ_S`` already assembled by the
    runtime, so this model closes the missing scalar without harmonic electric
    continuation or the raw ``R_M`` toroidal boundary term.
    """

    description: str = "upper-side toroidal update from thin-shell current continuity"
    _state: Any = field(default=None, init=False, repr=False, compare=False)
    _dtjr_from_dtalpha_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _dtpsi_from_dtalpha_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        self._state = state
        self._dtjr_from_dtalpha_cache = None
        self._dtpsi_from_dtalpha_cache = None

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError(
                "CurrentContinuityExteriorToroidalUpdateModel requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._state

    def build_dtjr_from_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._dtjr_from_dtalpha_cache is not None:
            return self._dtjr_from_dtalpha_cache

        st = self._require_state()
        n = int(toroidal_matrices.basis.index_length)
        alpha_to_jr = np.asarray(to_numpy(toroidal_matrices.alpha_to_jr_coeff_operator), dtype=float)
        alpha_to_psi = np.asarray(
            to_numpy(toroidal_matrices.alpha_to_psi_coeff_operator), dtype=float
        )
        psi_to_js = _get_toroidal_to_js_dense(st, n)
        div_omega = _get_vector_divergence_from_js_coeff_dense(
            st, n, toroidal_matrices.basis, int(psi_to_js.shape[0])
        )
        self._dtjr_from_dtalpha_cache = np.asarray(
            alpha_to_jr - (1.0 / float(toroidal_matrices.RI)) * (div_omega @ psi_to_js @ alpha_to_psi),
            dtype=float,
        )
        return self._dtjr_from_dtalpha_cache

    def build_dtpsi_from_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._dtpsi_from_dtalpha_cache is None:
            dtjr = np.asarray(self.build_dtjr_from_dtalpha_operator(toroidal_matrices), dtype=float)
            jr_to_psi = np.asarray(to_numpy(toroidal_matrices.jr_to_psi_coeff_operator), dtype=float)
            self._dtpsi_from_dtalpha_cache = np.asarray(jr_to_psi @ dtjr, dtype=float)
        return self._dtpsi_from_dtalpha_cache


@dataclass
class IncrementalIdealityCorrectedExteriorToroidalUpdateModel(ExteriorToroidalUpdateModel):
    """Upper-side toroidal update corrected by incremental ideality/Faraday traces.

    This model keeps the explicit ``R_M`` toroidal connector, but adds the
    shell-electric correction implied by the linearized ideality/Faraday
    identity

        ``R_I * dt_psi^+ = Pi0(d_r U - E_r)``.

    Operationally, the model uses:

    - a base explicit exterior toroidal update model, typically the raw
      ``R_M`` connector, and
    - a shell-electric trace model that turns the live shell response
      ``dt_alpha -> dt_psi -> E_shell`` into the missing scalar increment
      ``(d_r U - E_r) / R_I``.

    The resulting ``dt_psi^+`` is therefore the raw connector plus the
    incremental correction required to make the connector and shell-electric
    feedback representations consistent under the same adopted assumptions.
    """

    base_exterior_update_model: ExteriorToroidalUpdateModel
    shell_trace_model: Optional[ShellElectricTraceModel] = None
    shell_feedback_response_model: Optional["RadialShellResponseModel"] = None
    description: str = (
        "upper-side toroidal update with incremental ideality/Faraday correction"
    )
    _state: Any = field(default=None, init=False, repr=False, compare=False)
    _base_dtpsi_from_dtalpha_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _trace_dtpsi_from_dtalpha_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _correction_dtpsi_from_dtalpha_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _total_dtpsi_from_dtalpha_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _total_dtjr_from_dtalpha_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        self._state = state
        self._base_dtpsi_from_dtalpha_cache = None
        self._trace_dtpsi_from_dtalpha_cache = None
        self._correction_dtpsi_from_dtalpha_cache = None
        self._total_dtpsi_from_dtalpha_cache = None
        self._total_dtjr_from_dtalpha_cache = None
        if self.shell_trace_model is not None and hasattr(self.shell_trace_model, "bind_state"):
            self.shell_trace_model.bind_state(state)
        if self.shell_feedback_response_model is not None and hasattr(
            self.shell_feedback_response_model, "bind_state"
        ):
            self.shell_feedback_response_model.bind_state(state)
        if hasattr(self.base_exterior_update_model, "bind_state"):
            self.base_exterior_update_model.bind_state(state)

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError(
                "IncrementalIdealityCorrectedExteriorToroidalUpdateModel requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._state

    def _dtpsi_from_dtjr(self, toroidal_matrices: Any, dtjr_op: np.ndarray) -> np.ndarray:
        jr_to_psi = np.asarray(to_numpy(toroidal_matrices.jr_to_psi_coeff_operator), dtype=float)
        return np.asarray(jr_to_psi @ np.asarray(dtjr_op, dtype=float), dtype=float)

    def _dtjr_from_dtpsi(self, toroidal_matrices: Any, dtpsi_op: np.ndarray) -> np.ndarray:
        n = int(toroidal_matrices.basis.index_length)
        lap = np.asarray(
            coerce_dense_operator_matrix(
                toroidal_matrices.basis.get_laplacian_operator(r=toroidal_matrices.RI),
                n_cols=n,
            ),
            dtype=float,
        )
        return np.asarray(
            (-(float(toroidal_matrices.RI)) / float(mu0)) * (lap @ np.asarray(dtpsi_op, dtype=float)),
            dtype=float,
        )

    def build_base_dtpsi_from_dtalpha_operator(self, toroidal_matrices: Any) -> np.ndarray:
        if self._base_dtpsi_from_dtalpha_cache is not None:
            return self._base_dtpsi_from_dtalpha_cache

        n = int(toroidal_matrices.basis.index_length)
        dtpsi = self.base_exterior_update_model.build_dtpsi_from_dtalpha_operator(toroidal_matrices)
        if dtpsi is not None:
            dense = np.asarray(coerce_dense_operator_matrix(dtpsi, n_cols=n), dtype=float)
        else:
            dtjr = self.base_exterior_update_model.build_dtjr_from_dtalpha_operator(toroidal_matrices)
            if dtjr is None:
                raise NotImplementedError(
                    "Base exterior update model must supply either dtpsi_from_dtalpha "
                    "or dtjr_from_dtalpha for the corrected connector."
                )
            dense_dtjr = np.asarray(coerce_dense_operator_matrix(dtjr, n_cols=n), dtype=float)
            dense = self._dtpsi_from_dtjr(toroidal_matrices, dense_dtjr)
        self._base_dtpsi_from_dtalpha_cache = np.asarray(dense, dtype=float)
        return self._base_dtpsi_from_dtalpha_cache

    def build_matched_dtpsi_from_dtalpha_operator(self, toroidal_matrices: Any) -> np.ndarray:
        if self._trace_dtpsi_from_dtalpha_cache is not None:
            return self._trace_dtpsi_from_dtalpha_cache

        st = self._require_state()
        n = int(toroidal_matrices.basis.index_length)
        psi_to_e = _get_factorized_toroidal_to_e_dense(st, n)
        alpha_to_psi = np.asarray(to_numpy(toroidal_matrices.alpha_to_psi_coeff_operator), dtype=float)

        if self.shell_feedback_response_model is not None:
            rhs_from_e = self.shell_feedback_response_model.build_rhs_operator(toroidal_matrices)
            if rhs_from_e is None:
                raise NotImplementedError(
                    "IncrementalIdealityCorrectedExteriorToroidalUpdateModel requires a shell "
                    "feedback response model with an explicit rhs operator."
                )
            dense_rhs = np.asarray(coerce_dense_operator_matrix(rhs_from_e, n_cols=2 * n), dtype=float)
            feedback_rhs_from_dtalpha = dense_rhs @ psi_to_e @ alpha_to_psi
            dense = self._dtpsi_from_dtjr(
                toroidal_matrices, (1.0 / float(mu0)) * feedback_rhs_from_dtalpha
            )
        elif self.shell_trace_model is not None:
            trace_op = self.shell_trace_model.build_trace_operator(toroidal_matrices)
            if trace_op is None:
                raise NotImplementedError(
                    "IncrementalIdealityCorrectedExteriorToroidalUpdateModel requires a shell "
                    "trace model with an explicit trace operator."
                )
            dense_trace = np.asarray(
                coerce_dense_operator_matrix(trace_op, n_cols=2 * n), dtype=float
            )
            dudr_from_e = dense_trace[:n]
            er_from_e = dense_trace[n:]
            dense = (1.0 / float(toroidal_matrices.RI)) * (
                (dudr_from_e - er_from_e) @ psi_to_e @ alpha_to_psi
            )
        else:
            raise NotImplementedError(
                "IncrementalIdealityCorrectedExteriorToroidalUpdateModel requires either "
                "shell_feedback_response_model or shell_trace_model."
            )
        self._trace_dtpsi_from_dtalpha_cache = np.asarray(dense, dtype=float)
        return self._trace_dtpsi_from_dtalpha_cache

    def build_trace_dtpsi_from_dtalpha_operator(self, toroidal_matrices: Any) -> np.ndarray:
        """Backward-compatible alias for the matched dtpsi operator."""
        return self.build_matched_dtpsi_from_dtalpha_operator(toroidal_matrices)

    def build_correction_dtpsi_from_dtalpha_operator(self, toroidal_matrices: Any) -> np.ndarray:
        if self._correction_dtpsi_from_dtalpha_cache is not None:
            return self._correction_dtpsi_from_dtalpha_cache

        trace_dtpsi = self.build_matched_dtpsi_from_dtalpha_operator(toroidal_matrices)
        base_dtpsi = self.build_base_dtpsi_from_dtalpha_operator(toroidal_matrices)
        self._correction_dtpsi_from_dtalpha_cache = np.asarray(
            trace_dtpsi - base_dtpsi, dtype=float
        )
        return self._correction_dtpsi_from_dtalpha_cache

    def build_dtpsi_from_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._total_dtpsi_from_dtalpha_cache is None:
            base_dtpsi = self.build_base_dtpsi_from_dtalpha_operator(toroidal_matrices)
            correction_dtpsi = self.build_correction_dtpsi_from_dtalpha_operator(toroidal_matrices)
            self._total_dtpsi_from_dtalpha_cache = np.asarray(
                base_dtpsi + correction_dtpsi, dtype=float
            )
        return self._total_dtpsi_from_dtalpha_cache

    def build_dtjr_from_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._total_dtjr_from_dtalpha_cache is None:
            total_dtpsi = self.build_dtpsi_from_dtalpha_operator(toroidal_matrices)
            self._total_dtjr_from_dtalpha_cache = self._dtjr_from_dtpsi(
                toroidal_matrices, np.asarray(total_dtpsi, dtype=float)
            )
        return self._total_dtjr_from_dtalpha_cache


@dataclass
class EquivalentNonlocalRadialShellResponseModel(RadialShellResponseModel):
    """Equivalent nonlocal radial-shell response induced by the live toroidal solve.

    This model does not derive ``Q_I`` and ``E_{r,I}`` from an explicit 3D
    upper-region field model. Instead, it re-expresses one benchmark
    tangential full-induction toroidal closure as an equivalent scalar shell
    response:

        ``E_known -> f_ind(E_known) -> dt_alpha_tangential``
        ``        -> mu0 * B0r * dt_alpha_tangential``.

    Here ``f_ind`` is the live reduced inductive tangential forcing block
    assembled from the shell tangential electric field via the radial curl
    scalar ``hat(r)·curl(E)``. So this model should be read as the condensed
    forcing-side scalar response of the projected tangential closure, not as a
    direct radial-trace derivation.

    The resulting operator is nonlocal because the intermediate
    ``dt_alpha_tangential`` solve uses the same PFAC coupling, low-latitude
    constraints, and auxiliary closure basis as the live full-induction
    toroidal path. The benchmark shadow closure is the fuller
    ``tangential_full`` system. The model should therefore be interpreted as
    an explicit equivalent radial-shell response for the existing runtime
    model, not as a new first-principles derivation of
    ``Q_I - Delta_Omega E_{r,I}``.
    """

    description: str = "equivalent nonlocal radial-shell response from live tangential closure"
    benchmark_closure_mode: str = "tangential_full"
    _state: Any = field(default=None, init=False, repr=False, compare=False)
    _shadow_toroidal_matrices: Any = field(default=None, init=False, repr=False, compare=False)
    _rhs_operator_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        self._state = state
        self._shadow_toroidal_matrices = None
        self._rhs_operator_cache = None

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError(
                "EquivalentNonlocalRadialShellResponseModel requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._state

    def _get_shadow_toroidal_matrices(self, toroidal_matrices: Any) -> Any:
        if self._shadow_toroidal_matrices is not None:
            return self._shadow_toroidal_matrices

        closure_mode = str(self.benchmark_closure_mode).strip().lower()
        if closure_mode != "tangential_full":
            raise ValueError(
                "EquivalentNonlocalRadialShellResponseModel benchmark_closure_mode "
                "must be 'tangential_full', "
                f"got {self.benchmark_closure_mode!r}."
            )

        mats_cls = toroidal_matrices.__class__
        shadow = mats_cls(
            basis=toroidal_matrices.basis,
            grid=toroidal_matrices.grid,
            b_field=toroidal_matrices.b_field,
            RI=toroidal_matrices.RI,
            RM=toroidal_matrices.RM,
            closure_mode=closure_mode,
            radial_shell_response_model=None,
            closure_derivative_basis=toroidal_matrices.closure_derivative_basis,
            rhs_derivative_basis=toroidal_matrices.rhs_derivative_basis,
            radial_derivative_basis=toroidal_matrices.radial_derivative_basis,
            toroidal_solver=toroidal_matrices.toroidal_solver,
            toroidal_preconditioner=toroidal_matrices.toroidal_preconditioner,
            toroidal_tolerance=toroidal_matrices.toroidal_tolerance,
        )
        self._shadow_toroidal_matrices = shadow
        return shadow

    def build_rhs_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._rhs_operator_cache is not None:
            return self._rhs_operator_cache

        st = self._require_state()
        shadow = self._get_shadow_toroidal_matrices(toroidal_matrices)
        constraint_system = st.dt_alpha_constraint_system

        dtalpha_from_rhs = np.asarray(
            shadow.build_dtalpha_from_toroidal_rhs_matrix(
                constraint_operator=constraint_system.hard_operator,
                weighting=st.toroidal_weighting,
                regularization_lambda=st.toroidal_regularization_lambda,
                penalty_operator=constraint_system.soft_operator,
                penalty_scaling=float(constraint_system.soft_scaling),
                hinv_rtol=0.0,
            ),
            dtype=float,
        )
        tangential_rhs_from_e = np.asarray(shadow.toroidal_rhs_from_E_operator, dtype=float)
        alpha_to_jr = np.asarray(to_numpy(shadow.alpha_to_jr_coeff_operator), dtype=float)
        self._rhs_operator_cache = np.asarray(
            mu0 * alpha_to_jr @ dtalpha_from_rhs @ tangential_rhs_from_e,
            dtype=float,
        )
        return self._rhs_operator_cache

    def build_gap_coenergy_blocks(
        self, toroidal_matrices: Any
    ) -> Optional[GapCoenergyBlocks]:
        """Return a nondegenerate shell-gap realization using the runtime ``Pi_shell``.

        The condensed nonlocal forcing operator is still taken from the live
        tangential closure, but the Schur-complement realization is no longer
        the default zero-internal-variable embedding. Instead, the internal gap
        variable is the document-style shell correction ``Pi_shell``, enforced
        through

            ``x = Pi_shell * chi``

        with homogeneous outer magnetic update. Because ``Pi_shell`` is already
        exposed in the active solution coefficient space, the same realization
        carries over to the transform-backed modes without a separate gap-basis
        implementation. It is still not an independently assembled gap PDE
        solve, but it is a genuine shell-gap internal-variable model rather
        than a purely formal condensation of already-condensed operators.
        """

        st = self._require_state()
        gamma_known = self.build_gamma_known_operator(toroidal_matrices)
        if gamma_known is None:
            return None
        lambda_gap = self.build_lambda_gap_operator(toroidal_matrices)
        shell_pi = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_effective_operator, dtype=float
        )
        return build_shell_pi_gap_coenergy_blocks_from_condensed_operators(
            lambda_gap=np.asarray(lambda_gap, dtype=float),
            gamma_known=np.asarray(gamma_known, dtype=float),
            shell_pi_operator=shell_pi,
        )


@dataclass
class PFACNonlocalRadialShellResponseModel(RadialShellResponseModel):
    """Built-in PFAC/RM-based nonlocal radial-shell response model.

    This is the strongest explicit radial-shell model currently available from
    the runtime ingredients already assembled by the code:

    - forcing-side radial-shell response always runs through the exact
      ``q = d_r U - E_r`` shell interface,
    - the default forcing branch now uses the explicit current-first shell law
      ``delta E_S -> delta K_S -> delta j_r^+`` with the live shell
      constitutive adapter and exact shell inversion,
    - the built-in forcing branch is the explicit current-first shell law
      only; older benchmark forcing branches are no longer selectable through
      this model,
    - feedback-side closure uses the explicit thin-shell current-continuity law
      together with the runtime ``psi -> J_S`` operator, so the missing
      upper-side toroidal scalar is supplied by

          ``dt_jr^+ = dt_jr - div_S(dtK_S)``

      rather than by harmonic electric continuation or the raw ``R_M``
      toroidal boundary branch.

    It is therefore an explicit PFAC/RM radial-shell closure built from the
    existing response operators without making harmonic ``U`` part of the
    built-in default path. Trace-based electric continuation models remain
    available explicitly, but they are no longer the default.

    The split is intentional and non-overlapping:
    - forcing-side radial-shell RHS comes only from the selected forcing
      branch,
    - induced dt_alpha feedback comes only from the corrected upper-side
      toroidal update branch.
    This keeps the radial-shell scalar closure from double-counting the same
    response through both shell-electric and upper-toroidal pathways.
    """

    rm_boundary_mode: str = "auto"
    forcing_mode: RadialShellForcingMode | str = (
        RadialShellForcingMode.FROZEN_CONDUCTANCE_INCREMENTAL
    )
    description: str = "built-in PFAC/RM nonlocal radial-shell response model"
    _delegate: Optional["NonlocalShellElectricRadialResponseModel"] = field(
        default=None, init=False, repr=False, compare=False
    )

    def _resolve_rm_boundary_mode(self, state: Any) -> str:
        mode = str(self.rm_boundary_mode).lower()
        if mode != "auto":
            return mode
        if getattr(state, "RM", None) not in (None, 0) and bool(
            getattr(state, "magnetospheric_shielding", True)
        ):
            return "closed"
        return "open"

    def _require_delegate(self) -> "NonlocalShellElectricRadialResponseModel":
        if self._delegate is None:
            raise RuntimeError(
                "PFACNonlocalRadialShellResponseModel requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._delegate

    def _resolve_forcing_mode(self) -> str:
        mode = self.forcing_mode
        if isinstance(mode, RadialShellForcingMode):
            return mode.value
        normalized = str(mode).strip().lower()
        valid = {"frozen_conductance_incremental"}
        if normalized not in valid:
            raise ValueError(
                "PFACNonlocalRadialShellResponseModel forcing_mode must be one of "
                f"{sorted(valid)}, got {mode!r}."
            )
        return normalized

    def bind_state(self, state: Any) -> None:
        forcing_mode = self._resolve_forcing_mode()
        if forcing_mode != "frozen_conductance_incremental":
            raise AssertionError(f"Unhandled radial shell forcing mode {forcing_mode!r}.")
        shell_response_model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()

        if hasattr(shell_response_model, "bind_state"):
            shell_response_model.bind_state(state)

        q_trace_model: KnownSourceTraceModel = SchurComplementGapKnownSourceTraceModel(
            blocks_builder=lambda toroidal_matrices, model=shell_response_model: model.build_gap_coenergy_blocks(
                toroidal_matrices
            ),
            source_model=shell_response_model,
        )

        delegate = NonlocalShellElectricRadialResponseModel(
            shell_response_model=QTraceKnownSourceRadialResponseModel(
                q_trace_model=q_trace_model
            ),
            exterior_update_model=CurrentContinuityExteriorToroidalUpdateModel(),
        )
        delegate.bind_state(state)
        self._delegate = delegate
        self.description = (
            "built-in PFAC/RM nonlocal radial-shell response model "
            f"(forcing_mode={forcing_mode})"
        )

    def build_rhs_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self._require_delegate().build_rhs_operator(toroidal_matrices)

    def build_feedback_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self._require_delegate().build_feedback_dtalpha_operator(toroidal_matrices)

    def build_known_source_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self._require_delegate().build_known_source_from_js_operator(toroidal_matrices)

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self._require_delegate().build_q_trace_from_js_operator(toroidal_matrices)


@dataclass
class NonlocalShellElectricRadialResponseModel(RadialShellResponseModel):
    """Nonlocal radial-shell closure built from the live ``psi -> E`` shell response.

    This model keeps the radial-shell equation in scalar form, but uses the
    current nonlocal shell electric response already present in the code:

        ``dt_alpha -> dt_psi -> E_shell(dt_psi) -> rhs_radial_shell``.

    The electric response ``psi -> E_shell`` is taken from the live
    conductivity/PFAC pathway, so the feedback is nonlocal and includes the
    same hemispheric magnetic coupling as the current full-induction path.

    It is still not the final first-principles gap-region model unless a
    concrete ``ShellElectricTraceModel`` is supplied. Without that, the shell
    scalar is evaluated with a shell operator acting on shell electric
    coefficients rather than from explicit ``E_{r,I}`` and ``Q_I`` traces.

    If no explicit ``shell_response_model`` is supplied, the runtime falls
    back to the canonical equivalent nonlocal shell response built from the
    live tangential full-induction operator, not to any shell-local
    approximation.
    """

    shell_response_model: RadialShellResponseModel = field(
        default_factory=lambda: EquivalentNonlocalRadialShellResponseModel()
    )
    shell_trace_model: Optional[ShellElectricTraceModel] = None
    exterior_update_model: Optional[ExteriorToroidalUpdateModel] = None
    description: str = (
        "nonlocal shell-electric radial response using live psi->E coupling and "
        "a shell scalar response operator"
    )
    _state: Any = field(default=None, init=False, repr=False, compare=False)
    _rhs_operator_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _feedback_operator_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        self._state = state
        self._rhs_operator_cache = None
        self._feedback_operator_cache = None
        if hasattr(self.shell_response_model, "bind_state"):
            self.shell_response_model.bind_state(state)
        if self.shell_trace_model is not None and hasattr(self.shell_trace_model, "bind_state"):
            self.shell_trace_model.bind_state(state)
        if self.exterior_update_model is not None and hasattr(
            self.exterior_update_model, "bind_state"
        ):
            self.exterior_update_model.bind_state(state)

    def _build_rhs_from_trace_model(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self.shell_trace_model is None:
            return None
        trace_op = self.shell_trace_model.build_trace_operator(toroidal_matrices)
        if trace_op is None:
            return None
        return build_radial_shell_rhs_from_trace_operator(
            toroidal_matrices, self.shell_trace_model
        )

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError(
                "NonlocalShellElectricRadialResponseModel requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._state

    def build_rhs_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._rhs_operator_cache is None:
            op = self._build_rhs_from_trace_model(toroidal_matrices)
            if op is None:
                op = self.shell_response_model.build_rhs_operator(toroidal_matrices)
            if op is None:
                return None
            self._rhs_operator_cache = np.asarray(op, dtype=float)
        return self._rhs_operator_cache

    def build_feedback_dtalpha_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._feedback_operator_cache is not None:
            return self._feedback_operator_cache

        if self.exterior_update_model is not None:
            adapter = ExteriorToroidalScalarRadialResponseModel(
                exterior_update_model=self.exterior_update_model
            )
            adapter.bind_state(self._require_state())
            op = adapter.build_feedback_dtalpha_operator(toroidal_matrices)
            if op is not None:
                self._feedback_operator_cache = np.asarray(op, dtype=float)
                return self._feedback_operator_cache

        st = self._require_state()
        rhs_from_e = self.build_rhs_operator(toroidal_matrices)
        if rhs_from_e is None:
            return None

        n = int(toroidal_matrices.basis.index_length)
        psi_to_e = _get_factorized_toroidal_to_e_dense(st, n)
        alpha_to_psi = np.asarray(to_numpy(toroidal_matrices.alpha_to_psi_coeff_operator), dtype=float)
        self._feedback_operator_cache = np.asarray(rhs_from_e @ psi_to_e @ alpha_to_psi, dtype=float)
        return self._feedback_operator_cache

    def build_known_source_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if hasattr(self.shell_response_model, "build_known_source_from_js_operator"):
            op = self.shell_response_model.build_known_source_from_js_operator(toroidal_matrices)
            if op is not None:
                return op

        n = int(toroidal_matrices.basis.index_length)
        dense_gamma = _optional_dense_operator(
            self.build_known_source_operator(toroidal_matrices), n_cols=2 * n
        )
        if dense_gamma is None:
            return None

        return _adapt_e_operator_to_js(self._require_state(), n, dense_gamma)

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if hasattr(self.shell_response_model, "build_q_trace_from_js_operator"):
            op = self.shell_response_model.build_q_trace_from_js_operator(toroidal_matrices)
            if op is not None:
                return op

        n = int(toroidal_matrices.basis.index_length)
        dense_q = _optional_dense_operator(self.build_q_trace_operator(toroidal_matrices), n_cols=2 * n)
        if dense_q is None:
            return None

        return _adapt_e_operator_to_js(self._require_state(), n, dense_q)

@dataclass
class ThinSheetCurrentContinuityKnownShellCurrentSourceModel:
    """Current-first forcing-side shell source from thin-sheet continuity.

    This class exposes the primitive shell-current law used by the explicit
    frozen-conductance forcing branch:

        ``dtK_S,known -> dt_jr,known^+``

    under the adopted thin-sheet/no-stored-charge shell model,

        ``dt_jr,known^+ = -(1/RI) * div_Omega(dtK_S,known)``.

    The shell-current coefficients are the pre-resistivity sheet-current
    coefficients used by the live ``J_S -> E`` constitutive operator.
    """

    description: str = "known shell-current source from thin-sheet current continuity"
    _state: Any = field(default=None, init=False, repr=False, compare=False)
    _known_source_from_js_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        self._state = state
        self._known_source_from_js_cache = None

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError(
                "ThinSheetCurrentContinuityKnownShellCurrentSourceModel requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._state

    def build_known_source_from_js_operator(self, toroidal_matrices: Any) -> np.ndarray:
        if self._known_source_from_js_cache is not None:
            return self._known_source_from_js_cache

        st = self._require_state()
        n = int(toroidal_matrices.basis.index_length)
        js_n_rows = 2 * n
        div_omega = _get_vector_divergence_from_js_coeff_dense(
            st, n, toroidal_matrices.basis, js_n_rows
        )
        self._known_source_from_js_cache = np.asarray(
            (-(1.0 / float(toroidal_matrices.RI))) * div_omega,
            dtype=float,
        )
        return self._known_source_from_js_cache

    def build_rhs_from_js_operator(self, toroidal_matrices: Any) -> np.ndarray:
        return np.asarray(
            float(mu0) * self.build_known_source_from_js_operator(toroidal_matrices),
            dtype=float,
        )

    def build_q_trace_from_js_operator(self, toroidal_matrices: Any) -> np.ndarray:
        source = np.asarray(self.build_known_source_from_js_operator(toroidal_matrices), dtype=float)
        jr_to_psi = np.asarray(to_numpy(toroidal_matrices.jr_to_psi_coeff_operator), dtype=float)
        return np.asarray(float(toroidal_matrices.RI) * (jr_to_psi @ source), dtype=float)


@dataclass
class ShellCurrentDrivenKnownElectricRadialResponseModel(RadialShellResponseModel):
    """Generic ``E -> J_S -> dtj_r^+`` forcing-side radial-shell response.

    This class unifies forcing-side radial-shell models whose primitive law is
    current-first,

        ``dtK_S,known -> dt_jr,known^+``,

    while the runtime forcing state is still expressed through shell electric
    coefficients. The shell-electric step is then supplied by the live
    constitutive adapter

        ``dtE_S,known -> dtK_S,known``.

    Concretely, with the live shell constitutive operator exposed as
    ``J_S -> E``, the current-driven forcing path becomes

        ``E_known -> J_S,known -> dt_jr,known^+``.

    This is the common code path behind current-first forcing models such as
    the frozen-conductance incremental branch.
    """

    shell_current_source_model: Any
    description: str = "forcing-side radial-shell response from current-first shell source law"
    _state: Any = field(default=None, init=False, repr=False, compare=False)
    _rhs_operator_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )
    _known_source_operator_cache: Optional[np.ndarray] = field(
        default=None, init=False, repr=False, compare=False
    )

    def bind_state(self, state: Any) -> None:
        self._state = state
        self._rhs_operator_cache = None
        self._known_source_operator_cache = None
        if hasattr(self.shell_current_source_model, "bind_state"):
            self.shell_current_source_model.bind_state(state)

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a bound State. "
                "Pass the model into Dynamics(...) or run_pynamit(...), so State can bind it."
            )
        return self._state

    def build_rhs_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._rhs_operator_cache is None:
            source = self.build_known_source_operator(toroidal_matrices)
            if source is None:
                return None
            self._rhs_operator_cache = np.asarray(float(mu0) * np.asarray(source, dtype=float), dtype=float)
        return self._rhs_operator_cache

    def build_known_source_from_js_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        return self.shell_current_source_model.build_known_source_from_js_operator(toroidal_matrices)

    def build_known_source_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self._known_source_operator_cache is not None:
            return self._known_source_operator_cache

        st = self._require_state()
        n = int(toroidal_matrices.basis.index_length)
        e_to_js = _get_e_to_js_dense(st, n)
        source_from_js = np.asarray(
            self.build_known_source_from_js_operator(toroidal_matrices), dtype=float
        )
        source_from_js = _adapt_source_from_js_to_live_representation(
            st, n, toroidal_matrices.basis, source_from_js, int(e_to_js.shape[0])
        )
        self._known_source_operator_cache = np.asarray(source_from_js @ e_to_js, dtype=float)
        return self._known_source_operator_cache

    def build_gap_coenergy_blocks(
        self, toroidal_matrices: Any
    ) -> Optional[GapCoenergyBlocks]:
        """Return a nondegenerate one-boundary shell-gap realization.

        The current-first forcing branches are already closed at the shell:

            ``E_known -> J_S,known -> dt_jr,known^+ -> chi``.

        They therefore do not need the borrowed tangential benchmark to define
        ``Gamma_known``. To keep the shell-gap story aligned with the document
        model, we realize the same condensed forcing operator through the
        runtime shell correction ``Pi_shell`` as an explicit internal gap
        variable. This gives a genuine one-boundary shell-gap block model
        without claiming that the forcing-side source has been derived from the
        PFAC magnetic blocks alone.
        """
        gamma_known = self.build_gamma_known_operator(toroidal_matrices)
        if gamma_known is None:
            return None

        st = self._require_state()
        shell_pi = np.asarray(
            st.poloidal_matrices.dynamic_toroidal_shell_pi_effective_operator, dtype=float
        )
        lambda_gap = np.asarray(self.build_lambda_gap_operator(toroidal_matrices), dtype=float)
        return build_shell_pi_gap_coenergy_blocks_from_condensed_operators(
            lambda_gap=lambda_gap,
            gamma_known=np.asarray(gamma_known, dtype=float),
            shell_pi_operator=shell_pi,
        )


@dataclass
class FrozenConductanceIncrementalKnownElectricRadialResponseModel(
    ShellCurrentDrivenKnownElectricRadialResponseModel
):
    """Forcing-side radial-shell response for incremental shell forcing.

    This model specializes the generic current-driven forcing path to the
    current-first thin-sheet shell law

        ``dtK_S,known -> dt_jr,known^+``

    from thin-sheet current continuity, with the frozen-conductance adapter

        ``delta K_S = Sigma * delta E_S``.

    So the forcing-side path becomes

        ``delta E_known -> delta K_S -> delta j_r,known^+``.

    In the coefficient-space runtime the shell-current increment map
    ``delta E_known -> delta K_S`` is taken as the discrete inverse of the
    live shell constitutive operator ``J_S -> E`` in the active coefficient
    space. The radial-shell forcing scalar then follows from the current-first
    source law:

        ``rhs_known = mu0 * delta j_r,known^+``
        ``          = -(mu0/R_I) * div_Omega(delta K_S)``.

    Runtime interpretation
    ----------------------
    This law is exact *as an incremental law*. If the forcing input supplied to
    the toroidal solve is interpreted as ``delta E_S`` (or equivalently as the
    linearized shell-electric increment for the same substep), then the model
    is first-principles under frozen conductance.

    The current built-in runtime forcing path does not generally expose its
    forcing that way; it passes instantaneous non-inductive shell electric
    coefficients ``E_known`` into the reduced tangential forcing operator.
    So this model should currently be treated as an explicit incremental
    alternative forcing law, not as the default replacement for the condensed
    forcing branch.
    """

    description: str = (
        "incremental forcing-side radial-shell response from frozen shell "
        "conductance and thin-sheet current continuity"
    )
    shell_current_source_model: ThinSheetCurrentContinuityKnownShellCurrentSourceModel = field(
        default_factory=ThinSheetCurrentContinuityKnownShellCurrentSourceModel
    )


@dataclass
class ShellCurrentContinuityKnownElectricRadialResponseModel(
    FrozenConductanceIncrementalKnownElectricRadialResponseModel
):
    """Backward-compatible alias for the historical shell-current forcing model.

    This class keeps the previous public name, but its correct interpretation is
    now explicit: it is the frozen-conductance incremental forcing law
    implemented by
    ``FrozenConductanceIncrementalKnownElectricRadialResponseModel``.
    """

    description: str = (
        "backward-compatible alias for frozen-conductance incremental shell-current forcing"
    )


@dataclass(frozen=True)
class CallableRadialShellResponseModel(RadialShellResponseModel):
    """Small adapter for injected callable radial-shell response models."""

    operator_builder: Optional[Callable[[Any], Any]] = None
    rhs_evaluator: Optional[Callable[[Any, np.ndarray], np.ndarray]] = None
    description: str = "callable radial-shell response model"

    def build_rhs_operator(self, toroidal_matrices: Any) -> Optional[np.ndarray]:
        if self.operator_builder is None:
            return None
        return self.operator_builder(toroidal_matrices)

    def compute_rhs(self, toroidal_matrices: Any, E_coeffs: np.ndarray) -> np.ndarray:
        if self.rhs_evaluator is not None:
            return asarray(self.rhs_evaluator(toroidal_matrices, E_coeffs)).reshape(-1)
        return super().compute_rhs(toroidal_matrices, E_coeffs)
