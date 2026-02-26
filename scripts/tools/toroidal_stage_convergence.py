"""Benchmark toroidal stage parity between ST-CS and CS-dominant backends.

This script avoids time integration and file output. It only builds initialized
states and compares stage operators in a common SH-analysis space.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import tempfile
import time

import numpy as np

# Avoid expensive per-run Matplotlib/font cache initialization in non-writable HOME.
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = os.path.join(tempfile.gettempdir(), "pynamit-mpl-cache")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_dir
if "XDG_CACHE_HOME" not in os.environ:
    xdg_dir = os.path.join(tempfile.gettempdir(), "pynamit-xdg-cache")
    os.makedirs(xdg_dir, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = xdg_dir

from pynamit.math.linear_map import as_linear_map
from pynamit.primitives.field import Field
from pynamit.simulation.dynamics import Dynamics, SimulationMode
from pynamit.simulation.geometry_utils import to_dense
from pynamit.spherical_harmonics.sh_basis import SHBasis


@dataclass(frozen=True)
class StageErrors:
    psi_to_e: float
    e_to_forcing: float
    dtjr_from_k: float
    jr_to_psi: float


@dataclass(frozen=True)
class StageTimings:
    build_states: float
    psi_to_e: float
    e_to_forcing: float
    dtjr_from_k: float
    jr_to_psi: float


def _lift_two_channel(op: np.ndarray) -> np.ndarray:
    z = np.zeros_like(op)
    return np.block([[op, z], [z, op]])


def _build_state(
    sim_mode: SimulationMode,
    *,
    nmax: int,
    mmax: int,
    ncs: int,
    backend: str,
    dense_full_operators: bool,
    mainfield_kind: str,
    ignore_pfac: bool,
    connect_hemispheres: bool,
) -> Dynamics:
    dynamics = Dynamics(
        filename_prefix="benchmark_stage",
        Nmax=nmax,
        Mmax=mmax,
        Ncs=ncs,
        dynamics_mode="full_induction",
        simulation_mode=sim_mode,
        ignore_PFAC=ignore_pfac,
        mainfield_kind=mainfield_kind,
        mainfield_epoch=2020,
        connect_hemispheres=connect_hemispheres,
        least_squares_solver="svd",
        magnetospheric_toroidal_lock=False,
        northern_hemisphere_apex_constraints=True,
        backend=backend,
        dense_full_operators=dense_full_operators,
        benchmark_mode=True,
    )

    grid = dynamics.state.geometry.grid

    theta = np.deg2rad(90.0 - grid.lat)
    hall = 0.3 + 0.1 * np.sin(theta)
    pedersen = 1.0 + 0.2 * np.cos(theta) ** 2
    eta_p = pedersen / (hall ** 2 + pedersen ** 2)
    eta_h = hall / (hall ** 2 + pedersen ** 2)
    interp_basis = dynamics.interpolation_bases["conductance"]
    dynamics.state.etaP = Field.from_grid_values_expansion(
        interp_basis,
        np.asarray(eta_p).reshape(-1),
        grid,
        field_type="scalar",
    )
    dynamics.state.etaH = Field.from_grid_values_expansion(
        interp_basis,
        np.asarray(eta_h).reshape(-1),
        grid,
        field_type="scalar",
    )
    return dynamics


def _scalar_transfer_operators(state, sh_basis: SHBasis) -> tuple[np.ndarray, np.ndarray]:
    grid = state.geometry.grid
    sol = state.solution_basis

    g_sol = np.asarray(to_dense(sol.get_evaluation_matrix(grid)))
    p_sol = np.asarray(to_dense(sol.construct_scalar_projection_matrix(grid)))
    g_sh = np.asarray(to_dense(sh_basis.get_evaluation_matrix(grid)))
    p_sh = np.asarray(to_dense(sh_basis.construct_scalar_projection_matrix(grid)))

    # SH analysis coefficients -> solution coefficients
    lift_in = p_sol @ g_sh
    # solution coefficients -> SH analysis coefficients
    lift_out = p_sh @ g_sol
    return lift_in, lift_out


def _probe_rel_error(
    apply_st,
    apply_cs,
    *,
    input_size: int,
    n_probe: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    num = 0.0
    den = 0.0
    for _ in range(max(int(n_probe), 1)):
        x = rng.standard_normal(input_size)
        y_st = np.asarray(apply_st(x))
        y_cs = np.asarray(apply_cs(x))
        dy = y_cs - y_st
        num += float(np.dot(dy.ravel(), dy.ravel()))
        den += float(np.dot(y_st.ravel(), y_st.ravel()))
    if den <= 0:
        return 0.0
    return float(np.sqrt(num / den))


def compute_stage_errors(
    *,
    nmax: int,
    mmax: int,
    ncs: int,
    stages: set[str],
    probe_vectors: int,
    probe_seed: int,
    backend: str,
    dense_full_operators: bool,
    mainfield_kind: str,
    ignore_pfac: bool,
    connect_hemispheres: bool,
    probe_mode: str = "matrix-free",
    progress: bool = False,
) -> tuple[StageErrors, StageTimings]:
    def _log(msg: str) -> None:
        if progress:
            print(msg, flush=True)

    _log(
        f"[build] start N={nmax} M={mmax} Ncs={ncs} "
        f"(mainfield={mainfield_kind}, ignore_pfac={ignore_pfac}, "
        f"connect_hemispheres={connect_hemispheres})"
    )
    t0 = time.perf_counter()
    sh_basis = SHBasis(nmax, mmax)
    state_st = _build_state(
        SimulationMode.SPECTRAL_TRANSFORM_CS,
        nmax=nmax,
        mmax=mmax,
        ncs=ncs,
        backend=backend,
        dense_full_operators=dense_full_operators,
        mainfield_kind=mainfield_kind,
        ignore_pfac=ignore_pfac,
        connect_hemispheres=connect_hemispheres,
    ).state
    state_cs = _build_state(
        SimulationMode.CS_DOMINANT,
        nmax=nmax,
        mmax=mmax,
        ncs=ncs,
        backend=backend,
        dense_full_operators=dense_full_operators,
        mainfield_kind=mainfield_kind,
        ignore_pfac=ignore_pfac,
        connect_hemispheres=connect_hemispheres,
    ).state
    t_build = time.perf_counter() - t0
    _log(f"[build] done in {t_build:.2f}s")

    a_st, b_st = _scalar_transfer_operators(state_st, sh_basis)
    a_cs, b_cs = _scalar_transfer_operators(state_cs, sh_basis)
    a2_st, b2_st = _lift_two_channel(a_st), _lift_two_channel(b_st)
    a2_cs, b2_cs = _lift_two_channel(a_cs), _lift_two_channel(b_cs)
    n_sh = int(a_st.shape[1])
    n_sh2 = int(a2_st.shape[1])

    t1 = time.perf_counter()
    psi_to_e_err = np.nan
    if "psi_to_e" in stages:
        if probe_mode == "dense":
            _log("[psi_to_e] densify operators...")
            psi_to_e_st = np.asarray(to_dense(as_linear_map(state_st.toroidal_to_E_coeffs)))
            psi_to_e_cs = np.asarray(to_dense(as_linear_map(state_cs.toroidal_to_E_coeffs)))

            def _apply_psi_to_e_st(v_sh):
                x = a_st @ v_sh
                return b2_st @ (psi_to_e_st @ x)

            def _apply_psi_to_e_cs(v_sh):
                x = a_cs @ v_sh
                return b2_cs @ (psi_to_e_cs @ x)
        else:
            _log("[psi_to_e] matrix-free operators...")
            psi_to_e_st_op = as_linear_map(state_st.toroidal_to_E_coeffs)
            psi_to_e_cs_op = as_linear_map(state_cs.toroidal_to_E_coeffs)

            def _apply_psi_to_e_st(v_sh):
                x = a_st @ v_sh
                return b2_st @ np.asarray(psi_to_e_st_op.matvec(x))

            def _apply_psi_to_e_cs(v_sh):
                x = a_cs @ v_sh
                return b2_cs @ np.asarray(psi_to_e_cs_op.matvec(x))

        _log("[psi_to_e] probe...")

        psi_to_e_err = _probe_rel_error(
            _apply_psi_to_e_st,
            _apply_psi_to_e_cs,
            input_size=n_sh,
            n_probe=probe_vectors,
            seed=probe_seed + 101,
        )
    t_psi_to_e = time.perf_counter() - t1
    if "psi_to_e" in stages:
        _log(f"[psi_to_e] done in {t_psi_to_e:.2f}s (err={psi_to_e_err:.4e})")

    t2 = time.perf_counter()
    e_to_forcing_err = np.nan
    if "e_to_forcing" in stages:
        if probe_mode == "dense":
            _log("[e_to_forcing] densify operators...")
            e_to_forcing_st = np.asarray(
                to_dense(as_linear_map(state_st.toroidal_matrices.E_to_dtjr_forcing_matrix))
            )
            e_to_forcing_cs = np.asarray(
                to_dense(as_linear_map(state_cs.toroidal_matrices.E_to_dtjr_forcing_matrix))
            )
            e_to_forcing_st = e_to_forcing_st.reshape(e_to_forcing_st.shape[0], -1)
            e_to_forcing_cs = e_to_forcing_cs.reshape(e_to_forcing_cs.shape[0], -1)

            def _apply_e_to_forcing_st(v_sh2):
                x = a2_st @ v_sh2
                y = e_to_forcing_st @ x
                return b_st @ y

            def _apply_e_to_forcing_cs(v_sh2):
                x = a2_cs @ v_sh2
                y = e_to_forcing_cs @ x
                return b_cs @ y
        else:
            _log("[e_to_forcing] matrix-free operators...")

            def _apply_e_to_forcing_st(v_sh2):
                x = (a2_st @ v_sh2).reshape(2, -1)
                y = state_st.toroidal_matrices.compute_dtjr_forcing_from_E(x)
                return b_st @ np.asarray(y)

            def _apply_e_to_forcing_cs(v_sh2):
                x = (a2_cs @ v_sh2).reshape(2, -1)
                y = state_cs.toroidal_matrices.compute_dtjr_forcing_from_E(x)
                return b_cs @ np.asarray(y)

        _log("[e_to_forcing] probe...")

        e_to_forcing_err = _probe_rel_error(
            _apply_e_to_forcing_st,
            _apply_e_to_forcing_cs,
            input_size=n_sh2,
            n_probe=probe_vectors,
            seed=probe_seed + 211,
        )
    t_e_to_forcing = time.perf_counter() - t2
    if "e_to_forcing" in stages:
        _log(f"[e_to_forcing] done in {t_e_to_forcing:.2f}s (err={e_to_forcing_err:.4e})")

    t3 = time.perf_counter()
    dtjr_from_k_err = np.nan
    if "dtjr_from_k" in stages:
        if probe_mode == "dense":
            _log("[dtjr_from_k] build dense maps...")
            dtjr_from_k_st_map = np.asarray(
                state_st.toroidal_matrices._get_unconstrained_dtjr_map_cached(
                    weighting="none",
                    regularization_lambda=0.0,
                    penalty_operator=None,
                    penalty_scaling=0.0,
                    hinv_rtol=0.0,
                )
            )
            dtjr_from_k_cs_map = np.asarray(
                state_cs.toroidal_matrices._get_unconstrained_dtjr_map_cached(
                    weighting="none",
                    regularization_lambda=0.0,
                    penalty_operator=None,
                    penalty_scaling=0.0,
                    hinv_rtol=0.0,
                )
            )

            def _apply_dtjr_from_k_st(v_sh):
                x = a_st @ v_sh
                y = dtjr_from_k_st_map @ x
                return b_st @ np.asarray(y)

            def _apply_dtjr_from_k_cs(v_sh):
                x = a_cs @ v_sh
                y = dtjr_from_k_cs_map @ x
                return b_cs @ np.asarray(y)
        else:
            _log("[dtjr_from_k] matrix-free solves...")

            def _apply_dtjr_from_k_st(v_sh):
                x = a_st @ v_sh
                y = state_st.toroidal_matrices.solve_dt_jr_physics(
                    rhs_physics=x,
                    weighting="none",
                    regularization_lambda=0.0,
                    penalty_operator=None,
                    penalty_scaling=0.0,
                    hinv_rtol=0.0,
                )
                return b_st @ np.asarray(y)

            def _apply_dtjr_from_k_cs(v_sh):
                x = a_cs @ v_sh
                y = state_cs.toroidal_matrices.solve_dt_jr_physics(
                    rhs_physics=x,
                    weighting="none",
                    regularization_lambda=0.0,
                    penalty_operator=None,
                    penalty_scaling=0.0,
                    hinv_rtol=0.0,
                )
                return b_cs @ np.asarray(y)

        _log("[dtjr_from_k] probe...")

        dtjr_from_k_err = _probe_rel_error(
            _apply_dtjr_from_k_st,
            _apply_dtjr_from_k_cs,
            input_size=n_sh,
            n_probe=probe_vectors,
            seed=probe_seed + 307,
        )
    t_dtjr_from_k = time.perf_counter() - t3
    if "dtjr_from_k" in stages:
        _log(f"[dtjr_from_k] done in {t_dtjr_from_k:.2f}s (err={dtjr_from_k_err:.4e})")

    t4 = time.perf_counter()
    jr_to_psi_err = np.nan
    if "jr_to_psi" in stages:
        m_imp_to_jr_st = state_st.geometry.get_potential_to_JS_operator("m_imp", mode=None)
        m_imp_to_jr_cs = state_cs.geometry.get_potential_to_JS_operator("m_imp", mode=None)
        if probe_mode == "dense":
            _log("[jr_to_psi] build dense maps...")
            jr_to_psi_st_map = np.asarray(
                state_st.toroidal_matrices._get_jr_to_psi_dense(m_imp_to_jr_st, use_pinning=False)
            )
            jr_to_psi_cs_map = np.asarray(
                state_cs.toroidal_matrices._get_jr_to_psi_dense(m_imp_to_jr_cs, use_pinning=False)
            )

            def _apply_jr_to_psi_st(v_sh):
                x = a2_st @ v_sh
                y = jr_to_psi_st_map @ x
                return b_st @ np.asarray(y)

            def _apply_jr_to_psi_cs(v_sh):
                x = a2_cs @ v_sh
                y = jr_to_psi_cs_map @ x
                return b_cs @ np.asarray(y)
        else:
            _log("[jr_to_psi] matrix-free maps...")

            def _apply_jr_to_psi_st(v_sh):
                x = a2_st @ v_sh
                y = state_st.toroidal_matrices.compute_rates(
                    x,
                    m_imp_to_jr_st,
                    use_pinning=False,
                )
                return b_st @ np.asarray(y)

            def _apply_jr_to_psi_cs(v_sh):
                x = a2_cs @ v_sh
                y = state_cs.toroidal_matrices.compute_rates(
                    x,
                    m_imp_to_jr_cs,
                    use_pinning=False,
                )
                return b_cs @ np.asarray(y)

        _log("[jr_to_psi] probe...")

        jr_to_psi_err = _probe_rel_error(
            _apply_jr_to_psi_st,
            _apply_jr_to_psi_cs,
            input_size=n_sh2,
            n_probe=probe_vectors,
            seed=probe_seed + 401,
        )
    t_jr_to_psi = time.perf_counter() - t4
    if "jr_to_psi" in stages:
        _log(f"[jr_to_psi] done in {t_jr_to_psi:.2f}s (err={jr_to_psi_err:.4e})")

    return (
        StageErrors(
            psi_to_e=psi_to_e_err,
            e_to_forcing=e_to_forcing_err,
            dtjr_from_k=dtjr_from_k_err,
            jr_to_psi=jr_to_psi_err,
        ),
        StageTimings(
            build_states=t_build,
            psi_to_e=t_psi_to_e,
            e_to_forcing=t_e_to_forcing,
            dtjr_from_k=t_dtjr_from_k,
            jr_to_psi=t_jr_to_psi,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nmax-list",
        default="6,8,10,12,14",
        help="Comma-separated Nmax values. Mmax=Nmax.",
    )
    parser.add_argument(
        "--ncs-ratio",
        type=float,
        default=1.75,
        help="Ncs ratio relative to Nmax (rounded up).",
    )
    parser.add_argument(
        "--ncs-min",
        type=int,
        default=10,
        help="Minimum Ncs.",
    )
    parser.add_argument(
        "--stages",
        default="psi_to_e,e_to_forcing,dtjr_from_k,jr_to_psi",
        help="Comma-separated stages to run.",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Print per-stage timing summary for each resolution.",
    )
    parser.add_argument(
        "--probe-vectors",
        type=int,
        default=4,
        help="Number of random vectors used for operator parity estimates.",
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=0,
        help="Random seed for probe vectors.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "jax"),
        default="numpy",
        help="Backend for state construction (default: numpy).",
    )
    parser.add_argument(
        "--dense-full-operators",
        action="store_true",
        help="Enable dense full operators in Dynamics for benchmarking.",
    )
    parser.add_argument(
        "--mainfield-kind",
        choices=("dipole", "igrf", "radial"),
        default="dipole",
        help="Main field used during benchmark initialization (default: dipole).",
    )
    parser.add_argument(
        "--ignore-pfac",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip PFAC matrix setup to keep toroidal stage benchmark lightweight.",
    )
    parser.add_argument(
        "--connect-hemispheres",
        action="store_true",
        default=False,
        help="Enable IH coupling during setup (default: off for speed).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print per-resolution stage progress.",
    )
    parser.add_argument(
        "--probe-mode",
        choices=("matrix-free", "dense"),
        default="matrix-free",
        help="Operator comparison mode (default: matrix-free).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nmax_values = [int(v) for v in args.nmax_list.split(",") if v.strip()]

    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}

    print("N  M  Ncs  psi_to_e  e_to_forcing  dtjr_from_k  jr_to_psi")
    print("-- -- ---- --------  ------------  -----------  ---------")
    for nmax in nmax_values:
        mmax = nmax
        ncs = max(args.ncs_min, int(np.ceil(args.ncs_ratio * nmax)))
        if ncs % 2 != 0:
            ncs += 1
        errors, timings = compute_stage_errors(
            nmax=nmax,
            mmax=mmax,
            ncs=ncs,
            stages=stages,
            probe_vectors=args.probe_vectors,
            probe_seed=args.probe_seed,
            backend=args.backend,
            dense_full_operators=bool(args.dense_full_operators),
            mainfield_kind=args.mainfield_kind,
            ignore_pfac=bool(args.ignore_pfac),
            connect_hemispheres=bool(args.connect_hemispheres),
            probe_mode=args.probe_mode,
            progress=bool(args.progress),
        )
        print(
            f"{nmax:2d} {mmax:2d} {ncs:4d} "
            f"{errors.psi_to_e:8.4f}  {errors.e_to_forcing:12.4f}  "
            f"{errors.dtjr_from_k:11.4f}  {errors.jr_to_psi:9.4f}"
        )
        if args.timings:
            print(
                "  timings[s]: "
                f"build={timings.build_states:.2f}, "
                f"psi_to_e={timings.psi_to_e:.2f}, "
                f"e_to_forcing={timings.e_to_forcing:.2f}, "
                f"dtjr_from_k={timings.dtjr_from_k:.2f}, "
                f"jr_to_psi={timings.jr_to_psi:.2f}"
            )


if __name__ == "__main__":
    main()
