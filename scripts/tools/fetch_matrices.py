"""Fetch dense poloidal operator blocks for the PFAC + IH-coupled path.

This script builds a minimal `Dynamics` object in the single-state induction
path, uses the same conductance input helper as the default run setup, and
extracts dense poloidal matrices:

- ``u -> dm_ind/dt``
- ``jr -> dm_ind/dt``  (includes PFAC + interhemispheric `m_imp` solve)
- ``m_ind -> dm_ind/dt``

No external data files or time integration are required.
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np

from pynamit.data import get_conductance_inputs
from pynamit.math.constants import RE
from pynamit.math.linear_map import as_linear_map
from pynamit.simulation.dynamics import Dynamics
from pynamit.simulation.geometry_utils import to_dense


def build_dynamics(
    *,
    nmax: int,
    mmax: int,
    ncs: int,
    mainfield_kind: str,
    rm_re: float | None,
) -> Dynamics:
    """Build minimal `Dynamics` configured for PFAC + IH matrix extraction."""
    RI = RE + 110.0e3
    RM = None if rm_re is None else float(rm_re) * RE
    dynamics = Dynamics(
        filename_prefix="poloidal_dense_mats",
        Nmax=nmax,
        Mmax=mmax,
        Ncs=ncs,
        RI=RI,
        RM=RM,
        mainfield_kind=mainfield_kind,
        ignore_PFAC=False,
        connect_hemispheres=True,
        benchmark_mode=True,
        # Keep this script focused on dense matrix extraction, not iterative solves.
        dense_full_operators=True,
    )

    grid = dynamics.state.geometry.grid
    date = datetime.datetime(2001, 5, 12, 21, 45)
    hall, pedersen, cond_lat, cond_lon = get_conductance_inputs(date, grid.lat, grid.lon, None)
    dynamics.set_conductance(hall, pedersen, lat=cond_lat, lon=cond_lon)
    dynamics.state.update(dynamics.input_manager, time=0.0, interpolation=False)
    return dynamics


def fetch_poloidal_dense_matrices(
    dynamics: Dynamics,
    *,
    df_only: bool = False,
) -> dict[str, np.ndarray]:
    """Return dense poloidal matrices with PFAC + IH coupling active.

    If ``df_only`` is true, returns maps into the div-free electric component
    (``E_df``) instead of the time-derivative maps ``dm_ind/dt``.
    """
    st = dynamics.state

    scale = float(st.poloidal_matrices.E_df_to_d_m_ind_dt)
    e_df_from_e = np.asarray(st.E_coeffs_to_E_df_matrix, dtype=float)
    dt_m_ind_from_E = scale * e_df_from_e

    # Wind-input basis -> E coefficients (dense).
    e_from_u = np.asarray(to_dense(as_linear_map(st.u_coeffs_to_E_coeffs)), dtype=float)

    # jr-input basis -> m_imp (PFAC + IH constrained solve) -> E coefficients.
    jr_input_basis = dynamics.interpolation_bases["jr"]
    m_imp_from_jr = np.asarray(st.get_m_imp_from_jr_matrix(input_basis=jr_input_basis), dtype=float)
    e_from_mimp = np.asarray(to_dense(as_linear_map(st.m_imp_to_E_coeffs)), dtype=float)
    e_from_jr = e_from_mimp @ m_imp_from_jr

    if df_only:
        return {
            "edf_from_u": np.asarray(e_df_from_e @ e_from_u, dtype=float),
            "edf_from_jr": np.asarray(e_df_from_e @ e_from_jr, dtype=float),
            "edf_from_mind": np.asarray(st.m_ind_to_E_df_matrix, dtype=float),
        }

    dt_m_ind_from_u = dt_m_ind_from_E @ e_from_u
    dt_m_ind_from_jr = dt_m_ind_from_E @ e_from_jr
    dt_m_ind_from_m_ind = scale * np.asarray(st.m_ind_to_E_df_matrix, dtype=float)

    return {
        "dt_m_ind_from_u": np.asarray(dt_m_ind_from_u, dtype=float),
        "dt_m_ind_from_jr": np.asarray(dt_m_ind_from_jr, dtype=float),
        "dt_m_ind_from_m_ind": np.asarray(dt_m_ind_from_m_ind, dtype=float),
    }


def _print_summary(mats: dict[str, np.ndarray]) -> None:
    if {"edf_from_u", "edf_from_jr", "edf_from_mind"}.issubset(mats):
        keys = ["edf_from_u", "edf_from_jr", "edf_from_mind"]
    else:
        keys = ["dt_m_ind_from_u", "dt_m_ind_from_jr", "dt_m_ind_from_m_ind"]
    for key in keys:
        mat = np.asarray(mats[key])
        print(
            f"{key:16s} shape={mat.shape} "
            f"norm={np.linalg.norm(mat):.6e} "
            f"maxabs={np.max(np.abs(mat)):.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dense poloidal matrices for PFAC + IH coupling with minimal setup."
    )
    parser.add_argument("--nmax", type=int, default=12)
    parser.add_argument("--mmax", type=int, default=12)
    parser.add_argument("--ncs", type=int, default=22)
    parser.add_argument(
        "--mainfield-kind",
        type=str,
        default="dipole",
        choices=["dipole", "igrf", "radial"],
    )
    parser.add_argument(
        "--rm-over-re",
        type=float,
        default=None,
        help="Optional magnetospheric boundary radius in Earth radii (enables RM shielding).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional .npz path for saving the extracted dense matrices.",
    )
    parser.add_argument(
        "--df-only",
        action="store_true",
        help="Return E_df maps instead of dm_ind/dt maps.",
    )
    args = parser.parse_args()
    dyn = build_dynamics(
        nmax=args.nmax,
        mmax=args.mmax,
        ncs=args.ncs,
        mainfield_kind=args.mainfield_kind,
        rm_re=args.rm_over_re,
    )
    mats = fetch_poloidal_dense_matrices(dyn, df_only=bool(args.df_only))
    _print_summary(mats)

    if args.out:
        np.savez_compressed(args.out, **mats)
        print(f"Saved matrices to {args.out}")


if __name__ == "__main__":
    main()
