"""Fetch dense poloidal operator blocks used by the simulation.

This script builds a minimal ``Dynamics`` object, sets one conductance
snapshot, and extracts dense matrices through ``State`` accessors.  The
accessors use the same operators and response matrices as the time
integration path.
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np

from pynamit.external_inputs import get_conductance_inputs
from pynamit.math.constants import RE
from pynamit.simulation.dynamics import Dynamics


def build_dynamics(
    *,
    nmax: int,
    mmax: int,
    ncs: int,
    mainfield_kind: str,
    rm_re: float | None,
    horizontal_basis_kind: str,
    ignore_pfac: bool,
    connect_hemispheres: bool,
    run_directory: str | None,
    artifact_storage: str,
    least_squares_solver: str,
) -> Dynamics:
    """Build a minimal ``Dynamics`` configured for matrix extraction."""
    RI = RE + 110.0e3
    RM = None if rm_re is None else float(rm_re) * RE
    dynamics = Dynamics(
        run_directory=run_directory,
        Nmax=nmax,
        Mmax=mmax,
        Ncs=ncs,
        RI=RI,
        RM=RM,
        mainfield_kind=mainfield_kind,
        ignore_PFAC=ignore_pfac,
        connect_hemispheres=connect_hemispheres,
        least_squares_solver=least_squares_solver,
        horizontal_basis_kind=horizontal_basis_kind,
        artifact_storage=artifact_storage,
    )

    grid = dynamics.state.geometry.grid
    date = datetime.datetime(2001, 5, 12, 21, 45)
    hall, pedersen, cond_lat, cond_lon = get_conductance_inputs(
        date, grid.lat, grid.lon, None
    )
    dynamics.set_conductance(hall, pedersen, lat=cond_lat, lon=cond_lon)
    dynamics.state.update(dynamics.input_timeseries, time=0.0, interpolation=False)
    return dynamics


def fetch_poloidal_dense_matrices(
    dynamics: Dynamics, *, df_only: bool = False, include_Br: bool = True
) -> dict[str, np.ndarray]:
    """Return dense matrices from the active simulation state."""
    return dynamics.state.get_poloidal_model_matrices(
        df_only=df_only, include_Br=include_Br
    )


def _print_summary(matrices: dict[str, np.ndarray]) -> None:
    order = (
        "dt_m_ind_from_u",
        "dt_m_ind_from_jr",
        "dt_m_ind_from_Br",
        "dt_m_ind_from_m_ind",
        "edf_from_u",
        "edf_from_jr",
        "edf_from_Br",
        "edf_from_m_ind",
    )
    for key in [key for key in order if key in matrices]:
        matrix = np.asarray(matrices[key])
        print(
            f"{key:20s} shape={matrix.shape} "
            f"norm={np.linalg.norm(matrix):.6e} "
            f"maxabs={np.max(np.abs(matrix)):.6e}"
        )


def main() -> None:
    """Run the dense matrix extraction CLI."""
    parser = argparse.ArgumentParser(
        description="Build dense poloidal matrices from the active simulation operators."
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
        help="Optional magnetospheric boundary radius in Earth radii.",
    )
    parser.add_argument(
        "--horizontal-basis-kind",
        type=str,
        default="SH",
        choices=["SH", "CS", "sh", "cs"],
    )
    parser.add_argument(
        "--ignore-pfac",
        action="store_true",
        help="Disable PFAC contribution when assembling the model operators.",
    )
    parser.add_argument(
        "--no-connect-hemispheres",
        action="store_false",
        dest="connect_hemispheres",
        default=True,
        help="Disable interhemispheric E-map feedback.",
    )
    parser.add_argument(
        "--least-squares-solver",
        type=str,
        default="normal_pinv",
        help="Least-squares solver used for m_imp response matrices.",
    )
    parser.add_argument(
        "--artifact-storage",
        type=str,
        default="netcdf",
        choices=["auto", "netcdf", "zarr"],
        help="Storage backend for the temporary Dynamics artifacts.",
    )
    parser.add_argument(
        "--run-directory",
        type=str,
        default=None,
        help="Optional run directory. Defaults to a temporary directory.",
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
        help="Return E_df maps instead of d(m_ind)/dt maps.",
    )
    parser.add_argument(
        "--exclude-br",
        action="store_true",
        help="Do not include Br forcing matrices when RM is configured.",
    )
    args = parser.parse_args()

    dynamics = build_dynamics(
        nmax=args.nmax,
        mmax=args.mmax,
        ncs=args.ncs,
        mainfield_kind=args.mainfield_kind,
        rm_re=args.rm_over_re,
        horizontal_basis_kind=args.horizontal_basis_kind,
        ignore_pfac=bool(args.ignore_pfac),
        connect_hemispheres=bool(args.connect_hemispheres),
        run_directory=args.run_directory,
        artifact_storage=args.artifact_storage,
        least_squares_solver=args.least_squares_solver,
    )
    matrices = fetch_poloidal_dense_matrices(
        dynamics, df_only=bool(args.df_only), include_Br=not bool(args.exclude_br)
    )
    _print_summary(matrices)

    if args.out:
        np.savez_compressed(args.out, **matrices)
        print(f"Saved matrices to {args.out}")


if __name__ == "__main__":
    main()
