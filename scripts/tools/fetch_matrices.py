"""Fetch dense model operator blocks used by the simulation.

This script builds a minimal ``Simulation`` object, sets one conductance
snapshot, and extracts dense matrices through
``ElectrodynamicResponse``. The accessors use the same operators and
response matrices as the time-integration path.
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np
from kompe.constants import EARTH_RADIUS_M
from kompe.math import block_until_ready, to_numpy

from pynamit.external_input_contracts import ExternalInputRequest
from pynamit.external_inputs import get_conductance_inputs
from pynamit.simulation.api import Simulation


def build_simulation(
    *,
    nmax: int,
    mmax: int,
    ncs: int,
    main_field_kind: str,
    rm_re: float | None,
    horizontal_basis_kind: str,
    enable_pfac_coupling: bool,
    enable_interhemispheric_coupling: bool,
    run_directory: str | None,
    artifact_storage: str,
    least_squares_solver: str,
) -> Simulation:
    """Build a minimal simulation for matrix extraction."""
    RI = EARTH_RADIUS_M + 110.0e3
    RM = None if rm_re is None else float(rm_re) * EARTH_RADIUS_M
    simulation = Simulation(
        run_directory=run_directory,
        Nmax=nmax,
        Mmax=mmax,
        Ncs=ncs,
        RI=RI,
        RM=RM,
        main_field_kind=main_field_kind,
        enable_pfac_coupling=enable_pfac_coupling,
        enable_interhemispheric_coupling=enable_interhemispheric_coupling,
        least_squares_solver=least_squares_solver,
        horizontal_basis_kind=horizontal_basis_kind,
        artifact_storage=artifact_storage,
    )

    grid = simulation.geometry.model_grid
    date = datetime.datetime(2001, 5, 12, 21, 45)
    geo_lat, geo_lon = simulation.geometry.main_field.model_to_geo_coordinates(
        grid.lat, grid.lon, event_time=date
    )
    request = ExternalInputRequest.from_model_coordinates(
        grid.lat,
        grid.lon,
        geographic_lat=geo_lat,
        geographic_lon=geo_lon,
        coordinate_system=simulation.geometry.main_field.horizontal_coordinate_system,
        model_epoch=simulation.geometry.main_field.epoch,
        grid_id="matrix-extraction-model-grid",
    )
    hall, pedersen, _, _ = get_conductance_inputs(date, None, None, None, request=request)
    simulation.set_conductance(hall, pedersen, lat=grid.lat, lon=grid.lon)
    simulation.response.activate_inputs_at_time(
        simulation.run_data.input_series, time=0.0, interpolation=False
    )
    return simulation


def fetch_model_dense_matrices(
    simulation: Simulation, *, df_only: bool = False, include_boundary_Br: bool = True
) -> dict[str, np.ndarray]:
    """Return dense matrices from the active simulation response."""
    response = simulation.response
    matrices = (
        response.E_df_matrices(include_boundary_Br=include_boundary_Br)
        if df_only
        else response.induced_Br_rate_matrices(include_boundary_Br=include_boundary_Br)
    )
    return {
        key: np.asarray(to_numpy(block_until_ready(matrix))) for key, matrix in matrices.items()
    }


def _print_summary(matrices: dict[str, np.ndarray]) -> None:
    order = (
        "d_induced_Br_dt_from_u",
        "d_induced_Br_dt_from_boundary_jr",
        "d_induced_Br_dt_from_boundary_Br",
        "d_induced_Br_dt_from_induced_Br",
        "E_df_from_u",
        "E_df_from_boundary_jr",
        "E_df_from_boundary_Br",
        "E_df_from_induced_Br",
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
        description="Build dense matrices from the active simulation operators."
    )
    parser.add_argument("--nmax", type=int, default=12)
    parser.add_argument("--mmax", type=int, default=12)
    parser.add_argument("--ncs", type=int, default=22)
    parser.add_argument(
        "--main_field-kind", type=str, default="dipole", choices=["dipole", "igrf", "radial"]
    )
    parser.add_argument(
        "--rm-over-re",
        type=float,
        default=None,
        help="Optional magnetospheric boundary radius in Earth radii.",
    )
    parser.add_argument(
        "--horizontal-basis-kind", type=str, default="SH", choices=["SH", "CS", "sh", "cs"]
    )
    parser.add_argument(
        "--disable-pfac-coupling",
        action="store_false",
        dest="enable_pfac_coupling",
        default=True,
        help="Disable PFAC contribution when assembling the model operators.",
    )
    parser.add_argument(
        "--disable-interhemispheric-coupling",
        action="store_false",
        dest="enable_interhemispheric_coupling",
        default=True,
        help="Disable interhemispheric current and electric-field constraints.",
    )
    parser.add_argument(
        "--least-squares-solver",
        type=str,
        default="normal_pinv",
        help="Least-squares solver used for toroidal-potential responses.",
    )
    parser.add_argument(
        "--artifact-storage",
        type=str,
        default="netcdf",
        choices=["auto", "netcdf", "zarr"],
        help="Storage backend for the temporary Simulation artifacts.",
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
        "--df-only", action="store_true", help="Return E_df maps instead of d(induced_Br)/dt maps."
    )
    parser.add_argument(
        "--exclude-br",
        action="store_true",
        help="Do not include Br forcing matrices when RM is configured.",
    )
    args = parser.parse_args()

    simulation = build_simulation(
        nmax=args.nmax,
        mmax=args.mmax,
        ncs=args.ncs,
        main_field_kind=args.main_field_kind,
        rm_re=args.rm_over_re,
        horizontal_basis_kind=args.horizontal_basis_kind,
        enable_pfac_coupling=bool(args.enable_pfac_coupling),
        enable_interhemispheric_coupling=bool(args.enable_interhemispheric_coupling),
        run_directory=args.run_directory,
        artifact_storage=args.artifact_storage,
        least_squares_solver=args.least_squares_solver,
    )
    matrices = fetch_model_dense_matrices(
        simulation, df_only=bool(args.df_only), include_boundary_Br=not bool(args.exclude_br)
    )
    _print_summary(matrices)

    if args.out:
        np.savez_compressed(args.out, **matrices)
        print(f"Saved matrices to {args.out}")


if __name__ == "__main__":
    main()
