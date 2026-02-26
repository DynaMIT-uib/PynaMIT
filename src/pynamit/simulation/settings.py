"""Simulation Settings Module.

This module contains configuration classes and enums for PynaMIT simulations:
- SimulationMode: Enum defining operational modes
- DynamicsSettings: Dataclass for simulation configuration
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Literal
from enum import Enum

import numpy as np
import xarray as xr

from pynamit.math.constants import RE


class SimulationMode(str, Enum):
    """Defines the operational mode of the simulation.

    Attributes
    ----------
    PURE_SPECTRAL : str
        "pure_spectral" - Fully analytical spectral method.
        Solver, differentiation, and products happen in spectral coefficients.
        Physics: Exact (subject to truncation), Ground Truth.
        Cost: O(N^4).

    SPECTRAL_TRANSFORM_CS : str
        "spectral_transform_cs" - Pseudo-Spectral method with Cubed-Sphere grid.
        Solver is spectral. Nonlinear products happen on CS grid.
        SH<->Grid transforms use pseudo-inverse (approximate).
        Physics: Fast, includes aliasing.
        Cost: O(N^3).

    SPECTRAL_TRANSFORM_GL : str
        "spectral_transform_gl" - Pseudo-Spectral method with Gauss-Legendre grid.
        Solver is spectral. Nonlinear products happen on GL grid.
        SH<->Grid transforms are exact via quadrature weights.
        Physics: Fast, machine-precision transforms.
        Cost: O(N^3).

    CS_DOMINANT : str
        "cs_dominant" - Cubed-Sphere Hybrid method.
        Solver is spectral (Laplacian inverse).
        differentiation and products happen via Finite Differences on Cubed Sphere.
        Physics: Local, fast parallel, numerical dissipation.
        In full_induction runs, toroidal closure assembly may use one
        auxiliary SH basis for basis-consistent closure semantics while
        retaining CS state/grid representation.

    """
    PURE_SPECTRAL = "pure_spectral"
    SPECTRAL_TRANSFORM_CS = "spectral_transform_cs"
    SPECTRAL_TRANSFORM_GL = "spectral_transform_gl"
    CS_DOMINANT = "cs_dominant"


# Safety margin for floating point errors
FLOAT_ERROR_MARGIN = 1e-6


@dataclass
class DynamicsSettings:
    """Settings for the Dynamics simulation.

    Parameters
    ----------
    Nmax : int
        Maximum spherical harmonic degree.
    Mmax : int
        Maximum spherical harmonic order.
    Ncs : int
        Cubed sphere grid resolution (N x N per face).
    RI : float
        Radius of the ionosphere in meters.
    RM : float, optional
        Radius of the magnetosphere in meters.
    mainfield_kind : str
        Type of main field model: "dipole", "igrf", or "radial".
    mainfield_epoch : int
        Epoch year for IGRF model.
    mainfield_B0 : float, optional
        Reference magnetic field strength.
    FAC_integration_steps : array-like
        Radial steps for FAC integration.
    ignore_PFAC : bool
        Whether to ignore PFAC calculations.
    connect_hemispheres : bool
        Whether to connect hemispheres via field lines.
    latitude_boundary : float
        Latitude boundary for hemisphere connection.
    ih_constraint_scaling : float
        Scaling for interhemispheric constraint.
    vector_jr : bool
        Use vector representation for radial current.
    vector_Br : bool
        Use vector representation for radial B field.
    vector_conductance : bool
        Use vector representation for conductance.
    vector_u : bool
        Use vector representation for wind.
    t0 : str
        Initial time string.
    save_steady_states : bool
        Whether to save steady state solutions.
    integrator : str
        Time integration method: "euler" or "exponential".
    backend : str
        Computation backend: "auto", "numpy", or "jax".
    filename_prefix : str
        Prefix for output files.
    simulation_mode : SimulationMode
        Operational mode of the simulation.
    least_squares_solver : str
        Solver type for least squares problems.
    m_imp_regularization_lambda : float
        Regularization parameter for imposed field.
    solution_basis_kind : str
        Basis for solution: "SH" or "CS".
    """

    Nmax: int = 20
    Mmax: int = 20
    Ncs: int = 30
    RI: float = RE + 110.0e3
    RM: Optional[float] = None
    mainfield_kind: Literal["dipole", "igrf", "radial"] = "dipole"
    mainfield_epoch: int = 2020
    mainfield_B0: Optional[float] = None
    FAC_integration_steps: Union[np.ndarray, List[float]] = field(
        default_factory=lambda: np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11)
    )
    ignore_PFAC: bool = False
    connect_hemispheres: bool = False
    latitude_boundary: float = 50.0
    ih_constraint_scaling: float = 1e-5
    apply_psi_gauge: bool = True
    apply_m_ind_gauge: bool = True
    magnetospheric_toroidal_lock: bool = False
    magnetospheric_poloidal_lock: bool = True
    northern_hemisphere_apex_constraints: bool = False
    vector_jr: bool = True
    vector_Br: bool = True
    vector_conductance: bool = True
    vector_u: bool = True
    t0: str = "2020-01-01 00:00:00"
    save_steady_states: bool = True
    integrator: Literal["euler", "exponential"] = "euler"
    backend: Union[Literal["auto", "numpy", "jax"], bool] = "auto"
    filename_prefix: str = "simulation"
    dynamics_mode: Literal["legacy", "full_induction"] = "legacy"
    simulation_mode: SimulationMode = SimulationMode.SPECTRAL_TRANSFORM_CS
    least_squares_solver: str = "lsmr"
    m_imp_regularization_lambda: float = 0.0
    # Weighting strategies for handling equatorial singularity (Br -> 0)
    toroidal_weighting: Literal["none", "linear", "quadratic"] = "none"
    poloidal_weighting: Literal["none", "linear", "quadratic"] = "none"
    # Preconditioner for least-squares solver
    least_squares_preconditioner: Optional[Literal["jacobi", "pinv"]] = "pinv"
    # Conductance input interpolation policy:
    # - legacy_eta_linear: convert Sigma->eta first, then interpolate eta (legacy behavior)
    # - sigma_linear: interpolate Sigma directly, then convert to eta at state update
    # - sigma_log: interpolate log(Sigma + floor), then convert to eta at state update
    conductance_interpolation_mode: Literal[
        "legacy_eta_linear", "sigma_linear", "sigma_log"
    ] = "sigma_log",
    # Floor used for sigma_log encoding and for robust Sigma->eta conversion in
    # non-legacy modes (denominator floor uses floor^2).
    conductance_interpolation_floor: float = 1e-3
    # Tikhonov regularization for toroidal system (only used in full_induction mode)
    toroidal_regularization_lambda: float = 1e-10
    # Force dense assembly/use of full linear evolution operators for both
    # legacy and full-induction dynamics paths.
    dense_full_operators: bool = False
    # Use SH fast input projection path on regular lat/lon grids when available.
    # Disabled by default to preserve legacy baseline behavior.
    enable_fast_input_path: bool = False
    # Exponential affine-step implementation (when integrator="exponential").
    # "expm" uses a dense matrix exponential on the augmented affine system and
    # therefore requires ``dense_full_operators=True`` when ``integrator="exponential"``.
    # "expm_multiply" uses expm_multiply. Combined with ``dense_full_operators``,
    # this yields either dense-action or matrix-free-action stepping.
    exponential_solver: Literal["expm", "expm_multiply"] = "expm"

    # Computed fields
    solution_basis_kind: Literal["SH", "CS"] = "SH"

    def to_dataset(self) -> xr.Dataset:
        """Convert settings to an xarray Dataset for storage."""
        attrs = asdict(self)
        # Handle types that might not serialize well or need specific handling
        attrs["RM"] = 0 if self.RM is None else self.RM
        attrs["mainfield_B0"] = 0 if self.mainfield_B0 is None else self.mainfield_B0
        attrs["ignore_PFAC"] = int(self.ignore_PFAC)
        attrs["connect_hemispheres"] = int(self.connect_hemispheres)
        attrs["vector_jr"] = int(self.vector_jr)
        attrs["vector_Br"] = int(self.vector_Br)
        attrs["vector_conductance"] = int(self.vector_conductance)
        attrs["vector_u"] = int(self.vector_u)
        attrs["save_steady_states"] = int(self.save_steady_states)
        attrs["northern_hemisphere_apex_constraints"] = int(self.northern_hemisphere_apex_constraints)
        attrs["apply_psi_gauge"] = int(self.apply_psi_gauge)
        attrs["apply_m_ind_gauge"] = int(self.apply_m_ind_gauge)
        attrs["magnetospheric_toroidal_lock"] = int(self.magnetospheric_toroidal_lock)
        attrs["magnetospheric_poloidal_lock"] = int(self.magnetospheric_poloidal_lock)
        attrs["dense_full_operators"] = int(self.dense_full_operators)
        attrs["enable_fast_input_path"] = int(self.enable_fast_input_path)
        attrs["exponential_solver"] = self.exponential_solver

        # Serialize Simulation Mode
        attrs["simulation_mode"] = self.simulation_mode.value
        attrs["least_squares_solver"] = self.least_squares_solver
        attrs["least_squares_preconditioner"] = self.least_squares_preconditioner
        attrs["toroidal_weighting"] = self.toroidal_weighting
        attrs["poloidal_weighting"] = self.poloidal_weighting
        attrs["conductance_interpolation_mode"] = self.conductance_interpolation_mode
        attrs["conductance_interpolation_floor"] = self.conductance_interpolation_floor
        # Remove backend as it is runtime configuration
        if "backend" in attrs:
            del attrs["backend"]
        if "filename_prefix" in attrs:
            del attrs["filename_prefix"]

        return xr.Dataset(attrs=attrs)

    @staticmethod
    def from_dataset(ds: xr.Dataset, defaults: "DynamicsSettings") -> "DynamicsSettings":
        """Create settings from a dataset, using defaults as a base."""
        attrs = ds.attrs

        # Helper to safely get and convert
        def get(key, default, converter=lambda x: x):
            return converter(attrs.get(key, default))

        # Handle Enum deserialization
        mode_str = get("simulation_mode", defaults.simulation_mode.value)
        try:
            sim_mode = SimulationMode(mode_str)
        except ValueError:
            sim_mode = defaults.simulation_mode

        exp_solver = get("exponential_solver", defaults.exponential_solver)
        if exp_solver == "dense_expm":
            exp_solver = "expm"

        return DynamicsSettings(
            simulation_mode=sim_mode,
            least_squares_solver=get("least_squares_solver", defaults.least_squares_solver),
            Nmax=get("Nmax", defaults.Nmax),
            Mmax=get("Mmax", defaults.Mmax),
            Ncs=get("Ncs", defaults.Ncs),
            RI=get("RI", defaults.RI),
            RM=get("RM", defaults.RM, lambda x: None if x == 0 else x),
            mainfield_kind=get("mainfield_kind", defaults.mainfield_kind),
            mainfield_epoch=get("mainfield_epoch", defaults.mainfield_epoch),
            mainfield_B0=get(
                "mainfield_B0", defaults.mainfield_B0, lambda x: None if x == 0 else x
            ),
            FAC_integration_steps=get("FAC_integration_steps", defaults.FAC_integration_steps),
            ignore_PFAC=bool(get("ignore_PFAC", defaults.ignore_PFAC)),
            connect_hemispheres=bool(get("connect_hemispheres", defaults.connect_hemispheres)),
            latitude_boundary=get("latitude_boundary", defaults.latitude_boundary),
            ih_constraint_scaling=get("ih_constraint_scaling", defaults.ih_constraint_scaling),
            apply_psi_gauge=bool(get("apply_psi_gauge", defaults.apply_psi_gauge)),
            apply_m_ind_gauge=bool(get("apply_m_ind_gauge", defaults.apply_m_ind_gauge)),
            magnetospheric_toroidal_lock=bool(
                get("magnetospheric_toroidal_lock", defaults.magnetospheric_toroidal_lock)
            ),
            magnetospheric_poloidal_lock=bool(
                get("magnetospheric_poloidal_lock", defaults.magnetospheric_poloidal_lock)
            ),
            northern_hemisphere_apex_constraints=bool(
                get("northern_hemisphere_apex_constraints", defaults.northern_hemisphere_apex_constraints)
            ),
            vector_jr=bool(get("vector_jr", defaults.vector_jr)),
            vector_Br=bool(get("vector_Br", defaults.vector_Br)),
            vector_conductance=bool(get("vector_conductance", defaults.vector_conductance)),
            vector_u=bool(get("vector_u", defaults.vector_u)),
            t0=get("t0", defaults.t0),
            save_steady_states=bool(get("save_steady_states", defaults.save_steady_states)),

            integrator=get("integrator", defaults.integrator),
            # Runtime fields not in file
            backend=defaults.backend,
            filename_prefix=get("filename_prefix", defaults.filename_prefix),
            solution_basis_kind=get("solution_basis_kind", defaults.solution_basis_kind),
            dynamics_mode=get("dynamics_mode", defaults.dynamics_mode),
            toroidal_weighting=get("toroidal_weighting", defaults.toroidal_weighting),
            poloidal_weighting=get("poloidal_weighting", defaults.poloidal_weighting),
            least_squares_preconditioner=get("least_squares_preconditioner", defaults.least_squares_preconditioner),
            conductance_interpolation_mode=get(
                "conductance_interpolation_mode",
                defaults.conductance_interpolation_mode,
            ),
            conductance_interpolation_floor=get(
                "conductance_interpolation_floor",
                defaults.conductance_interpolation_floor,
            ),
            m_imp_regularization_lambda=get("m_imp_regularization_lambda", defaults.m_imp_regularization_lambda),
            toroidal_regularization_lambda=get("toroidal_regularization_lambda", defaults.toroidal_regularization_lambda),
            dense_full_operators=bool(
                get("dense_full_operators", defaults.dense_full_operators)
            ),
            enable_fast_input_path=bool(
                get("enable_fast_input_path", defaults.enable_fast_input_path)
            ),
            exponential_solver=exp_solver,
        )
