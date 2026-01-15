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

    SPECTRAL_TRANSFORM : str
        Alias for SPECTRAL_TRANSFORM_CS (backward compatibility).
    """
    PURE_SPECTRAL = "pure_spectral"
    SPECTRAL_TRANSFORM_CS = "spectral_transform_cs"
    SPECTRAL_TRANSFORM_GL = "spectral_transform_gl"
    CS_DOMINANT = "cs_dominant"
    # Backward compatibility alias
    SPECTRAL_TRANSFORM = "spectral_transform_cs"


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
    pure_spectral : bool
        Deprecated flag for pure spectral mode.
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
    vector_jr: bool = True
    vector_Br: bool = True
    vector_conductance: bool = True
    vector_u: bool = True
    t0: str = "2020-01-01 00:00:00"
    save_steady_states: bool = True
    integrator: Literal["euler", "exponential"] = "euler"
    backend: Union[Literal["auto", "numpy", "jax"], bool] = "auto"
    filename_prefix: str = "simulation"
    simulation_mode: SimulationMode = SimulationMode.SPECTRAL_TRANSFORM_CS
    least_squares_solver: str = "cg"
    m_imp_regularization_lambda: float = 0.0

    # Deprecated / Computed fields
    solution_basis_kind: Literal["SH", "CS"] = "SH"
    pure_spectral: bool = False

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

        # Serialize Simulation Mode
        attrs["simulation_mode"] = self.simulation_mode.value
        attrs["least_squares_solver"] = self.least_squares_solver

        # Deprecated Serialization (for consistency)
        attrs["pure_spectral"] = int(self.pure_spectral)

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

        # Handle legacy pure_spectral override if present and mode not explicitly set
        if "pure_spectral" in attrs and "simulation_mode" not in attrs:
            if bool(attrs["pure_spectral"]):
                sim_mode = SimulationMode.PURE_SPECTRAL

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
            pure_spectral=bool(get("pure_spectral", defaults.pure_spectral)),
        )
