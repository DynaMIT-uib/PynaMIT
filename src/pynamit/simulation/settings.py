"""Simulation Settings Module.

This module contains configuration classes and enums for PynaMIT simulations:
- SimulationMode: Enum defining operational modes
- DynamicsSettings: Dataclass for simulation configuration
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
from typing import Any, List, Mapping, Optional, Union, Literal
from enum import Enum
from difflib import get_close_matches

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


class StringChoiceEnum(str, Enum):
    """String enum with human-friendly ``str(...)`` behavior."""

    def __str__(self) -> str:
        return self.value


class MainfieldKind(StringChoiceEnum):
    DIPOLE = "dipole"
    IGRF = "igrf"
    RADIAL = "radial"


class IntegratorKind(StringChoiceEnum):
    EULER = "euler"
    EXPONENTIAL = "exponential"


class DynamicsMode(StringChoiceEnum):
    LEGACY = "legacy"
    FULL_INDUCTION = "full_induction"


class WeightingMode(StringChoiceEnum):
    NONE = "none"
    LINEAR = "linear"
    QUADRATIC = "quadratic"


class LLConstraintMode(StringChoiceEnum):
    AUTO = "auto"
    OFF = "off"
    SOFT = "soft"
    HARD = "hard"


class ConductanceInterpolationMode(StringChoiceEnum):
    LEGACY_ETA_LINEAR = "legacy_eta_linear"
    SIGMA_LINEAR = "sigma_linear"
    SIGMA_LOG = "sigma_log"


class ExponentialSolverKind(StringChoiceEnum):
    EXPM = "expm"
    EXPM_MULTIPLY = "expm_multiply"
    DENSE_EXPM = "dense_expm"


class SolutionBasisKind(StringChoiceEnum):
    SH = "SH"
    CS = "CS"


class ArtifactStorageKind(StringChoiceEnum):
    AUTO = "auto"
    NETCDF = "netcdf"
    ZARR = "zarr"


# Safety margin for floating point errors
FLOAT_ERROR_MARGIN = 1e-6


def _enum_values(enum_cls: type[StringChoiceEnum]) -> set[str]:
    """Return the canonical string values for a string enum."""
    return {member.value for member in enum_cls}


_VALID_MAINFIELD_KINDS = _enum_values(MainfieldKind)
_SCIPY_SOLVE_IVP_INTEGRATORS = {
    "rk23": "RK23",
    "rk45": "RK45",
    "dop853": "DOP853",
    "radau": "Radau",
    "bdf": "BDF",
    "lsoda": "LSODA",
}
_VALID_INTEGRATORS = _enum_values(IntegratorKind) | set(_SCIPY_SOLVE_IVP_INTEGRATORS)
_VALID_BACKENDS = {"auto", "numpy", "jax"}
_VALID_DYNAMICS_MODES = _enum_values(DynamicsMode)
_VALID_WEIGHTINGS = _enum_values(WeightingMode)
_VALID_PRECONDITIONERS = {"jacobi", "pinv"}
_VALID_CONDUCTANCE_INTERPOLATION_MODES = _enum_values(ConductanceInterpolationMode)
_VALID_EXPONENTIAL_SOLVERS = _enum_values(ExponentialSolverKind)
_VALID_SOLUTION_BASES = _enum_values(SolutionBasisKind)
_VALID_ARTIFACT_STORAGES = _enum_values(ArtifactStorageKind)


def _normalize_choice(name: str, value: Any, *, valid: set[str], allow_none: bool = False) -> Any:
    """Return normalized string choice or raise with a clear message."""
    if allow_none and value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be one of {sorted(valid)}"
            + (" or None" if allow_none else "")
            + f", got {value!r}."
        )
    normalized = str(value).strip()
    if normalized not in valid:
        suggestion = ""
        matches = get_close_matches(normalized, sorted(valid), n=1, cutoff=0.7)
        if matches:
            suggestion = f" Did you mean {matches[0]!r}?"
        raise ValueError(
            f"Invalid {name} {normalized!r}. Valid options: {sorted(valid)}"
            + (" or None." if allow_none else ".")
            + suggestion
        )
    return normalized


def _normalize_lower_choice(
    name: str, value: Any, *, valid: set[str], allow_none: bool = False
) -> Any:
    """Return lower-case normalized choice or raise with a clear message."""
    if isinstance(value, str):
        value = value.strip().lower()
    return _normalize_choice(name, value, valid=valid, allow_none=allow_none)


def _normalize_upper_choice(name: str, value: Any, *, valid: set[str]) -> str:
    """Return upper-case normalized choice or raise with a clear message."""
    if isinstance(value, str):
        value = value.strip().upper()
    return _normalize_choice(name, value, valid=valid)


def _coerce_enum_choice(
    enum_cls: type[StringChoiceEnum],
    name: str,
    value: Any,
    *,
    case: Literal["lower", "upper", "preserve"] = "preserve",
) -> StringChoiceEnum:
    """Return a normalized string enum instance with typo suggestions."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be one of {[member.value for member in enum_cls]}, got {value!r}."
        )
    normalized = value.strip()
    if case == "lower":
        normalized = normalized.lower()
    elif case == "upper":
        normalized = normalized.upper()
    try:
        return enum_cls(normalized)
    except ValueError:
        valid = [member.value for member in enum_cls]
        matches = get_close_matches(normalized, valid, n=1, cutoff=0.7)
        suggestion = f" Did you mean {matches[0]!r}?" if matches else ""
        raise ValueError(
            f"Invalid {name} {normalized!r}. Valid options: {valid}.{suggestion}"
        ) from None


def _normalize_integrator_choice(value: Any) -> IntegratorKind | str:
    """Return one canonical integrator choice.

    Built-in PynaMIT integrators remain ``IntegratorKind`` enum values.
    Supported SciPy ``solve_ivp`` methods are normalized to the canonical
    mixed-case strings expected by SciPy, e.g. ``"DOP853"``.
    """
    if isinstance(value, IntegratorKind):
        return value
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str):
        valid = [member.value for member in IntegratorKind] + list(
            _SCIPY_SOLVE_IVP_INTEGRATORS.values()
        )
        raise ValueError(f"integrator must be one of {valid}, got {value!r}.")

    normalized = value.strip()
    lowered = normalized.lower()
    if lowered in _enum_values(IntegratorKind):
        return IntegratorKind(lowered)
    if lowered in _SCIPY_SOLVE_IVP_INTEGRATORS:
        return _SCIPY_SOLVE_IVP_INTEGRATORS[lowered]

    valid = [member.value for member in IntegratorKind] + list(
        _SCIPY_SOLVE_IVP_INTEGRATORS.values()
    )
    matches = get_close_matches(normalized, valid, n=1, cutoff=0.7)
    if not matches:
        matches = get_close_matches(lowered, [item.lower() for item in valid], n=1, cutoff=0.7)
        suggestion = ""
        if matches:
            lower_to_canonical = {item.lower(): item for item in valid}
            suggestion = f" Did you mean {lower_to_canonical[matches[0]]!r}?"
        else:
            suggestion = ""
    else:
        suggestion = f" Did you mean {matches[0]!r}?"
    raise ValueError(f"Invalid integrator {normalized!r}. Valid options: {valid}.{suggestion}")


def _normalize_backend_choice(value: Any) -> str:
    """Return canonical backend string."""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "jax" if value else "numpy"
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise ValueError(
            f"backend must be one of {sorted(_VALID_BACKENDS)} or a boolean, got {value!r}."
        )
    normalized = value.strip().lower()
    if normalized == "np":
        normalized = "numpy"
    return _normalize_choice("backend", normalized, valid=_VALID_BACKENDS)


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
    ll_constraint_mode : str
        Low-latitude ionospheric compatibility policy:
        "auto", "off", "soft", or "hard".
    magnetospheric_shielding : bool
        If True, apply the RM shielding closure to induced above-``R_M``
        poloidal response pathways, including the dynamic ``psi -> Ve`` PFAC
        response.
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
        Time integration method: "euler", "exponential", or one supported
        SciPy ``solve_ivp`` method such as "DOP853" or "RK45".
    backend : str
        Computation backend: "auto", "numpy", or "jax".
    run_directory : str, optional
        Directory for persisted run files. If omitted, the caller decides
        whether to use a temporary run directory or an explicit project-local
        directory.
    artifact_storage : str
        Preferred on-disk artifact format for new saved datasets:
        "auto", "netcdf", or "zarr".
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
    mainfield_kind: MainfieldKind = MainfieldKind.DIPOLE
    mainfield_epoch: int = 2020
    mainfield_B0: Optional[float] = None
    FAC_integration_steps: Union[np.ndarray, List[float]] = field(
        default_factory=lambda: np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11)
    )
    ignore_PFAC: bool = False
    connect_hemispheres: bool = False
    latitude_boundary: float = 50.0
    ih_constraint_scaling: float = 1e-5
    ll_constraint_mode: LLConstraintMode = LLConstraintMode.AUTO
    apply_psi_gauge: bool = True
    apply_m_ind_gauge: bool = True
    apply_m_imp_gauge: bool = True
    magnetospheric_shielding: bool = True
    northern_hemisphere_apex_constraints: bool = False
    vector_jr: bool = True
    vector_Br: bool = True
    vector_conductance: bool = True
    vector_u: bool = True
    t0: str = "2020-01-01 00:00:00"
    save_steady_states: bool = True
    integrator: IntegratorKind | str = IntegratorKind.EULER
    backend: Union[Literal["auto", "numpy", "jax"], bool] = "auto"
    run_directory: Optional[str] = None
    artifact_storage: ArtifactStorageKind = ArtifactStorageKind.AUTO
    dynamics_mode: DynamicsMode = DynamicsMode.LEGACY
    simulation_mode: SimulationMode = SimulationMode.SPECTRAL_TRANSFORM_CS
    least_squares_solver: str = "lsmr"
    m_imp_regularization_lambda: float = 0.0
    # Weighting strategies for handling equatorial singularity (Br -> 0)
    toroidal_weighting: WeightingMode = WeightingMode.NONE
    poloidal_weighting: WeightingMode = WeightingMode.NONE
    # Preconditioner for least-squares solver
    least_squares_preconditioner: Optional[Literal["jacobi", "pinv"]] = "pinv"
    # Conductance input interpolation policy:
    # - legacy_eta_linear: convert Sigma->eta first, then interpolate eta (legacy behavior)
    # - sigma_linear: interpolate Sigma directly, then convert to eta at state update
    # - sigma_log: interpolate log(Sigma + floor), then convert to eta at state update
    conductance_interpolation_mode: ConductanceInterpolationMode = (
        ConductanceInterpolationMode.LEGACY_ETA_LINEAR
    )
    # Floor used for sigma_log encoding and for robust Sigma->eta conversion in
    # non-legacy modes (denominator floor uses floor^2).
    conductance_interpolation_floor: float = 1e-3
    # Tikhonov regularization for toroidal system (only used in full_induction mode)
    toroidal_regularization_lambda: float = 1e-10
    # Force dense assembly/use of full linear evolution operators for both
    # legacy and full-induction dynamics paths.
    dense_full_operators: bool = False
    # Optional diagnostics for coupled full-induction near-null modes.
    # When enabled, the run computes a near-null basis for the coupled operator
    # and warns if the forcing projects strongly onto it.
    induction_null_diagnostics: bool = False
    induction_null_svd_rtol: float = 1e-8
    induction_null_warn_ratio: float = 0.5
    # Use SH fast input projection path on regular lat/lon grids when available.
    # Disabled by default to preserve legacy baseline behavior.
    enable_fast_input_path: bool = False
    # Exponential affine-step implementation (when integrator="exponential").
    # "expm" uses a dense matrix exponential on the augmented affine system and
    # therefore requires ``dense_full_operators=True`` when ``integrator="exponential"``.
    # "expm_multiply" uses expm_multiply. Combined with ``dense_full_operators``,
    # this yields either dense-action or matrix-free-action stepping.
    exponential_solver: ExponentialSolverKind = ExponentialSolverKind.EXPM

    # Computed fields
    solution_basis_kind: SolutionBasisKind = SolutionBasisKind.SH

    def __post_init__(self) -> None:
        """Normalize derived/defaulted settings once at construction time."""
        if isinstance(self.simulation_mode, str):
            self.simulation_mode = SimulationMode(self.simulation_mode)
        elif not isinstance(self.simulation_mode, SimulationMode):
            raise ValueError(
                f"simulation_mode must be one of {[mode.value for mode in SimulationMode]}, "
                f"got {self.simulation_mode!r}."
            )

        # Guard against accidental tuple defaults from trailing commas in older
        # code paths or serialized settings adapters.
        if isinstance(self.conductance_interpolation_mode, tuple):
            if len(self.conductance_interpolation_mode) != 1:
                raise ValueError(
                    "conductance_interpolation_mode tuple input must have length 1, "
                    f"got {self.conductance_interpolation_mode!r}."
                )
            self.conductance_interpolation_mode = str(self.conductance_interpolation_mode[0])

        self.mainfield_kind = _coerce_enum_choice(
            MainfieldKind, "mainfield_kind", self.mainfield_kind, case="lower"
        )
        self.integrator = _normalize_integrator_choice(self.integrator)
        self.backend = _normalize_backend_choice(self.backend)
        self.dynamics_mode = _coerce_enum_choice(
            DynamicsMode, "dynamics_mode", self.dynamics_mode, case="lower"
        )
        self.toroidal_weighting = _coerce_enum_choice(
            WeightingMode, "toroidal_weighting", self.toroidal_weighting, case="lower"
        )
        self.poloidal_weighting = _coerce_enum_choice(
            WeightingMode, "poloidal_weighting", self.poloidal_weighting, case="lower"
        )
        self.ll_constraint_mode = _coerce_enum_choice(
            LLConstraintMode, "ll_constraint_mode", self.ll_constraint_mode, case="lower"
        )
        self.least_squares_preconditioner = _normalize_lower_choice(
            "least_squares_preconditioner",
            self.least_squares_preconditioner,
            valid=_VALID_PRECONDITIONERS,
            allow_none=True,
        )
        self.conductance_interpolation_mode = _coerce_enum_choice(
            ConductanceInterpolationMode,
            "conductance_interpolation_mode",
            self.conductance_interpolation_mode,
            case="lower",
        )
        self.exponential_solver = _coerce_enum_choice(
            ExponentialSolverKind, "exponential_solver", self.exponential_solver, case="lower"
        )
        self.solution_basis_kind = _coerce_enum_choice(
            SolutionBasisKind, "solution_basis_kind", self.solution_basis_kind, case="upper"
        )
        self.artifact_storage = _coerce_enum_choice(
            ArtifactStorageKind, "artifact_storage", self.artifact_storage, case="lower"
        )

        from pynamit.math.least_squares_solver import LeastSquaresSolver

        self.least_squares_solver = _normalize_lower_choice(
            "least_squares_solver",
            self.least_squares_solver,
            valid=set(LeastSquaresSolver.VALID_SOLVERS),
        )

        if self.RM == 0:
            self.RM = None

        self.conductance_interpolation_floor = float(
            max(self.conductance_interpolation_floor, 0.0)
        )
        self.induction_null_svd_rtol = float(self.induction_null_svd_rtol)
        if self.induction_null_svd_rtol < 0.0:
            raise ValueError("induction_null_svd_rtol must be >= 0.0.")
        self.induction_null_warn_ratio = float(self.induction_null_warn_ratio)
        if not (0.0 <= self.induction_null_warn_ratio <= 1.0):
            raise ValueError("induction_null_warn_ratio must be between 0.0 and 1.0.")

        if self.simulation_mode == SimulationMode.CS_DOMINANT:
            self.solution_basis_kind = SolutionBasisKind.CS

        # CS-dominant electrostatic solves need mild regularization near the
        # magnetic equator when the user has not chosen one explicitly.
        if (
            self.simulation_mode == SimulationMode.CS_DOMINANT
            and self.m_imp_regularization_lambda == 0.0
        ):
            self.m_imp_regularization_lambda = 1e-4

        # Full-induction defaults are stability defaults, not user-facing
        # behavior changes. Keep explicit user choices, only fill "none"/0 cases.
        if self.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            if self.toroidal_weighting == WeightingMode.NONE:
                self.toroidal_weighting = WeightingMode.QUADRATIC
            if self.poloidal_weighting == WeightingMode.NONE:
                self.poloidal_weighting = WeightingMode.QUADRATIC
            if self.toroidal_regularization_lambda == 0.0:
                self.toroidal_regularization_lambda = 1e-10

        if self.exponential_solver == ExponentialSolverKind.DENSE_EXPM:
            self.exponential_solver = ExponentialSolverKind.EXPM

        if self.exponential_solver not in {
            ExponentialSolverKind.EXPM,
            ExponentialSolverKind.EXPM_MULTIPLY,
        }:
            raise ValueError(
                "exponential_solver must be one of {'expm', 'expm_multiply'}, "
                f"got {self.exponential_solver!r}."
            )

        if (
            self.integrator == IntegratorKind.EXPONENTIAL
            and self.dynamics_mode == DynamicsMode.FULL_INDUCTION
            and self.exponential_solver == ExponentialSolverKind.EXPM
            and not self.dense_full_operators
        ):
            raise ValueError(
                "dynamics_mode='full_induction' with integrator='exponential' and "
                "exponential_solver='expm' requires dense_full_operators=True."
            )

    @classmethod
    def coerce(cls, settings: Optional[Any] = None, /, **overrides: Any) -> "DynamicsSettings":
        """Return normalized settings from a full or partial settings object."""
        field_names = {field_def.name for field_def in fields(cls)}
        values: dict[str, Any] = {}
        unknown_overrides = [key for key in overrides if key not in field_names]

        if unknown_overrides:
            unknown = sorted(unknown_overrides)
            detail_parts = []
            for key in unknown:
                matches = get_close_matches(key, sorted(field_names), n=1, cutoff=0.7)
                if matches:
                    detail_parts.append(f"{key!r} (did you mean {matches[0]!r}?)")
                else:
                    detail_parts.append(repr(key))
            raise TypeError(f"Unknown DynamicsSettings override(s): {', '.join(detail_parts)}.")

        if settings is not None:
            if isinstance(settings, Mapping):
                for name in field_names:
                    if name in settings:
                        value = settings[name]
                        if name == "FAC_integration_steps" and value is None:
                            continue
                        values[name] = value
            else:
                for name in field_names:
                    if hasattr(settings, name):
                        value = getattr(settings, name)
                        if name == "FAC_integration_steps" and value is None:
                            continue
                        values[name] = value
        for key, value in overrides.items():
            values[key] = value

        return cls(**values)

    def to_dataset(self) -> xr.Dataset:
        """Convert settings to an xarray Dataset for storage."""
        attrs = asdict(self)
        for key, value in list(attrs.items()):
            if isinstance(value, Enum):
                attrs[key] = value.value
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
        attrs["northern_hemisphere_apex_constraints"] = int(
            self.northern_hemisphere_apex_constraints
        )
        attrs["apply_psi_gauge"] = int(self.apply_psi_gauge)
        attrs["apply_m_ind_gauge"] = int(self.apply_m_ind_gauge)
        attrs["apply_m_imp_gauge"] = int(self.apply_m_imp_gauge)
        attrs["magnetospheric_shielding"] = int(self.magnetospheric_shielding)
        attrs["dense_full_operators"] = int(self.dense_full_operators)
        attrs["induction_null_diagnostics"] = int(self.induction_null_diagnostics)
        attrs["induction_null_svd_rtol"] = self.induction_null_svd_rtol
        attrs["induction_null_warn_ratio"] = self.induction_null_warn_ratio
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
        attrs["ll_constraint_mode"] = self.ll_constraint_mode
        # Remove backend as it is runtime configuration
        if "backend" in attrs:
            del attrs["backend"]
        if "run_directory" in attrs:
            del attrs["run_directory"]

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
            valid_modes = [mode.value for mode in SimulationMode]
            matches = get_close_matches(str(mode_str), valid_modes, n=1, cutoff=0.7)
            suggestion = f" Did you mean {matches[0]!r}?" if matches else ""
            raise ValueError(
                f"Invalid simulation_mode {mode_str!r} in saved settings. "
                f"Valid options: {valid_modes}.{suggestion}"
            ) from None

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
            ll_constraint_mode=get("ll_constraint_mode", defaults.ll_constraint_mode),
            apply_psi_gauge=bool(get("apply_psi_gauge", defaults.apply_psi_gauge)),
            apply_m_ind_gauge=bool(get("apply_m_ind_gauge", defaults.apply_m_ind_gauge)),
            apply_m_imp_gauge=bool(get("apply_m_imp_gauge", defaults.apply_m_imp_gauge)),
            magnetospheric_shielding=bool(
                get("magnetospheric_shielding", defaults.magnetospheric_shielding)
            ),
            northern_hemisphere_apex_constraints=bool(
                get(
                    "northern_hemisphere_apex_constraints",
                    defaults.northern_hemisphere_apex_constraints,
                )
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
            run_directory=get("run_directory", defaults.run_directory),
            artifact_storage=get("artifact_storage", defaults.artifact_storage),
            solution_basis_kind=get("solution_basis_kind", defaults.solution_basis_kind),
            dynamics_mode=get("dynamics_mode", defaults.dynamics_mode),
            toroidal_weighting=get("toroidal_weighting", defaults.toroidal_weighting),
            poloidal_weighting=get("poloidal_weighting", defaults.poloidal_weighting),
            least_squares_preconditioner=get(
                "least_squares_preconditioner", defaults.least_squares_preconditioner
            ),
            conductance_interpolation_mode=get(
                "conductance_interpolation_mode", defaults.conductance_interpolation_mode
            ),
            conductance_interpolation_floor=get(
                "conductance_interpolation_floor", defaults.conductance_interpolation_floor
            ),
            m_imp_regularization_lambda=get(
                "m_imp_regularization_lambda", defaults.m_imp_regularization_lambda
            ),
            toroidal_regularization_lambda=get(
                "toroidal_regularization_lambda", defaults.toroidal_regularization_lambda
            ),
            dense_full_operators=bool(get("dense_full_operators", defaults.dense_full_operators)),
            induction_null_diagnostics=bool(
                get("induction_null_diagnostics", defaults.induction_null_diagnostics)
            ),
            induction_null_svd_rtol=get(
                "induction_null_svd_rtol", defaults.induction_null_svd_rtol
            ),
            induction_null_warn_ratio=get(
                "induction_null_warn_ratio", defaults.induction_null_warn_ratio
            ),
            enable_fast_input_path=bool(
                get("enable_fast_input_path", defaults.enable_fast_input_path)
            ),
            exponential_solver=exp_solver,
        )
