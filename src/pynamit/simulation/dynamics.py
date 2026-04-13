"""Dynamics module.

This module contains the Dynamics class for simulating dynamic MIT
coupling.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pynamit.gauss_legendre.gl_basis import GLBasis
from pynamit.primitives.grid import Grid
from pynamit.primitives.io import IO
from pynamit.simulation.state import State
from pynamit.primitives.input_manager import InputManager
from pynamit.utils import asarray, set_backend, xp
from pynamit.simulation.input import encode_conductance_input_for_storage
from pynamit.simulation.data import SimulationData

# Import settings from dedicated module
from pynamit.simulation.settings import (
    DynamicsMode,
    SimulationMode,
    DynamicsSettings,
    FLOAT_ERROR_MARGIN,
    IntegratorKind,
)


logger = logging.getLogger(__name__)


class Dynamics:
    """Class for simulating dynamic MIT coupling.

    Manages the temporal evolution of the state of the ionosphere in
    response to field-aligned currents and neutral winds, giving rise to
    dynamic magnetosphere-ionosphere-thermosphere (MIT) coupling. Saves
    and loads simulation data to and from persisted xarray artifacts.

    Attributes
    ----------
    current_time : float
        Current simulation time in seconds.
    state : State
        Current state of the system.
    RI : float
        Radius of the ionosphere in meters.
    basis : Basis
        The mathematical basis used for the simulation.
    mainfield : Mainfield
        Main magnetic field model.
    """

    def __init__(
        self,
        settings: Optional[Any] = None,
        *,
        benchmark_mode: bool = False,
        radial_shell_response_model: Optional[Any] = None,
        **settings_overrides: Any,
    ):
        """Initialize the Dynamics class."""
        if isinstance(settings, (str, Path)):
            settings_path = Path(settings)
            io: IO | None = None
            resolved_directory = None
            if settings_path.exists() and settings_path.is_dir():
                resolved_directory = IO.discover_run_directory(settings_path)
                io = IO(resolved_directory)
                loaded_settings_dataset = io.load_dataset("settings")
            elif settings_path.exists():
                raise ValueError(
                    f"Expected a run directory for restart, got existing non-directory path "
                    f"{str(settings_path)!r}."
                )
            else:
                loaded_settings_dataset = None
            if loaded_settings_dataset is not None:
                settings_storage = io.get_dataset_storage_kind("settings") or "auto"
                loaded_settings = DynamicsSettings.from_dataset(
                    loaded_settings_dataset,
                    DynamicsSettings(
                        run_directory=resolved_directory, artifact_storage=settings_storage
                    ),
                )
                settings = DynamicsSettings.coerce(loaded_settings, **settings_overrides)
            else:
                settings_overrides = {"run_directory": str(settings_path), **settings_overrides}
                settings = None

        if settings is None:
            self.settings = DynamicsSettings(**settings_overrides)
        else:
            self.settings = DynamicsSettings.coerce(settings, **settings_overrides)
        self.backend = set_backend(self.settings.backend)
        self.benchmark_mode = bool(benchmark_mode)
        self._validate_supported_full_induction_runtime_model(
            radial_shell_response_model=radial_shell_response_model
        )

        self._uses_temporary_run_directory = False
        if (not self.benchmark_mode) and self.settings.run_directory is None:
            temporary_run_directory = IO.build_temporary_run_directory()
            self.settings.run_directory = temporary_run_directory
            self._uses_temporary_run_directory = True

        self.run_directory = self.settings.run_directory
        self.data = SimulationData.create(
            self.run_directory,
            self.settings,
            load_existing=not self.benchmark_mode,
            print_info=not self.benchmark_mode,
        )
        self.settings = self.data.settings
        cs_basis = self.data.cs_basis
        sh_basis = self.data.sh_basis

        # Select grid basis based on simulation mode
        # GL grid for exact SH transforms (pure spectral and GL transform modes)
        # CS grid for cubed-sphere based modes
        if self.settings.simulation_mode in (
            SimulationMode.PURE_SPECTRAL,
            SimulationMode.SPECTRAL_TRANSFORM_GL,
        ):
            grid_basis = GLBasis(self.settings.Nmax)
        else:
            grid_basis = cs_basis

        self.input_manager = InputManager(
            self.data.input_timeseries,
            grid_basis,
            self.data.input_variables,
            enable_fast_path=self.settings.enable_fast_input_path,
        )
        solution_spec = self.data.solution_spec

        self.interpolation_bases = {
            "jr": sh_basis if bool(self.settings.vector_jr) else grid_basis,
            "Br": sh_basis if bool(self.settings.vector_Br) else grid_basis,
            "conductance": sh_basis if bool(self.settings.vector_conductance) else grid_basis,
            "u": sh_basis if bool(self.settings.vector_u) else grid_basis,
            # Add psi to interpolation bases to support output timeseries loading
            "psi": sh_basis,
        }

        # Initialize the state of the ionosphere, restarting from the
        # last state checkpoint if available.
        self.state = State(
            basis=solution_spec,
            mainfield=self.data.mainfield,
            grid_basis=grid_basis,
            settings=self.settings,
            PFAC_matrix=self.data.pfac_matrix,
            solution_space=solution_spec,
            radial_shell_response_model=radial_shell_response_model,
        )

        if self.data.has_dataset("state"):
            self.current_time = self.data.get_latest_output_time("state")
        else:
            self.current_time = np.float64(0)
        self._current_m_ind = None
        if self.data.has_dataset("state"):
            state_entry = self.data.get_output_entry(
                "state", self.current_time, interpolation=False
            )
            if state_entry is not None and state_entry.get("m_ind") is not None:
                self._current_m_ind = asarray(state_entry["m_ind"])
            if (
                self.settings.dynamics_mode == DynamicsMode.FULL_INDUCTION
                and state_entry is not None
            ):
                psi_entry = state_entry.get("psi")
                if psi_entry is not None:
                    self.state.psi = asarray(psi_entry)

        # Store settings and PFAC matrix on file.
        if (not self.benchmark_mode) and not self.data.settings_from_file:
            self.data.save_settings(print_info=True)

        if (not self.benchmark_mode) and not self.data.pfac_from_file:
            self.data.save_pfac_matrix(self.state.geometry.T_to_Ve, print_info=True)

    def _validate_supported_full_induction_runtime_model(
        self, *, radial_shell_response_model: Optional[Any]
    ) -> None:
        """Restrict operational full-induction runs to the canonical shell-gap model.

        Older tangential closures and non-canonical radial-shell forcing variants
        are kept only for benchmark/diagnostic use. This preserves internal
        operator-verification paths while keeping one supported runtime model.
        """
        if self.settings.dynamics_mode != DynamicsMode.FULL_INDUCTION:
            return
        if self.benchmark_mode:
            return

        if radial_shell_response_model is not None:
            raise ValueError(
                "Operational full_induction uses the built-in canonical "
                "shell-gap response model. Explicit radial_shell_response_model "
                "overrides are available only with benchmark_mode=True."
            )

    @classmethod
    def from_directory(cls, run_directory: str | Path, **settings_overrides: Any) -> "Dynamics":
        """Construct a simulation by restarting from one saved run directory."""
        return cls(IO.discover_run_directory(run_directory), **settings_overrides)

    @property
    def io(self):
        """Persistence backend for this simulation run."""
        return self.data.io

    @property
    def mainfield(self):
        """Main magnetic field model used by the run."""
        return self.data.mainfield

    @property
    def input_timeseries(self):
        """Input timeseries storage owned by the persisted run package."""
        return self.data.input_timeseries

    @property
    def output_timeseries(self):
        """Output timeseries storage owned by the persisted run package."""
        return self.data.output_timeseries

    @property
    def input_variables(self):
        """Input variable schema."""
        return self.data.input_variables

    @property
    def output_variables(self):
        """Output variable schema."""
        return self.data.output_variables

    @property
    def uses_temporary_run_directory(self) -> bool:
        """Whether the run is persisting to an auto-generated temporary directory."""
        return self._uses_temporary_run_directory

    def evolve_to_time(
        self,
        t,
        dt=np.float64(5e-4),
        sampling_step_interval=200,
        saving_sample_interval=10,
        quiet=False,
        steady_state_initialization=True,
    ):
        """Evolve the system state to a specified time.

        Parameters
        ----------
        t : float
            Target time to evolve to in seconds.
        dt : float, optional
            Time step size in seconds.
        sampling_step_interval : int, optional
            Number of steps between samples.
        saving_sample_interval : int, optional
            Number of samples between saves.
        quiet : bool, optional
            Whether to suppress progress output.
        """
        step = 0

        if self._current_m_ind is not None:
            inductive_m_ind = asarray(self._current_m_ind)
            m_ind_finite = bool(np.all(np.isfinite(np.asarray(inductive_m_ind))))
            psi_finite = True
            if (
                self.settings.dynamics_mode == DynamicsMode.FULL_INDUCTION
                and self.state.psi is not None
            ):
                psi_finite = bool(np.all(np.isfinite(np.asarray(self.state.psi))))
            if not (m_ind_finite and psi_finite):
                run_directory = self.settings.run_directory
                raise ValueError(
                    "Non-finite values found in saved state used for resume "
                    f"(run_directory={run_directory!r}, time={float(self.current_time):.3f}s). "
                    "Delete the saved state/steady_state output files for this run "
                    "or use a new run_directory to start from a clean initialization."
                )
        else:
            if steady_state_initialization:
                self.state.update(self.input_manager, self.current_time, interpolation=True)
                E_coeffs_noind, m_imp_noind = self.state.calculate_noind_coeffs()
                psi, inductive_m_ind = self.state.solve_steady_state_model_variables(
                    E_coeffs_noind
                )
            else:
                self.current_time = np.float64(0)
                zeros = xp.zeros((self.output_timeseries.get_storage_spec("state").index_length,))
                inductive_m_ind = zeros
            self._current_m_ind = asarray(inductive_m_ind)

        # Sync state psi if dynamic mode
        if self.settings.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            if self.state.psi is None:
                self.state.psi = xp.zeros((self.state.solution_space.index_length,))
            psi = self.state.psi
        else:
            psi = None

        while True:
            self.state.update(self.input_manager, self.current_time, interpolation=True)

            E_coeffs_noind, m_imp_noind = self.state.calculate_noind_coeffs()
            if self.settings.dynamics_mode == DynamicsMode.FULL_INDUCTION:
                psi = self.state.psi

            # Prepare data for logging/storage
            if self.settings.dynamics_mode == DynamicsMode.FULL_INDUCTION:
                current_m_ind = inductive_m_ind
                current_psi = psi
                need_steady_state_for_step = (
                    self.settings.integrator == IntegratorKind.EXPONENTIAL
                    and self.state.get_effective_exponential_step_form() == "centered"
                )
                need_steady_state_for_output = (
                    bool(self.settings.save_steady_states) and step % sampling_step_interval == 0
                )
                if need_steady_state_for_step or need_steady_state_for_output:
                    steady_state_psi, steady_state_m_ind = (
                        self.state.solve_steady_state_model_variables(
                            E_coeffs_noind, update_state=False
                        )
                    )
                    steady_state_psi = np.asarray(steady_state_psi)
                    steady_state_m_ind = np.asarray(steady_state_m_ind)
                else:
                    steady_state_m_ind = None
                    steady_state_psi = None
            else:
                current_m_ind = inductive_m_ind
                current_psi = None
                need_steady_state_for_step = (
                    self.settings.integrator == IntegratorKind.EXPONENTIAL
                    and self.state.get_effective_exponential_step_form() == "centered"
                )
                if need_steady_state_for_step or (
                    bool(self.settings.save_steady_states) and step % sampling_step_interval == 0
                ):
                    steady_state_psi, steady_state_m_ind = (
                        self.state.solve_steady_state_model_variables(
                            E_coeffs_noind, update_state=False
                        )
                    )
                    steady_state_m_ind = np.asarray(steady_state_m_ind)
                else:
                    steady_state_m_ind = None
                steady_state_psi = None

            if step % sampling_step_interval == 0:
                self.add_state_to_timeseries(
                    "state", current_m_ind, E_coeffs_noind, m_imp_noind, psi=current_psi
                )

                if bool(self.settings.save_steady_states) and steady_state_m_ind is not None:
                    self.add_state_to_timeseries(
                        "steady_state",
                        steady_state_m_ind,
                        E_coeffs_noind,
                        m_imp_noind,
                        psi=steady_state_psi,
                    )

                # Save state and steady state time series.
                if step % (sampling_step_interval * saving_sample_interval) == 0:
                    self.data.save_output_dataset("state")

                    if quiet:
                        pass
                    else:
                        print(
                            "Saved state at t = {:.2f} s".format(self.current_time),
                            end="\n" if bool(self.settings.save_steady_states) else "\r",
                            flush=True,
                        )

                    if bool(self.settings.save_steady_states) and steady_state_m_ind is not None:
                        self.data.save_output_dataset("steady_state")

                        if quiet:
                            pass
                        else:
                            print(
                                "Saved steady state at t = {:.2f} s".format(self.current_time),
                                end="\x1b[F",
                                flush=True,
                            )

            next_time = self.current_time + dt

            if next_time > t + FLOAT_ERROR_MARGIN:
                if quiet:
                    pass
                else:
                    print("\n\n")
                break

            # Evolve State (single-state or coupled-state handled by State).
            psi_new, inductive_m_ind = self.state.evolve_model_variables(
                m_ind=inductive_m_ind,
                dt=dt,
                E_coeffs_noind=E_coeffs_noind,
                steady_state_m_ind=steady_state_m_ind,
                steady_state_psi=steady_state_psi,
                psi=psi,
            )
            if self.settings.dynamics_mode == DynamicsMode.FULL_INDUCTION:
                psi = psi_new

            self.current_time = next_time
            self._current_m_ind = asarray(inductive_m_ind)

            step += 1

    def impose_steady_state(
        self,
        time: Optional[float] = None,
        *,
        interpolation: bool = True,
        save: bool = True,
        quiet: bool = False,
    ):
        """Replace the live model state by the steady-state solution at one time.

        Parameters
        ----------
        time : float, optional
            Simulation time at which to impose the steady state. If omitted,
            uses the current simulation time.
        interpolation : bool, optional
            Whether to interpolate inputs to the requested time before solving.
        save : bool, optional
            Whether to persist the imposed state/steady-state entries.
        quiet : bool, optional
            Suppress status output when saving.
        """
        if time is not None:
            self.current_time = np.float64(time)

        self.state.update(self.input_manager, self.current_time, interpolation=interpolation)
        E_coeffs_noind, m_imp_noind = self.state.calculate_noind_coeffs()
        psi_ss, m_ind_ss = self.state.solve_steady_state_model_variables(
            E_coeffs_noind, update_state=True
        )
        m_ind_ss = asarray(m_ind_ss)
        self._current_m_ind = m_ind_ss
        if self.settings.dynamics_mode == DynamicsMode.FULL_INDUCTION:
            self.state.psi = None if psi_ss is None else asarray(psi_ss)

        if save and not self.benchmark_mode:
            self.add_state_to_timeseries(
                "state", m_ind_ss, E_coeffs_noind, m_imp_noind, psi=psi_ss
            )
            if bool(self.settings.save_steady_states):
                self.add_state_to_timeseries(
                    "steady_state", m_ind_ss, E_coeffs_noind, m_imp_noind, psi=psi_ss
                )
            self.data.save_output_dataset("state")
            if bool(self.settings.save_steady_states):
                self.data.save_output_dataset("steady_state")
            if not quiet:
                print(f"Imposed steady state at t = {float(self.current_time):.2f} s")

        return psi_ss, m_ind_ss

    def add_state_to_timeseries(self, key, m_ind, E_coeffs, m_imp, psi=None):
        """Add the current state to the time series.

        Parameters
        ----------
        key : str
            Key for the time series entry.
        m_ind : array-like
            Inductive magnetic field coefficients.
        E_coeffs : tuple
            Electric field coefficients without induced effects.
        m_imp : array-like
            Imposed magnetic field coefficients without induced effects.
        psi : array-like, optional
            Toroidal Stream Function coefficients.
        """
        # Include dynamic inductive toroidal residual contribution (psi -> E).
        if psi is not None:
            E_coeffs = np.asarray(E_coeffs) + self.state.calculate_psi_E_coeffs(np.asarray(psi))

        # Calculate full fields if needed (only if m_ind provided)
        # If full induction, m_ind might be None.
        if m_ind is not None:
            E_coeffs_ind, m_imp_ind = self.state.calculate_ind_coeffs(m_ind)
            E_coeffs = np.asarray(E_coeffs) + E_coeffs_ind
            m_imp = np.asarray(m_imp) + m_imp_ind
        else:
            # Just use what was passed (no poloidal induction added)
            E_coeffs = np.asarray(E_coeffs) if E_coeffs is not None else (np.zeros(1), np.zeros(1))
            m_imp = np.asarray(m_imp)

        # Append current state to time series.
        state_data = {
            "m_ind": np.asarray(m_ind) if m_ind is not None else None,
            "m_imp": np.asarray(m_imp),
            "psi": np.asarray(psi) if psi is not None else None,
            "Phi": np.asarray(E_coeffs[0]),
            "W": np.asarray(E_coeffs[1]),
        }

        # Ensure dummy zeros for missing fields if required by schema
        if state_data["m_ind"] is None and "m_ind" in self.output_variables[key]:
            state_data["m_ind"] = np.zeros(self.state.solution_space.index_length)

        if state_data["psi"] is None and "psi" in self.output_variables[key]:
            state_data["psi"] = np.zeros(self.state.solution_space.index_length)

        self.data.add_output_entry(key, state_data, time=self.current_time)

    def _interpolate_and_store_input(
        self,
        key: str,
        input_data: dict[str, np.ndarray],
        *,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ) -> None:
        """Interpolate one input payload and persist it when applicable."""
        self.input_manager.interpolate_and_add_entry(
            key,
            input_data,
            self.adapt_input_time(time, input_data),
            self.interpolation_bases[key],
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        if not self.benchmark_mode:
            self.data.save_input_dataset(key)

    def set_FAC(
        self,
        FAC,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set field-aligned current (FAC) input.

        Converts FAC to radial current density by multiplying with the
        radial component of the main field, and sets the radial current
        density as input.

        Parameters
        ----------
        FAC : array-like
            Field-aligned current density in A/m².
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the FAC data.
        sqrt_weights : array-like, optional
            sqrt_weights for the FAC data points.
        reg_lambda : float, optional
            Regularization parameter for the least squares solver.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        FAC_b_field = self.mainfield.discretize(
            Grid(lat=lat, lon=lon, theta=theta, phi=phi), self.settings.RI
        )
        fac_array = np.asarray(FAC, dtype=float)
        radial_factor = np.asarray(FAC_b_field.vec.r / FAC_b_field.magnitude, dtype=float).reshape(
            -1
        )
        n_points = radial_factor.size
        if fac_array.size % n_points != 0:
            raise ValueError(
                "FAC input size must be an integer multiple of the spatial grid size. "
                f"Got FAC.size={fac_array.size} for grid size {n_points}."
            )
        jr = fac_array.reshape(-1, n_points) * radial_factor.reshape(1, -1)

        self.set_jr(
            jr,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_jr(
        self,
        jr,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set radial current density input.

        Parameters
        ----------
        jr : array-like
            Radial current density in A/m².
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the current data.
        sqrt_weights : array-like, optional
            sqrt_weights for the current data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        input_data = {"jr": np.atleast_2d(jr)}

        self._interpolate_and_store_input(
            "jr",
            input_data,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_Br(
        self,
        Br,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set radial component of magnetic field input.

        Parameters
        ----------
        Br : array-like
            Radial component of magnetic field.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the current data.
        sqrt_weights : array-like, optional
            sqrt_weights for the current data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        if self.settings.RM == 0:
            raise ValueError("Br can only be set if magnetospheric radius (RM) is set.")

        input_data = {"Br": np.atleast_2d(Br)}

        self._interpolate_and_store_input(
            "Br",
            input_data,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_conductance(
        self,
        Hall,
        Pedersen,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set Hall and Pedersen conductance values.

        Parameters
        ----------
        Hall : array-like
            Hall conductance.
        Pedersen : array-like
            Pedersen conductance.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the conductance data.
        sqrt_weights : array-like, optional
            sqrt_weights for the conductance data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        Hall = np.atleast_2d(Hall)
        Pedersen = np.atleast_2d(Pedersen)
        mode = self.settings.conductance_interpolation_mode
        sigma_floor = float(self.settings.conductance_interpolation_floor)

        input_data = encode_conductance_input_for_storage(
            Hall=Hall, Pedersen=Pedersen, mode=mode, sigma_floor=sigma_floor, logger=logger
        )

        self._interpolate_and_store_input(
            "conductance",
            input_data,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_u(
        self,
        u_theta,
        u_phi,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set neutral wind velocities.

        Parameters
        ----------
        u_theta : array-like
            Meridional (south) wind velocity in m/s.
        u_phi : array-like
            Zonal (east) wind velocity in m/s.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the wind data.
        sqrt_weights : array-like, optional
            sqrt_weights for the wind data points.
        reg_lambda : float, optional
            Regularization parameter.
        """
        # If u_theta and u_phi, are 1D arrays, convert to 2D.
        input_data = {"u": np.array([np.atleast_2d(u_theta), np.atleast_2d(u_phi)])}
        # Reorder time to first dimension and component to second.
        input_data["u"] = np.moveaxis(input_data["u"], [0, 1], [1, 0])

        self._interpolate_and_store_input(
            "u",
            input_data,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def adapt_input_time(self, time, data):
        """Adapt array of time values given with the input data.

        Parameters
        ----------
        time : array-like, optional
            Time values for the input data.
        data : dict
            Dictionary containing input data variables.

        Returns
        -------
        time : array-like
            Adapted time values.

        Notes
        -----
        If time is None, the current time is used.

        Raises
        ------
        ValueError
            If time is None and data is of a shape that suggests
            multiple time values.
        """
        if time is None:
            if any([data[var].shape[0] > 1 for var in data.keys()]):
                raise ValueError(
                    "Time must be specified if the input data is given for multiple time values."
                )
            return np.atleast_1d(self.current_time)
        else:
            return np.atleast_1d(time)
