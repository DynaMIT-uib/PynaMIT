"""Dynamics module.

This module contains the Dynamics class for simulating dynamic MIT
coupling.
"""

from __future__ import annotations
import logging
from typing import Any, List, Optional, Union, Literal

import numpy as np
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.gauss_legendre.gl_basis import GLBasis
from pynamit.math.constants import RE
from pynamit.primitives.grid import Grid
from pynamit.primitives.io import IO
from pynamit.primitives.mainfield import Mainfield
from pynamit.simulation.state import State
from pynamit.primitives.timeseries import Timeseries
from pynamit.primitives.input_manager import InputManager
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.utils import asarray, set_backend, xp

# Import settings from dedicated module
from pynamit.simulation.settings import (
    SimulationMode,
    DynamicsSettings,
    FLOAT_ERROR_MARGIN,
)


logger = logging.getLogger(__name__)


class Dynamics:
    """Class for simulating dynamic MIT coupling.

    Manages the temporal evolution of the state of the ionosphere in
    response to field-aligned currents and neutral winds, giving rise to
    dynamic magnetosphere-ionosphere-thermosphere (MIT) coupling. Saves
    and loads simulation data to and from NetCDF files.

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
        filename_prefix: str = "simulation",
        Nmax: int = 20,
        Mmax: int = 20,
        Ncs: int = 30,
        RI: float = RE + 110.0e3,
        RM: Optional[float] = None,
        mainfield_kind: Literal["dipole", "igrf", "radial"] = "dipole",
        mainfield_epoch: int = 2020,
        mainfield_B0: Optional[float] = None,
        FAC_integration_steps: Union[np.ndarray, List[float]] = None,
        ignore_PFAC: bool = False,
        connect_hemispheres: bool = False,
        latitude_boundary: float = 50.0,
        ih_constraint_scaling: float = 1e-5,
        apply_psi_gauge: bool = True,
        apply_m_ind_gauge: bool = True,
        magnetospheric_toroidal_lock: bool = False,
        magnetospheric_poloidal_lock: bool = True,
        northern_hemisphere_apex_constraints: bool = False,
        vector_jr: bool = True,
        vector_Br: bool = True,
        vector_conductance: bool = True,
        vector_u: bool = True,
        t0: str = "2020-01-01 00:00:00",
        save_steady_states: bool = True,
        integrator: Literal["euler", "exponential"] = "euler",
        backend: Union[Literal["auto", "numpy", "jax"], bool] = "auto",
        solution_basis_kind: Literal["SH", "CS"] = "SH",
        simulation_mode: Optional[SimulationMode] = None,
        least_squares_solver: str = "lsmr",
        m_imp_regularization_lambda: float = 0.0,
        dynamics_mode: Literal["legacy", "full_induction"] = "legacy",
        toroidal_weighting: Literal["none", "linear", "quadratic"] = "none",
        poloidal_weighting: Literal["none", "linear", "quadratic"] = "none",
        least_squares_preconditioner: Optional[Literal["jacobi", "pinv"]] = "pinv",
        conductance_interpolation_mode: Literal[
            "legacy_eta_linear", "sigma_linear", "sigma_log"
        ] = "sigma_log",
        conductance_interpolation_floor: float = 1e-3,
        toroidal_regularization_lambda: float = 1e-10,
        dense_full_operators: bool = False,
        benchmark_mode: bool = False,
    ):
        """Initialize the Dynamics class."""
        if FAC_integration_steps is None:
            FAC_integration_steps = np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11)
            
        if simulation_mode == SimulationMode.CS_DOMINANT:
            solution_basis_kind = "CS"

        initial_settings = DynamicsSettings(
            Nmax=Nmax,
            Mmax=Mmax,
            Ncs=Ncs,
            RI=RI,
            RM=RM,
            mainfield_kind=mainfield_kind,
            mainfield_epoch=mainfield_epoch,
            mainfield_B0=mainfield_B0,
            FAC_integration_steps=FAC_integration_steps,
            ignore_PFAC=ignore_PFAC,
            connect_hemispheres=connect_hemispheres,
            latitude_boundary=latitude_boundary,
            ih_constraint_scaling=ih_constraint_scaling,
            apply_psi_gauge=apply_psi_gauge,
            apply_m_ind_gauge=apply_m_ind_gauge,
            magnetospheric_toroidal_lock=magnetospheric_toroidal_lock,
            magnetospheric_poloidal_lock=magnetospheric_poloidal_lock,
            northern_hemisphere_apex_constraints=northern_hemisphere_apex_constraints,
            vector_jr=vector_jr,
            vector_Br=vector_Br,
            vector_conductance=vector_conductance,
            vector_u=vector_u,
            t0=t0,
            save_steady_states=save_steady_states,
            integrator=integrator,
            backend=backend,
            solution_basis_kind=solution_basis_kind,
            simulation_mode=(
                SimulationMode.SPECTRAL_TRANSFORM_CS if simulation_mode is None else simulation_mode
            ),
            least_squares_solver=least_squares_solver,
            m_imp_regularization_lambda=m_imp_regularization_lambda,
            dynamics_mode=dynamics_mode,
            toroidal_weighting=toroidal_weighting,
            poloidal_weighting=poloidal_weighting,
            least_squares_preconditioner=least_squares_preconditioner,
            conductance_interpolation_mode=conductance_interpolation_mode,
            conductance_interpolation_floor=conductance_interpolation_floor,
            toroidal_regularization_lambda=toroidal_regularization_lambda,
            dense_full_operators=dense_full_operators,
        )
        
        if simulation_mode is not None:
            initial_settings.simulation_mode = simulation_mode
        self.settings = initial_settings
        self.backend = set_backend(backend)

        self.filename_prefix = filename_prefix
        self.io = IO(filename_prefix)
        self.benchmark_mode = bool(benchmark_mode)

        # Check if settings are consistent with previously saved runs.
        settings_on_file = None
        if not self.benchmark_mode:
            settings_on_file = self.io.load_dataset("settings", print_info=True)

        if settings_on_file is not None:
            if not self.settings.to_dataset().identical(settings_on_file):
                raise ValueError(
                    "Mismatch between Dynamics object arguments and settings on file."
                )

        PFAC_matrix_on_file = None
        if not self.benchmark_mode:
            PFAC_matrix_on_file = self.io.load_dataarray("PFAC_matrix", print_info=True)

        sh_basis = SHBasis(self.settings.Nmax, self.settings.Mmax, Nmin=0)
        sh_basis_zero_removed = SHBasis(self.settings.Nmax, self.settings.Mmax)

        cs_basis = CSBasis(self.settings.Ncs)

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

        # Specify input format and load input data.
        conductance_mode = getattr(
            self.settings, "conductance_interpolation_mode", "legacy_eta_linear"
        )
        if conductance_mode == "legacy_eta_linear":
            conductance_vars = {"etaP": "scalar", "etaH": "scalar"}
        elif conductance_mode == "sigma_linear":
            conductance_vars = {"SigmaP": "scalar", "SigmaH": "scalar"}
        elif conductance_mode == "sigma_log":
            conductance_vars = {"logSigmaP": "scalar", "logSigmaH": "scalar"}
        else:
            raise ValueError(
                "Invalid conductance_interpolation_mode="
                f"{conductance_mode!r}. "
                "Valid modes: 'legacy_eta_linear', 'sigma_linear', 'sigma_log'."
            )
        self.input_variables = {
            "jr": {"jr": "scalar"},
            "Br": {"Br": "scalar"},
            "conductance": conductance_vars,
            "u": {"u": "tangential"},
        }

        self.input_storage_bases = {
            "jr": sh_basis_zero_removed,
            "Br": sh_basis_zero_removed,
            "conductance": sh_basis,
            "u": sh_basis_zero_removed,
        }

        self.input_timeseries = Timeseries(self.input_storage_bases, self.input_variables)
        self.input_manager = InputManager(self.input_timeseries, grid_basis, self.input_variables)
        if not self.benchmark_mode:
            self.input_timeseries.load_all(self.io)

        # Specify output format and load output data.
        self.output_variables = {
            "state": {"m_ind": "scalar", "psi": "scalar", "m_imp": "scalar", "Phi": "scalar", "W": "scalar"},
            "steady_state": {"m_ind": "scalar", "psi": "scalar", "m_imp": "scalar", "Phi": "scalar", "W": "scalar"},
        }

        # Select solution basis
        if self.settings.solution_basis_kind == "CS":
            solution_basis = cs_basis
            state_output_basis = cs_basis
        else:
            solution_basis = sh_basis_zero_removed
            state_output_basis = sh_basis_zero_removed

        self.output_storage_bases = {
            "state": state_output_basis,
            "steady_state": state_output_basis,
        }

        self.output_timeseries = Timeseries(self.output_storage_bases, self.output_variables)
        if not self.benchmark_mode:
            self.output_timeseries.load_all(self.io)

        self.interpolation_bases = {
            "jr": sh_basis_zero_removed if bool(self.settings.vector_jr) else grid_basis,
            "Br": sh_basis_zero_removed if bool(self.settings.vector_Br) else grid_basis,
            "conductance": sh_basis if bool(self.settings.vector_conductance) else grid_basis,
            "u": sh_basis_zero_removed if bool(self.settings.vector_u) else grid_basis,
            # Add psi to interpolation bases to support output timeseries loading
            "psi": sh_basis_zero_removed, 
        }

        self.mainfield = Mainfield(
            kind=self.settings.mainfield_kind,
            epoch=self.settings.mainfield_epoch,
            hI=(self.settings.RI - RE) * 1e-3,
            B0=None if self.settings.mainfield_B0 == 0 else self.settings.mainfield_B0,
        )

        # Initialize the state of the ionosphere, restarting from the
        # last state checkpoint if available.
        self.state = State(
            basis=solution_basis,
            mainfield=self.mainfield,
            grid_basis=grid_basis,
            settings=self.settings,
            PFAC_matrix=PFAC_matrix_on_file,
            solution_basis=solution_basis,
        )

        if "state" in self.output_timeseries.datasets.keys():
            self.current_time = np.max(self.output_timeseries.datasets["state"].time.values)
        else:
            self.current_time = np.float64(0)

        # Store settings and PFAC matrix on file.
        if filename_prefix is None:
            self.io.filename_prefix = "simulation"

        if (not self.benchmark_mode) and settings_on_file is None:
            self.io.save_dataset(self.settings.to_dataset(), "settings", print_info=True)

        if (not self.benchmark_mode) and PFAC_matrix_on_file is None:
            self.io.save_dataarray(self.state.geometry.T_to_Ve, "PFAC_matrix", print_info=True)

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

        if "state" in self.output_timeseries.datasets.keys():
            self.current_time = np.max(self.output_timeseries.datasets["state"].time.values)
            inductive_m_ind = self.output_timeseries.get_entry(
                "state", self.current_time, interpolation=False
            )["m_ind"]
            inductive_m_ind = asarray(inductive_m_ind)
        else:
            if steady_state_initialization:
                self.state.update(self.input_manager, self.current_time, interpolation=True)
                E_coeffs_noind, m_imp_noind = self.state.calculate_noind_coeffs()
                psi, inductive_m_ind = self.state.solve_steady_state_model_variables(
                    E_coeffs_noind,
                )
            else:
                self.current_time = np.float64(0)
                zeros = xp.zeros((self.output_storage_bases["state"].index_length,))
                inductive_m_ind = zeros

        # Sync state psi if dynamic mode
        if self.settings.dynamics_mode == "full_induction":
             if self.state.psi is None:
                  self.state.psi = xp.zeros((self.state.solution_basis.index_length,))
             psi = self.state.psi
        else:
             psi = None

        while True:
            self.state.update(self.input_manager, self.current_time, interpolation=True)

            E_coeffs_noind, m_imp_noind = self.state.calculate_noind_coeffs()
            if self.settings.dynamics_mode == "full_induction":
                psi = self.state.psi

            # Prepare data for logging/storage
            if self.settings.dynamics_mode == "full_induction":
                current_m_ind = inductive_m_ind
                current_psi = psi
                if bool(self.settings.save_steady_states) and step % sampling_step_interval == 0:
                    steady_state_psi, steady_state_m_ind = self.state.solve_steady_state_model_variables(
                        E_coeffs_noind,
                        update_state=False,
                    )
                    steady_state_psi = np.asarray(steady_state_psi)
                    steady_state_m_ind = np.asarray(steady_state_m_ind)
                else:
                    steady_state_m_ind = None
                    steady_state_psi = None
            else:
                current_m_ind = inductive_m_ind
                current_psi = None
                if self.settings.integrator == "exponential" or (
                    bool(self.settings.save_steady_states) and step % sampling_step_interval == 0
                ):
                    steady_state_psi, steady_state_m_ind = self.state.solve_steady_state_model_variables(
                        E_coeffs_noind,
                        update_state=False,
                    )
                    steady_state_m_ind = np.asarray(steady_state_m_ind)
                else:
                    steady_state_m_ind = None
                steady_state_psi = None

            if step % sampling_step_interval == 0:
                self.add_state_to_timeseries("state", current_m_ind, E_coeffs_noind, m_imp_noind, psi=current_psi)

                if bool(self.settings.save_steady_states) and steady_state_m_ind is not None:
                    self.add_state_to_timeseries(
                        "steady_state", steady_state_m_ind, E_coeffs_noind, m_imp_noind, psi=steady_state_psi
                    )

                # Save state and steady state time series.
                if step % (sampling_step_interval * saving_sample_interval) == 0:
                    self.output_timeseries.save("state", self.io)

                    if quiet:
                        pass
                    else:
                        print(
                            "Saved state at t = {:.2f} s".format(self.current_time),
                            end="\n" if bool(self.settings.save_steady_states) else "\r",
                            flush=True,
                        )

                    if bool(self.settings.save_steady_states) and steady_state_m_ind is not None:
                        self.output_timeseries.save("steady_state", self.io)

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
                psi=psi,
            )
            if self.settings.dynamics_mode == "full_induction":
                psi = psi_new
            
            self.current_time = next_time

            step += 1

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
             state_data["m_ind"] = np.zeros(self.state.solution_basis.index_length)
             
        if state_data["psi"] is None and "psi" in self.output_variables[key]:
             state_data["psi"] = np.zeros(self.state.solution_basis.index_length)

        self.output_timeseries.add_entry(key, state_data, time=self.current_time)

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

        self.set_jr(
            FAC.flatten() * FAC_b_field.vec.r / FAC_b_field.magnitude,
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

        self.input_manager.interpolate_and_add_entry(
            "jr",
            input_data,
            self.adapt_input_time(time, input_data),
            self.interpolation_bases["jr"],
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        if not self.benchmark_mode:
            self.input_timeseries.save("jr", self.io)

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

        self.input_manager.interpolate_and_add_entry(
            "Br",
            input_data,
            self.adapt_input_time(time, input_data),
            self.interpolation_bases["Br"],
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        if not self.benchmark_mode:
            self.input_timeseries.save("Br", self.io)

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
        mode = getattr(self.settings, "conductance_interpolation_mode", "legacy_eta_linear")
        sigma_floor = float(
            max(getattr(self.settings, "conductance_interpolation_floor", 1e-3), 0.0)
        )

        if mode == "legacy_eta_linear":
            input_data = {"etaP": np.empty_like(Pedersen), "etaH": np.empty_like(Hall)}

            # Legacy behavior: convert conductances to resistivities first, then
            # interpolate in eta-space.
            for i in range(max(input_data["etaP"].shape[0], 1)):
                input_data["etaP"][i] = Pedersen[i] / (Hall[i] ** 2 + Pedersen[i] ** 2)

            for i in range(max(input_data["etaH"].shape[0], 1)):
                input_data["etaH"][i] = Hall[i] / (Hall[i] ** 2 + Pedersen[i] ** 2)
        else:
            ped = np.asarray(Pedersen, dtype=float)
            hall = np.asarray(Hall, dtype=float)
            if np.any(ped < 0.0):
                logger.warning(
                    "Negative Pedersen conductance supplied; clipping to nonnegative "
                    "for conductance_interpolation_mode=%s.",
                    mode,
                )
                ped = np.maximum(ped, 0.0)
            if np.any(hall < 0.0):
                logger.warning(
                    "Negative Hall conductance supplied; clipping to nonnegative "
                    "for conductance_interpolation_mode=%s (Hall sign is geometry-driven).",
                    mode,
                )
                hall = np.maximum(hall, 0.0)

            if mode == "sigma_linear":
                input_data = {"SigmaP": ped, "SigmaH": hall}
            elif mode == "sigma_log":
                floor = max(sigma_floor, np.finfo(float).tiny)
                input_data = {
                    "logSigmaP": np.log(ped + floor),
                    "logSigmaH": np.log(hall + floor),
                }
            else:
                raise ValueError(
                    "Invalid conductance_interpolation_mode="
                    f"{mode!r}. "
                    "Valid modes: 'legacy_eta_linear', 'sigma_linear', 'sigma_log'."
                )

        self.input_manager.interpolate_and_add_entry(
            "conductance",
            input_data,
            self.adapt_input_time(time, input_data),
            self.interpolation_bases["conductance"],
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        if not self.benchmark_mode:
            self.input_timeseries.save("conductance", self.io)

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

        self.input_manager.interpolate_and_add_entry(
            "u",
            input_data,
            self.adapt_input_time(time, input_data),
            self.interpolation_bases["u"],
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        if not self.benchmark_mode:
            self.input_timeseries.save("u", self.io)

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
