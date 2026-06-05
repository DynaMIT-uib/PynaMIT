"""Dynamics module.

This module contains the Dynamics class for simulating dynamic MIT
coupling.
"""

import numpy as np
import xarray as xr
from pynamit.math.constants import RE
from pynamit.math.least_squares_solver import get_default_least_squares_solver
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.simulation.schema import normalize_horizontal_basis_kind
from pynamit.sphere.spherical_transform import SphericalTransform
from pynamit.sphere import Grid, is_grid_basis
from pynamit.primitives.io import IO
from pynamit.simulation.data import SimulationData
from pynamit.simulation.mainfield import Mainfield
from pynamit.simulation.state import State
from pynamit.math.backend import set_backend, to_jax, to_numpy, use_jax

FLOAT_ERROR_MARGIN = 1e-6  # Safety margin for floating point errors


class Dynamics(object):
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
    mainfield : Mainfield
        Main magnetic field model.
    """

    def __init__(
        self,
        run_directory=None,
        Nmax=20,
        Mmax=20,
        Ncs=30,
        RI=RE + 110.0e3,
        RM=None,
        mainfield_kind="dipole",
        mainfield_epoch=2020,
        mainfield_B0=None,
        FAC_integration_steps=np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11),
        ignore_PFAC=False,
        connect_hemispheres=False,
        latitude_boundary=50,
        ih_constraint_scaling=1e-5,
        vector_jr=True,
        vector_Br=True,
        vector_conductance=True,
        vector_u=True,
        t0="2020-01-01 00:00:00",
        save_steady_states=True,
        integrator="euler",
        least_squares_solver=None,
        least_squares_preconditioner="pinv",
        artifact_storage="auto",
        backend="auto",
        horizontal_basis_kind="SH",
        area_weighted_least_squares=False,
        project_conductance=True,
    ):
        """Initialize the Dynamics class.

        Parameters
        ----------
        run_directory : str, optional
            Preferred directory for this persisted run. Artifacts are
            written as fixed names like ``settings.zarr`` inside this
            directory.
        Nmax : int, optional
            Maximum spherical harmonic degree.
        Mmax : int, optional
            Maximum spherical harmonic order.
        Ncs : int, optional
            Number of cubed sphere grid points per edge.
        RI : float, optional
            Ionospheric radius in meters.
        mainfield_kind : {'dipole', 'igrf',  'radial'}, optional
            Type of main magnetic field model.
        mainfield_epoch : int, optional
            Epoch year for main field model.
        mainfield_B0 : float, optional
            Main field strength.
        FAC_integration_steps : array-like, optional
            Integration radii for FAC poloidal field calculation.
        ignore_PFAC : bool, optional
            Whether to ignore FAC poloidal fields.
        connect_hemispheres : bool, optional
            Whether hemispheres are electrically connected.
        latitude_boundary : float, optional
            Simulation boundary latitude in degrees.
        ih_constraint_scaling : float, optional
            Scaling for interhemispheric coupling constraint.
        vector_jr : bool, optional
            Use vector representation for radial current.
        vector_Br : bool, optional
            Use vector representation for radial magnetic field
            component.
        vector_conductance : bool, optional
            Use vector representation for conductances.
        vector_u : bool, optional
            Use vector representation for neutral wind.
        t0 : str, optional
            Start time in UTC format.
        save_steady_states : bool, optional
            Default for whether ``evolve_to_time`` calculates and saves
            steady states.
        integrator : {'euler', 'exponential'}, optional
            Integrator type for time evolution.
        least_squares_solver : str, optional
            Least-squares solver used by state feedback solves.
        least_squares_preconditioner : {'jacobi', 'pinv', None},
            optional
            Preconditioner used by iterative least-squares state solves.
        artifact_storage : {'auto', 'netcdf', 'zarr'}, optional
            Preferred backend for new saved xarray artifacts. Existing
            artifacts keep their format on restart.
        backend : {'auto', 'numpy', 'jax', bool}, optional
            Array backend to use. ``"auto"`` respects the current global
            setting (environment variable or previous choice).
            ``"numpy"``/``False`` enforce NumPy arrays, while
            ``"jax"``/``True`` enables JAX.
        horizontal_basis_kind : {'SH', 'CS'}, optional
            Basis requested for horizontal state coefficients and
            surface operators. ``'SH'`` is the default. ``'CS'`` uses
            cubed-sphere nodal coefficients and finite-difference
            derivatives for horizontal surface operators. Radial
            Laplace-continuation terms use the SH radial-continuation
            basis.
        area_weighted_least_squares : bool, optional
            Use surface-area weights for least-squares projections when
            no explicit ``sqrt_weights`` are supplied. Cubed-sphere
            grids use their native cell areas; ordinary spherical grids
            use ``sin(theta)``.
        project_conductance : bool, optional
            If True, project conductance/resistance inputs into the
            configured conductance storage basis. If False, store
            resistance values directly on the model grid; the supplied
            conductance grid must match the state geometry grid and
            conductance regularization/weights are not used.
        """
        self.backend = set_backend(backend)
        horizontal_basis_kind = normalize_horizontal_basis_kind(horizontal_basis_kind)

        # Store setting arguments in xarray dataset.
        self.settings = xr.Dataset(
            attrs={
                "Nmax": Nmax,
                "Mmax": Mmax,
                "Ncs": Ncs,
                "RI": RI,
                "RM": 0 if RM is None else RM,
                "latitude_boundary": latitude_boundary,
                "ignore_PFAC": int(ignore_PFAC),
                "connect_hemispheres": int(connect_hemispheres),
                "FAC_integration_steps": FAC_integration_steps,
                "ih_constraint_scaling": ih_constraint_scaling,
                "mainfield_kind": mainfield_kind,
                "mainfield_epoch": mainfield_epoch,
                "mainfield_B0": 0 if mainfield_B0 is None else mainfield_B0,
                "vector_jr": int(vector_jr),
                "vector_Br": int(vector_Br),
                "vector_conductance": int(vector_conductance),
                "vector_u": int(vector_u),
                "horizontal_basis_kind": horizontal_basis_kind,
                "area_weighted_least_squares": int(area_weighted_least_squares),
                "project_conductance": int(project_conductance),
                "t0": t0,
                "save_steady_states": int(save_steady_states),
                "integrator": integrator,
                "least_squares_solver": least_squares_solver or get_default_least_squares_solver(),
                "least_squares_preconditioner": least_squares_preconditioner,
            }
        )

        self.data = SimulationData.create(
            self.settings,
            run_directory=run_directory,
            artifact_storage=artifact_storage,
            horizontal_basis_kind=horizontal_basis_kind,
            area_weighted_least_squares=self.settings.area_weighted_least_squares,
            print_info=True,
        )
        self.uses_temporary_run_directory = self.data.uses_temporary_run_directory
        self.io = self.data.io
        self.run_directory = self.data.run_directory

        self.schema = self.data.schema
        self.cs_basis = self.schema.cs_basis
        self.sh_basis = self.schema.sh_basis
        self.sh_basis_mean_free = self.schema.sh_basis_mean_free
        self.horizontal_basis = self.schema.horizontal_basis
        self.solid_harmonics = self.schema.solid_harmonics

        self.input_vars = self.schema.input_vars
        self.input_field_spaces = self.schema.input_field_spaces
        self.input_timeseries = self.data.input_timeseries

        self.output_vars = self.schema.output_vars
        self.output_field_spaces = self.schema.output_field_spaces
        self.output_timeseries = self.data.output_timeseries

        self.interpolation_bases = self.schema.interpolation_bases

        input_grid = Grid(
            theta=self.cs_basis.arr_theta,
            phi=self.cs_basis.arr_phi,
            area_weights=self.cs_basis.unit_area,
        )
        self.input_transforms = {
            key: SphericalTransform(
                self.input_field_spaces[key].representation,
                input_grid,
                interpolation_basis=self.cs_basis,
                area_weighted=bool(self.settings.area_weighted_least_squares),
            )
            for key in self.input_vars
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
            self.horizontal_basis,
            self.mainfield,
            self.cs_basis,
            self.settings,
            PFAC_matrix=self.data.pfac_matrix,
            solid_harmonics=self.solid_harmonics,
        )
        self.horizontal_spherical_transform = self.state.geometry.spherical_transform

        if "state" in self.output_timeseries.datasets.keys():
            self.current_time = np.max(self.output_timeseries.datasets["state"].time.values)
        else:
            self.current_time = np.float64(0)

        self.data.save_settings_if_missing(print_info=True)
        self.data.save_pfac_matrix_if_missing(self.state.geometry.T_to_Ve, print_info=True)

    @classmethod
    def from_directory(cls, run_directory, **kwargs):
        """Construct a simulation from one run directory."""
        return cls(run_directory=IO.discover_run_directory(run_directory), **kwargs)

    def evolve_to_time(
        self,
        t,
        dt=np.float64(5e-4),
        sampling_step_interval=200,
        saving_sample_interval=10,
        quiet=False,
        steady_state_initialization=True,
        run_inductive=True,
        run_steady_state=None,
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
        steady_state_initialization : bool, optional
            Whether to initialize a new inductive run from steady state.
        run_inductive : bool, optional
            Whether to run and save the inductive time-dependent state.
        run_steady_state : bool, optional
            Whether to calculate and save the algebraic steady-state
            solution. Defaults to ``self.settings.save_steady_states``.
        """
        run_inductive = bool(run_inductive)
        if run_steady_state is None:
            run_steady_state = bool(self.settings.save_steady_states)
        else:
            run_steady_state = bool(run_steady_state)

        if not run_inductive and not run_steady_state:
            raise ValueError("At least one of run_inductive or run_steady_state must be True.")

        sampling_step_interval = int(sampling_step_interval)
        saving_sample_interval = int(saving_sample_interval)
        if sampling_step_interval < 1:
            raise ValueError("sampling_step_interval must be >= 1.")
        if saving_sample_interval < 1:
            raise ValueError("saving_sample_interval must be >= 1.")

        step = 0

        inductive_m_ind = None
        if run_inductive and "state" in self.output_timeseries.datasets.keys():
            self.current_time = np.max(self.output_timeseries.datasets["state"].time.values)
            inductive_m_ind = self.output_timeseries.get_entry(
                "state", self.current_time, interpolation=False
            )["m_ind"]
            inductive_m_ind = to_jax(inductive_m_ind) if use_jax() else inductive_m_ind
            inductive_m_ind = self.state.project_scalar_mean_free(inductive_m_ind)
        elif run_inductive:
            if steady_state_initialization:
                self.state.update(self.input_timeseries, self.current_time)
                E_coeffs_noind, _ = self.state.calculate_noind_coeffs()
                inductive_m_ind = self.state.steady_state_m_ind(E_coeffs_noind)
            else:
                self.current_time = np.float64(0)
                zeros = np.zeros(self.output_field_spaces["state"].index_length)
                inductive_m_ind = to_jax(zeros) if use_jax() else zeros
                inductive_m_ind = self.state.project_scalar_mean_free(inductive_m_ind)
        elif "steady_state" in self.output_timeseries.datasets.keys():
            self.current_time = np.max(self.output_timeseries.datasets["steady_state"].time.values)
        else:
            self.current_time = np.float64(0)

        step_increment = 1 if run_inductive else sampling_step_interval

        while True:
            self.state.update(self.input_timeseries, self.current_time)

            E_coeffs_noind, m_imp_noind = self.state.calculate_noind_coeffs()

            is_sample_step = step % sampling_step_interval == 0
            should_save_sample = is_sample_step and step % (
                sampling_step_interval * saving_sample_interval
            ) == 0
            needs_steady_state = (run_inductive and self.settings.integrator == "exponential") or (
                run_steady_state and is_sample_step
            )

            if needs_steady_state:
                steady_state_m_ind = self.state.steady_state_m_ind(E_coeffs_noind)
            else:
                steady_state_m_ind = None

            if is_sample_step:
                if run_inductive:
                    self.add_state_to_timeseries(
                        "state", inductive_m_ind, E_coeffs_noind, m_imp_noind
                    )

                if run_steady_state:
                    self.add_state_to_timeseries(
                        "steady_state", steady_state_m_ind, E_coeffs_noind, m_imp_noind
                    )

                # Save state and steady state time series.
                if should_save_sample:
                    if run_inductive:
                        self.data.save_output_dataset("state")
                        if not quiet:
                            print(
                                "Saved state at t = {:.2f} s".format(self.current_time),
                                end="\n" if run_steady_state else "\r",
                                flush=True,
                            )

                    if run_steady_state:
                        self.data.save_output_dataset("steady_state")
                        if not quiet:
                            print(
                                "Saved steady state at t = {:.2f} s".format(self.current_time),
                                end="\x1b[F" if run_inductive else "\r",
                                flush=True,
                            )

            next_time = self.current_time + dt * step_increment

            if next_time > t + FLOAT_ERROR_MARGIN:
                if not quiet:
                    print("\n\n")
                break

            if run_inductive:
                inductive_m_ind = self.state.evolve_m_ind(
                    inductive_m_ind, dt, E_coeffs_noind, steady_state_m_ind
                )
            self.current_time = next_time

            step += step_increment

    def impose_steady_state(self, time=None, interpolation=True, save=True, quiet=False):
        """Replace the current model state with the steady state."""
        if time is not None:
            self.current_time = np.float64(time)

        self.state.update(self.input_timeseries, self.current_time, interpolation=interpolation)
        E_coeffs_noind, m_imp_noind = self.state.calculate_noind_coeffs()
        steady_state_m_ind = self.state.steady_state_m_ind(E_coeffs_noind)

        if save:
            self.add_state_to_timeseries("state", steady_state_m_ind, E_coeffs_noind, m_imp_noind)
            if bool(self.settings.save_steady_states):
                self.add_state_to_timeseries(
                    "steady_state", steady_state_m_ind, E_coeffs_noind, m_imp_noind
                )

            self.data.save_output_dataset("state")
            if bool(self.settings.save_steady_states):
                self.data.save_output_dataset("steady_state")

            if not quiet:
                print(f"Imposed steady state at t = {float(self.current_time):.2f} s")

        return steady_state_m_ind

    def add_state_to_timeseries(self, key, m_ind, E_coeffs_noind, m_imp_noind):
        """Add the current state to the time series.

        Parameters
        ----------
        key : str
            Key for the time series entry.
        m_ind : array-like
            Inductive magnetic field coefficients.
        E_coeffs_noind : tuple
            Electric field coefficients without induced effects.
        m_imp_noind : array-like
            Imposed magnetic field coefficients without induced effects.
        """
        m_ind = self.state.project_scalar_mean_free(m_ind)
        E_coeffs_ind, m_imp_ind = self.state.calculate_ind_coeffs(m_ind)

        E_coeffs = self.state.project_helmholtz_mean_free(E_coeffs_noind + E_coeffs_ind)
        m_imp = self.state.project_scalar_mean_free(m_imp_noind + m_imp_ind)

        # Append current state to time series.
        state_data = {
            "m_ind": to_numpy(m_ind),
            "m_imp": to_numpy(m_imp),
            "Phi": to_numpy(
                self.state.geometry.helmholtz_curl_free_potential_operator.matvec(
                    E_coeffs
                )
            ),
            "W": to_numpy(
                self.state.geometry.helmholtz_divergence_free_potential_operator.matvec(
                    E_coeffs
                )
            ),
        }

        self.data.add_output_entry(key, state_data, time=self.current_time)

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
        FAC_b_evaluator = FieldEvaluator(
            self.mainfield, Grid(lat=lat, lon=lon, theta=theta, phi=phi), self.settings.RI
        )

        self.set_jr(
            FAC * FAC_b_evaluator.br,
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
        *,
        coefficients=False,
    ):
        """Set radial current density input.

        Parameters
        ----------
        jr : array-like
            Radial current density in A/m², or storage-basis
            coefficients if ``coefficients=True``.
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
        coefficients : bool, optional
            If True, ``jr`` is already in the input storage basis and is
            stored directly without interpolation or projection.
        """
        input_data = {"jr": np.atleast_2d(jr)}

        if coefficients:
            self._add_input_coefficients("jr", input_data, time)
            return

        self._project_and_add_input(
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
        *,
        coefficients=False,
    ):
        """Set radial component of magnetic field input.

        Parameters
        ----------
        Br : array-like
            Radial component of magnetic field, or storage-basis
            coefficients if ``coefficients=True``.
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
        coefficients : bool, optional
            If True, ``Br`` is already in the input storage basis and is
            stored directly without interpolation or projection.
        """
        if self.settings.RM == 0:
            raise ValueError("Br can only be set if magnetospheric radius (RM) is set.")

        input_data = {"Br": np.atleast_2d(Br)}

        if coefficients:
            self._add_input_coefficients("Br", input_data, time)
            return

        self._project_and_add_input(
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

    def set_resistance(
        self,
        Pedersen,
        Hall,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        coefficients=False,
        project=None,
    ):
        """Set Pedersen and Hall resistance inputs.

        Parameters
        ----------
        Pedersen : array-like
            Pedersen resistance values, or storage-basis coefficients if
            ``coefficients=True``.
        Hall : array-like
            Hall resistance values, or storage-basis coefficients if
            ``coefficients=True``.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the resistance data.
        sqrt_weights : array-like, optional
            sqrt_weights for the resistance data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        coefficients : bool, optional
            If True, ``Pedersen`` and ``Hall`` are already in the input
            storage basis and are stored directly without interpolation
            or projection.
        project : bool, optional
            Override ``self.settings.project_conductance`` for this
            call. If False, ``Pedersen`` and ``Hall`` must be grid
            values on the state/model grid and are stored directly in
            the grid conductance basis.
        """
        input_data = {"etaP": np.atleast_2d(Pedersen), "etaH": np.atleast_2d(Hall)}

        if coefficients:
            self._add_input_coefficients("conductance", input_data, time)
            return

        if not self._should_project_conductance(project):
            self._add_native_grid_input(
                "conductance",
                input_data,
                lat=lat,
                lon=lon,
                theta=theta,
                phi=phi,
                time=time,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
            )
            return

        self._project_and_add_input(
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
        *,
        project=None,
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
        project : bool, optional
            Override ``self.settings.project_conductance`` for this
            call. If False, the conductance grid must match the
            state/model grid and the converted resistance values are
            stored directly without projection.
        """
        Hall = np.atleast_2d(Hall)
        Pedersen = np.atleast_2d(Pedersen)

        etaP = np.empty_like(Pedersen)
        etaH = np.empty_like(Hall)

        # Convert conductances to resistances for all time points.
        for i in range(max(etaP.shape[0], 1)):
            etaP[i] = Pedersen[i] / (Hall[i] ** 2 + Pedersen[i] ** 2)

        for i in range(max(etaH.shape[0], 1)):
            etaH[i] = Hall[i] / (Hall[i] ** 2 + Pedersen[i] ** 2)

        self.set_resistance(
            etaP,
            etaH,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
            project=project,
        )

    def set_neutral_wind(
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
        *,
        coefficients=False,
    ):
        """Set neutral wind velocities.

        Parameters
        ----------
        u_theta : array-like
            Meridional (south) wind velocity in m/s, or curl-free
            storage-basis coefficients if ``coefficients=True``.
        u_phi : array-like
            Zonal (east) wind velocity in m/s, or divergence-free
            storage-basis coefficients if ``coefficients=True``.
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
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        coefficients : bool, optional
            If True, ``u_theta`` and ``u_phi`` are interpreted as the
            curl-free and divergence-free Helmholtz coefficients,
            respectively, and stored directly without interpolation or
            projection.
        """
        input_data = self._wind_input_data(u_theta, u_phi)

        if coefficients:
            self._add_input_coefficients("u", input_data, time)
            return

        self._project_and_add_input(
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

    def set_u(self, *args, **kwargs):
        """Set neutral wind velocities using the historical API name."""
        return self.set_neutral_wind(*args, **kwargs)

    def _wind_input_data(self, u_theta, u_phi):
        """Return wind input data with time before component."""
        input_data = {"u": np.array([np.atleast_2d(u_theta), np.atleast_2d(u_phi)])}
        input_data["u"] = np.moveaxis(input_data["u"], [0, 1], [1, 0])
        return input_data

    def _project_and_add_input(
        self,
        key,
        input_data,
        *,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Project gridded input data and store coefficient entries."""
        input_time = self.adapt_input_time(time, input_data)
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)
        transform = self.input_transforms[key]
        field_space = self.input_field_spaces[key]
        project = (
            transform.project_helmholtz
            if field_space.field_type == "tangential"
            else transform.project_scalar
        )

        projected_data = {}
        for var, values in input_data.items():
            projected_values = project(
                values,
                input_grid=input_grid,
                projection_basis=self.interpolation_bases[key],
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
            )
            if projected_values.shape[0] != input_time.size:
                raise ValueError(
                    f"{key}.{var} has {projected_values.shape[0]} projected time "
                    f"slices, but {input_time.size} time values were supplied."
                )
            projected_data[var] = projected_values

        for time_index in range(input_time.size):
            self.input_timeseries.add_entry(
                key,
                {var: projected_data[var][time_index] for var in projected_data},
                input_time[time_index],
            )

        self.data.save_input_dataset(key)

    def _should_project_conductance(self, project):
        """Return the effective conductance projection setting."""
        if project is None:
            return bool(self.settings.project_conductance)
        return bool(project)

    def _add_native_grid_input(
        self,
        key,
        input_data,
        *,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
    ):
        """Store grid-basis input values after model-grid validation."""
        if sqrt_weights is not None:
            raise ValueError("sqrt_weights are not supported when conductance projection is off.")
        if reg_lambda is not None:
            raise ValueError("reg_lambda is not supported when conductance projection is off.")

        storage_representation = self.input_field_spaces[key].representation
        if not is_grid_basis(storage_representation):
            raise ValueError(
                "Direct grid conductance storage requires Dynamics(project_conductance=False)."
            )

        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)
        model_grid = self.state.geometry.grid
        if not input_grid.same_as(model_grid):
            raise ValueError(
                "Direct grid conductance storage requires the input grid to match "
                "the state/model grid."
            )

        if hasattr(storage_representation, "arr_theta") and hasattr(
            storage_representation, "arr_phi"
        ):
            storage_grid = Grid(
                theta=storage_representation.arr_theta,
                phi=storage_representation.arr_phi,
            )
            if not storage_grid.same_as(model_grid):
                raise ValueError(
                    "Conductance storage basis grid does not match the state/model grid."
                )

        transform = self.input_transforms[key]
        normalize = (
            transform.normalize_helmholtz_value_batch
            if self.input_field_spaces[key].field_type == "tangential"
            else transform.normalize_scalar_value_batch
        )
        direct_data = {
            var: normalize(values, input_grid)
            for var, values in input_data.items()
        }
        input_time = self.adapt_input_time(time, direct_data)

        for var, values in direct_data.items():
            if values.shape[0] != input_time.size:
                raise ValueError(
                    f"{key}.{var} has {values.shape[0]} grid time slices, "
                    f"but {input_time.size} time values were supplied."
                )

        for time_index in range(input_time.size):
            self.input_timeseries.add_entry(
                key,
                {var: direct_data[var][time_index] for var in direct_data},
                input_time[time_index],
            )

        self.data.save_input_dataset(key)

    def _add_input_coefficients(self, key, input_data, time):
        """Store input-basis coefficients directly in a time series."""
        input_time = self.adapt_input_time(time, input_data)

        for time_index in range(input_time.size):
            self.input_timeseries.add_entry(
                key,
                {var: input_data[var][time_index] for var in input_data},
                input_time[time_index],
            )

        self.data.save_input_dataset(key)

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
