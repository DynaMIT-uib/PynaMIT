"""User-facing input preparation and simulation objects."""

import numpy as np
from kompe.constants import EARTH_RADIUS_M
from kompe.math import set_backend

from pynamit.simulation.config import SimulationConfig
from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.simulation.geometry import SimulationGeometry, build_main_field
from pynamit.simulation.input_manifest import write_input_manifest
from pynamit.simulation.inputs import InputPipeline
from pynamit.simulation.response import ElectrodynamicResponse
from pynamit.simulation.runner import (
    DEFAULT_DT_SECONDS,
    DEFAULT_SAMPLING_STEP_INTERVAL,
    DEFAULT_WRITE_SAMPLE_INTERVAL,
    SimulationRunner,
)
from pynamit.simulation.schema import INPUT_DATASET_KEYS
from pynamit.simulation.simulation_data import SimulationData
from pynamit.storage import ArrayCache, ArtifactStore


class InputPreparation:
    """Project and persist inputs for a later PynaMIT simulation.

    Input values can be supplied as samples or as coefficients through
    the ``set_*`` methods. Sampled values are projected immediately and
    the resulting coefficient time series are saved in
    ``input_directory``. The preparation object does not construct the
    time-evolution runner.

    Attributes
    ----------
    config : SimulationConfig
        Coefficient-space and geometry configuration.
    data : SimulationData
        Persisted settings, schema, and input time series.
    input_directory : str
        Directory containing the prepared input artifacts.
    model_grid : kompe.SphericalGrid
        Grid on which sampled input fields are projected.
    inputs : dict
        Projected input datasets, keyed by stream name.
    geometry : SimulationGeometry
        Spatial realization used while preparing the inputs.
    operator_cache : pynamit.storage.ArrayCache, optional
        Shared cache for deterministic materialized operators.
    """

    def __init__(
        self,
        input_directory=None,
        Nmax=20,
        Mmax=20,
        Ncs=30,
        RI=EARTH_RADIUS_M + 110.0e3,
        RM=None,
        main_field_kind="dipole",
        main_field_epoch=None,
        main_field_B0=None,
        boundary_jr_projection_basis=None,
        boundary_Br_projection_basis=None,
        conductance_projection_basis=None,
        u_projection_basis=None,
        Q_eff_projection_basis=None,
        E_neutral_wind_projection_basis=None,
        t0="2020-01-01 00:00:00",
        artifact_storage="auto",
        operator_cache_directory=None,
        backend="auto",
        horizontal_basis_kind="SH",
        area_weighted_least_squares=False,
    ):
        """Initialize an input preparation.

        Parameters
        ----------
        input_directory : path-like, optional
            Directory in which projected input coefficients are stored.
        Nmax : int, optional
            Maximum spherical harmonic degree.
        Mmax : int, optional
            Maximum spherical harmonic order.
        Ncs : int, optional
            Number of cubed sphere grid points per edge.
        RI : float, optional
            Ionospheric radius in meters.
        RM : float, optional
            Magnetospheric boundary radius in meters.
        main_field_kind : {'dipole', 'kaiju_dipole', 'igrf', 'radial'}
            Type of main magnetic field model.
        main_field_epoch : float, optional
            Decimal year for the main field. Defaults to the decimal
            year of ``t0``.
        main_field_B0 : float, optional
            Main field strength.
        boundary_jr_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting radial-current inputs.
            Defaults to ``horizontal_basis_kind``.
        boundary_Br_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting radial magnetic-field
            inputs. Defaults to ``horizontal_basis_kind``.
        conductance_projection_basis : {'SH', 'CS'}, optional
            Basis used to store the dimensionless log conductance
            magnitude and log Hall/Pedersen ratio. ``'CS'`` makes
            matching model-grid inputs a no-op. Defaults to
            ``horizontal_basis_kind``.
        u_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting neutral-wind inputs.
            Defaults to ``horizontal_basis_kind``.
        Q_eff_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting effective wind-current
            inputs. Defaults to ``u_projection_basis``.
        E_neutral_wind_projection_basis : {'SH', 'CS'}, optional
            Basis route used when projecting equivalent neutral-wind
            electric fields. Defaults to ``horizontal_basis_kind``.
        t0 : str, optional
            Start time in UTC format.
        artifact_storage : {'auto', 'netcdf', 'zarr'}, optional
            Preferred backend for new saved xarray artifacts. Existing
            artifacts keep their format on restart.
        operator_cache_directory : path-like, optional
            Shared content-addressed cache for expensive deterministic
            numerical arrays, including spherical-harmonic evaluation
            matrices. This runtime optimization is not part of the
            persisted physical configuration.
        backend : {'auto', 'numpy', 'jax', bool}, optional
            Array backend to use. ``"auto"`` respects the current global
            setting (environment variable or previous choice).
            ``"numpy"``/``False`` enforce NumPy arrays, while
            ``"jax"``/``True`` enables JAX.
        horizontal_basis_kind : {'SH', 'CS'}, optional
            Basis requested for horizontal surface potentials and
            operators. ``'SH'`` is the default. ``'CS'`` uses
            cubed-sphere nodal coefficients and finite-difference
            derivatives. ``induced_Br`` and radial continuation
            remain in the configured mean-free poloidal SH space.
        area_weighted_least_squares : bool, optional
            Use surface-area weights for least-squares projections when
            no explicit ``sqrt_weights`` are supplied. Cubed-sphere
            grids use their native cell areas; ordinary spherical grids
            use ``sin(theta)``. Disabled by default to preserve the
            established projection norm.
        """
        config = SimulationConfig(
            Nmax=Nmax,
            Mmax=Mmax,
            Ncs=Ncs,
            RI=RI,
            RM=RM,
            enable_pfac_coupling=False,
            main_field_kind=main_field_kind,
            main_field_epoch=main_field_epoch,
            main_field_B0=main_field_B0,
            boundary_jr_projection_basis=boundary_jr_projection_basis,
            boundary_Br_projection_basis=boundary_Br_projection_basis,
            conductance_projection_basis=conductance_projection_basis,
            u_projection_basis=u_projection_basis,
            Q_eff_projection_basis=Q_eff_projection_basis,
            E_neutral_wind_projection_basis=E_neutral_wind_projection_basis,
            horizontal_basis_kind=horizontal_basis_kind,
            area_weighted_least_squares=area_weighted_least_squares,
            t0=t0,
        )
        self._open_input_preparation(
            config,
            directory=input_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
            backend=backend,
        )

    def _open_input_preparation(
        self, config, *, directory, artifact_storage, operator_cache_directory, backend
    ):
        """Open the projection state shared with ``Simulation``."""
        set_backend(backend)
        self.operator_cache = (
            None if operator_cache_directory is None else ArrayCache(operator_cache_directory)
        )
        self.data = SimulationData.open(
            config,
            simulation_directory=directory,
            artifact_storage=artifact_storage,
            operator_cache=self.operator_cache,
            print_info=True,
        )
        self.config = self.data.config
        self.input_directory = self.data.simulation_directory
        self.inputs = self.data.input_series.datasets
        schema = self.data.schema
        main_field = build_main_field(self.config)

        self.geometry = SimulationGeometry(
            schema.horizontal_basis,
            schema.cs_basis,
            main_field,
            self.config,
            gap_Br_response_matrix=self.data.gap_Br_response,
            solid_harmonics=schema.solid_harmonics,
            operator_cache=self.operator_cache,
        )
        self.model_grid = self.geometry.model_grid
        self._input_pipeline = InputPipeline(self)
        self._response = None
        self.current_time = np.float64(0)

        self.data.save_settings_if_missing(print_info=True)

    def __repr__(self):
        """Summarize projected inputs for interactive sessions."""
        inputs = ", ".join(sorted(self.inputs)) or "none"
        return (
            f"InputPreparation(Nmax={self.config.Nmax}, Mmax={self.config.Mmax}, "
            f"Ncs={self.config.Ncs}, inputs=[{inputs}], "
            f"input_directory={self.input_directory!r})"
        )

    def _require_response(self):
        """Construct the electrodynamic response when it is needed.

        Ordinary projection does not need it. The response is required
        only when deriving ``Q_eff`` from neutral wind and conductance.
        """
        if self._response is None:
            self._response = ElectrodynamicResponse(self.geometry, self.config)
        return self._response

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        *,
        input_directory=None,
        artifact_storage="auto",
        operator_cache_directory=None,
        backend="auto",
    ):
        """Construct an input preparation from normalized configuration.

        The input directory, artifact storage, operator-cache directory,
        and backend are runtime preferences rather than persisted model
        settings.
        """
        if not isinstance(config, SimulationConfig):
            raise TypeError("InputPreparation.from_config requires a SimulationConfig.")
        preparation = cls.__new__(cls)
        preparation._open_input_preparation(
            config,
            directory=input_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
            backend=backend,
        )
        return preparation

    @classmethod
    def from_directory(cls, input_directory, **kwargs):
        """Open an existing prepared-input directory."""
        input_directory = ArtifactStore.require_artifact_directory(input_directory, ("settings",))
        artifact_storage = kwargs.get("artifact_storage", "auto")
        settings = ArtifactStore(
            input_directory, preferred_dataset_storage=artifact_storage
        ).load_dataset("settings")

        stored_config = SimulationConfig.from_settings(settings)
        config_values = stored_config.to_kwargs()
        config_overrides = {
            name: value
            for name, value in kwargs.items()
            if name in config_values and value is not None
        }
        config = SimulationConfig(**{**config_values, **config_overrides})
        runtime_kwargs = {
            name: value for name, value in kwargs.items() if name not in config_values
        }
        return cls.from_config(config, input_directory=input_directory, **runtime_kwargs)

    def write_manifest(self, *, source="manual", notes=(), metadata=None):
        """Write the manifest for this reusable input package."""
        return write_input_manifest(
            self.input_directory,
            self.config,
            input_datasets=tuple(key for key in INPUT_DATASET_KEYS if key in self.inputs),
            source=source,
            notes=notes,
            metadata=metadata,
        )

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

        Projects FAC onto the radial direction with the background-field
        direction cosine, then stores that radial current density.
        Positive FAC follows the background magnetic-field vector; it is
        not defined as upward in both hemispheres.

        Parameters
        ----------
        FAC : array-like
            Signed field-parallel current density in A/m². Positive
            values follow the background magnetic-field vector.
        lat, lon : array-like, optional
            Latitude/longitude in the simulation model frame, in
            degrees.
        theta, phi : array-like, optional
            Model-frame colatitude/azimuth in degrees.
        time : array-like, optional
            Time points for the FAC data.
        sqrt_weights : array-like, optional
            sqrt_weights for the FAC data points.
        reg_lambda : float, optional
            Regularization parameter for the least squares solver.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        radial_current = self._input_pipeline.radial_current_from_FAC(
            FAC, lat=lat, lon=lon, theta=theta, phi=phi
        )

        self.set_boundary_jr(
            radial_current,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_boundary_jr(
        self,
        boundary_jr=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        boundary_jr_coefficients=None,
    ):
        """Set radial current at the upper ionospheric boundary.

        Parameters
        ----------
        boundary_jr : array-like
            Radial current density in A/m².
        boundary_jr_coefficients : array-like, optional
            Radial-current coefficients in the input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude in the simulation model frame, in
            degrees.
        theta, phi : array-like, optional
            Model-frame colatitude/azimuth in degrees.
        time : array-like, optional
            Time points for the current data.
        sqrt_weights : array-like, optional
            sqrt_weights for the current data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self._input_pipeline.set_scalar_input(
            "boundary_jr",
            samples={"boundary_jr": boundary_jr},
            coefficients={"boundary_jr": boundary_jr_coefficients},
            sample_label="boundary_jr samples",
            coefficient_label="boundary_jr_coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_boundary_Br(
        self,
        boundary_Br=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        boundary_Br_coefficients=None,
    ):
        """Set radial magnetic field at the outer boundary.

        Parameters
        ----------
        boundary_Br : array-like
            Radial component of magnetic field.
        boundary_Br_coefficients : array-like, optional
            Radial magnetic-field coefficients in the input storage
            basis.
        lat, lon : array-like, optional
            Latitude/longitude in the simulation model frame, in
            degrees.
        theta, phi : array-like, optional
            Model-frame colatitude/azimuth in degrees.
        time : array-like, optional
            Time points for the magnetic-field data.
        sqrt_weights : array-like, optional
            sqrt_weights for the magnetic-field data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        if self.config.RM is None:
            raise ValueError("boundary_Br can only be set if magnetospheric radius (RM) is set.")

        self._input_pipeline.set_scalar_input(
            "boundary_Br",
            samples={"boundary_Br": boundary_Br},
            coefficients={"boundary_Br": boundary_Br_coefficients},
            sample_label="boundary_Br samples",
            coefficient_label="boundary_Br_coefficients",
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
        *,
        etaP=None,
        etaH=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set positive resistance through canonical coordinates.

        Parameters
        ----------
        etaP : array-like
            Strictly positive Pedersen resistance values.
        etaH : array-like
            Strictly positive Hall resistance values.
        lat, lon : array-like, optional
            Latitude/longitude in the simulation model frame, in
            degrees.
        theta, phi : array-like, optional
            Model-frame colatitude/azimuth in degrees.
        time : array-like, optional
            Time points for the resistance data.
        sqrt_weights : array-like, optional
            sqrt_weights for the resistance data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self._input_pipeline.require_complete_values("resistance samples", etaP=etaP, etaH=etaH)
        log_magnitude, log_ratio = ionospheric_closure.resistance_to_log_conductance_coordinates(
            etaP, etaH
        )
        self._store_conductance_coordinates(
            log_magnitude,
            log_ratio,
            sample_label="resistance samples",
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
        *,
        pedersen=None,
        hall=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        log_magnitude_coefficients=None,
        log_ratio_coefficients=None,
    ):
        """Set positive conductance through canonical log coordinates.

        Sampled Pedersen/Hall values are converted to
        ``log(hypot(SigmaP, SigmaH) / 1 S)`` and
        ``log(SigmaH / SigmaP)`` before projection. This guarantees
        positive reconstructed conductance and treats the reciprocal
        resistance tensor symmetrically.

        Parameters
        ----------
        pedersen : array-like
            Strictly positive Pedersen conductance in siemens.
        hall : array-like
            Strictly positive Hall conductance in siemens.
        log_magnitude_coefficients, log_ratio_coefficients
            : array-like, optional
            Canonical conductance coordinates already represented in
            the configured input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude in the simulation model frame, in
            degrees.
        theta, phi : array-like, optional
            Model-frame colatitude/azimuth in degrees.
        time : array-like, optional
            Time points for the conductance data.
        sqrt_weights : array-like, optional
            sqrt_weights for the conductance data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        coefficients_supplied = (
            log_magnitude_coefficients is not None or log_ratio_coefficients is not None
        )
        if coefficients_supplied:
            log_magnitude = None
            log_ratio = None
        else:
            self._input_pipeline.require_complete_values(
                "conductance samples", pedersen=pedersen, hall=hall
            )
            log_magnitude, log_ratio = ionospheric_closure.conductance_to_log_coordinates(
                pedersen, hall
            )

        self._store_conductance_coordinates(
            log_magnitude,
            log_ratio,
            log_magnitude_coefficients=log_magnitude_coefficients,
            log_ratio_coefficients=log_ratio_coefficients,
            sample_label="conductance samples",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def _store_conductance_coordinates(
        self,
        log_magnitude,
        log_ratio,
        *,
        log_magnitude_coefficients=None,
        log_ratio_coefficients=None,
        sample_label,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Store canonical conductance samples or coefficients."""
        self._input_pipeline.set_scalar_input(
            "conductance",
            samples={
                "log_conductance_magnitude": log_magnitude,
                "log_hall_to_pedersen_ratio": log_ratio,
            },
            coefficients={
                "log_conductance_magnitude": log_magnitude_coefficients,
                "log_hall_to_pedersen_ratio": log_ratio_coefficients,
            },
            sample_label=sample_label,
            coefficient_label="log-conductance coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_neutral_wind(
        self,
        u_theta=None,
        u_phi=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        u_cf=None,
        u_df=None,
    ):
        """Set neutral wind velocities.

        Parameters
        ----------
        u_theta : array-like
            Meridional (south) wind velocity in m/s.
        u_phi : array-like
            Zonal (east) wind velocity in m/s.
        u_cf, u_df : array-like, optional
            Curl-free and divergence-free Helmholtz coefficients in the
            input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude in the simulation model frame, in
            degrees.
        theta, phi : array-like, optional
            Model-frame colatitude/azimuth in degrees.
        time : array-like, optional
            Time points for the wind data.
        sqrt_weights : array-like, optional
            sqrt_weights for the wind data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self._input_pipeline.set_tangential_input(
            "u",
            theta_component=u_theta,
            phi_component=u_phi,
            cf_coefficients=u_cf,
            df_coefficients=u_df,
            sample_label="wind samples",
            coefficient_label="wind coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_Q_eff(
        self,
        Q_eff_theta=None,
        Q_eff_phi=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        Q_eff_cf=None,
        Q_eff_df=None,
    ):
        """Set effective wind-current input.

        ``Q_eff`` is a tangential JS proxy for the neutral
        wind ``u x B`` term. It is added to the JS before
        the conductance/resistance tensor maps currents to the electric
        field.

        Parameters
        ----------
        Q_eff_theta : array-like
            Southward effective-current component in A/m.
        Q_eff_phi : array-like
            Eastward effective-current component in A/m.
        Q_eff_cf, Q_eff_df : array-like, optional
            Curl-free and divergence-free Helmholtz coefficients in the
            input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude in the simulation model frame, in
            degrees.
        theta, phi : array-like, optional
            Model-frame colatitude/azimuth in degrees.
        time : array-like, optional
            Time points for the Q_eff data.
        sqrt_weights : array-like, optional
            sqrt_weights for the Q_eff data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self._input_pipeline.set_tangential_input(
            "Q_eff",
            theta_component=Q_eff_theta,
            phi_component=Q_eff_phi,
            cf_coefficients=Q_eff_cf,
            df_coefficients=Q_eff_df,
            sample_label="Q_eff samples",
            coefficient_label="Q_eff coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_E_neutral_wind(
        self,
        E_neutral_wind_theta=None,
        E_neutral_wind_phi=None,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
        *,
        E_neutral_wind_cf=None,
        E_neutral_wind_df=None,
    ):
        """Set an equivalent neutral-wind electric-field input.

        ``E_neutral_wind`` is added to the non-induced electric field
        before the imposed-current coupling is solved. It is an electric
        field in V/m, not an effective current. Use this route for
        externally prepared neutral-wind electrodynamics, such as
        separate Pedersen- and Hall-weighted winds. It is an alternative
        to ``set_neutral_wind`` and ``set_Q_eff``; use only one of the
        three representations.

        Parameters
        ----------
        E_neutral_wind_theta : array-like
            Southward neutral-wind electric-field component in V/m.
        E_neutral_wind_phi : array-like
            Eastward neutral-wind electric-field component in V/m.
        E_neutral_wind_cf, E_neutral_wind_df : array-like, optional
            Curl-free and divergence-free Helmholtz coefficients in the
            input storage basis.
        lat, lon : array-like, optional
            Latitude/longitude in the simulation model frame, in
            degrees.
        theta, phi : array-like, optional
            Model-frame colatitude/azimuth in degrees.
        time : array-like, optional
            Time points for the neutral-wind electric-field data.
        sqrt_weights : array-like, optional
            Square-root weights for the neutral-wind electric-field
            samples.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        self._input_pipeline.set_tangential_input(
            "E_neutral_wind",
            theta_component=E_neutral_wind_theta,
            phi_component=E_neutral_wind_phi,
            cf_coefficients=E_neutral_wind_cf,
            df_coefficients=E_neutral_wind_df,
            sample_label="neutral-wind electric-field samples",
            coefficient_label="neutral-wind electric-field coefficients",
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_Q_eff_from_neutral_wind(
        self,
        u_theta,
        u_phi,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        wind_reg_lambda=None,
        Q_eff_reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Fit and store Q_eff coefficients from wind and conductance.

        The wind is projected once, then ``Q_eff`` is solved in its
        storage basis so that the resistance-weighted electric response
        matches the wind forcing. Use
        ``calculate_Q_eff_from_neutral_wind`` to inspect the equivalent
        model-grid field without storing it.
        """
        self._input_pipeline.require_no_exclusive_conflict("Q_eff")
        input_time, wind_coeff_rows = self._input_pipeline.project_tangential_samples(
            "u",
            u_theta,
            u_phi,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=wind_reg_lambda,
            pinv_rtol=pinv_rtol,
        )
        q_coeff_rows = self._input_pipeline.fit_Q_eff_from_neutral_wind(
            input_time, wind_coeff_rows, reg_lambda=Q_eff_reg_lambda, pinv_rtol=pinv_rtol
        )
        self._input_pipeline.add_input_coefficients("Q_eff", {"Q_eff": q_coeff_rows}, input_time)

    def calculate_Q_eff_from_neutral_wind(
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
        """Return model-grid Q_eff equivalent to wind forcing."""
        if "conductance" not in self.data.input_series.datasets:
            raise RuntimeError(
                "Ionospheric resistance or conductance must be set before "
                "calculating Q_eff from wind."
            )

        input_time, wind_coeff_rows = self._input_pipeline.project_tangential_samples(
            "u",
            u_theta,
            u_phi,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )
        return self._input_pipeline.evaluate_Q_eff_from_neutral_wind(input_time, wind_coeff_rows)


class Simulation(InputPreparation):
    """Configure, drive, evolve, and persist one coupled MIT simulation.

    A simulation supports the same input setters as
    :class:`InputPreparation`, then adds the electrodynamic response and
    time-evolution runner.
    """

    def __init__(
        self,
        simulation_directory=None,
        Nmax=20,
        Mmax=20,
        Ncs=30,
        RI=EARTH_RADIUS_M + 110.0e3,
        RM=None,
        magnetic_boundary_shielding=False,
        main_field_kind="dipole",
        main_field_epoch=None,
        main_field_B0=None,
        fac_integration_radii=None,
        enable_pfac_coupling=True,
        enable_interhemispheric_coupling=False,
        interhemispheric_coupling_latitude=50,
        interhemispheric_electric_field_weight=1e-5,
        boundary_jr_projection_basis=None,
        boundary_Br_projection_basis=None,
        conductance_projection_basis=None,
        u_projection_basis=None,
        Q_eff_projection_basis=None,
        E_neutral_wind_projection_basis=None,
        t0="2020-01-01 00:00:00",
        save_equilibria=True,
        integrator="euler",
        least_squares_solver=None,
        least_squares_preconditioner="pinv",
        reuse_preconditioner=False,
        toroidal_potential_regularization_lambda=0.0,
        artifact_storage="auto",
        operator_cache_directory=None,
        backend="auto",
        horizontal_basis_kind="SH",
        area_weighted_least_squares=False,
    ):
        """Initialize a coupled MIT simulation.

        Parameters
        ----------
        simulation_directory : path-like, optional
            Directory for persisted settings, inputs, and outputs.
        Nmax, Mmax, Ncs : int, optional
            Spherical-harmonic truncation and cubed-sphere resolution.
        RI, RM : float, optional
            Ionospheric and magnetospheric radii in meters.
        magnetic_boundary_shielding : bool, optional
            Impose a shielding condition at ``RM``.
        main_field_kind, main_field_epoch, main_field_B0 : optional
            Main-field model, decimal-year epoch, and optional field
            magnitude override.
        fac_integration_radii : array-like, optional
            Radii used to integrate the FAC poloidal field.
        enable_pfac_coupling : bool, optional
            Include the FAC poloidal field in the coupled response.
        enable_interhemispheric_coupling : bool, optional
            Couple conjugate current and electric-field solutions.
        interhemispheric_coupling_latitude : float, optional
            Absolute latitude bounding the coupled low-latitude region.
        interhemispheric_electric_field_weight : float, optional
            Relative least-squares weight of the conjugate constraint.
        boundary_jr_projection_basis, boundary_Br_projection_basis,
        conductance_projection_basis, u_projection_basis,
        Q_eff_projection_basis, E_neutral_wind_projection_basis :
            Input storage bases. Each defaults to the corresponding
            choice derived by :class:`SimulationConfig`.
        t0 : str, optional
            Physical start time.
        save_equilibria : bool, optional
            Save instantaneous induction equilibria by default.
        integrator : str, optional
            Integrator used for ``induced_Br`` evolution.
        least_squares_solver, least_squares_preconditioner : optional
            Toroidal-potential solver and preconditioner.
        reuse_preconditioner : bool, optional
            Reuse a compatible iterative-solver preconditioner.
        toroidal_potential_regularization_lambda : float, optional
            Regularization strength for the toroidal-potential solve.
        artifact_storage : {'auto', 'netcdf', 'zarr'}, optional
            Preferred storage backend for newly saved artifacts.
        operator_cache_directory : path-like, optional
            Shared cache for deterministic numerical operators.
        backend : {'auto', 'numpy', 'jax', bool}, optional
            Array backend.
        horizontal_basis_kind : {'SH', 'CS'}, optional
            Horizontal surface basis.
        area_weighted_least_squares : bool, optional
            Use surface-area weights for projections without explicit
            weights.
        """
        config = SimulationConfig(
            Nmax=Nmax,
            Mmax=Mmax,
            Ncs=Ncs,
            RI=RI,
            RM=RM,
            magnetic_boundary_shielding=magnetic_boundary_shielding,
            interhemispheric_coupling_latitude=interhemispheric_coupling_latitude,
            enable_pfac_coupling=enable_pfac_coupling,
            enable_interhemispheric_coupling=enable_interhemispheric_coupling,
            fac_integration_radii=fac_integration_radii,
            interhemispheric_electric_field_weight=interhemispheric_electric_field_weight,
            main_field_kind=main_field_kind,
            main_field_epoch=main_field_epoch,
            main_field_B0=main_field_B0,
            boundary_jr_projection_basis=boundary_jr_projection_basis,
            boundary_Br_projection_basis=boundary_Br_projection_basis,
            conductance_projection_basis=conductance_projection_basis,
            u_projection_basis=u_projection_basis,
            Q_eff_projection_basis=Q_eff_projection_basis,
            E_neutral_wind_projection_basis=E_neutral_wind_projection_basis,
            horizontal_basis_kind=horizontal_basis_kind,
            area_weighted_least_squares=area_weighted_least_squares,
            t0=t0,
            save_equilibria=save_equilibria,
            integrator=integrator,
            least_squares_solver=least_squares_solver,
            least_squares_preconditioner=least_squares_preconditioner,
            reuse_preconditioner=reuse_preconditioner,
            toroidal_potential_regularization_lambda=toroidal_potential_regularization_lambda,
        )
        self._open_input_preparation(
            config,
            directory=simulation_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
            backend=backend,
        )
        self._open_simulation_runtime()

    def _open_simulation_runtime(self):
        """Open outputs and the runner after the shared input state."""
        self.simulation_directory = self.data.simulation_directory
        self.outputs = self.data.output_series.datasets
        self._require_response()
        current_output = self.outputs.get("dynamic", self.outputs.get("equilibrium"))
        self.current_time = (
            np.max(current_output.time.values) if current_output is not None else np.float64(0)
        )
        self._runner = SimulationRunner(self)

    def __repr__(self):
        """Summarize the live simulation for interactive sessions."""
        inputs = ", ".join(sorted(self.inputs)) or "none"
        outputs = ", ".join(sorted(self.outputs)) or "none"
        return (
            f"Simulation(Nmax={self.config.Nmax}, Mmax={self.config.Mmax}, "
            f"Ncs={self.config.Ncs}, current_time={float(self.current_time):g}, "
            f"inputs=[{inputs}], outputs=[{outputs}], "
            f"simulation_directory={self.simulation_directory!r})"
        )

    @property
    def response(self):
        """Return the lazily constructed electrodynamic response."""
        return self._require_response()

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        *,
        simulation_directory=None,
        artifact_storage="auto",
        operator_cache_directory=None,
        backend="auto",
    ):
        """Construct a simulation from a normalized configuration."""
        if not isinstance(config, SimulationConfig):
            raise TypeError("Simulation.from_config requires a SimulationConfig.")
        simulation = cls.__new__(cls)
        simulation._open_input_preparation(
            config,
            directory=simulation_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
            backend=backend,
        )
        simulation._open_simulation_runtime()
        return simulation

    @classmethod
    def from_directory(cls, simulation_directory, **kwargs):
        """Construct a simulation from one simulation directory."""
        simulation_directory = ArtifactStore.require_artifact_directory(
            simulation_directory, ("settings",)
        )
        artifact_storage = kwargs.get("artifact_storage", "auto")
        settings = ArtifactStore(
            simulation_directory, preferred_dataset_storage=artifact_storage
        ).load_dataset("settings")

        stored_config = SimulationConfig.from_settings(settings)
        config_values = stored_config.to_kwargs()
        config_overrides = {
            name: value
            for name, value in kwargs.items()
            if name in config_values and value is not None
        }
        config = SimulationConfig(**{**config_values, **config_overrides})
        runtime_kwargs = {
            name: value for name, value in kwargs.items() if name not in config_values
        }
        return cls.from_config(config, simulation_directory=simulation_directory, **runtime_kwargs)

    def evolve_to_time(
        self,
        t,
        dt=DEFAULT_DT_SECONDS,
        sampling_step_interval=DEFAULT_SAMPLING_STEP_INTERVAL,
        write_sample_interval=DEFAULT_WRITE_SAMPLE_INTERVAL,
        quiet=False,
        initialize_from_equilibrium=True,
        run_dynamic=True,
        run_equilibrium=None,
    ):
        """Evolve the inductive solution to ``t`` seconds after ``t0``.

        ``sampling_step_interval`` controls how often output is
        retained; ``write_sample_interval`` controls how many retained
        samples are accumulated between persistence writes.
        """
        return self._runner.evolve_to_time(
            t,
            dt=dt,
            sampling_step_interval=sampling_step_interval,
            write_sample_interval=write_sample_interval,
            quiet=quiet,
            initialize_from_equilibrium=initialize_from_equilibrium,
            run_dynamic=run_dynamic,
            run_equilibrium=run_equilibrium,
        )

    def impose_equilibrium(self, time=None, interpolation=True, save=True, quiet=False):
        """Solve the instantaneous induction equilibrium."""
        return self._runner.impose_equilibrium(
            time=time, interpolation=interpolation, save=save, quiet=quiet
        )


__all__ = ["InputPreparation", "Simulation"]
