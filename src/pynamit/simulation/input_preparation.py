"""Project and persist physical inputs for later simulations."""

from collections.abc import Mapping
from functools import cached_property

from kompe.cache import PersistentArrayCache
from kompe.constants import EARTH_RADIUS_M
from kompe.math import get_array_module

from pynamit.simulation.config import SIMULATION_SCHEMA_VERSION, SimulationConfig
from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.simulation.geometry import SimulationGeometry
from pynamit.simulation.input_manifest import write_input_manifest
from pynamit.simulation.input_projection import _InputProjector, _scalar_sample_rows
from pynamit.simulation.schema import (
    INPUT_DATASET_KEYS,
    WIND_FORCING_INPUTS,
    build_simulation_schema,
)
from pynamit.storage import ArtifactStore


class InputPreparation(Mapping):
    """Project and persist inputs for a later PynaMIT simulation.

    Physical ``set_*`` methods take samples on a ``SphericalGrid``;
    ``set_coefficients`` accepts already-projected arrays. Samples
    are projected immediately and
    the resulting coefficient time series stay in memory unless an
    ``input_directory`` is supplied. A directory enables automatic
    persistence; ``save(directory)`` can attach one later. The object
    also provides read-only mapping access to its editable xarray
    datasets. It does not construct time-evolution state.

    Sample fields may retain the broadcast coordinate shape, such as
    ``(latitude, longitude)``, or use a flat point axis. For a time
    series, add one leading or trailing time axis and supply ``time``.
    Vector setters take the two components separately in this layout.
    A single untimed sample defaults to zero seconds after ``t0``.
    Preparation has no live simulation clock; to update a running
    experiment, pass ``time=simulation.current_time`` explicitly.

    Attributes
    ----------
    config : SimulationConfig
        Coefficient-space and geometry configuration.
    schema : pynamit.simulation.schema.SimulationSchema
        Coefficient spaces and physical storage metadata.
    model_grid : kompe.SphericalGrid
        Grid on which sampled input fields are projected.
    main_field : pynamit.geomagnetism.MainField
        Background magnetic field used to interpret physical inputs.
    geometry : SimulationGeometry
        Reusable bases, transforms, and main field. Expensive physical
        operators are constructed only when first used.
    operator_cache : kompe.cache.PersistentArrayCache, optional
        Shared cache for deterministic materialized operators.
    """

    def __init__(
        self,
        input_directory=None,
        *,
        Nmax=20,
        Mmax=20,
        Ncs=30,
        RI=EARTH_RADIUS_M + 110.0e3,
        RM=None,
        main_field_kind="dipole",
        main_field_epoch=None,
        main_field_B0=None,
        boundary_jr_remapping=None,
        boundary_Br_remapping=None,
        conductance_basis=None,
        u_remapping=None,
        Q_eff_remapping=None,
        E_neutral_wind_remapping=None,
        t0="2020-01-01 00:00:00",
        artifact_storage="auto",
        operator_cache_directory=None,
        horizontal_basis_kind="SH",
        area_weighted_least_squares=False,
        least_squares_solver=None,
        least_squares_tolerance=1e-15,
        least_squares_preconditioner=None,
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
            Number of cells along each cubed-sphere face edge.
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
        boundary_jr_remapping : {'direct', 'CS'}, optional
            Sample remapping before fitting radial-current inputs.
            Defaults to direct fitting for SH and CS remapping for CS.
        boundary_Br_remapping : {'direct', 'CS'}, optional
            Sample remapping before fitting radial magnetic-field
            inputs. Defaults to direct fitting for SH; remapping for CS.
        conductance_basis : {'SH', 'CS'}, optional
            Basis used to store the dimensionless log conductance
            magnitude and log Hall/Pedersen ratio. ``'CS'`` makes
            matching model-grid inputs a no-op. Defaults to
            ``horizontal_basis_kind``.
        u_remapping : {'direct', 'CS'}, optional
            Sample remapping before fitting neutral-wind inputs.
            Defaults to direct fitting for SH and CS remapping for CS.
        Q_eff_remapping : {'direct', 'CS'}, optional
            Sample remapping before fitting effective wind-current
            inputs. Defaults to ``u_remapping``.
        E_neutral_wind_remapping : {'direct', 'CS'}, optional
            Sample remapping before fitting equivalent neutral-wind
            electric fields. Defaults to direct fitting for SH;
            CS remapping otherwise.
        t0 : str, optional
            Start time in UTC format.
        artifact_storage : {'auto', 'netcdf', 'zarr'}, optional
            Preferred backend for new saved xarray artifacts. Existing
            artifacts keep their format on restart.
        operator_cache_directory : path-like, optional
            Shared content-addressed cache for expensive deterministic
            numerical arrays, including spherical-harmonic evaluation
            arrays. This runtime optimization is not part of the
            persisted physical configuration.
        horizontal_basis_kind : {'SH', 'CS'}, optional
            Basis requested for horizontal surface potentials and
            operators. ``'SH'`` is the default. ``'CS'`` uses
            cubed-sphere nodal coefficients and finite-difference
            derivatives. ``induced_Br`` and radial continuation
            remain in the configured mean-free poloidal SH space.
        area_weighted_least_squares : bool, optional
            Use surface-area weights for least-squares projections when
            no explicit ``sqrt_weights`` are supplied. Cubed-sphere
            model-grid fits use their native cell areas. Direct fits to
            other sample locations use the supplied grid's area weights
            or explicit ``sqrt_weights``. Coordinates alone do not
            define integration weights.
            Disabled by default to preserve the established norm.
        least_squares_solver, least_squares_tolerance,
        least_squares_preconditioner : optional
            Shared settings for all input fits, also saved for the
            consuming simulation. Defaults are SH ``normal_pinv``,
            CS ``lsmr``,
            tolerance ``1e-15``, and no preconditioner. See
            ``kompe.math.LeastSquaresSolver`` for tolerance semantics.
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
            boundary_jr_remapping=boundary_jr_remapping,
            boundary_Br_remapping=boundary_Br_remapping,
            conductance_basis=conductance_basis,
            u_remapping=u_remapping,
            Q_eff_remapping=Q_eff_remapping,
            E_neutral_wind_remapping=E_neutral_wind_remapping,
            horizontal_basis_kind=horizontal_basis_kind,
            area_weighted_least_squares=area_weighted_least_squares,
            least_squares_solver=least_squares_solver,
            least_squares_tolerance=least_squares_tolerance,
            least_squares_preconditioner=least_squares_preconditioner,
            t0=t0,
        )
        self._open_input_preparation(
            config,
            directory=input_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
        )
        self.load_input_series()
        self.save_settings_if_missing()

    def _open_input_preparation(
        self,
        config,
        *,
        directory,
        artifact_storage,
        operator_cache_directory=None,
        operator_cache=None,
        geometry=None,
        print_info=False,
    ):
        """Open input histories and geometry, without results."""
        config = None if config is None else SimulationConfig.from_settings(config)
        if config is None and geometry is not None:
            config = SimulationConfig.from_geometry(geometry)
        if config is None and directory is None:
            raise ValueError("A simulation directory is required when settings are omitted.")
        storage = ArtifactStore._normalize_storage_kind(artifact_storage)
        store = (
            None
            if directory is None
            else ArtifactStore(directory, preferred_dataset_storage=storage)
        )
        saved = None if store is None else store.load_dataset("settings", print_info=print_info)
        if (
            saved is not None
            and saved.attrs.get("simulation_schema_version") != SIMULATION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Saved simulation uses schema {saved.attrs.get('simulation_schema_version')!r}; "
                f"expected {SIMULATION_SCHEMA_VERSION}."
            )
        if config is None:
            if saved is None:
                raise ValueError(f"No saved 'settings' dataset exists at {store.directory!r}")
            config = SimulationConfig.from_settings(saved)
        elif saved is not None and not config.to_dataset().identical(
            SimulationConfig.from_settings(saved).to_dataset()
        ):
            raise ValueError("Mismatch between Simulation object arguments and settings on file.")
        if operator_cache_directory is not None:
            operator_cache = PersistentArrayCache(operator_cache_directory)
        if geometry is None:
            geometry = SimulationGeometry.from_config(config, operator_cache=operator_cache)
        else:
            if operator_cache is not None and operator_cache is not geometry.operator_cache:
                raise ValueError("Supplied geometry already owns its operator cache.")
            geometry = geometry.with_config(config)
        self.config = config
        self.geometry = geometry
        self.schema = build_simulation_schema(geometry, config)
        self.artifact_store = store
        self.artifact_storage = storage
        self.settings_saved = saved is not None
        self._input_series = self.schema.create_input_series(time_origin=config.t0)
        self._loaded_streams = set()
        self.main_field = geometry.main_field
        self.model_grid = geometry.model_grid
        self.operator_cache = geometry.operator_cache

    @cached_property
    def _input_projector(self):
        return _InputProjector(self)

    @property
    def input_series(self):
        """Return input streams, loading saved datasets once."""
        return self.load_input_series()

    @input_series.setter
    def input_series(self, series):
        self._input_series = series
        self._loaded_streams.update(INPUT_DATASET_KEYS)

    def load_input_series(self, *keys):
        """Read named streams once, or all inputs when omitted."""
        series = self._input_series
        unread = tuple(
            key for key in (keys or INPUT_DATASET_KEYS) if key not in self._loaded_streams
        )
        if not unread:
            return series
        for key in unread:
            if key not in series.variables:
                raise KeyError(f"Unknown input stream {key!r}.")
            if self.artifact_store is not None and key not in series.datasets:
                series.load(key, self.artifact_store)
        active_wind_forcings = sorted(WIND_FORCING_INPUTS.intersection(series.datasets))
        if len(active_wind_forcings) > 1:
            raise ValueError(
                f"Wind-forcing representations {active_wind_forcings} are mutually exclusive; use only one."
            )
        if "boundary_Br" in series.datasets and self.config.RM is None:
            raise ValueError("Stored boundary_Br input requires magnetospheric radius RM.")
        self._loaded_streams.update(unread)
        return series

    @property
    def settings(self):
        """Return the persisted physical and numerical configuration."""
        return self.config.to_dataset()

    def save_settings_if_missing(self, *, print_info=False):
        """Persist settings for a newly bound input directory."""
        if not self.settings_saved and self.artifact_store is not None:
            self.config.validate_geometry_for_storage(self.geometry)
            self.artifact_store.save_dataset(self.settings, "settings", print_info=print_info)
            self.settings_saved = True

    @property
    def datasets(self):
        """Return projected input datasets, keyed by stream name."""
        return self.input_series.datasets

    @property
    def input_directory(self):
        """Return the directory, or None for in-memory inputs."""
        return None if self.artifact_store is None else self.artifact_store.directory

    def __getitem__(self, key):
        """Return one input dataset."""
        return self.datasets[key]

    def __iter__(self):
        """Iterate over available input names."""
        return iter(self.datasets)

    def __len__(self):
        """Return the number of available input streams."""
        return len(self.datasets)

    def save(self, directory=None, *, artifact_storage=None):
        """Save prepared data and enable automatic later writes.

        This saves inputs only. Use ``simulation.save`` to save or
        relocate the complete trajectory, including recorded outputs.
        """
        self._save_input_package(
            directory, artifact_storage, existing_directory=self.input_directory
        )

    def _save_input_package(self, directory, artifact_storage, *, existing_directory):
        """Write inputs, replacing only the caller's own archive."""
        self.config.validate_geometry_for_storage(self.geometry)
        original_directory = self.input_directory
        storage = (
            self.artifact_storage
            if artifact_storage is None
            else ArtifactStore._normalize_storage_kind(artifact_storage)
        )
        series = self.input_series  # Read from the original store before relocation.
        store = (
            self.artifact_store
            if directory is None
            else ArtifactStore(directory, preferred_dataset_storage=storage)
        )
        if store is None:
            raise ValueError("Supply a directory when first saving in-memory data.")
        saved = store.load_dataset("settings")
        if saved is not None:
            if saved.attrs.get("simulation_schema_version") != SIMULATION_SCHEMA_VERSION:
                raise ValueError("The destination uses an incompatible simulation schema.")
            if not self.settings.identical(SimulationConfig.from_settings(saved).to_dataset()):
                raise ValueError("The destination contains different simulation settings.")
            if store.directory != existing_directory:
                raise ValueError(
                    "The destination already contains a simulation; choose a new directory or reopen it."
                )
        store.preferred_dataset_storage = storage
        store.save_dataset(self.settings, "settings")
        for key in series.datasets:
            series.save(key, store, incremental=False)
        if original_directory is not None and original_directory != store.directory:
            for key in series.datasets:
                series.load(key, store)
        self.artifact_store = store
        self.artifact_storage = storage
        self.settings_saved = True

    def __repr__(self):
        """Summarize projected inputs for interactive sessions."""
        inputs = ", ".join(sorted(self)) or "none"
        return (
            f"InputPreparation(Nmax={self.config.Nmax}, Mmax={self.config.Mmax}, "
            f"Ncs={self.config.Ncs}, inputs=[{inputs}], "
            f"input_directory={self.input_directory!r})"
        )

    @classmethod
    def from_geometry(cls, geometry, *, input_directory=None, artifact_storage="auto", **settings):
        """Reuse geometry with independent coefficient histories.

        Geometry supplies spatial settings. Other experiment controls
        use their normal defaults unless explicitly supplied here.
        Overrides may change physics, but not coefficient bases.
        Configure persistent operator caching on the supplied SH basis.
        """
        preparation = cls.__new__(cls)
        preparation._open_input_preparation(
            SimulationConfig.from_geometry(geometry, **settings),
            directory=input_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=None,
            geometry=geometry,
        )
        preparation.load_input_series()
        preparation.save_settings_if_missing()
        return preparation

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        *,
        input_directory=None,
        artifact_storage="auto",
        operator_cache_directory=None,
        geometry=None,
    ):
        """Construct an input preparation from normalized configuration.

        Storage and operator-cache directories are runtime preferences,
        not persisted model settings.
        """
        if not isinstance(config, SimulationConfig):
            raise TypeError("InputPreparation.from_config requires a SimulationConfig.")
        preparation = cls.__new__(cls)
        preparation._open_input_preparation(
            config,
            directory=input_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
            geometry=geometry,
        )
        preparation.load_input_series()
        preparation.save_settings_if_missing()
        return preparation

    @classmethod
    def from_directory(
        cls, input_directory, *, artifact_storage="auto", operator_cache_directory=None
    ):
        """Reopen saved inputs, retaining their physical configuration.

        Only storage and cache preferences can change.
        """
        preparation = cls.__new__(cls)
        preparation._open_input_preparation(
            None,
            directory=input_directory,
            artifact_storage=artifact_storage,
            operator_cache_directory=operator_cache_directory,
        )
        preparation.load_input_series()
        return preparation

    def write_manifest(self, *, source="manual", notes=(), metadata=None):
        """Write the manifest for this reusable input package."""
        if self.input_directory is None:
            raise ValueError("Save the prepared inputs before writing a manifest.")
        return write_input_manifest(
            self.input_directory,
            self.config,
            input_datasets=tuple(key for key in INPUT_DATASET_KEYS if key in self),
            source=source,
            notes=notes,
            metadata=metadata,
        )

    def set_coefficients(self, key, values, *, time=None):
        """Store already-projected input coefficients without fitting.

        ``key`` names an input stream. Supply an array for a single
        variable, or a dictionary for conductance with keys
        ``log_conductance_magnitude`` and
        ``log_hall_to_pedersen_ratio``.
        One field has the schema's coefficient shape; multiple fields
        have a leading time axis. Helmholtz coefficients use curl-free
        then divergence-free potentials, not theta/phi components.
        """
        projector = self._input_projector
        names = self.schema.input_variables[key]
        data = {names[0]: values} if len(names) == 1 else values
        if not isinstance(data, Mapping) or set(data) != set(names):
            raise ValueError(f"{key} requires coefficient variables {names}.")
        shape = self.schema.input_field_spaces[key].shape
        rows = {}
        for name, value in data.items():
            xp = get_array_module(value)
            array = xp.asarray(value)
            if array.shape == shape:
                array = array[None, ...]
            elif array.ndim != len(shape) + 1 or array.shape[1:] != shape:
                raise ValueError(
                    f"{key}.{name} requires shape {shape}, optionally preceded by a time axis."
                )
            rows[name] = array
        projector.store_input_coefficients(key, rows, time)

    def set_FAC(self, FAC, *, grid, time=None, sqrt_weights=None, reg_lambda=None):
        """Project field-parallel current (A/m²) onto radial current.

        ``grid`` contains model-frame points. Positive FAC follows
        the background magnetic field, not upward in both hemispheres.
        """
        unit_br = self.main_field.unit_vector(grid, self.config.RI)[0]
        self.set_boundary_jr(
            _scalar_sample_rows(FAC, grid) * unit_br,
            grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )

    def set_boundary_jr(self, boundary_jr, *, grid, time=None, sqrt_weights=None, reg_lambda=None):
        """Fit upper-boundary outward radial current samples in A/m².

        ``grid`` is a SphericalGrid in the model frame, including any
        quadrature weights. Values may retain its point shape; a time
        series adds a leading or trailing time axis. ``sqrt_weights``
        overrides fitting weights; ``reg_lambda`` controls smoothness.
        """
        self._input_projector.project_and_store_input(
            "boundary_jr",
            {"boundary_jr": boundary_jr},
            input_grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )

    def set_boundary_Br(self, boundary_Br, *, grid, time=None, sqrt_weights=None, reg_lambda=None):
        """Fit outer-boundary radial magnetic samples in tesla.

        Requires a finite ``RM``. ``grid`` and fitting controls follow
        ``set_boundary_jr``. Coefficients use poloidal SH space.
        """
        self._input_projector.project_and_store_input(
            "boundary_Br",
            {"boundary_Br": boundary_Br},
            input_grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )

    def set_resistance(self, *, etaP, etaH, grid, time=None, sqrt_weights=None, reg_lambda=None):
        """Fit positive Pedersen/Hall resistance samples in ohms.

        Both quantities share the model-frame ``grid``. Canonical log
        conductance coordinates preserve positivity after fitting.
        """
        log_magnitude, log_ratio = ionospheric_closure.resistance_to_log_conductance_coordinates(
            etaP, etaH
        )
        self._store_conductance_coordinates(
            log_magnitude,
            log_ratio,
            grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )

    def set_conductance(
        self, *, pedersen, hall, grid, time=None, sqrt_weights=None, reg_lambda=None
    ):
        """Fit positive Pedersen/Hall conductance samples in siemens.

        Fit ``log(hypot(SigmaP, SigmaH) / 1 S)`` and
        ``log(SigmaH / SigmaP)`` on the model-frame ``grid``. These
        coordinates guarantee positive reconstructed conductance and
        treat reciprocal resistance symmetrically.
        """
        log_magnitude, log_ratio = ionospheric_closure.conductance_to_log_coordinates(
            pedersen, hall
        )
        self._store_conductance_coordinates(
            log_magnitude,
            log_ratio,
            grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )

    def _store_conductance_coordinates(
        self, log_magnitude, log_ratio, *, grid, time, sqrt_weights, reg_lambda
    ):
        """Fit the two canonical conductance coordinates."""
        self._input_projector.project_and_store_input(
            "conductance",
            {"log_conductance_magnitude": log_magnitude, "log_hall_to_pedersen_ratio": log_ratio},
            input_grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )

    def set_neutral_wind(
        self, u_theta, u_phi, *, grid, time=None, sqrt_weights=None, reg_lambda=None
    ):
        """Fit south/east neutral wind samples in m/s on a model grid.

        Stored coefficients are curl-free/divergence-free potentials.
        Each fitting weight applies to both components unless supplied
        with shape ``(2, N)``. Use only one wind-forcing representation.
        """
        self._input_projector.set_tangential_input(
            "u",
            u_theta,
            u_phi,
            grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )

    def set_Q_eff(
        self, Q_eff_theta, Q_eff_phi, *, grid, time=None, sqrt_weights=None, reg_lambda=None
    ):
        """Fit south/east effective wind-current samples in A/m.

        Q_eff is added to sheet current before the resistance tensor
        maps current to electric field. It is an alternative to direct
        neutral wind or equivalent neutral-wind electric field.
        """
        self._input_projector.set_tangential_input(
            "Q_eff",
            Q_eff_theta,
            Q_eff_phi,
            grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )

    def set_E_neutral_wind(
        self,
        E_neutral_wind_theta,
        E_neutral_wind_phi,
        *,
        grid,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
    ):
        """Fit south/east equivalent wind electric-field samples in V/m.

        This field is added before imposed-current closure. Use it for
        externally prepared electrodynamics, such as separately
        Pedersen/Hall-weighted winds, instead of ``u`` or ``Q_eff``.
        """
        self._input_projector.set_tangential_input(
            "E_neutral_wind",
            E_neutral_wind_theta,
            E_neutral_wind_phi,
            grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )

    def set_Q_eff_from_neutral_wind(
        self,
        u_theta,
        u_phi,
        *,
        grid,
        time=None,
        sqrt_weights=None,
        wind_reg_lambda=None,
        Q_eff_reg_lambda=None,
    ):
        """Fit Q_eff coefficients to the wind-driven electric response.

        Project wind once, then solve Q_eff in its coefficient space
        using the conductance applicable at each supplied time.
        """
        projector = self._input_projector
        projector.require_no_exclusive_conflict("Q_eff")
        input_time, wind_rows = projector.project_tangential_samples(
            "u",
            u_theta,
            u_phi,
            grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=wind_reg_lambda,
        )
        rows = projector.fit_Q_eff_from_neutral_wind(
            input_time, wind_rows, reg_lambda=Q_eff_reg_lambda
        )
        projector.store_input_coefficients("Q_eff", {"Q_eff": rows}, input_time)

    def evaluate_Q_eff_from_neutral_wind(
        self, u_theta, u_phi, *, grid, time=None, sqrt_weights=None, reg_lambda=None
    ):
        """Evaluate model-grid Q_eff equivalent to supplied wind.

        Return south/east values and model latitude/longitude without
        storing another wind-forcing representation.
        """
        if "conductance" not in self.input_series.datasets:
            raise RuntimeError("Conductance must be set before evaluating Q_eff from wind.")
        input_time, wind_rows = self._input_projector.project_tangential_samples(
            "u",
            u_theta,
            u_phi,
            grid=grid,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )
        return self._input_projector.evaluate_Q_eff_from_neutral_wind(input_time, wind_rows)


__all__ = ["InputPreparation"]
