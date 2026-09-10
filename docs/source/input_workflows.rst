Input workflows
===============

Input preparation owns physical conversion and coefficient projection. Provider
adapters own external coordinates, units, file formats, and provenance. Neither
requires a running induction model. See :doc:`architecture` for object ownership.

Input preparation and projection
--------------------------------

Input projection is intentionally separated from ``Simulation`` in
``pynamit.simulation.input_projection``. The simulation schema declares each stream's
variables, field type, and coefficient space. ``_InputProjector`` owns sample
remapping and fitting, independently of the storage layout:

* gridded scalar and tangential projection;
* coefficient row and time-row validation;
* storage of projected input rows;
* mutual exclusivity between the alternative wind representations ``u``,
  ``Q_eff``, and ``E_neutral_wind``; and
* time-series coordination when deriving ``Q_eff`` from neutral wind.

Public setters such as ``set_boundary_jr``, ``set_resistance``,
``set_neutral_wind``, ``set_Q_eff``, and ``set_E_neutral_wind`` accept physical
samples and one ``SphericalGrid`` in the model coordinate frame. They make
physical conversions explicit and share the numerical projector. Constructing
a grid explicitly retains its shape and quadrature weights across calls::

    from kompe import SphericalGrid
    from pynamit import InputPreparation

    inputs = InputPreparation()
    grid = SphericalGrid(lat=latitude, lon=longitude)
    inputs.set_conductance(pedersen=SigmaP, hall=SigmaH, grid=grid)
    inputs.set_neutral_wind(u_theta, u_phi, grid=grid)

Use ``inputs.model_grid`` directly for values evaluated at model nodes.
Single samples default to time zero; multiple samples require explicit times.
An input preparation does not have a running clock. For an interactive change
during evolution, use ``simulation.inputs.set_conductance(...,
time=simulation.current_time)``. This makes the physical activation time visible
and gives independent preparations and live simulations the same setter semantics.

Already-projected values use the separate
``set_coefficients(key, values, time=...)`` path, without coordinates or fitting
options. A single-variable stream takes an array. The ``conductance`` stream
takes a dictionary with ``log_conductance_magnitude`` and
``log_hall_to_pedersen_ratio`` arrays. Tangential coefficients have shape
``(2, N)`` (or ``(T, 2, N)`` for multiple times), ordered as scalar Helmholtz
potentials ``(curl-free, divergence-free)``, not physical theta/phi components.

``SimulationConfig`` distinguishes coefficient representation from sample
remapping. ``conductance_basis='SH'`` or ``'CS'`` chooses the stored conductance
basis and defaults to ``horizontal_basis_kind``. Other streams keep their
schema-defined coefficient spaces; ``boundary_jr_remapping``,
``boundary_Br_remapping``, ``u_remapping``, ``Q_eff_remapping``, and
``E_neutral_wind_remapping`` select ``'direct'`` fitting at supplied points or
``'CS'`` remapping to model nodes before fitting. Defaults follow the horizontal
model choice, except ``Q_eff_remapping`` inherits ``u_remapping``.
The version-4 file format retains its historical ``*_projection_basis``
settings keys (``'SH'`` means direct fitting for non-conductance streams);
translation happens only at the settings serialization boundary. Existing
saved inputs and manifests remain readable without regeneration.

PynaMIT's input/storage boundary keeps time first. It normalizes flat or
mesh-shaped provider samples, then explicitly moves time to the last axis for
Kompe analysis and back to the first axis for storage. Kompe's transforms and
linear maps always keep scientific axes before batch axes. The coefficient
representation remains in the schema's ``input_field_spaces``.
The projector constructs the scalar or tangential remapping operator explicitly
and passes it to Kompe before fitting. It does not inspect SH/CS storage layouts.

Area weighting uses known integration weights, not a latitude-based guess.
Passing the model grid retains its cubed-sphere cell areas. A direct
area-weighted fit uses the supplied grid's ``area_weights`` or explicit
``sqrt_weights``; coordinates alone do not specify quadrature. For a grid uniform in
theta and phi these can be proportional to ``sqrt(sin(theta))``; for equal-area
samples they are constant. Statistical observation weights are a separate
choice. Remapping first uses the target model grid's integration measure.
MAGE projection and diagnostics share a reader that supplies the established
``sin(theta)`` measure for the regular TIEGCM latitude/longitude grid, checking
its uniform sampling at the file boundary. The magnetospheric grid retains
its stored solid angles. No MAGE forcing-file regeneration is needed.

``set_Q_eff_from_neutral_wind`` follows one coefficient-space route: it fits
the stored ``Q_eff`` so its resistance-weighted electric response matches the
projected wind forcing. ``evaluate_Q_eff_from_neutral_wind`` is the separate
non-persisting diagnostic for callers that need the equivalent model-grid
field.

Prepared external inputs
------------------------

``pynamit.workflows.example``, ``pynamit.workflows.example_inputs``,
``pynamit.workflows.prepared_inputs``, and
``pynamit.workflows.mage`` contain reusable, script-independent orchestration
for standard simulations, prepared forcing, and MAGE/GAMERA projection. The prepared
input manifest and compatibility rules live below them in
``pynamit.simulation.input_manifest`` so the core simulation never imports a
convenience workflow. Scripts under
``scripts/simulation`` are workflow entry points: they own editable experiment
settings, paths, and directory naming while delegating reusable validation and
numerical work to package modules.

Empirical event conditions are owned by the caller. The preparation workflow
requires the event time, Kp, solar-wind and IMF values, dipole tilt, F10.7, and
Ap explicitly; PynaMIT does not define a physically privileged default event.
The regression tests keep their shared 12 May 2001 scenario under ``tests/``.

External empirical inputs use immutable value objects with separate geographic,
model-frame, library-interface, and output semantics. ``InputProviderSpec`` describes one
library adapter and independently declares its request contract, output
contract, fields, and vector basis. Hardy, AMPS, and HWM remain independently
configurable even though their current request contracts are equal.

PynaMIT's geographic grid is a geocentric spherical GEO grid at the ionospheric
radius. The external libraries describe their interfaces as geographic or
geodetic. PynaMIT deliberately applies a simple spherical-Earth approximation
at this boundary: numerical latitude and longitude are passed through
unchanged, and the same nominal 110-km altitude is supplied to the library.
The approximation is centralized in ``pynamit.geographic_approximation`` rather than being
repeated implicitly in each adapter.

``ExternalInputCoordinates`` owns the physical geocentric-GEO grid and the
model-frame view of the same ordered samples, then caches library-facing views
by coordinate-convention signature. In a GEO simulation the geographic and model grids
are the same object. A generic dipole simulation retains an additional
``centered_dipole`` model grid. Hardy, AMPS, and HWM reference one interned
``LIBRARY_GEOGRAPHIC_110KM`` contract for their shared physical sampling grid.

The PynaMIT adapter evaluates Lompe's Hardy and EUV primitives explicitly.
For a generic dipole simulation, Hardy receives the retained centered-dipole
model coordinates and magnetic local time uses the simulation's decimal-year
dipole epoch. For a GEO model frame, Hardy receives modified-Apex coordinates
derived at the full event time and MLT uses the event's decimal year. EUV
always receives the paired physical GEO positions. This avoids Lompe's
``hardy_EUV`` convenience path, which constructs its MLT dipole from the
integer event year. AMPS always starts from the physical GEO positions and
derives its QD/Apex coordinates independently. HWM is
evaluated at the same requested positions through
``pyhwm2014.hwm14_vectorized`` with the event's YYDDD date code and full UTC
time. Naive datetimes retain PynaMIT's historical UTC interpretation; aware
datetimes are normalized to UTC before any provider is called. Under the
spherical approximation, library east/north wind components map directly to
PynaMIT ``u_phi`` and ``-u_theta``. HWM therefore introduces no separate
regular grid, seam handling, or second spatial fit.

All adapters return values associated with the original geographic-grid
ordering.
Prepared-input construction stores those values at the corresponding
simulation model-grid nodes. Existing PynaMIT and Kompe caches consequently
reuse compatible projection operators according to mathematical signatures
rather than provider names.

Fallback files contain a shared registry of geographic and library-request grids.
Each ``ProviderSnapshot`` references both grid objects and its independent
provider specification. Coordinate identities hash the full coordinate
contract together with normalized ordered coordinate pairs. Equal contracts
and grids are structurally shared after loading, while different contracts
remain semantically distinct even when their numerical arrays happen to be
equal.

Empirical adapters return one snapshot for one physical event. They do not
accept simulation-relative times or manufacture a time history. The bundled
fallback file records its generating event and provider conditions alongside
the cached arrays, while the regression scenario that requests those values
lives under ``tests/``. Users provide genuine time-dependent samples or
coefficients directly through ``InputPreparation``; the test suite uses an
explicit coefficient history to exercise storage, interpolation, and
evolution.

MAGE remains on its native spherical coordinate path. Preparation aligns
GAMERA and ReMIX through Kaiju/Geopack, projection requires
``main_field_kind='kaiju_dipole'``, and reusable simulations reject projected packages
whose saved main-field kind differs. No ellipsoidal conversion is introduced
into the MAGE workflow or PynaMIT's spherical differential operators.

The MAGE workflow has three stages with different reuse boundaries.
``mage_prepare.py`` owns optional Kaiju, NetCDF, and HDF5 access, signed GAMERA
dipole-axis interpretation, time-dependent external-coordinate conversion, and
the stepwise external-data ETL loop. TIEGCM scalars and conductivity-weighted
winds remain on the native geographic grid; only the conventional
east/north-to-spherical component conversion is required. REMIX FAC and GAMERA
fields originate on timestamped SM grids, so preparation maps each history
onto fixed GEO target grids. The boundary target is the first GAMERA shell
mapped into GEO and then held fixed. GAMERA boundary positions use the same
volume-barycentric trilinear-cell quadrature as Kaiju, matching the
cell-centered magnetic data before radial projection and remapping. Boundary
fitting uses the corresponding spherical cell solid angles computed from the
true vertices, rather than treating the nonuniform GAMERA mesh as regular
latitude--longitude sampling. The output is a minimal, versioned forcing
contract independent of PynaMIT spectral resolution.
The saved main-field strength converts ``MagM0`` between GAMERA's length-scale
reference radius and PynaMIT's dipole reference radius, preserving the same
physical field. ``delta_Br`` subtracts GAMERA's split numerical ``B0`` rather
than its separately evaluated analytic dipole, so finite-volume background
representation error is not reinterpreted as an external perturbation.
Preparation validates GAMERA/TIEGCM time correspondence and atomically
publishes the HDF5 file only after every source step succeeds. The contract
converts ReMIX's parallel-positive FAC explicitly to upward-positive FAC using
Kaiju's northern and southern grid orientations, then converts it to PynaMIT's
outward radial current with ``jr = FAC_upward * abs(unit_br)``. It records both
the SM source and GEO model frames rather than exposing either as a projection
knob.

Visualization preserves the same frame boundary. Global maps are evaluated
and drawn in GEO. A timestamp-dependent Plate Carrée projection moves the map
seam so geographic noon/12 solar LT is central; fields and coastlines remain
in ordinary GEO coordinates and are transformed by the map projection.
Hemisphere maps evaluate the same model samples, transform their positions to
MAG (or apex coordinates for an IGRF main field), and use magnetic local time
for orientation and labeling. Plotting therefore does not silently reinterpret
GEO model longitudes as magnetic longitudes.
``pynamit.workflows.mage.projection`` owns forcing-schema validation, fixed
projection geometry, least-squares weighting, and construction of a reusable
coefficient-space input package. Internally, one private projector owns the
grids and numerical operators that are invariant across forcing times; the
public ``prepare_inputs`` function owns the HDF5 and manifest workflow. It
builds in a temporary sibling directory and replaces the published artifacts
only after every projected time and the package manifest succeed.
Finally, ``mage_run.py`` creates any number of named simulations from one projected
package. These boundaries prevent external-reader concerns, numerical
projection, and simulation experiments from sharing mutable state.

The simulation manifest snapshots the prepared-input manifest and selected streams.
An existing trajectory can be resumed or extended only when that identity and
its evolution policy still match; a different projection or experiment uses a
new simulation directory. This prevents newly copied inputs from being paired with
inductive outputs computed from an older forcing package.

This boundary matters because prepared-input logic is both user-facing and
testable.  If a script needs behavior that should remain correct over time,
move it into the package and test the behavior there.

A consuming simulation owns its selected inputs. The workflow removes stale,
unselected input artifacts, opens only the selected prepared streams, copies
those datasets into the simulation directory, and reloads them through the simulation's own
``ArtifactStore`` object. Lazy Zarr
arrays therefore do not leave a live simulation dependent on the preparation
directory. PFAC integration samples are re-derived from the consuming simulation's
radial domain unless that simulation supplies them explicitly; they are not part of
the prepared coefficient contract. The MAGE simulation uses
``dipole_fac_integration_radii`` with an editable point count, so this PFAC
discretization remains a simulation choice rather than a projection side effect.

For generic prepared inputs, coordinates name physical positions, not just
array axes.
Geographic wind positions and tangent-vector components are rotated into the
configured model coordinates before projection. Native providers share one
physical spherical-GEO sample grid, while Hardy may also consume the retained
centered-dipole model view. Providers may derive Apex or magnetic coordinates
internally, but their returned values remain attached to the original ordered
physical positions before projection in model coordinates. The forcing event
time is persisted as the input time origin ``t0``. By default,
``main_field_epoch`` resolves to that event's decimal year and defines both the
background-field coefficients and centered-dipole axis. An explicit epoch may
still select a deliberately fixed reference field; the resolved value is always
persisted.

The versioned prepared-input manifest has one canonical ``input_contract``.
Coefficient-space settings, geometry requirements, and the dataset list live
only inside that contract; top-level provenance and notes do not mirror contract
fields. Consumers validate the manifest version and contract before loading any
forcing into a simulation.
