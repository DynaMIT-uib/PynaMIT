Architecture
============

PynaMIT is organized around a small number of durable responsibilities:
configuration and schema construction, spatial bases, input preparation,
time evolution, persistence, and visualization.  The simulation APIs should
stay readable when used interactively, while the implementation keeps the
physics-specific transformations in focused modules.

API boundaries
--------------

PynaMIT has four API tiers:

* The stable primary API is the explicit export set of ``pynamit`` and
  ``pynamit.simulation``. Saved results, plotting, the GUI, and reusable
  workflows have the explicit ``pynamit.results``, ``pynamit.plotting``,
  ``pynamit.gui``, and ``pynamit.workflows`` namespaces. Together they contain
  the simulation facade, normalized configuration, field values and spaces,
  background-field utilities, backend selection, saved simulation access, and
  renderers. Spherical and numerical APIs are imported from ``kompe``.
* Reusable packages such as ``pynamit.geomagnetism`` and
  ``pynamit.storage`` provide advanced scientific and infrastructure APIs.
  Their package export lists define the supported
  entry points; implementation submodules do not extend that promise merely
  because Python can import them.
  Focused electrodynamics and workflow functions are advanced module APIs at
  their documented module paths rather than primary package exports.
* ``Simulation.config``, ``Simulation.current_time``,
  ``Simulation.data``, ``Simulation.geometry``, and
  ``Simulation.response`` are stable attributes for inspection and
  diagnostics. This guarantees the ownership path, not every constructor or
  non-underscored member of the collaborator's concrete class. Named geometry
  maps and response operators are advanced diagnostic surfaces and may be
  expensive to materialize.
* Underscored attributes, including ``simulation._input_projector`` and
  ``simulation._time_evolution``, caches, and scheduling helpers are internal. Tests
  may exercise these objects directly without turning them into user API.

Spherical grids, bases, and transforms are imported directly from Kompe.

High-level flow
---------------

``Simulation`` remains the public orchestration object.  It normalizes user
configuration with ``SimulationConfig``, builds a ``SimulationSchema``,
creates persistence through ``SimulationData``, exposes lazily constructed
``SimulationGeometry`` and ``ElectrodynamicResponse`` collaborators, and keeps
the user-facing simulation methods in one place.

The main setup path is:

1. ``SimulationConfig`` normalizes constructor settings, defaults, numerical
   policies, basis choices, physical domains, time values, and persisted
   xarray attributes.
2. ``build_simulation_schema`` creates the spherical-harmonic, solid-harmonic,
   and cubed-sphere basis objects, then declares the input and output
   ``FieldSpace`` metadata used by storage and transforms.
3. ``Simulation`` attaches an ``_InputProjector``, which lazily builds reusable
   input transforms on the geometry's canonical model grid and owns all input
   validation, projection, and coefficient storage.
4. ``Simulation`` attaches a ``_TimeEvolution`` for restart handling, sample
   scheduling, progress reporting, and output save decisions.
5. ``SimulationGeometry`` is constructed once, on first use, as the numerical
   spatial context. ``ElectrodynamicResponse`` is likewise constructed when a
   response calculation first needs it; it receives the geometry and owns the
   instantaneous forcing coefficients, constraint solve, and closure-dependent
   operator caches.
6. ``SimulationData`` owns persisted settings, input time series, and output
   time series.

Persisted objects are reached through ``simulation.data`` (for example
``simulation.data.output_series`` and ``simulation.data.schema``),
while the horizontal basis, solid harmonics, and background field are reached
through ``simulation.geometry``. ``Simulation`` does not copy those
references onto parallel top-level attributes. This keeps its initialization
small and makes the owner of each object unambiguous.

Keep this layering intact: configuration should not perform numerical work,
schema should not read simulation data, input projection should not evolve the model,
and visualization should not mutate simulation state.

Package map and dependency direction
------------------------------------

The top-level packages separate reusable scientific and infrastructural
concepts from the PynaMIT simulation model::

    pynamit/
      fields.py              coefficient-space metadata and owned values
      coordinates.py         generic longitude and local-time conversions
      geomagnetism/          background fields and magnetic coordinates
        main_field.py        background-field models and magnetic mapping
      external_inputs/       empirical input adapters and coordinate semantics
        coordinates.py       geographic, model, and library coordinate views
        provider_definitions.py
                             provider fields and interface conventions
        fallback_data.py     provider snapshots and bundled data
        providers.py         native and fallback evaluation
      storage/               named artifacts and coefficient time series
      simulation/            the coupled PynaMIT model
      results/               read-only saved simulations and field evaluation
      plotting/              Matplotlib figure construction
      gui/                   optional interactive Panel application
      workflows/             reusable preparation and simulation orchestration

    kompe/                   spherical geometry, bases, operators, and solvers

The simulation package is grouped by runtime role::

    simulation/
      __init__.py            stable preparation, simulation, and config exports
      input_preparation.py   public input projection and persistence workflow
      simulation.py          coupled response and evolution workflow
      response.py            active inputs, response solves, and operator caches
      geometry.py            simulation-specific spatial and magnetic mappings
      evolution.py           time evolution, sampling, and persistence
      input_projection.py    input validation, projection, and storage
      config.py/schema.py    normalized configuration and field spaces
      simulation_data.py     persisted simulation context
      input_manifest.py      prepared-input file contract
      electrodynamics/       magnetic boundary, closure, and induction equations

Dependencies should point inward from workflows and facades toward focused
implementation modules.  In particular:

* ``workflows`` may construct and operate ``Simulation``;
* ``Simulation`` may coordinate projection, response, evolution, and persistence;
* ``ElectrodynamicResponse`` and ``SimulationGeometry`` may compose ``electrodynamics`` functions; and
* ``electrodynamics`` must remain free of imports from simulation
  orchestration and persistence modules.

The reusable packages must not import the simulation layer. In particular,
``geomagnetism`` owns ``MainField`` and magnetic-coordinate conversion without
knowing about ``SimulationConfig``;
``geometry.build_main_field`` is the small adapter from simulation configuration to
that reusable API. ``storage`` knows how to persist named datasets and field
series but does not know the simulation artifact vocabulary.

This direction keeps the equation modules reusable without introducing
stateful physics-manager objects.

Configuration and schema
------------------------

``pynamit.simulation.config`` is the canonical home for settings and default
normalization.  New simulation settings should be added to
``SimulationConfig`` first, then serialized through ``to_attrs``/
``to_dataset``.  Avoid adding loose settings directly to ``Simulation`` once a
setting needs persistence or restart compatibility.

Configuration also owns physical-domain invariants. The FAC integration grid
is immutable simulation configuration and, when omitted, is derived from that simulation's
``RI`` and optional ``RM``. Basis resolutions, radial boundaries, main-field
parameters, boolean choices, solver names, and integrators are validated
before schema or geometry construction so invalid simulations cannot partially
initialize. Restart comparison is performed between normalized configurations
within the current persisted schema, so equivalent settings compare
canonically without allowing an actual mismatch.

Once normalized, one immutable ``SimulationConfig`` instance is shared by
``Simulation``, ``SimulationData``, and ``ElectrodynamicResponse``. Schema and persistence
builders do not accept parallel setting overrides; callers that need to adapt
legacy settings do so through ``SimulationConfig.from_settings`` before
crossing those boundaries. Runtime policy is read from that shared config
rather than copied into mutable state attributes.

Programmatic callers can pass a normalized configuration through
``Simulation.from_config``. Storage format, simulation directory, and array backend
remain separate arguments because they are execution preferences rather than
persisted physical model settings.

``pynamit.simulation.schema`` is the canonical home for basis and field-space
selection.  New persisted input or output streams should be declared through
the schema tables and built into ``FieldSpace`` objects there.  This keeps
storage names, field types, mean-free choices, and projection bases visible in
one place. A built ``SimulationSchema`` owns immutable mapping copies so
storage metadata cannot drift after the corresponding time series exist.

Numerical policies that can change an established calculation are explicit
and persisted. ``area_weighted_least_squares=False`` retains the historical
unweighted projection convention; enabling it changes the least-squares norm,
not the underlying field equations. Likewise,
``toroidal_potential_regularization_lambda=0.0`` leaves the physical
toroidal-potential problem unregularized. Reproducibility workflows pin both
choices explicitly instead of inheriting a future policy change.

Numerical operators and value identity
----------------------------------------

``LinearMap`` is the common operator abstraction for dense, sparse,
matrix-free, and structured-einsum calculations. Least-squares problems and
physical compositions should retain ``LinearMap`` objects until an explicit
representation is genuinely required. ``to_matrix()`` returns a flat 2-D
matrix, while ``to_array()`` retains the shaped domain and codomain axes. The
tensor helper module contains only
contractions and pseudoinverses that still operate on multidimensional arrays;
do not add parallel wrappers for operations already expressed by
``LinearMap``.
Creating an explicit map is also an execution decision. The response layer does
this only for runtime maps where dense multiplication is advantageous; plotting
and export code should otherwise retain structured maps until explicit values
are actually needed.
When a rectangular map must become explicit, materialization probes the
smaller side: input columns for tall maps and adjoint output rows for wide
maps. This is important for rectangular surface-to-poloidal and
boundary-current-to-gap-field maps, whose compact physical output should
determine temporary memory use.

Sparse equality-constrained least-squares factorization is a solver-layer
primitive rather than a cubed-sphere implementation detail. Native CS
Helmholtz analysis supplies its sparse synthesis matrix and its two constant
gauge rows to that primitive, then retains the resulting analysis as a
``LinearMap`` for both NumPy and JAX arrays. This avoids a dense
``(2 Ncs_surface)^2`` pseudoinverse while preserving the same weighted field
fit and fixing both potential gauges exactly. Non-native grids, regularized
fits, and explicitly selected alternate solvers continue through the general
transform solver.
Mean-free SH Helmholtz transforms use the corresponding dense full-rank
factorization. Their coefficient spaces omit both constant gauges, so a
cached normal-system factor replaces the tall SVD while retaining a
structured adjoint. Spaces retaining a mean mode, undersampled grids, and
otherwise rank-deficient transforms keep the pseudoinverse fallback.

``SphericalGrid`` is an immutable coordinate value. Its coordinate signature defines
grid-value compatibility and remapping identity. Its separate analysis
signature also includes optional area weights: equal coordinates can share
synthesis matrices while requiring different weighted least-squares analyses.
Transform caches must choose deliberately between those two identities.
Mean-freedom is owned by coefficient spaces, not by post-step cleanup. The
schema's ``sh_basis`` retains the mean/monopole term for quantities such as
conductance, while ``mean_free_sh_basis`` is used for radial magnetic fields.
The evolving ``induced_Br`` at ``RI`` and prescribed ``boundary_Br`` at ``RM``
therefore live directly in the mean-free poloidal SH space.
``boundary_jr``, ``Phi``, and ``W`` live in the selected horizontal surface
space. In CS mode the surface potentials use an explicit zero-area-mean gauge
constraint. Saved ``boundary_jr`` is kept in the exact range of the discrete
surface Laplacian instead of applying a second, slightly different
area-mean projection. Each stored representation supplies its own synthesis
operator on the model grid, so ``SimulationGeometry`` does not carry a
parallel full-scalar transform.

``FieldCoefficients`` is likewise an owned value. NumPy inputs are copied and
made read-only, and NumPy-to-JAX construction first breaks any shared host
buffer. This is required for correctness, not merely style: ``ElectrodynamicResponse`` caches
operators derived from coefficient values, so mutation behind the container
would make those caches scientifically stale.

Basis and grid metadata follow the same rule. Coefficient-index names are
tuples, and coefficient indices, CS coordinates, metric tensors, and cell
areas are owned read-only arrays. A basis cache key and its persisted
coefficient identity therefore cannot diverge through mutation by a caller.

Spatial bases
-------------

Spherical harmonics and cubed-sphere support have different implementation
needs, but they should present the same public basis contract wherever
possible. ``GlobalCSBasis`` is Kompe's public cubed-sphere basis. Its private
implementation is split into:

* ``cs_coordinates`` for panel and coordinate transforms;
* ``global_mesh`` for mesh shape and indexing;
* ``global_remapping`` for scattered scalar and vector remapping;
* ``global_differencing`` and ``finite_differences`` for derivative
  stencils and sparse operators; and
* ``cs_vectors`` for vector-basis conversions.

Prefer adding focused CS behavior to one of these collaborators instead of
growing ``GlobalCSBasis`` again. ``GlobalCSProjection`` owns the public
continuous coordinate and vector transformations; ``GlobalCSMesh`` owns
sampled geometry; and ``GlobalCSBasis`` owns expansion, interpolation, and
closed-surface differential behavior. Keep the ``CS`` abbreviation in class
names that represent actual objects.

``kompe`` is a warranted standalone package because its bases, grids,
analysis/synthesis transforms, and solid-harmonic continuation form a coherent
numerical domain that can be used without a PynaMIT simulation.

``geomagnetism`` is the corresponding standalone physical domain. ``MainField``
owns background-field models, field-line and conjugate mapping, magnetic
coordinates, and apex basis vectors. ``MainField.evaluate(grid, radius)``
returns ``(Br, Btheta, Bphi)`` directly on the active array backend. Its
bounded cache reuses values for equal coordinate signatures, radii, and
backends. ``unit_vector``, ``horizontal_to_apex_array``, and
``radial_to_apex_scale`` expose the related local geometry without another
bound evaluation object. The ``(2, 2, N)`` array contains one pointwise
component transform per sample; sampled fields use component or value names
instead.

The simulation frame is explicit and follows the main-field kind.
``kaiju_dipole``, ``igrf``, and ``radial`` use
``geocentric_geographic`` coordinates, so the SH and cubed-sphere positions
denote locations on Earth directly. The generic idealized ``dipole`` model
uses ``centered_dipole`` coordinates. For ``kaiju_dipole``,
``MainField`` alone owns the fixed GEO-to-MAG rotation: it evaluates the
analytic dipole, field-line mapping, conjugacy, magnetic latitude, and apex
basis vectors in MAG and returns coordinates and vector components in GEO.
The Kaiju/Geopack dipole coefficients and axis are frozen at
``main_field_epoch``. ``SM`` remains a timestamped external-source coordinate
system whose Sun-facing longitude origin is transformed at every source time;
it is never the coordinate space of the model coefficients. The background
field is not re-tilted every timestep.

Do not promote every focused module to a package. A standalone package is
warranted when the concept has a reusable public vocabulary and a dependency
boundary of its own. The PynaMIT boundary, ionospheric closure, and induction
equations are mutually coupled parts of one simulation model, so they remain
together under ``simulation.electrodynamics``.

Input preparation and projection
--------------------------------

Input projection is intentionally separated from ``Simulation`` in
``pynamit.simulation.input_projection``. The simulation schema declares each stream's
variables, field type, storage space, and projection basis; ``_InputProjector``
applies the shared validation and projection rules. Every persisted input also
has an explicit projection-basis setting in ``SimulationConfig``. ``Q_eff``
defaults to the ``u`` route because it is an alternative representation of the
same wind forcing;
``E_neutral_wind`` defaults to the horizontal model basis because it stores an
equivalent electric field directly. Field type remains canonical in the
schema's ``FieldSpace``. ``_InputProjector`` owns:

* sample-vs-coefficient validation;
* gridded scalar and tangential projection;
* coefficient row and time-row validation;
* storage of projected input rows; and
* mutual exclusivity between the alternative wind representations ``u``,
  ``Q_eff``, and ``E_neutral_wind``; and
* time-series coordination when deriving ``Q_eff`` from neutral wind.

Public setters such as ``set_boundary_jr``, ``set_resistance``,
``set_neutral_wind``, ``set_Q_eff``, and ``set_E_neutral_wind`` should remain
thin API methods.  When a new input stream is added, prefer extending the
schema and the shared projector over hand-writing a new projection path inside
``Simulation``.
Tangential setters accept either sampled ``theta``/``phi`` components or one
canonical coefficient array. The coefficient array has shape ``(2, N)`` (or
``(T, 2, N)`` for multiple times), with scalar Helmholtz-potential coefficients
ordered ``(curl-free, divergence-free)``. Keeping both potential blocks in one
argument makes the stored representation explicit and prevents incomplete
decompositions.
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
The approximation is centralized in ``pynamit.geodesy`` rather than being
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

Electrodynamic physics
----------------------

The ``electrodynamics`` modules follow the direction of the model equations
instead of mirroring individual coefficient names:

* ``electrodynamics.magnetic_boundary`` maps the physical
  ``induced_Br``, ``boundary_Br``, and ``boundary_jr`` quantities to the
  derived horizontal sheet current ``JS``. Solid harmonics own generic radial
  continuation; this module owns the particular potential jump and shielding
  relations used by PynaMIT. The private poloidal and toroidal potentials
  appear here only as convenient operator coordinates.
* ``electrodynamics.ionospheric_closure`` applies the height-integrated
  Ohm-law closure. It maps physical Hall/Pedersen conductance or resistance
  into the canonical log-conductance coordinates, reconstructs the resistance
  tensor, and maps neutral motion or sheet current through the magnetic
  geometry to ``E``. It owns both the direct grid law
  ``E = R JS - u x B`` and the coefficient-operator compositions used by
  ``ElectrodynamicResponse``. It also owns the Pedersen/Hall geometry tensors
  and collisional Joule-heating kernel. Joule heating is the Pedersen
  dissipation ``etaP * J.T @ P @ J``; ``J dot E`` is electromagnetic work and
  is not generally the same quantity when neutral motion contributes to the
  closure. Its functions remain numerical kernels; iteration over input times
  and coefficient storage belong to ``_InputProjector``.
* ``electrodynamics.induction`` owns Faraday evolution of physical
  ``induced_Br``, including Euler, exponential, and SciPy integration and the
  corresponding instantaneous equilibrium. It integrates in the private
  induced-poloidal-potential coordinate where that improves conditioning, then
  converts exactly at the module boundary. Motion of an SM forcing pattern
  appears through timestamped SM-to-GEO preparation, not through a rotating
  model basis.

``SimulationGeometry`` supplies simulation-specific grids, magnetic-field factors,
transforms, radial field-line mapping, exact physical-to-private coordinate
maps, and interhemispheric constraint geometry to these equations. It is a
numerical object: persisted xarray values are unwrapped at construction, and
``SimulationData`` alone wraps the numerical ``gap_Br_response`` for storage.
``ElectrodynamicResponse`` receives that geometry, owns the inputs active at
one simulation time, solves the toroidal-potential constraint behind
``boundary_jr``, and caches the composed operators whose values depend on the
current resistance distribution. It also exposes named physical operator and
matrix compositions for inspection. Keeping those compositions with their
caches avoids split ownership between an operator facade and the response
object whose private state they mutate.

The combined field that drives the toroidal-potential response is named
``driving_E``. It can contain one neutral-wind representation, a
``boundary_Br`` response, or an ``induced_Br`` response. The name therefore describes
its role without suggesting that it contains only one neutral-wind
representation. The private toroidal potential then completes the response
required by the ``boundary_jr`` and optional interhemispheric constraints.
``E_neutral_wind`` is the public name for an externally prepared equivalent
neutral-wind electric field. It does not claim that the stored field is the
total electric field. In the MAGE workflow it is derived from separate
Pedersen- and Hall-weighted winds; the model composes the other terms
while solving the closure.

The poloidal and horizontal surface spaces coincide in the default SH mode.
They are intentionally distinct in CS mode. ``surface_to_poloidal_operator``
is the only bridge in Faraday's law: it projects the surface ``W`` potential
onto the configured poloidal harmonics.
``boundary_jr_to_gap_Br_operator`` has the corresponding rectangular shape.
Its input is radial current at ``RI+`` and its output is the unshielded
poloidal ``Br`` produced in the gap at ``RI``. That gap field and the inward
continuation of ``boundary_Br`` are both external-source radial fields, so the
same divergence-free ionospheric sheet-current response shields them. This
factorization prevents unobservable high-resolution CS modes from being
carried as part of ``induced_Br``.

Expensive optional geometry follows use rather than construction. The
``gap_Br_response`` is reused when persisted, but a new one is not built merely
to construct ``Simulation`` or project inputs. It is constructed and saved
when an equilibrium or dynamic path first requests model output. When PFAC
coupling is disabled (or the main field is radial), the optional contribution
is represented by absence rather than by constructing and multiplying a dense
zero matrix.

Surface-sized operators should remain structured in CS mode. In particular,
native Helmholtz analysis, wind and sheet-current closure compositions, and
ordinary runtime toroidal-potential responses must not materialize dense
surface maps.
The surface-to-poloidal bridge composes a factorized full-column-rank SH
analysis with the horizontal synthesis operator; native CS nodal synthesis
therefore stays an implicit identity rather than allocating a dense grid-sized
identity. The analogous poloidal-current fit used while constructing the gap
response also keeps a factorized normal system and an implicit adjoint instead
of forming an SVD pseudoinverse of its tall grid matrix.
The compact poloidal feedback matrix remains intentionally dense because its
matrix exponential and equilibrium pseudoinverse require an explicit
poloidal operator. The equilibrium response composes that compact
pseudoinverse with the structured surface-to-poloidal bridge; its full
cross-space matrix is materialized only when explicitly requested for
diagnostics. The generic dense ``normal_pinv`` solver remains the SH default
and is retained for reproducibility. CS toroidal-potential solves default to
matrix-free ``lsmr``; ``cgls`` and the ``jacobi`` preconditioner remain
explicit alternatives when their cost is justified for a particular system.

The toroidal-potential runtime follows the same rule. A single active
``boundary_jr`` or driving-E field is assembled as one physical least-squares
right-hand side and solved directly. The full
``boundary_jr_to_toroidal_potential_operator`` and
``driving_E_to_toroidal_potential_operator`` remain available for matrix
diagnostics, but normal simulation steps do not construct them.
Interhemispheric induction solves only the source columns reachable from
``induced_Br``, rather than first constructing the response to every possible
horizontal-E coefficient.

Magnetic-boundary maps also retain structured ``LinearMap`` compositions.
This matters in CS mode, where the native gradient and Helmholtz synthesis
operators are sparse. Repeated E-response maps are cached densely for compact
poloidal inputs and when the horizontal and poloidal SH spaces coincide; large
CS surface-to-surface maps remain structured until an explicit matrix is
requested.

The persisted input artifact is named ``conductance``. Its canonical scalar
fields are
``log_conductance_magnitude = log(hypot(SigmaP, SigmaH) / 1 S)`` and
``log_hall_to_pedersen_ratio = log(SigmaH / SigmaP)``. The fixed one-siemens
reference makes the first logarithm dimensionless without changing numeric
values already expressed in siemens. Both components are required to be
strictly positive. Fitting these two unconstrained coordinates guarantees
positive reconstructed Pedersen and Hall conductance. It also treats the
reciprocal resistance naturally: conductance and resistance have opposite
log magnitudes and the same Hall/Pedersen ratio.

``set_conductance`` is the canonical physical input API and can also accept
already projected log-coordinate coefficients. ``set_resistance`` remains a
sample-level convenience for physical resistance values. It maps the
reciprocal resistance magnitude and unchanged Hall/Pedersen ratio directly
onto the same canonical log coordinates, without constructing intermediate
conductance components. The response synthesizes the two log fields once per
active input, reconstructs the resistance tensor on the model grid, and
caches the current closure-dependent operators. An exact fingerprint of the
canonical coefficients determines whether those caches remain valid after an
input update.

The shared surface convention is
``F = -grad(phi) + rhat x grad(psi)``. Stored ``Phi`` and ``W`` are the
curl-free and divergence-free electric-potential coefficients normalized by
``RI``; they therefore have units of V/m, while visualization multiplies them
by ``RI`` to display volts. Stored magnetic variables are instead physical:
``induced_Br`` is the continuous induced radial field at ``RI``,
``boundary_jr`` is radial current density at ``RI+``, and ``boundary_Br`` is
the prescribed radial field at ``RM``. Private potential coordinates obey
``boundary_jr = RI / mu0 * surface_laplacian(toroidal_potential)``,
``induced_Br = -RI**2 * poloidal_laplacian(induced_poloidal_potential)``, and
``d(induced_poloidal_potential)/dt = W / RI``. Exact forward and inverse maps
keep these numerical coordinates out of the public schema without changing
the established sign, radius, or time-evolution convention.

The CS toroidal-potential system has one physically irrelevant constant gauge.
``surface_gauge_operator`` adds the exact zero-area-mean equation needed to
make its coefficient solution unique. This is a constraint, not Tikhonov
regularization: ``toroidal_potential_regularization_lambda`` remains an optional numerical
policy that also damps physically observable coefficient directions when it
is positive.

``SimulationGeometry`` names describe the physical map rather than the symbols
used in one derivation. In particular, ``pedersen_geometry_tensor``,
``hall_geometry_tensor``, and ``wind_motional_E_tensor`` are pointwise maps,
while ``induced_poloidal_potential_faraday_rate_scale`` converts the
divergence-free electric potential to the private potential rate.
``boundary_jr_to_gap_Br_matrix`` denotes the specific PFAC-derived physical
map from upper-boundary current to unshielded gap radial field; its persisted
artifact is ``gap_Br_response``.

The interhemispheric names distinguish a physical region from a boundary
condition. ``interhemispheric_coupling_latitude`` bounds the low-latitude
region where conjugate points are compared; it is not another magnetic
boundary. ``radial_current_constraint_operator`` is the assembled map used by
the toroidal-potential solve: it is the local radial-to-apex map outside that
region and the local-minus-conjugate map inside it.
``interhemispheric_electric_field_weight`` is specifically the relative
least-squares weight of the conjugate electric-field residual. Conjugate grids,
transforms, and masks are absent when this coupling is disabled or cannot
apply to the radial-field model.

Response and evolution
----------------------

``Simulation`` exposes ``evolve_to_time`` as the stable public API, but delegates
execution to ``_TimeEvolution``. ``_TimeEvolution`` owns restart
short-circuiting, sample scheduling, progress reporting, and output save
decisions, including assembly of complete output snapshots and the one-shot
``impose_equilibrium`` implementation behind the public facade. The requested
target time is always an exact final sample and checkpoint, including when it
is not an integer multiple of the nominal time step. Sampling and saving
intervals are validated as positive integers, and a later active checkpoint
cannot be used to fabricate a missing earlier output. The evolution object also
reuses exponential propagators while the closure operator and step duration remain
unchanged. It may decide when to update active inputs or save outputs, but
time-stepping equations belong in
``electrodynamics.induction``, closure-dependent operator caches belong in
``ElectrodynamicResponse``, and artifact details belong in ``SimulationData``.

Imposing an equilibrium always updates the in-memory ``dynamic`` stream; its
``save`` option controls only whether that live checkpoint is persisted.
Imposition before a later active checkpoint is rejected because retaining both
would create two trajectory branches in one linear time-series artifact.

``ElectrodynamicResponse`` stores the active input coefficients and the
algebraic response implied by the current resistance. The evolving
``induced_Br`` remains an explicit value owned by the running branch because
one simulation can carry both dynamic and equilibrium solutions for the same active
inputs.
``_TimeEvolution`` manages those branch lifetimes and passes their
coefficient vectors into response methods. The persisted artifact name
``dynamic`` identifies the time-dependent output stream; ``equilibrium``
identifies the instantaneous zero-Faraday-rate comparison.

Persistence
-----------

``SimulationData``, ``ArtifactStore``, and ``FieldTimeSeries`` form the
persistence boundary.
Numerical modules should pass normalized coefficient rows and times to the
time-series APIs instead of writing xarray artifacts directly.  This makes
restart behavior, netCDF/zarr differences, and schema compatibility easier to
maintain. Time values are finite scalar coordinates. Near-equal floating
checkpoint labels replace one another with the declared absolute tolerance so
roundoff cannot create duplicate logical times.

Coefficient time series have two intentional representations. In memory, each
coefficient dimension carries its schema ``MultiIndex``. A group with one
coefficient space keeps the compact ``i`` dimension; a mixed group, such as CS
output with SH ``induced_Br`` and CS surface quantities, uses distinct
``sh_i`` and ``cs_i`` dimensions. On disk, those indexes are reset to explicit
coordinate columns so NetCDF and Zarr persist the same portable structure.
Loading validates every column and reconstructs the in-memory indexes before
exposing the series.

The time tolerance is a time-coordinate policy measured in seconds. It is
shared with evolution checkpoint decisions but is distinct from the relative
tolerance used to decide whether coefficient values changed.

An ``ArtifactStore`` instance is bound to one resolved artifact directory and one preferred
storage policy. Create another instance for another simulation instead of retargeting
an existing handle. Time-series storage owns artifact append/rewrite decisions;
projection policy and spatial weights do not belong in persisted coefficient
containers.

One logical artifact has one physical storage representation. A successful
format change removes the alternate NetCDF/Zarr path, and ambiguous legacy
duplicates fail loudly instead of silently preferring stale data. Complete
NetCDF and Zarr rewrites use unique sibling temporary paths before replacement;
incremental Zarr appends remain the time-series layer's explicit fast path.

The simulation schema owns the complete vocabulary of simulation artifact names.
``ArtifactStore`` remains generic: directory validation requires an explicit collection
of artifact names, and the persistence primitive contains no knowledge of
settings, gap responses, or output-stream identities. Artifact names are single
path-safe components.

When adding persisted data, decide first whether it is configuration, input,
output, or derived visualization metadata.  Configuration belongs in
``SimulationConfig``; input and output coefficient time series belong in the
simulation schema; visualization defaults belong in serializable figure settings.

Visualization
-------------

Visualization code should treat saved simulations as read-only inputs.  The
``FigureSettings`` and panel binding layer make figure configuration
serializable, reusable, and testable outside the GUI.  Keep option validation
close to the settings, rendering close to figure builders, and widget binding close
to panel-specific modules.

Saved simulation loading has one persistence path. ``SimulationResults`` uses the same
``ArtifactStore`` abstraction as simulation persistence and constructs the canonical
configuration, schema, main field, and optional geometry. The plotting-grid
``PlotData`` builds on that loaded context rather than
reimplementing artifact discovery. Figure renderers directly retain their
serializable settings and cached grid fields, then own only
their respective figure families.
Coefficient artifacts are validated by the schema-aware ``FieldTimeSeries`` loader;
plotting asks that schema for the exact stored variable name instead of guessing from
prefixes or suffixes. Numeric saved times are interpreted relative to the simulation's
physical ``t0`` and are never silently displayed as Unix-epoch times.

The saved-view cache has one versioned slot per resolved simulation and plotting
resolution. Artifact changes replace the previous heavyweight view in that
slot, preventing live simulations from accumulating obsolete transform and
geometry objects. Its fingerprint descends into directory-backed artifacts so
an incremental Zarr chunk append invalidates the view even when the store's
top-level directory timestamp does not change.

Specialized saved views should retain canonical context, not reconstruct or
re-own it. ``SimulationResults`` is the sole owner of the simulation directory, artifacts,
configuration, schema, main field, and optional geometry.
``PlotData`` owns only plotting grids, spherical transforms, and derived
field caches. SimulationGeometry and dense sheet-current maps are built only when an
output-field calculation needs them, and the sheet-current maps are
specifically deferred until Joule heating is requested. Input-driver and
ordinary scalar-field figures do not pay that cost merely because output
artifacts also exist. Renderers consume the shared objects instead of
rebuilding main fields, schemas, and transforms from raw settings. An
``equilibrium`` artifact is valid output even when no ``dynamic`` artifact is
present; only difference plots require both.

Saved simulation behavior enters through ``SimulationResults`` or ``PlotData``
and the ``FigureSettings`` renderer path, rather than maintaining a second
loading or rendering stack.

Avoid adding simulation-specific calculations directly to GUI callbacks.  If a
plot needs computed fields, expose them through saved-results, grid-field, or
figure-builder helpers so command-line scripts, notebooks, tests, and the GUI
all use the same path. Derived physical quantities should call the equation
kernels that define them. In particular, both saved-results frontends use the
closure's total-sheet-current Joule-heating definition, including prescribed
magnetic-boundary current, rather than reconstructing a second approximation
inside visualization.

Extension rules
---------------

Use these rules when extending the codebase:

* Add settings to ``SimulationConfig`` before threading ad hoc constructor
  values through the system.
* Add persisted streams to ``simulation.schema`` before adding storage code.
* Add input projection behavior to ``_InputProjector`` and its specification
  table before adding logic to ``Simulation`` setters.
* Put reusable boundary-current equations in
  ``electrodynamics.magnetic_boundary``, constitutive ``JS``/wind-to-``E``
  equations in ``electrodynamics.ionospheric_closure``, and magnetic
  time-stepping equations in ``electrodynamics.induction``.
* Add execution behavior to ``_TimeEvolution`` before expanding
  ``Simulation.evolve_to_time``.
* Add cubed-sphere internals to the focused CS collaborator that owns the
  concept.
* Move reusable script logic into package modules before testing it.
* Keep visualization configuration serializable and renderer-agnostic.
* Prefer functionality tests that assert scientific or user-facing behavior
  over tests that encode a refactor's private implementation shape.
