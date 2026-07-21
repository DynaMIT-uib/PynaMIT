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
  ``pynamit.simulation``. The high-level optional visualization API is the
  explicit lazy export set of ``pynamit.visualization``. Together they contain
  the simulation facade, normalized configuration, field values and spaces,
  spherical bases and transforms, background-field utilities, backend
  selection, saved-run views, and renderers.
* Reusable packages such as ``pynamit.sphere``, ``pynamit.geomagnetism``,
  ``pynamit.math``, and ``pynamit.storage`` provide advanced scientific and
  infrastructure APIs. Their package export lists define the supported
  entry points; implementation submodules do not extend that promise merely
  because Python can import them.
  Focused electrodynamics and workflow functions are advanced module APIs at
  their documented module paths rather than primary package exports.
* ``Simulation.config``, ``Simulation.current_time``,
  ``Simulation.run_data``, ``Simulation.geometry``, and
  ``Simulation.response`` are stable attributes for inspection and
  diagnostics. This guarantees the ownership path, not every constructor or
  non-underscored member of the collaborator's concrete class. Named geometry
  maps and response operators are advanced diagnostic surfaces and may be
  expensive to materialize.
* Underscored attributes, including ``simulation._input_pipeline`` and
  ``simulation._runner``, caches, and scheduling helpers are internal. Tests
  may exercise these objects directly without turning them into user API.

Compatibility names such as ``BasisEvaluator``, ``set_u``, and the
``SphericalTransform.G*`` properties remain supported, but implementation
code uses the canonical descriptive names. New code should enter through the
primary names rather than adding further aliases.

High-level flow
---------------

``Simulation`` remains the public orchestration object.  It normalizes user
configuration with ``SimulationConfig``, builds a ``SimulationSchema``,
creates persistence through ``RunData``, constructs sibling
``SimulationGeometry`` and ``ElectrodynamicResponse`` collaborators, and keeps the user-facing
simulation methods in one place.

The main setup path is:

1. ``SimulationConfig`` normalizes constructor settings, defaults, numerical
   policies, basis choices, physical domains, time values, and persisted
   xarray attributes.
2. ``build_simulation_schema`` creates the spherical-harmonic, solid-harmonic,
   and cubed-sphere basis objects, then declares the input and output
   ``FieldSpace`` metadata used by storage and transforms.
3. ``Simulation`` attaches an ``InputPipeline``, which lazily builds reusable
   input transforms on the geometry's canonical model grid and owns all input
   validation, projection, and coefficient storage.
4. ``Simulation`` attaches a ``SimulationRunner`` for restart handling, sample
   scheduling, progress reporting, and output save decisions.
5. ``SimulationGeometry`` is constructed once as the numerical spatial context.
   ``ElectrodynamicResponse`` receives it and owns the instantaneous forcing
   coefficients, constraint solve, and closure-dependent operator caches.
6. ``RunData`` owns persisted settings, input time series, and output
   time series.

Persisted objects are reached through ``simulation.run_data`` (for example
``simulation.run_data.output_series`` and ``simulation.run_data.schema``),
while the horizontal basis, solid harmonics, and background field are reached
through ``simulation.geometry``. ``Simulation`` does not copy those
references onto parallel top-level attributes. This keeps its initialization
small and makes the owner of each object unambiguous.

Keep this layering intact: configuration should not perform numerical work,
schema should not read run data, input projection should not evolve the model,
and visualization should not mutate simulation state.

Package map and dependency direction
------------------------------------

The top-level packages separate reusable scientific and infrastructural
concepts from the PynaMIT simulation model::

    pynamit/
      fields.py              coefficient-space metadata and owned values
      coordinates.py         generic longitude and local-time conversions
      math/                  backend-neutral operators and solvers
      sphere/                spherical bases, grids, and transforms
      geomagnetism/          background fields and magnetic coordinates
      storage/               named artifacts and coefficient time series
      simulation/            the coupled PynaMIT model
      visualization/         read-only run views and rendering

The simulation package is grouped by runtime role::

    simulation/
      __init__.py            stable Simulation and SimulationConfig exports
      api.py                 public simulation facade
      response.py            active inputs, response solves, and operator caches
      geometry.py            run-specific spatial and magnetic mappings
      runner.py              execution, sampling, and persistence decisions
      inputs.py              input validation, projection, and storage
      config.py/schema.py    normalized configuration and field spaces
      run_data.py            persisted-run context
      electrodynamics/       magnetic boundary, closure, and induction equations
      workflows/             end-to-end preparation and run orchestration

Dependencies should point inward from workflows and facades toward focused
implementation modules.  In particular:

* ``workflows`` may construct and operate ``Simulation``;
* ``Simulation`` may coordinate projection, response, evolution, and persistence;
* ``ElectrodynamicResponse`` and ``SimulationGeometry`` may compose ``electrodynamics`` functions; and
* ``electrodynamics`` must remain free of imports from simulation
  orchestration and persistence modules.

The reusable packages must not import the simulation layer. In particular,
``geomagnetism`` owns ``MainField``, magnetic-coordinate conversion, and
``MagneticFieldEvaluation`` without knowing about ``SimulationConfig``;
``geometry.build_main_field`` is the small adapter from run configuration to
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
is immutable run configuration and, when omitted, is derived from that run's
``RI`` and optional ``RM``. Basis resolutions, radial boundaries, main-field
parameters, boolean choices, solver names, and integrators are validated
before schema or geometry construction so invalid runs cannot partially
initialize. Restart comparison is performed between normalized configurations,
allowing old artifacts to inherit newly introduced defaults without allowing
an actual setting mismatch.

Once normalized, one immutable ``SimulationConfig`` instance is shared by
``Simulation``, ``RunData``, and ``ElectrodynamicResponse``. Schema and persistence
builders do not accept parallel setting overrides; callers that need to adapt
legacy settings do so through ``SimulationConfig.from_settings`` before
crossing those boundaries. Runtime policy is read from that shared config
rather than copied into mutable state attributes.

Programmatic callers can pass a normalized configuration through
``Simulation.from_config``. Storage format, run directory, and array backend
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
``m_imp_regularization_lambda=0.0`` leaves the physical imposed-potential
problem unregularized. Reproducibility workflows pin both choices explicitly
instead of inheriting a future policy change.

Numerical operators and value identity
----------------------------------------

``LinearMap`` is the common operator abstraction for dense, sparse,
matrix-free, and structured-einsum calculations. Least-squares problems and
physical compositions should retain ``LinearMap`` objects until an explicit
matrix is genuinely required. The tensor helper module contains only
contractions and pseudoinverses that still operate on multidimensional arrays;
do not add parallel wrappers for operations already expressed by
``LinearMap``.
When a rectangular map must become explicit, materialization probes the
smaller side: input columns for tall maps and adjoint output rows for wide
maps. This is important for rectangular surface-to-poloidal and PFAC maps,
whose compact physical output should determine temporary memory use.

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

``Grid`` is an immutable coordinate value. Its coordinate signature defines
grid-value compatibility and remapping identity. Its separate analysis
signature also includes optional area weights: equal coordinates can share
synthesis matrices while requiring different weighted least-squares analyses.
Transform caches must choose deliberately between those two identities.
Mean-freedom is owned by coefficient spaces, not by post-step cleanup. The
schema's ``sh_basis`` retains the mean/monopole term for quantities such as
resistance, while ``mean_free_sh_basis`` is used for radial magnetic fields.
The evolving ``m_ind`` and prescribed boundary ``Br`` therefore live directly
in the mean-free poloidal SH space. ``m_imp``, ``Phi``, and ``W`` live in the
selected horizontal surface space. In CS mode the latter space retains its
nodal layout and uses an explicit zero-area-mean gauge constraint. Each stored
representation supplies its own synthesis operator on the model grid, so
``SimulationGeometry`` does not carry a parallel full-scalar transform.

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
possible. ``CSBasis`` is the public cubed-sphere basis facade. Its private
implementation is split into:

* ``cs_coordinates`` for panel and coordinate transforms.
* ``CSGridGeometry`` and ``CSGridRemapper`` for grid shape, indexing, and
  remapping, including scattered scalar and vector interpolation.
* ``CSFiniteDifferences`` for derivative stencils and sparse operators.
* ``cs_vectors`` for vector-basis conversions.

Prefer adding focused CS behavior to one of these collaborators instead of
growing ``CSBasis`` again. Stateless coordinate and vector calculations are
module functions rather than artificial namespace objects. The familiar
coordinate and interpolation methods remain on ``CSBasis`` as its public
facade, while their implementations belong to those focused modules. Keep the
``CS`` abbreviation in class names that represent actual objects.

``sphere`` is a warranted standalone package because its bases, grids,
analysis/synthesis transforms, and solid-harmonic continuation form a coherent
numerical domain that can be used without a PynaMIT run.

``geomagnetism`` is the corresponding standalone physical domain. ``MainField``
owns background-field models, field-line and conjugate mapping, magnetic
coordinates, and apex basis vectors. ``MagneticFieldEvaluation`` binds one
main field to one spherical grid and radius, caching field components and local
maps used by ``SimulationGeometry``. Full field components use ``Br``, ``Btheta``, and
``Bphi``; normalized components are explicitly named ``unit_br``,
``unit_btheta``, and ``unit_bphi`` so a case-only spelling difference cannot
change the physics.

Simulation state coordinates are Earth-fixed. ``kaiju_dipole``, ``igrf``, and
``radial`` use geocentric geographic coordinates, so the SH and cubed-sphere
positions denote locations on Earth directly. The generic idealized ``dipole``
model retains centered-dipole ``MAG`` coordinates. For ``kaiju_dipole``,
``MainField`` alone owns the fixed GEO-to-MAG rotation: it evaluates the
analytic dipole, field-line mapping, conjugacy, magnetic latitude, and apex
basis vectors in MAG and returns coordinates and vector components in GEO.
The Kaiju/Geopack dipole coefficients and axis are frozen at
``main_field_epoch``. ``SM`` remains a timestamped external-source coordinate
system whose Sun-facing longitude origin is transformed at every source time;
it is never the coordinate space of ``m_ind`` or another evolving PynaMIT
state. The background field is not re-tilted every timestep.

Do not promote every focused module to a package. A standalone package is
warranted when the concept has a reusable public vocabulary and a dependency
boundary of its own. The PynaMIT boundary, ionospheric closure, and induction
equations are mutually coupled parts of one simulation model, so they remain
together under ``simulation.electrodynamics``.

Input preparation and projection
--------------------------------

Input projection is intentionally separated from ``Simulation`` in
``pynamit.simulation.inputs``.  Its private specification table declares the
variables, mutual-exclusion group, and projection-control restrictions for
each input stream. Every persisted input also has an explicit projection-basis
setting in ``SimulationConfig``. ``Q_eff`` defaults to the ``u`` route because
it is an alternative representation of the same wind forcing; the independent
``E_source`` route defaults to the horizontal model basis. Field type remains
canonical in the schema's ``FieldSpace``. ``InputPipeline`` owns:

* sample-vs-coefficient validation;
* gridded scalar and tangential projection;
* coefficient row and time-row validation;
* storage of projected input rows; and
* mutual exclusivity between the alternative wind representations ``u``
  and ``Q_eff``;
* additive composition of independent ``E_source`` forcing; and
* time-series coordination when deriving ``Q_eff`` from neutral wind.

Public setters such as ``set_jr``, ``set_resistance``, ``set_neutral_wind``,
``set_Q_eff``, and ``set_E_source`` should remain thin API methods.  When a
new input stream is added, prefer extending the schema and the private input
specification table over hand-writing a new projection path inside ``Simulation``.
``set_Q_eff_from_neutral_wind`` follows one coefficient-space route: it fits
the stored ``Q_eff`` so its resistance-weighted electric response matches the
projected wind forcing. ``calculate_Q_eff_from_neutral_wind`` is the separate
non-persisting diagnostic for callers that need the equivalent model-grid
field.

Prepared external inputs
------------------------

``pynamit.simulation.workflows.standard``, ``prepared_inputs``, and
``mage_projection`` contain reusable, script-independent orchestration for
standard runs, prepared forcing, and MAGE/GAMERA projection. Scripts under
``scripts/simulation`` are workflow entry points: they own editable experiment
settings, paths, and directory naming while delegating reusable validation and
numerical work to package modules.

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
``mage_projection`` owns forcing-schema validation, fixed projection geometry,
least-squares weighting, and construction of a reusable coefficient-space input
package. Internally, one private projector owns the grids and numerical
operators that are invariant across forcing times; the public ``project_inputs``
function owns the HDF5 and manifest workflow. It builds in a temporary sibling
directory and replaces the published artifacts only after every projected time
and the package manifest succeed.
Finally, ``mage_run.py`` creates any number of named runs from one projected
package. These boundaries prevent external-reader concerns, numerical
projection, and simulation experiments from sharing mutable state.

The run manifest snapshots the prepared-input manifest and selected streams.
An existing trajectory can be resumed or extended only when that identity and
its evolution policy still match; a different projection or experiment uses a
new run directory. This prevents newly copied inputs from being paired with
state outputs computed from an older forcing package.

This boundary matters because prepared-input logic is both user-facing and
testable.  If a script needs behavior that should remain correct over time,
move it into the package and test the behavior there.

A consuming run owns its selected inputs. The workflow removes stale,
unselected input artifacts, opens only the selected prepared streams, copies
those datasets into the run directory, and reloads them through the run's own
``ArtifactStore`` object. Lazy Zarr
arrays therefore do not leave a live run dependent on the preparation
directory. PFAC integration samples are re-derived from the consuming run's
radial domain unless that run supplies them explicitly; they are not part of
the prepared coefficient contract. The MAGE run uses
``dipole_fac_integration_radii`` with an editable point count, so this PFAC
discretization remains a run choice rather than a projection side effect.

For generic prepared inputs, coordinates name physical positions, not just
array axes.
Geographic wind positions and tangent-vector components are rotated into the
configured model coordinates before projection. Native Hardy/AMPS scalar
models are instead queried in their event-epoch centered-dipole coordinates;
their values are then attached to the corresponding positions on the model
grid. This two-sided conversion is necessary when the run uses IGRF/GEO or a
centered dipole with another epoch. The forcing event time is persisted
as the input time origin ``t0``; ``main_field_epoch`` remains an independent
choice for the Earth-fixed background-field coefficients and coordinate axis.

The versioned prepared-input manifest has one canonical ``input_contract``.
Coefficient-space settings, geometry requirements, and the dataset list live
only inside that contract; top-level provenance and notes do not mirror contract
fields. Consumers validate the manifest version and contract before loading any
forcing into a run.

Electrodynamic physics
----------------------

The ``electrodynamics`` modules follow the direction of the model equations
instead of mirroring the names of state variables:

* ``electrodynamics.magnetic_boundary`` maps magnetic boundary potentials and
  prescribed boundary ``Br`` to the derived horizontal sheet current ``JS``.
  Solid harmonics own generic radial continuation; this module owns the
  particular potential jump, shielding, and boundary-current relations used
  by PynaMIT.
* ``electrodynamics.ionospheric_closure`` applies the height-integrated
  Ohm-law closure.  It converts Hall/Pedersen conductance to the stored
  resistance variables and maps neutral motion or sheet current through the
  magnetic geometry and resistance tensor to ``E``. It owns both the direct
  grid law ``E = R JS - u x B`` and the coefficient-operator compositions
  used by ``ElectrodynamicResponse``. It also owns the Pedersen/Hall geometry tensors and
  collisional Joule-heating kernel. Joule heating is the Pedersen dissipation
  ``etaP * J.T @ P @ J``; ``J dot E`` is electromagnetic work and is not
  generally the same quantity when neutral motion contributes to the closure.
  Its functions remain numerical kernels; iteration over input times and
  coefficient storage belong to ``InputPipeline``.
* ``electrodynamics.induction`` owns Faraday evolution of ``m_ind``, including
  Euler, exponential, and SciPy integration and the corresponding steady
  state. Its coefficients and electric-field inputs share the fixed model
  frame. Motion of an SM forcing pattern therefore appears through the
  timestamped SM-to-GEO preparation, not through a rotating state basis.

``SimulationGeometry`` supplies run-specific grids, magnetic-field factors, transforms,
radial field-line mapping, and interhemispheric constraint geometry to these
equations. It is a numerical object: persisted xarray values are unwrapped at
construction, and ``RunData`` alone wraps a numerical PFAC matrix for
storage. ``ElectrodynamicResponse`` receives that geometry, owns the inputs active at
one simulation time, solves the imposed-potential constraint, and caches the
composed operators whose values depend on the current resistance distribution.
It also exposes named operator and matrix compositions for inspection. Keeping
those compositions with their caches avoids split ownership between an
operator facade and the response object whose private state they mutate.

The combined field that drives the imposed-potential response is named
``driving_E``. It can contain wind, ``Q_eff``, ``E_source``, boundary ``Br``,
or an ``m_ind`` response. The name therefore describes its role without
suggesting that it contains only the direct electric-field input or encoding
the absence of ``m_imp``. The imposed potential then completes the response
required by the radial-current and optional interhemispheric constraints.
``E_source`` is the public name for an independently supplied additive
electric-field term. It does not claim that the stored field is the total
electric field. In the MAGE workflow it contains only the wind-driven term
derived from Pedersen/Hall-weighted winds; the model composes the other terms
while solving the closure.

The poloidal and horizontal surface spaces coincide in the default SH mode.
They are intentionally distinct in CS mode. ``surface_to_poloidal_operator``
is the only bridge in Faraday's law: it projects the surface ``W`` potential
onto the configured poloidal harmonics. PFAC coupling has the corresponding
rectangular shape. ``m_imp`` directly describes the imposed sheet-current
contribution in the surface space; the field-aligned current above the
ionosphere also produces a poloidal magnetic contribution and a potential
jump across the sheet, which the PFAC map expresses in poloidal coefficients.
This prevents unobservable high-resolution CS modes from being carried as
part of the evolving poloidal state.

Expensive optional geometry follows use rather than construction. The PFAC
coupling matrix is reused when persisted, but a new one is not built merely to
construct ``Simulation`` or project inputs. It is constructed and saved when a
steady-state or evolution path first requests model output. When PFAC coupling
is disabled (or the main field is radial), the optional contribution is
represented by absence rather than by constructing and multiplying a dense
zero matrix.

Surface-sized operators should remain structured in CS mode. In particular,
native Helmholtz analysis, wind and sheet-current closure compositions, and
ordinary runtime ``m_imp`` responses must not materialize dense surface maps.
The surface-to-poloidal bridge composes a factorized full-column-rank SH
analysis with the horizontal synthesis operator; native CS nodal synthesis
therefore stays an implicit identity rather than allocating a dense grid-sized
identity. The analogous poloidal-current fit used while constructing PFAC also
keeps a factorized normal system and an implicit adjoint instead of forming an
SVD pseudoinverse of its tall grid matrix.
The compact poloidal feedback matrix remains intentionally dense because its
matrix exponential and steady-state pseudoinverse require an explicit
poloidal operator. The steady-state response composes that compact
pseudoinverse with the structured surface-to-poloidal bridge; its full
cross-space matrix is materialized only when explicitly requested for
diagnostics. The generic dense ``normal_pinv`` solver is retained for
reproducibility; large CS imposed-potential solves should select ``lsmr`` (or
``cgls``) with the ``jacobi`` preconditioner.

The imposed-potential runtime follows the same rule. A single active ``jr`` or
driving-E field is assembled as one physical least-squares right-hand side and
solved directly. The full ``jr_to_m_imp_operator`` and
``driving_E_to_m_imp_operator`` remain available for matrix diagnostics, but
normal simulation steps do not construct them. Interhemispheric induction
solves only the source columns reachable from the poloidal ``m_ind`` state,
rather than first constructing the response to every possible horizontal-E
coefficient.

Magnetic-boundary maps also retain structured ``LinearMap`` compositions.
This matters in CS mode, where the native gradient and Helmholtz synthesis
operators are sparse. Repeated E-response maps are cached densely for compact
poloidal inputs and when the horizontal and poloidal SH spaces coincide; large
CS surface-to-surface maps remain structured until an explicit matrix is
requested.

The persisted input artifact is named ``resistance`` because its canonical
variables are the Pedersen and Hall resistance coefficients ``etaP`` and
``etaH``. ``set_conductance`` remains the physical convenience API: it accepts
``sigmaP`` and ``sigmaH`` in siemens, performs the pointwise tensor inversion,
and then projects the resulting resistance. ``set_resistance`` accepts the
canonical stored variables directly. Visualization converts resistance back
to conductance when a figure actually displays conductance.

The shared surface convention is
``F = -grad(phi) + rhat x grad(psi)``. Stored ``Phi`` and ``W`` are the
curl-free and divergence-free electric-potential coefficients normalized by
``RI``; they therefore have units of V/m, while visualization multiplies them
by ``RI`` to display volts. Stored ``m_ind`` and ``m_imp`` follow the normalized
magnetic-potential convention documented with ``SolidHarmonics``. With these
definitions, ``m_imp_to_jr = RI / mu0 * surface_laplacian``,
``m_ind_to_Br = -RI**2 * poloidal_laplacian``, and
``d(m_ind)/dt = W / RI`` use one sign and radius convention across the code.

The CS imposed-potential system has one physically irrelevant constant gauge.
``surface_gauge_operator`` adds the exact zero-area-mean equation needed to
make its coefficient solution unique. This is a constraint, not Tikhonov
regularization: ``m_imp_regularization_lambda`` remains an optional numerical
policy that also damps physically observable coefficient directions when it
is positive.

``SimulationGeometry`` names describe the physical map rather than the symbols
used in one derivation. In particular, ``pedersen_geometry_tensor``,
``hall_geometry_tensor``, and ``wind_motional_E_tensor`` are pointwise maps,
while ``faraday_rate_scale`` converts the divergence-free electric potential
to the magnetic-potential rate. ``pfac_coupling_matrix`` denotes the specific
PFAC toroidal-to-poloidal coupling from the model equations; its persisted
artifact remains ``PFAC_matrix``.

The interhemispheric names distinguish a physical region from a boundary
condition. ``interhemispheric_coupling_latitude`` bounds the low-latitude
region where conjugate points are compared; it is not another magnetic
boundary. ``radial_current_constraint_operator`` is the assembled map used by
the imposed-potential solve: it is the local radial-to-apex map outside that
region and the local-minus-conjugate map inside it.
``interhemispheric_electric_field_weight`` is specifically the relative
least-squares weight of the conjugate electric-field residual. Conjugate grids,
transforms, and masks are absent when this coupling is disabled or cannot
apply to the radial-field model.

Response and evolution
----------------------

``Simulation`` exposes ``evolve_to_time`` as the stable public API, but delegates
execution to ``SimulationRunner``. ``SimulationRunner`` owns restart
short-circuiting, sample scheduling, progress reporting, and output save
decisions, including assembly of complete output snapshots and the one-shot
``impose_steady_state`` implementation behind the public facade. The requested
target time is always an exact final sample and checkpoint, including when it
is not an integer multiple of the nominal time step. Sampling and saving
intervals are validated as positive integers, and a later active checkpoint
cannot be used to fabricate a missing earlier output. The runner also reuses
exponential propagators while the closure operator and step duration remain
unchanged. The runner may decide when to update active inputs or save outputs, but
time-stepping equations belong in
``electrodynamics.induction``, closure-dependent operator caches belong in
``ElectrodynamicResponse``, and artifact details belong in ``RunData``.

Imposing a steady state always updates the in-memory ``state`` stream; its
``save`` option controls only whether that live checkpoint is persisted.
Imposition before a later active checkpoint is rejected because retaining both
would create two trajectory branches in one linear time-series artifact.

``ElectrodynamicResponse`` is intentionally not called ``State``. It stores the active
input coefficients and the algebraic response implied by the current
resistance, but it does not own a unique evolving ``m_ind``. One run can carry
both inductive and steady-state ``m_ind`` branches for the same active inputs.
``SimulationRunner`` owns those branch lifetimes and passes their coefficient
vectors into response methods. The persisted artifact name ``state`` continues
to identify the inductive output stream, not an in-memory object hierarchy.

Persistence
-----------

``RunData``, ``ArtifactStore``, and ``FieldTimeSeries`` form the
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
output with SH ``m_ind`` and CS surface potentials, uses distinct ``sh_i`` and
``cs_i`` dimensions. On disk, those indexes are reset to explicit coordinate
columns so NetCDF and Zarr persist the same portable structure. Loading
validates every column and reconstructs the in-memory indexes before exposing
the series.

The time tolerance is a time-coordinate policy measured in seconds. It is
shared with evolution checkpoint decisions but is distinct from the relative
tolerance used to decide whether coefficient values changed.

An ``ArtifactStore`` instance is bound to one resolved artifact directory and one preferred
storage policy. Create another instance for another run instead of retargeting
an existing handle. Time-series storage owns artifact append/rewrite decisions;
projection policy and spatial weights do not belong in persisted coefficient
containers.

One logical artifact has one physical storage representation. A successful
format change removes the alternate NetCDF/Zarr path, and ambiguous legacy
duplicates fail loudly instead of silently preferring stale data. Complete
NetCDF and Zarr rewrites use unique sibling temporary paths before replacement;
incremental Zarr appends remain the time-series layer's explicit fast path.

The simulation schema owns the complete vocabulary of run artifact names.
``ArtifactStore`` remains generic: directory validation requires an explicit collection
of artifact names, and the persistence primitive contains no knowledge of
settings, PFAC, or field-stream identities. Artifact names are single
path-safe components.

When adding persisted data, decide first whether it is configuration, input,
output, or derived visualization metadata.  Configuration belongs in
``SimulationConfig``; input and output coefficient time series belong in the
simulation schema; visualization defaults belong in serializable figure specs.

Visualization
-------------

Visualization code should treat saved runs as read-only inputs.  The newer
``PynamitFigureSpec`` and panel binding layer make figure configuration
serializable, reusable, and testable outside the GUI.  Keep option validation
close to the spec, rendering close to figure builders, and widget binding close
to panel-specific modules.

Saved-run loading has one persistence path. ``SavedRunView`` uses the same
``ArtifactStore`` abstraction as simulation persistence and constructs the canonical
configuration, schema, main field, and optional geometry. The plotting-grid
``SavedCoefficientFieldView`` builds on that loaded context rather than
reimplementing artifact discovery. Figure renderers directly retain their
serializable specification and cached coefficient-field view, then own only
their respective figure families.

The saved-view cache has one versioned slot per resolved run and plotting
resolution. Artifact changes replace the previous heavyweight view in that
slot, preventing live simulations from accumulating obsolete transform and
geometry objects. Its fingerprint descends into directory-backed artifacts so
an incremental Zarr chunk append invalidates the view even when the store's
top-level directory timestamp does not change.

Specialized saved views should retain canonical context, not reconstruct or
re-own it. ``SavedRunView`` is the sole owner of the run directory, artifacts,
configuration, schema, main field, and optional geometry.
``SavedCoefficientFieldView`` owns only plotting grids, evaluators, and derived
field caches. SimulationGeometry and dense sheet-current maps are built only when an
output-field calculation needs them, and the sheet-current maps are
specifically deferred until Joule heating is requested. Input-driver and
ordinary scalar-field figures do not pay that cost merely because output
artifacts also exist. Renderers consume the shared objects instead of
rebuilding main fields, schemas, and transforms from raw settings. A
steady-state artifact is valid output even when no inductive ``state`` artifact
is present; only difference plots require both.

``PynamEye`` and ``visualization.results`` are legacy frontends retained for
existing scripts. New saved-run behavior should enter through ``SavedRunView``
or ``SavedCoefficientFieldView`` and the figure-spec renderer path, not create a
third loading or rendering stack.

Avoid adding simulation-specific calculations directly to GUI callbacks.  If a
plot needs computed fields, expose them through saved-run, run-field, or
figure-builder helpers so command-line scripts, notebooks, tests, and the GUI
all use the same path. Derived physical quantities should call the equation
kernels that define them. In particular, both saved-run frontends use the
closure's total-sheet-current Joule-heating definition, including prescribed
magnetic-boundary current, rather than reconstructing a second approximation
inside visualization.

Extension rules
---------------

Use these rules when extending the codebase:

* Add settings to ``SimulationConfig`` before threading ad hoc constructor
  values through the system.
* Add persisted streams to ``simulation.schema`` before adding storage code.
* Add input projection behavior to ``InputPipeline`` and its specification
  table before adding logic to ``Simulation`` setters.
* Put reusable boundary-current equations in
  ``electrodynamics.magnetic_boundary``, constitutive ``JS``/wind-to-``E``
  equations in ``electrodynamics.ionospheric_closure``, and magnetic
  time-stepping equations in ``electrodynamics.induction``.
* Add execution behavior to ``SimulationRunner`` before expanding
  ``Simulation.evolve_to_time``.
* Add cubed-sphere internals to the focused CS collaborator that owns the
  concept.
* Move reusable script logic into package modules before testing it.
* Keep visualization configuration serializable and renderer-agnostic.
* Prefer functionality tests that assert scientific or user-facing behavior
  over tests that encode a refactor's private implementation shape.
