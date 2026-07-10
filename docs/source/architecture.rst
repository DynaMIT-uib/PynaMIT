Architecture
============

PynaMIT is organized around a small number of durable responsibilities:
configuration and schema construction, spatial bases, input preparation,
time evolution, persistence, and visualization.  The simulation APIs should
stay readable when used interactively, while the implementation keeps the
physics-specific transformations in focused modules.

High-level flow
---------------

``Dynamics`` remains the public orchestration object.  It normalizes user
configuration with ``SimulationConfig``, builds a ``SimulationSchema``,
creates persistence through ``SimulationData``, initializes ``State``, and
keeps the user-facing simulation methods in one place.

The main setup path is:

1. ``SimulationConfig`` normalizes constructor settings, defaults, backend
   options, basis choices, and persisted xarray attributes.
2. ``build_simulation_schema`` creates the spherical-harmonic, solid-harmonic,
   and cubed-sphere basis objects, then declares the input and output
   ``FieldSpace`` metadata used by storage and transforms.
3. ``Dynamics`` builds reusable input transforms from those field spaces and
   attaches an ``InputProjector`` for all input validation, projection, and
   coefficient storage.
4. ``Dynamics`` attaches an ``EvolutionRunner`` for restart handling, sample
   scheduling, progress reporting, and output save decisions.
5. ``State`` owns the mutable numerical state and cached, run-specific
   operator compositions.
6. ``SimulationData`` owns persisted settings, input time series, and output
   time series.

Keep this layering intact: configuration should not perform numerical work,
schema should not read run data, input projection should not evolve the model,
and visualization should not mutate simulation state.

Configuration and schema
------------------------

``pynamit.simulation.config`` is the canonical home for settings and default
normalization.  New simulation settings should be added to
``SimulationConfig`` first, then serialized through ``to_attrs``/
``to_dataset``.  Avoid adding loose settings directly to ``Dynamics`` once a
setting needs persistence or restart compatibility.

``pynamit.simulation.schema`` is the canonical home for basis and field-space
selection.  New persisted input or output streams should be declared through
the schema tables and built into ``FieldSpace`` objects there.  This keeps
storage names, field types, mean-free choices, and projection bases visible in
one place.

Spatial bases
-------------

Spherical harmonics and cubed-sphere support have different implementation
needs, but they should present the same public basis contract wherever
possible.  ``CSBasis`` is the public cubed-sphere basis facade.  Its
implementation is split into:

* ``CSCoordinateSystem`` for panel and coordinate transforms.
* ``CSGridGeometry`` and ``CSGridRemapper`` for grid shape, indexing, and
  remapping.
* ``CSFiniteDifferences`` for derivative stencils and sparse operators.
* ``CSVectorTransforms`` for vector-basis conversions.

Prefer adding focused CS behavior to one of these collaborators instead of
growing ``CSBasis`` again.  Keep the ``CS`` abbreviation in class names.

Input preparation and projection
--------------------------------

Input projection is intentionally separated from ``Dynamics`` in
``pynamit.simulation.inputs``.  ``InputSpec`` declares the variables, field
type, mutual-exclusion group, and projection-control restrictions for each
input stream.  ``InputProjector`` owns:

* sample-vs-coefficient validation;
* gridded scalar and tangential projection;
* coefficient row and time-row validation;
* storage of projected input rows; and
* mutual exclusivity between ``u``, ``Q_eff``, and ``E_source``.

Public setters such as ``set_jr``, ``set_resistance``, ``set_neutral_wind``,
``set_Q_eff``, and ``set_E_source`` should remain thin API methods.  When a
new input stream is added, prefer extending the schema and ``InputSpec`` table
over hand-writing a new projection path inside ``Dynamics``.

Prepared external inputs
------------------------

``pynamit.simulation.prepared_inputs`` and ``pynamit.simulation.mage_workflow``
contain reusable, script-independent helpers for MAGE/GAMERA forcing.  Scripts
under ``scripts/simulation`` should be workflow entry points: they may parse
paths and settings, but they should delegate coordinate conversion, metadata
validation, cadence summaries, source-current construction, and projection
directory naming to package modules.

This boundary matters because prepared-input logic is both user-facing and
testable.  If a script needs behavior that should remain correct over time,
move it into the package and test the behavior there.

Electrodynamic physics
----------------------

The physics modules follow the direction of the model equations instead of
mirroring the names of state variables:

* ``magnetic_boundary`` maps magnetic boundary potentials and prescribed
  boundary ``Br`` to the derived horizontal sheet current ``JS``.  Solid
  harmonics own generic radial continuation; this module owns the particular
  potential jump, shielding, and boundary-current relations used by PynaMIT.
* ``ionospheric_closure`` applies the height-integrated Ohm-law closure.  It
  converts Hall/Pedersen conductance to the stored resistance variables and
  maps neutral motion or sheet current through the magnetic geometry and
  resistance tensor to ``E``.
* ``induction`` owns Faraday evolution of ``m_ind``, including Euler,
  exponential, and SciPy integration and the corresponding steady state.

``Geometry`` supplies run-specific grids, magnetic-field factors, transforms,
radial field-line mapping, and interhemispheric constraint geometry to these
equations.  ``State`` owns mutable input coefficients and caches the composed
operators whose values depend on the current conductance.  ``StateOperators``
exposes named compositions of those maps for inspection and solver use.  This
keeps equation implementations out of the orchestration objects without
introducing stateful physics collaborators.

State and evolution
-------------------

``Dynamics`` exposes ``evolve_to_time`` as the stable public API, but delegates
the run loop to ``EvolutionRunner``.  ``EvolutionRunner`` owns restart
short-circuiting, sample scheduling, progress reporting, and output save
decisions.  The runner may decide when to update state or save outputs, but
time-stepping equations belong in ``induction``, run-specific operator caches
belong in ``State``, and artifact details belong in ``SimulationData``.

Persistence
-----------

``SimulationData`` and the time-series primitives are the persistence boundary.
Numerical modules should pass normalized coefficient rows and times to the
time-series APIs instead of writing xarray artifacts directly.  This makes
restart behavior, netCDF/zarr differences, and schema compatibility easier to
maintain.

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

Avoid adding simulation-specific calculations directly to GUI callbacks.  If a
plot needs computed fields, expose them through saved-run, run-field, or
figure-builder helpers so command-line scripts, notebooks, tests, and the GUI
all use the same path.

Extension rules
---------------

Use these rules when extending the codebase:

* Add settings to ``SimulationConfig`` before threading ad hoc constructor
  values through the system.
* Add persisted streams to ``simulation.schema`` before adding storage code.
* Add input projection behavior to ``InputProjector`` and ``InputSpec`` before
  adding logic to ``Dynamics`` setters.
* Put reusable boundary-current equations in ``magnetic_boundary``,
  constitutive ``JS``/wind-to-``E`` equations in ``ionospheric_closure``, and
  magnetic time-stepping equations in ``induction``.
* Add run-loop behavior to ``EvolutionRunner`` before expanding
  ``Dynamics.evolve_to_time``.
* Add cubed-sphere internals to the focused CS collaborator that owns the
  concept.
* Move reusable script logic into package modules before testing it.
* Keep visualization configuration serializable and renderer-agnostic.
* Prefer functionality tests that assert scientific or user-facing behavior
  over tests that encode a refactor's private implementation shape.
