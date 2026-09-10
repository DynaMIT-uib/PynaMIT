Architecture
============

The objective is a short, inspectable researcher workflow backed by reusable
scientific objects. Modularity should follow independent equations and state
lifetimes, not line counts. Numerical kernels may be sophisticated; the
physics-facing functions should show how those kernels compose into the model.

Public workflow
---------------

Import the primary workflow from one canonical namespace::

    from pynamit import InputPreparation, Simulation, SimulationResults
    from kompe import SphericalGrid

    preparation = InputPreparation(Nmax=20, Mmax=20)
    # preparation.set_conductance(...), preparation.set_boundary_jr(...)

    simulation = Simulation.from_inputs(preparation)
    simulation.evolve_to_time(10.0, output_interval=0.1)
    results = simulation.results

A simulation owns an input preparation rather than inheriting from it.
``InputPreparation`` can be used without creating an induction solver.
``SimulationResults`` owns histories and scientific evaluation context, not
the evolving response. Omitting a directory keeps the workflow in memory;
``save(directory)`` enables persistence. Live and saved results have the
same interface.

``pynamit.simulation`` is an implementation namespace, not a second set of
workflow re-exports. This avoids loading the high-level simulation while result
evaluation imports lower-level schema, geometry, or equation modules.
Advanced functions are imported from their defining modules.
``pynamit.results``, ``pynamit.plotting``, ``pynamit.gui``, and
``pynamit.workflows`` provide their explicit scientific/workflow APIs.
Spherical and general numerical objects are imported directly from Kompe.

Ownership
---------

.. list-table::
   :header-rows: 1
   :widths: 24 45 31

   * - Object or module
     - Owns
     - Does not own
   * - ``SimulationConfig``
     - Immutable physical and numerical choices; normalization
     - Operators, datasets, or solver state
   * - ``SimulationSchema``
     - Field spaces, units, and storage metadata
     - Numerical basis construction or file access
   * - ``SimulationGeometry``
     - Bases, main field, and fixed MIT-specific spatial maps
     - Time-dependent forcing, experiment controls, or full saved settings
   * - ``InputPreparation``
     - Physical input conversion, projection, and input histories
     - A live clock or inductive state
   * - ``ElectrodynamicResponse``
     - Fixed-conductance closure, constraint solve, and reusable maps
     - Forcing histories or evolving magnetic state
   * - ``Simulation``
     - Live time and induced Br; explicit initial conditions and restoration
     - State implicitly reconstructed from editable output rows
   * - ``_TimeEvolution``
     - Input intervals, reusable forcing response, sampling, and writes
     - Generic integrator algorithms
   * - ``SimulationResults``
     - Prepared inputs together with output histories and restart artifacts
     - Live dynamic state or a second response model
   * - ``FieldTimeSeries`` / ``ArtifactStore``
     - Coefficient layout/time selection; named-file I/O
     - MIT equations or input projection policy
   * - Result evaluators
     - Coefficients to physical fields; units and causal time selection
     - Figure configuration
   * - ``PlotData`` / renderers / GUI
     - Display grids and operator caches; figures; session interaction
     - Alternate physics or mutable simulation state

The core equation sequence remains visible: magnetic boundary relations
produce sheet current, the ionospheric closure produces electric field under
the current constraints, and Faraday's law evolves induced radial magnetic
field. Their functions live together by equation in
``simulation.electrodynamics``, while ``geometry.py`` and
``response.py`` own reusable state. See :doc:`numerical_model`.

Package and dependency boundaries
---------------------------------

The maintained packages have distinct scientific scopes::

    kompe/
      spherical geometry, SH/CS/SECS bases, grids and meshes
      analysis/synthesis, differential and solid-harmonic operators
      LinearMap, least squares, affine evolution, array backend, caches

    pynamit/
      geomagnetism/       MainField models and magnetic coordinates
      simulation/         configuration, geometry, inputs, response, evolution
        electrodynamics/  magnetic boundary, ionospheric closure, Faraday law
      storage/            coefficient histories and named artifacts
      results/            live/saved histories and physical field evaluation
      external_inputs/    empirical provider conventions and fallback data
      workflows/          experiment preparation and execution
        mage/             external forcing and coefficient projection
      plotting/           scientific figures and display-grid evaluation
      gui/                optional session and process boundaries

Kompe primitives must have complete mathematical meaning without PynaMIT's
physical model. MIT closure, induction coordinates, input contracts, and result
interpretation remain in PynaMIT even when reused. General irregular-grid
differentiation and affine linear evolution belong in Kompe; datetime policy
and the construction of the Faraday equation do not.

Imports follow these responsibilities. Workflows may construct simulations.
Simulations coordinate geometry, response, inputs, results, and persistence.
Geometry and response compose the equation modules; equation modules do not
import orchestration or persistence. ``geomagnetism`` does not know
``SimulationConfig``; ``storage`` does not know MIT artifact names.
Result evaluation may use equation modules, without constructing a live
simulation. The GUI calls the same workflows as scripts.

Composition and lifetime
------------------------

``SimulationGeometry.from_bases`` accepts actual Kompe objects.
``InputPreparation.from_geometry`` and ``Simulation.from_geometry``
reuse that geometry. ``Simulation.from_inputs`` shares compatible geometry,
including materialized operators, while copying coefficient histories into an
independent trajectory. Physical changes create fresh geometry caches;
run-policy changes do not discard compatible spatial operators.

Geometry retains only spatial construction settings. Time origin, integrator,
and solver policy are supplied to the consuming preparation or simulation;
``SimulationConfig.from_geometry`` combines them with the spatial defaults.
In-memory equations use the actual bases. File-format restrictions are checked
by ``SimulationConfig.validate_geometry_for_storage`` before any writes, not
while defining the mathematical objects. The standard saved recipe describes
Schmidt-normalized SH truncations and global CS bases; unsupported spaces fail
explicitly at saving rather than reopening with different mathematics.
The horizontal basis must still provide the operations required by MIT,
including a Laplacian within its coefficient space; an arbitrary selection of
CS nodes need not define such a discretization.

``InputPreparation`` owns its coefficient series directly. ``SimulationResults``
references that preparation and owns the output series, so inputs exist without
any results object. Saved-result readers initialize the projector lazily only
if input fitting is actually requested. Input-only export leaves the trajectory
directory unchanged; ``simulation.save`` or ``simulation.results.save`` relocates
the complete experiment and rebinds subsequent input and output writes together.

Array-backend choice is an explicit script/session policy, not an option on
individual simulations. Workflow construction does not mutate that policy.
Use ``set_backend`` once, or a ``backend_context`` around both construction
and computation; numerical primitives continue to recognize JAX operands.

``_InputProjector`` is a meaningful private collaborator: it owns reusable
projection problems and batched sample/coefficient conversion. It is not
another object researchers must construct. Provider/file interpretation remains
outside it. See :doc:`input_workflows`.

A response is fixed for one conductance distribution. Changed conductance
creates a new response without mutating earlier diagnostic objects. Only
conductance-independent solver state is reused across that change. Winds,
boundary forcing, and induced Br remain explicit calculation arguments.
This is why merging geometry and response would make ownership less clear.

``MainField``, bases, transforms, and responses cache their own numerical
quantities. Shared cache primitives implement eviction or persistence, while
the scientific owner supplies the key. There is no generic evaluation-object
hierarchy or duplicate cache of sampled result fields. See
:doc:`results_and_storage` for live edits and lazy reads.

``LinearMap`` retains domain/codomain shapes and dense, sparse, diagonal, or
matrix-free structure. Physics code composes maps and applies their scientific
axes; flat matrices appear at explicit solver/materialization boundaries.
Diagonal maps remain vectors. A materialized map is reused rather than ignored.
The chosen least-squares algorithm and tolerance remain explicit, not inferred
from a series of hidden numerical heuristics.

Configuration and extension
---------------------------

``SimulationConfig`` normalizes physical-domain invariants, bases, solver
policies, and persisted settings once, before constructing numerical state.
Use ``from_config`` for a normalized configuration or
``dataclasses.replace`` for a deliberate validated change. Constructor
keywords remain explicit for IPython inspection; replacing them with nested
configuration bags or generic keyword dispatch would hide the choices.
Dependent defaults must be supplied explicitly or reset to ``None`` when
the parent choice changes.

``simulation.schema`` declares each persisted stream's variables, units,
coefficient spaces, and gauge. It describes supplied numerical geometry;
it does not create a parallel collection of bases. Stored rows return in their
canonical scientific shape, so downstream code need not guess layouts or
reapply storage validation.

When extending the code:

* Put the equation where its physical or mathematical meaning is complete.
  Keep signs, units, component order, and coupling terms visible there.
* Add a class only for meaningful state, an independent workflow, or a necessary
  boundary. Reused equations can remain functions.
* Add input/output contracts to the schema, not another storage path.
* Keep generic numerical kernels in Kompe and provider/I/O/plotting transfers
  at explicit CPU boundaries. Test numerical paths on both NumPy and JAX.
* Test identities and manufactured solutions as well as researcher workflows.
  Assert current scientific contracts rather than removed names.
* Require measured benefit before adding a new solver strategy or dense cache.

No additional package split is currently needed. Plotting and the GUI already
have optional dependency boundaries, and another distribution would not remove
their need for PynaMIT result semantics. Likewise, the current grid / mesh /
basis / transform distinction and geometry / response distinction each capture
different mathematical contracts or state lifetimes. Reducing their class count
would combine responsibilities rather than remove machinery.

Performance tradeoffs to measure
--------------------------------

Two deliberate numerical boundaries remain important at larger resolutions.
Native CS fixed inverse maps use reusable SciPy sparse factors, including an
explicit CPU boundary for JAX callers. Dense poloidal exponential evolution
has cubic setup cost and quadratic map storage. GPU-native sparse solves or
exponential-action methods could help particular workloads, but they must
preserve the fixed operator, gauge, tolerance, and input-change contracts.
They are not interchangeable implementation details to switch heuristically.
Profile representative resolutions and changing-conductance histories before
replacing these paths; warm constant-input timing alone is insufficient.
