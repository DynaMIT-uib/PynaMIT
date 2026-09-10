Scientific API
--------------

The ordinary workflow starts with :class:`pynamit.Simulation`. Its
``inputs.set_*`` methods accept sampled physical fields or already projected
coefficients, and :meth:`~pynamit.Simulation.evolve_to_time` advances the
inductive solution. :class:`pynamit.InputPreparation` exposes the same input
path without constructing the time-evolution machinery. A simulation owns an
input preparation rather than inheriting from it. ``Simulation.from_inputs``
copies a preparation for an independent experiment. Without a directory, both
objects stay in memory; ``save(directory)`` enables persistence explicitly.
Compatible experiments share numerical geometry, not coefficient histories.
``SimulationGeometry.from_bases`` accepts existing Kompe objects;
``InputPreparation.from_geometry`` and ``Simulation.from_geometry`` reuse them.

``simulation.induced_Br`` and ``simulation.current_time`` are the live dynamic
state. Recorded output is not its backing storage. ``set_state(Br, time=...)``
sets an initial condition; ``restore_state(results, time=...)`` restores an exact
dynamic checkpoint, including into a new independent simulation. To inspect
the live response without recording it, call ``simulation.output_coefficients()``
and pass the returned dictionary to ``OutputEvaluation.evaluate``.

Select the shared array backend at script/session level, before constructing
objects: ``pynamit.set_backend("jax")`` or ``pynamit.set_backend("numpy")``.
Constructors and ``from_*`` methods no longer accept ``backend`` or change the
process setting. For temporary work, use ``kompe.math.backend_context`` around
both construction and calculation. Objects do not own separate execution engines;
explicit JAX operands still select JAX for compatible operations.

Simulation and input preparation
--------------------------------

.. autoclass:: pynamit.InputPreparation
   :members:
   :exclude-members: geometry

.. autoclass:: pynamit.Simulation
   :members:

.. autoclass:: pynamit.SimulationConfig
   :members:

.. autoclass:: pynamit.SimulationGeometry
   :members: from_config, from_bases, with_config

Live and saved results
----------------------

``simulation.results`` and ``SimulationResults.from_directory(...)`` have the
same interface. Pass either to the result evaluators below or to
``pynamit.plotting.PlotData.from_results``. A live plot view reflects new samples
and direct coefficient edits; it caches only geometry-dependent operators.

.. autoclass:: pynamit.SimulationResults
   :members:

.. autofunction:: pynamit.results.evaluate_projected_input

.. autofunction:: pynamit.results.evaluate_simulation_output

.. autoclass:: pynamit.results.OutputEvaluation
   :members:

.. autofunction:: pynamit.results.evaluate_ground_magnetic_field

Reusable workflows
------------------

.. autofunction:: pynamit.workflows.prepare_example_inputs

.. autofunction:: pynamit.workflows.run_example

.. autofunction:: pynamit.workflows.run_from_inputs

Fields and background geometry
------------------------------

Coefficient layouts and values belong to Kompe; PynaMIT supplies the MIT
schemas that associate them with physical input and output quantities.

.. autoclass:: kompe.CoefficientSpace
   :members:

.. autoclass:: pynamit.MainField
   :members:

API migration
-------------

The architecture cleanup uses one canonical spelling, without compatibility
aliases:

* Input setters default to time zero, not the simulation clock. Pass
  ``time=simulation.current_time`` when updating a running experiment.
* Editing or deleting output rows no longer changes continuation. Use
  ``set_state`` or ``restore_state`` deliberately. ``save`` persists recorded
  histories; ``record_state()`` first records an
  edited live state when a new checkpoint is wanted.
* ``equilibrium_coefficients(time)`` calculates equilibria without changing
  state or histories. ``sample_equilibria(times)`` records independent
  diagnostics and replaces ``evolve_to_time(..., run_dynamic=False)``.
  Equilibrium queries default to held inputs; interpolation is explicit.
* ``impose_equilibrium`` is replaced by explicit calculation,
  ``set_state(fields["induced_Br"], time=...)``, and optional ``record_state()``.
  Recording uses the live held inputs, not a previous interpolated diagnostic.
* Kompe's prepared evolution now yields bounded arrays with trailing sample
  axes. Pass ``batch_size`` to control these blocks; it does not alter the
  integrator's accepted trajectory. ``build_induction_stepper`` follows the
  same contract, while ``evolve_induced_Br`` still returns one final state.
* ``FieldTimeSeries.iter_intervals`` replaces ``change_times`` and carries
  selected coefficients alongside boundaries, including endpoint inputs.
  ``ElectrodynamicResponse.from_conductance`` accepts already selected
  conductance instead of selecting timestamps through ``from_inputs``.
* ``Simulation.output_coefficients()`` evaluates the live state and replaces
  ``output_at_current_time``. To inspect recorded output instead, use
  ``simulation.results.output_series.get_entry``.
* ``plot_output_quicklook`` uses live coefficients and returns an editable
  figure without displaying or closing it. Field evaluation stays on the
  configured backend; only the evaluated plot arrays move to NumPy.
* Import workflow classes from ``pynamit``, not ``pynamit.simulation``.
  Implementation modules no longer eagerly re-export workflow classes.
* ``FieldTimeSeries.get_entry`` returns canonical scalar ``(N,)`` or
  Helmholtz ``(2, N)`` arrays rather than always flattening coefficient rows.
  Optional ``variables=(...)`` reads a subset without materializing other fields.
* ``evaluate_simulation_output`` uses ``field_names={...}`` instead of
  ``include_derived``. Returned fields retain the requested grid's broadcast
  shape. An explicit Joule-heating request fails when conductance is unavailable.
* Reuse ``OutputEvaluation(geometry, grid)`` through ``evaluation=`` instead
  of supplying output/current operator dictionaries. Its ``evaluate`` method
  replaces ``evaluate_output_coefficients`` for direct coefficient arrays.
  Physical input/output evaluators accept time arrays with a trailing time axis.
* ``PlotData.input_plot_data_at_time`` accepts model seconds after ``t0``,
  consistently with result evaluators. Datetimes remain display/observation
  coordinates, not an intermediate representation of saved simulation time.
* ``simulation.data`` becomes ``simulation.results``; ``SimulationData`` is
  removed. Input preparation owns ``input_series``, ``schema``, and ``geometry``
  directly; it no longer constructs or exposes ``results``. A live simulation's
  ``simulation.results.inputs`` is its ``simulation.inputs`` object.
* Numerical bases move from ``results.schema`` to ``results.geometry``.
  ``geometry.poloidal_basis`` names the mean-free SH magnetic basis.
* Output evaluators take ``SimulationResults``; pass ``simulation.results``.
  Input evaluation accepts ``InputPreparation`` directly as well as results.
  ``from_directory`` no longer needs a ``build_geometry`` flag: numerical
  geometry exists immediately, and expensive operators build on first use.
* ``Simulation.from_inputs`` shares the preparation's operator cache. Set
  ``operator_cache_directory`` during preparation, or supply a cached SH basis
  through ``SimulationGeometry.from_bases``.
* ``pynamit.FieldSpace`` becomes ``kompe.CoefficientSpace``. Its
  ``representation`` is ``"scalar"`` or ``"helmholtz"``; ``shape`` and
  ``size`` replace ``coefficient_shape`` and ``coefficient_length``.
* ``FieldCoefficients`` is removed. Use ordinary backend arrays and
  ``CoefficientSpace.project_mean_free(values)`` when normalization is needed.
  Stored rows are already normalized. Arrays are interpreted in the receiving
  transform's basis; compare coefficient spaces explicitly when exchanging
  values between unrelated bases.
* Geometry retains spatial settings only, not a complete experiment config.
  Use ``SimulationConfig.from_geometry(geometry, t0=..., integrator=...)`` or
  supply those controls directly to ``Simulation.from_geometry``. Supplied
  nonstandard bases may be used in memory; saving rejects coefficient spaces
  the standard SH/CS file recipe cannot reconstruct.
* Scalar bases expose ``coefficient_count`` instead of ``index_length``.
  Global CS constructors use ``cells_per_edge`` instead of ``cells_per_face``.
* ``LeastSquaresSolver.prepare`` replaces ``build_response_solver``;
  ``pointwise_component_map`` replaces ``pointwise_matrix_linear_map``.
* Set ``least_squares_solver`` and ``least_squares_tolerance`` once on
  ``InputPreparation``, ``Simulation``, or ``SimulationConfig``. The shared
  settings now cover both input and response fits. The ``set_*`` input methods
  no longer take ``tolerance=``. Saved preparations retain these defaults
  when consumed by ``run_from_inputs``; explicit arguments override them.
* Response diagnostics expose one canonical operator interface. Replace
  ``source_to_W_matrices`` and ``source_to_induced_Br_rate_matrices`` with
  their ``*_operators`` methods, then call ``operator.to_matrix()`` only
  when a flat matrix is needed. Likewise, materialize the
  ``induced_poloidal_potential_feedback_operator`` or
  ``noninductive_W_to_equilibrium_induced_poloidal_potential_operator``
  instead of using the former ``*_matrix`` properties. ``to_array()``
  retains the map's scientific axes; both exports accept ``backend=``.
* ``PersistentArrayCache`` is imported from ``kompe.cache``, not
  ``pynamit.storage``.
* Provider ``SampleGrid`` becomes ``SampleCoordinates``. The spherical
  provider approximation is named ``pynamit.geographic_approximation``, not
  ``pynamit.geodesy``.

These Python API changes do not change the saved coefficient artifact format.
Update Kompe and PynaMIT together.

Array backend
-------------

.. autofunction:: pynamit.set_backend

Spherical grids, bases, transforms, operators, and least-squares solvers are
provided by ``kompe``. The :doc:`sphere_operators` page records the shared
surface-field conventions used by PynaMIT.
