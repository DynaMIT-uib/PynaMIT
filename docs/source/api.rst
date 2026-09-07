Scientific API
--------------

The ordinary workflow needs one object: :class:`pynamit.Simulation`. Its
``set_*`` methods accept sampled physical fields or already projected
coefficients, and :meth:`~pynamit.Simulation.evolve_to_time` advances the
inductive solution. :class:`pynamit.InputPreparation` exposes the same input
path without constructing the time-evolution machinery.

Simulation and input preparation
--------------------------------

.. autoclass:: pynamit.InputPreparation
   :members:
   :exclude-members: geometry

.. autoclass:: pynamit.Simulation
   :members:

.. autoclass:: pynamit.SimulationConfig
   :members:

Saved results
-------------

.. autoclass:: pynamit.SimulationResults
   :members:

.. autofunction:: pynamit.results.evaluate_projected_input

.. autofunction:: pynamit.results.evaluate_simulation_output

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

.. autoclass:: kompe.FieldCoefficients
   :members:

.. autoclass:: pynamit.MainField
   :members:

API migration
-------------

The architecture cleanup uses one canonical spelling, without compatibility
aliases:

* ``pynamit.FieldSpace`` becomes ``kompe.CoefficientSpace``. Its
  ``representation`` is ``"scalar"`` or ``"helmholtz"``; ``shape`` and
  ``size`` replace ``coefficient_shape`` and ``coefficient_length``.
* ``FieldCoefficients`` is imported from Kompe. Its ``field_space`` still
  identifies the coefficient layout and gauge.
* Scalar bases expose ``coefficient_count`` instead of ``index_length``.
  Global CS constructors use ``cells_per_edge`` instead of ``cells_per_face``.
* ``LeastSquaresSolver.prepare`` replaces ``build_response_solver``;
  ``pointwise_component_map`` replaces ``pointwise_matrix_linear_map``.
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
