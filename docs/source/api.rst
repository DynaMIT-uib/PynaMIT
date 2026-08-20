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

Reusable workflows
------------------

.. autofunction:: pynamit.workflows.prepare_example_inputs

.. autofunction:: pynamit.workflows.run_example

.. autofunction:: pynamit.workflows.run_from_inputs

Fields and background geometry
------------------------------

.. autoclass:: pynamit.FieldSpace
   :members:

.. autoclass:: pynamit.FieldCoefficients
   :members:

.. autoclass:: pynamit.MainField
   :members:

.. autoclass:: pynamit.MagneticFieldEvaluation
   :members:

Array backend
-------------

.. autofunction:: pynamit.set_backend

Spherical grids, bases, transforms, operators, and least-squares solvers are
provided by ``kompe``. The :doc:`sphere_operators` page records the shared
surface-field conventions used by PynaMIT.
