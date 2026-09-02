Legacy simulation scripts
=========================

These scripts predate the prepared-input/simulation split.  They are kept for
reference, regression checks, or one-off reproduction work, but they are
not the active workflow.

Use the modern scripts one directory up for new work:

- `mage_prepare.py`
- `mage_project.py`
- `mage_run.py`
- `pynamit_paper_simulation.py`

The active pattern is:

1. Prepare resolution-independent forcing.
2. Project one reusable input package for each desired resolution and main
   field model.
3. Create one or more simulations from a projected package with settings such
   as PFAC handling, hemisphere coupling, shielding, and integrator.
