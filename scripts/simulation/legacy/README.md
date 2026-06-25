Legacy simulation scripts
=========================

These scripts predate the prepared-input/run split.  They are kept for
reference, regression checks, or one-off reproduction work, but they are
not the active workflow.

Use the modern scripts one directory up for new work:

- `mage_prepare_forcing.py`
- `mage_project_inputs.py`
- `mage_forcing_final.py`
- `pynamit_paper_simulation.py`

The active pattern is:

1. Prepare/project inputs into a reusable input package.
2. Run PynaMIT from that package with run-only settings such as the main
   field model, PFAC handling, hemisphere coupling, shielding, and
   integrator.
