# Simulation workflows

The scripts in this directory are editable entry points. Reusable simulation
and input-processing behavior belongs under `pynamit.simulation.workflows`;
paths, experiment names, and machine-specific settings remain here.

## MAGE/GAMERA/TIEGCM

The active MAGE workflow has three stages:

1. Run `mage_prepare.py` once to read GAMERA, REMIX, and TIEGCM and write
   resolution-independent forcing under `mage_prepared/`. The prepared file
   contains only projection inputs and atomically replaces an older complete
   file after successful preparation.
2. Run `mage_project.py` once for each desired spectral resolution or set of
   projection choices. Each projected package is reusable, and a failed
   reprojection leaves the previous complete package in place.
3. Run `mage_run.py` for each simulation experiment. Change `run_name` to keep
   alternatives such as integrators, shielding choices, and regularization in
   separate directories. A run directory may be resumed or extended only with
   the same projected-input manifest, input selection, and evolution policy.

The default layout is:

```text
mage_prepared/
  mage_prepared_forcing.h5
mage_cases/<case>/
  projections/N<nmax>_M<mmax>_Ncs<ncs>/
  runs/N<nmax>_M<mmax>_Ncs<ncs>/<run_name>/
```

Set `projection_directory` or `run_directory` explicitly when resolution alone
does not distinguish an experiment. Preparation settings belong in
`mage_prepare.py`, coefficient-fitting settings in `mage_project.py`, and time
evolution settings in `mage_run.py`.

Older, monolithic scripts are retained under `legacy/` for reference only.
