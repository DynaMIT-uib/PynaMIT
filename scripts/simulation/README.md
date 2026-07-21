# Simulation workflows

The scripts in this directory are editable entry points. Reusable simulation
and input-processing behavior belongs under `pynamit.simulation.workflows`;
paths, experiment names, and machine-specific settings remain here.

## MAGE/GAMERA/TIEGCM

The active MAGE workflow has three stages:

1. Run `mage_prepare.py` once to read GAMERA, REMIX, and TIEGCM and write
   resolution-independent forcing under `mage_prepared/`. The prepared file
   contains only projection inputs and atomically replaces an older complete
   file after successful preparation. TIEGCM conductance and weighted winds
   remain on their native geographic grid. REMIX FAC and GAMERA fields
   originate in timestamped SM coordinates and are remapped onto fixed GEO
   grids for each history. GAMERA `delta_Br` is first evaluated at the true
   three-dimensional cell centers, and its fit uses solid-angle weights from
   the actual GAMERA cell vertices. The fixed prepared grids let all spectral
   projection matrices be built once and reused for every forcing time.
   ReMIX's parallel-positive FAC is converted explicitly to a common
   upward-positive convention using Kaiju's northern and southern grid
   orientations, then to PynaMIT's outward radial current with the local
   main-field direction cosine before projection.
   GAMERA `MagM0` is also converted from GAMERA's length-scale reference
   radius to PynaMIT's dipole reference radius, so the analytic main field
   agrees in physical SI coordinates. Both reference radii are recorded in
   the prepared forcing metadata.
2. Run `mage_project.py` once for each desired spectral resolution or set of
   projection choices. Each projected package is reusable, and a failed
   reprojection leaves the previous complete package in place.
3. Run `mage_run.py` for each simulation experiment. Change `run_name` to keep
   alternatives such as integrators, shielding choices, and regularization in
   separate directories. A run directory may be resumed or extended only with
   the same projected-input manifest, input selection, and evolution policy.
   By default, the inductive state starts from the initial steady state and a
   steady-state comparison is saved throughout the run. Optional shielding of
   the evolving induced field at the GAMERA boundary is disabled by default.
   With no explicit `final_time`, the run stops at the last projected input
   time rather than extrapolating beyond the prepared forcing.

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

The projected `E_source` stream is the direct wind-driven electric-field term,
not the total model electric field. PynaMIT adds boundary, induced, and imposed
responses when it solves the electrodynamic closure. Prescribed `delta_Br` at
the GAMERA boundary is always continued consistently to the ionosphere;
`magnetic_boundary_shielding` is a separate optional image response applied
only to the evolving `m_ind` field.

Prepared `delta_Br` is GAMERA total field minus its split numerical `B0`
background. This isolates the evolved perturbation without importing the
finite-volume background representation error as a physical external field.

PynaMIT state is not evolved in SM or MAG. The MAGE/Kaiju run uses fixed
geocentric geographic coordinates, matching the IGRF/paper workflow.
`MainField` transforms into MAG internally for analytic centered-dipole
physics and returns results in GEO. The main field is frozen at the event
epoch. Time dependence of solar-local-time MAGE patterns enters through the
SM-to-GEO preparation transform and plot centering, not by rotating the
simulation state or background field.

Global maps are evaluated and drawn in GEO. Their timestamp-dependent map
projection places geographic noon/12 solar LT at the center while the data
and coastlines retain ordinary GEO coordinates. Hemisphere maps use the same
model samples but express their positions in MAG (or apex coordinates for an
IGRF main field) and center magnetic noon/12 MLT at the top.

Older, monolithic scripts are retained under `legacy/` for reference only.
