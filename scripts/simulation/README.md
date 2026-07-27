# Simulation workflows

The scripts in this directory are editable entry points. Reusable simulation
and input-processing behavior belongs under `pynamit.simulation.workflows`;
paths, experiment names, and machine-specific settings remain here.

## MAGE/GAMERA/TIEGCM

The configured case is the 24 October 2011 sudden commencement simulated for
Shi et al. (2022), *Geospace Concussion: Global Reversal of Ionospheric
Vertical Plasma Drift in Response to a Sudden Commencement*,
https://doi.org/10.1029/2022GL100014.

The active MAGE workflow has three stages:

1. Run `mage_prepare.py` once to read GAMERA, REMIX, and TIEGCM and write
   resolution-independent `forcing.h5` in the event directory. The prepared
   file contains only projection inputs and atomically replaces an older
   complete file after successful preparation. TIEGCM conductance and weighted
   winds remain on their native geographic grid. REMIX FAC and GAMERA fields
   originate in timestamped SM coordinates and are remapped onto fixed GEO
   grids for each history. GAMERA `delta_Br` is first evaluated at the true
   three-dimensional cell centers, and its fit uses solid-angle weights from
   the actual GAMERA cell vertices. The fixed prepared grids let all spectral
   projection matrices be built once and reused for every forcing time.
   ReMIX FAC follows Kaiju's native periodic grid interpolation, including its
   polar-cell rule. GAMERA `delta_Br` uses four-point bilinear interpolation on
   GAMERA's own periodic native angular grid; its omitted cell-center polar
   values are reconstructed from the adjacent ring means.
   The prepared `time` dataset is the validated, uniform TIEGCM `mtime`
   schedule. GAMERA's adaptive stepper can write a nominal 10-second output
   a few hundredths of a second after that target; preparation retains those
   exact GAMERA and ReMIX timestamps and their signed offsets as provenance,
   requires both sources to remain within 0.1 seconds of the nominal history,
   and performs each SM-to-GEO transform at the GAMERA time rounded to the
   nearest whole second, exactly following Kaiju's `mjdRECALC`. This
   lets PynaMIT apply each record on its intended fixed-step clock without
   interpolating conductance or changing the dense-exponential evolution.
   Sheet conductance is a radial geometric-height integral of TIEGCM's
   geographic `SIGMA_PED` and `SIGMA_HAL` profiles. Below the first saved
   interface, preparation reproduces TIEGCM's `pdynamo` continuation to its
   -8.5 log-pressure interface (90 km): Pedersen and Hall conductivity use
   5 km and 3 km exponential scale lengths, respectively, and the lowest
   saved winds are held constant. The extension uses the source grid's own
   vertical spacing (six added layers for this high-resolution file).
   Pedersen- and Hall-weighted winds include those same radial layer
   integrals. This is the thin-sheet reduction derived in Appendix A of
   Laundal et al. (2025): the separately weighted winds compactly represent
   `integral(sigma_P u dr)` and `integral(sigma_H u dr)`. Projection converts
   these wind-current moments to the equivalent `E_neutral_wind` field
   without explicitly constructing `Q_eff`; the two formulations agree away
   from the dip equator, while the E formulation remains regular at the
   equator.

   Preparation also reproduces ReMIX's mandatory hard minimum:
   2 S Pedersen and 1 S Hall inside the saved ReMIX SM polar domain. The
   domain is transformed onto the fixed GEO grid at every GAMERA source time;
   equatorward conductance remains the unfloored TIEGCM radial integral.
   These values are part of this MAGE case rather than an experiment option,
   and the prepared schema records both floors and the source grid's
   equatorward boundary. The added background conductance inherits the
   corresponding conductivity-weighted neutral wind, equivalent to assuming
   that it scales the associated TIEGCM vertical conductivity profile.

   The case XML sets ReMIX `doStarlight=T`, but the coupled `doGCM=T` branch
   replaces ReMIX's internally calculated EUV conductance with the GCM input
   and then applies these hard minima. Consequently the faithful treatment
   here is the hard 2/1 S background, not an additional quadratic
   `sqrt(Sigma**2 + floor**2)` correction.

   The MAGE-specific
   `gzigm1`/`gzigm2` diagnostics are intentionally not a second input path:
   TIEGCM forms them as Pedersen/Hall conductivity integrals along modified-
   apex field lines for its equipotential-field-line dynamo, then regrids the
   results to geographic coordinates. ReMIX used those effective field-line
   quantities, whereas PynaMIT's spherical thin-sheet law requires the local
   radial-column conductance derived in Appendix A. The resulting difference
   from the archived ReMIX potential is therefore expected rather than an
   interpolation error; substituting the saved ReMIX conductance is useful as
   a diagnostic, but would change the constitutive model.
   ReMIX's parallel-positive FAC is converted explicitly to a common
   upward-positive convention using Kaiju's northern and southern grid
   orientations, then to PynaMIT's outward radial current with the local
   main-field direction cosine before projection.
   GAMERA `MagM0` is also converted from GAMERA's length-scale reference
   radius to PynaMIT's dipole reference radius, so the analytic main field
   agrees in physical SI coordinates. Both reference radii are recorded in
   the prepared forcing metadata.
2. Run `mage_project.py` to project every resolution listed in its
   `resolutions` setting. The default `(20, 40, 60, 80)` sweep supports a
   direct convergence comparison. `projection_name` keeps alternative fitting
   choices separate. Each projected package is reusable, and a failed
   reprojection leaves the previous complete package in place.
3. Run `mage_run.py` to evolve every resolution in its `resolutions` setting.
   Select its `projection_name`, then change `run_name` to keep alternatives
   such as integrators, shielding choices, and regularization in separate
   directories. The full sweep is validated before its first simulation
   starts. A matching completed run is skipped before simulation geometry is
   constructed; an interrupted run resumes from its last persisted `m_ind`
   state. A run directory may be resumed or extended only with the same
   projected-input manifest, input selection, run settings, and evolution
   policy. By default, the inductive state starts from the initial steady state
   and a steady-state comparison is saved throughout the run. Optional
   shielding of the evolving induced field at the GAMERA boundary is disabled
   by default. With no explicit `final_time`, the run stops at the last
   projected input time rather than extrapolating beyond the prepared forcing.

The default layout is:

```text
mage_output/2011-10-24/
  forcing.h5
  resolutions/
    N<nmax>_M<mmax>_Ncs<ncs>/
      operator_cache/
      projections/<projection_name>/
      runs/<run_name>/
```

The event is the natural ownership boundary: its resolution-independent
prepared forcing sits above the resolution-specific numerical operators,
projected inputs, and simulation runs. Projection and run names are independent
namespaces. Every run records and owns a copy of the exact projected inputs it
consumes, so dependency validation does not rely on directory nesting. Set
`output_path` or `resolutions_directory` explicitly when the default layout is
unsuitable. Preparation settings belong in `mage_prepare.py`,
coefficient-fitting settings in `mage_project.py`, and time evolution settings
in `mage_run.py`.

Run directories own copies of their projected inputs, saved state and
steady-state histories, settings and provenance manifests. They also persist
the time-independent PFAC coupling matrix, which is restored directly on a
restart. Healthy input copies are reused instead of being recopied from the
projection package.

Projection and run stages share the `operator_cache/` belonging to their
resolution. Materialized SH evaluation matrices, projection normal
pseudo-inverses, the fixed model-grid Helmholtz Cholesky factor, and the final
PFAC coupling matrix are stored there as read-only NumPy arrays and
memory-mapped on reuse. Their identities include implementation-version tags,
the full basis and field-space signatures, exact grid coordinates and
weights, regularization, solver tolerance where applicable, main-field
geometry, and PFAC integration radii. Thus a normalization, coordinate,
weight, regularization, or geometry change cannot silently reuse the wrong
array. PFAC's one-use radial-quadrature grids are deliberately excluded; only
their final integrated matrix is persisted. The run still owns a PFAC copy so
restart correctness does not depend on the optional shared cache.
The cache is only an optimization and can be deleted without affecting a
run's physical identity or restartability. At high resolution these exact
float64 arrays can occupy several GiB, trading disk space for avoiding
repeated basis evaluation and dense factorization. Set
`cache_operators=False` to disable the trade.
Native CS identity, interpolation, and finite-difference operators remain
structured or sparse, so the cache does not turn them into enormous dense
arrays merely to persist them. SH Helmholtz and PFAC analysis likewise
compose their two derivative operators directly; they do not assemble an
additional four-dimensional synthesis tensor just to obtain a factor.

The active Pedersen and Hall resistance coefficients also receive an exact
content fingerprint. In-memory exponential-propagator reuse requires that
fingerprint and the integration step to match; it does not rely merely on
object identity. Resistance-dependent closure matrices, steady-state inverses,
and propagators are not persisted by default. At N=80 one dense square matrix
is already hundreds of MiB, and a distinct conductance normally occurs at
every forcing time, so storing each dynamic matrix would turn one history into
hundreds of GiB without helping its forward continuation. The fixed geometric
and basis constructions, together with the existing PFAC sidecar, are the
high-value persistent cache boundary.

The projected `E_neutral_wind` stream is the equivalent electric-field
contribution derived from Pedersen- and Hall-weighted neutral winds, not the
total model electric field. PynaMIT adds boundary, induced, and imposed
responses when it solves the electrodynamic closure. Prescribed `delta_Br` at
the GAMERA boundary is always continued consistently to the ionosphere;
`magnetic_boundary_shielding` is a separate optional image response applied
only to the evolving `m_ind` field.

Prepared `delta_Br` is GAMERA total field minus its split numerical `B0`
background. This isolates the evolved perturbation without importing the
finite-volume background representation error as a physical external field.
Both fields are saved cell-volume averages; `BxD`/`ByD`/`BzD` are point
samples at the volume barycentres and therefore are not the matching
background subtraction. The single fitted boundary radius is the
solid-angle-weighted mean of those barycentre radii.

PynaMIT state is not evolved in SM or MAG. The MAGE/Kaiju run uses fixed
geocentric geographic coordinates, matching the IGRF/paper workflow.
`MainField` transforms into MAG internally for analytic centered-dipole
physics and returns results in GEO. The main field is frozen at the event
epoch. Time dependence of solar-local-time MAGE patterns enters through the
SM-to-GEO preparation transform and plot centering, not by rotating the
simulation state or background field.

Global maps are evaluated and drawn in GEO. Their timestamp-dependent map
projection places 12 mean-solar LT at the center while the data and coastlines
retain ordinary GEO coordinates. Hemisphere maps use the same
model samples but express their positions in MAG (or apex coordinates for an
IGRF main field) and center magnetic noon/12 MLT at the top.

Older, monolithic scripts are retained under `legacy/` for reference only.
