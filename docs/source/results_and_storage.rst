Results, storage, and visualization
===================================

A live simulation and a saved run use the same ``SimulationResults`` and
scientific evaluators. Plotting owns display grids and units, not another
physical interpretation of the coefficients.

Field evaluation
----------------

For example::

    from kompe import SphericalGrid
    from pynamit import SimulationResults
    from pynamit.results import evaluate_simulation_output

    results = SimulationResults.from_directory("simulation")
    grid = SphericalGrid(lat=[[30.0], [60.0]], lon=[[0.0, 90.0, 180.0]])
    fields = evaluate_simulation_output(
        results, 10.0, grid=grid, field_names={"induced_Br", "Phi"}
    )

Each returned field has ``grid.shape`` and remains on the numerical backend.
Passing a non-empty 1-D array of times adds a trailing time axis. Input
evaluation supports the same time blocks; ground-field evaluation always
retains its time axis, including for a single query.
``Phi`` and ``W`` are evaluated in volts, not their stored radius-normalized
coefficient units. Result evaluators read only the coefficients needed for the
requested fields. For repeated queries, retain a single physical evaluator::

    from pynamit.results import OutputEvaluation

    evaluation = OutputEvaluation(results.geometry, grid)
    fields = evaluate_simulation_output(
        results, [0.0, 10.0, 20.0], evaluation=evaluation
    )

``OutputEvaluation`` owns lazily constructed physical maps and the shared
basis transforms on that grid. Evaluated values are not cached, so live edits
remain visible. Plotting reuses this object too, not parallel operator bags.

``evaluate_simulation_output`` owns causal input/output selection for scripts
and ``PlotData``. An explicit Joule-heating request requires conductance at the
requested time. The default selection omits Joule heating when conductance is
unavailable; plots represent unavailable fields as NaN. An output stream that
begins later is never sampled from the future to fill an earlier frame.
``evaluation.evaluate(coefficients, ...)`` accepts coefficient arrays directly,
including trailing batch axes. Quicklook uses it when no history lookup is needed.
Both paths share the same physical equations. A time block requires all requested
output samples to be available; an explicit Joule request also requires conductance
throughout the block. Default selection omits Joule heating otherwise. Optional
boundary Br contributes zero before its first sample and its actual value afterward.

``scripts/tools/benchmark_result_evaluation.py`` compares repeated single-time
calls with one block, using the same warm maps and synthetic coefficients.
A local float64 CPU comparison (SH degree 6, 600 points, 32 times, all fields
including Joule heating) measured about 12.7 ms versus 3.9 ms with NumPy and
120.6 ms versus 4.5 ms with JAX. Single-time throughput stayed comparable to
the pre-change implementation. These are warm evaluation measurements, not
integration, I/O, first-compilation, or GPU speedups.

Simulation time is seconds after ``t0`` throughout numerical evaluation.
``PlotData.input_plot_data_at_time(time)`` accepts those seconds too. Indexed
plots use the saved numeric time directly, without a datetime round trip.
``timestamp_at_index`` and ``time_index`` supply datetimes for labels and
observation matching. Thus datetime rounding cannot move an input change to
the following plot frame.

Ground magnetic signals use ``evaluate_ground_magnetic_field``. This continues
induced Br inward, not the background field. Geographic components use
``(theta, phi) = (south, east)``; radial arrays retain ``grid.shape + (time,)``
and tangential arrays prepend a two-component axis. See :doc:`api` for arguments.

``results.time_series`` owns datetime decoding, observation resampling, and
physical time units. ``time_derivative`` subtracts an integer-nanosecond origin
before converting to relative seconds and calls ``kompe.math.centered_derivative``.
Kompe owns the irregular three-point quadratic stencil and batched NumPy/JAX
array calculation. Complete stencils differentiate at the central sample, not
the midpoint of its neighbours; unavailable edge/missing stencils remain NaN.
``results.peaks`` owns event selection, and ``results.magnetic_signals`` owns
station/model alignment and nT or nT/s observational signals.

Persistence
-----------

``InputPreparation`` owns input histories and input-package persistence.
``SimulationResults`` combines that same input object with output histories
and restart artifacts. It does not own a second input catalog. Input
preparation never constructs output histories or a results object.
``ArtifactStore`` and ``FieldTimeSeries`` implement the common file and
coefficient-series boundaries.
``SimulationSchema`` declares the physical coefficient spaces and creates
input and output series with their units and UTC time origin. Live simulations,
prepared-input copies, and saved-result readers use these same constructors;
each resulting series owns its datasets independently. The storage layer
handles coefficient layouts and persistence, without knowing the MIT quantities.
Numerical modules should pass normalized coefficient rows and times to the
time-series APIs instead of writing xarray artifacts directly.  This makes
restart behavior, netCDF/zarr differences, and schema compatibility easier to
maintain. Time values are finite scalar coordinates. Near-equal floating
checkpoint labels match within four float64 rounding units so
roundoff cannot create duplicate logical times. This is not a fixed physical
time window.

``FieldTimeSeries.add_entries(key, data, times)`` stores a complete batch
with time on the leading axis. This boundary moves time behind the coefficient
axes for Kompe's mean-free normalization, then restores time-first rows for
storage; analysis, normalization, and synthesis otherwise share Kompe's
scientific-axes-first layout. The saved dataset layout is unchanged.
Input preparation uses this boundary to
reuse coefficient metadata and transfer each variable to the CPU once,
while retaining the coefficient space's gauge projection. ``add_entry``
uses the same path for one checkpoint. Overlapping times within a batch
retain insertion-order replacement semantics; ordinary batches merge once.

Evolution retains each requested sample's state, forcing, and conductance
response until the next write. It evaluates induced electric potentials and
boundary current together for samples sharing a response, then appends one
batch per output stream. ``samples_per_write`` controls this existing boundary;
it does not change output times or integration steps. At an input change,
the magnetic state stays continuous while algebraic fields use the new inputs.
Completed pending samples are also evaluated and retained on interruption.
Batch evaluation reduces repeated operator dispatch, but first-use JAX
compilation can cost more. Measure first and warm runs separately with
``scripts/tools/benchmark_evolution.py``; this is not a promise of faster
one-shot execution or GPU performance.

Monotonic appends copy normalized rows directly into coefficient arrays whose
capacity doubles when needed, without an intermediate incoming xarray dataset
or concatenating the full coefficient history. Only the resulting xarray view
is constructed. The exposed xarray
datasets contain views of occupied rows only. Appending does not expand an
earlier view's time axis; use ``dataset.copy(deep=True)`` when an independent
snapshot is needed. In-place edits and replaced variables remain authoritative
when recording resumes. Out-of-order insertions and overlapping replacements
use the sorted merge path and reset the append buffers. Storage formats and
checkpoint selection are unchanged; Zarr still writes only pending rows.
``get_entry(key, times)`` reads unique selected rows together and returns
coefficient-first arrays with time trailing. Scalar queries retain the direct
one-row/two-row slice used by evolution. Interpolation retains stored floating
precision; output queries do not interpolate unless explicitly requested.
Time indexes remain ordinary xarray/pandas indexes and are rebuilt on append;
the large coefficient histories no longer require repeated full copies.
Direct dataset edits require ``save(..., incremental=False)`` to rewrite them,
as before.

An array-first append comparison using ``benchmark_evolution.py`` measured the
following warm continuation times on an arm64 macOS CPU (Python 3.12, JAX x64,
single-threaded Accelerate). Each continuation used 1,000 Euler steps,
``dt=0.0002``, 100 output samples, 10 samples per write, degree/order 5, and
``Ncs=12``; figures are medians of 15 continuations after the first run.
Only the append implementation was switched; output sample counts and final
magnetic-state norms agreed exactly within each backend.

.. list-table:: Warm evolution including in-memory output, milliseconds
   :header-rows: 1

   * - Backend / horizontal basis
     - Dataset-first append
     - Array-first append
   * - NumPy / SH
     - 33.7
     - 21.8
   * - NumPy / CS
     - 83.2
     - 66.5
   * - JAX CPU / SH
     - 44.4
     - 31.7
   * - JAX CPU / CS
     - 73.1
     - 60.1

These measure output overhead for a small model, not a general solver or GPU
speedup. First-use JAX compilation and disk I/O are separate costs.

Coefficient time series have two intentional representations. In memory, each
coefficient dimension carries its schema ``MultiIndex``. A group with one
coefficient space keeps the compact ``i`` dimension; a mixed group, such as CS
output with SH ``induced_Br`` and CS surface quantities, uses distinct
``sh_i`` and ``cs_i`` dimensions. On disk, those indexes are reset to explicit
coordinate columns so NetCDF and Zarr persist the same portable structure.
Loading validates every column and reconstructs the in-memory indexes before
exposing the series.

The time tolerance is a time-coordinate policy measured in seconds. It is
shared with evolution checkpoint decisions; coefficient changes use exact
equality. ``FieldTimeSeries.get_entry()`` is stateless; change history belongs
to the consumer. ``simulation.inputs`` always exposes the current input series,
including after loading a different prepared package.

Time lookup uses binary search over the sorted coordinates validated at
insertion and loading. One bracket serves all variables in the stream;
selection returns owned arrays with each variable's ``CoefficientSpace.shape``:
scalar ``(N,)`` or Helmholtz ``(2, N)``. Flat on-disk storage is decoded here,
not repeatedly inside numerical consumers. Validation and gauge projection
happen at insertion, so readers return ordinary arrays without re-wrapping
or re-normalizing a trusted stored row. ``variables=(...)`` selects a subset;
``variables=()`` checks availability without reading coefficient values.
A known stream without a sample yet returns ``None``; an unknown stream key
is an error. Runtime input selection uses this same missing-data contract.
It retains the owned selection as its comparison snapshot without copying
those CPU arrays again.

An ``ArtifactStore`` instance is bound to one resolved artifact directory and one preferred
storage policy. Create another instance for another simulation instead of retargeting
an existing handle. Time-series storage owns artifact append/rewrite decisions;
projection policy and spatial weights do not belong in persisted coefficient
containers.

One logical artifact has one physical storage representation. A successful
format change removes the alternate NetCDF/Zarr path, and ambiguous legacy
duplicates fail loudly instead of silently preferring stale data. Complete
NetCDF and Zarr rewrites use unique sibling temporary paths before replacement;
incremental Zarr appends remain the time-series layer's explicit fast path.

The simulation schema owns the complete vocabulary of simulation artifact names.
``ArtifactStore`` remains generic: directory validation requires an explicit collection
of artifact names, and the persistence primitive contains no knowledge of
settings, gap responses, or output-stream identities. Artifact names are single
path-safe components.

When adding persisted data, decide first whether it is configuration, input,
output, or derived visualization metadata.  Configuration belongs in
``SimulationConfig``; input and output coefficient time series belong in the
simulation schema; visualization defaults belong in serializable figure settings.

Visualization
-------------

Visualization code should treat saved simulations as read-only inputs.  The
``FigureSettings`` and panel binding layer make figure configuration
serializable, reusable, and testable outside the GUI.  Keep option validation
close to the settings, rendering close to figure builders, and widget binding close
to panel-specific modules.

Each Panel browser session owns one GUI, layout, and rendered-settings snapshot.
Changing the directory invalidates loaded state; control edits remain drafts until
rendered. Figure and script exports use the same snapshot, not partially edited
controls. Startup selection is explicit (argument, environment preference, or
artifacts in the current directory), never an arbitrary child of a result tree.
Generic figure defaults are event independent. Case-specific plotting choices
belong in the case script and the simulation's ``pynamit_plot_defaults.json``;
they do not belong in GUI constructors or directory-name heuristics.

Long preparation, simulation, and movie jobs run through ``gui.workflow_runner``
in isolated Python processes. This is a process/serialization boundary, not a
second workflow implementation: it calls the ordinary preparation, simulation,
and plotting functions. Backend and JAX precision settings are preserved without
sharing mutable runtime state across sessions. The GUI retains only the active
process and a bounded log; stopping or session expiry releases that process.
Scientific validation remains in the underlying workflows, apart from GUI-specific
checks such as preventing preparation from overwriting a populated directory.

``save_movie`` renders exactly the inclusive ``FigureSettings.time_range``:
``(3, 3)`` writes one frame, not an automatically expanded interval. Out-of-range
indices and invalid FPS/DPI overrides are errors. Color scaling likewise keeps
small finite values regardless of their units; positive-field scales reject
negative data rather than silently clipping them. Only constant fields need an
artificial display interval.

Saved simulation loading has one persistence path. ``SimulationResults`` uses the same
``ArtifactStore`` abstraction as simulation persistence and constructs the canonical
configuration, schema, and lightweight numerical geometry. The plotting-grid
``PlotData.from_results`` accepts either live or saved results rather than
reimplementing artifact discovery. Figure renderers directly retain their
serializable settings and grid-evaluation operators, then own only
their respective figure families. Evaluated ground fields are not cached:
coefficient edits and added samples must be visible even when their numerical
operators are reused. ``PlotData.from_directory`` simply opens results and
uses the same construction path.
Coefficient artifacts are validated by the schema-aware ``FieldTimeSeries`` loader;
plotting asks that schema for the exact stored variable name instead of guessing from
prefixes or suffixes. Numeric saved times are interpreted relative to the simulation's
physical ``t0`` and are never silently displayed as Unix-epoch times.

The saved-view cache has one versioned slot per resolved simulation and plotting
resolution. Artifact changes replace the previous heavyweight view in that
slot, preventing live simulations from accumulating obsolete transform and
geometry objects. Its fingerprint descends into directory-backed artifacts so
an incremental Zarr chunk append invalidates the view even when the store's
top-level directory timestamp does not change.

Specialized saved views should retain canonical context, not reconstruct or
re-own it. ``SimulationResults`` owns the output archive and combines it with
``InputPreparation``, the input-series owner. Configuration, schema, and geometry
are shared with that preparation; geometry owns the bases and main field.
``PlotData`` owns plotting grids, spherical transforms, and evaluation-operator
caches. Expensive physical maps are built only when a field calculation needs
them, and the sheet-current maps are
specifically deferred until Joule heating is requested. Input-driver and
ordinary scalar-field figures do not pay that cost merely because output
artifacts also exist. Renderers consume the shared objects instead of
rebuilding main fields, schemas, and transforms from raw settings. An
``equilibrium`` artifact is valid output even when no ``dynamic`` artifact is
present; only difference plots require both.

Saved coefficient datasets have one owner: their ``FieldTimeSeries``.
``SimulationResults.datasets`` returns a catalog of the currently loaded
datasets, not a second store. ``load_input_series("conductance")`` and
``load_output_series("dynamic")`` load only the requested streams; omitting
keys explicitly loads all streams of that kind. Field lookup also loads only
the stream it needs.

Saved simulation behavior enters through ``SimulationResults`` or ``PlotData``
and the ``FigureSettings`` renderer path, rather than maintaining a second
loading or rendering stack.

Avoid adding simulation-specific calculations directly to GUI callbacks.  If a
plot needs computed fields, expose them through saved-results, grid-field, or
figure-builder helpers so command-line scripts, notebooks, tests, and the GUI
all use the same path. Derived physical quantities should call the equation
kernels that define them. In particular, both saved-results frontends use the
closure's total-sheet-current Joule-heating definition, including prescribed
magnetic-boundary current, rather than reconstructing a second approximation
inside visualization.
