# PynaMIT

PynaMIT is a Python package for simulating global inductive
magnetosphere-ionosphere-thermosphere (MIT) coupling on a two-dimensional
ionospheric shell. It supports time-dependent and steady-state simulations
with configurable conductance, neutral wind, field-aligned current, main-field
geometry, numerical basis, array backend, and output storage.

The package is developed as part of the
[DynaMIT project](https://dynamit-uib.github.io/). The archived software record
is available on [Zenodo](https://zenodo.org/records/17421994)
([DOI: 10.5281/zenodo.17421994](https://doi.org/10.5281/zenodo.17421994)).

## Installation

The dependency lists are kept in `requirements/`:

- `requirements/conda-common.txt`: common conda-forge dependencies for running,
  testing, plotting, and documenting PynaMIT.
- `requirements/pip-common.txt`: Python packages that are installed from PyPI
  or directly from GitHub.
- `requirements/pip-input-models.txt`: optional Lompe and PyAMPS models used
  to generate simulation inputs.

From the repository root, one way to install PynaMIT in a new conda
environment is:

```bash
conda create -n pynamit -c conda-forge --file requirements/conda-common.txt jax
conda activate pynamit

pip install -r requirements/pip-common.txt
pip install "kompe @ git+https://github.com/DynaMIT-uib/kompe.git@main"
pip install -r requirements/pip-input-models.txt
pip install --no-deps \
  "pyhwm2014 @ git+https://github.com/rilma/pyHWM14.git@main"
pip install -e .
```

The native input models are optional; omit the two commands immediately before
`pip install -e .` when the bundled fallback inputs are sufficient. pyHWM2014
currently declares a nonexistent `timeutil` dependency, so it must be installed
without dependencies; NumPy is already part of PynaMIT's core environment.
The current pyHWM2014 main branch requires Python 3.12 or newer.

`pip install -e .` performs an editable install. The environment imports the
package from this source tree, so local source-code changes are picked up
without reinstalling. Use `pip install .` instead for a regular non-editable
install.

The same pattern works with `mamba`:

```bash
mamba create -n pynamit -c conda-forge --file requirements/conda-common.txt jax
```

The `jax` and `zarr` packages are optional runtime dependencies. The development
requirements include `zarr` for persistence tests; add `jax` explicitly when
creating the environment to enable the JAX backend.
For pip installations, use `pip install 'pynamit[zarr]'` to enable Zarr
persistence. This requires Zarr 3.2 and Python 3.12 or newer for strict
missing-chunk detection. Core and GUI workflows can instead use NetCDF on
older supported Python versions.
For NumPy-equivalent 64-bit precision with JAX, set `JAX_ENABLE_X64=1` before
importing JAX; backend selection does not change JAX's process-wide precision
policy.

Select the array backend once near the top of a script, before constructing
the simulation:

```python
import pynamit

pynamit.set_backend("jax")  # or "numpy"
```

Array mathematics and reusable operators then stay on that backend. SciPy-only
algorithms and xarray persistence remain explicit CPU boundaries.
`Simulation` and `InputPreparation` do not take a `backend` keyword or change
this session setting. For a temporary selection, use
`with kompe.math.backend_context("numpy"):` around both construction and work.
This is a session context, not a per-object backend; explicit JAX operands
still select JAX for compatible numerical operations.

JAX accelerator support depends on the operating system, drivers, and hardware.
The generic `jax` package in the environment command is suitable for a standard
environment, but GPU/TPU-specific installations may require replacing it
according to the
[official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).

The core dependencies declared in `pyproject.toml` are sufficient to import
and run the simulation API. Install `pynamit[plot]` for plotting,
`pynamit[gui]` for the interactive GUI and its storage backends,
and `pynamit[mage]` for MAGE HDF5 preparation. The requirements files above
remain the reproducible development-environment definition.

PynaMIT's spherical and numerical machinery is provided by
[`kompe`](https://github.com/DynaMIT-uib/kompe), which is installed as an
ordinary package dependency. Kompe is deliberately not baked into PynaMIT's
test-container images; CI installs the source revision it is testing so that
changes in the two repositories can be checked together.

## Interactive use

The main workflow is deliberately usable as a short IPython session. Create a
simulation, inspect its model grid, add inputs, and evolve it:

```python
import numpy as np
import pynamit

simulation = pynamit.Simulation(Nmax=4, Mmax=4, Ncs=8)
grid = simulation.model_grid

# Replace these uniform arrays with measured or modelled grid samples.
simulation.inputs.set_conductance(
    pedersen=2 * np.ones(grid.size), hall=np.ones(grid.size), grid=grid
)
simulation.inputs.set_boundary_jr(np.zeros(grid.size), grid=grid)
simulation.evolve_to_time(0.01, quiet=True)
```

`simulation.inputs` is an `InputPreparation`: it provides the setters and
maps input names to live xarray datasets, e.g. `simulation.inputs["conductance"]`.
Output datasets are available as `simulation.outputs["dynamic"]`. The more
specialized `simulation.geometry`, `simulation.response`, and
`simulation.results` objects remain available when their lower-level
operators or persistence details are needed.

Without a directory, preparation and evolution stay in memory. Call
`simulation.save("simulation")` to save settings, inputs, and outputs and enable
automatic later writes. Passing `simulation_directory` at construction enables
persistence from the start.

Equilibria and recorded states are separate from evolution:

```python
fields = simulation.equilibrium_coefficients(time=0.01)  # no state change
simulation.set_state(fields["induced_Br"], time=0.01)
simulation.record_state()  # no time integration
simulation.sample_equilibria([0.0, 0.005, 0.01])  # independent diagnostics
```

Inputs can also be prepared independently and reused in several experiments.
Ordinary projection constructs neither time-evolution state nor expensive
magnetic-response operators:

```python
preparation = pynamit.InputPreparation(Nmax=4, Mmax=4, Ncs=8)
grid = preparation.model_grid
preparation.set_conductance(pedersen=2 * np.ones(grid.size), hall=np.ones(grid.size), grid=grid)
simulation = pynamit.Simulation.from_inputs(preparation, enable_pfac_coupling=True)
simulation.evolve_to_time(0.01, quiet=True)

# Optional reusable package for scripts and the GUI:
preparation.save("prepared_inputs")
preparation.write_manifest(source="example")
```

`from_inputs` takes an independent snapshot and inherits the preparation's
configuration, with explicit overrides for simulation settings. Later input
edits in either object do not change the other. Coefficient spaces, coordinates,
and time origin must remain compatible.

Choose `least_squares_solver` and `least_squares_tolerance` once during
preparation (or on `SimulationConfig`). Input projection, wind-current fitting,
and response fits share those settings; individual input setters do not have
their own tolerances. Defaults are SH `normal_pinv`, CS `lsmr`, and `1e-15`.
The numerical-policy section of `docs/source/architecture.rst` documents
the tolerance's meaning for each algorithm and distinguishes data fits from
fixed geometric inverse maps.

Compatible experiments reuse their bases, transforms, and geometric operators;
their input and output histories remain independent. For direct composition with
Kompe objects, use the existing geometry role:

```python
from kompe import SHBasis, GlobalCSBasis

geometry = pynamit.SimulationGeometry.from_bases(
    SHBasis(4, 4), GlobalCSBasis(8), enable_pfac_coupling=False
)
preparation = pynamit.InputPreparation.from_geometry(geometry, t0="2001-05-12")
# Set inputs as above, then make independently editable experiments:
first = pynamit.Simulation.from_inputs(preparation)
second = pynamit.Simulation.from_inputs(preparation, integrator="exponential")
assert first.geometry is second.geometry
```

`from_bases` derives resolution settings from the objects. Pass the same CS basis
as `horizontal_basis` to select CS surface operators; poloidal magnetic fields
still use SH. Geometry retains spatial choices only. Time origin and numerical
fit/evolution policy belong to the preparation or simulation; they are not
inherited from an earlier experiment that happened to build the geometry.
A physics change rebuilds affected geometry while retaining compatible bases.
Configure a persistent operator cache on the supplied SH basis, rather than
replacing it on each experiment. Alternative bases can be explored in memory;
saving requires coefficient spaces that the standard SH/CS file recipe can
reconstruct, and rejects unsupported spaces before writing any artifacts.

Prepared inputs expose `preparation.input_series`, `preparation.schema`, and
`preparation.geometry` directly. They do not construct simulation results.
`simulation.results.inputs is simulation.inputs`: the result view combines
that input preparation with recorded outputs. `preparation.save(...)` saves
inputs only; `simulation.save(...)` saves or relocates the complete experiment.

The name is deliberately `InputPreparation`, rather than `InputProjection`:
the object can project sampled fields, but it can also accept coefficients
that are already projected and package either form for later simulations. The
corresponding convenience functions share one workflow namespace:

```python
from pynamit.workflows import prepare_example_inputs, run_example, run_from_inputs
```

`prepare_example_inputs` prepares one explicitly specified event through the
configured empirical input providers, `run_example` prepares and runs such an
event in one call, and `run_from_inputs` runs any compatible prepared package.
The event time, Kp, solar-wind and IMF values, tilt, F10.7, and Ap are ordinary
function arguments rather than hidden PynaMIT defaults. The regression suite
keeps its shared 12 May 2001 case in `tests/example_scenario.py`.

Live and saved histories have the same result interface:

```python
results = simulation.results
# Or, without constructing time evolution:
# results = pynamit.SimulationResults.from_directory("simulation")
results.inputs
results.outputs
results.times

from pynamit.results import evaluate_simulation_output
from pynamit.plotting import PlotData

fields = evaluate_simulation_output(results, 0.01)
view = PlotData.from_results(results)
```

The live view sees additional samples and coefficient edits without reloading.
Result evaluation accepts `SimulationResults`, not a separate live-simulation
branch. The former `simulation.data` owner is replaced by `simulation.results`.

Result evaluation, plotting, the GUI, and specialized MAGE processing have
separate namespaces:

```python
from pynamit.results import evaluate_projected_input, evaluate_simulation_output
from pynamit.plotting import FigureSettings, render_figure
from pynamit.gui import build_gui
from pynamit.workflows.mage import ForcingSettings, prepare_forcing, prepare_inputs
```

Both evaluation functions return ordinary dictionaries of arrays, so they are
convenient in IPython without introducing a separate projection/result object.
For example, ``evaluate_simulation_output(results, 10.0)`` evaluates the saved
physical fields at 10 seconds on the model grid.

## Interactive GUI

Install `pynamit[gui]`, then open a saved simulation or projected-input package:

```bash
pynamit-gui /path/to/simulation
```

Without a path, the GUI uses `PYNAMIT_SIMULATION_DIR`, or the current directory
if it contains PynaMIT settings. Otherwise it starts empty. It does not search
through result trees and silently choose one simulation. The same preference
works with `build_gui()` and `scripts/visualization/pynamit_panel.py`.

For a MAGE case, select the specific resolution and named simulation, for example:

```bash
export PYNAMIT_SIMULATION_DIR=/path/to/mage_output/2011-10-24/resolutions/N20_M20_Ncs20/simulations/default
pynamit-gui
```

The GUI's **Load** action reads data and optional `pynamit_plot_defaults.json`
from that directory. Ordinary plots start with model-only global maps and no
time shift or event marker. The MAGE run script keeps its event-specific choices
in `PLOT_DEFAULTS`, writing them beside each simulation only if no defaults file
exists. Edit that case configuration for new simulations, or use **Save plot
defaults** in the GUI to persist the current controls for an existing simulation.
Scripts can do the same with `FigureSettings(...).save_defaults(directory)`.
Relative station-data paths in that file are relative to the simulation directory.

Controls are drafts until **Redraw**. Figure, Python-script, and settings exports
describe the displayed figure, even if controls have subsequently changed. Saving
plot defaults or a movie instead uses the current controls. Existing output files
require overwrite confirmation.

The other GUI modes prepare empirical example inputs and run or resume simulations
from prepared packages. Event time and physical driver values are editable under
**Empirical model inputs**; native-provider selection follows `PYNAMIT_INPUT_SOURCE`.
Preparation requires a new or empty destination, preserving existing packages.
Simulation output cadence is an explicit interval in seconds, independent of solver
steps. Euler has a fixed internal `dt`; SciPy integrators adapt using `rtol` and
`atol`, and exponential evolution needs no `dt`. Preparation, simulation, and movie
export run in separate processes, with a live log and a stop
button. Stopping retains completed artifacts; inspect interrupted output before
resuming. The server also stops a job when its owning browser session expires.

The GUI is a trusted local tool: it can read and write files and start computations
as the server user. Keep the default loopback address or use a secured SSH tunnel;
do not expose it as an unauthenticated public service.

## Testing

Install the test tools and optional GUI/plotting/storage dependencies, then
run the test suite from the repository root:

```bash
python -m pip install -e ".[test,gui,zarr]"
python -m pytest
```

With no selection flags, pytest runs the complete suite with fallback inputs on
each available backend. Tests marked `native_input_validation` additionally run
with live input models when they are installed. Explicit `--data-source` options
apply the requested sources to the complete suite.

CI gives the two environments separate jobs: the ordinary environment proves
that no native input models are installed, while the native-input environment
checks model outputs against the fallback dataset through both NumPy and JAX
projections. The corresponding local commands are:

```bash
pytest --backend numpy --backend jax --data-source fallback
pytest -m native_input_validation \
  --backend numpy --backend jax --data-source native
```

For a focused check, pass only the backend and input source of interest.
See [CONTRIBUTING.md](https://github.com/DynaMIT-uib/PynaMIT/blob/main/CONTRIBUTING.md) for the two-repository development
setup, test ownership, formatting, and documentation checks.

## Examples and Paper Scripts

The `scripts/` directory contains example scripts and the simulation and
visualization scripts used in preparation of Laundal et al. (Ann. Geophys.,
43, 803-833, 2025). These scripts are useful references, but some are research
scripts rather than maintained examples: they may use deprecated features or
older abstractions that are not representative of the current public API.

## Citation

Publications using PynaMIT should cite:

```bibtex
@Article{angeo-43-803-2025,
AUTHOR = {Laundal, K. M. and Skeidsvoll, A. S. and Popescu Braileanu, B. and Hatch, S. M. and Olsen, N. and Vanham\"aki, H.},
TITLE = {Global inductive magnetosphere-ionosphere-thermosphere coupling},
JOURNAL = {Annales Geophysicae},
VOLUME = {43},
YEAR = {2025},
NUMBER = {2},
PAGES = {803--833},
URL = {https://angeo.copernicus.org/articles/43/803/2025/},
DOI = {10.5194/angeo-43-803-2025}
}
```

## License

PynaMIT is distributed under the MIT License. See `LICENSE` for details.
