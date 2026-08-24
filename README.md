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
simulation.set_conductance(
    pedersen=2 * np.ones(grid.size),
    hall=np.ones(grid.size),
    lat=grid.lat,
    lon=grid.lon,
)
simulation.set_boundary_jr(
    np.zeros(grid.size), lat=grid.lat, lon=grid.lon
)
simulation.evolve_to_time(0.01, quiet=True)
```

The live xarray datasets are available directly as `simulation.inputs` and
`simulation.outputs`; for example, `simulation.outputs["dynamic"]`. The more
specialized `simulation.geometry`, `simulation.response`, and
`simulation.data` objects remain available when their lower-level
operators or persistence details are needed.

Inputs that will be reused by several simulations can instead be prepared in their
own directory. `InputPreparation` has the same `set_*` methods as
`Simulation`, but ordinary projection constructs neither the time-evolution
state nor the full simulation response geometry:

```python
preparation = pynamit.InputPreparation(
    input_directory="prepared_inputs", Nmax=4, Mmax=4, Ncs=8
)
grid = preparation.model_grid
preparation.set_conductance(
    pedersen=2 * np.ones(grid.size),
    hall=np.ones(grid.size),
    lat=grid.lat,
    lon=grid.lon,
)
preparation.write_manifest(source="example")
```

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

Completed simulations can be inspected without rebuilding a live simulation:

```python
results = pynamit.SimulationResults.from_directory("simulation")
results.inputs
results.outputs
results.times
```

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

`PynamEye` is deprecated and lives explicitly at
`pynamit.plotting.legacy.PynamEye`.

## Testing

After installation, run the test suite from the repository root:

```bash
pytest
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
