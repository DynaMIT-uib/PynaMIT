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
- `requirements/pip-data.txt`: optional data/model dependencies used by some
  example and paper-preparation scripts.

From the repository root, one way to install PynaMIT in a new conda
environment is:

```bash
conda create -n pynamit -c conda-forge --file requirements/conda-common.txt jax zarr
conda activate pynamit

pip install -r requirements/pip-common.txt
pip install -r requirements/pip-data.txt

pip install -e .
```

`pip install -e .` performs an editable install. The environment imports the
package from this source tree, so local source-code changes are picked up
without reinstalling. Use `pip install .` instead for a regular non-editable
install.

The same pattern works with `mamba`:

```bash
mamba create -n pynamit -c conda-forge --file requirements/conda-common.txt jax zarr
```

The `jax` and `zarr` packages are optional. `jax` enables the optional JAX
backend, while `zarr` enables efficient xarray-based reads and writes.

JAX accelerator support depends on the operating system, drivers, and hardware.
The generic dependency in `requirements/conda-common.txt` is suitable for a
standard environment, but GPU/TPU-specific installations may require replacing
or updating the JAX packages according to the
[official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).

The core dependencies declared in `pyproject.toml` are sufficient to import
and run the simulation API. Install `pynamit[gui]` for the Panel and plotting
stack, and `pynamit[inputs]` for the optional native input models. The
requirements files above remain the reproducible development-environment
definition.

PynaMIT's spherical and numerical machinery is provided by
[`kompe`](https://github.com/DynaMIT-uib/kompe). Until the first Kompe release
is available from PyPI, `requirements/pip-common.txt` installs it directly
from that repository; install the requirements file before invoking
`pip install .` on a fresh environment.

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
    hall=np.ones(grid.size),
    pedersen=2 * np.ones(grid.size),
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
`simulation.run_data` objects remain available when their lower-level
operators or persistence details are needed.

## Testing

After installation, run the test suite from the repository root:

```bash
pytest
```

Some tests are skipped automatically if optional dependencies or native input
datasets are not available.

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
