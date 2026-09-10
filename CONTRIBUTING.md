# Developing PynaMIT and Kompe

The scientific and modularity guidelines are in [AGENTS.md](AGENTS.md).
Keep MIT-specific equations and workflows in PynaMIT and general spherical
mathematics in Kompe. Tests follow the same ownership boundary.

## Local environment

Use Python 3.12 or newer for the complete development environment. From the
PynaMIT checkout, with Kompe checked out alongside it:

```bash
python -m pip install -e "../kompe[test]"
python -m pip install -e ".[test,gui,jax,zarr]"
```

Omit `jax` for a NumPy-only environment. The `gui` extra includes plotting,
HDF5, and NetCDF; the `zarr` extra enables strict Zarr persistence. Older Python
versions can use the core and GUI with NetCDF, without `zarr`. Native
empirical input models are deliberately separate. Building ApexPy and some
native input models requires a Fortran compiler.

`pytest.ini` selects this checkout's `src` and, when present, `../kompe/src`.
That sibling checkout therefore takes precedence over an installed Kompe wheel
**during tests**. Editable installations make scripts and IPython use the same
sources. Check both when diagnosing source/version mismatches:

```bash
python -c "import pynamit, kompe; print(pynamit.__file__); print(kompe.__file__)"
```

## Tests

```bash
python -m pytest --backend numpy --backend jax --data-source fallback
```

With no selection flags, the complete suite runs on each available backend
with fallback inputs. Only tests marked `native_input_validation` also run
with installed native providers. Explicit `--data-source` options apply that
selection to the entire suite. Backend selection and isolated runtime state
are implemented in `tests/conftest.py`.

The shared physical scenario belongs to `tests/example_scenario.py`, not production
defaults. Stored regressions remain useful, but numerical changes also need
independent physical identities or manufactured solutions. Use realistic
nonuniform coordinates, component conventions, and rank deficiencies where
they matter. Performance tests should guard meaningful operator reuse or
allocation costs, not incidental helper calls.

Test groups mirror scientific roles: `tests/inputs/`, `tests/mage/`,
`tests/regression/`, and `tests/visualization/`; model-equation, storage, and
simulation tests are directly under `tests/`. Plotting tests distinguish field evaluation,
rendering, figure settings, and exports. Pure Kompe contracts are tested in
Kompe, while PynaMIT tests their composition into MIT workflows.

For focused native-provider equivalence (not another complete simulation run):

```bash
python -m pip install -r requirements/pip-input-models.txt
# Upstream pyHWM2014 declares an unavailable timeutil dependency.
python -m pip install --no-deps "pyhwm2014 @ git+https://github.com/rilma/pyHWM14.git@main"
python -m pytest -m native_input_validation \
  --backend numpy --backend jax --data-source native
```

## Before submitting

```bash
ruff check .
ruff format .
python -m pytest --backend numpy --backend jax --data-source fallback
python -m pip install sphinx sphinx-rtd-theme myst-parser
sphinx-build -W --keep-going -b html docs/source docs/build
```

Run Kompe's tests in its own checkout, with both its default NumPy environment
and `KOMPE_USE_JAX=1 JAX_ENABLE_X64=1`. Its CI additionally checks supported
Python versions, minimum NumPy/SciPy versions, wheel packaging, and a scheduled
PynaMIT consumer run. PynaMIT CI installs current Kompe and the declared test
extras into the JAX image; native providers are installed only in their separate
job. Changes to either source repository do not require rebuilding the image.

For cross-repository API changes, land Kompe first: PynaMIT CI consumes Kompe's
`main`, not unpublished sibling-checkout changes. Before publishing a PynaMIT
release, require a Kompe release that actually supplies its APIs; passing tests
against the sibling checkout does not establish compatibility with the minimum
version in package metadata. Do not add runtime compatibility fallbacks to hide
a mismatched installation.

Update examples and public API documentation together with intentional API
changes. Keep obsolete-name migration notes in documentation rather than
adding tests solely to remember that an old private name no longer exists.
