# Images contain third-party dependencies only. CI installs Kompe and PynaMIT
# from the source revisions being tested.
FROM condaforge/miniforge3 AS base
WORKDIR /opt/app
COPY requirements/conda-common.txt requirements/pip-common.txt ./
RUN mamba install -y --file conda-common.txt \
  && mamba clean -afy
RUN python -m pip install --no-cache-dir -r pip-common.txt

# JAX is the only image-specific layer. CI installs native input providers
# dynamically when a test run needs them.
FROM base AS jax
RUN mamba install -y jax \
  && mamba clean -afy
