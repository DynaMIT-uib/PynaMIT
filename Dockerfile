# Variant 1: With JAX
FROM condaforge/miniforge3 AS jax
WORKDIR /opt/app
COPY requirements/conda-common.txt requirements/pip-common.txt ./
RUN mamba install -y \
      --file conda-common.txt \
      jax \
  && mamba clean -afy
RUN pip install -r pip-common.txt

# Variant 2: With data stack
FROM condaforge/miniforge3 AS data
WORKDIR /opt/app
COPY requirements/conda-common.txt requirements/pip-common.txt requirements/pip-data.txt ./
RUN mamba install -y \
      --file conda-common.txt \
  && mamba clean -afy
RUN pip install -r pip-common.txt -r pip-data.txt
