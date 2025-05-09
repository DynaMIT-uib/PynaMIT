FROM condaforge/miniforge3

# Install compilers.
RUN apt update && apt install -y gcc gfortran

# Set compiler environment variables.
ENV CC=gcc
ENV CXX=g++
ENV FC=gfortran

# Install required packages, Python 3.10 is used for compatibility with pyHWM14.
RUN mamba install -y \
    cartopy \
    coveralls \
    git \
    hdf5 \
    matplotlib \
    mscorefonts \
    myst-parser \
    make \
    numpy \
    pandas \
    pip \
    pkg-config \
    pytest \
    python=3.10 \
    python-build \
    ruff \
    rust \
    scipy \
    sphinx \
    sphinx-rtd-theme

# Install Lompe.
RUN pip install "lompe[deps-from-github,extras] @ git+https://github.com/klaundal/lompe.git@main"

# Install pyHWM14.
RUN pip install "git+https://github.com/rilma/pyHWM14.git@main"
