FROM ubuntu:22.04

# Install system dependencies
RUN apt update
RUN apt install -y wget git cargo gfortran-12 gcc-12 libhdf5-dev pkg-config
ENV FC=gfortran-12
ENV CC=gcc-12
ENV CXX=g++-12

# Install Mamba through Miniforge
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3.sh -b -p "/miniforge3"
ENV PATH="${PATH}:/miniforge3/bin"

# Create PynaMIT environment
RUN mamba create -y -n pynamit-env pip numpy scipy pandas matplotlib cartopy pytest

# Make RUN commands use the new environment
SHELL ["mamba", "run", "-n", "pynamit-env", "/bin/bash", "-c"]

# Install the Lompe dependency apexpy after removing -lquadmath flag (incompatible with aarch64)
RUN git clone https://github.com/aburrell/apexpy.git
RUN sed -i '46d' /apexpy/meson.build
RUN pip install -e /apexpy

# Install Lompe
RUN pip install "lompe[deps-from-github,extras] @ git+https://github.com/klaundal/lompe.git@main"

# Install PynaMIT
ADD ./ /PynaMIT
RUN pip install -e /PynaMIT

ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "pynamit-env", "pytest", "/PynaMIT/tests/2d_dipole.py"]