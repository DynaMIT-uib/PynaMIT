FROM ubuntu:22.04

# Install system dependencies
RUN apt update
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt install -y wget git cargo gfortran-12 gcc-12 libhdf5-dev pkg-config ttf-mscorefonts-installer
ENV FC=gfortran-12
ENV CC=gcc-12
ENV CXX=g++-12

# Install Mamba through Miniforge
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3.sh -b -p "/miniforge3"
ENV PATH="${PATH}:/miniforge3/bin"
RUN mamba update -y -n base -c conda-forge mamba

# Create PynaMIT environment
# Python 3.11 is used to avoid a bug in pip encountered during apexpy installation with Python 3.12
RUN mamba create -y -n pynamit-env pip numpy scipy pandas matplotlib cartopy pytest build python=3.11

# Make RUN commands use the new environment
SHELL ["mamba", "run", "-n", "pynamit-env", "/bin/bash", "-c"]

# Install the Lompe dependency "apexpy" after removing -lquadmath flag (incompatible with aarch64)
RUN git clone https://github.com/aburrell/apexpy.git
WORKDIR /apexpy
RUN sed -i '46d' meson.build
RUN python -m build .
RUN pip install .
WORKDIR /

# Install Lompe
RUN pip install "lompe[deps-from-github,extras] @ git+https://github.com/klaundal/lompe.git@main"

# Install PynaMIT
ADD ./ /PynaMIT
RUN pip install -e /PynaMIT

ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "pynamit-env"]