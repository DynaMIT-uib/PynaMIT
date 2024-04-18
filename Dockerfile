FROM ubuntu:22.04

# We need to use a interactive bash shell to source conda.sh and mamba.sh, and to activate mamba environments
SHELL ["/bin/bash", "-i", "-c"]

# Install system dependencies
RUN apt update
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt install -y wget git cargo gfortran-12 gcc-12 libhdf5-dev pkg-config ttf-mscorefonts-installer
ENV FC=gfortran-12
ENV CC=gcc-12
ENV CXX=g++-12

# Install Mamba through Miniforge
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3.sh -b -p "/opt/miniforge3" && \
    echo "source /opt/miniforge3/etc/profile.d/conda.sh" >> ${HOME}/.bashrc && \
    echo "source /opt/miniforge3/etc/profile.d/mamba.sh" >> ${HOME}/.bashrc && \
    echo "mamba activate" >> ${HOME}/.bashrc

# Update Mamba
RUN mamba update -y -c conda-forge mamba

# Create PynaMIT environment
# Python 3.11 is used to avoid a bug in pip encountered during apexpy installation with Python 3.12
RUN mamba create -y -n pynamit-env pip numpy scipy pandas matplotlib cartopy pytest build python=3.11 && \
    echo "mamba activate pynamit-env" >> ${HOME}/.bashrc

# Install the Lompe dependency "apexpy" after removing -lquadmath flag (incompatible with aarch64)
RUN git clone https://github.com/aburrell/apexpy.git
WORKDIR /apexpy
RUN sed -i '46d' meson.build
RUN python -m build .
RUN pip install .
WORKDIR /

# Install Lompe
RUN pip install "lompe[deps-from-github,extras] @ git+https://github.com/klaundal/lompe.git@main"