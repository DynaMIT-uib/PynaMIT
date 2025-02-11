FROM ubuntu:24.04

# Use login bash shell to ensure activation of Mamba and environments
SHELL ["/bin/bash", "-l", "-c"]

# Install system dependencies
RUN apt update
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt install -y wget git cargo gfortran gcc libhdf5-dev pkg-config ttf-mscorefonts-installer make
ENV FC=gfortran
ENV CC=gcc
ENV CXX=g++

# Install Mamba through Miniforge
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3.sh -b -p "/opt/miniforge3" && \
    echo "source /opt/miniforge3/etc/profile.d/conda.sh" >> /etc/profile.d/activate_env.sh && \
    echo "source /opt/miniforge3/etc/profile.d/mamba.sh" >> /etc/profile.d/activate_env.sh && \
    echo "mamba activate base" >> /etc/profile.d/activate_env.sh

# Update Mamba
RUN mamba update -y -c conda-forge mamba

# Create PynaMIT environment, Python 3.10 is used for compatibility with pyHWM14
RUN mamba create -y -n pynamit-env pip numpy scipy pandas matplotlib cartopy pytest python-build python=3.10 && \
    echo "mamba activate pynamit-env" >> /etc/profile.d/activate_env.sh

# Install Lompe
RUN pip install "lompe[deps-from-github,extras] @ git+https://github.com/klaundal/lompe.git@main"

# Install pyHWM14
RUN git clone https://github.com/rilma/pyHWM14.git
WORKDIR /pyHWM14
RUN make install
WORKDIR /