"""Simulation."""

import numpy as np
import pynamit
import dipole
import datetime
import matplotlib.pyplot as plt
import h5py as h5

RE = 6381e3
RI = 6.5e6
latitude_boundary = 35
latitude_step = 0.5

PLOT_BR = False
PLOT_CONDUCTANCE = False
PLOT_JR = False

BR_LAMBDA = 0.1
CONDUCTANCE_LAMBDA = 1
JR_LAMBDA = 0.1

dt = 10

filename_prefix = "mage-forcing"
Nmax, Mmax, Ncs = 40, 40, 40
rk = RI / np.cos(np.deg2rad(np.r_[0:70:2])) ** 2

date = datetime.datetime(2011, 10, 24, 18)
d = dipole.Dipole(date.year)

file = h5.File("mage_2011/data.h5", "r")

# Magnetosphere grid is 96x128, while ionosphere grid is 144x288.
# 1.5x density from 96 to 144, 2.25x (1.5^2) density from 128 to 288.
# Does this come from the 1.5x radius quotient?
ionosphere_lat = file["glat"][:]
ionosphere_lon = file["glon"][:]

ionosphere_grid = pynamit.Grid(lat=ionosphere_lat, lon=ionosphere_lon)

magnetosphere_lat = file["Blat"][:]
magnetosphere_lon = file["Blon"][:]

magnetosphere_grid = pynamit.Grid(lat=magnetosphere_lat, lon=magnetosphere_lon)

print("Setting up simulation object")
# Set up simulation object.
dynamics = pynamit.Dynamics(
    filename_prefix=filename_prefix,
    Nmax=Nmax,
    Mmax=Mmax,
    Ncs=Ncs,
    RI=RI,
    RM=1.5 * RI,
    mainfield_kind="dipole",
    FAC_integration_steps=rk,
    ignore_PFAC=True,
    connect_hemispheres=False,
    latitude_boundary=latitude_boundary,
    ih_constraint_scaling=1e-5,
    t0=str(date),
    integrator="exponential",
)

Br_basis_evaluator = pynamit.BasisEvaluator(
    dynamics.state.basis,
    magnetosphere_grid,
    weights=np.sin(np.deg2rad((90 - magnetosphere_lat).flatten())),
    reg_lambda=BR_LAMBDA,
)

if PLOT_BR or PLOT_CONDUCTANCE or PLOT_JR:
    plt_lat, plt_lon = np.linspace(-89.9, 89.9, 60), np.linspace(-180, 180, 100)
    plt_lat, plt_lon = np.meshgrid(plt_lat, plt_lon)
    plt_grid = pynamit.Grid(lat=plt_lat, lon=plt_lon)
    plt_evaluator = pynamit.BasisEvaluator(dynamics.state.basis, plt_grid)
    conductance_plt_evaluator = pynamit.BasisEvaluator(dynamics.state.conductance_basis, plt_grid)

time = file["time"][:]
nstep = time.shape[0]

for step in range(0, nstep):
    print("Processing input step", step, "of", nstep)
    print("Setting Br")

    # Get and set Br input.
    delta_Br = file["Bu"][:][step, :, :]
    # Probably need to remove dipole field from delta_Br?
    # Need to find out typical delta_Br values at 1.5 RI.

    if np.any(np.isnan(delta_Br)):
        raise ValueError("Br input contains NaN values.")

    if PLOT_BR:
        pynamit.globalplot(
            magnetosphere_lon, magnetosphere_lat, delta_Br, cmap=plt.cm.bwr, extend="both"
        )

    Br_expansion = pynamit.FieldExpansion(
        dynamics.state.basis,
        basis_evaluator=Br_basis_evaluator,
        grid_values=delta_Br.flatten(),
        field_type="scalar",
    )

    if PLOT_BR:
        pynamit.globalplot(
            plt_lon,
            plt_lat,
            Br_expansion.to_grid(plt_evaluator).reshape(plt_lon.shape),
            cmap=plt.cm.bwr,
            extend="both",
            title="Br at 1.5 RI",
        )

    dynamics.set_Br(
        Br_expansion.to_grid(dynamics.state.basis_evaluator),
        theta=dynamics.state.grid.theta,
        phi=dynamics.state.grid.phi,
        time=dt * step,
    )

    # Get and set jr input.
    print("Setting jr")
    FAC = file["FAC"][:][step, :, :] * 1e-6  # Convert from muA/m^2 to A/m^2

    FAC_b_evaluator = pynamit.FieldEvaluator(
        dynamics.mainfield, pynamit.Grid(lat=ionosphere_lat, lon=ionosphere_lon), RI
    )

    FAC[np.isnan(FAC)] = 0

    jr = FAC.flatten() * FAC_b_evaluator.br

    dynamics.set_jr(
        jr,
        lat=ionosphere_lat.flatten(),
        lon=ionosphere_lon.flatten(),
        time=dt * step,
        weights=np.sin(np.deg2rad((90 - ionosphere_lat).flatten())),
        reg_lambda=JR_LAMBDA,
    )

    # Get and set conductance input. Should not use _G conductances
    # ("GAMERA"?), as these are NaN at low latitudes.
    print("Setting conductance")
    conductance_hall = file["SH"][:][step, :, :].flatten()
    conductance_pedersen = file["SP"][:][step, :, :].flatten()

    if np.any(np.isnan(conductance_hall)):
        raise ValueError("Hall conductance input contains NaN values.")
    if np.any(np.isnan(conductance_pedersen)):
        raise ValueError("Pedersen conductance input contains NaN values.")
    if np.any(conductance_hall <= 0):
        raise ValueError("Hall conductance input contains non-positive values.")
    if np.any(conductance_pedersen <= 0):
        raise ValueError("Pedersen conductance input contains non-positive values.")

    conductance_pedersen_input = conductance_pedersen.flatten()

    dynamics.set_conductance(
        conductance_hall,
        conductance_pedersen,
        lat=ionosphere_lat.flatten(),
        lon=ionosphere_lon.flatten(),
        time=dt * step,
        weights=np.sin(np.deg2rad((90 - ionosphere_lat).flatten())),
        reg_lambda=CONDUCTANCE_LAMBDA,
    )

    dynamics.set_input_state_variables()

    if PLOT_JR:
        # Note: no minlat, 50 deg default?
        pynamit.globalplot(
            plt_lon,
            plt_lat,
            dynamics.state.jr.to_grid(plt_evaluator).reshape(plt_lon.shape),
            cmap=plt.cm.bwr,
            extend="both",
            title="jr",
        )

    if PLOT_CONDUCTANCE:
        pynamit.globalplot(
            plt_lon,
            plt_lat,
            dynamics.state.etaP.to_grid(conductance_plt_evaluator).reshape(plt_lon.shape),
            cmap=plt.cm.viridis,
            extend="both",
            title="etaP",
        )

        pynamit.globalplot(
            plt_lon,
            plt_lat,
            dynamics.state.etaH.to_grid(conductance_plt_evaluator).reshape(plt_lon.shape),
            cmap=plt.cm.viridis,
            extend="both",
            title="etaH",
        )

print("Imposing steady state")
dynamics.impose_steady_state()

print("Time evolution")
final_time = 3600  # seconds
dynamics.evolve_to_time(final_time, dt=dt, sampling_step_interval=1, saving_sample_interval=1)
