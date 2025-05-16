"""Simulation."""

import numpy as np
import pynamit
import dipole
import datetime
import matplotlib.pyplot as plt
import h5py as h5
import cartopy.crs as ccrs

RE = 6381e3
RI = 6.5e6
latitude_boundary = 35
latitude_step = 0.5

PLOT_BR = False
PLOT_CONDUCTANCE = False
PLOT_JR = False
PLOT_U = False

BR_LAMBDA = 0.1
CONDUCTANCE_LAMBDA = 1
JR_LAMBDA = 0.1
U_LAMBDA = 0.1


def dipole_radial_sampling(r_min, r_max, n_steps):
    ratio = r_min / r_max
    max_angle = np.rad2deg(np.arccos(np.sqrt(ratio)))
    angles = np.linspace(0, max_angle, n_steps)
    rk = r_min / np.cos(np.deg2rad(angles)) ** 2
    return rk, angles


filename_prefix = "results_mage_2011"
Nmax, Mmax, Ncs = 40, 40, 40
# rk = RI / np.cos(np.deg2rad(np.r_[0:70:2])) ** 2
rk, _ = dipole_radial_sampling(RI, 1.5 * RI, n_steps=40)

noon_lon = 0
dt = 10

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
    ignore_PFAC=False,
    connect_hemispheres=True,
    latitude_boundary=latitude_boundary,
    ih_constraint_scaling=1e-5,
    t0=str(date),
    integrator="exponential",
)

FAC_b_evaluator = pynamit.FieldEvaluator(
    dynamics.mainfield, pynamit.Grid(lat=ionosphere_lat, lon=ionosphere_lon), RI
)

plt_lat, plt_lon = np.linspace(-89.9, 89.9, 60), np.linspace(-180, 180, 100)
plt_lat, plt_lon = np.meshgrid(plt_lat, plt_lon)
plt_grid = pynamit.Grid(lat=plt_lat, lon=plt_lon)
plt_evaluator = pynamit.BasisEvaluator(dynamics.state.basis, plt_grid)
conductance_plt_evaluator = pynamit.BasisEvaluator(dynamics.state.conductance_basis, plt_grid)

time = file["time"][:]
nstep = time.shape[0]
nstep = 1

for step in range(0, nstep):
    print("Processing input step", step + 1, "of", nstep)
    # Get and set Delta Br input (given in nT).
    delta_Br = file["Bu"][:][step, :, :].flatten() * 1e-9  # Convert from nT to T.

    if np.any(np.isnan(delta_Br)):
        raise ValueError("Br input contains NaN values.")

    print("Setting Br with RMS: ", np.sqrt(np.mean(delta_Br**2)))
    dynamics.set_Br(
        delta_Br,
        lat=magnetosphere_lat,
        lon=magnetosphere_lon,
        time=dt * step,
        weights=np.sin(np.deg2rad((90 - magnetosphere_lat).flatten())),
        reg_lambda=BR_LAMBDA,
    )

    # Get and set jr input (FAC given in muA/m^2).
    FAC = file["FAC"][:][step, :, :] * 1e-6  # Convert from muA/m^2 to A/m^2

    if np.any(np.isnan(FAC)):
        print("Warning: FAC input contains NaN values. Setting to 0.")
        FAC[np.isnan(FAC)] = 0

    jr = FAC.flatten() * FAC_b_evaluator.br

    print("Setting jr with RMS: ", np.sqrt(np.mean(jr**2)))
    # dynamics.set_jr(
    #    jr,
    #    lat=ionosphere_lat,
    #    lon=ionosphere_lon,
    #    time=dt * step,
    #    weights=np.sin(np.deg2rad((90 - ionosphere_lat).flatten())),
    #    reg_lambda=JR_LAMBDA,
    # )

    # Get and set conductance input (given in S).
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

    print(
        "Setting conductances with RMSes: ",
        np.sqrt(np.mean(conductance_hall**2)),
        np.sqrt(np.mean(conductance_pedersen**2)),
    )
    dynamics.set_conductance(
        conductance_hall,
        conductance_pedersen,
        lat=ionosphere_lat,
        lon=ionosphere_lon,
        time=dt * step,
        weights=np.sin(np.deg2rad((90 - ionosphere_lat).flatten())),
        reg_lambda=CONDUCTANCE_LAMBDA,
    )

    # Get and set wind input (given in m/s).
    u_east = file["We"][:][step, :, :]
    u_north = file["Wn"][:][step, :, :]

    u_theta, u_phi = (-u_north.flatten(), u_east.flatten())
    u_lat, u_lon = ionosphere_lat, ionosphere_lon

    if np.any(np.isnan(u_theta)):
        raise ValueError("Wind input contains NaN values.")
    if np.any(np.isnan(u_phi)):
        raise ValueError("Wind input contains NaN values.")

    print("Setting wind with RMS: ", np.sqrt(np.mean(u_theta**2 + u_phi**2)))
    # dynamics.set_u(
    #    u_theta=u_theta,
    #    u_phi=u_phi,
    #    lat=u_lat,
    #    lon=u_lon,
    #    time=dt * step,
    #    weights=np.tile(np.sin(np.deg2rad(90 - u_lat.flatten())), (2, 1)),
    #    reg_lambda=U_LAMBDA,
    # )

    print("Setting input state variables")
    dynamics.set_input_state_variables()

    if PLOT_BR:
        pynamit.globalplot(
            plt_lon,
            plt_lat,
            dynamics.state.Br.to_grid(plt_evaluator).reshape(plt_lon.shape),
            cmap=plt.cm.bwr,
            extend="both",
            title="Br at 1.5 RI",
        )

    if PLOT_JR:
        # Note: no minlat, 50 deg default?
        pynamit.globalplot(
            plt_lon,
            plt_lat,
            dynamics.state.jr.to_grid(plt_evaluator).reshape(plt_lon.shape),
            cmap=plt.cm.bwr,
            extend="both",
            title="jr at RI",
        )

    if PLOT_CONDUCTANCE:
        pynamit.globalplot(
            plt_lon,
            plt_lat,
            dynamics.state.etaP.to_grid(conductance_plt_evaluator).reshape(plt_lon.shape),
            cmap=plt.cm.viridis,
            extend="both",
            title="etaP at RI",
        )

        pynamit.globalplot(
            plt_lon,
            plt_lat,
            dynamics.state.etaH.to_grid(conductance_plt_evaluator).reshape(plt_lon.shape),
            cmap=plt.cm.viridis,
            extend="both",
            title="etaH at RI",
        )

    if PLOT_U:
        # Quiver plot tangential vector field.
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(15, 6),
            subplot_kw={"projection": ccrs.PlateCarree(central_longitude=noon_lon)},
        )

        ax.coastlines()

        ax.quiver(
            plt_lon,
            plt_lat,
            dynamics.state.u.to_grid(plt_evaluator)[1].flatten(),
            -dynamics.state.u.to_grid(plt_evaluator)[0].flatten(),
            color="blue",
            transform=ccrs.PlateCarree(),
        )

        plt.show()

        # Alternative: plot theta and phi components separately.
        # pynamit.globalplot(
        #    plt_lon,
        #    plt_lat,
        #    dynamics.state.u.to_grid(plt_evaluator)[0].reshape(plt_lon.shape),
        #    cmap=plt.cm.viridis,
        #    extend="both",
        #    title="u at RI",
        # )

        # pynamit.globalplot(
        #    plt_lon,
        #    plt_lat,
        #    dynamics.state.u.to_grid(plt_evaluator)[1].reshape(plt_lon.shape),
        #    cmap=plt.cm.viridis,
        #    extend="both",
        #    title="u at RI",
        # )

print("Imposing steady state")
dynamics.impose_steady_state()
# Compare m_ind mapped to the earth to Br.coeffs / self.m_ind_to_Br mapped to the earth.
# Add effect of pfac
m_ind_mapped = dynamics.state.m_ind.coeffs * dynamics.state.basis.radial_shift_Ve(RI, RE)
pfac_mapped = dynamics.state.T_to_Ve.values.dot(
    dynamics.state.m_imp.coeffs
)  # * dynamics.state.basis.radial_shift_Ve(RI, RE)
Br_mapped = (
    dynamics.state.basis.radial_shift_Ve(1.5 * RI, RE)
    * dynamics.state.Br.coeffs
    / dynamics.state.m_ind_to_Br
)

# Global plot of m_ind_mapped and Br_mapped.
pynamit.globalplot(
    plt_lon,
    plt_lat,
    plt_evaluator.basis_to_grid(m_ind_mapped).reshape(plt_lon.shape),
    cmap=plt.cm.viridis,
    extend="both",
    title="m_ind_mapped at RE",
)
pynamit.globalplot(
    plt_lon,
    plt_lat,
    plt_evaluator.basis_to_grid(Br_mapped).reshape(plt_lon.shape),
    cmap=plt.cm.viridis,
    extend="both",
    title="Br_mapped at RE",
)

print(m_ind_mapped)
print(pfac_mapped)
print(Br_mapped)
# print("Norm of Br mapped: ", np.linalg.norm(Br_mapped))
# print("Norm of difference: ", np.linalg.norm(m_ind_mapped - Br_mapped))
dynamics.state.update_E()
print("Norm of E cf: ", np.linalg.norm(dynamics.state.E.coeffs[0]))
print("Norm of E df: ", np.linalg.norm(dynamics.state.E.coeffs[1]))
exit()

print("Time evolution")
final_time = 3600  # seconds
dynamics.evolve_to_time(final_time, dt=dt, sampling_step_interval=1, saving_sample_interval=1)
